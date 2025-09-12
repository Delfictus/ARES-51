//! Comprehensive test suite for intent storage system

use super::*;
use crate::intent::*;
use std::time::Instant;
use tempfile::TempDir;
use tokio::time::timeout;

/// Test utilities for storage testing
pub struct StorageTestHarness {
    pub storage: IntentStorage,
    pub _temp_dir: TempDir, // Keep alive for duration of test
}

impl StorageTestHarness {
    /// Create a new test harness with isolated storage
    pub async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_storage").to_string_lossy().to_string();
        config.max_concurrent_transactions = 100;
        config.cache_size_mb = 64; // Small cache for tests
        
        let storage = IntentStorage::new(config).await.unwrap();
        
        Self {
            storage,
            _temp_dir: temp_dir,
        }
    }
    
    /// Create a test intent with specified parameters
    pub fn create_test_intent(&self, module: &str, priority: Priority) -> OptimizationIntent {
        OptimizationIntent::builder()
            .target_module(module)
            .add_objective(Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(priority)
            .build()
            .unwrap()
    }
    
    /// Create multiple test intents
    pub fn create_test_intents(&self, count: usize) -> Vec<OptimizationIntent> {
        (0..count)
            .map(|i| {
                let priority = match i % 4 {
                    0 => Priority::Low,
                    1 => Priority::Medium,
                    2 => Priority::High,
                    3 => Priority::Critical,
                    _ => unreachable!(),
                };
                self.create_test_intent(&format!("module_{}", i), priority)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_storage_operations() {
        let harness = StorageTestHarness::new().await;
        
        let intent = harness.create_test_intent("test_module", Priority::High);
        let intent_id = intent.id.clone();
        
        // Store intent
        let version = harness.storage.store_intent(intent.clone()).await.unwrap();
        assert_eq!(version.version, 1);
        
        // Retrieve intent
        let retrieved = harness.storage.get_intent(&intent_id).await.unwrap();
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, intent_id);
        assert_eq!(retrieved.priority, Priority::High);
        assert_eq!(retrieved.objectives.len(), 1);
        assert_eq!(retrieved.constraints.len(), 1);
    }
    
    #[tokio::test]
    async fn test_versioning_system() {
        let harness = StorageTestHarness::new().await;
        
        let mut intent = harness.create_test_intent("test_module", Priority::Medium);
        let intent_id = intent.id.clone();
        
        // Store version 1
        let v1 = harness.storage.store_intent(intent.clone()).await.unwrap();
        assert_eq!(v1.version, 1);
        
        // Update and store version 2
        intent.priority = Priority::High;
        let v2 = harness.storage.store_intent(intent.clone()).await.unwrap();
        assert_eq!(v2.version, 2);
        
        // List all versions
        let versions = harness.storage.list_intent_versions(&intent_id).await.unwrap();
        assert_eq!(versions.len(), 2);
        
        // Get specific versions
        let v1_intent = harness.storage.get_intent_version(&intent_id, 1).await.unwrap().unwrap();
        assert_eq!(v1_intent.priority, Priority::Medium);
        
        let v2_intent = harness.storage.get_intent_version(&intent_id, 2).await.unwrap().unwrap();
        assert_eq!(v2_intent.priority, Priority::High);
        
        // Latest version should be version 2
        let latest = harness.storage.get_intent(&intent_id).await.unwrap().unwrap();
        assert_eq!(latest.priority, Priority::High);
    }
    
    #[tokio::test]
    async fn test_transaction_acid_properties() {
        let harness = StorageTestHarness::new().await;
        
        let intent1 = harness.create_test_intent("module_1", Priority::High);
        let intent2 = harness.create_test_intent("module_2", Priority::Medium);
        
        // Test atomic transaction
        let tx_manager = harness.storage.transaction_manager();
        let mut tx = tx_manager.begin_transaction().await.unwrap();
        
        tx.put_intent(intent1.clone(), 1).await.unwrap();
        tx.put_intent(intent2.clone(), 1).await.unwrap();
        
        // Both should not be visible until commit
        assert!(harness.storage.get_intent(&intent1.id).await.unwrap().is_none());
        assert!(harness.storage.get_intent(&intent2.id).await.unwrap().is_none());
        
        // Commit transaction
        tx.commit().await.unwrap();
        
        // Now both should be visible
        assert!(harness.storage.get_intent(&intent1.id).await.unwrap().is_some());
        assert!(harness.storage.get_intent(&intent2.id).await.unwrap().is_some());
    }
    
    #[tokio::test]
    async fn test_transaction_rollback() {
        let harness = StorageTestHarness::new().await;
        
        let intent = harness.create_test_intent("test_module", Priority::High);
        let intent_id = intent.id.clone();
        
        // Store initial intent
        harness.storage.store_intent(intent.clone()).await.unwrap();
        
        // Start transaction to modify intent
        let tx_manager = harness.storage.transaction_manager();
        let mut tx = tx_manager.begin_transaction().await.unwrap();
        
        let mut modified_intent = intent.clone();
        modified_intent.priority = Priority::Critical;
        tx.update_intent(modified_intent, 2).await.unwrap();
        
        // Rollback transaction
        tx.rollback().await.unwrap();
        
        // Original intent should be unchanged
        let retrieved = harness.storage.get_intent(&intent_id).await.unwrap().unwrap();
        assert_eq!(retrieved.priority, Priority::High); // Not Critical
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        let harness = StorageTestHarness::new().await;
        
        const NUM_OPERATIONS: usize = 100;
        let intents = harness.create_test_intents(NUM_OPERATIONS);
        
        // Spawn concurrent storage operations
        let mut handles = Vec::new();
        
        for intent in intents {
            let storage = harness.storage.clone();
            let handle = tokio::spawn(async move {
                // Store intent
                let version = storage.store_intent(intent.clone()).await.unwrap();
                assert_eq!(version.version, 1);
                
                // Immediately retrieve it
                let retrieved = storage.get_intent(&intent.id).await.unwrap();
                assert!(retrieved.is_some());
                
                let retrieved = retrieved.unwrap();
                assert_eq!(retrieved.id, intent.id);
                assert_eq!(retrieved.priority, intent.priority);
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify total count
        let stats = harness.storage.get_stats().await.unwrap();
        assert_eq!(stats.total_intents, NUM_OPERATIONS as u64);
    }
    
    #[tokio::test]
    async fn test_performance_requirements() {
        let harness = StorageTestHarness::new().await;
        
        // Pre-populate storage with many intents
        const POPULATION_SIZE: usize = 1000;
        let intents = harness.create_test_intents(POPULATION_SIZE);
        
        // Store all intents
        for intent in &intents {
            harness.storage.store_intent(intent.clone()).await.unwrap();
        }
        
        // Test retrieval performance - should be <1ms each
        const TEST_COUNT: usize = 100;
        let mut total_time = std::time::Duration::ZERO;
        
        for i in 0..TEST_COUNT {
            let intent_id = &intents[i].id;
            
            let start = Instant::now();
            let retrieved = harness.storage.get_intent(intent_id).await.unwrap();
            let elapsed = start.elapsed();
            
            total_time += elapsed;
            
            assert!(retrieved.is_some());
            assert!(
                elapsed.as_millis() < 1,
                "Retrieval took {}ms, expected <1ms", elapsed.as_millis()
            );
        }
        
        let avg_time = total_time / TEST_COUNT as u32;
        println!("Average retrieval time: {}μs", avg_time.as_micros());
        
        // Check cache hit rate
        let cache_hit_rate = harness.storage.indexing.cache_hit_rate().await.unwrap();
        assert!(cache_hit_rate > 0.5, "Cache hit rate too low: {}", cache_hit_rate);
    }
    
    #[tokio::test]
    async fn test_search_functionality() {
        let harness = StorageTestHarness::new().await;
        
        // Create intents with different priorities
        let high_intent = harness.create_test_intent("high_module", Priority::High);
        let medium_intent = harness.create_test_intent("medium_module", Priority::Medium);
        let low_intent = harness.create_test_intent("low_module", Priority::Low);
        
        // Store all intents
        harness.storage.store_intent(high_intent.clone()).await.unwrap();
        harness.storage.store_intent(medium_intent.clone()).await.unwrap();
        harness.storage.store_intent(low_intent.clone()).await.unwrap();
        
        // Search by priority
        let high_priority_query = IntentSearchQuery {
            priority: Some(Priority::High),
            target_module: None,
            text_search: None,
            date_range: None,
            limit: None,
        };
        
        let results = harness.storage.search_intents(high_priority_query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].priority, Priority::High);
        
        // Search by target module
        let module_query = IntentSearchQuery {
            priority: None,
            target_module: Some("medium_module".to_string()),
            text_search: None,
            date_range: None,
            limit: None,
        };
        
        let results = harness.storage.search_intents(module_query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].priority, Priority::Medium);
    }
    
    #[tokio::test]
    async fn test_storage_capacity() {
        let harness = StorageTestHarness::new().await;
        
        // Test storing 10,000 intents to verify capacity
        const LARGE_COUNT: usize = 10_000;
        println!("Testing storage capacity with {} intents", LARGE_COUNT);
        
        let start = Instant::now();
        
        // Create and store intents in batches for better performance
        const BATCH_SIZE: usize = 100;
        let mut stored_count = 0;
        
        for batch in 0..(LARGE_COUNT / BATCH_SIZE) {
            let batch_intents = harness.create_test_intents(BATCH_SIZE);
            
            // Store batch concurrently
            let mut handles = Vec::new();
            for intent in batch_intents {
                let storage = harness.storage.clone();
                let handle = tokio::spawn(async move {
                    storage.store_intent(intent).await.unwrap();
                });
                handles.push(handle);
            }
            
            // Wait for batch to complete
            for handle in handles {
                handle.await.unwrap();
                stored_count += 1;
            }
            
            if batch % 10 == 0 {
                println!("Stored {} intents...", stored_count);
            }
        }
        
        let elapsed = start.elapsed();
        println!("Stored {} intents in {:?} ({:.2} intents/sec)", 
                 stored_count, elapsed, stored_count as f64 / elapsed.as_secs_f64());
        
        // Verify final count
        let stats = harness.storage.get_stats().await.unwrap();
        assert_eq!(stats.total_intents, stored_count as u64);
        
        println!("Storage stats: {:?}", stats);
    }
    
    #[tokio::test]
    async fn test_error_handling() {
        let harness = StorageTestHarness::new().await;
        
        // Test retrieving non-existent intent
        let fake_id = IntentId::new();
        let result = harness.storage.get_intent(&fake_id).await.unwrap();
        assert!(result.is_none());
        
        // Test invalid intent version
        let intent = harness.create_test_intent("test", Priority::Medium);
        harness.storage.store_intent(intent.clone()).await.unwrap();
        
        let result = harness.storage.get_intent_version(&intent.id, 999).await.unwrap();
        assert!(result.is_none());
    }
    
    #[tokio::test]
    async fn test_maintenance_operations() {
        let harness = StorageTestHarness::new().await;
        
        // Store some intents
        let intents = harness.create_test_intents(50);
        for intent in &intents {
            harness.storage.store_intent(intent.clone()).await.unwrap();
        }
        
        // Run maintenance
        harness.storage.maintenance().await.unwrap();
        
        // Verify intents are still accessible after maintenance
        for intent in &intents {
            let retrieved = harness.storage.get_intent(&intent.id).await.unwrap();
            assert!(retrieved.is_some());
        }
        
        let stats = harness.storage.get_stats().await.unwrap();
        assert_eq!(stats.total_intents, intents.len() as u64);
    }
    
    #[tokio::test]
    async fn test_timeout_resilience() {
        let harness = StorageTestHarness::new().await;
        
        // Test that operations complete within reasonable time limits
        let intent = harness.create_test_intent("timeout_test", Priority::Medium);
        
        // Store with timeout
        let store_result = timeout(
            std::time::Duration::from_secs(5),
            harness.storage.store_intent(intent.clone())
        ).await;
        
        assert!(store_result.is_ok(), "Store operation timed out");
        let version = store_result.unwrap().unwrap();
        assert_eq!(version.version, 1);
        
        // Retrieve with timeout
        let retrieve_result = timeout(
            std::time::Duration::from_millis(100), // Very strict timeout
            harness.storage.get_intent(&intent.id)
        ).await;
        
        assert!(retrieve_result.is_ok(), "Retrieve operation timed out");
        let retrieved = retrieve_result.unwrap().unwrap();
        assert!(retrieved.is_some());
    }
}

/// Stress tests for extreme conditions
#[cfg(test)]
mod stress_tests {
    use super::*;
    
    #[tokio::test]
    #[ignore = "Long running stress test"]
    async fn stress_test_concurrent_access() {
        let harness = StorageTestHarness::new().await;
        
        const CONCURRENT_OPERATIONS: usize = 1000;
        const OPERATIONS_PER_TASK: usize = 100;
        
        println!("Starting stress test with {} concurrent tasks", CONCURRENT_OPERATIONS);
        
        let start = Instant::now();
        let mut handles = Vec::new();
        
        for task_id in 0..CONCURRENT_OPERATIONS {
            let storage = harness.storage.clone();
            let handle = tokio::spawn(async move {
                for op_id in 0..OPERATIONS_PER_TASK {
                    let intent = OptimizationIntent::builder()
                        .target_module(format!("stress_module_{}_{}", task_id, op_id))
                        .add_objective(Objective::MinimizeLatency {
                            percentile: 99.0,
                            target_ms: 10.0,
                        })
                        .priority(Priority::Medium)
                        .build()
                        .unwrap();
                    
                    // Store and immediately retrieve
                    let version = storage.store_intent(intent.clone()).await.unwrap();
                    assert_eq!(version.version, 1);
                    
                    let retrieved = storage.get_intent(&intent.id).await.unwrap();
                    assert!(retrieved.is_some());
                }
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        let elapsed = start.elapsed();
        let total_ops = CONCURRENT_OPERATIONS * OPERATIONS_PER_TASK * 2; // store + retrieve
        
        println!("Completed {} operations in {:?} ({:.2} ops/sec)", 
                 total_ops, elapsed, total_ops as f64 / elapsed.as_secs_f64());
        
        let stats = harness.storage.get_stats().await.unwrap();
        println!("Final storage stats: {:?}", stats);
        
        assert_eq!(stats.total_intents, (CONCURRENT_OPERATIONS * OPERATIONS_PER_TASK) as u64);
    }
}