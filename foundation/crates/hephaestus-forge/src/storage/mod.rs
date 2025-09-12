//! High-performance persistent storage for optimization intents
//! 
//! This module provides ACID-compliant storage with sub-millisecond retrieval times,
//! supporting 1M+ intents with concurrent access, versioning, and rollback capabilities.

pub mod intent_store;
pub mod indexing;
pub mod versioning;
pub mod transactions;

#[cfg(test)]
pub mod tests;

pub use intent_store::*;
pub use indexing::*;
pub use versioning::*;
pub use transactions::*;

use crate::{intent::*, types::ForgeResult, ForgeError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Configuration for intent storage system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database path
    pub db_path: String,
    
    /// Maximum number of concurrent transactions
    pub max_concurrent_transactions: usize,
    
    /// Cache size for hot intents (in MB)
    pub cache_size_mb: usize,
    
    /// Compression settings
    pub compression: CompressionConfig,
    
    /// Index configuration
    pub indexing: IndexConfig,
    
    /// Versioning settings
    pub versioning: VersioningConfig,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: i32,
}

/// Available compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Snappy,
    Lz4,
    Zstd,
}

/// Index configuration for fast lookups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Enable bloom filters for existence checks
    pub bloom_filters: bool,
    
    /// Block cache size (in MB)
    pub block_cache_mb: usize,
    
    /// Write buffer size (in MB)  
    pub write_buffer_mb: usize,
    
    /// Number of levels for LSM tree
    pub max_levels: i32,
}

/// Versioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Maximum number of versions to keep per intent
    pub max_versions: u32,
    
    /// Automatic cleanup of old versions
    pub auto_cleanup: bool,
    
    /// Cleanup interval in seconds
    pub cleanup_interval_sec: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: "./data/intents".to_string(),
            max_concurrent_transactions: 1000,
            cache_size_mb: 512,
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Snappy,
                level: 6,
            },
            indexing: IndexConfig {
                bloom_filters: true,
                block_cache_mb: 256,
                write_buffer_mb: 64,
                max_levels: 7,
            },
            versioning: VersioningConfig {
                max_versions: 100,
                auto_cleanup: true,
                cleanup_interval_sec: 3600, // 1 hour
            },
        }
    }
}

/// Main intent storage interface
#[derive(Clone)]
pub struct IntentStorage {
    store: Arc<IntentStore>,
    indexing: Arc<IndexingEngine>,
    versioning: Arc<VersioningEngine>,
    transactions: Arc<TransactionManager>,
    config: StorageConfig,
}

impl IntentStorage {
    /// Create a new intent storage system
    pub async fn new(config: StorageConfig) -> ForgeResult<Self> {
        // Ensure database directory exists
        std::fs::create_dir_all(&config.db_path)
            .map_err(|e| ForgeError::StorageError(format!("Failed to create db directory: {}", e)))?;
            
        // Initialize storage components
        let store = Arc::new(IntentStore::new(&config).await?);
        let indexing = Arc::new(IndexingEngine::new(store.clone(), &config).await?);
        let versioning = Arc::new(VersioningEngine::new(store.clone(), &config).await?);
        let transactions = Arc::new(TransactionManager::new(store.clone(), &config).await?);
        
        Ok(Self {
            store,
            indexing,
            versioning,
            transactions,
            config,
        })
    }
    
    /// Store a new optimization intent with automatic versioning
    pub async fn store_intent(&self, intent: OptimizationIntent) -> ForgeResult<IntentVersion> {
        let tx = self.transactions.begin_transaction().await?;
        
        let result = async {
            // Create new version
            let version = self.versioning.create_version(&intent.id, &intent).await?;
            
            // Store in primary store
            self.store.put_intent(&intent, version.version).await?;
            
            // Update indexes
            self.indexing.index_intent(&intent, version.version).await?;
            
            Ok(version)
        }.await;
        
        match result {
            Ok(version) => {
                tx.commit().await?;
                Ok(version)
            }
            Err(e) => {
                tx.rollback().await?;
                Err(e)
            }
        }
    }
    
    /// Retrieve an optimization intent by ID with sub-millisecond performance
    pub async fn get_intent(&self, intent_id: &IntentId) -> ForgeResult<Option<OptimizationIntent>> {
        // Check cache first for hot paths
        if let Some(cached) = self.indexing.get_cached_intent(intent_id).await? {
            return Ok(Some(cached));
        }
        
        // Retrieve from storage
        let intent = self.store.get_intent(intent_id).await?;
        
        // Update cache if found
        if let Some(ref intent) = intent {
            self.indexing.cache_intent(intent).await?;
        }
        
        Ok(intent)
    }
    
    /// Get specific version of an intent
    pub async fn get_intent_version(&self, intent_id: &IntentId, version: u32) -> ForgeResult<Option<OptimizationIntent>> {
        self.versioning.get_version(intent_id, version).await
    }
    
    /// List all versions of an intent
    pub async fn list_intent_versions(&self, intent_id: &IntentId) -> ForgeResult<Vec<IntentVersion>> {
        self.versioning.list_versions(intent_id).await
    }
    
    /// Update an existing intent (creates new version)
    pub async fn update_intent(&self, intent: OptimizationIntent) -> ForgeResult<IntentVersion> {
        self.store_intent(intent).await
    }
    
    /// Delete an intent and all its versions
    pub async fn delete_intent(&self, intent_id: &IntentId) -> ForgeResult<()> {
        let tx = self.transactions.begin_transaction().await?;
        
        let result = async {
            // Remove from indexes
            self.indexing.remove_intent(intent_id).await?;
            
            // Remove all versions
            self.versioning.delete_all_versions(intent_id).await?;
            
            // Remove from primary store
            self.store.delete_intent(intent_id).await?;
            
            Ok(())
        }.await;
        
        match result {
            Ok(()) => {
                tx.commit().await?;
                Ok(())
            }
            Err(e) => {
                tx.rollback().await?;
                Err(e)
            }
        }
    }
    
    /// Search intents by various criteria
    pub async fn search_intents(&self, query: IntentSearchQuery) -> ForgeResult<Vec<OptimizationIntent>> {
        self.indexing.search_intents(query).await
    }
    
    /// Get storage statistics
    pub async fn get_stats(&self) -> ForgeResult<StorageStats> {
        Ok(StorageStats {
            total_intents: self.store.count_intents().await?,
            total_versions: self.versioning.count_versions().await?,
            cache_hit_rate: self.indexing.cache_hit_rate().await?,
            storage_size_bytes: self.store.storage_size().await?,
            index_size_bytes: self.indexing.index_size().await?,
        })
    }
    
    /// Perform maintenance operations (cleanup old versions, compact storage)
    pub async fn maintenance(&self) -> ForgeResult<()> {
        tokio::try_join!(
            self.versioning.cleanup_old_versions(),
            self.store.compact(),
            self.indexing.rebuild_indexes()
        )?;
        
        Ok(())
    }
    
    /// Create a snapshot of the current state
    pub async fn create_snapshot(&self, path: &Path) -> ForgeResult<SnapshotInfo> {
        self.store.create_snapshot(path).await
    }
    
    /// Restore from a snapshot
    pub async fn restore_snapshot(&self, path: &Path) -> ForgeResult<()> {
        self.store.restore_snapshot(path).await
    }
    
    /// Get transaction manager for custom operations
    pub fn transaction_manager(&self) -> &TransactionManager {
        &self.transactions
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_intents: u64,
    pub total_versions: u64,
    pub cache_hit_rate: f64,
    pub storage_size_bytes: u64,
    pub index_size_bytes: u64,
}

/// Snapshot information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub path: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub size_bytes: u64,
    pub intent_count: u64,
}

/// Intent search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSearchQuery {
    /// Filter by priority
    pub priority: Option<Priority>,
    
    /// Filter by target module
    pub target_module: Option<String>,
    
    /// Search in objectives/constraints
    pub text_search: Option<String>,
    
    /// Date range filter
    pub date_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    
    /// Maximum results to return
    pub limit: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_storage() -> (IntentStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();
        
        let storage = IntentStorage::new(config).await.unwrap();
        (storage, temp_dir)
    }
    
    #[tokio::test]
    async fn test_basic_intent_operations() {
        let (storage, _temp_dir) = create_test_storage().await;
        
        // Create test intent
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Store intent
        let version = storage.store_intent(intent.clone()).await.unwrap();
        assert_eq!(version.version, 1);
        
        // Retrieve intent
        let retrieved = storage.get_intent(&intent_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, intent_id);
        
        // Update intent (creates new version)
        let mut updated_intent = intent.clone();
        updated_intent.priority = Priority::Critical;
        let new_version = storage.update_intent(updated_intent).await.unwrap();
        assert_eq!(new_version.version, 2);
        
        // List versions
        let versions = storage.list_intent_versions(&intent_id).await.unwrap();
        assert_eq!(versions.len(), 2);
        
        // Delete intent
        storage.delete_intent(&intent_id).await.unwrap();
        let deleted = storage.get_intent(&intent_id).await.unwrap();
        assert!(deleted.is_none());
    }
    
    #[tokio::test]
    async fn test_performance_requirements() {
        let (storage, _temp_dir) = create_test_storage().await;
        
        // Create 1000 test intents
        let mut intents = Vec::new();
        for i in 0..1000 {
            let intent = OptimizationIntent::builder()
                .target_module(format!("module_{}", i))
                .add_objective(crate::intent::Objective::MinimizeLatency {
                    percentile: 99.0,
                    target_ms: 10.0,
                })
                .priority(Priority::Medium)
                .build()
                .unwrap();
            
            storage.store_intent(intent.clone()).await.unwrap();
            intents.push(intent);
        }
        
        // Test retrieval performance (should be <1ms each)
        for intent in &intents[..100] { // Test first 100
            let start = std::time::Instant::now();
            let retrieved = storage.get_intent(&intent.id).await.unwrap();
            let elapsed = start.elapsed();
            
            assert!(retrieved.is_some());
            assert!(elapsed.as_millis() < 1, "Retrieval took {}ms, expected <1ms", elapsed.as_millis());
        }
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        let (storage, _temp_dir) = create_test_storage().await;
        
        // Spawn multiple concurrent operations
        let mut handles = Vec::new();
        
        for i in 0..100 {
            let storage = storage.clone();
            let handle = tokio::spawn(async move {
                let intent = OptimizationIntent::builder()
                    .target_module(format!("concurrent_module_{}", i))
                    .add_objective(crate::intent::Objective::MinimizeLatency {
                        percentile: 99.0,
                        target_ms: 10.0,
                    })
                    .priority(Priority::Medium)
                    .build()
                    .unwrap();
                
                // Store and immediately retrieve
                storage.store_intent(intent.clone()).await.unwrap();
                let retrieved = storage.get_intent(&intent.id).await.unwrap();
                assert!(retrieved.is_some());
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        futures::future::try_join_all(handles).await.unwrap();
    }
}