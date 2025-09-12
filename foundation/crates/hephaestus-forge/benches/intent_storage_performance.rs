//! Performance benchmarks for intent storage system
//! 
//! Verifies the critical <1ms retrieval requirement and other performance metrics

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hephaestus_forge::storage::{IntentStorage, StorageConfig, IntentSearchQuery};
use hephaestus_forge::intent::*;
use tempfile::TempDir;
use tokio::runtime::Runtime;
use std::time::Duration;
use rand::Rng;

/// Benchmark harness for storage performance testing
struct BenchmarkHarness {
    storage: IntentStorage,
    _temp_dir: TempDir,
    runtime: Runtime,
}

impl BenchmarkHarness {
    /// Create a new benchmark harness
    fn new() -> Self {
        let runtime = Runtime::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("bench_storage").to_string_lossy().to_string();
        config.cache_size_mb = 512; // Large cache for benchmarks
        config.max_concurrent_transactions = 1000;
        
        let storage = runtime.block_on(IntentStorage::new(config)).unwrap();
        
        Self {
            storage,
            _temp_dir: temp_dir,
            runtime,
        }
    }
    
    /// Create a test intent
    fn create_intent(&self, id: usize) -> OptimizationIntent {
        OptimizationIntent::builder()
            .target_module(format!("benchmark_module_{}", id))
            .add_objective(Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .add_objective(Objective::MaximizeThroughput {
                target_ops_per_sec: 1000.0,
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .add_constraint(Constraint::MaxMemoryMB(512))
            .priority(match id % 4 {
                0 => Priority::Low,
                1 => Priority::Medium,
                2 => Priority::High,
                3 => Priority::Critical,
                _ => unreachable!(),
            })
            .build()
            .unwrap()
    }
    
    /// Pre-populate storage with test data
    fn populate_storage(&mut self, count: usize) -> Vec<IntentId> {
        let intents: Vec<_> = (0..count).map(|i| self.create_intent(i)).collect();
        let intent_ids: Vec<_> = intents.iter().map(|i| i.id.clone()).collect();
        
        self.runtime.block_on(async {
            for intent in intents {
                self.storage.store_intent(intent).await.unwrap();
            }
        });
        
        intent_ids
    }
}

/// Benchmark single intent retrieval (critical <1ms requirement)
fn bench_single_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_retrieval");
    
    // Test different storage sizes
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let mut harness = BenchmarkHarness::new();
        let intent_ids = harness.populate_storage(*size);
        
        group.bench_with_input(
            BenchmarkId::new("get_intent", size),
            size,
            |b, _| {
                let mut counter = 0;
                b.iter(|| {
                    let intent_id = &intent_ids[counter % intent_ids.len()];
                    counter += 1;
                    
                    let result = harness.runtime.block_on(
                        harness.storage.get_intent(black_box(intent_id))
                    );
                    
                    black_box(result.unwrap().unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache performance
fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    
    let mut harness = BenchmarkHarness::new();
    let intent_ids = harness.populate_storage(1000);
    
    // Cold cache - first access
    group.bench_function("cold_cache", |b| {
        b.iter(|| {
            let intent_id = &intent_ids[rand::thread_rng().gen_range(0..intent_ids.len())];
            let result = harness.runtime.block_on(
                harness.storage.get_intent(black_box(intent_id))
            );
            black_box(result.unwrap().unwrap());
        });
    });
    
    // Warm up cache
    harness.runtime.block_on(async {
        for intent_id in &intent_ids[0..100] {
            harness.storage.get_intent(intent_id).await.unwrap();
        }
    });
    
    // Hot cache - repeated access to same intents
    group.bench_function("hot_cache", |b| {
        b.iter(|| {
            let intent_id = &intent_ids[rand::random::<usize>() % 100]; // Only first 100
            let result = harness.runtime.block_on(
                harness.storage.get_intent(black_box(intent_id))
            );
            black_box(result.unwrap().unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark concurrent access patterns
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    
    let mut harness = BenchmarkHarness::new();
    let intent_ids = harness.populate_storage(10_000);
    
    for concurrency in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", concurrency),
            concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    let handles: Vec<_> = (0..concurrency).map(|_| {
                        let storage = harness.storage.clone();
                        let intent_ids = intent_ids.clone();
                        tokio::spawn(async move {
                            let intent_id = &intent_ids[rand::thread_rng().gen_range(0..intent_ids.len())];
                            storage.get_intent(intent_id).await.unwrap().unwrap()
                        })
                    }).collect();
                    
                    harness.runtime.block_on(async {
                        for handle in handles {
                            black_box(handle.await.unwrap());
                        }
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark storage operations throughput
fn bench_storage_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_throughput");
    group.throughput(Throughput::Elements(1));
    
    let mut harness = BenchmarkHarness::new();
    
    // Benchmark store operations
    group.bench_function("store_intent", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent = harness.create_intent(counter);
            counter += 1;
            
            let result = harness.runtime.block_on(
                harness.storage.store_intent(black_box(intent))
            );
            black_box(result.unwrap());
        });
    });
    
    // Pre-populate for retrieval benchmarks
    let intent_ids = harness.populate_storage(10_000);
    
    // Benchmark retrieval operations
    group.bench_function("retrieve_intent", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent_id = &intent_ids[counter % intent_ids.len()];
            counter += 1;
            
            let result = harness.runtime.block_on(
                harness.storage.get_intent(black_box(intent_id))
            );
            black_box(result.unwrap().unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark search operations
fn bench_search_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_operations");
    
    let mut harness = BenchmarkHarness::new();
    let _intent_ids = harness.populate_storage(10_000);
    
    // Search by priority
    group.bench_function("search_by_priority", |b| {
        b.iter(|| {
            let query = IntentSearchQuery {
                priority: Some(Priority::High),
                target_module: None,
                text_search: None,
                date_range: None,
                limit: Some(100),
            };
            
            let result = harness.runtime.block_on(
                harness.storage.search_intents(black_box(query))
            );
            black_box(result.unwrap());
        });
    });
    
    // Search by target module
    group.bench_function("search_by_module", |b| {
        b.iter(|| {
            let query = IntentSearchQuery {
                priority: None,
                target_module: Some("benchmark_module_500".to_string()),
                text_search: None,
                date_range: None,
                limit: Some(100),
            };
            
            let result = harness.runtime.block_on(
                harness.storage.search_intents(black_box(query))
            );
            black_box(result.unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark transaction operations
fn bench_transaction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_operations");
    
    let mut harness = BenchmarkHarness::new();
    
    // Simple transaction
    group.bench_function("simple_transaction", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent = harness.create_intent(counter);
            counter += 1;
            
            harness.runtime.block_on(async {
                let tx_manager = harness.storage.transaction_manager();
                let mut tx = tx_manager.begin_transaction().await.unwrap();
                tx.put_intent(black_box(intent), 1).await.unwrap();
                tx.commit().await.unwrap();
            });
        });
    });
    
    // Complex transaction with multiple operations
    group.bench_function("complex_transaction", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent1 = harness.create_intent(counter);
            let intent2 = harness.create_intent(counter + 1);
            let intent3 = harness.create_intent(counter + 2);
            counter += 3;
            
            harness.runtime.block_on(async {
                let tx_manager = harness.storage.transaction_manager();
                let mut tx = tx_manager.begin_transaction().await.unwrap();
                tx.put_intent(black_box(intent1), 1).await.unwrap();
                tx.put_intent(black_box(intent2), 1).await.unwrap();
                tx.put_intent(black_box(intent3), 1).await.unwrap();
                tx.commit().await.unwrap();
            });
        });
    });
    
    group.finish();
}

/// Benchmark versioning operations
fn bench_versioning_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("versioning_operations");
    
    let mut harness = BenchmarkHarness::new();
    let intent_ids = harness.populate_storage(1000);
    
    // Create multiple versions
    harness.runtime.block_on(async {
        for intent_id in &intent_ids[0..100] {
            let intent = harness.storage.get_intent(intent_id).await.unwrap().unwrap();
            // Create versions 2-5
            for version in 2..=5 {
                let mut updated_intent = intent.clone();
                updated_intent.priority = match version % 4 {
                    0 => Priority::Low,
                    1 => Priority::Medium,
                    2 => Priority::High,
                    3 => Priority::Critical,
                    _ => unreachable!(),
                };
                harness.storage.store_intent(updated_intent).await.unwrap();
            }
        }
    });
    
    group.bench_function("get_latest_version", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent_id = &intent_ids[counter % 100];
            counter += 1;
            
            let result = harness.runtime.block_on(
                harness.storage.get_intent(black_box(intent_id))
            );
            black_box(result.unwrap().unwrap());
        });
    });
    
    group.bench_function("get_specific_version", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent_id = &intent_ids[counter % 100];
            let version = (counter % 4) + 2; // Versions 2-5
            counter += 1;
            
            let result = harness.runtime.block_on(
                harness.storage.get_intent_version(black_box(intent_id), black_box(version as u32))
            );
            black_box(result.unwrap());
        });
    });
    
    group.bench_function("list_versions", |b| {
        let mut counter = 0;
        b.iter(|| {
            let intent_id = &intent_ids[counter % 100];
            counter += 1;
            
            let result = harness.runtime.block_on(
                harness.storage.list_intent_versions(black_box(intent_id))
            );
            black_box(result.unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark under high load conditions
fn bench_high_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_load");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for stability
    
    let mut harness = BenchmarkHarness::new();
    let intent_ids = harness.populate_storage(100_000);
    
    group.bench_function("mixed_workload", |b| {
        b.iter(|| {
            // Simulate mixed workload: 70% reads, 20% writes, 10% searches
            let operation = rand::thread_rng().gen::<f64>();
            
            harness.runtime.block_on(async {
                if operation < 0.7 {
                    // Read operation
                    let intent_id = &intent_ids[rand::thread_rng().gen_range(0..intent_ids.len())];
                    let result = harness.storage.get_intent(intent_id).await.unwrap();
                    black_box(result);
                } else if operation < 0.9 {
                    // Write operation
                    let intent = harness.create_intent(rand::thread_rng().gen::<usize>());
                    let result = harness.storage.store_intent(intent).await.unwrap();
                    black_box(result);
                } else {
                    // Search operation
                    let query = IntentSearchQuery {
                        priority: Some(Priority::High),
                        target_module: None,
                        text_search: None,
                        date_range: None,
                        limit: Some(10),
                    };
                    let result = harness.storage.search_intents(query).await.unwrap();
                    black_box(result);
                }
            });
        });
    });
    
    group.finish();
}

/// Critical benchmark: Verify <1ms retrieval requirement
fn bench_critical_retrieval_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("critical_latency");
    group.measurement_time(Duration::from_secs(60)); // Extended measurement
    group.sample_size(10000); // Large sample size for statistical significance
    
    let mut harness = BenchmarkHarness::new();
    let intent_ids = harness.populate_storage(1_000_000); // 1M intents as required
    
    // Warm up cache with random access pattern
    harness.runtime.block_on(async {
        for _ in 0..10_000 {
            let intent_id = &intent_ids[rand::thread_rng().gen_range(0..intent_ids.len())];
            harness.storage.get_intent(intent_id).await.unwrap();
        }
    });
    
    group.bench_function("sub_millisecond_retrieval", |b| {
        let mut counter = 0;
        b.iter_custom(|iters| {
            let mut total_time = Duration::ZERO;
            
            for _ in 0..iters {
                let intent_id = &intent_ids[counter % intent_ids.len()];
                counter += 1;
                
                let start = std::time::Instant::now();
                let result = harness.runtime.block_on(
                    harness.storage.get_intent(black_box(intent_id))
                );
                let elapsed = start.elapsed();
                
                // Verify result
                black_box(result.unwrap().unwrap());
                
                // Accumulate time
                total_time += elapsed;
                
                // Assert <1ms requirement
                assert!(
                    elapsed.as_millis() < 1,
                    "Retrieval took {}μs, exceeds 1ms requirement", 
                    elapsed.as_micros()
                );
            }
            
            total_time
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_retrieval,
    bench_cache_performance,
    bench_concurrent_access,
    bench_storage_throughput,
    bench_search_operations,
    bench_transaction_operations,
    bench_versioning_operations,
    bench_high_load,
    bench_critical_retrieval_latency,
);

criterion_main!(benches);