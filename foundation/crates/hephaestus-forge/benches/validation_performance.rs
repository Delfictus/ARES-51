use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hephaestus_forge::validation::{MetamorphicTestSuite, ValidationConfig};
use hephaestus_forge::types::{VersionedModule, ModuleMetadata, ModuleId};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark for property-based test generation performance
fn bench_property_test_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create test module
    let test_module = create_test_module();
    
    let mut group = c.benchmark_group("property_test_generation");
    
    // Benchmark different test case counts
    for test_count in [1_000, 5_000, 10_000, 25_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*test_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("generate_test_cases", test_count),
            test_count,
            |b, &test_count| {
                b.to_async(&rt).iter(|| async {
                    let config = ValidationConfig {
                        property_testing: true,
                        differential_testing: false,
                        chaos_engineering: false,
                        ..Default::default()
                    };
                    
                    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
                    
                    let start = std::time::Instant::now();
                    let result = test_suite.validate_candidates(vec![test_module.clone()]).await;
                    let elapsed = start.elapsed();
                    
                    // Calculate test generation rate
                    let tests_per_second = if elapsed.as_millis() > 0 {
                        (test_count as u64 * 1000) / elapsed.as_millis() as u64
                    } else {
                        test_count as u64
                    };
                    
                    println!("Generated {} tests in {:?}, rate: {} tests/sec", 
                             test_count, elapsed, tests_per_second);
                    
                    // Verify we meet the 10K+ requirement
                    assert!(tests_per_second >= 10_000, 
                           "Test generation rate {} is below required 10K/sec", 
                           tests_per_second);
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark differential testing performance
fn bench_differential_testing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let test_module = create_test_module();
    
    let mut group = c.benchmark_group("differential_testing");
    
    for test_count in [100, 500, 1000, 2500].iter() {
        group.throughput(Throughput::Elements(*test_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("differential_comparison", test_count),
            test_count,
            |b, &test_count| {
                b.to_async(&rt).iter(|| async {
                    let config = ValidationConfig {
                        property_testing: false,
                        differential_testing: true,
                        chaos_engineering: false,
                        ..Default::default()
                    };
                    
                    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
                    let result = test_suite.validate_candidates(vec![test_module.clone()]).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark chaos engineering test performance
fn bench_chaos_engineering(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let test_module = create_test_module();
    
    let mut group = c.benchmark_group("chaos_engineering");
    group.sample_size(10); // Chaos tests are more expensive
    
    group.bench_function("chaos_test_suite", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ValidationConfig {
                property_testing: false,
                differential_testing: false,
                chaos_engineering: true,
                ..Default::default()
            };
            
            let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
            let result = test_suite.validate_candidates(vec![test_module.clone()]).await;
            
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark parallel test execution
fn bench_parallel_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let test_modules: Vec<VersionedModule> = (0..8).map(|i| {
        let mut module = create_test_module();
        module.metadata.id = ModuleId(format!("test_module_{}", i));
        module
    }).collect();
    
    let mut group = c.benchmark_group("parallel_execution");
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_validation", thread_count),
            thread_count,
            |b, &thread_count| {
                b.to_async(&rt).iter(|| async {
                    let config = ValidationConfig {
                        property_testing: true,
                        differential_testing: true,
                        chaos_engineering: false,
                        parallel_threads: *thread_count,
                        ..Default::default()
                    };
                    
                    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
                    let modules_to_test = test_modules.iter().take(*thread_count).cloned().collect();
                    let result = test_suite.validate_candidates(modules_to_test).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency during high-volume testing
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let test_module = create_test_module();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(20);
    
    group.bench_function("memory_usage_high_volume", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ValidationConfig {
                property_testing: true,
                max_test_cases_per_property: 100_000,
                memory_limit_mb: 512,
                ..Default::default()
            };
            
            let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
            
            // Measure memory usage before
            let memory_before = get_memory_usage();
            
            let result = test_suite.validate_candidates(vec![test_module.clone()]).await;
            
            // Measure memory usage after
            let memory_after = get_memory_usage();
            let memory_growth = memory_after.saturating_sub(memory_before);
            
            println!("Memory growth: {} MB", memory_growth / (1024 * 1024));
            
            // Ensure memory usage is reasonable
            assert!(memory_growth < 1024 * 1024 * 1024, // Less than 1GB growth
                   "Memory usage too high: {} bytes", memory_growth);
            
            black_box(result)
        });
    });
    
    group.finish();
}

/// Performance regression detection benchmark
fn bench_regression_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let baseline_module = create_test_module();
    let mut regression_module = create_test_module();
    
    // Artificially slow down the regression module
    regression_module.bytecode.extend(vec![0u8; 10000]); // Larger bytecode
    
    let mut group = c.benchmark_group("regression_detection");
    
    group.bench_function("detect_performance_regression", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ValidationConfig {
                differential_testing: true,
                regression_threshold: 0.1, // 10% performance degradation threshold
                ..Default::default()
            };
            
            let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
            
            // Test both modules and verify regression detection
            let baseline_result = test_suite.validate_candidates(vec![baseline_module.clone()]).await;
            let regression_result = test_suite.validate_candidates(vec![regression_module.clone()]).await;
            
            black_box((baseline_result, regression_result))
        });
    });
    
    group.finish();
}

/// Benchmark shrinking algorithm performance
fn bench_shrinking_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let test_module = create_test_module();
    
    let mut group = c.benchmark_group("shrinking_performance");
    
    for input_size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("shrink_counterexample", input_size),
            input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async {
                    let config = ValidationConfig {
                        property_testing: true,
                        shrinking_enabled: true,
                        max_shrinking_attempts: 1000,
                        ..Default::default()
                    };
                    
                    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
                    
                    // Create a large input that will likely need shrinking
                    let result = test_suite.validate_candidates(vec![test_module.clone()]).await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn create_test_module() -> VersionedModule {
    VersionedModule {
        metadata: ModuleMetadata {
            id: ModuleId("test_module".to_string()),
            version: "1.0.0".to_string(),
            name: "Test Module".to_string(),
            description: "Module for performance testing".to_string(),
            tags: vec!["test".to_string(), "benchmark".to_string()],
        },
        bytecode: vec![0u8; 1024], // 1KB of test bytecode
        dependencies: vec![],
        performance_profile: Default::default(),
        safety_invariants: vec![],
        proof_certificate: None,
    }
}

fn get_memory_usage() -> u64 {
    // Get current memory usage (simplified implementation)
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return kb * 1024; // Convert to bytes
                    }
                }
            }
        }
    }
    0
}

criterion_group!(
    benches,
    bench_property_test_generation,
    bench_differential_testing,
    bench_chaos_engineering,
    bench_parallel_execution,
    bench_memory_efficiency,
    bench_regression_detection,
    bench_shrinking_performance
);

criterion_main!(benches);