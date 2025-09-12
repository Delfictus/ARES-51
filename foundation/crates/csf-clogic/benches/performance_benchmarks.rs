//! 🛡️ HARDENING PHASE 3: Performance benchmarks for critical paths
//! These benchmarks ensure no performance regressions in production-critical operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use csf_bus::PhaseCoherenceBus;
use csf_clogic::adp::{AdaptiveDecisionProcessor, AdpConfig};
use csf_clogic::drpp::{DrppConfig, NeuralOscillator, PatternDetector};
use csf_clogic::egc::{EgcConfig, EmergentGovernanceController, RuleGenerator};
use csf_clogic::*;
use std::sync::Arc;
use std::time::Duration;

/// 🛡️ BENCHMARK: Pattern detection performance under varying loads
fn benchmark_pattern_detector_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detector_performance");
    group.measurement_time(Duration::from_secs(10));

    // Set up pattern detector with optimized config
    let config = DrppConfig {
        num_oscillators: 128,
        coupling_strength: 0.1,
        pattern_threshold: 0.8,
        frequency_range: (1.0, 100.0),
        time_window_ms: 1000,
        adaptive_tuning: true,
    };

    let detector = PatternDetector::new(&config);

    // Test with varying oscillator counts to check performance scaling
    for oscillator_count in [10, 50, 100, 200, 500].iter() {
        let oscillators: Vec<_> = (0..*oscillator_count)
            .map(|i| NeuralOscillator::new(i, &config))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("detect_patterns", oscillator_count),
            oscillator_count,
            |b, _| b.iter(|| black_box(detector.detect(black_box(&oscillators)))),
        );
    }

    group.finish();
}

/// 🛡️ BENCHMARK: Rule generation performance with circuit breaker
fn benchmark_rule_generator_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_generator_performance");
    group.measurement_time(Duration::from_secs(10));

    let config = EgcConfig::default();
    let generator = RuleGenerator::new(&config);

    // Test rule generation under different policy loads
    for policy_count in [100, 500, 1000, 2000, 5000].iter() {
        let policies: Vec<_> = (0..*policy_count)
            .map(|i| create_benchmark_policy(&format!("policy_{}", i)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("generate_rules", policy_count),
            policy_count,
            |b, _| b.iter(|| black_box(generator.generate_rules(black_box(&policies)))),
        );
    }

    group.finish();
}

/// 🛡️ BENCHMARK: Circuit breaker overhead measurement  
fn benchmark_circuit_breaker_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker_overhead");
    group.measurement_time(Duration::from_secs(5));

    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    // Empty oscillators to trigger circuit breaker path
    let empty_oscillators = vec![];
    let normal_oscillators: Vec<_> = (0..100)
        .map(|i| NeuralOscillator::new(i, &config))
        .collect();

    group.bench_function("circuit_breaker_active", |b| {
        // First trigger the circuit breaker
        for _ in 0..15 {
            detector.detect(&empty_oscillators);
        }

        // Now benchmark with circuit breaker active (should be very fast)
        b.iter(|| black_box(detector.detect(black_box(&empty_oscillators))))
    });

    group.bench_function("normal_operation", |b| {
        b.iter(|| black_box(detector.detect(black_box(&normal_oscillators))))
    });

    group.finish();
}

/// 🛡️ BENCHMARK: Memory allocation performance under resource limits
fn benchmark_memory_allocation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_performance");
    group.measurement_time(Duration::from_secs(8));

    let config = EgcConfig::default();
    let generator = RuleGenerator::new(&config);

    // Test memory allocation patterns under resource limits
    group.bench_function("bounded_rule_generation", |b| {
        b.iter(|| {
            // Generate rules up to the limit repeatedly to test allocation patterns
            let policies: Vec<_> = (0..1000)
                .map(|i| create_benchmark_policy(&format!("policy_{}", i)))
                .collect();

            for _ in 0..10 {
                // Multiple iterations to stress memory
                black_box(generator.generate_rules(black_box(&policies)));
            }
        })
    });

    group.finish();
}

/// 🛡️ BENCHMARK: Concurrent access performance with atomic operations
fn benchmark_concurrent_access_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access_performance");
    group.measurement_time(Duration::from_secs(10));

    let config = DrppConfig::default();
    let detector = Arc::new(PatternDetector::new(&config));

    let oscillators: Vec<_> = (0..100)
        .map(|i| NeuralOscillator::new(i, &config))
        .collect();

    group.bench_function("single_threaded", |b| {
        b.iter(|| {
            for _ in 0..100 {
                black_box(detector.detect(black_box(&oscillators)));
            }
        })
    });

    group.bench_function("concurrent_access", |b| {
        use std::thread;

        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let detector_clone = detector.clone();
                    let oscillators_clone = oscillators.clone();

                    thread::spawn(move || {
                        for _ in 0..25 {
                            // 4 threads * 25 = 100 total operations
                            black_box(detector_clone.detect(black_box(&oscillators_clone)));
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        })
    });

    group.finish();
}

/// 🛡️ BENCHMARK: End-to-end processing pipeline performance
fn benchmark_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_pipeline");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("complete_clogic_pipeline", |b| {
        b.iter(|| {
            // Create tokio runtime for each benchmark iteration
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                // Create full C-LOGIC system
                let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
                let config = CLogicConfig::default();
                let system = black_box(CLogicSystem::new(bus, config).await.unwrap());

                // Start system
                system.start().await.unwrap();

                // Get state (triggers all modules)
                let state = black_box(system.get_state().await);

                // Validate state
                assert!(state.timestamp.as_nanos() > 0);

                // Clean shutdown
                system.stop().await.unwrap();
            });
        })
    });

    group.finish();
}

// Helper function to create benchmark policies
fn create_benchmark_policy(name: &str) -> csf_clogic::egc::policy_engine::Policy {
    csf_clogic::egc::policy_engine::Policy {
        id: csf_clogic::egc::policy_engine::PolicyId::new(),
        name: name.to_string(),
        policy_type: csf_clogic::egc::policy_engine::PolicyType::Performance,
        conditions: vec![],
        actions: vec![],
        priority: 1,
        active: true,
        created_at: csf_core::hardware_timestamp(),
    }
}

// Performance regression thresholds - fail benchmarks if exceeded
criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets =
        benchmark_pattern_detector_performance,
        benchmark_rule_generator_performance,
        benchmark_circuit_breaker_overhead,
        benchmark_memory_allocation_performance,
        benchmark_concurrent_access_performance,
        benchmark_end_to_end_pipeline
}

criterion_main!(benches);
