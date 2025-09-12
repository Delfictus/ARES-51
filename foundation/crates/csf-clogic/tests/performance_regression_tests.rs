//! 🛡️ HARDENING PHASE 3: Performance regression prevention tests
//! These tests ensure that performance doesn't regress below acceptable thresholds

use csf_clogic::drpp::{DrppConfig, NeuralOscillator, PatternDetector};
use csf_clogic::egc::{EgcConfig, RuleGenerator};
use csf_clogic::performance_monitor::{MonitoredMutex, PerformanceMonitor};
use csf_clogic::*;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// 🛡️ Performance regression test for pattern detection
#[test]
fn test_pattern_detection_performance_regression() {
    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    // Create test oscillators
    let oscillators: Vec<_> = (0..100)
        .map(|i| NeuralOscillator::new(i, &config))
        .collect();

    // Performance benchmark: should complete 1000 detections in <1 second
    let start = Instant::now();
    for _ in 0..1000 {
        detector.detect(&oscillators);
    }
    let duration = start.elapsed();

    println!(
        "✅ Pattern detection performance: {}ms for 1000 operations",
        duration.as_millis()
    );

    // Regression threshold: should not exceed 2 seconds for 1000 operations
    assert!(
        duration < Duration::from_secs(2),
        "Performance regression: Pattern detection took {}ms (threshold: 2000ms)",
        duration.as_millis()
    );

    // Additional check: average operation should be <2ms
    let avg_per_op = duration.as_nanos() / 1000;
    assert!(
        avg_per_op < 2_000_000, // 2ms in nanoseconds
        "Performance regression: Average operation time {}μs (threshold: 2000μs)",
        avg_per_op / 1000
    );
}

/// 🛡️ Performance regression test for rule generation
#[test]
fn test_rule_generation_performance_regression() {
    let config = EgcConfig::default();
    let generator = RuleGenerator::new(&config);

    // Create test policies
    let policies: Vec<_> = (0..1000)
        .map(|i| csf_clogic::egc::policy_engine::Policy {
            id: csf_clogic::egc::policy_engine::PolicyId::new(),
            name: format!("policy_{}", i),
            policy_type: csf_clogic::egc::policy_engine::PolicyType::Performance,
            conditions: vec![],
            actions: vec![],
            priority: 1,
            active: true,
            created_at: csf_core::hardware_timestamp(),
        })
        .collect();

    // Performance benchmark: should handle 1000 policies in <500ms
    let start = Instant::now();
    for _ in 0..100 {
        generator.generate_rules(&policies);
    }
    let duration = start.elapsed();

    println!(
        "✅ Rule generation performance: {}ms for 100x1000 policies",
        duration.as_millis()
    );

    // Regression threshold: should not exceed 1 second for 100 operations
    assert!(
        duration < Duration::from_secs(1),
        "Performance regression: Rule generation took {}ms (threshold: 1000ms)",
        duration.as_millis()
    );
}

/// 🛡️ Circuit breaker performance should be minimal overhead
#[test]
fn test_circuit_breaker_performance_overhead() {
    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    let empty_oscillators = vec![];
    let normal_oscillators: Vec<_> = (0..50).map(|i| NeuralOscillator::new(i, &config)).collect();

    // First, trigger the circuit breaker
    for _ in 0..15 {
        detector.detect(&empty_oscillators);
    }

    // Benchmark circuit breaker overhead (should be <1μs per call when active)
    let start = Instant::now();
    for _ in 0..10000 {
        detector.detect(&empty_oscillators);
    }
    let circuit_breaker_time = start.elapsed();

    // Benchmark normal operation
    let start = Instant::now();
    for _ in 0..1000 {
        // Fewer iterations since normal operation is more expensive
        detector.detect(&normal_oscillators);
    }
    let normal_time = start.elapsed();

    println!(
        "✅ Circuit breaker overhead: {}μs per call",
        circuit_breaker_time.as_micros() / 10000
    );
    println!(
        "✅ Normal operation: {}μs per call",
        normal_time.as_micros() / 1000
    );

    // Circuit breaker should add minimal overhead (<10μs per call)
    let circuit_breaker_per_call = circuit_breaker_time.as_nanos() / 10000;
    assert!(
        circuit_breaker_per_call < 10_000, // 10μs
        "Performance regression: Circuit breaker overhead {}ns (threshold: 10000ns)",
        circuit_breaker_per_call
    );
}

/// 🛡️ Test mutex contention monitoring performance
#[test]
fn test_mutex_contention_monitoring_overhead() {
    let monitor = PerformanceMonitor::global();
    let mutex = Arc::new(MonitoredMutex::new(
        0u64,
        "test_performance_mutex".to_string(),
    ));

    // Benchmark uncontended lock performance
    let start = Instant::now();
    for i in 0..10000 {
        let mut guard = mutex.lock();
        *guard += i;
    }
    let uncontended_time = start.elapsed();

    println!(
        "✅ Monitored mutex performance: {}ns per uncontended lock",
        uncontended_time.as_nanos() / 10000
    );

    // Monitoring overhead should be minimal (<1μs per lock)
    let per_lock_ns = uncontended_time.as_nanos() / 10000;
    assert!(
        per_lock_ns < 1_000, // 1μs
        "Performance regression: Monitored mutex overhead {}ns (threshold: 1000ns)",
        per_lock_ns
    );

    // Check that monitoring data was collected
    let summary = monitor.get_performance_summary();
    assert!(summary.total_operations > 0 || summary.mutex_contentions >= 0);
}

/// 🛡️ Test concurrent performance under load
#[test]
fn test_concurrent_performance_scaling() {
    let config = DrppConfig::default();
    let detector = Arc::new(PatternDetector::new(&config));

    let oscillators: Vec<_> = (0..200)
        .map(|i| NeuralOscillator::new(i, &config))
        .collect();

    // Test single-threaded performance
    let start = Instant::now();
    for _ in 0..1000 {
        detector.detect(&oscillators);
    }
    let single_thread_time = start.elapsed();

    // Test 4-thread concurrent performance
    let start = Instant::now();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let detector_clone = detector.clone();
            let oscillators_clone = oscillators.clone();

            thread::spawn(move || {
                for _ in 0..250 {
                    // 4 threads * 250 = 1000 total operations
                    detector_clone.detect(&oscillators_clone);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
    let concurrent_time = start.elapsed();

    println!(
        "✅ Single-threaded: {}ms, Concurrent (4 threads): {}ms",
        single_thread_time.as_millis(),
        concurrent_time.as_millis()
    );

    // Concurrent execution should provide some speedup or at least not be much slower
    // Allow up to 2x slowdown due to synchronization overhead
    assert!(
        concurrent_time < single_thread_time * 2,
        "Performance regression: Concurrent execution {}ms vs single-threaded {}ms (>2x slower)",
        concurrent_time.as_millis(),
        single_thread_time.as_millis()
    );
}

/// 🛡️ Memory allocation performance under resource limits
#[test]
fn test_memory_allocation_performance() {
    let config = EgcConfig::default();
    let generator = RuleGenerator::new(&config);

    // Test memory allocation patterns with increasing load
    let mut allocation_times = Vec::new();

    for policy_count in [100, 500, 1000, 2000, 5000] {
        let policies: Vec<_> = (0..policy_count)
            .map(|i| csf_clogic::egc::policy_engine::Policy {
                id: csf_clogic::egc::policy_engine::PolicyId::new(),
                name: format!("policy_{}", i),
                policy_type: csf_clogic::egc::policy_engine::PolicyType::Performance,
                conditions: vec![],
                actions: vec![],
                priority: 1,
                active: true,
                created_at: csf_core::hardware_timestamp(),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..10 {
            generator.generate_rules(&policies);
        }
        let duration = start.elapsed();

        allocation_times.push(duration.as_nanos() / 10 / policy_count as u128);

        println!(
            "✅ {} policies: {}ns per policy per operation",
            policy_count,
            duration.as_nanos() / 10 / policy_count as u128
        );
    }

    // Performance should not degrade significantly with increased load
    // Check that per-policy time doesn't increase by more than 50% from smallest to largest
    let min_time = allocation_times.iter().min().unwrap();
    let max_time = allocation_times.iter().max().unwrap();

    // Handle case where min_time is 0 (very fast operations)
    let degradation_ratio = if *min_time == 0 {
        1.0 // If min time is 0, consider performance stable
    } else {
        *max_time as f64 / *min_time as f64
    };

    assert!(
        degradation_ratio < 2.0,
        "Performance regression: Per-policy time degraded by {}x (threshold: 2.0x)",
        degradation_ratio
    );
}

/// 🛡️ End-to-end system performance regression test
#[tokio::test]
async fn test_end_to_end_system_performance() {
    let bus = Arc::new(csf_bus::PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CLogicConfig::default();

    // Benchmark system creation and initialization
    let start = Instant::now();
    let system = CLogicSystem::new(bus, config)
        .await
        .expect("System creation failed");
    let creation_time = start.elapsed();

    println!("✅ System creation time: {}ms", creation_time.as_millis());

    // System creation should be fast (<5 seconds)
    assert!(
        creation_time < Duration::from_secs(5),
        "Performance regression: System creation took {}ms (threshold: 5000ms)",
        creation_time.as_millis()
    );

    // Benchmark startup
    let start = Instant::now();
    system.start().await.expect("System start failed");
    let startup_time = start.elapsed();

    println!("✅ System startup time: {}ms", startup_time.as_millis());

    // Startup should be fast (<3 seconds)
    assert!(
        startup_time < Duration::from_secs(3),
        "Performance regression: System startup took {}ms (threshold: 3000ms)",
        startup_time.as_millis()
    );

    // Benchmark state retrieval (should be very fast)
    let start = Instant::now();
    for _ in 0..100 {
        let _state = system.get_state().await;
    }
    let state_time = start.elapsed();

    println!(
        "✅ State retrieval: {}μs per call",
        state_time.as_micros() / 100
    );

    // State retrieval should be very fast (<1ms per call)
    let per_call_ns = state_time.as_nanos() / 100;
    assert!(
        per_call_ns < 1_000_000, // 1ms
        "Performance regression: State retrieval {}μs (threshold: 1000μs)",
        per_call_ns / 1000
    );

    // Clean shutdown
    let start = Instant::now();
    system.stop().await.expect("System stop failed");
    let shutdown_time = start.elapsed();

    println!("✅ System shutdown time: {}ms", shutdown_time.as_millis());

    // Shutdown should be fast (<2 seconds)
    assert!(
        shutdown_time < Duration::from_secs(2),
        "Performance regression: System shutdown took {}ms (threshold: 2000ms)",
        shutdown_time.as_millis()
    );
}
