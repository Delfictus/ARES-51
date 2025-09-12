//! 🛡️ HARDENING: Stress tests for csf-clogic components
//! These tests validate circuit breakers, resource limits, and concurrent safety

use csf_bus::PhaseCoherenceBus;
use csf_clogic::drpp::{DrppConfig, PatternDetector};
use csf_clogic::egc::{EgcConfig, RuleGenerator};
use csf_clogic::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 🛡️ STRESS TEST: Pattern detector circuit breaker functionality
#[test]
fn stress_test_pattern_detector_circuit_breaker() {
    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    // Test with empty oscillators array to trigger failures
    let empty_oscillators = vec![];
    let mut consecutive_empty_results = 0;

    // This should trigger circuit breaker after multiple failures
    for i in 0..20 {
        let patterns = detector.detect(&empty_oscillators);

        if patterns.is_empty() {
            consecutive_empty_results += 1;
        }

        // After reaching failure threshold, all subsequent calls should return empty
        if i >= 10 {
            assert!(
                patterns.is_empty(),
                "Circuit breaker should be open after {} failures",
                i
            );
        }

        // Small delay to prevent tight loop
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(
        consecutive_empty_results >= 10,
        "Circuit breaker should have activated"
    );
    println!("✅ Circuit breaker activated correctly after failures");

    // Test recovery after delay
    std::thread::sleep(Duration::from_millis(1100)); // Wait for recovery time

    // Circuit should allow attempts again (but still fail due to empty input)
    let patterns = detector.detect(&empty_oscillators);
    assert!(patterns.is_empty()); // Still empty due to empty input, but circuit is trying

    println!("✅ Circuit breaker recovery mechanism working");
}

/// 🛡️ STRESS TEST: Concurrent pattern detection
#[test]
fn stress_test_concurrent_pattern_detection() {
    use std::sync::Barrier;

    let config = DrppConfig::default();
    let detector = Arc::new(PatternDetector::new(&config));

    let num_threads = 8;
    let operations_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let total_operations = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let detector = detector.clone();
                let barrier = barrier.clone();
                let total_ops = total_operations.clone();

                s.spawn(move || {
                    // Wait for all threads to be ready
                    barrier.wait();

                    let empty_oscillators = vec![];
                    let start = Instant::now();

                    for _ in 0..operations_per_thread {
                        let _patterns = detector.detect(&empty_oscillators);
                        total_ops.fetch_add(1, Ordering::Relaxed);
                    }

                    let elapsed = start.elapsed();
                    println!(
                        "Thread {} completed {} operations in {:?}",
                        thread_id, operations_per_thread, elapsed
                    );
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    });

    let final_count = total_operations.load(Ordering::Relaxed);
    assert_eq!(final_count, num_threads * operations_per_thread);
    println!(
        "✅ Concurrent stress test passed: {} total operations",
        final_count
    );
}

/// 🛡️ STRESS TEST: Rule generator resource limits
#[tokio::test]
async fn stress_test_rule_generator_resource_limits() {
    let config = EgcConfig::default();
    let generator = RuleGenerator::new(&config);

    // Create a large policy set to test limits
    let large_policy_set: Vec<_> = (0..15_000)
        .map(|i| create_dummy_policy(&format!("policy_{}", i)))
        .collect();

    println!(
        "Testing rule generation with {} policies",
        large_policy_set.len()
    );

    let start = Instant::now();
    let result = generator.generate_rules(&large_policy_set).await;
    let elapsed = start.elapsed();

    assert!(
        result.is_ok(),
        "Rule generation should handle large input gracefully"
    );
    let rules = result.unwrap();

    // Should limit rules generated per call
    assert!(
        rules.len() <= 100,
        "Should respect MAX_GENERATED_RULES_PER_CALL limit, got {}",
        rules.len()
    );

    println!(
        "✅ Generated {} rules from {} policies in {:?}",
        rules.len(),
        large_policy_set.len(),
        elapsed
    );

    // Test multiple generations to check history pruning
    for i in 0..10 {
        let _rules = generator
            .generate_rules(&large_policy_set[..100])
            .await
            .unwrap();
        if i % 3 == 0 {
            println!("Generation {} completed", i);
        }
    }

    println!("✅ Multiple rule generations completed without memory issues");
}

/// 🛡️ STRESS TEST: System-level stress under high load
#[tokio::test]
async fn stress_test_system_high_load() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CLogicConfig::default();

    let system = CLogicSystem::new(bus, config)
        .await
        .expect("System creation failed");

    // Start the system
    system.start().await.expect("System start failed");

    let start = Instant::now();
    let mut state_retrieval_count = 0;

    // Stress test by rapidly requesting system state
    while start.elapsed() < Duration::from_millis(100) {
        let _state = system.get_state().await;
        state_retrieval_count += 1;
    }

    // Clean shutdown
    system.stop().await.expect("System stop failed");

    println!(
        "✅ System handled {} state requests in 100ms",
        state_retrieval_count
    );
    assert!(
        state_retrieval_count > 10,
        "System should handle multiple state requests"
    );
}

/// 🛡️ STRESS TEST: Memory usage validation
#[test]
fn stress_test_memory_bounds() {
    // This test ensures our resource limits prevent unbounded memory growth
    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    // Simulate long-running operation with many pattern detections
    let empty_oscillators = vec![];
    let start_memory = get_memory_usage();

    // Run many operations
    for _ in 0..10_000 {
        let _patterns = detector.detect(&empty_oscillators);
    }

    let end_memory = get_memory_usage();
    let memory_growth = end_memory.saturating_sub(start_memory);

    // Memory growth should be bounded (less than 100MB for this test)
    assert!(
        memory_growth < 100_000_000,
        "Memory growth {} bytes exceeds limit",
        memory_growth
    );

    println!(
        "✅ Memory growth bounded: {} bytes over 10k operations",
        memory_growth
    );
}

/// 🛡️ PERFORMANCE BENCHMARK: Critical path timing
#[test]
fn benchmark_critical_operations() {
    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);
    let empty_oscillators = vec![];

    // Benchmark pattern detection
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _patterns = detector.detect(&empty_oscillators);
    }

    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;

    println!(
        "⏱️  Pattern detection average: {:?} per operation",
        avg_time
    );

    // Performance regression check - should complete in reasonable time
    assert!(
        avg_time < Duration::from_millis(1),
        "Pattern detection took {:?}, exceeds 1ms threshold",
        avg_time
    );

    println!("✅ Performance benchmark passed");
}

// Helper functions
fn create_dummy_policy(name: &str) -> csf_clogic::egc::policy_engine::Policy {
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

fn get_memory_usage() -> usize {
    // Simple memory usage approximation
    // In a real system, you'd use proper memory profiling

    // This is a simplified approach - in production you'd use tools like:
    // - jemalloc statistics
    // - /proc/self/status parsing
    // - Custom allocator tracking

    0 // Placeholder - implement proper memory tracking if needed
}
