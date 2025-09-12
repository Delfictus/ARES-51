//! Performance validation tests for TTW Temporal Task Weaver
//!
//! These tests validate that the TTW system meets its performance requirements:
//! - Sub-microsecond latency for critical operations
//! - >1M operations per second throughput
//! - Consistent performance under load
//!
//! Unlike benchmarks, these tests are part of the regular test suite and will
//! fail if performance requirements are not met.

use csf_time::clock::HlcClock;
use csf_time::deadline::{Task, TaskPriority};
use csf_time::*;
use std::sync::Arc;
use std::time::Instant;

/// Performance thresholds for TTW operations
struct PerformanceThresholds {
    /// Maximum allowed latency for time source operations (nanoseconds)
    time_source_max_latency_ns: u64,
    /// Maximum allowed latency for HLC clock operations (nanoseconds)
    hlc_clock_max_latency_ns: u64,
    /// Maximum allowed latency for deadline scheduling (nanoseconds)
    scheduler_max_latency_ns: u64,
    /// Minimum required throughput (operations per second)
    min_throughput_ops_sec: u64,
}

impl PerformanceThresholds {
    /// Production performance thresholds as specified in NovaCore requirements
    fn production() -> Self {
        Self {
            time_source_max_latency_ns: 1_000, // 1μs
            hlc_clock_max_latency_ns: 1_000,   // 1μs
            scheduler_max_latency_ns: 5_000,   // 5μs (slightly higher due to complexity)
            min_throughput_ops_sec: 1_000_000, // 1M ops/sec
        }
    }

    /// Relaxed thresholds for CI environments
    #[allow(dead_code)]
    fn ci_environment() -> Self {
        Self {
            time_source_max_latency_ns: 10_000, // 10μs
            hlc_clock_max_latency_ns: 10_000,   // 10μs
            scheduler_max_latency_ns: 50_000,   // 50μs
            min_throughput_ops_sec: 100_000,    // 100K ops/sec
        }
    }
}

/// Measure the latency of a single operation
fn measure_latency<F, R>(operation: F) -> (R, u64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = operation();
    let elapsed = start.elapsed().as_nanos() as u64;
    (result, elapsed)
}

/// Measure throughput by running operations for a fixed duration
fn measure_throughput<F>(operation: F, duration_ms: u64) -> u64
where
    F: Fn(),
{
    let start = Instant::now();
    let duration = std::time::Duration::from_millis(duration_ms);
    let mut operations = 0u64;

    while start.elapsed() < duration {
        operation();
        operations += 1;
    }

    // Convert to operations per second
    (operations * 1000) / duration_ms
}

#[test]
fn test_time_source_latency_performance() {
    let thresholds = PerformanceThresholds::production();

    // Test simulated time source (most commonly used in tests)
    let time_source = SimulatedTimeSource::new(NanoTime::from_nanos(1000));

    // Warm up
    for _ in 0..100 {
        let _ = time_source.now_ns();
    }

    // Measure latency over multiple operations
    let mut max_latency = 0u64;
    let mut total_latency = 0u64;
    let num_operations = 1000;

    for _ in 0..num_operations {
        let (_, latency) =
            measure_latency(|| time_source.now_ns().expect("Time source should work"));

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_latency = total_latency / num_operations;

    println!(
        "TimeSource Performance: avg={}ns, max={}ns, target={}ns",
        avg_latency, max_latency, thresholds.time_source_max_latency_ns
    );

    // Validate performance requirements
    assert!(
        avg_latency <= thresholds.time_source_max_latency_ns,
        "TimeSource average latency {}ns exceeds target {}ns",
        avg_latency,
        thresholds.time_source_max_latency_ns
    );

    // Allow some outliers but max should be reasonable
    assert!(
        max_latency <= thresholds.time_source_max_latency_ns * 10,
        "TimeSource max latency {}ns is excessive (>10x target {}ns)",
        max_latency,
        thresholds.time_source_max_latency_ns
    );
}

#[test]
fn test_production_time_source_latency_performance() {
    let thresholds = PerformanceThresholds::production();

    // Test production time source
    let time_source = TimeSourceImpl::new().expect("Failed to create production time source");

    // Warm up
    for _ in 0..100 {
        let _ = time_source.now_ns();
    }

    // Measure latency over multiple operations
    let mut max_latency = 0u64;
    let mut total_latency = 0u64;
    let num_operations = 1000;

    for _ in 0..num_operations {
        let (_, latency) =
            measure_latency(|| time_source.now_ns().expect("Time source should work"));

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_latency = total_latency / num_operations;

    println!(
        "Production TimeSource Performance: avg={}ns, max={}ns, target={}ns",
        avg_latency, max_latency, thresholds.time_source_max_latency_ns
    );

    // Production time source may be slower due to system calls, so we allow higher latency
    let production_threshold = thresholds.time_source_max_latency_ns * 50; // Allow 50μs for production

    assert!(
        avg_latency <= production_threshold,
        "Production TimeSource average latency {}ns exceeds relaxed target {}ns",
        avg_latency,
        production_threshold
    );
}

#[test]
fn test_hlc_clock_latency_performance() {
    let thresholds = PerformanceThresholds::production();
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");

    // Warm up
    for _ in 0..100 {
        let _ = hlc_clock.current_time();
    }

    // Test current_time() operation latency
    let mut max_latency = 0u64;
    let mut total_latency = 0u64;
    let num_operations = 1000;

    for _ in 0..num_operations {
        let (_, latency) = measure_latency(|| hlc_clock.current_time().expect("HLC should work"));

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_latency = total_latency / num_operations;

    println!(
        "HLC Clock current_time() Performance: avg={}ns, max={}ns, target={}ns",
        avg_latency, max_latency, thresholds.hlc_clock_max_latency_ns
    );

    assert!(
        avg_latency <= thresholds.hlc_clock_max_latency_ns,
        "HLC Clock current_time() average latency {}ns exceeds target {}ns",
        avg_latency,
        thresholds.hlc_clock_max_latency_ns
    );

    // Test update() operation latency
    let message_time = LogicalTime::new(2000, 100, 2);
    max_latency = 0;
    total_latency = 0;

    for _ in 0..num_operations {
        let (_, latency) = measure_latency(|| {
            hlc_clock
                .update(message_time)
                .expect("HLC update should work")
        });

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_update_latency = total_latency / num_operations;

    println!(
        "HLC Clock update() Performance: avg={}ns, max={}ns, target={}ns",
        avg_update_latency, max_latency, thresholds.hlc_clock_max_latency_ns
    );

    // Update operations are more complex, allow 2x threshold
    let update_threshold = thresholds.hlc_clock_max_latency_ns * 2;
    assert!(
        avg_update_latency <= update_threshold,
        "HLC Clock update() average latency {}ns exceeds relaxed target {}ns",
        avg_update_latency,
        update_threshold
    );
}

#[test]
fn test_deadline_scheduler_latency_performance() {
    let thresholds = PerformanceThresholds::production();
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());

    // Warm up
    for i in 0..100 {
        let task = Task::new(
            format!("warmup_{}", i),
            TaskPriority::Normal,
            NanoTime::from_nanos(10000),
            Duration::from_micros(100),
        );
        let _ = scheduler.schedule_task(task, NanoTime::from_nanos(10000));
    }

    // Measure scheduling latency
    let mut max_latency = 0u64;
    let mut total_latency = 0u64;
    let num_operations = 1000;

    for i in 0..num_operations {
        let (_, latency) = measure_latency(|| {
            let task = Task::new(
                format!("perf_task_{}", i),
                TaskPriority::Normal,
                NanoTime::from_nanos(10000 + i * 100),
                Duration::from_micros(50),
            );
            scheduler.schedule_task(task, NanoTime::from_nanos(10000 + i * 100))
        });

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_latency = total_latency / num_operations;

    println!(
        "Deadline Scheduler Performance: avg={}ns, max={}ns, target={}ns",
        avg_latency, max_latency, thresholds.scheduler_max_latency_ns
    );

    assert!(
        avg_latency <= thresholds.scheduler_max_latency_ns,
        "Deadline Scheduler average latency {}ns exceeds target {}ns",
        avg_latency,
        thresholds.scheduler_max_latency_ns
    );
}

#[test]
fn test_quantum_oracle_latency_performance() {
    let thresholds = PerformanceThresholds::production();
    let oracle = QuantumTimeOracle::new();

    // Warm up
    for _ in 0..100 {
        let _ = oracle.current_offset();
    }

    // Measure quantum offset calculation latency
    let mut max_latency = 0u64;
    let mut total_latency = 0u64;
    let num_operations = 1000;

    for i in 0..num_operations {
        let (_, latency) =
            measure_latency(|| oracle.current_offset_with_time(NanoTime::from_nanos(1000 + i)));

        max_latency = max_latency.max(latency);
        total_latency += latency;
    }

    let avg_latency = total_latency / num_operations;

    println!(
        "Quantum Oracle Performance: avg={}ns, max={}ns, target={}ns",
        avg_latency, max_latency, thresholds.time_source_max_latency_ns
    );

    // Quantum oracle should be very fast (same target as time source)
    assert!(
        avg_latency <= thresholds.time_source_max_latency_ns,
        "Quantum Oracle average latency {}ns exceeds target {}ns",
        avg_latency,
        thresholds.time_source_max_latency_ns
    );
}

#[test]
fn test_throughput_performance() {
    let thresholds = PerformanceThresholds::production();
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));

    println!("Testing throughput over 1 second duration...");

    // Test TimeSource throughput
    let time_source_throughput = measure_throughput(
        || {
            let _ = time_source.now_ns();
        },
        1000,
    ); // 1 second

    println!(
        "TimeSource Throughput: {} ops/sec, target: {} ops/sec",
        time_source_throughput, thresholds.min_throughput_ops_sec
    );

    assert!(
        time_source_throughput >= thresholds.min_throughput_ops_sec,
        "TimeSource throughput {} ops/sec below target {} ops/sec",
        time_source_throughput,
        thresholds.min_throughput_ops_sec
    );

    // Test HLC Clock throughput
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");
    let hlc_throughput = measure_throughput(
        || {
            let _ = hlc_clock.current_time();
        },
        1000,
    );

    println!(
        "HLC Clock Throughput: {} ops/sec, target: {} ops/sec",
        hlc_throughput, thresholds.min_throughput_ops_sec
    );

    // HLC operations are more complex, allow lower throughput
    let hlc_target = thresholds.min_throughput_ops_sec / 2; // 500K ops/sec
    assert!(
        hlc_throughput >= hlc_target,
        "HLC Clock throughput {} ops/sec below target {} ops/sec",
        hlc_throughput,
        hlc_target
    );

    // Test Quantum Oracle throughput
    let oracle = QuantumTimeOracle::new();
    let oracle_throughput = measure_throughput(
        || {
            let _ = oracle.current_offset();
        },
        1000,
    );

    println!(
        "Quantum Oracle Throughput: {} ops/sec, target: {} ops/sec",
        oracle_throughput, thresholds.min_throughput_ops_sec
    );

    assert!(
        oracle_throughput >= thresholds.min_throughput_ops_sec,
        "Quantum Oracle throughput {} ops/sec below target {} ops/sec",
        oracle_throughput,
        thresholds.min_throughput_ops_sec
    );
}

#[test]
fn test_integrated_ttw_performance() {
    let thresholds = PerformanceThresholds::production();
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());
    let oracle = QuantumTimeOracle::new();

    // Test integrated TTW operation cycle
    let integrated_throughput = measure_throughput(
        || {
            // Complete TTW cycle: time -> HLC -> quantum optimization -> scheduling
            let current_time = time_source.now_ns().unwrap_or(NanoTime::ZERO);
            let _logical_time = hlc_clock.current_time().unwrap();
            let quantum_offset = oracle.current_offset_with_time(current_time);
            let deadline = NanoTime::from_nanos(
                current_time
                    .as_nanos()
                    .saturating_add(Duration::from_millis(1).as_nanos()),
            );
            let optimized_deadline = quantum_offset.apply(deadline);

            let task = Task::new(
                "integrated_task".to_string(),
                TaskPriority::Normal,
                optimized_deadline,
                Duration::from_micros(100),
            );

            let _ = scheduler.schedule_task(task, optimized_deadline);
        },
        1000,
    );

    println!(
        "Integrated TTW Cycle Throughput: {} ops/sec",
        integrated_throughput
    );

    // Integrated operations are complex, expect lower throughput
    let integrated_target = thresholds.min_throughput_ops_sec / 10; // 100K ops/sec
    assert!(
        integrated_throughput >= integrated_target,
        "Integrated TTW throughput {} ops/sec below target {} ops/sec",
        integrated_throughput,
        integrated_target
    );

    println!("✅ All TTW performance requirements validated!");
}

#[test]
fn test_performance_consistency_under_load() {
    println!("Testing performance consistency under concurrent load...");

    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let num_threads = 4;
    let operations_per_thread = 10000;

    let mut handles = Vec::new();

    // Spawn concurrent threads
    for thread_id in 0..num_threads {
        let time_source_clone = time_source.clone();
        let handle = std::thread::spawn(move || {
            let mut latencies = Vec::new();

            for i in 0..operations_per_thread {
                let (_, latency) = measure_latency(|| {
                    time_source_clone.now_ns().expect("Time source should work")
                });

                latencies.push(latency);

                // Simulate some load
                if i % 1000 == 0 {
                    std::thread::yield_now();
                }
            }

            // Return statistics
            let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
            let max_latency = *latencies.iter().max().unwrap();
            let min_latency = *latencies.iter().min().unwrap();

            (thread_id, avg_latency, max_latency, min_latency)
        });
        handles.push(handle);
    }

    // Collect results
    let mut all_avg_latencies = Vec::new();
    let mut all_max_latencies = Vec::new();

    for handle in handles {
        let (thread_id, avg_latency, max_latency, min_latency) =
            handle.join().expect("Thread failed");
        println!(
            "Thread {}: avg={}ns, max={}ns, min={}ns",
            thread_id, avg_latency, max_latency, min_latency
        );

        all_avg_latencies.push(avg_latency);
        all_max_latencies.push(max_latency);
    }

    // Verify consistency across threads
    let overall_avg = all_avg_latencies.iter().sum::<u64>() / all_avg_latencies.len() as u64;
    let overall_max = *all_max_latencies.iter().max().unwrap();

    println!(
        "Concurrent Performance: overall_avg={}ns, overall_max={}ns",
        overall_avg, overall_max
    );

    // Check that performance doesn't degrade significantly under concurrent load
    let consistency_threshold = 10_000; // 10μs
    assert!(
        overall_avg <= consistency_threshold,
        "Performance degrades under concurrent load: avg={}ns > {}ns",
        overall_avg,
        consistency_threshold
    );

    println!("✅ Performance remains consistent under concurrent load!");
}
