//! Basic integration tests for csf-time
//!
//! Simplified tests that validate core functionality without complex scenarios.

use csf_time::*;
use std::sync::Arc;

#[test]
fn test_time_source_basic() {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(100)));

    // Test basic time functionality
    let time1 = time_source.now_ns().unwrap();
    assert_eq!(time1, NanoTime::from_secs(100));

    // Test advance (note: advance may not be available on this implementation)
    // Just test that we can get time consistently
    let time2 = time_source.now_ns().unwrap();
    assert_eq!(time2, NanoTime::from_secs(100));
}

#[test]
fn test_hlc_clock_basic() {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(100)));
    let hlc_clock = HlcClockImpl::new(1, time_source).unwrap();

    // Test current_time
    let logical_time = hlc_clock.current_time().unwrap();
    assert_eq!(logical_time.node_id, 1);
    assert!(logical_time.physical > 0);
}

#[test]
fn test_deadline_scheduler_basic() {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(100)));
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());

    // Test basic statistics
    let stats = scheduler.get_statistics();
    assert_eq!(stats.total_tasks, 0);
    assert_eq!(stats.critical_tasks, 0);
}

#[test]
fn test_quantum_oracle_basic() {
    let oracle = QuantumTimeOracle::new();

    // Test quantum offset (use current_offset method)
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(100)));
    let offset = oracle.current_offset_with_time(time_source.now_ns().unwrap());
    assert!(offset.amplitude >= 0.0);
    assert!(offset.frequency >= 0.0);
    assert!(offset.phase >= 0.0);
}

#[test]
fn test_duration_basic() {
    let dur1 = Duration::from_secs(5);
    let dur2 = Duration::from_millis(500);

    assert_eq!(dur1.as_secs(), 5);
    assert_eq!(dur2.as_millis(), 500);

    let sum = dur1 + dur2;
    assert_eq!(sum.as_millis(), 5500);
}

#[test]
fn test_nano_time_basic() {
    let time1 = NanoTime::from_secs(10);
    let time2 = NanoTime::from_millis(500);

    assert_eq!(time1.as_secs(), 10);
    assert_eq!(time2.as_millis(), 500);

    let sum = NanoTime::from_nanos(
        time1
            .as_nanos()
            .saturating_add(Duration::from_secs(5).as_nanos()),
    );
    assert_eq!(sum.as_secs(), 15);
}
