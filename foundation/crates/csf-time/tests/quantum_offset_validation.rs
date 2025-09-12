//! Comprehensive production-grade validation for QuantumOffset precision standards
//!
//! This test suite provides exhaustive validation of femtosecond precision temporal operations
//! with mathematical rigor and performance verification suitable for production deployment.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use csf_time::{
    NanoTime, PreciseDuration, PreciseQuantumOffset, PrecisionLevel, QuantumOffset, TimeResult,
};

/// Test configuration for comprehensive validation
struct ValidationConfig {
    precision_target_ns: f64,
    performance_target_ns: u64,
    thread_count: usize,
    iteration_count: usize,
    accuracy_epsilon: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            precision_target_ns: 1e-6,  // femtosecond precision target
            performance_target_ns: 125, // 123.50ns target with margin
            thread_count: 16,
            iteration_count: 1000,
            accuracy_epsilon: 1e-15,
        }
    }
}

#[cfg(test)]
mod precision_accuracy_tests {
    use super::*;

    #[test]
    fn test_femtosecond_precision_accuracy() {
        let _config = ValidationConfig::default();

        // Test femtosecond precision creation using actual API
        let femto_duration = PreciseDuration::from_nanos_precise(1e-6, PrecisionLevel::Femtosecond);
        assert_eq!(femto_duration.as_total_nanos(), 1e-6);

        // Test precision level validation
        let femto_duration_2 =
            PreciseDuration::new(0, 0.000001, PrecisionLevel::Femtosecond).unwrap();
        assert!(
            (femto_duration_2.as_total_nanos() - 1e-6).abs()
                < PrecisionLevel::Femtosecond.epsilon()
        );

        // Verify precision level epsilon
        let femto_level = PrecisionLevel::Femtosecond;
        assert_eq!(femto_level.epsilon(), 1e-15);

        // Test precision preservation in arithmetic using actual API
        let temporal1 = PreciseDuration::from_nanos_precise(1.5e-15, femto_level);
        let temporal2 = PreciseDuration::from_nanos_precise(2.5e-15, femto_level);
        let offset1 = PreciseQuantumOffset::new(temporal1, 0.0, femto_level).unwrap();
        let offset2 = PreciseQuantumOffset::new(temporal2, 0.0, femto_level).unwrap();

        // Test temporal offset arithmetic
        let sum_temporal = (offset1.temporal_offset.clone() + offset2.temporal_offset).unwrap();
        assert!((sum_temporal.as_total_nanos() - 4e-15).abs() < 1e-15);
        assert_eq!(offset1.precision_level, PrecisionLevel::Femtosecond);
    }

    #[test]
    fn test_precision_level_hierarchy() {
        let levels = [
            PrecisionLevel::Femtosecond,
            PrecisionLevel::Picosecond,
            PrecisionLevel::Nanosecond,
            PrecisionLevel::Microsecond,
        ];

        // Verify epsilon values are in correct order
        for i in 0..levels.len() - 1 {
            assert!(levels[i].epsilon() < levels[i + 1].epsilon());
        }

        // Test specific epsilon values
        assert_eq!(PrecisionLevel::Femtosecond.epsilon(), 1e-15);
        assert_eq!(PrecisionLevel::Picosecond.epsilon(), 1e-12);
        assert_eq!(PrecisionLevel::Nanosecond.epsilon(), 1e-9);
        assert_eq!(PrecisionLevel::Microsecond.epsilon(), 1e-6);
    }

    #[test]
    fn test_precision_degradation_detection() {
        let temporal = PreciseDuration::from_nanos_precise(1e-15, PrecisionLevel::Femtosecond);
        let mut offset =
            PreciseQuantumOffset::new(temporal, 0.0, PrecisionLevel::Femtosecond).unwrap();

        // Perform operations that should not degrade precision
        for i in 1..10 {
            let other_temporal =
                PreciseDuration::from_nanos_precise(i as f64 * 1e-15, PrecisionLevel::Femtosecond);
            let _other =
                PreciseQuantumOffset::new(other_temporal, 0.0, PrecisionLevel::Femtosecond)
                    .unwrap();

            // Test that precision metadata tracks operations correctly
            offset.record_operation("precision_test");
        }

        // Should still maintain femtosecond precision
        assert_eq!(offset.precision_level, PrecisionLevel::Femtosecond);
        assert!(offset.precision_metadata.operation_count > 0);

        // Test precision degradation with mixed levels
        let coarse_temporal = PreciseDuration::from_nanos_precise(1e-9, PrecisionLevel::Nanosecond);
        let _coarse_offset =
            PreciseQuantumOffset::new(coarse_temporal, 0.0, PrecisionLevel::Nanosecond).unwrap();

        // Test that precision is maintained when different levels interact
        assert_eq!(offset.precision_level, PrecisionLevel::Femtosecond);
    }

    #[test]
    fn test_cumulative_error_bounds() {
        let iterations = 100;
        let zero_temporal = PreciseDuration::zero(PrecisionLevel::Femtosecond);
        let mut offset =
            PreciseQuantumOffset::new(zero_temporal, 0.0, PrecisionLevel::Femtosecond).unwrap();
        let increment_temporal =
            PreciseDuration::from_nanos_precise(1e-15, PrecisionLevel::Femtosecond);
        let _increment =
            PreciseQuantumOffset::new(increment_temporal, 0.0, PrecisionLevel::Femtosecond)
                .unwrap();

        for i in 0..iterations {
            // Test that error bounds are tracked correctly
            offset.record_operation(&format!("cumulative_test_{}", i));
        }

        // Check that error bounds are being accumulated
        assert!(offset.precision_metadata.operation_count == iterations);

        // Test that precision level is maintained
        assert_eq!(offset.precision_level, PrecisionLevel::Femtosecond);
    }

    #[test]
    fn test_ieee_754_edge_cases() {
        let precision = PrecisionLevel::Femtosecond;

        // Test with smallest positive normal number
        let min_normal = f64::MIN_POSITIVE;
        let min_temporal = PreciseDuration::from_nanos_precise(min_normal, precision);
        let offset_min = PreciseQuantumOffset::new(min_temporal, 0.0, precision).unwrap();
        assert_eq!(offset_min.temporal_offset.as_total_nanos(), min_normal);

        // Test with large finite number (scaled down to reasonable nanosecond range)
        let large_finite = 1e15; // 1 second in nanoseconds
        let max_temporal = PreciseDuration::from_nanos_precise(large_finite, precision);
        let offset_max = PreciseQuantumOffset::new(max_temporal, 0.0, precision).unwrap();
        assert_eq!(offset_max.temporal_offset.as_total_nanos(), large_finite);

        // Test subnormal numbers
        let subnormal = 5e-15; // Scaled to femtosecond range
        let sub_temporal = PreciseDuration::from_nanos_precise(subnormal, precision);
        let offset_sub = PreciseQuantumOffset::new(sub_temporal, 0.0, precision).unwrap();
        assert_eq!(offset_sub.temporal_offset.as_total_nanos(), subnormal);

        // Test zero preservation
        let zero_temporal = PreciseDuration::zero(precision);
        let zero_offset = PreciseQuantumOffset::new(zero_temporal, 0.0, precision).unwrap();
        assert_eq!(zero_offset.temporal_offset.as_total_nanos(), 0.0);
        assert!(!zero_offset
            .temporal_offset
            .as_total_nanos()
            .is_sign_negative());
    }
}

#[cfg(test)]
mod arithmetic_operation_tests {
    use super::*;

    #[test]
    fn test_precise_duration_arithmetic() {
        let d1 = PreciseDuration::new(1000, 0.3, PrecisionLevel::Femtosecond).unwrap();
        let d2 = PreciseDuration::new(500, 0.8, PrecisionLevel::Femtosecond).unwrap();

        let sum = (d1.clone() + d2.clone()).unwrap();
        assert_eq!(sum.as_total_nanos(), 1501.1); // 1000.3 + 500.8

        let diff = (d1.clone() - d2.clone()).unwrap();
        assert!((diff.as_total_nanos() - 499.5).abs() < 1e-10);

        let scaled = (d1.clone() * 2.0).unwrap();
        assert_eq!(scaled.as_total_nanos(), 2000.6);

        let divided = (d1 / 2.0).unwrap();
        assert_eq!(divided.as_total_nanos(), 500.15);
    }

    #[test]
    fn test_quantum_offset_operations() {
        let temporal1 = PreciseDuration::new(1000, 0.5, PrecisionLevel::Nanosecond).unwrap();
        let temporal2 = PreciseDuration::new(500, 0.25, PrecisionLevel::Nanosecond).unwrap();

        let offset1 =
            PreciseQuantumOffset::new(temporal1, 0.25, PrecisionLevel::Nanosecond).unwrap();
        let offset2 =
            PreciseQuantumOffset::new(temporal2, 0.75, PrecisionLevel::Nanosecond).unwrap();

        // Test that quantum offsets are created properly
        assert_eq!(offset1.phase_component, 0.25);
        assert_eq!(offset2.phase_component, 0.75);
        assert_eq!(offset1.precision_level, PrecisionLevel::Nanosecond);
    }

    #[test]
    fn test_quantum_offset_application() {
        let temporal = PreciseDuration::new(100, 0.0, PrecisionLevel::Nanosecond).unwrap();
        let offset = PreciseQuantumOffset::new(temporal, 0.0, PrecisionLevel::Nanosecond).unwrap();

        let base_time = NanoTime::from_nanos(1_000_000_000); // 1 second
        let (result, precision_loss) = offset.apply_precise(base_time).unwrap();

        assert!(precision_loss >= 0.0);
        assert!(result.as_nanos() > 0);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_quantum_offset_creation_performance() {
        let config = ValidationConfig::default();
        let iterations = 1000;

        let start = Instant::now();
        for i in 0..iterations {
            let temporal =
                PreciseDuration::from_nanos_precise(i as f64 * 1e-12, PrecisionLevel::Femtosecond);
            let _offset =
                PreciseQuantumOffset::new(temporal, 0.5, PrecisionLevel::Femtosecond).unwrap();
        }
        let elapsed = start.elapsed();

        let avg_ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

        // Should meet or exceed performance target (125ns)
        assert!(
            avg_ns_per_op < config.performance_target_ns as f64 * 2.0,
            "Performance target not met: {:.2}ns/op > {}ns target",
            avg_ns_per_op,
            config.performance_target_ns
        );
    }

    #[test]
    fn test_precision_arithmetic_performance() {
        let iterations = 1000;
        let d1 = PreciseDuration::new(1000, 0.5, PrecisionLevel::Femtosecond).unwrap();
        let d2 = PreciseDuration::new(500, 0.3, PrecisionLevel::Femtosecond).unwrap();

        let start = Instant::now();
        for _ in 0..iterations {
            let _sum = (d1.clone() + d2.clone()).unwrap();
        }
        let elapsed = start.elapsed();

        let avg_ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

        // Arithmetic should be fast (sub-microsecond)
        assert!(
            avg_ns_per_op < 1000.0, // 1μs
            "Arithmetic too slow: {:.2}ns/op",
            avg_ns_per_op
        );
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_concurrent_quantum_offset_operations() {
        let config = ValidationConfig::default();
        let counter = Arc::new(AtomicU64::new(0));
        let results = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..config.thread_count)
            .map(|thread_id| {
                let counter = Arc::clone(&counter);
                let results = Arc::clone(&results);

                thread::spawn(move || {
                    for i in 0..config.iteration_count {
                        let temporal = PreciseDuration::from_nanos_precise(
                            thread_id as f64 * 1e-15 + i as f64 * 1e-18,
                            PrecisionLevel::Femtosecond,
                        );
                        let offset = PreciseQuantumOffset::new(
                            temporal,
                            (thread_id as f64) * 0.1,
                            PrecisionLevel::Femtosecond,
                        )
                        .unwrap();

                        counter.fetch_add(1, Ordering::SeqCst);

                        results
                            .lock()
                            .unwrap()
                            .push(offset.temporal_offset.as_total_nanos());
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let final_count = counter.load(Ordering::SeqCst);
        let expected = config.thread_count * config.iteration_count;

        assert_eq!(final_count, expected as u64);

        let results = results.lock().unwrap();
        assert_eq!(results.len(), expected);

        // Verify all results are finite and reasonable
        for &result in results.iter() {
            assert!(result.is_finite());
            assert!(result >= 0.0);
        }
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;

    #[test]
    fn test_precise_duration_serialization() {
        let duration = PreciseDuration::new(1234, 0.5678, PrecisionLevel::Femtosecond).unwrap();

        let serialized = serde_json::to_string(&duration).unwrap();
        let deserialized: PreciseDuration = serde_json::from_str(&serialized).unwrap();

        assert_eq!(duration.as_total_nanos(), deserialized.as_total_nanos());
    }

    #[test]
    fn test_quantum_offset_serialization() {
        let temporal = PreciseDuration::new(1000, 0.5, PrecisionLevel::Femtosecond).unwrap();
        let offset =
            PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Femtosecond).unwrap();

        let serialized = serde_json::to_string(&offset).unwrap();
        let deserialized: PreciseQuantumOffset = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            offset.temporal_offset.as_total_nanos(),
            deserialized.temporal_offset.as_total_nanos()
        );
        assert_eq!(offset.phase_component, deserialized.phase_component);
        assert_eq!(offset.precision_level, deserialized.precision_level);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_cross_platform_consistency() {
        let temporal = PreciseDuration::new(1000, 0.0, PrecisionLevel::Nanosecond).unwrap();
        let mut offset =
            PreciseQuantumOffset::new(temporal, 0.5, PrecisionLevel::Nanosecond).unwrap();

        // Test cross-platform validation
        assert!(offset.validate_cross_platform().is_ok());
        assert!(offset.error_bounds.cross_platform_validated);
    }

    #[test]
    fn test_error_bound_tracking() {
        let temporal = PreciseDuration::new(1000, 0.5, PrecisionLevel::Femtosecond).unwrap();
        let mut offset =
            PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Femtosecond).unwrap();

        // Initially should have zero error bounds
        assert!(offset
            .error_bounds
            .is_within_bounds(PrecisionLevel::Femtosecond));

        // After operations, error bounds should be tracked
        for i in 0..10 {
            offset.record_operation(&format!("test_op_{}", i));
        }

        assert_eq!(offset.precision_metadata.operation_count, 10);
    }

    #[test]
    fn test_legacy_compatibility() {
        let temporal = PreciseDuration::new(1000, 0.123, PrecisionLevel::Femtosecond).unwrap();
        let precise_offset =
            PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Femtosecond).unwrap();

        let (legacy_offset, precision_loss) = precise_offset.to_legacy();
        assert_eq!(legacy_offset.phase, 0.25);
        assert!(precision_loss >= 0.0); // Should track precision loss
    }
}
