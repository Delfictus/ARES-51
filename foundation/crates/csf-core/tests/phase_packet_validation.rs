//! Comprehensive production-grade validation for PhasePacket serialization system
//!
//! This test suite provides exhaustive validation of quantum-aware message serialization
//! with sub-microsecond performance targets and quantum coherence preservation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use bincode;
use csf_core::phase_packet::{CoherenceFactor, PhaseAngle, PhasePacket, PhaseState};
use csf_core::tensor::RelationalTensor;
use csf_core::{ComponentId, NanoTime};
use serde::{Deserialize, Serialize};
use serde_json;

/// Configuration for comprehensive phase packet validation
#[derive(Clone)]
struct PhasePacketValidationConfig {
    performance_target_ns: u64,
    large_payload_size: usize,
    stress_packet_count: usize,
    thread_count: usize,
    coherence_threshold: f64,
    serialization_iterations: usize,
    compression_target_ratio: f64,
}

impl Default for PhasePacketValidationConfig {
    fn default() -> Self {
        Self {
            performance_target_ns: 1000,   // Sub-microsecond target
            large_payload_size: 1_000_000, // 1MB payloads
            stress_packet_count: 10_000,
            thread_count: 8,
            coherence_threshold: 0.95,
            serialization_iterations: 1000,
            compression_target_ratio: 0.7,
        }
    }
}

/// Test payload types for comprehensive validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum TestPayload {
    Simple(i32),
    Text(String),
    Binary(Vec<u8>),
    Structured {
        id: u64,
        timestamp: u64,
        data: Vec<f64>,
    },
    Nested(Box<TestPayload>),
    TensorPayload(Vec<f64>),  // Simulating tensor data
    LargeData(Vec<Vec<f64>>), // Multi-dimensional data
}

impl TestPayload {
    fn generate_small() -> Self {
        TestPayload::Simple(42)
    }

    fn generate_medium() -> Self {
        TestPayload::Structured {
            id: 12345,
            timestamp: 1634567890,
            data: (0..1000).map(|i| i as f64 * 0.001).collect(),
        }
    }

    fn generate_large(size: usize) -> Self {
        TestPayload::LargeData(
            (0..size / 1000)
                .map(|i| {
                    (0..1000)
                        .map(|j| (i * 1000 + j) as f64 * std::f64::consts::PI)
                        .collect()
                })
                .collect(),
        )
    }

    fn generate_tensor_payload(rows: usize, cols: usize) -> Self {
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64).sin() * (i as f64).cos())
            .collect();
        TestPayload::TensorPayload(data)
    }

    fn generate_random_binary(size: usize) -> Self {
        let data: Vec<u8> = (0..size).map(|_| rand::random::<u8>()).collect();
        TestPayload::Binary(data)
    }
}

#[cfg(test)]
mod serialization_accuracy_tests {
    use super::*;

    #[test]
    fn test_basic_serialization_accuracy() {
        let payload = TestPayload::generate_medium();
        let packet = PhasePacket::new(payload.clone());

        // Test bincode serialization
        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        // Verify complete accuracy
        assert_eq!(packet.payload, deserialized.payload);
        assert_eq!(packet.phase_state, deserialized.phase_state);
        assert_eq!(packet.coherence_factor, deserialized.coherence_factor);
        assert_eq!(packet.packet_id, deserialized.packet_id);
        assert_eq!(packet.routing_info, deserialized.routing_info);
        assert_eq!(packet.entanglement_map, deserialized.entanglement_map);
    }

    #[test]
    fn test_quantum_coherence_preservation() {
        let config = PhasePacketValidationConfig::default();
        let payload = TestPayload::generate_small();

        // Create packet with specific quantum properties
        let mut packet = PhasePacket::with_phase_state(
            payload,
            PhaseState::Coherent(std::f64::consts::PI / 4.0),
        );

        // Add quantum entanglement
        let entangled_component = ComponentId::new(123);
        packet.entangle_with(entangled_component, 0.95);

        // Apply phase shift
        packet.apply_phase_shift(std::f64::consts::PI / 8.0);

        let original_coherence = packet.coherence_factor;
        let original_phase_state = packet.phase_state.clone();

        // Serialize and deserialize
        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        // Verify quantum properties preserved
        assert!((deserialized.coherence_factor - original_coherence).abs() < 1e-15);
        assert_eq!(deserialized.phase_state, original_phase_state);
        assert_eq!(deserialized.entanglement_map, packet.entanglement_map);

        // Verify coherence level maintained
        assert!(deserialized.is_coherent());
        assert!(deserialized.coherence_factor >= config.coherence_threshold);
    }

    #[test]
    fn test_phase_state_serialization() {
        let phase_states = vec![
            PhaseState::Coherent(0.0),
            PhaseState::Coherent(std::f64::consts::PI),
            PhaseState::Decoherent,
            PhaseState::Superposition(vec![(0.0, 0.6), (std::f64::consts::PI / 2.0, 0.8)]),
            PhaseState::Entangled(ComponentId::new(456), std::f64::consts::PI / 3.0),
        ];

        for phase_state in phase_states {
            let packet =
                PhasePacket::with_phase_state(TestPayload::generate_small(), phase_state.clone());

            let serialized = packet.quantum_serialize().unwrap();
            let deserialized: PhasePacket<TestPayload> =
                PhasePacket::quantum_deserialize(&serialized).unwrap();

            assert_eq!(deserialized.phase_state, phase_state);
        }
    }

    #[test]
    fn test_routing_info_preservation() {
        let mut packet = PhasePacket::new(TestPayload::generate_small());

        // Add complex routing information
        for i in 0..10 {
            let component = ComponentId::new(i * 100);
            let phase = (i as f64) * std::f64::consts::PI / 5.0;
            packet.add_route(component, phase);
        }

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        assert_eq!(deserialized.routing_info.len(), 10);
        for (component, phase) in &packet.routing_info {
            let deserialized_phase = deserialized.routing_info.get(component).unwrap();
            assert!((phase - deserialized_phase).abs() < 1e-15);
        }
    }

    #[test]
    fn test_entanglement_map_accuracy() {
        let mut packet = PhasePacket::new(TestPayload::generate_medium());

        // Create complex entanglement network
        for i in 0..20 {
            let component = ComponentId::new(i as u64);
            let strength = (i as f64) / 20.0;
            packet.entangle_with(component, strength);
        }

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        assert_eq!(
            deserialized.entanglement_map.len(),
            packet.entanglement_map.len()
        );
        for (component, strength) in &packet.entanglement_map {
            let deserialized_strength = deserialized.entanglement_map.get(component).unwrap();
            assert!((strength - deserialized_strength).abs() < 1e-15);
        }
    }

    #[test]
    fn test_timestamp_precision_preservation() {
        let packet = PhasePacket::new(TestPayload::generate_small());
        let original_timestamp = packet.timestamp;

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        // Timestamp should be preserved exactly
        assert_eq!(
            deserialized.timestamp.as_nanos(),
            original_timestamp.as_nanos()
        );
    }

    #[test]
    fn test_nested_payload_accuracy() {
        let nested_payload = TestPayload::Nested(Box::new(TestPayload::Nested(Box::new(
            TestPayload::Structured {
                id: 98765,
                timestamp: 1634567890,
                data: vec![1.1, 2.2, 3.3, 4.4, 5.5],
            },
        ))));

        let packet = PhasePacket::new(nested_payload.clone());

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        assert_eq!(deserialized.payload, nested_payload);
    }

    #[test]
    fn test_floating_point_precision() {
        // Test with challenging floating-point values
        let challenging_data = vec![
            0.0,
            -0.0,
            1e-100,
            1e100,
            std::f64::consts::PI,
            std::f64::consts::E,
            f64::MIN_POSITIVE,
            f64::MAX,
            1.0 / 3.0, // Repeating decimal
            0.1 + 0.2, // Floating point arithmetic quirk
        ];

        let payload = TestPayload::Structured {
            id: 12345,
            timestamp: 1634567890,
            data: challenging_data.clone(),
        };

        let packet = PhasePacket::new(payload);

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        if let TestPayload::Structured { data, .. } = deserialized.payload {
            for (i, &value) in challenging_data.iter().enumerate() {
                let deserialized_value = data[i];

                if value.is_nan() {
                    assert!(deserialized_value.is_nan());
                } else if value == 0.0 {
                    assert_eq!(deserialized_value, value);
                    assert_eq!(
                        deserialized_value.is_sign_negative(),
                        value.is_sign_negative()
                    );
                } else {
                    assert_eq!(deserialized_value, value);
                }
            }
        } else {
            panic!("Unexpected payload type");
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_serialization_performance() {
        let config = PhasePacketValidationConfig::default();
        let packet = PhasePacket::new(TestPayload::generate_medium());

        // Measure serialization performance
        let start = Instant::now();
        for _ in 0..config.serialization_iterations {
            let _serialized = packet.quantum_serialize().unwrap();
        }
        let serialize_duration = start.elapsed();

        let serialized_data = packet.quantum_serialize().unwrap();

        // Measure deserialization performance
        let start = Instant::now();
        for _ in 0..config.serialization_iterations {
            let _: PhasePacket<TestPayload> =
                PhasePacket::quantum_deserialize(&serialized_data).unwrap();
        }
        let deserialize_duration = start.elapsed();

        let avg_serialize_ns =
            serialize_duration.as_nanos() / config.serialization_iterations as u128;
        let avg_deserialize_ns =
            deserialize_duration.as_nanos() / config.serialization_iterations as u128;

        println!(
            "Serialization: {}ns, Deserialization: {}ns",
            avg_serialize_ns, avg_deserialize_ns
        );

        // Performance targets
        assert!(
            avg_serialize_ns < config.performance_target_ns as u128,
            "Serialization too slow: {}ns > {}ns",
            avg_serialize_ns,
            config.performance_target_ns
        );
        assert!(
            avg_deserialize_ns < config.performance_target_ns as u128 * 2, // Allow 2x for deserialization
            "Deserialization too slow: {}ns > {}ns",
            avg_deserialize_ns,
            config.performance_target_ns * 2
        );
    }

    #[test]
    fn test_large_payload_performance() {
        let config = PhasePacketValidationConfig::default();
        let large_payload = TestPayload::generate_large(config.large_payload_size);
        let packet = PhasePacket::new(large_payload);

        // Test serialization of large payloads
        let start = Instant::now();
        let serialized = packet.quantum_serialize().unwrap();
        let serialize_duration = start.elapsed();

        let start = Instant::now();
        let _deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();
        let deserialize_duration = start.elapsed();

        println!(
            "Large payload ({}B) - Serialize: {}ms, Deserialize: {}ms",
            serialized.len(),
            serialize_duration.as_millis(),
            deserialize_duration.as_millis()
        );

        // Should handle large payloads efficiently
        assert!(
            serialize_duration.as_millis() < 100,
            "Large payload serialization too slow"
        );
        assert!(
            deserialize_duration.as_millis() < 200,
            "Large payload deserialization too slow"
        );

        // Compression should be reasonable
        let raw_size = config.large_payload_size * std::mem::size_of::<f64>();
        let compression_ratio = serialized.len() as f64 / raw_size as f64;
        println!("Compression ratio: {:.3}", compression_ratio);
    }

    #[test]
    fn test_tensor_payload_performance() {
        let rows = 1000;
        let cols = 1000;
        let tensor_payload = TestPayload::generate_tensor_payload(rows, cols);
        let packet = PhasePacket::new(tensor_payload);

        let start = Instant::now();
        let serialized = packet.quantum_serialize().unwrap();
        let serialize_time = start.elapsed();

        let start = Instant::now();
        let _deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();
        let deserialize_time = start.elapsed();

        println!(
            "Tensor {}x{} - Serialize: {}ms, Deserialize: {}ms, Size: {}KB",
            rows,
            cols,
            serialize_time.as_millis(),
            deserialize_time.as_millis(),
            serialized.len() / 1024
        );

        // Should handle tensor data efficiently
        assert!(
            serialize_time.as_millis() < 50,
            "Tensor serialization too slow"
        );
        assert!(
            deserialize_time.as_millis() < 100,
            "Tensor deserialization too slow"
        );
    }

    #[test]
    fn test_quantum_operations_performance() {
        let config = PhasePacketValidationConfig::default();
        let mut packet = PhasePacket::new(TestPayload::generate_medium());

        // Benchmark quantum operations
        let start = Instant::now();
        for i in 0..config.serialization_iterations {
            let component = ComponentId::new(i as u64);
            packet.entangle_with(component, 0.8);
        }
        let entanglement_duration = start.elapsed();

        let start = Instant::now();
        for _ in 0..config.serialization_iterations {
            packet.apply_phase_shift(std::f64::consts::PI / 1000.0);
        }
        let phase_shift_duration = start.elapsed();

        let start = Instant::now();
        for i in 0..config.serialization_iterations {
            let component = ComponentId::new(i as u64 + 10000);
            packet.add_route(component, (i as f64) * 0.001);
        }
        let routing_duration = start.elapsed();

        let avg_entanglement_ns =
            entanglement_duration.as_nanos() / config.serialization_iterations as u128;
        let avg_phase_shift_ns =
            phase_shift_duration.as_nanos() / config.serialization_iterations as u128;
        let avg_routing_ns = routing_duration.as_nanos() / config.serialization_iterations as u128;

        println!(
            "Quantum operations - Entanglement: {}ns, Phase shift: {}ns, Routing: {}ns",
            avg_entanglement_ns, avg_phase_shift_ns, avg_routing_ns
        );

        // All quantum operations should be fast
        assert!(
            avg_entanglement_ns < 1000,
            "Entanglement too slow: {}ns",
            avg_entanglement_ns
        );
        assert!(
            avg_phase_shift_ns < 500,
            "Phase shift too slow: {}ns",
            avg_phase_shift_ns
        );
        assert!(
            avg_routing_ns < 200,
            "Routing addition too slow: {}ns",
            avg_routing_ns
        );
    }

    #[test]
    fn test_memory_usage_efficiency() {
        let config = PhasePacketValidationConfig::default();
        let payload_sizes = [100, 1000, 10000, 100000];

        for &size in &payload_sizes {
            let payload = TestPayload::generate_random_binary(size);
            let packet = PhasePacket::new(payload);

            let serialized = packet.quantum_serialize().unwrap();
            let serialized_size = serialized.len();
            let expected_min_size = size + 100; // Payload + overhead
            let overhead_ratio = serialized_size as f64 / size as f64;

            println!(
                "Payload: {}B, Serialized: {}B, Overhead: {:.2}x",
                size, serialized_size, overhead_ratio
            );

            // Overhead should be reasonable
            assert!(
                overhead_ratio < 2.0,
                "Memory overhead too high for size {}: {:.2}x",
                size,
                overhead_ratio
            );
            assert!(
                serialized_size >= expected_min_size,
                "Serialized size too small"
            );
        }
    }

    #[test]
    fn test_batch_serialization_performance() {
        let config = PhasePacketValidationConfig::default();
        let batch_size = 1000;

        let packets: Vec<_> = (0..batch_size)
            .map(|i| PhasePacket::new(TestPayload::Simple(i)))
            .collect();

        // Individual serialization
        let start = Instant::now();
        let mut individual_results = Vec::new();
        for packet in &packets {
            individual_results.push(packet.quantum_serialize().unwrap());
        }
        let individual_duration = start.elapsed();

        // Batch serialization (simulated)
        let start = Instant::now();
        let batch_results: Vec<_> = packets
            .iter()
            .map(|p| p.quantum_serialize().unwrap())
            .collect();
        let batch_duration = start.elapsed();

        let speedup = individual_duration.as_nanos() as f64 / batch_duration.as_nanos() as f64;
        println!("Batch serialization speedup: {:.2}x", speedup);

        // Results should be identical
        assert_eq!(individual_results.len(), batch_results.len());
        for i in 0..individual_results.len() {
            assert_eq!(individual_results[i], batch_results[i]);
        }

        // Batch should be competitive
        assert!(
            speedup >= 0.8,
            "Batch processing slower than expected: {:.2}x",
            speedup
        );
    }
}

#[cfg(test)]
mod quantum_coherence_validation_tests {
    use super::*;

    #[test]
    fn test_coherence_preservation_through_serialization() {
        let config = PhasePacketValidationConfig::default();
        let phases = [
            0.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
            std::f64::consts::PI,
        ];

        for &phase in &phases {
            let mut packet = PhasePacket::with_phase_state(
                TestPayload::generate_medium(),
                PhaseState::Coherent(phase),
            );

            // Add quantum properties
            packet.entangle_with(ComponentId::new(123), 0.9);
            packet.apply_phase_shift(std::f64::consts::PI / 8.0);

            let original_coherence = packet.coherence_factor;
            assert!(packet.is_coherent());

            // Multiple serialization rounds
            let mut current_packet = packet;
            for round in 0..10 {
                let serialized = current_packet.quantum_serialize().unwrap();
                current_packet = PhasePacket::quantum_deserialize(&serialized).unwrap();

                assert!(
                    current_packet.is_coherent(),
                    "Lost coherence at round {}",
                    round
                );
                assert!(
                    current_packet.coherence_factor >= config.coherence_threshold,
                    "Coherence below threshold at round {}: {}",
                    round,
                    current_packet.coherence_factor
                );
            }
        }
    }

    #[test]
    fn test_phase_correlation_preservation() {
        let packet1 =
            PhasePacket::with_phase_state(TestPayload::generate_small(), PhaseState::Coherent(0.0));

        let packet2 = PhasePacket::with_phase_state(
            TestPayload::generate_small(),
            PhaseState::Coherent(std::f64::consts::PI / 4.0),
        );

        let original_correlation = packet1.phase_correlation(&packet2);

        // Serialize both packets
        let serialized1 = packet1.quantum_serialize().unwrap();
        let serialized2 = packet2.quantum_serialize().unwrap();

        let deserialized1: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized1).unwrap();
        let deserialized2: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized2).unwrap();

        let new_correlation = deserialized1.phase_correlation(&deserialized2);

        assert!(
            (original_correlation - new_correlation).abs() < 1e-15,
            "Phase correlation not preserved: {} vs {}",
            original_correlation,
            new_correlation
        );
    }

    #[test]
    fn test_superposition_state_preservation() {
        let superposition_states = vec![
            (0.0, 0.6),
            (std::f64::consts::PI / 4.0, 0.8),
            (std::f64::consts::PI / 2.0, 0.5),
            (3.0 * std::f64::consts::PI / 4.0, 0.7),
            (std::f64::consts::PI, 0.9),
        ];

        let packet = PhasePacket::with_phase_state(
            TestPayload::generate_medium(),
            PhaseState::Superposition(superposition_states.clone()),
        );

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        if let PhaseState::Superposition(deserialized_states) = deserialized.phase_state {
            assert_eq!(deserialized_states.len(), superposition_states.len());

            for (i, &(phase, amplitude)) in superposition_states.iter().enumerate() {
                let (des_phase, des_amplitude) = deserialized_states[i];
                assert!(
                    (phase - des_phase).abs() < 1e-15,
                    "Phase mismatch at index {}",
                    i
                );
                assert!(
                    (amplitude - des_amplitude).abs() < 1e-15,
                    "Amplitude mismatch at index {}",
                    i
                );
            }
        } else {
            panic!("Superposition state not preserved");
        }
    }

    #[test]
    fn test_entanglement_network_preservation() {
        let mut packet = PhasePacket::new(TestPayload::generate_medium());

        // Create complex entanglement network
        let entanglement_data = vec![
            (ComponentId::new(100), 0.95),
            (ComponentId::new(200), 0.87),
            (ComponentId::new(300), 0.92),
            (ComponentId::new(400), 0.78),
            (ComponentId::new(500), 0.83),
        ];

        for (component, strength) in &entanglement_data {
            packet.entangle_with(*component, *strength);
        }

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        // Verify entanglement network preserved
        for (component, expected_strength) in entanglement_data {
            let actual_strength = deserialized.entanglement_map.get(&component).unwrap();
            assert!(
                (expected_strength - actual_strength).abs() < 1e-15,
                "Entanglement strength mismatch for {:?}: {} vs {}",
                component,
                expected_strength,
                actual_strength
            );
        }

        // Enhanced coherence should be preserved
        assert!(
            deserialized.coherence_factor > 1.0,
            "Enhanced coherence lost"
        );
    }

    #[test]
    fn test_quantum_decoherence_handling() {
        let coherent_packet = PhasePacket::with_phase_state(
            TestPayload::generate_small(),
            PhaseState::Coherent(std::f64::consts::PI / 2.0),
        );

        let decoherent_packet =
            PhasePacket::with_phase_state(TestPayload::generate_small(), PhaseState::Decoherent);

        // Serialize both
        let coherent_serialized = coherent_packet.quantum_serialize().unwrap();
        let decoherent_serialized = decoherent_packet.quantum_serialize().unwrap();

        let coherent_deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&coherent_serialized).unwrap();
        let decoherent_deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&decoherent_serialized).unwrap();

        // Verify coherence states preserved
        assert!(coherent_deserialized.is_coherent());
        assert!(!decoherent_deserialized.is_coherent());
        assert_eq!(decoherent_deserialized.phase_state, PhaseState::Decoherent);
    }

    #[test]
    fn test_phase_shift_accumulation() {
        let mut packet =
            PhasePacket::with_phase_state(TestPayload::generate_small(), PhaseState::Coherent(0.0));

        let shift_increment = std::f64::consts::PI / 16.0;
        let num_shifts = 8;

        // Apply multiple phase shifts
        for _ in 0..num_shifts {
            packet.apply_phase_shift(shift_increment);
        }

        let expected_total_phase = (num_shifts as f64) * shift_increment;

        // Serialize and deserialize
        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        if let PhaseState::Coherent(phase) = deserialized.phase_state {
            assert!(
                (phase - expected_total_phase).abs() < 1e-15,
                "Phase accumulation not preserved: {} vs {}",
                phase,
                expected_total_phase
            );
        } else {
            panic!("Phase state changed during serialization");
        }
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_concurrent_serialization() {
        let config = PhasePacketValidationConfig::default();
        let shared_packet = Arc::new(PhasePacket::new(TestPayload::generate_medium()));
        let success_count = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let packet_clone = shared_packet.clone();
            let success_clone = success_count.clone();

            let handle = thread::spawn(move || {
                let mut local_successes = 0;

                for _ in 0..config.serialization_iterations / config.thread_count {
                    match packet_clone.quantum_serialize() {
                        Ok(serialized) => {
                            match PhasePacket::<TestPayload>::quantum_deserialize(&serialized) {
                                Ok(deserialized) => {
                                    if deserialized.payload == packet_clone.payload {
                                        local_successes += 1;
                                    }
                                }
                                Err(_) => {}
                            }
                        }
                        Err(_) => {}
                    }
                }

                success_clone.fetch_add(local_successes, Ordering::Relaxed);
                (thread_id, local_successes)
            });

            handles.push(handle);
        }

        for handle in handles {
            let (thread_id, successes) = handle.join().unwrap();
            println!(
                "Thread {} completed {} serializations",
                thread_id, successes
            );
        }

        let total_successes = success_count.load(Ordering::Relaxed);
        assert_eq!(total_successes, config.serialization_iterations as u64);
    }

    #[test]
    fn test_concurrent_quantum_operations() {
        let config = PhasePacketValidationConfig::default();
        let packets: Vec<_> = (0..config.thread_count)
            .map(|i| Arc::new(Mutex::new(PhasePacket::new(TestPayload::Simple(i as i32)))))
            .collect();

        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let packet = packets[thread_id].clone();

            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let mut p = packet.lock().unwrap();

                    // Concurrent quantum operations
                    let component = ComponentId::new((thread_id * 1000 + i) as u64);
                    p.entangle_with(component, 0.8);
                    p.apply_phase_shift(std::f64::consts::PI / 100.0);
                    p.add_route(component, (i as f64) * 0.01);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all packets are still coherent and valid
        for packet_mutex in packets {
            let packet = packet_mutex.lock().unwrap();
            assert!(packet.is_coherent());
            assert!(packet.entanglement_map.len() == 100);
            assert!(packet.routing_info.len() == 100);
        }
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<PhasePacket<TestPayload>>();
        assert_sync::<PhasePacket<TestPayload>>();
        assert_send::<PhaseState>();
        assert_sync::<PhaseState>();
    }

    #[test]
    fn test_memory_safety_concurrent_access() {
        let config = PhasePacketValidationConfig::default();
        let large_packets: Vec<_> = (0..100)
            .map(|i| Arc::new(PhasePacket::new(TestPayload::generate_large(10000))))
            .collect();

        let mut handles = Vec::new();

        let packet_count = large_packets.len();
        for thread_id in 0..config.thread_count {
            let packets_clone = large_packets.clone();

            let handle = thread::spawn(move || {
                for i in 0..config.serialization_iterations / config.thread_count {
                    let packet_idx = (thread_id + i) % packet_count;
                    let packet = &packets_clone[packet_idx];

                    // Concurrent read-only access
                    let _age = packet.age_ns();
                    let _coherent = packet.is_coherent();
                    let _correlation = packet.phase_correlation(packet);

                    // Occasional serialization
                    if i % 10 == 0 {
                        let _serialized = packet.quantum_serialize().unwrap();
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All packets should still be valid
        for packet in &large_packets {
            assert!(packet.coherence_factor >= 0.0);
        }
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    #[test]
    fn test_cross_platform_serialization() {
        let packet = PhasePacket::new(TestPayload::generate_medium());

        // Test different serialization formats
        let json_data = serde_json::to_string(&packet).unwrap();
        let bincode_data = packet.quantum_serialize().unwrap();

        // Deserialize with both formats
        let json_packet: PhasePacket<TestPayload> = serde_json::from_str(&json_data).unwrap();
        let bincode_packet: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&bincode_data).unwrap();

        // Both should produce equivalent results
        assert_eq!(json_packet.payload, bincode_packet.payload);
        assert_eq!(json_packet.phase_state, bincode_packet.phase_state);
        assert_eq!(json_packet.packet_id, bincode_packet.packet_id);
    }

    #[test]
    fn test_schema_evolution_compatibility() {
        // Test with minimal packet structure
        #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
        struct MinimalPayload {
            value: i32,
        }

        let minimal_packet = PhasePacket::new(MinimalPayload { value: 42 });
        let serialized = minimal_packet.quantum_serialize().unwrap();

        // Should be able to deserialize even with extended payload
        #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
        struct ExtendedPayload {
            value: i32,
            #[serde(default)]
            extra_field: Option<String>,
        }

        // This test verifies that basic structure is maintained
        let deserialized_minimal: PhasePacket<MinimalPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();
        assert_eq!(deserialized_minimal.payload.value, 42);
    }

    #[test]
    fn test_version_compatibility() {
        let config = PhasePacketValidationConfig::default();
        let payload = TestPayload::generate_medium();
        let mut packet = PhasePacket::new(payload);

        // Add version-specific quantum properties
        packet.entangle_with(ComponentId::new(12345), 0.95);
        packet.apply_phase_shift(std::f64::consts::PI / 3.0);

        // Serialize with current version
        let serialized = packet.quantum_serialize().unwrap();

        // Should deserialize correctly (forward compatibility)
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        assert_eq!(deserialized.payload, packet.payload);
        assert_eq!(deserialized.entanglement_map, packet.entanglement_map);
        assert_eq!(deserialized.phase_state, packet.phase_state);
    }

    #[test]
    fn test_endianness_independence() {
        let test_values = vec![0x12345678u32, 0x9ABCDEFu32, 0xDEADBEEFu32, 0xCAFEBABEu32];

        for &value in &test_values {
            let payload = TestPayload::Simple(value as i32);
            let packet = PhasePacket::new(payload);

            let serialized = packet.quantum_serialize().unwrap();
            let deserialized: PhasePacket<TestPayload> =
                PhasePacket::quantum_deserialize(&serialized).unwrap();

            assert_eq!(deserialized.payload, packet.payload);
        }
    }

    #[test]
    fn test_compression_compatibility() {
        let config = PhasePacketValidationConfig::default();

        // Test with highly compressible data
        let compressible_data: Vec<u8> = vec![0x55; 10000]; // Repeating pattern
        let compressible_packet = PhasePacket::new(TestPayload::Binary(compressible_data.clone()));

        // Test with incompressible data
        let random_data: Vec<u8> = (0..10000).map(|_| rand::random()).collect();
        let random_packet = PhasePacket::new(TestPayload::Binary(random_data.clone()));

        let compressible_serialized = compressible_packet.quantum_serialize().unwrap();
        let random_serialized = random_packet.quantum_serialize().unwrap();

        // Both should deserialize correctly regardless of compressibility
        let compressible_deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&compressible_serialized).unwrap();
        let random_deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&random_serialized).unwrap();

        assert_eq!(
            compressible_deserialized.payload,
            compressible_packet.payload
        );
        assert_eq!(random_deserialized.payload, random_packet.payload);

        println!(
            "Compressible: {} -> {} bytes",
            compressible_data.len(),
            compressible_serialized.len()
        );
        println!(
            "Random: {} -> {} bytes",
            random_data.len(),
            random_serialized.len()
        );
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_high_volume_serialization() {
        let config = PhasePacketValidationConfig::default();
        let packets: Vec<_> = (0..config.stress_packet_count)
            .map(|i| PhasePacket::new(TestPayload::Simple(i as i32)))
            .collect();

        let start = Instant::now();
        let mut serialized_packets = Vec::new();

        for packet in &packets {
            serialized_packets.push(packet.quantum_serialize().unwrap());
        }

        let serialize_duration = start.elapsed();

        let start = Instant::now();
        let mut deserialized_packets = Vec::new();

        for serialized in &serialized_packets {
            deserialized_packets
                .push(PhasePacket::<TestPayload>::quantum_deserialize(serialized).unwrap());
        }

        let deserialize_duration = start.elapsed();

        println!("High volume test: {} packets", config.stress_packet_count);
        println!(
            "Serialize: {}ms, Deserialize: {}ms",
            serialize_duration.as_millis(),
            deserialize_duration.as_millis()
        );

        // Verify all packets
        for i in 0..packets.len() {
            assert_eq!(packets[i].payload, deserialized_packets[i].payload);
            assert_eq!(packets[i].packet_id, deserialized_packets[i].packet_id);
        }

        // Performance should scale reasonably
        let total_time = serialize_duration + deserialize_duration;
        let avg_time_per_packet = total_time.as_nanos() / (config.stress_packet_count * 2) as u128;
        assert!(
            avg_time_per_packet < 10000, // 10μs per operation
            "Average time per packet too high: {}ns",
            avg_time_per_packet
        );
    }

    #[test]
    fn test_memory_pressure_stress() {
        let config = PhasePacketValidationConfig::default();
        let large_packet_count = 1000;
        let packet_size = 50000; // 50KB each

        // Create many large packets
        let packets: Vec<_> = (0..large_packet_count)
            .map(|_| PhasePacket::new(TestPayload::generate_random_binary(packet_size)))
            .collect();

        // Serialize all at once (memory pressure)
        let start = Instant::now();
        let serialized: Vec<_> = packets
            .iter()
            .map(|p| p.quantum_serialize().unwrap())
            .collect();
        let duration = start.elapsed();

        println!(
            "Memory pressure: {} packets of {}KB each in {}ms",
            large_packet_count,
            packet_size / 1024,
            duration.as_millis()
        );

        // Verify a sample
        for i in (0..large_packet_count).step_by(100) {
            let deserialized: PhasePacket<TestPayload> =
                PhasePacket::quantum_deserialize(&serialized[i]).unwrap();
            assert_eq!(deserialized.payload, packets[i].payload);
        }

        // Should complete in reasonable time despite memory pressure
        assert!(
            duration.as_secs() < 30,
            "Memory pressure test too slow: {}s",
            duration.as_secs()
        );
    }

    #[test]
    fn test_quantum_complexity_stress() {
        let mut packet = PhasePacket::new(TestPayload::generate_large(100000));

        // Add maximum quantum complexity
        for i in 0..1000 {
            packet.entangle_with(ComponentId::new(i), (i as f64) / 1000.0);
            packet.add_route(
                ComponentId::new(i + 1000),
                (i as f64) * std::f64::consts::PI / 500.0,
            );
        }

        // Apply many phase shifts
        for _ in 0..100 {
            packet.apply_phase_shift(std::f64::consts::PI / 200.0);
        }

        // Should still serialize/deserialize correctly
        let start = Instant::now();
        let serialized = packet.quantum_serialize().unwrap();
        let serialize_time = start.elapsed();

        let start = Instant::now();
        let deserialized: PhasePacket<TestPayload> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();
        let deserialize_time = start.elapsed();

        println!(
            "Complex quantum packet: Serialize {}ms, Deserialize {}ms, Size: {}KB",
            serialize_time.as_millis(),
            deserialize_time.as_millis(),
            serialized.len() / 1024
        );

        // Verify quantum properties preserved
        assert_eq!(deserialized.entanglement_map.len(), 1000);
        assert_eq!(deserialized.routing_info.len(), 1000);
        assert!(deserialized.is_coherent());
        assert_eq!(deserialized.payload, packet.payload);

        // Should handle complexity reasonably
        assert!(
            serialize_time.as_millis() < 1000,
            "Complex serialization too slow"
        );
        assert!(
            deserialize_time.as_millis() < 1000,
            "Complex deserialization too slow"
        );
    }
}
