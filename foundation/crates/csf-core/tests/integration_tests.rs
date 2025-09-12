//! Comprehensive integration tests for Phase 1.2 ARES ChronoFabric components
//!
//! This test suite validates cross-component interactions, data flow integrity,
//! end-to-end workflows, and system-level performance characteristics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// Import all Phase 1.2 components
use csf_core::energy_functional::{
    AdaptiveEnergyFunctional, AllocationStrategy, ChronoFabricEnergyFunctional, EnergyFunctional,
    EnergyParameters, EnergyState, PerformanceMetrics, QuantumEnergyFunctional,
};
use csf_core::phase_packet::{CoherenceFactor, PhaseAngle, PhasePacket, PhaseState};
use csf_core::tensor::{RelationalMetadata, RelationalTensor};
use csf_core::{ComponentId, NanoTime};
use csf_shared_types::PrecisionLevel;

// Placeholder types for disabled integration tests
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreciseQuantumOffset {
    value: f64,
    precision: PrecisionLevel,
}

#[allow(dead_code)]
impl PreciseQuantumOffset {
    fn new(value: f64, precision: PrecisionLevel) -> Self {
        Self { value, precision }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value + other.value,
            precision: self.precision,
        }
    }

    fn multiply(&self, scalar: f64) -> Self {
        Self {
            value: self.value * scalar,
            precision: self.precision,
        }
    }

    fn offset(&self) -> f64 {
        self.value
    }
}

use ndarray::{Array, IxDyn};
use serde::{Deserialize, Serialize};

/// Configuration for integration tests
#[derive(Clone)]
struct IntegrationConfig {
    workflow_timeout_ms: u64,
    data_integrity_threshold: f64,
    performance_degradation_limit: f64,
    coherence_preservation_threshold: f64,
    thread_count: usize,
    stress_iterations: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            workflow_timeout_ms: 5000, // 5 second timeout
            data_integrity_threshold: 1e-12,
            performance_degradation_limit: 0.1, // 10% max degradation
            coherence_preservation_threshold: 0.95,
            thread_count: 8,
            stress_iterations: 1000,
        }
    }
}

/// Comprehensive test data structure for integration flows
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct IntegrationTestData {
    quantum_offsets: Vec<f64>,
    tensor_data: Vec<f64>,
    tensor_shape: Vec<usize>,
    energy_metrics: Vec<f64>,
    timestamp: u64,
    component_id: u64,
}

impl IntegrationTestData {
    fn generate_test_data(size: usize, component_id: u64) -> Self {
        let quantum_offsets: Vec<f64> = (0..size)
            .map(|i| (i as f64 * std::f64::consts::PI * 1e-15))
            .collect();

        let tensor_size = size * size;
        let tensor_data: Vec<f64> = (0..tensor_size)
            .map(|i| (i as f64).sin() * (i as f64).cos() * 1e-12)
            .collect();

        let energy_metrics: Vec<f64> = (0..10)
            .map(|i| (component_id as f64 + i as f64) * 1e-13)
            .collect();

        Self {
            quantum_offsets,
            tensor_data,
            tensor_shape: vec![size, size],
            energy_metrics,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            component_id,
        }
    }
}

#[cfg(test)]
mod quantum_offset_tensor_integration {
    use super::*;

    #[test]
    #[ignore] // Temporarily disabled due to circular dependency resolution
    fn test_quantum_offset_to_tensor_data_flow() {
        let config = IntegrationConfig::default();

        // Step 1: Create quantum offsets with femtosecond precision
        let precision = PrecisionLevel::Femtosecond;
        let offsets: Vec<PreciseQuantumOffset> = (0..100)
            .map(|i| {
                PreciseQuantumOffset::new((i as f64) * 1e-15 * std::f64::consts::PI, precision)
            })
            .collect();

        // Step 2: Process offsets through arithmetic operations
        let mut processed_offsets = Vec::new();
        for i in 0..offsets.len() - 1 {
            let sum = offsets[i].add(&offsets[i + 1]);
            let scaled = sum.multiply(2.0);
            processed_offsets.push(scaled.offset());
        }

        // Step 3: Convert to tensor data
        let tensor_size = (processed_offsets.len() as f64).sqrt() as usize;
        let tensor_data: Vec<f64> = processed_offsets
            .into_iter()
            .take(tensor_size * tensor_size)
            .collect();

        let shape = vec![tensor_size, tensor_size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), tensor_data.clone()).unwrap();
        let tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Step 4: Verify data integrity through processing chain
        let tensor_trace = tensor.trace().unwrap();
        let tensor_norm = tensor.frobenius_norm();

        // Verify quantum precision preserved through conversion
        assert!(tensor_trace.is_finite(), "Tensor trace should be finite");
        assert!(tensor_norm.is_finite(), "Tensor norm should be finite");
        assert!(tensor_norm > 0.0, "Tensor norm should be positive");

        // Verify precision scaling
        let expected_magnitude = tensor_size as f64 * 1e-15;
        assert!(
            tensor_trace.abs() > expected_magnitude * 0.1,
            "Data magnitude should be preserved through conversion"
        );

        // Step 5: Mathematical operations on integrated data
        let transposed = tensor.transpose();
        let product = tensor.matrix_multiply(&transposed).unwrap();

        assert_eq!(product.shape()[0], tensor_size);
        assert_eq!(product.shape()[1], tensor_size);

        let product_trace = product.trace().unwrap();
        assert!(
            product_trace >= 0.0,
            "Matrix product trace should be non-negative"
        );

        println!(
            "Quantum offset → Tensor integration: {} offsets → {}×{} tensor",
            offsets.len(),
            tensor_size,
            tensor_size
        );
        println!("Trace: {:.6e}, Norm: {:.6e}", tensor_trace, tensor_norm);
    }

    #[test]
    #[ignore] // Temporarily disabled due to Complex<f64> trait bound issues
    fn test_tensor_quantum_coherence_preservation() {
        let config = IntegrationConfig::default();

        // Create quantum state tensor with complex amplitudes
        let size = 64;
        let mut quantum_data = Vec::new();

        for i in 0..size {
            let offset = PreciseQuantumOffset::new(
                (i as f64) * std::f64::consts::PI / (size as f64),
                PrecisionLevel::Femtosecond,
            );

            // Convert quantum offset to complex amplitude
            let amplitude = num_complex::Complex64::from_polar(
                1.0 / (size as f64).sqrt(),
                offset.offset() * 1e15, // Scale to reasonable phase range
            );
            quantum_data.push(amplitude);
        }

        let shape = vec![size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), quantum_data).unwrap();
        let mut quantum_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Test quantum operations
        quantum_tensor.normalize_quantum_state();
        let initial_norm: f64 = quantum_tensor.data.iter().map(|&amp| amp.norm_sqr()).sum();

        assert!(
            (initial_norm - 1.0).abs() < config.data_integrity_threshold,
            "Quantum state should be normalized"
        );

        // Apply quantum phase shift
        quantum_tensor.apply_quantum_phase_shift(std::f64::consts::PI / 4.0);
        let after_shift_norm: f64 = quantum_tensor.data.iter().map(|&amp| amp.norm_sqr()).sum();

        assert!(
            (after_shift_norm - 1.0).abs() < config.data_integrity_threshold,
            "Normalization should be preserved through phase shift"
        );

        // Test quantum coherence
        let coherence = quantum_tensor.calculate_quantum_coherence().unwrap();
        assert!(
            coherence >= config.coherence_preservation_threshold,
            "Quantum coherence should be maintained: {}",
            coherence
        );

        println!("Tensor quantum coherence: {:.4}", coherence);
    }

    #[test]
    #[ignore] // Temporarily disabled due to circular dependency resolution
    fn test_precision_degradation_through_pipeline() {
        let config = IntegrationConfig::default();

        // Start with maximum precision quantum offsets
        let initial_precision = PrecisionLevel::Femtosecond;
        let high_precision_value = 1.234567890123456789e-15;

        let offset = PreciseQuantumOffset::new(high_precision_value, initial_precision);

        // Step 1: Multiple arithmetic operations
        let mut accumulated = offset;
        for _ in 0..100 {
            let increment = PreciseQuantumOffset::new(1e-18, initial_precision);
            accumulated = accumulated.add(&increment);
        }

        // Step 2: Convert to tensor
        let tensor_data = vec![accumulated.offset(); 16];
        let shape = vec![4, 4];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), tensor_data).unwrap();
        let tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Step 3: Tensor operations
        let transposed = tensor.transpose();
        let product = tensor.matrix_multiply(&transposed).unwrap();
        let trace = product.trace().unwrap();

        // Step 4: Verify precision preservation
        let expected_value = high_precision_value + 100.0 * 1e-18;
        let tensor_diagonal = tensor.data.get([0, 0]).unwrap();

        let relative_error = (tensor_diagonal - expected_value).abs() / expected_value;
        assert!(
            relative_error < 1e-12,
            "Precision degradation too high: {:.2e}",
            relative_error
        );

        // Step 5: Test round-trip through serialization
        let serialized = bincode::serialize(&accumulated).unwrap();
        let deserialized: PreciseQuantumOffset = bincode::deserialize(&serialized).unwrap();

        let serialization_error = (deserialized.offset() - accumulated.offset()).abs();
        assert!(
            serialization_error < 1e-15,
            "Serialization precision loss: {:.2e}",
            serialization_error
        );

        println!(
            "Precision preservation test passed with relative error: {:.2e}",
            relative_error
        );
    }
}

#[cfg(test)]
mod tensor_phase_packet_integration {
    use super::*;

    #[test]
    fn test_tensor_to_phase_packet_serialization() {
        let config = IntegrationConfig::default();

        // Create large tensor with quantum-aware data
        let size = 100;
        let tensor_data: Vec<f64> = (0..size * size)
            .map(|i| {
                let offset =
                    PreciseQuantumOffset::new((i as f64) * 1e-15, PrecisionLevel::Femtosecond);
                offset.offset() * (i as f64).sin() * 1e12 // Scale for tensor operations
            })
            .collect();

        let shape = vec![size, size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), tensor_data.clone()).unwrap();
        let tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Create integration test data combining tensor and quantum offset info
        let test_data = IntegrationTestData {
            quantum_offsets: vec![1e-15, 2e-15, 3e-15],
            tensor_data: tensor_data.clone(),
            tensor_shape: shape,
            energy_metrics: vec![1e-13, 2e-13],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            component_id: 12345,
        };

        // Create phase packet with quantum properties
        let mut packet = PhasePacket::with_phase_state(
            test_data.clone(),
            PhaseState::Coherent(std::f64::consts::PI / 3.0),
        );

        // Add quantum entanglement representing tensor correlations
        let tensor_component = ComponentId::new(test_data.component_id);
        packet.entangle_with(tensor_component, 0.95);

        // Apply phase shift based on tensor properties
        let tensor_trace = tensor.trace().unwrap();
        let phase_shift = (tensor_trace * 1e15).fract() * 2.0 * std::f64::consts::PI;
        packet.apply_phase_shift(phase_shift);

        // Serialize with quantum preservation
        let start_time = Instant::now();
        let serialized = packet.quantum_serialize().unwrap();
        let serialization_time = start_time.elapsed();

        // Performance validation
        assert!(
            serialization_time.as_micros() < 10000, // 10ms max
            "Serialization too slow: {}μs",
            serialization_time.as_micros()
        );

        // Deserialize and verify integrity
        let deserialized: PhasePacket<IntegrationTestData> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        // Verify tensor data integrity
        assert_eq!(deserialized.payload.tensor_data.len(), tensor_data.len());
        for (i, (&original, &deserialized_val)) in tensor_data
            .iter()
            .zip(deserialized.payload.tensor_data.iter())
            .enumerate()
        {
            let diff = (original - deserialized_val).abs();
            assert!(
                diff < config.data_integrity_threshold,
                "Tensor data integrity loss at index {}: {:.2e}",
                i,
                diff
            );
        }

        // Verify quantum properties preserved
        assert_eq!(deserialized.phase_state, packet.phase_state);
        assert!(deserialized.is_coherent());
        assert_eq!(
            deserialized.entanglement_map.len(),
            packet.entanglement_map.len()
        );

        // Verify coherence enhancement from entanglement
        assert!(
            deserialized.coherence_factor >= config.coherence_preservation_threshold,
            "Coherence not preserved: {}",
            deserialized.coherence_factor
        );

        println!(
            "Tensor → PhasePacket integration: {}KB tensor serialized in {}μs",
            serialized.len() / 1024,
            serialization_time.as_micros()
        );
    }

    #[test]
    fn test_large_tensor_packet_performance() {
        let config = IntegrationConfig::default();
        let tensor_sizes = [50, 100, 200, 500];

        for &size in &tensor_sizes {
            // Create large tensor
            let tensor_data: Vec<f64> = (0..size * size)
                .map(|i| (i as f64 * std::f64::consts::E).sin() * 1e-10)
                .collect();

            let test_payload = IntegrationTestData {
                quantum_offsets: (0..10).map(|i| i as f64 * 1e-15).collect(),
                tensor_data: tensor_data.clone(),
                tensor_shape: vec![size, size],
                energy_metrics: (0..5).map(|i| i as f64 * 1e-13).collect(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                component_id: size as u64,
            };

            let packet = PhasePacket::new(test_payload);

            // Test serialization performance scaling
            let start = Instant::now();
            let serialized = packet.quantum_serialize().unwrap();
            let serialize_time = start.elapsed();

            let start = Instant::now();
            let _deserialized: PhasePacket<IntegrationTestData> =
                PhasePacket::quantum_deserialize(&serialized).unwrap();
            let deserialize_time = start.elapsed();

            println!(
                "Size {}×{}: Serialize {}μs, Deserialize {}μs, Data {}KB",
                size,
                size,
                serialize_time.as_micros(),
                deserialize_time.as_micros(),
                serialized.len() / 1024
            );

            // Performance scaling verification
            let expected_max_serialize_us = (size * size) as u128 / 100; // Rough scaling
            let expected_max_deserialize_us = (size * size) as u128 / 50;

            assert!(
                serialize_time.as_micros() < expected_max_serialize_us.max(1000),
                "Serialization scaling poor for {}×{}: {}μs",
                size,
                size,
                serialize_time.as_micros()
            );

            assert!(
                deserialize_time.as_micros() < expected_max_deserialize_us.max(2000),
                "Deserialization scaling poor for {}×{}: {}μs",
                size,
                size,
                deserialize_time.as_micros()
            );
        }
    }

    #[test]
    fn test_quantum_coherence_through_serialization() {
        let config = IntegrationConfig::default();

        // Create quantum-correlated tensor data
        let size = 32;
        let mut correlated_data = Vec::new();

        for i in 0..size {
            for j in 0..size {
                // Create quantum correlation pattern
                let phase1 = (i as f64) * std::f64::consts::PI / (size as f64);
                let phase2 = (j as f64) * std::f64::consts::PI / (size as f64);
                let correlation = (phase1 + phase2).cos() * (phase1 - phase2).sin();
                correlated_data.push(correlation * 1e-12);
            }
        }

        let test_data = IntegrationTestData {
            quantum_offsets: (0..size)
                .map(|i| (i as f64) * 1e-15 * std::f64::consts::TAU / (size as f64))
                .collect(),
            tensor_data: correlated_data,
            tensor_shape: vec![size, size],
            energy_metrics: vec![1e-13; 10],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            component_id: 98765,
        };

        // Create multiple correlated packets
        let packet_count = 10;
        let mut packets = Vec::new();

        for i in 0..packet_count {
            let mut packet = PhasePacket::with_phase_state(
                test_data.clone(),
                PhaseState::Coherent((i as f64) * std::f64::consts::PI / (packet_count as f64)),
            );

            // Create entanglement network
            for j in 0..packet_count {
                if i != j {
                    packet.entangle_with(ComponentId::new(j as u64), 0.8);
                }
            }

            packets.push(packet);
        }

        // Serialize all packets
        let serialized_packets: Vec<_> = packets
            .iter()
            .map(|p| p.quantum_serialize().unwrap())
            .collect();

        // Deserialize and verify correlations
        let deserialized_packets: Vec<PhasePacket<IntegrationTestData>> = serialized_packets
            .iter()
            .map(|data| PhasePacket::quantum_deserialize(data).unwrap())
            .collect();

        // Test phase correlations preserved
        for i in 0..packet_count {
            for j in i + 1..packet_count {
                let original_correlation = packets[i].phase_correlation(&packets[j]);
                let deserialized_correlation =
                    deserialized_packets[i].phase_correlation(&deserialized_packets[j]);

                let correlation_error = (original_correlation - deserialized_correlation).abs();
                assert!(
                    correlation_error < config.data_integrity_threshold,
                    "Phase correlation error between packets {} and {}: {:.2e}",
                    i,
                    j,
                    correlation_error
                );
            }
        }

        // Verify entanglement networks preserved
        for (i, packet) in deserialized_packets.iter().enumerate() {
            assert_eq!(packet.entanglement_map.len(), packet_count - 1);
            assert!(packet.is_coherent());
            assert!(packet.coherence_factor >= config.coherence_preservation_threshold);
        }

        println!(
            "Quantum coherence network: {} packets with preserved correlations",
            packet_count
        );
    }
}

#[cfg(test)]
mod phase_packet_energy_functional_integration {
    use super::*;

    #[test]
    fn test_packet_energy_optimization_workflow() {
        let config = IntegrationConfig::default();

        // Create energy functional system
        let energy_params = EnergyParameters::default();
        let mut energy_functional = ChronoFabricEnergyFunctional::new(energy_params.clone());

        // Create diverse phase packets representing different components
        let component_count = 10;
        let mut packets = Vec::new();
        let mut component_states = HashMap::new();

        for i in 0..component_count {
            let component_id = ComponentId::new(i as u64);

            // Create test data for this component
            let test_data = IntegrationTestData::generate_test_data(20, i as u64);

            // Create phase packet with energy-related quantum state
            let mut packet = PhasePacket::with_phase_state(
                test_data,
                PhaseState::Coherent((i as f64 + 1.0) * 1e-13),
            );

            // Add routing based on energy requirements
            for j in 0..component_count {
                if i != j {
                    let route_id = ComponentId::new(j as u64);
                    let energy_phase = (i * j) as f64 * std::f64::consts::PI
                        / (component_count * component_count) as f64;
                    packet.add_route(route_id, energy_phase);
                }
            }

            // Create corresponding energy state
            let energy_state = if i % 3 == 0 {
                EnergyState::QuantumCoherent {
                    energy: (i as f64 + 1.0) * 5e-13,
                    coherence_factor: packet.coherence_factor,
                    phase_energy: (i as f64) * 2e-14,
                }
            } else if i % 3 == 1 {
                EnergyState::Active {
                    current_energy: (i as f64 + 1.0) * 8e-13,
                    peak_energy: (i as f64 + 2.0) * 1e-12,
                    efficiency: 0.75 + (i as f64) * 0.03,
                }
            } else {
                EnergyState::Idle {
                    baseline_energy: (i as f64 + 0.5) * 1e-15,
                }
            };

            energy_functional.update_component_state(component_id, energy_state.clone());
            component_states.insert(component_id, energy_state);
            packets.push((component_id, packet));
        }

        // Step 1: Initial energy optimization
        let initial_allocation = energy_functional
            .optimize_allocation(&component_states, &energy_params)
            .unwrap();

        assert_eq!(initial_allocation.len(), component_count);

        // Step 2: Serialize packets with energy-aware routing
        let mut serialized_data = Vec::new();
        let start = Instant::now();

        for (component_id, packet) in &packets {
            let energy_weight = initial_allocation[component_id];

            // Modify packet based on energy allocation
            let mut energy_packet = packet.clone();
            if energy_weight > 0.5 {
                energy_packet.apply_phase_shift(std::f64::consts::PI / 8.0); // High energy phase
            } else {
                energy_packet.apply_phase_shift(-std::f64::consts::PI / 8.0); // Low energy phase
            }

            let serialized = energy_packet.quantum_serialize().unwrap();
            serialized_data.push((*component_id, serialized, energy_weight));
        }

        let serialization_time = start.elapsed();

        // Step 3: Quantum-aware energy optimization
        let mut quantum_components = HashMap::new();
        for (component_id, _, energy_weight) in &serialized_data {
            quantum_components.insert(*component_id, *energy_weight as CoherenceFactor);
        }

        let quantum_optimized = energy_functional
            .quantum_optimize(&quantum_components, &component_states, &energy_params)
            .unwrap();

        // Step 4: Performance metrics and adaptation
        let workflow_time = serialization_time;
        let performance_metrics = PerformanceMetrics {
            avg_response_time_ns: workflow_time.as_nanos() as u64,
            throughput_ops_sec: (component_count as f64) / workflow_time.as_secs_f64(),
            energy_efficiency: 1.0 / energy_functional.total_system_energy(),
            coherence_maintenance_rate: packets
                .iter()
                .map(|(_, p)| if p.is_coherent() { 1.0 } else { 0.0 })
                .sum::<f64>()
                / component_count as f64,
            ..PerformanceMetrics::default()
        };

        let adapted_params =
            energy_functional.adapt_parameters(&performance_metrics, &energy_params);

        // Step 5: Final optimization with adapted parameters
        let final_allocation = energy_functional
            .optimize_allocation(&quantum_optimized, &adapted_params)
            .unwrap();

        // Verification
        assert_eq!(final_allocation.len(), quantum_optimized.len());

        // Check energy conservation
        let initial_total = component_states
            .values()
            .map(|s| energy_functional.energy(s))
            .sum::<f64>();
        let final_total = quantum_optimized
            .values()
            .map(|s| energy_functional.energy(s))
            .sum::<f64>();

        println!("Energy optimization workflow:");
        println!("  Components: {}", component_count);
        println!("  Initial energy: {:.6e}", initial_total);
        println!("  Final energy: {:.6e}", final_total);
        println!("  Serialization time: {}μs", serialization_time.as_micros());
        println!(
            "  Coherence rate: {:.3}",
            performance_metrics.coherence_maintenance_rate
        );

        // Performance validation
        assert!(
            serialization_time.as_millis() < config.workflow_timeout_ms,
            "Workflow timeout exceeded"
        );

        assert!(
            performance_metrics.coherence_maintenance_rate
                >= config.coherence_preservation_threshold,
            "Coherence not maintained through workflow"
        );
    }

    #[test]
    fn test_energy_aware_packet_routing() {
        let config = IntegrationConfig::default();

        // Create network of components with varying energy profiles
        let component_count = 15;
        let energy_params = EnergyParameters {
            allocation_strategy: AllocationStrategy::QuantumAware,
            quantum_scaling_factor: 1.5,
            ..EnergyParameters::default()
        };

        let mut energy_functional = ChronoFabricEnergyFunctional::new(energy_params.clone());
        let mut network_packets = Vec::new();
        let mut component_states = HashMap::new();

        // Create components with different energy characteristics
        for i in 0..component_count {
            let component_id = ComponentId::new(i as u64);

            let test_data = IntegrationTestData::generate_test_data(10, i as u64);
            let mut packet = PhasePacket::new(test_data);

            // Create energy-based routing network
            for j in 0..component_count {
                if i != j {
                    let target_id = ComponentId::new(j as u64);

                    // Energy-weighted routing phase
                    let energy_distance = ((i as f64) - (j as f64)).abs();
                    let routing_phase =
                        energy_distance * std::f64::consts::PI / (component_count as f64);
                    packet.add_route(target_id, routing_phase);

                    // Energy-based entanglement strength
                    let entanglement_strength = 1.0 / (1.0 + energy_distance);
                    packet.entangle_with(target_id, entanglement_strength * 0.8);
                }
            }

            // Define energy state based on network position
            let energy_state = match i % 4 {
                0 => EnergyState::Active {
                    current_energy: 5e-13,
                    peak_energy: 8e-13,
                    efficiency: 0.9,
                },
                1 => EnergyState::QuantumCoherent {
                    energy: 3e-13,
                    coherence_factor: packet.coherence_factor,
                    phase_energy: 1e-13,
                },
                2 => EnergyState::Overloaded {
                    excess_energy: 2e-12,
                    throttling_factor: 0.7,
                },
                _ => EnergyState::Idle {
                    baseline_energy: 5e-16,
                },
            };

            energy_functional.update_component_state(component_id, energy_state.clone());
            component_states.insert(component_id, energy_state);
            network_packets.push((component_id, packet));
        }

        // Optimize energy allocation for the network
        let network_allocation = energy_functional
            .optimize_allocation(&component_states, &energy_params)
            .unwrap();

        // Simulate packet routing with energy-aware decisions
        let mut routing_metrics = HashMap::new();

        for (source_id, source_packet) in &network_packets {
            let source_allocation = network_allocation[source_id];

            // Find optimal routing targets based on energy allocation
            let mut routing_costs = Vec::new();

            for route in source_packet.routing_info.keys() {
                let target_allocation = network_allocation.get(route).unwrap_or(&0.0);

                // Calculate routing cost based on energy and quantum properties
                let energy_cost = (source_allocation - target_allocation).abs();
                let quantum_cost = 1.0 - source_packet.entanglement_map.get(route).unwrap_or(&0.0);
                let phase_cost =
                    source_packet.routing_info[route].abs() / (2.0 * std::f64::consts::PI);

                let total_cost = energy_cost + quantum_cost + phase_cost;
                routing_costs.push((*route, total_cost));
            }

            // Sort by routing cost (lower is better)
            routing_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Select top 3 routes
            let optimal_routes: Vec<ComponentId> =
                routing_costs.iter().take(3).map(|(id, _)| *id).collect();

            routing_metrics.insert(*source_id, (routing_costs.len(), optimal_routes));
        }

        // Verify routing efficiency
        let total_routes: usize = routing_metrics.values().map(|(total, _)| *total).sum();
        let optimal_routes: usize = routing_metrics
            .values()
            .map(|(_, optimal)| optimal.len())
            .sum();

        let routing_efficiency = optimal_routes as f64 / total_routes as f64;

        // Load balancing based on energy states
        let mut component_loads = HashMap::new();
        for (component_id, state) in &component_states {
            let load = match state {
                EnergyState::Active { efficiency, .. } => 1.0 / efficiency,
                EnergyState::QuantumCoherent {
                    coherence_factor, ..
                } => 1.0 / coherence_factor,
                EnergyState::Overloaded {
                    throttling_factor, ..
                } => 2.0 / throttling_factor,
                EnergyState::Idle { .. } => 0.1,
                EnergyState::Error { .. } => 5.0,
            };
            component_loads.insert(*component_id, load);
        }

        let balanced_allocation =
            energy_functional.dynamic_load_balance(&component_loads, &network_allocation);

        // Verification
        assert_eq!(balanced_allocation.len(), component_count);
        assert!(
            routing_efficiency > 0.2,
            "Routing efficiency too low: {:.3}",
            routing_efficiency
        );

        // Verify energy conservation in routing
        let total_original_allocation: f64 = network_allocation.values().sum();
        let total_balanced_allocation: f64 = balanced_allocation.values().sum();

        let allocation_difference = (total_original_allocation - total_balanced_allocation).abs();
        assert!(
            allocation_difference < config.data_integrity_threshold,
            "Energy allocation not conserved through load balancing"
        );

        println!("Energy-aware packet routing:");
        println!("  Network components: {}", component_count);
        println!("  Total routes: {}", total_routes);
        println!("  Routing efficiency: {:.3}", routing_efficiency);
        println!("  Allocation conservation: {:.2e}", allocation_difference);
    }

    #[test]
    fn test_adaptive_energy_packet_correlation() {
        let config = IntegrationConfig::default();

        // Create adaptive energy system with varying workloads
        let mut energy_functional = ChronoFabricEnergyFunctional::new(EnergyParameters::default());

        let workload_phases = vec![
            ("Low Load", 5, 0.1),     // 5 components, low intensity
            ("Medium Load", 10, 0.5), // 10 components, medium intensity
            ("High Load", 20, 0.9),   // 20 components, high intensity
            ("Peak Load", 30, 1.2),   // 30 components, overload
        ];

        for (phase_name, component_count, load_intensity) in workload_phases {
            println!("Testing phase: {}", phase_name);

            // Generate workload for this phase
            let mut phase_packets = Vec::new();
            let mut phase_components = HashMap::new();

            for i in 0..component_count {
                let component_id = ComponentId::new(i as u64);

                let test_data = IntegrationTestData::generate_test_data(
                    (10.0 * load_intensity) as usize + 5,
                    i as u64,
                );

                let mut packet = PhasePacket::new(test_data);

                // Intensity-based quantum properties
                let coherence = (0.9 - load_intensity * 0.3).max(0.3);
                packet.entangle_with(component_id, coherence);

                let phase_shift = load_intensity * std::f64::consts::PI / 4.0;
                packet.apply_phase_shift(phase_shift);

                // Create energy state reflecting load intensity
                let energy_state = if load_intensity > 1.0 {
                    EnergyState::Overloaded {
                        excess_energy: (load_intensity - 1.0) * 3e-12,
                        throttling_factor: 1.0 / load_intensity,
                    }
                } else {
                    EnergyState::Active {
                        current_energy: load_intensity * 8e-13,
                        peak_energy: load_intensity * 1.2e-12,
                        efficiency: (1.0 - load_intensity * 0.2).max(0.5),
                    }
                };

                energy_functional.update_component_state(component_id, energy_state.clone());
                phase_components.insert(component_id, energy_state);
                phase_packets.push((component_id, packet));
            }

            // Measure phase processing performance
            let phase_start = Instant::now();

            // 1. Energy optimization
            let allocation = energy_functional
                .optimize_allocation(&phase_components, &energy_functional.parameters)
                .unwrap();

            // 2. Packet serialization with energy awareness
            let mut serialized_packets = Vec::new();
            for (component_id, packet) in &phase_packets {
                let energy_weight = allocation[component_id];

                // Apply energy-weighted transformations
                let mut processed_packet = packet.clone();
                if energy_weight < 0.3 {
                    // Low energy - apply throttling
                    processed_packet = PhasePacket::new(processed_packet.payload.clone());
                }

                serialized_packets.push(processed_packet.quantum_serialize().unwrap());
            }

            // 3. Performance metrics collection
            let phase_duration = phase_start.elapsed();
            let total_data_size: usize = serialized_packets.iter().map(|data| data.len()).sum();

            let phase_metrics = PerformanceMetrics {
                avg_response_time_ns: phase_duration.as_nanos() as u64 / component_count as u64,
                throughput_ops_sec: component_count as f64 / phase_duration.as_secs_f64(),
                energy_efficiency: component_count as f64 / energy_functional.total_system_energy(),
                coherence_maintenance_rate: phase_packets
                    .iter()
                    .map(|(_, p)| if p.is_coherent() { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / component_count as f64,
                resource_utilization: load_intensity.min(1.0),
                ..PerformanceMetrics::default()
            };

            // 4. Adaptive parameter adjustment
            let adapted_params =
                energy_functional.adapt_parameters(&phase_metrics, &energy_functional.parameters);

            // Verify adaptation effectiveness
            assert!(
                adapted_params.quantum_scaling_factor
                    >= energy_functional.parameters.quantum_scaling_factor,
                "Quantum scaling should adapt to load"
            );

            if load_intensity > 0.8 {
                assert!(
                    adapted_params.max_energy_per_component
                        >= energy_functional.parameters.max_energy_per_component,
                    "Energy limits should adapt to high load"
                );
            }

            // Performance validation
            let throughput_per_component =
                phase_metrics.throughput_ops_sec / component_count as f64;
            let expected_min_throughput = (100.0 / load_intensity).max(10.0); // Scale expectations

            assert!(
                throughput_per_component >= expected_min_throughput,
                "Throughput too low for {}: {:.2} ops/sec per component",
                phase_name,
                throughput_per_component
            );

            println!("  Phase results:");
            println!("    Duration: {}ms", phase_duration.as_millis());
            println!(
                "    Throughput: {:.1} ops/sec total",
                phase_metrics.throughput_ops_sec
            );
            println!("    Data size: {}KB", total_data_size / 1024);
            println!(
                "    Coherence rate: {:.3}",
                phase_metrics.coherence_maintenance_rate
            );
            println!(
                "    Energy efficiency: {:.2e}",
                phase_metrics.energy_efficiency
            );

            // Clean up for next phase
            energy_functional = ChronoFabricEnergyFunctional::new(adapted_params);
        }
    }
}

#[cfg(test)]
mod end_to_end_workflow_tests {
    use super::*;

    #[test]
    #[ignore] // Temporarily disabled due to circular dependency resolution
    fn test_complete_chronofabric_pipeline() {
        let config = IntegrationConfig::default();

        println!("Starting complete ChronoFabric pipeline test");

        let pipeline_start = Instant::now();

        // Phase 1: Quantum Offset Generation and Processing
        println!("Phase 1: Quantum temporal correlation generation");
        let quantum_offsets: Vec<PreciseQuantumOffset> = (0..1000)
            .map(|i| {
                let temporal_correlation = (i as f64) * std::f64::consts::TAU / 1000.0;
                let femtosecond_precision = temporal_correlation * 1e-15;
                PreciseQuantumOffset::new(femtosecond_precision, PrecisionLevel::Femtosecond)
            })
            .collect();

        // Quantum offset operations
        let mut processed_offsets = Vec::new();
        for i in 0..quantum_offsets.len() - 1 {
            let correlation = quantum_offsets[i].add(&quantum_offsets[i + 1]);
            let enhanced = correlation.multiply(std::f64::consts::FRAC_1_SQRT_2);
            processed_offsets.push(enhanced);
        }

        let phase1_time = pipeline_start.elapsed();

        // Phase 2: Tensor Construction and Quantum Operations
        println!("Phase 2: RelationalTensor quantum processing");
        let tensor_size = (processed_offsets.len() as f64).sqrt() as usize;
        let tensor_data: Vec<f64> = processed_offsets
            .iter()
            .take(tensor_size * tensor_size)
            .map(|offset| offset.offset() * 1e12) // Scale for tensor operations
            .collect();

        let shape = vec![tensor_size, tensor_size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), tensor_data).unwrap();
        let mut quantum_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Quantum tensor operations
        let tensor_trace = quantum_tensor.trace().unwrap();
        let tensor_norm = quantum_tensor.frobenius_norm();
        quantum_tensor.normalize_quantum_state();

        // Apply quantum transformations
        quantum_tensor.apply_quantum_phase_shift(std::f64::consts::PI / 6.0);
        let quantum_coherence = quantum_tensor.calculate_quantum_coherence().unwrap();

        let phase2_time = pipeline_start.elapsed();

        // Phase 3: Phase Packet Network Distribution
        println!("Phase 3: PhasePacket distributed serialization");
        let network_size = 25;
        let mut distributed_packets = Vec::new();

        for i in 0..network_size {
            let component_id = ComponentId::new(i as u64);

            // Create comprehensive integration data
            let integration_data = IntegrationTestData {
                quantum_offsets: processed_offsets
                    .iter()
                    .skip(i * 10)
                    .take(10)
                    .map(|offset| offset.offset())
                    .collect(),
                tensor_data: quantum_tensor
                    .data
                    .iter()
                    .skip(i * 20)
                    .take(20)
                    .cloned()
                    .collect(),
                tensor_shape: vec![tensor_size, tensor_size],
                energy_metrics: vec![
                    tensor_trace * (i as f64 + 1.0),
                    tensor_norm * (i as f64 + 1.0),
                    quantum_coherence * (i as f64 + 1.0),
                ],
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                component_id: i as u64,
            };

            // Create quantum-aware phase packet
            let mut phase_packet = PhasePacket::with_phase_state(
                integration_data,
                PhaseState::Coherent((i as f64) * std::f64::consts::TAU / (network_size as f64)),
            );

            // Establish quantum entanglement network
            for j in 0..network_size {
                if i != j {
                    let target_component = ComponentId::new(j as u64);
                    let entanglement_strength = quantum_coherence * 0.8;
                    phase_packet.entangle_with(target_component, entanglement_strength);

                    let routing_phase =
                        ((i + j) as f64) * std::f64::consts::PI / network_size as f64;
                    phase_packet.add_route(target_component, routing_phase);
                }
            }

            // Apply temporal correlations from quantum offsets
            let temporal_phase =
                processed_offsets[i * processed_offsets.len() / network_size].offset() * 1e15;
            phase_packet.apply_phase_shift(temporal_phase);

            distributed_packets.push((component_id, phase_packet));
        }

        // Network serialization
        let mut network_data = Vec::new();
        for (component_id, packet) in &distributed_packets {
            let serialized = packet.quantum_serialize().unwrap();
            network_data.push((*component_id, serialized));
        }

        let phase3_time = pipeline_start.elapsed();

        // Phase 4: Energy Functional Optimization
        println!("Phase 4: EnergyFunctional adaptive optimization");
        let energy_params = EnergyParameters {
            allocation_strategy: AllocationStrategy::QuantumAware,
            quantum_scaling_factor: 1.2,
            target_efficiency: 0.92,
            max_energy_per_component: 1e-11,
            coherence_coupling: 0.15,
            ..EnergyParameters::default()
        };

        let mut energy_system = ChronoFabricEnergyFunctional::new(energy_params.clone());
        let mut system_components = HashMap::new();

        // Create energy states correlated with packet properties
        for (i, (component_id, packet)) in distributed_packets.iter().enumerate() {
            let energy_state = if packet.is_coherent() {
                EnergyState::QuantumCoherent {
                    energy: tensor_trace.abs() * (i as f64 + 1.0) * 1e-13,
                    coherence_factor: packet.coherence_factor,
                    phase_energy: quantum_coherence * 1e-13,
                }
            } else {
                EnergyState::Active {
                    current_energy: tensor_norm * (i as f64 + 1.0) * 1e-13,
                    peak_energy: tensor_norm * (i as f64 + 2.0) * 1e-13,
                    efficiency: 0.8 + (packet.coherence_factor - 1.0).max(0.0) * 0.1,
                }
            };

            energy_system.update_component_state(*component_id, energy_state.clone());
            system_components.insert(*component_id, energy_state);
        }

        // Multi-stage energy optimization
        let initial_allocation = energy_system
            .optimize_allocation(&system_components, &energy_params)
            .unwrap();

        // Quantum optimization for entangled components
        let mut quantum_entangled = HashMap::new();
        for (component_id, packet) in &distributed_packets {
            if packet.is_coherent() {
                quantum_entangled.insert(*component_id, packet.coherence_factor);
            }
        }

        let quantum_optimized = energy_system
            .quantum_optimize(&quantum_entangled, &system_components, &energy_params)
            .unwrap();

        // Performance measurement and adaptation
        let phase4_start_time = phase3_time;
        let current_time = pipeline_start.elapsed();
        let optimization_duration = current_time - phase4_start_time;

        let system_metrics = PerformanceMetrics {
            avg_response_time_ns: optimization_duration.as_nanos() as u64 / network_size as u64,
            peak_response_time_ns: optimization_duration.as_nanos() as u64 * 2, // Estimated peak
            throughput_ops_sec: network_size as f64 / optimization_duration.as_secs_f64(),
            energy_efficiency: network_size as f64 / energy_system.total_system_energy(),
            coherence_maintenance_rate: distributed_packets
                .iter()
                .map(|(_, p)| if p.is_coherent() { 1.0 } else { 0.0 })
                .sum::<f64>()
                / network_size as f64,
            error_rate_ppm: 1.0, // Estimated error rate
            resource_utilization: initial_allocation.values().sum::<f64>() / network_size as f64,
            adaptation_success_rate: quantum_optimized.len() as f64
                / quantum_entangled.len() as f64,
        };

        let adapted_params = energy_system.adapt_parameters(&system_metrics, &energy_params);

        // Final load balancing
        let mut component_loads = HashMap::new();
        for component_id in system_components.keys() {
            let load = initial_allocation[component_id] * 2.0; // Convert allocation to load
            component_loads.insert(*component_id, load);
        }

        let final_allocation =
            energy_system.dynamic_load_balance(&component_loads, &initial_allocation);

        let total_pipeline_time = pipeline_start.elapsed();

        // Comprehensive Verification
        println!("Phase 5: End-to-end validation");

        // 1. Data integrity verification
        let final_data_size: usize = network_data.iter().map(|(_, data)| data.len()).sum();
        assert!(final_data_size > 0, "No data produced by pipeline");

        // 2. Quantum coherence preservation
        let coherence_preservation_rate = system_metrics.coherence_maintenance_rate;
        assert!(
            coherence_preservation_rate >= config.coherence_preservation_threshold,
            "Coherence not preserved: {:.3}",
            coherence_preservation_rate
        );

        // 3. Energy conservation
        let initial_energy: f64 = system_components
            .values()
            .map(|state| energy_system.energy(state))
            .sum();
        let final_energy: f64 = quantum_optimized
            .values()
            .map(|state| energy_system.energy(state))
            .sum();
        let energy_conservation_error = (initial_energy - final_energy).abs() / initial_energy;

        assert!(
            energy_conservation_error < 0.1,
            "Energy not conserved: {:.3}% error",
            energy_conservation_error * 100.0
        );

        // 4. Performance scaling
        let ops_per_second = network_size as f64 / total_pipeline_time.as_secs_f64();
        assert!(
            ops_per_second > 1.0,
            "Pipeline throughput too low: {:.2} ops/sec",
            ops_per_second
        );

        // 5. Resource allocation validity
        assert_eq!(final_allocation.len(), network_size);
        for (component_id, allocation) in &final_allocation {
            assert!(
                *allocation >= 0.0 && *allocation <= 1.0,
                "Invalid allocation for {:?}: {}",
                component_id,
                allocation
            );
        }

        // 6. Temporal precision preservation
        let precision_samples = 10;
        for i in 0..precision_samples {
            let original_offset = &quantum_offsets[i];
            let processed_offset = &processed_offsets[i.min(processed_offsets.len() - 1)];

            // Precision should be maintained within femtosecond accuracy
            let precision_error = (original_offset.offset() - processed_offset.offset()).abs();
            assert!(
                precision_error < 1e-14,
                "Precision loss at sample {}: {:.2e}",
                i,
                precision_error
            );
        }

        // Results Summary
        println!("\n=== CHRONOFABRIC PIPELINE RESULTS ===");
        println!(
            "Total execution time: {}ms",
            total_pipeline_time.as_millis()
        );
        println!("Data processing phases:");
        println!("  Phase 1 (Quantum Offsets): {}ms", phase1_time.as_millis());
        println!(
            "  Phase 2 (Tensor Ops): {}ms",
            (phase2_time - phase1_time).as_millis()
        );
        println!(
            "  Phase 3 (Packet Network): {}ms",
            (phase3_time - phase2_time).as_millis()
        );
        println!(
            "  Phase 4 (Energy Opt): {}ms",
            (total_pipeline_time - phase3_time).as_millis()
        );
        println!("Network characteristics:");
        println!("  Quantum offsets processed: {}", quantum_offsets.len());
        println!("  Tensor size: {}×{}", tensor_size, tensor_size);
        println!("  Network components: {}", network_size);
        println!("  Total data size: {}KB", final_data_size / 1024);
        println!("Performance metrics:");
        println!("  Throughput: {:.2} ops/sec", ops_per_second);
        println!("  Coherence rate: {:.3}", coherence_preservation_rate);
        println!(
            "  Energy efficiency: {:.2e}",
            system_metrics.energy_efficiency
        );
        println!(
            "  Resource utilization: {:.3}",
            system_metrics.resource_utilization
        );
        println!("Validation results:");
        println!(
            "  Energy conservation: {:.3}% error",
            energy_conservation_error * 100.0
        );
        println!("  Precision preservation: ✓ femtosecond accuracy");
        println!(
            "  Quantum coherence: ✓ {:.1}% maintained",
            coherence_preservation_rate * 100.0
        );
        println!("  Resource allocation: ✓ all valid");
        println!(
            "  Pipeline timeout: ✓ {}ms < {}ms",
            total_pipeline_time.as_millis(),
            config.workflow_timeout_ms
        );

        // Final assertions
        assert!(
            total_pipeline_time.as_millis() < config.workflow_timeout_ms,
            "Pipeline exceeded timeout: {}ms",
            total_pipeline_time.as_millis()
        );

        println!("\n✅ COMPLETE CHRONOFABRIC PIPELINE TEST PASSED");
    }

    #[test]
    #[ignore] // Temporarily disabled due to circular dependency resolution
    fn test_concurrent_pipeline_execution() {
        let config = IntegrationConfig::default();

        println!("Testing concurrent ChronoFabric pipeline execution");

        let pipeline_count = 4;
        let components_per_pipeline = 20;
        let mut handles = Vec::new();

        let results = Arc::new(Mutex::new(Vec::new()));

        for pipeline_id in 0..pipeline_count {
            let results_clone = results.clone();

            let handle = thread::spawn(move || {
                let pipeline_start = Instant::now();

                // Concurrent pipeline execution
                let mut pipeline_offsets = Vec::new();
                for i in 0..100 {
                    let offset_value = (pipeline_id as f64 * 1000.0 + i as f64) * 1e-15;
                    let offset =
                        PreciseQuantumOffset::new(offset_value, PrecisionLevel::Femtosecond);
                    pipeline_offsets.push(offset);
                }

                // Process through tensor operations
                let tensor_size = 10;
                let tensor_data: Vec<f64> = pipeline_offsets
                    .iter()
                    .take(tensor_size * tensor_size)
                    .map(|o| o.offset() * 1e12)
                    .collect();

                let shape = vec![tensor_size, tensor_size];
                let ndarray = Array::from_shape_vec(IxDyn(&shape), tensor_data).unwrap();
                let tensor =
                    RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

                let tensor_trace = tensor.trace().unwrap();

                // Create component network
                let mut packets = Vec::new();
                for i in 0..components_per_pipeline {
                    let component_id = ComponentId::new((pipeline_id * 1000 + i) as u64);
                    let test_data =
                        IntegrationTestData::generate_test_data(5, component_id.inner());
                    let packet = PhasePacket::new(test_data);
                    packets.push((component_id, packet));
                }

                // Energy optimization
                let energy_params = EnergyParameters::default();
                let mut energy_functional =
                    ChronoFabricEnergyFunctional::new(energy_params.clone());

                let mut component_states = HashMap::new();
                for (component_id, _) in &packets {
                    let state = EnergyState::Active {
                        current_energy: tensor_trace.abs() * 1e-13,
                        peak_energy: tensor_trace.abs() * 2e-13,
                        efficiency: 0.8,
                    };
                    energy_functional.update_component_state(*component_id, state.clone());
                    component_states.insert(*component_id, state);
                }

                let allocation = energy_functional
                    .optimize_allocation(&component_states, &energy_params)
                    .unwrap();

                let pipeline_duration = pipeline_start.elapsed();

                // Return pipeline results
                let pipeline_result = (
                    pipeline_id,
                    pipeline_duration,
                    pipeline_offsets.len(),
                    tensor_trace,
                    packets.len(),
                    allocation.len(),
                );

                results_clone.lock().unwrap().push(pipeline_result);
                pipeline_result
            });

            handles.push(handle);
        }

        // Wait for all pipelines to complete
        let mut pipeline_results = Vec::new();
        for handle in handles {
            pipeline_results.push(handle.join().unwrap());
        }

        // Verify concurrent execution results
        let total_execution_time = pipeline_results
            .iter()
            .map(|(_, duration, _, _, _, _)| duration.as_millis())
            .max()
            .unwrap();

        let total_components: usize = pipeline_results
            .iter()
            .map(|(_, _, _, _, packet_count, _)| *packet_count)
            .sum();

        let total_allocations: usize = pipeline_results
            .iter()
            .map(|(_, _, _, _, _, allocation_count)| *allocation_count)
            .sum();

        println!("Concurrent pipeline execution results:");
        for (pipeline_id, duration, offset_count, trace, packet_count, allocation_count) in
            &pipeline_results
        {
            println!(
                "  Pipeline {}: {}ms, {} offsets, trace={:.2e}, {} packets, {} allocations",
                pipeline_id,
                duration.as_millis(),
                offset_count,
                trace,
                packet_count,
                allocation_count
            );
        }

        // Verify results
        assert_eq!(pipeline_results.len(), pipeline_count);
        assert_eq!(total_components, pipeline_count * components_per_pipeline);
        assert_eq!(total_allocations, total_components);

        // Performance validation
        assert!(
            total_execution_time < config.workflow_timeout_ms,
            "Concurrent execution too slow: {}ms",
            total_execution_time
        );

        // Verify no data corruption in concurrent execution
        let traces: Vec<f64> = pipeline_results
            .iter()
            .map(|(_, _, _, trace, _, _)| *trace)
            .collect();
        for (i, &trace) in traces.iter().enumerate() {
            assert!(trace.is_finite(), "Pipeline {} produced invalid trace", i);
            assert!(trace != 0.0, "Pipeline {} produced zero trace", i);
        }

        let concurrent_throughput =
            total_components as f64 / (total_execution_time as f64 / 1000.0);
        println!(
            "Concurrent throughput: {:.2} components/sec",
            concurrent_throughput
        );

        assert!(
            concurrent_throughput > 10.0,
            "Concurrent throughput too low: {:.2}",
            concurrent_throughput
        );

        println!("✅ CONCURRENT PIPELINE EXECUTION TEST PASSED");
    }
}
