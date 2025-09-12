//! Phase 1.2 Performance Benchmarks for ARES ChronoFabric
//!
//! Comprehensive benchmarking suite to validate performance targets and detect regressions.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Import Phase 1.2 components
use csf_core::energy_functional::{
    ChronoFabricEnergyFunctional, EnergyFunctional, EnergyParameters, EnergyState,
};
use csf_core::phase_packet::PhasePacket;
use csf_core::tensor::RelationalTensor;
use csf_core::ComponentId;
use serde::{Deserialize, Serialize};

/// Benchmark test data
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchmarkData {
    values: Vec<f64>,
    metadata: String,
}

impl BenchmarkData {
    fn generate(size: usize) -> Self {
        Self {
            values: (0..size).map(|i| i as f64 * std::f64::consts::PI).collect(),
            metadata: format!("benchmark_data_{}", size),
        }
    }
}

/// Benchmark quantum data structures
fn bench_quantum_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_data_structures");

    group.bench_function("component_id_creation", |b| {
        b.iter(|| {
            let _id = ComponentId::new(12345);
        });
    });

    group.bench_function("benchmark_data_creation", |b| {
        b.iter(|| {
            let _data = BenchmarkData::generate(100);
        });
    });

    group.finish();
}

/// Benchmark tensor operations
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    group.sample_size(50); // Reduce sample size for large operations

    let sizes = [10, 50, 100];

    for &size in &sizes {
        // Create test tensor using correct API
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let shape = vec![size, size];
        let tensor = RelationalTensor::new(data, shape).unwrap();

        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, _| {
            b.iter(|| {
                let _result = tensor.matmul(&tensor).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |b, _| {
            b.iter(|| {
                let _transposed = tensor.clone().transpose(0, 1).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("simd_multiply", size), &size, |b, _| {
            b.iter(|| {
                let _result = tensor.simd_element_wise_multiply(&tensor).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark phase packet serialization
fn bench_phase_packet_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase_packet_serialization");
    group.measurement_time(Duration::from_secs(10));

    let data_sizes = [100, 1000, 10000];

    for &size in &data_sizes {
        let test_data = BenchmarkData::generate(size);
        let mut packet = PhasePacket::new(test_data);

        // Add quantum properties
        packet.entangle_with(ComponentId::new(123), 0.9);
        packet.apply_phase_shift(std::f64::consts::PI / 4.0);

        group.bench_with_input(BenchmarkId::new("serialize", size), &size, |b, _| {
            b.iter(|| {
                let _serialized = packet.quantum_serialize().unwrap();
            });
        });

        let serialized = packet.quantum_serialize().unwrap();

        group.bench_with_input(BenchmarkId::new("deserialize", size), &size, |b, _| {
            b.iter(|| {
                let _: PhasePacket<BenchmarkData> =
                    PhasePacket::quantum_deserialize(&serialized).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark energy functional optimization
fn bench_energy_functional_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_optimization");
    group.sample_size(30);

    let component_counts = [5, 10, 25];

    for &count in &component_counts {
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Create component system
        let mut components = std::collections::HashMap::new();
        for i in 0..count {
            let component_id = ComponentId::new(i as u64);
            let state = match i % 3 {
                0 => EnergyState::Active {
                    current_energy: 8e-13,
                    peak_energy: 1e-12,
                    efficiency: 0.85,
                },
                1 => EnergyState::QuantumCoherent {
                    energy: 5e-13,
                    coherence_factor: 0.9,
                    phase_energy: 2e-13,
                },
                _ => EnergyState::Idle {
                    baseline_energy: 1e-15,
                },
            };
            components.insert(component_id, state);
        }

        group.bench_with_input(
            BenchmarkId::new("allocation_optimization", count),
            &count,
            |b, _| {
                b.iter(|| {
                    let _allocation = functional
                        .optimize_allocation(&components, &params)
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("energy_calculation", count),
            &count,
            |b, _| {
                b.iter(|| {
                    for state in components.values() {
                        let _energy = functional.energy(state);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark integrated workflows
fn bench_integrated_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrated_workflow");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("complete_pipeline", |b| {
        b.iter(|| {
            // Create tensor using correct API
            let tensor_data: Vec<f64> = (0..100).map(|i| i as f64 * 1e-15).collect();
            let shape = vec![10, 10];
            let tensor = RelationalTensor::new(tensor_data, shape).unwrap();

            // Create phase packet
            let test_data = BenchmarkData::generate(50);
            let mut packet = PhasePacket::new(test_data);
            packet.entangle_with(ComponentId::new(1), 0.8);

            // Serialize packet
            let _serialized = packet.quantum_serialize().unwrap();

            // Energy optimization
            let params = EnergyParameters::default();
            let functional = ChronoFabricEnergyFunctional::new(params.clone());

            let components = std::collections::HashMap::from([
                (
                    ComponentId::new(1),
                    EnergyState::Active {
                        current_energy: 8e-13,
                        peak_energy: 1e-12,
                        efficiency: 0.85,
                    },
                ),
                (
                    ComponentId::new(2),
                    EnergyState::QuantumCoherent {
                        energy: 5e-13,
                        coherence_factor: 0.9,
                        phase_energy: 2e-13,
                    },
                ),
            ]);

            let _allocation = functional
                .optimize_allocation(&components, &params)
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_quantum_data_structures,
    bench_tensor_operations,
    bench_phase_packet_serialization,
    bench_energy_functional_optimization,
    bench_integrated_workflow
);

criterion_main!(benches);
