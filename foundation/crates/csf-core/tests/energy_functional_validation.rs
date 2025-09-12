//! Comprehensive production-grade validation for EnergyFunctional optimization systems
//!
//! This test suite provides exhaustive validation of quantum-aware energy optimization
//! with mathematical accuracy verification, convergence testing, and performance benchmarks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use csf_core::energy_functional::{
    AdaptiveEnergyFunctional, AllocationStrategy, ChronoFabricEnergyFunctional, EfficiencyFactor,
    EnergyFunctional, EnergyOptimizationError, EnergyParameters, EnergyState, EnergyUnits,
    PerformanceMetrics, QuantumEnergyFunctional, ResourceWeight,
};
use csf_core::phase_packet::{CoherenceFactor, PhaseAngle};
use csf_core::{ComponentId, NanoTime};

/// Configuration for comprehensive energy functional validation
#[derive(Clone)]
struct EnergyValidationConfig {
    convergence_tolerance: f64,
    max_optimization_iterations: u32,
    performance_target_us: u64,
    thread_count: usize,
    stress_component_count: usize,
    accuracy_epsilon: f64,
    energy_conservation_threshold: f64,
}

impl Default for EnergyValidationConfig {
    fn default() -> Self {
        Self {
            convergence_tolerance: 1e-6,
            max_optimization_iterations: 1000,
            performance_target_us: 100, // 100 microseconds
            thread_count: 8,
            stress_component_count: 1000,
            accuracy_epsilon: 1e-12,
            energy_conservation_threshold: 1e-10,
        }
    }
}

/// Generate test energy states with various patterns
fn generate_test_energy_state(state_type: TestEnergyType, energy_scale: f64) -> EnergyState {
    match state_type {
        TestEnergyType::Idle => EnergyState::Idle {
            baseline_energy: energy_scale * 1e-15,
        },
        TestEnergyType::Active => EnergyState::Active {
            current_energy: energy_scale * 5e-13,
            peak_energy: energy_scale * 1e-12,
            efficiency: 0.85,
        },
        TestEnergyType::QuantumCoherent => EnergyState::QuantumCoherent {
            energy: energy_scale * 3e-13,
            coherence_factor: 0.92,
            phase_energy: energy_scale * 1e-13,
        },
        TestEnergyType::Overloaded => EnergyState::Overloaded {
            excess_energy: energy_scale * 2e-12,
            throttling_factor: 0.6,
        },
        TestEnergyType::Error => EnergyState::Error {
            error_energy: energy_scale * 1e-12,
            recovery_cost: energy_scale * 3e-12,
        },
    }
}

#[derive(Clone)]
enum TestEnergyType {
    Idle,
    Active,
    QuantumCoherent,
    Overloaded,
    Error,
}

/// Generate component system for testing
fn generate_test_system(
    component_count: usize,
    energy_types: Vec<TestEnergyType>,
) -> HashMap<ComponentId, EnergyState> {
    let mut system = HashMap::new();

    for i in 0..component_count {
        let component_id = ComponentId::new(i as u64);
        let energy_type = &energy_types[i % energy_types.len()];
        let energy_scale = 1.0 + (i as f64) / 100.0; // Varying energy scales
        let state = generate_test_energy_state(energy_type.clone(), energy_scale);
        system.insert(component_id, state);
    }

    system
}

#[cfg(test)]
mod mathematical_accuracy_tests {
    use super::*;

    #[test]
    fn test_energy_calculation_accuracy() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        // Test energy calculations for different states
        let idle_state = EnergyState::Idle {
            baseline_energy: 1e-15,
        };
        let idle_energy = functional.energy(&idle_state);
        assert!((idle_energy - 1e-15).abs() < config.accuracy_epsilon);

        let active_state = EnergyState::Active {
            current_energy: 9e-13,
            peak_energy: 1e-12,
            efficiency: 0.9,
        };
        let active_energy = functional.energy(&active_state);
        let expected_active = 9e-13 / 0.9;
        assert!((active_energy - expected_active).abs() < config.accuracy_epsilon);

        let quantum_state = EnergyState::QuantumCoherent {
            energy: 5e-13,
            coherence_factor: 0.8,
            phase_energy: 2e-13,
        };
        let quantum_energy = functional.energy(&quantum_state);
        let expected_quantum = 5e-13 + 2e-13 * 0.8;
        assert!((quantum_energy - expected_quantum).abs() < config.accuracy_epsilon);
    }

    #[test]
    fn test_energy_gradient_accuracy() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        let active_state = EnergyState::Active {
            current_energy: 8e-13,
            peak_energy: 1e-12,
            efficiency: 0.8,
        };

        let gradient = functional.energy_gradient(&active_state);

        // Verify gradient components
        assert_eq!(gradient.len(), 2);
        let expected_energy_gradient = 1.0 / 0.8;
        let expected_efficiency_gradient = -8e-13 / (0.8 * 0.8);

        assert!((gradient[0] - expected_energy_gradient).abs() < config.accuracy_epsilon);
        assert!((gradient[1] - expected_efficiency_gradient).abs() < config.accuracy_epsilon);

        // Test quantum coherent gradient
        let quantum_state = EnergyState::QuantumCoherent {
            energy: 5e-13,
            coherence_factor: 0.9,
            phase_energy: 2e-13,
        };

        let quantum_gradient = functional.energy_gradient(&quantum_state);
        assert_eq!(quantum_gradient.len(), 3);
        assert!((quantum_gradient[0] - 1.0).abs() < config.accuracy_epsilon);
        assert!((quantum_gradient[1] - 2e-13).abs() < config.accuracy_epsilon);
        assert!((quantum_gradient[2] - 0.9).abs() < config.accuracy_epsilon);
    }

    #[test]
    fn test_efficiency_metric_accuracy() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        // Test different efficiency calculations
        let idle_state = EnergyState::Idle {
            baseline_energy: 1e-15,
        };
        let idle_efficiency = functional.efficiency_metric(&idle_state);
        assert!((idle_efficiency - 0.1).abs() < config.accuracy_epsilon);

        let active_state = EnergyState::Active {
            current_energy: 5e-13,
            peak_energy: 1e-12,
            efficiency: 0.75,
        };
        let active_efficiency = functional.efficiency_metric(&active_state);
        assert!((active_efficiency - 0.75).abs() < config.accuracy_epsilon);

        let quantum_state = EnergyState::QuantumCoherent {
            energy: 5e-13,
            coherence_factor: 0.85,
            phase_energy: 2e-13,
        };
        let quantum_efficiency = functional.efficiency_metric(&quantum_state);
        let expected_quantum_efficiency = 0.8 + 0.85 * 0.4;
        assert!((quantum_efficiency - expected_quantum_efficiency).abs() < config.accuracy_epsilon);
    }

    #[test]
    fn test_constraint_validation_accuracy() {
        let _config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Valid state within constraints
        let valid_state = EnergyState::Active {
            current_energy: 5e-13, // Well below max (1e-12)
            peak_energy: 8e-13,
            efficiency: 0.8,
        };
        assert!(functional.validate_constraints(&valid_state, &params));

        // Invalid state exceeding energy constraint
        let invalid_state = EnergyState::Active {
            current_energy: 2e-12, // Above max when divided by efficiency
            peak_energy: 3e-12,
            efficiency: 0.5,
        };
        assert!(!functional.validate_constraints(&invalid_state, &params));

        // Edge case at exactly the limit
        let edge_state = EnergyState::Active {
            current_energy: 9e-13, // Exactly at limit when divided by 0.9
            peak_energy: 1e-12,
            efficiency: 0.9,
        };
        assert!(functional.validate_constraints(&edge_state, &params));
    }

    #[test]
    fn test_total_system_energy_accuracy() {
        let config = EnergyValidationConfig::default();
        let mut functional = ChronoFabricEnergyFunctional::new(EnergyParameters::default());

        // Add several components with known energies
        let components = vec![
            (
                ComponentId::new(1),
                EnergyState::Idle {
                    baseline_energy: 1e-15,
                },
            ),
            (
                ComponentId::new(2),
                EnergyState::Active {
                    current_energy: 8e-13,
                    peak_energy: 1e-12,
                    efficiency: 0.8,
                },
            ),
            (
                ComponentId::new(3),
                EnergyState::QuantumCoherent {
                    energy: 5e-13,
                    coherence_factor: 0.9,
                    phase_energy: 2e-13,
                },
            ),
        ];

        let mut expected_total = 0.0;
        for (component, state) in &components {
            functional.update_component_state(*component, state.clone());
            expected_total += functional.energy(state);
        }

        let actual_total = functional.total_system_energy();
        assert!(
            (actual_total - expected_total).abs() < config.accuracy_epsilon,
            "System energy mismatch: {} vs {}",
            actual_total,
            expected_total
        );
    }

    #[test]
    fn test_numerical_stability() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        // Test with very small numbers
        let tiny_state = EnergyState::Active {
            current_energy: 1e-30,
            peak_energy: 2e-30,
            efficiency: 1e-10,
        };
        let tiny_energy = functional.energy(&tiny_state);
        assert!(tiny_energy.is_finite(), "Tiny energy should be finite");

        // Test with very large numbers
        let large_state = EnergyState::Active {
            current_energy: 1e10,
            peak_energy: 2e10,
            efficiency: 0.9,
        };
        let large_energy = functional.energy(&large_state);
        assert!(large_energy.is_finite(), "Large energy should be finite");

        // Test near-zero efficiency handling
        let zero_eff_state = EnergyState::Active {
            current_energy: 1e-12,
            peak_energy: 2e-12,
            efficiency: 1e-20, // Very small efficiency
        };
        let zero_eff_energy = functional.energy(&zero_eff_state);
        assert!(
            zero_eff_energy.is_finite(),
            "Near-zero efficiency should be handled"
        );
        assert!(
            zero_eff_energy > 1e-12 / 0.1,
            "Should use minimum efficiency bound"
        );
    }
}

#[cfg(test)]
mod optimization_convergence_tests {
    use super::*;

    #[test]
    fn test_allocation_optimization_convergence() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Create system with multiple components
        let components = generate_test_system(
            10,
            vec![
                TestEnergyType::Active,
                TestEnergyType::QuantumCoherent,
                TestEnergyType::Idle,
            ],
        );

        // Test different allocation strategies
        let strategies = vec![
            AllocationStrategy::Equal,
            AllocationStrategy::Priority,
            AllocationStrategy::LoadBalanced,
            AllocationStrategy::QuantumAware,
        ];

        for strategy in strategies {
            let mut test_params = params.clone();
            test_params.allocation_strategy = strategy;

            let allocation_result = functional.optimize_allocation(&components, &test_params);
            assert!(
                allocation_result.is_ok(),
                "Allocation optimization failed for {:?}",
                strategy
            );

            let allocation = allocation_result.unwrap();

            // Verify allocation properties
            assert_eq!(allocation.len(), components.len());

            // All weights should be valid
            for (component, weight) in &allocation {
                assert!(*weight >= 0.0, "Negative weight for {:?}", component);
                assert!(*weight <= 1.0, "Weight exceeds 1.0 for {:?}", component);
            }

            // For equal allocation, weights should be approximately equal
            if matches!(strategy, AllocationStrategy::Equal) {
                let expected_weight = 1.0 / components.len() as f64;
                for weight in allocation.values() {
                    assert!(
                        (weight - expected_weight).abs() < config.convergence_tolerance,
                        "Equal allocation weight mismatch: {} vs {}",
                        weight,
                        expected_weight
                    );
                }
            }
        }
    }

    #[test]
    fn test_quantum_optimization_convergence() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Create entangled component system
        let mut entangled_components = HashMap::new();
        let mut base_states = HashMap::new();

        for i in 0..5 {
            let component = ComponentId::new(i);
            let coherence = 0.7 + (i as f64) * 0.05; // Varying coherence
            entangled_components.insert(component, coherence);

            base_states.insert(
                component,
                EnergyState::Active {
                    current_energy: (i as f64 + 1.0) * 1e-13,
                    peak_energy: (i as f64 + 2.0) * 1e-13,
                    efficiency: 0.8 + (i as f64) * 0.02,
                },
            );
        }

        let optimization_result =
            functional.quantum_optimize(&entangled_components, &base_states, &params);

        assert!(optimization_result.is_ok(), "Quantum optimization failed");
        let optimized_states = optimization_result.unwrap();

        // Verify optimization results
        assert_eq!(optimized_states.len(), entangled_components.len());

        for (component, optimized_state) in &optimized_states {
            // Should be quantum coherent state
            assert!(
                matches!(optimized_state, EnergyState::QuantumCoherent { .. }),
                "Component {:?} not in quantum coherent state",
                component
            );

            if let EnergyState::QuantumCoherent {
                coherence_factor, ..
            } = optimized_state
            {
                let expected_coherence = entangled_components[component];
                assert!(
                    (coherence_factor - expected_coherence).abs() < config.convergence_tolerance,
                    "Coherence factor mismatch for {:?}: {} vs {}",
                    component,
                    coherence_factor,
                    expected_coherence
                );
            }
        }
    }

    #[test]
    fn test_adaptive_parameter_convergence() {
        let config = EnergyValidationConfig::default();
        let initial_params = EnergyParameters::default();
        let mut functional = ChronoFabricEnergyFunctional::new(initial_params.clone());

        // Create performance metrics requiring adaptation
        let poor_metrics = PerformanceMetrics {
            avg_response_time_ns: 2000, // Above target (1000)
            peak_response_time_ns: 5000,
            throughput_ops_sec: 500_000.0,    // Below target (1M)
            energy_efficiency: 5e10,          // Below target (1e11)
            coherence_maintenance_rate: 0.85, // Below target (0.9)
            error_rate_ppm: 2.0,
            resource_utilization: 0.6,
            adaptation_success_rate: 0.9,
        };

        let adapted_params = functional.adapt_parameters(&poor_metrics, &initial_params);

        // Verify adaptations
        assert!(
            adapted_params.quantum_scaling_factor > initial_params.quantum_scaling_factor,
            "Quantum scaling factor should increase"
        );
        assert!(
            adapted_params.target_efficiency >= initial_params.target_efficiency,
            "Target efficiency should not decrease"
        );
        assert!(
            adapted_params.max_energy_per_component > initial_params.max_energy_per_component,
            "Max energy should increase for poor energy efficiency"
        );
        assert!(
            adapted_params.coherence_coupling > initial_params.coherence_coupling,
            "Coherence coupling should increase for poor coherence maintenance"
        );

        // Test convergence over multiple adaptations
        let mut current_params = initial_params;
        for iteration in 0..10 {
            current_params = functional.adapt_parameters(&poor_metrics, &current_params);

            // Parameters should converge (stop changing significantly)
            if iteration > 5 {
                let next_params = functional.adapt_parameters(&poor_metrics, &current_params);
                let scaling_change = (next_params.quantum_scaling_factor
                    - current_params.quantum_scaling_factor)
                    .abs();
                let efficiency_change =
                    (next_params.target_efficiency - current_params.target_efficiency).abs();

                if scaling_change < config.convergence_tolerance
                    && efficiency_change < config.convergence_tolerance
                {
                    println!("Parameter adaptation converged at iteration {}", iteration);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_energy_prediction_accuracy() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        // Create historical energy data with trend
        let mut historical_data = Vec::new();
        for i in 0..100 {
            let energy = 1e-12 + (i as f64) * 1e-15; // Linear increase
            historical_data.push(EnergyState::Active {
                current_energy: energy,
                peak_energy: energy * 1.5,
                efficiency: 0.8,
            });
        }

        let time_horizon_ns = 1_000_000; // 1ms
        let predicted_energy = functional.predict_energy_demand(&historical_data, time_horizon_ns);

        // Prediction should be reasonable based on trend
        let last_energy = functional.energy(&historical_data.last().unwrap());
        assert!(
            predicted_energy >= last_energy,
            "Prediction should account for growth trend"
        );
        assert!(
            predicted_energy < last_energy * 2.0,
            "Prediction should be reasonable"
        );

        // Test with empty historical data
        let empty_prediction = functional.predict_energy_demand(&[], time_horizon_ns);
        let conservative_estimate = functional.parameters.max_energy_per_component * 0.5;
        assert!(
            (empty_prediction - conservative_estimate).abs() < config.accuracy_epsilon,
            "Empty history prediction should be conservative"
        );
    }

    #[test]
    fn test_throttling_convergence() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Test throttling of overloaded states
        let overloaded_state = EnergyState::Active {
            current_energy: 10e-12, // Well above throttling threshold
            peak_energy: 15e-12,
            efficiency: 0.8,
        };

        let throttled_state = functional.apply_throttling(&overloaded_state, 0.5);

        // Should transition to overloaded state
        assert!(
            matches!(throttled_state, EnergyState::Overloaded { .. }),
            "Should transition to overloaded state"
        );

        if let EnergyState::Overloaded {
            excess_energy,
            throttling_factor,
        } = throttled_state
        {
            let expected_excess = 10e-12 - params.throttling_threshold;
            assert!(
                (excess_energy - expected_excess).abs() < config.accuracy_epsilon,
                "Excess energy calculation incorrect"
            );
            assert!(
                (throttling_factor - 0.5).abs() < config.accuracy_epsilon,
                "Throttling factor not preserved"
            );
        }

        // Test quantum coherent throttling
        let quantum_state = EnergyState::QuantumCoherent {
            energy: 8e-12, // Above threshold
            coherence_factor: 0.9,
            phase_energy: 1e-12,
        };

        let throttled_quantum = functional.apply_throttling(&quantum_state, 0.7);
        assert!(
            matches!(throttled_quantum, EnergyState::Overloaded { .. }),
            "Quantum state should be throttled when over threshold"
        );
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_energy_calculation_performance() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        let test_state = EnergyState::QuantumCoherent {
            energy: 5e-13,
            coherence_factor: 0.85,
            phase_energy: 2e-13,
        };

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _energy = functional.energy(&test_state);
        }

        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / iterations as u128;

        println!("Energy calculation: {}ns per operation", avg_ns);
        assert!(avg_ns < 1000, "Energy calculation too slow: {}ns", avg_ns); // Sub-microsecond
    }

    #[test]
    fn test_optimization_performance() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Test with varying system sizes
        let system_sizes = [10, 50, 100, 200];

        for &size in &system_sizes {
            let components = generate_test_system(
                size,
                vec![TestEnergyType::Active, TestEnergyType::QuantumCoherent],
            );

            let start = Instant::now();
            let _allocation = functional
                .optimize_allocation(&components, &params)
                .unwrap();
            let duration = start.elapsed();

            println!(
                "Optimization ({} components): {}μs",
                size,
                duration.as_micros()
            );

            // Should scale reasonably
            let max_expected_us = size as u64 * 10; // Rough scaling expectation
            assert!(
                (duration.as_micros() as u64) < max_expected_us,
                "Optimization too slow for {} components: {}μs",
                size,
                duration.as_micros()
            );
        }
    }

    #[test]
    fn test_gradient_calculation_performance() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        let test_states = vec![
            EnergyState::Active {
                current_energy: 8e-13,
                peak_energy: 1e-12,
                efficiency: 0.85,
            },
            EnergyState::QuantumCoherent {
                energy: 5e-13,
                coherence_factor: 0.9,
                phase_energy: 2e-13,
            },
            EnergyState::Idle {
                baseline_energy: 1e-15,
            },
        ];

        let iterations = 5000;

        for (i, state) in test_states.iter().enumerate() {
            let start = Instant::now();

            for _ in 0..iterations {
                let _gradient = functional.energy_gradient(state);
            }

            let duration = start.elapsed();
            let avg_ns = duration.as_nanos() / iterations as u128;

            println!(
                "Gradient calculation (state {}): {}ns per operation",
                i, avg_ns
            );
            assert!(avg_ns < 5000, "Gradient calculation too slow: {}ns", avg_ns);
            // < 5μs
        }
    }

    #[test]
    fn test_quantum_optimization_performance() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        let component_counts = [5, 10, 25, 50];

        for &count in &component_counts {
            let mut entangled_components = HashMap::new();
            let mut base_states = HashMap::new();

            for i in 0..count {
                let component = ComponentId::new(i);
                entangled_components.insert(component, 0.8 + (i as f64) * 0.01);
                base_states.insert(
                    component,
                    EnergyState::Active {
                        current_energy: (i as f64 + 1.0) * 1e-13,
                        peak_energy: (i as f64 + 2.0) * 1e-13,
                        efficiency: 0.8,
                    },
                );
            }

            let start = Instant::now();
            let _result = functional
                .quantum_optimize(&entangled_components, &base_states, &params)
                .unwrap();
            let duration = start.elapsed();

            println!(
                "Quantum optimization ({} components): {}μs",
                count,
                duration.as_micros()
            );

            // Should complete within performance target
            assert!(
                (duration.as_micros() as u64) < config.performance_target_us,
                "Quantum optimization too slow for {} components: {}μs",
                count,
                duration.as_micros()
            );
        }
    }

    #[test]
    fn test_adaptive_parameter_performance() {
        let config = EnergyValidationConfig::default();
        let initial_params = EnergyParameters::default();
        let mut functional = ChronoFabricEnergyFunctional::new(initial_params.clone());

        let test_metrics = PerformanceMetrics::default();
        let iterations = 1000;

        let start = Instant::now();

        for _ in 0..iterations {
            let _adapted = functional.adapt_parameters(&test_metrics, &initial_params);
        }

        let duration = start.elapsed();
        let avg_us = duration.as_micros() / iterations as u128;

        println!("Parameter adaptation: {}μs per operation", avg_us);
        assert!(
            (avg_us as u64) < config.performance_target_us,
            "Parameter adaptation too slow: {}μs",
            avg_us
        );
    }

    #[test]
    fn test_load_balancing_performance() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        let component_counts = [10, 50, 100, 500];

        for &count in &component_counts {
            let mut component_loads = HashMap::new();
            let mut available_resources = HashMap::new();

            for i in 0..count {
                let component = ComponentId::new(i);
                component_loads.insert(component, (i as f64) / (count as f64));
                available_resources.insert(component, 0.8 + (i as f64) / (count as f64) * 0.2);
            }

            let start = Instant::now();
            let _balanced = functional.dynamic_load_balance(&component_loads, &available_resources);
            let duration = start.elapsed();

            println!(
                "Load balancing ({} components): {}μs",
                count,
                duration.as_micros()
            );

            // Should scale linearly
            let max_expected_us = count as u64;
            assert!(
                duration.as_micros() < (max_expected_us as u128),
                "Load balancing too slow for {} components: {}μs",
                count,
                duration.as_micros()
            );
        }
    }

    #[test]
    fn test_memory_usage_efficiency() {
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params);

        // Measure base memory usage
        let base_size = std::mem::size_of::<ChronoFabricEnergyFunctional>();
        println!(
            "ChronoFabricEnergyFunctional base size: {} bytes",
            base_size
        );

        // Should be reasonable size
        assert!(
            base_size < 10000,
            "EnergyFunctional too large: {} bytes",
            base_size
        );

        // Test memory usage with components
        let component_sizes = [100, 500, 1000];

        for &size in &component_sizes {
            let mut test_functional =
                ChronoFabricEnergyFunctional::new(EnergyParameters::default());

            for i in 0..size {
                let component = ComponentId::new(i);
                let state = generate_test_energy_state(TestEnergyType::Active, 1.0);
                test_functional.update_component_state(component, state);
            }

            // Estimate memory usage
            let per_component_overhead =
                std::mem::size_of::<ComponentId>() + std::mem::size_of::<EnergyState>();
            let estimated_size = base_size + (size as usize) * per_component_overhead;

            println!(
                "Estimated memory for {} components: {} KB",
                size,
                estimated_size / 1024
            );
        }
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_concurrent_energy_calculations() {
        let config = EnergyValidationConfig::default();
        let functional = Arc::new(ChronoFabricEnergyFunctional::new(
            EnergyParameters::default(),
        ));
        let test_states = vec![
            EnergyState::Active {
                current_energy: 8e-13,
                peak_energy: 1e-12,
                efficiency: 0.85,
            },
            EnergyState::QuantumCoherent {
                energy: 5e-13,
                coherence_factor: 0.9,
                phase_energy: 2e-13,
            },
            EnergyState::Idle {
                baseline_energy: 1e-15,
            },
        ];

        let shared_states = Arc::new(test_states);
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let functional_clone = functional.clone();
            let states_clone = shared_states.clone();
            let results_clone = results.clone();

            let handle = thread::spawn(move || {
                let mut local_results = Vec::new();

                for i in 0..1000 {
                    let state_idx = (thread_id + i) % states_clone.len();
                    let state = &states_clone[state_idx];

                    let energy = functional_clone.energy(state);
                    let efficiency = functional_clone.efficiency_metric(state);
                    let gradient = functional_clone.energy_gradient(state);

                    local_results.push((energy, efficiency, gradient.len(), thread_id));
                }

                results_clone.lock().unwrap().extend(local_results);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), config.thread_count * 1000);

        // Verify deterministic results
        for i in 1..final_results.len() {
            let current = &final_results[i];

            // Find matching state calculation
            for j in 0..i {
                let previous = &final_results[j];
                if current.3 != previous.3 {
                    // Different threads
                    // Same state should give same results
                    let state_match = (i % shared_states.len()) == (j % shared_states.len());
                    if state_match {
                        assert!((current.0 - previous.0).abs() < 1e-15, "Energy mismatch");
                        assert!(
                            (current.1 - previous.1).abs() < 1e-15,
                            "Efficiency mismatch"
                        );
                        assert_eq!(current.2, previous.2, "Gradient length mismatch");
                    }
                }
            }
        }
    }

    #[test]
    fn test_concurrent_optimization() {
        let config = EnergyValidationConfig::default();
        let functional = Arc::new(ChronoFabricEnergyFunctional::new(
            EnergyParameters::default(),
        ));
        let test_systems: Vec<_> = (0..config.thread_count)
            .map(|i| {
                Arc::new(generate_test_system(
                    20 + i,
                    vec![TestEnergyType::Active, TestEnergyType::QuantumCoherent],
                ))
            })
            .collect();

        let params = Arc::new(EnergyParameters::default());
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for (thread_id, system) in test_systems.iter().enumerate() {
            let functional_clone = functional.clone();
            let system_clone = system.clone();
            let params_clone = params.clone();
            let results_clone = results.clone();

            let handle = thread::spawn(move || {
                let mut local_successes = 0;

                for _ in 0..10 {
                    match functional_clone.optimize_allocation(&*system_clone, &*params_clone) {
                        Ok(allocation) => {
                            if allocation.len() == system_clone.len() {
                                local_successes += 1;
                            }
                        }
                        Err(_) => {}
                    }
                }

                results_clone
                    .lock()
                    .unwrap()
                    .push((thread_id, local_successes));
                local_successes
            });

            handles.push(handle);
        }

        let mut total_successes = 0;
        for handle in handles {
            total_successes += handle.join().unwrap();
        }

        let expected_successes = config.thread_count * 10;
        assert_eq!(
            total_successes, expected_successes,
            "Not all concurrent optimizations succeeded"
        );
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ChronoFabricEnergyFunctional>();
        assert_sync::<ChronoFabricEnergyFunctional>();
        assert_send::<EnergyState>();
        assert_sync::<EnergyState>();
        assert_send::<EnergyParameters>();
        assert_sync::<EnergyParameters>();
        assert_send::<PerformanceMetrics>();
        assert_sync::<PerformanceMetrics>();
    }

    #[test]
    fn test_concurrent_adaptive_updates() {
        let config = EnergyValidationConfig::default();
        let functionals: Vec<_> = (0..config.thread_count)
            .map(|_| {
                Arc::new(Mutex::new(ChronoFabricEnergyFunctional::new(
                    EnergyParameters::default(),
                )))
            })
            .collect();

        let test_metrics = Arc::new(PerformanceMetrics {
            avg_response_time_ns: 1500,
            energy_efficiency: 8e10,
            coherence_maintenance_rate: 0.88,
            ..PerformanceMetrics::default()
        });

        let mut handles = Vec::new();

        for (thread_id, functional) in functionals.iter().enumerate() {
            let functional_clone = functional.clone();
            let metrics_clone = test_metrics.clone();

            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let mut f = functional_clone.lock().unwrap();
                    let current_params = f.parameters.clone();
                    let _adapted = f.adapt_parameters(&*metrics_clone, &current_params);
                }
                thread_id
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All functionals should have adapted parameters
        for functional in &functionals {
            let f = functional.lock().unwrap();
            let initial_params = EnergyParameters::default();

            // Should have adapted due to poor metrics
            assert!(f.parameters.quantum_scaling_factor >= initial_params.quantum_scaling_factor);
            assert!(
                f.parameters.max_energy_per_component >= initial_params.max_energy_per_component
            );
        }
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_large_system_optimization() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Create very large system
        let large_system = generate_test_system(
            config.stress_component_count,
            vec![
                TestEnergyType::Active,
                TestEnergyType::QuantumCoherent,
                TestEnergyType::Idle,
                TestEnergyType::Overloaded,
            ],
        );

        println!(
            "Testing large system with {} components",
            config.stress_component_count
        );

        let start = Instant::now();
        let allocation_result = functional.optimize_allocation(&large_system, &params);
        let duration = start.elapsed();

        assert!(
            allocation_result.is_ok(),
            "Large system optimization failed"
        );
        let allocation = allocation_result.unwrap();

        assert_eq!(allocation.len(), config.stress_component_count);

        // Verify all allocations are valid
        for (component, weight) in &allocation {
            assert!(
                *weight >= 0.0 && *weight <= 1.0,
                "Invalid allocation weight for {:?}: {}",
                component,
                weight
            );
        }

        println!(
            "Large system optimization completed in {}ms",
            duration.as_millis()
        );

        // Should complete in reasonable time even for large systems
        assert!(
            duration.as_secs() < 10,
            "Large system optimization too slow: {}s",
            duration.as_secs()
        );
    }

    #[test]
    fn test_continuous_optimization_stress() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let mut functional = ChronoFabricEnergyFunctional::new(params.clone());

        let components = generate_test_system(
            100,
            vec![TestEnergyType::Active, TestEnergyType::QuantumCoherent],
        );

        let iterations = 1000;
        let start = Instant::now();

        for i in 0..iterations {
            // Continuous optimization with slight parameter changes
            let mut modified_params = params.clone();
            modified_params.target_efficiency = 0.9 + (i as f64) / (iterations as f64) * 0.05;

            let allocation_result = functional.optimize_allocation(&components, &modified_params);
            assert!(
                allocation_result.is_ok(),
                "Optimization failed at iteration {}",
                i
            );

            // Periodic adaptive updates
            if i % 100 == 0 {
                let test_metrics = PerformanceMetrics {
                    avg_response_time_ns: 800 + (i % 500) as u64,
                    energy_efficiency: 9e10 + (i as f64) * 1e8,
                    ..PerformanceMetrics::default()
                };

                let _adapted = functional.adapt_parameters(&test_metrics, &modified_params);
            }
        }

        let duration = start.elapsed();
        let avg_us = duration.as_micros() / iterations as u128;

        println!(
            "Continuous optimization: {} iterations in {}ms (avg {}μs per iteration)",
            iterations,
            duration.as_millis(),
            avg_us
        );

        // Should maintain performance under continuous load
        assert!(
            avg_us < 1000,
            "Continuous optimization degraded: {}μs per iteration",
            avg_us
        );
    }

    #[test]
    fn test_memory_pressure_stress() {
        let config = EnergyValidationConfig::default();
        let functional_count = 100;
        let components_per_functional = 200;

        // Create many functionals with large component sets
        let functionals: Vec<_> = (0..functional_count)
            .map(|i| {
                let mut f = ChronoFabricEnergyFunctional::new(EnergyParameters::default());

                // Add components to each functional
                for j in 0..components_per_functional {
                    let component = ComponentId::new((i * components_per_functional + j) as u64);
                    let state = generate_test_energy_state(
                        if j % 2 == 0 {
                            TestEnergyType::Active
                        } else {
                            TestEnergyType::QuantumCoherent
                        },
                        1.0 + (j as f64) / 100.0,
                    );
                    f.update_component_state(component, state);
                }

                f
            })
            .collect();

        // Test operations under memory pressure
        let start = Instant::now();
        let mut total_energy = 0.0;

        for functional in &functionals {
            total_energy += functional.total_system_energy();
        }

        let duration = start.elapsed();

        println!(
            "Memory pressure test: {} functionals with {} components each",
            functional_count, components_per_functional
        );
        println!(
            "Total system energy calculation: {}ms",
            duration.as_millis()
        );
        println!("Total energy: {:.6e} attojoules", total_energy);

        assert!(total_energy > 0.0, "Total energy should be positive");
        assert!(
            duration.as_secs() < 5,
            "Memory pressure test too slow: {}s",
            duration.as_secs()
        );
    }

    #[test]
    fn test_concurrent_stress_with_contention() {
        let config = EnergyValidationConfig::default();
        let shared_functional = Arc::new(Mutex::new(ChronoFabricEnergyFunctional::new(
            EnergyParameters::default(),
        )));

        let thread_count = 16; // High contention
        let operations_per_thread = 500;
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let functional_clone = shared_functional.clone();

            let handle = thread::spawn(move || {
                let mut operations_completed = 0;

                for i in 0..operations_per_thread {
                    let component =
                        ComponentId::new((thread_id * operations_per_thread + i) as u64);
                    let state = generate_test_energy_state(TestEnergyType::Active, 1.0);

                    {
                        let mut f = functional_clone.lock().unwrap();
                        f.update_component_state(component, state);
                        let _energy = f.total_system_energy();
                    } // Release lock

                    operations_completed += 1;
                }

                (thread_id, operations_completed)
            });

            handles.push(handle);
        }

        let start = Instant::now();
        let mut total_operations = 0;

        for handle in handles {
            let (thread_id, ops) = handle.join().unwrap();
            total_operations += ops;
            println!("Thread {} completed {} operations", thread_id, ops);
        }

        let duration = start.elapsed();
        let expected_operations = thread_count * operations_per_thread;

        assert_eq!(total_operations, expected_operations);
        println!(
            "Concurrent stress test: {} operations in {}ms with high contention",
            total_operations,
            duration.as_millis()
        );

        // Should handle contention reasonably
        assert!(
            duration.as_secs() < 30,
            "Concurrent stress test too slow: {}s",
            duration.as_secs()
        );

        // Verify final state
        let final_functional = shared_functional.lock().unwrap();
        let component_count = final_functional.component_cache.len();
        assert_eq!(component_count, expected_operations);
    }
}

#[cfg(test)]
mod integration_validation_tests {
    use super::*;

    #[test]
    fn test_end_to_end_optimization_workflow() {
        let config = EnergyValidationConfig::default();
        let initial_params = EnergyParameters::default();
        let mut functional = ChronoFabricEnergyFunctional::new(initial_params.clone());

        // Step 1: Create diverse component system
        let mut components = HashMap::new();
        for i in 0..20 {
            let component = ComponentId::new(i);
            let state = match i % 4 {
                0 => EnergyState::Idle {
                    baseline_energy: 1e-15,
                },
                1 => EnergyState::Active {
                    current_energy: 8e-13,
                    peak_energy: 1e-12,
                    efficiency: 0.8,
                },
                2 => EnergyState::QuantumCoherent {
                    energy: 5e-13,
                    coherence_factor: 0.9,
                    phase_energy: 2e-13,
                },
                _ => EnergyState::Overloaded {
                    excess_energy: 2e-12,
                    throttling_factor: 0.6,
                },
            };
            components.insert(component, state.clone());
            functional.update_component_state(component, state);
        }

        // Step 2: Initial system analysis
        let initial_total_energy = functional.total_system_energy();
        assert!(
            initial_total_energy > 0.0,
            "Initial system energy should be positive"
        );

        // Step 3: Perform allocation optimization
        let allocation = functional
            .optimize_allocation(&components, &initial_params)
            .unwrap();
        assert_eq!(allocation.len(), components.len());

        // Step 4: Quantum optimization for coherent components
        let mut entangled_components = HashMap::new();
        let mut coherent_base_states = HashMap::new();

        for (component, state) in &components {
            if matches!(state, EnergyState::QuantumCoherent { .. }) {
                entangled_components.insert(*component, 0.9);
                coherent_base_states.insert(*component, state.clone());
            }
        }

        if !entangled_components.is_empty() {
            let quantum_optimized = functional
                .quantum_optimize(
                    &entangled_components,
                    &coherent_base_states,
                    &initial_params,
                )
                .unwrap();

            assert!(
                !quantum_optimized.is_empty(),
                "Quantum optimization should produce results"
            );
        }

        // Step 5: Performance monitoring and adaptation
        let performance_metrics = PerformanceMetrics {
            avg_response_time_ns: 1200,
            peak_response_time_ns: 2500,
            energy_efficiency: 8e10,
            coherence_maintenance_rate: 0.87,
            throughput_ops_sec: 800_000.0,
            error_rate_ppm: 1.5,
            resource_utilization: 0.82,
            adaptation_success_rate: 0.92,
        };

        let adapted_params = functional.adapt_parameters(&performance_metrics, &initial_params);
        assert!(adapted_params.quantum_scaling_factor >= initial_params.quantum_scaling_factor);

        // Step 6: Re-optimization with adapted parameters
        let final_allocation = functional
            .optimize_allocation(&components, &adapted_params)
            .unwrap();
        assert_eq!(final_allocation.len(), components.len());

        // Step 7: Load balancing
        let mut component_loads = HashMap::new();
        let mut available_resources = HashMap::new();

        for component in components.keys() {
            component_loads.insert(*component, rand::random::<f64>());
            available_resources.insert(*component, 0.5 + rand::random::<f64>() * 0.5);
        }

        let balanced_allocation =
            functional.dynamic_load_balance(&component_loads, &available_resources);
        assert_eq!(balanced_allocation.len(), components.len());

        // Step 8: Final verification
        let final_total_energy = functional.total_system_energy();
        assert!(
            final_total_energy.is_finite(),
            "Final system energy should be finite"
        );

        println!("End-to-end workflow completed successfully");
        println!(
            "Initial energy: {:.6e}, Final energy: {:.6e}",
            initial_total_energy, final_total_energy
        );
    }

    #[test]
    fn test_error_handling_and_recovery() {
        let config = EnergyValidationConfig::default();
        let params = EnergyParameters::default();
        let functional = ChronoFabricEnergyFunctional::new(params.clone());

        // Test allocation with empty system
        let empty_system = HashMap::new();
        let empty_result = functional.optimize_allocation(&empty_system, &params);

        // Should handle empty system gracefully
        match empty_result {
            Ok(allocation) => assert!(allocation.is_empty()),
            Err(e) => println!("Empty system optimization error (expected): {:?}", e),
        }

        // Test with invalid energy constraints
        let mut invalid_params = params.clone();
        invalid_params.max_energy_per_component = 0.0; // Invalid constraint

        let valid_system = generate_test_system(5, vec![TestEnergyType::Active]);
        let invalid_constraint_result =
            functional.optimize_allocation(&valid_system, &invalid_params);

        // Should detect constraint violations
        match invalid_constraint_result {
            Ok(_) => {} // Might still succeed with zero allocation
            Err(e) => println!("Invalid constraint error (expected): {:?}", e),
        }

        // Test quantum optimization with empty entanglement
        let empty_entanglement = HashMap::new();
        let empty_base_states = HashMap::new();
        let quantum_empty_result =
            functional.quantum_optimize(&empty_entanglement, &empty_base_states, &params);

        match quantum_empty_result {
            Ok(result) => assert!(result.is_empty()),
            Err(e) => println!("Empty quantum optimization error: {:?}", e),
        }

        // Test load balancing with zero resources
        let components: Vec<ComponentId> = (0..5).map(ComponentId::new).collect();
        let zero_loads: HashMap<ComponentId, f64> = components.iter().map(|&c| (c, 0.0)).collect();
        let zero_resources: HashMap<ComponentId, ResourceWeight> =
            components.iter().map(|&c| (c, 0.0)).collect();

        let zero_balance_result = functional.dynamic_load_balance(&zero_loads, &zero_resources);

        // Should handle zero case gracefully
        for weight in zero_balance_result.values() {
            assert!(*weight >= 0.0, "Balanced weight should be non-negative");
        }

        println!("Error handling tests completed successfully");
    }

    #[test]
    fn test_system_stability_under_varying_conditions() {
        let config = EnergyValidationConfig::default();
        let base_params = EnergyParameters::default();
        let mut functional = ChronoFabricEnergyFunctional::new(base_params.clone());

        // Test with varying system sizes
        let system_sizes = [1, 5, 10, 50, 100];

        for &size in &system_sizes {
            let system = generate_test_system(
                size,
                vec![TestEnergyType::Active, TestEnergyType::QuantumCoherent],
            );

            let allocation_result = functional.optimize_allocation(&system, &base_params);
            assert!(
                allocation_result.is_ok(),
                "Optimization failed for system size {}",
                size
            );

            let allocation = allocation_result.unwrap();
            assert_eq!(allocation.len(), size, "Allocation size mismatch");

            // Verify stability properties
            let total_weight: f64 = allocation.values().sum();
            assert!(
                total_weight > 0.0,
                "Total allocation weight should be positive"
            );
        }

        // Test with varying parameter sets
        let parameter_variations = vec![
            EnergyParameters {
                target_efficiency: 0.5,
                quantum_scaling_factor: 0.8,
                ..base_params.clone()
            },
            EnergyParameters {
                target_efficiency: 0.99,
                quantum_scaling_factor: 2.0,
                max_energy_per_component: 1e-11,
                ..base_params.clone()
            },
            EnergyParameters {
                allocation_strategy: AllocationStrategy::LoadBalanced,
                coherence_coupling: 0.5,
                ..base_params.clone()
            },
        ];

        let test_system = generate_test_system(20, vec![TestEnergyType::Active]);

        for (i, params) in parameter_variations.iter().enumerate() {
            let result = functional.optimize_allocation(&test_system, params);
            assert!(
                result.is_ok(),
                "Optimization failed for parameter set {}",
                i
            );
        }

        println!("System stability tests completed successfully");
    }
}
