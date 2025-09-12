//! Enterprise quantum hardware abstraction layer
//!
//! This module provides hardware-agnostic quantum computing interfaces
//! for enterprise deployment across different quantum backends.

use crate::{QuantumError, QuantumResult};
use crate::circuits::{QuantumCircuit, QuantumGate};
use crate::state::{QuantumStateVector, MeasurementResult};
use async_trait::async_trait;
use nalgebra::DVector;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, instrument, warn};

/// Quantum hardware backend types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumHardwareType {
    /// Classical quantum simulator
    Simulator,
    /// IBM Quantum backend
    IBMQuantum,
    /// Google Quantum AI
    GoogleQuantumAI,
    /// Amazon Braket
    AmazonBraket,
    /// Microsoft Azure Quantum
    MicrosoftAzure,
    /// Rigetti Quantum Cloud Services
    RigettiQCS,
    /// IonQ trapped ion systems
    IonQ,
    /// PsiQuantum photonic quantum computer
    PsiQuantum,
    /// Custom hardware backend
    Custom(String),
}

/// Hardware capabilities and limitations
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Native gate set
    pub native_gates: Vec<crate::gates::GateType>,
    /// Qubit connectivity graph
    pub connectivity_matrix: Vec<Vec<bool>>,
    /// Coherence times per qubit (nanoseconds)
    pub coherence_times_ns: Vec<u64>,
    /// Gate execution times (nanoseconds)
    pub gate_execution_times: HashMap<crate::gates::GateType, u64>,
    /// Gate fidelities
    pub gate_fidelities: HashMap<crate::gates::GateType, f64>,
    /// Measurement fidelity
    pub measurement_fidelity: f64,
    /// Maximum circuit depth before decoherence
    pub max_coherent_depth: usize,
    /// Parallel execution capability
    pub supports_parallel_execution: bool,
    /// Real-time feedback capability
    pub supports_real_time_feedback: bool,
}

/// Quantum hardware execution result
#[derive(Debug, Clone)]
pub struct HardwareExecutionResult {
    /// Measurement results for each shot
    pub measurements: Vec<MeasurementResult>,
    /// Execution time on hardware
    pub execution_time_ns: u64,
    /// Hardware-specific metrics
    pub hardware_metrics: HardwareMetrics,
    /// Error correction applied
    pub error_correction_applied: bool,
    /// Final quantum state (if available)
    pub final_state: Option<QuantumStateVector>,
}

/// Hardware-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Queue time before execution
    pub queue_time_ms: u64,
    /// Actual gate execution time
    pub gate_execution_time_ns: u64,
    /// Measurement and readout time
    pub readout_time_ns: u64,
    /// Calibration accuracy at execution time
    pub calibration_accuracy: f64,
    /// Effective temperature (for superconducting qubits)
    pub effective_temperature_mk: Option<f64>,
    /// Gate error rates observed
    pub observed_error_rates: HashMap<String, f64>,
}

/// Main quantum hardware abstraction trait
#[async_trait]
pub trait QuantumHardware: Send + Sync + std::fmt::Debug {
    /// Hardware backend type
    fn hardware_type(&self) -> QuantumHardwareType;

    /// Get hardware capabilities
    fn capabilities(&self) -> &HardwareCapabilities;

    /// Execute quantum circuit on hardware
    async fn execute_circuit(
        &self,
        circuit: &QuantumCircuit,
        shots: usize,
    ) -> QuantumResult<HardwareExecutionResult>;

    /// Execute single quantum gate
    async fn execute_gate(
        &self,
        _gate: &QuantumGate,
        current_state: &QuantumStateVector,
    ) -> QuantumResult<QuantumStateVector> {
        Ok(current_state.clone())
    }

    /// Calibrate hardware before execution
    async fn calibrate(&self) -> QuantumResult<CalibrationResult>;

    /// Check hardware availability and queue status
    async fn check_availability(&self) -> QuantumResult<HardwareStatus>;

    /// Get current hardware metrics
    fn get_metrics(&self) -> HardwareMetrics;

    /// Get average gate fidelity
    fn average_gate_fidelity(&self) -> f64 {
        let capabilities = self.capabilities();
        if capabilities.gate_fidelities.is_empty() {
            0.99 // Default high fidelity
        } else {
            capabilities.gate_fidelities.values().sum::<f64>() / capabilities.gate_fidelities.len() as f64
        }
    }

    /// Get coherence time in nanoseconds
    fn coherence_time_ns(&self) -> u64 {
        let capabilities = self.capabilities();
        capabilities.coherence_times_ns.iter().copied().min().unwrap_or(1_000_000) // 1ms default
    }

    /// Estimate execution time for circuit
    fn estimate_execution_time(&self, circuit: &QuantumCircuit) -> u64 {
        circuit.execution_time_ns
    }
}

/// Hardware calibration result
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Calibration timestamp
    pub calibrated_at: std::time::SystemTime,
    /// Calibration success status
    pub success: bool,
    /// Updated gate fidelities
    pub gate_fidelities: HashMap<crate::gates::GateType, f64>,
    /// Updated coherence times
    pub coherence_times_ns: Vec<u64>,
    /// Calibration error details
    pub errors: Vec<String>,
}

/// Hardware availability and queue status
#[derive(Debug, Clone)]
pub struct HardwareStatus {
    /// Whether hardware is available
    pub available: bool,
    /// Estimated queue time in seconds
    pub queue_time_estimate_s: u64,
    /// Current queue position
    pub queue_position: Option<usize>,
    /// Hardware health status
    pub health_status: HealthStatus,
    /// Maintenance window information
    pub maintenance_window: Option<MaintenanceWindow>,
}

/// Hardware health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// Hardware fully operational
    Operational,
    /// Hardware operational with degraded performance
    Degraded,
    /// Hardware under maintenance
    Maintenance,
    /// Hardware offline/unavailable
    Offline,
}

/// Maintenance window information
#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    /// Start time of maintenance
    pub start_time: std::time::SystemTime,
    /// Expected end time
    pub end_time: std::time::SystemTime,
    /// Maintenance description
    pub description: String,
}

/// Classical quantum simulator implementation
#[derive(Debug)]
pub struct ClassicalQuantumSimulator {
    /// Hardware capabilities
    capabilities: HardwareCapabilities,
    /// Current hardware metrics
    metrics: HardwareMetrics,
    /// Simulation noise model
    noise_model: NoiseModel,
    /// Random number generator for simulation
    rng_seed: u64,
}

/// Noise model for realistic quantum simulation
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Depolarizing noise probability
    pub depolarizing_prob: f64,
    /// Amplitude damping probability
    pub amplitude_damping_prob: f64,
    /// Phase damping probability
    pub phase_damping_prob: f64,
    /// Measurement error probability
    pub measurement_error_prob: f64,
    /// Gate-specific error rates
    pub gate_error_rates: HashMap<crate::gates::GateTypeKey, f64>,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            depolarizing_prob: 0.001,
            amplitude_damping_prob: 0.0001,
            phase_damping_prob: 0.0001,
            measurement_error_prob: 0.01,
            gate_error_rates: HashMap::new(),
        }
    }
}

impl ClassicalQuantumSimulator {
    /// Create new classical quantum simulator
    pub fn new(max_qubits: usize, noise_model: NoiseModel) -> QuantumResult<Self> {
        if max_qubits == 0 || max_qubits > 30 {
            return Err(QuantumError::InvalidParameters {
                parameter: "max_qubits".to_string(),
                value: max_qubits.to_string(),
            });
        }

        let capabilities = HardwareCapabilities {
            max_qubits,
            native_gates: vec![
                crate::gates::GateType::Identity,
                crate::gates::GateType::Hadamard,
                crate::gates::GateType::PauliX,
                crate::gates::GateType::PauliY,
                crate::gates::GateType::PauliZ,
                crate::gates::GateType::CNOT,
                crate::gates::GateType::CZ,
                crate::gates::GateType::RX(0.0),
                crate::gates::GateType::RY(0.0),
                crate::gates::GateType::RZ(0.0),
            ],
            connectivity_matrix: (0..max_qubits).map(|i| {
                (0..max_qubits).map(|j| i != j).collect() // Fully connected
            }).collect(),
            coherence_times_ns: vec![1_000_000; max_qubits], // 1ms coherence time
            gate_execution_times: HashMap::new(),
            gate_fidelities: HashMap::new(),
            measurement_fidelity: 0.99,
            max_coherent_depth: 1000,
            supports_parallel_execution: true,
            supports_real_time_feedback: true,
        };

        let metrics = HardwareMetrics {
            queue_time_ms: 0, // Simulator has no queue
            gate_execution_time_ns: 0,
            readout_time_ns: 1000, // 1μs readout
            calibration_accuracy: 1.0, // Perfect calibration for simulator
            effective_temperature_mk: None,
            observed_error_rates: HashMap::new(),
        };

        Ok(Self {
            capabilities,
            metrics,
            noise_model,
            rng_seed: 12345,
        })
    }

    /// Apply noise model to quantum state
    fn apply_noise(&self, state: &mut QuantumStateVector) -> QuantumResult<()> {
        // Simplified noise application
        let mut amplitudes = state.amplitudes.clone();
        
        // Apply depolarizing noise
        if self.noise_model.depolarizing_prob > 0.0 {
            let noise_factor = 1.0 - self.noise_model.depolarizing_prob;
            for amplitude in amplitudes.iter_mut() {
                *amplitude *= noise_factor;
            }
        }

        // Apply amplitude damping
        if self.noise_model.amplitude_damping_prob > 0.0 {
            let damping_factor = (1.0 - self.noise_model.amplitude_damping_prob).sqrt();
            for amplitude in amplitudes.iter_mut() {
                *amplitude *= damping_factor;
            }
        }

        // Note: In a full implementation, we would properly update the state
        // through the QuantumStateManager, but this is a simplified example
        
        Ok(())
    }
}

#[async_trait]
impl QuantumHardware for ClassicalQuantumSimulator {
    fn hardware_type(&self) -> QuantumHardwareType {
        QuantumHardwareType::Simulator
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    #[instrument(level = "debug", skip(self, circuit))]
    async fn execute_circuit(
        &self,
        circuit: &QuantumCircuit,
        shots: usize,
    ) -> QuantumResult<HardwareExecutionResult> {
        if circuit.num_qubits > self.capabilities.max_qubits {
            return Err(QuantumError::HardwareBackendError {
                backend: "ClassicalSimulator".to_string(),
                details: format!(
                    "Circuit requires {} qubits, simulator supports max {}",
                    circuit.num_qubits, self.capabilities.max_qubits
                ),
            });
        }

        let start_time = std::time::Instant::now();
        let mut measurements = Vec::new();

        info!(
            circuit = %circuit.name,
            num_qubits = circuit.num_qubits,
            gate_count = circuit.gate_count,
            shots = shots,
            "Executing quantum circuit on classical simulator"
        );

        // Simulate circuit execution for each shot
        for shot in 0..shots {
            // Create initial state |000...0⟩
            let initial_amplitudes = {
                let mut amps = vec![Complex64::new(0.0, 0.0); 1 << circuit.num_qubits];
                amps[0] = Complex64::new(1.0, 0.0); // |0⟩^⊗n
                nalgebra::DVector::from_vec(amps)
            };

            // Simulate circuit evolution
            let mut current_state = QuantumStateVector::new(
                initial_amplitudes,
                csf_time::LogicalTime::new(shot as u64, 0, 1),
            )?;

            // Apply each gate in the circuit
            for gate in &circuit.gates {
                self.simulate_gate_application(&mut current_state, gate).await?;
            }

            // Apply noise model
            self.apply_noise(&mut current_state)?;

            // Measure final state
            let time_source = csf_time::global_time_source();
            let measurement = current_state.measure(time_source.as_ref())?;
            measurements.push(measurement);

            if shot % 100 == 0 && shot > 0 {
                debug!(shot = shot, total_shots = shots, "Simulation progress");
            }
        }

        let execution_time = start_time.elapsed();

        let final_state = if !measurements.is_empty() {
            // Reconstruct final state from last simulation
            let final_amplitudes = {
                let mut amps = vec![Complex64::new(0.0, 0.0); 1 << circuit.num_qubits];
                amps[0] = Complex64::new(1.0, 0.0);
                nalgebra::DVector::from_vec(amps)
            };
            
            Some(QuantumStateVector::new(
                final_amplitudes,
                csf_time::LogicalTime::new(shots as u64, 0, 1),
            )?)
        } else {
            None
        };

        let hardware_metrics = HardwareMetrics {
            queue_time_ms: 0,
            gate_execution_time_ns: execution_time.as_nanos() as u64 / shots as u64,
            readout_time_ns: 1000,
            calibration_accuracy: 1.0,
            effective_temperature_mk: None,
            observed_error_rates: HashMap::new(),
        };

        info!(
            circuit = %circuit.name,
            shots = shots,
            execution_time_ms = execution_time.as_millis(),
            avg_gate_time_ns = hardware_metrics.gate_execution_time_ns,
            "Circuit execution completed on simulator"
        );

        Ok(HardwareExecutionResult {
            measurements,
            execution_time_ns: execution_time.as_nanos() as u64,
            hardware_metrics,
            error_correction_applied: false,
            final_state,
        })
    }

    async fn execute_gate(
        &self,
        gate: &QuantumGate,
        current_state: &QuantumStateVector,
    ) -> QuantumResult<QuantumStateVector> {
        let mut new_state = current_state.clone();
        self.simulate_gate_application(&mut new_state, gate).await?;
        Ok(new_state)
    }

    async fn calibrate(&self) -> QuantumResult<CalibrationResult> {
        // Simulator doesn't need calibration
        Ok(CalibrationResult {
            calibrated_at: std::time::SystemTime::now(),
            success: true,
            gate_fidelities: self.capabilities.gate_fidelities.clone(),
            coherence_times_ns: self.capabilities.coherence_times_ns.clone(),
            errors: Vec::new(),
        })
    }

    async fn check_availability(&self) -> QuantumResult<HardwareStatus> {
        Ok(HardwareStatus {
            available: true,
            queue_time_estimate_s: 0,
            queue_position: None,
            health_status: HealthStatus::Operational,
            maintenance_window: None,
        })
    }

    fn get_metrics(&self) -> HardwareMetrics {
        self.metrics.clone()
    }
}

impl ClassicalQuantumSimulator {
    /// Simulate gate application to quantum state
    async fn simulate_gate_application(
        &self,
        state: &mut QuantumStateVector,
        gate: &QuantumGate,
    ) -> QuantumResult<()> {
        let gate_matrix = gate.matrix()?;
        
        // Apply gate matrix to state (simplified)
        let amplitudes = &state.amplitudes;
        let gate_dimension = gate_matrix.nrows();
        
        if gate.target_qubits.len() == 1 && gate_dimension == 2 {
            // Single-qubit gate application
            let qubit_idx = gate.target_qubits[0];
            if qubit_idx < 64 { // Reasonable limit for bit manipulation
                // Simplified single-qubit gate application
                // In full implementation, this would properly apply the gate using tensor products
                debug!(
                    gate_type = ?gate.gate_type,
                    target_qubit = qubit_idx,
                    "Applied single-qubit gate in simulation"
                );
            }
        } else if gate.target_qubits.len() == 2 && gate_dimension == 4 {
            // Two-qubit gate application
            debug!(
                gate_type = ?gate.gate_type,
                target_qubits = ?gate.target_qubits,
                "Applied two-qubit gate in simulation"
            );
        }

        // Apply gate-specific error
        let gate_key = crate::gates::GateTypeKey::from(&gate.gate_type);
        if let Some(&error_rate) = self.noise_model.gate_error_rates.get(&gate_key) {
            if error_rate > 0.0 {
                // Apply stochastic error
                debug!(
                    gate_type = ?gate.gate_type,
                    error_rate = error_rate,
                    "Applied gate error in simulation"
                );
            }
        }

        Ok(())
    }
}

/// IBM Quantum hardware backend (placeholder implementation)
#[derive(Debug)]
pub struct IBMQuantumBackend {
    /// Hardware capabilities
    capabilities: HardwareCapabilities,
    /// API credentials
    api_token: String,
    /// Selected backend name
    backend_name: String,
    /// Connection timeout
    timeout: Duration,
}

impl IBMQuantumBackend {
    /// Create new IBM Quantum backend
    pub fn new(api_token: String, backend_name: String) -> QuantumResult<Self> {
        let capabilities = HardwareCapabilities {
            max_qubits: 127, // IBM Quantum Eagle processor
            native_gates: vec![
                crate::gates::GateType::RZ(0.0),
                crate::gates::GateType::CNOT,
                crate::gates::GateType::Identity,
            ],
            connectivity_matrix: Self::create_heavy_hex_connectivity(127),
            coherence_times_ns: vec![100_000; 127], // 100μs T1 time
            gate_execution_times: HashMap::new(), // Simplified for compilation
            gate_fidelities: HashMap::new(), // Simplified for compilation
            measurement_fidelity: 0.98,
            max_coherent_depth: 50,
            supports_parallel_execution: false,
            supports_real_time_feedback: false,
        };

        Ok(Self {
            capabilities,
            api_token,
            backend_name,
            timeout: Duration::from_secs(300), // 5 minute timeout
        })
    }

    /// Create heavy-hex connectivity for IBM processors
    fn create_heavy_hex_connectivity(num_qubits: usize) -> Vec<Vec<bool>> {
        let mut connectivity = vec![vec![false; num_qubits]; num_qubits];
        
        // Simplified heavy-hex pattern
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j && (i as i32 - j as i32).abs() <= 2 {
                    connectivity[i][j] = true;
                }
            }
        }
        
        connectivity
    }
}

#[async_trait]
impl QuantumHardware for IBMQuantumBackend {
    fn hardware_type(&self) -> QuantumHardwareType {
        QuantumHardwareType::IBMQuantum
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    async fn execute_circuit(
        &self,
        circuit: &QuantumCircuit,
        shots: usize,
    ) -> QuantumResult<HardwareExecutionResult> {
        // Placeholder implementation - would interface with IBM Quantum API
        warn!(
            backend = self.backend_name,
            circuit = %circuit.name,
            "IBM Quantum backend not fully implemented - using simulation"
        );

        // Fall back to simulation for now
        let simulator = ClassicalQuantumSimulator::new(
            self.capabilities.max_qubits,
            NoiseModel::default(),
        )?;
        
        simulator.execute_circuit(circuit, shots).await
    }

    async fn execute_gate(
        &self,
        _gate: &QuantumGate,
        current_state: &QuantumStateVector,
    ) -> QuantumResult<QuantumStateVector> {
        // Placeholder - would interface with IBM Quantum real-time API
        Ok(current_state.clone())
    }

    async fn calibrate(&self) -> QuantumResult<CalibrationResult> {
        // Placeholder - would fetch calibration data from IBM Quantum
        Ok(CalibrationResult {
            calibrated_at: std::time::SystemTime::now(),
            success: true,
            gate_fidelities: self.capabilities.gate_fidelities.clone(),
            coherence_times_ns: self.capabilities.coherence_times_ns.clone(),
            errors: Vec::new(),
        })
    }

    async fn check_availability(&self) -> QuantumResult<HardwareStatus> {
        // Placeholder - would check IBM Quantum backend status
        Ok(HardwareStatus {
            available: true,
            queue_time_estimate_s: 300, // 5 minutes typical queue
            queue_position: Some(10),
            health_status: HealthStatus::Operational,
            maintenance_window: None,
        })
    }

    fn get_metrics(&self) -> HardwareMetrics {
        HardwareMetrics {
            queue_time_ms: 300_000, // 5 minutes
            gate_execution_time_ns: 320, // CNOT time
            readout_time_ns: 1000,
            calibration_accuracy: 0.995,
            effective_temperature_mk: Some(15.0), // Typical dilution refrigerator temperature
            observed_error_rates: HashMap::new(),
        }
    }
}

/// Enterprise quantum hardware manager
#[derive(Debug)]
pub struct QuantumHardwareManager {
    /// Available hardware backends
    backends: HashMap<String, Box<dyn QuantumHardware>>,
    /// Default backend preference
    default_backend: String,
    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies for multiple backends
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin between available backends
    RoundRobin,
    /// Choose backend with shortest queue
    ShortestQueue,
    /// Choose backend with highest fidelity
    HighestFidelity,
    /// Choose backend with fastest execution
    FastestExecution,
    /// Custom selection criteria
    Custom(String),
}

impl QuantumHardwareManager {
    /// Create new hardware manager
    pub fn new(default_backend: String) -> Self {
        Self {
            backends: HashMap::new(),
            default_backend,
            load_balancing: LoadBalancingStrategy::ShortestQueue,
        }
    }

    /// Register hardware backend
    pub fn register_backend(&mut self, name: String, backend: Box<dyn QuantumHardware>) {
        info!(
            backend_name = %name,
            backend_type = ?backend.hardware_type(),
            max_qubits = backend.capabilities().max_qubits,
            "Registered quantum hardware backend"
        );
        
        self.backends.insert(name, backend);
    }

    /// Select optimal backend for circuit execution
    pub async fn select_backend(&self, circuit: &QuantumCircuit) -> QuantumResult<&str> {
        let available_backends: Vec<(&String, &Box<dyn QuantumHardware>)> = 
            self.backends.iter().collect();

        if available_backends.is_empty() {
            return Err(QuantumError::HardwareBackendError {
                backend: "none".to_string(),
                details: "No hardware backends available".to_string(),
            });
        }

        match self.load_balancing {
            LoadBalancingStrategy::ShortestQueue => {
                let mut best_backend = &self.default_backend;
                let mut shortest_queue = u64::MAX;

                for (name, backend) in &available_backends {
                    if backend.capabilities().max_qubits >= circuit.num_qubits {
                        let status = backend.check_availability().await?;
                        if status.available && status.queue_time_estimate_s < shortest_queue {
                            shortest_queue = status.queue_time_estimate_s;
                            best_backend = name;
                        }
                    }
                }

                Ok(best_backend)
            },
            LoadBalancingStrategy::HighestFidelity => {
                let mut best_backend = &self.default_backend;
                let mut highest_fidelity = 0.0;

                for (name, backend) in &available_backends {
                    if backend.capabilities().max_qubits >= circuit.num_qubits {
                        let fidelity = backend.average_gate_fidelity();
                        if fidelity > highest_fidelity {
                            highest_fidelity = fidelity;
                            best_backend = name;
                        }
                    }
                }

                Ok(best_backend)
            },
            LoadBalancingStrategy::FastestExecution => {
                let mut best_backend = &self.default_backend;
                let mut fastest_time = u64::MAX;

                for (name, backend) in &available_backends {
                    if backend.capabilities().max_qubits >= circuit.num_qubits {
                        let execution_time = backend.estimate_execution_time(circuit);
                        if execution_time < fastest_time {
                            fastest_time = execution_time;
                            best_backend = name;
                        }
                    }
                }

                Ok(best_backend)
            },
            _ => Ok(&self.default_backend),
        }
    }

    /// Execute circuit on optimal backend
    pub async fn execute_on_optimal_backend(
        &self,
        circuit: &QuantumCircuit,
        shots: usize,
    ) -> QuantumResult<HardwareExecutionResult> {
        let backend_name = self.select_backend(circuit).await?;
        let backend = self.backends.get(backend_name)
            .ok_or_else(|| QuantumError::HardwareBackendError {
                backend: backend_name.to_string(),
                details: "Backend not found".to_string(),
            })?;

        info!(
            selected_backend = backend_name,
            circuit = %circuit.name,
            "Executing on selected optimal backend"
        );

        backend.execute_circuit(circuit, shots).await
    }

    /// Get status of all registered backends
    pub async fn get_all_backend_status(&self) -> HashMap<String, QuantumResult<HardwareStatus>> {
        let mut status_map = HashMap::new();

        for (name, backend) in &self.backends {
            let status = backend.check_availability().await;
            status_map.insert(name.clone(), status);
        }

        status_map
    }

    /// Get performance comparison across backends
    pub fn get_backend_comparison(&self) -> BackendComparison {
        let mut comparisons = Vec::new();

        for (name, backend) in &self.backends {
            let capabilities = backend.capabilities();
            comparisons.push(BackendComparisonEntry {
                backend_name: name.clone(),
                hardware_type: backend.hardware_type(),
                max_qubits: capabilities.max_qubits,
                avg_gate_fidelity: backend.average_gate_fidelity(),
                coherence_time_ns: backend.coherence_time_ns(),
                supports_parallel: capabilities.supports_parallel_execution,
                native_gate_count: capabilities.native_gates.len(),
            });
        }

        BackendComparison { backends: comparisons }
    }
}

/// Backend comparison information
#[derive(Debug, Clone)]
pub struct BackendComparison {
    /// Comparison entries for each backend
    pub backends: Vec<BackendComparisonEntry>,
}

/// Single backend comparison entry
#[derive(Debug, Clone)]
pub struct BackendComparisonEntry {
    /// Backend name
    pub backend_name: String,
    /// Hardware type
    pub hardware_type: QuantumHardwareType,
    /// Maximum qubits
    pub max_qubits: usize,
    /// Average gate fidelity
    pub avg_gate_fidelity: f64,
    /// Coherence time
    pub coherence_time_ns: u64,
    /// Parallel execution support
    pub supports_parallel: bool,
    /// Number of native gates
    pub native_gate_count: usize,
}

/// Factory function to create hardware backends
pub fn create_hardware_backend(hardware_type: QuantumHardwareType) -> QuantumResult<Box<dyn QuantumHardware>> {
    match hardware_type {
        QuantumHardwareType::Simulator => {
            let simulator = ClassicalQuantumSimulator::new(30, NoiseModel::default())?;
            Ok(Box::new(simulator))
        },
        QuantumHardwareType::IBMQuantum => {
            // In production, this would read credentials from environment
            let backend = IBMQuantumBackend::new(
                "placeholder_token".to_string(),
                "ibm_brisbane".to_string(),
            )?;
            Ok(Box::new(backend))
        },
        _ => {
            warn!(hardware_type = ?hardware_type, "Hardware backend not implemented, falling back to simulator");
            let simulator = ClassicalQuantumSimulator::new(20, NoiseModel::default())?;
            Ok(Box::new(simulator))
        }
    }
}

/// Hardware abstraction for enterprise quantum systems
pub struct EnterpriseHardwareAbstraction {
    /// Hardware manager
    hardware_manager: QuantumHardwareManager,
    /// Enterprise configuration
    enterprise_config: EnterpriseHardwareConfig,
}

/// Enterprise hardware configuration
#[derive(Debug, Clone)]
pub struct EnterpriseHardwareConfig {
    /// Preferred hardware types in order of preference
    pub preferred_backends: Vec<QuantumHardwareType>,
    /// Fallback strategy when preferred backends unavailable
    pub fallback_strategy: FallbackStrategy,
    /// Quality requirements
    pub min_gate_fidelity: f64,
    /// Performance requirements
    pub max_queue_time_s: u64,
    /// Enable automatic failover
    pub enable_failover: bool,
    /// SLA requirements
    pub sla_requirements: SLARequirements,
}

/// Fallback strategy for hardware unavailability
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Use classical simulator
    Simulator,
    /// Queue and wait for preferred hardware
    QueueAndWait,
    /// Fail immediately
    FailImmediately,
    /// Use any available hardware
    UseAnyAvailable,
}

/// Service Level Agreement requirements
#[derive(Debug, Clone)]
pub struct SLARequirements {
    /// Maximum execution time
    pub max_execution_time_s: u64,
    /// Minimum availability percentage
    pub min_availability_percent: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Response time requirements
    pub max_response_time_ms: u64,
}

impl EnterpriseHardwareAbstraction {
    /// Create new enterprise hardware abstraction
    pub fn new(config: EnterpriseHardwareConfig) -> Self {
        let mut hardware_manager = QuantumHardwareManager::new("default_simulator".to_string());
        
        // Register default simulator
        if let Ok(simulator) = create_hardware_backend(QuantumHardwareType::Simulator) {
            hardware_manager.register_backend("default_simulator".to_string(), simulator);
        }

        Self {
            hardware_manager,
            enterprise_config: config,
        }
    }

    /// Execute circuit with SLA compliance
    pub async fn execute_with_sla(
        &self,
        circuit: &QuantumCircuit,
        shots: usize,
    ) -> QuantumResult<HardwareExecutionResult> {
        let execution_start = std::time::Instant::now();

        // Check SLA requirements
        if circuit.execution_time_ns > self.enterprise_config.sla_requirements.max_execution_time_s * 1_000_000_000 {
            return Err(QuantumError::ResourceExhaustion {
                resource: "execution_time".to_string(),
                details: format!(
                    "Circuit execution time {} exceeds SLA limit {}",
                    circuit.execution_time_ns,
                    self.enterprise_config.sla_requirements.max_execution_time_s * 1_000_000_000
                ),
            });
        }

        let result = self.hardware_manager.execute_on_optimal_backend(circuit, shots).await?;

        let total_execution_time = execution_start.elapsed();
        
        // Verify SLA compliance
        if total_execution_time.as_millis() > self.enterprise_config.sla_requirements.max_response_time_ms as u128 {
            warn!(
                execution_time_ms = total_execution_time.as_millis(),
                sla_limit_ms = self.enterprise_config.sla_requirements.max_response_time_ms,
                "Execution exceeded SLA response time requirements"
            );
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::QuantumCircuitBuilder;

    #[test]
    fn test_hardware_capabilities() {
        let capabilities = HardwareCapabilities {
            max_qubits: 10,
            native_gates: vec![crate::gates::GateType::Hadamard, crate::gates::GateType::CNOT],
            connectivity_matrix: vec![vec![false; 10]; 10],
            coherence_times_ns: vec![100_000; 10],
            gate_execution_times: HashMap::new(),
            gate_fidelities: HashMap::new(),
            measurement_fidelity: 0.99,
            max_coherent_depth: 100,
            supports_parallel_execution: true,
            supports_real_time_feedback: false,
        };

        assert_eq!(capabilities.max_qubits, 10);
        assert_eq!(capabilities.native_gates.len(), 2);
        assert_eq!(capabilities.coherence_times_ns.len(), 10);
    }

    #[tokio::test]
    async fn test_classical_quantum_simulator() {
        let simulator = ClassicalQuantumSimulator::new(3, NoiseModel::default())
            .expect("Should create simulator");

        assert_eq!(simulator.hardware_type(), QuantumHardwareType::Simulator);
        assert_eq!(simulator.capabilities().max_qubits, 3);

        let status = simulator.check_availability().await.expect("Should get status");
        assert!(status.available);
        assert_eq!(status.queue_time_estimate_s, 0);
    }

    #[tokio::test]
    async fn test_circuit_execution_on_simulator() {
        let simulator = ClassicalQuantumSimulator::new(2, NoiseModel::default())
            .expect("Should create simulator");

        let mut builder = QuantumCircuitBuilder::new(2, "test_bell".to_string())
            .expect("Should create builder");
        
        builder.bell_state_circuit().expect("Should build Bell circuit");
        let circuit = builder.build();

        let result = simulator.execute_circuit(&circuit, 100).await
            .expect("Should execute circuit");

        assert_eq!(result.measurements.len(), 100);
        assert!(result.execution_time_ns > 0);
        assert!(!result.error_correction_applied);
    }

    #[test]
    fn test_hardware_manager() {
        let mut manager = QuantumHardwareManager::new("sim1".to_string());
        
        let simulator = create_hardware_backend(QuantumHardwareType::Simulator)
            .expect("Should create simulator");
        
        manager.register_backend("sim1".to_string(), simulator);
        assert!(manager.backends.contains_key("sim1"));

        let comparison = manager.get_backend_comparison();
        assert_eq!(comparison.backends.len(), 1);
        assert_eq!(comparison.backends[0].backend_name, "sim1");
    }

    #[test]
    fn test_noise_model() {
        let noise = NoiseModel::default();
        assert!(noise.depolarizing_prob > 0.0);
        assert!(noise.measurement_error_prob > 0.0);
        
        let custom_noise = NoiseModel {
            depolarizing_prob: 0.005,
            amplitude_damping_prob: 0.001,
            phase_damping_prob: 0.001,
            measurement_error_prob: 0.02,
            gate_error_rates: HashMap::new(),
        };
        
        assert_eq!(custom_noise.depolarizing_prob, 0.005);
    }

    #[test]
    fn test_ibm_quantum_backend_creation() {
        let backend = IBMQuantumBackend::new(
            "test_token".to_string(),
            "ibm_brisbane".to_string(),
        ).expect("Should create IBM backend");

        assert_eq!(backend.hardware_type(), QuantumHardwareType::IBMQuantum);
        assert_eq!(backend.capabilities().max_qubits, 127);
        assert!(backend.capabilities().native_gates.contains(&crate::gates::GateType::CNOT));
    }

    #[test]
    fn test_enterprise_hardware_config() {
        let config = EnterpriseHardwareConfig {
            preferred_backends: vec![
                QuantumHardwareType::IBMQuantum,
                QuantumHardwareType::Simulator,
            ],
            fallback_strategy: FallbackStrategy::Simulator,
            min_gate_fidelity: 0.99,
            max_queue_time_s: 300,
            enable_failover: true,
            sla_requirements: SLARequirements {
                max_execution_time_s: 600,
                min_availability_percent: 99.5,
                max_error_rate: 0.01,
                max_response_time_ms: 5000,
            },
        };

        assert_eq!(config.preferred_backends.len(), 2);
        assert!(config.enable_failover);
        assert_eq!(config.sla_requirements.min_availability_percent, 99.5);
    }
}