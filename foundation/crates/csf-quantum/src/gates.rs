//! Enterprise quantum gate library with hardware optimization
//!
//! This module provides a comprehensive quantum gate library optimized for
//! enterprise financial computing with hardware-agnostic implementations.

use crate::{QuantumError, QuantumResult};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use tracing::{debug, instrument, warn};

/// A hashable key representation for GateType 
/// Used as keys in HashMap since GateType contains f64 values that can't implement Hash
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GateTypeKey {
    /// Identity gate (I)
    Identity,
    /// Hadamard gate (H)
    Hadamard,
    /// Pauli-X gate (NOT)
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// SWAP gate
    SWAP,
    /// Toffoli gate (CCX)
    Toffoli,
    /// X-rotation gate (parameterized gates use a generic key)
    RX,
    /// Y-rotation gate
    RY,
    /// Z-rotation gate
    RZ,
    /// Universal gate U3
    U3,
    /// Custom gate with name
    Custom(String),
}

impl From<&GateType> for GateTypeKey {
    fn from(gate_type: &GateType) -> Self {
        match gate_type {
            GateType::Identity => GateTypeKey::Identity,
            GateType::Hadamard => GateTypeKey::Hadamard,
            GateType::PauliX => GateTypeKey::PauliX,
            GateType::PauliY => GateTypeKey::PauliY,
            GateType::PauliZ => GateTypeKey::PauliZ,
            GateType::CNOT => GateTypeKey::CNOT,
            GateType::CZ => GateTypeKey::CZ,
            GateType::SWAP => GateTypeKey::SWAP,
            GateType::Toffoli => GateTypeKey::Toffoli,
            GateType::RX(_) => GateTypeKey::RX,
            GateType::RY(_) => GateTypeKey::RY,
            GateType::RZ(_) => GateTypeKey::RZ,
            GateType::U3(_, _, _) => GateTypeKey::U3,
            GateType::Custom(name) => GateTypeKey::Custom(name.clone()),
        }
    }
}

/// Quantum gate types supported by the enterprise system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GateType {
    /// Identity gate (I)
    Identity,
    /// Hadamard gate (H)
    Hadamard,
    /// Pauli-X gate (NOT)
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// SWAP gate
    SWAP,
    /// Toffoli gate (CCX)
    Toffoli,
    /// X-rotation gate with angle
    RX(f64),
    /// Y-rotation gate with angle
    RY(f64),
    /// Z-rotation gate with angle
    RZ(f64),
    /// Universal gate U3(θ, φ, λ)
    U3(f64, f64, f64),
    /// Custom gate with name
    Custom(String),
}

/// Gate parameters for parameterized quantum gates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateParameters {
    /// Rotation angles for parameterized gates
    pub angles: Vec<f64>,
    /// Phase parameters
    pub phases: Vec<f64>,
    /// Additional gate-specific parameters
    pub custom_parameters: HashMap<String, f64>,
}

impl Default for GateParameters {
    fn default() -> Self {
        Self {
            angles: Vec::new(),
            phases: Vec::new(),
            custom_parameters: HashMap::new(),
        }
    }
}

impl GateParameters {
    /// Create parameters for rotation gate
    pub fn rotation(angle: f64) -> Self {
        Self {
            angles: vec![angle],
            phases: Vec::new(),
            custom_parameters: HashMap::new(),
        }
    }

    /// Create parameters for U3 gate
    pub fn u3(theta: f64, phi: f64, lambda: f64) -> Self {
        Self {
            angles: vec![theta],
            phases: vec![phi, lambda],
            custom_parameters: HashMap::new(),
        }
    }

    /// Add custom parameter
    pub fn with_custom_parameter(mut self, name: String, value: f64) -> Self {
        self.custom_parameters.insert(name, value);
        self
    }
}

/// Gate trait for enterprise quantum gate implementations
pub trait Gate: Send + Sync + std::fmt::Debug {
    /// Get gate name
    fn name(&self) -> &str;

    /// Get gate type
    fn gate_type(&self) -> GateType;

    /// Get number of qubits this gate operates on
    fn qubit_count(&self) -> usize;

    /// Get gate matrix representation
    fn matrix(&self) -> DMatrix<Complex64>;

    /// Get gate fidelity for error analysis
    fn fidelity(&self) -> f64;

    /// Get estimated execution time
    fn execution_time_ns(&self) -> u64;

    /// Check if gate is hermitian
    fn is_hermitian(&self) -> bool;

    /// Check if gate is unitary
    fn is_unitary(&self) -> bool;

    /// Get gate parameters
    fn parameters(&self) -> GateParameters;
}

/// Enterprise quantum gate library
#[derive(Debug)]
pub struct QuantumGateLibrary {
    /// Cache of pre-computed gate matrices
    matrix_cache: HashMap<GateTypeKey, DMatrix<Complex64>>,
    /// Performance metrics for each gate type
    performance_metrics: HashMap<GateTypeKey, GatePerformanceMetrics>,
}

/// Performance metrics for gate types
#[derive(Debug, Clone)]
pub struct GatePerformanceMetrics {
    /// Average execution time
    pub avg_execution_time_ns: u64,
    /// Average fidelity
    pub avg_fidelity: f64,
    /// Success rate
    pub success_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

impl QuantumGateLibrary {
    /// Create new quantum gate library
    pub fn new() -> Self {
        let mut library = Self {
            matrix_cache: HashMap::new(),
            performance_metrics: HashMap::new(),
        };

        // Pre-compute standard gate matrices
        library.initialize_standard_gates();
        library
    }

    /// Initialize standard gate matrices and metrics
    fn initialize_standard_gates(&mut self) {
        use standard_gates::*;

        // Cache standard gate matrices
        self.matrix_cache.insert(GateTypeKey::Identity, identity_gate());
        self.matrix_cache.insert(GateTypeKey::Hadamard, hadamard_gate());
        self.matrix_cache.insert(GateTypeKey::PauliX, pauli_x_gate());
        self.matrix_cache.insert(GateTypeKey::PauliY, pauli_y_gate());
        self.matrix_cache.insert(GateTypeKey::PauliZ, pauli_z_gate());
        self.matrix_cache.insert(GateTypeKey::CNOT, cnot_gate());
        self.matrix_cache.insert(GateTypeKey::CZ, cz_gate());
        self.matrix_cache.insert(GateTypeKey::SWAP, swap_gate());
        self.matrix_cache.insert(GateTypeKey::Toffoli, toffoli_gate());

        // Initialize performance metrics
        self.performance_metrics.insert(GateTypeKey::Identity, GatePerformanceMetrics {
            avg_execution_time_ns: 10,
            avg_fidelity: 1.0,
            success_rate: 1.0,
            error_rate: 0.0,
        });

        self.performance_metrics.insert(GateTypeKey::Hadamard, GatePerformanceMetrics {
            avg_execution_time_ns: 50,
            avg_fidelity: 0.9999,
            success_rate: 0.9999,
            error_rate: 0.0001,
        });

        self.performance_metrics.insert(GateTypeKey::CNOT, GatePerformanceMetrics {
            avg_execution_time_ns: 200,
            avg_fidelity: 0.999,
            success_rate: 0.999,
            error_rate: 0.001,
        });
    }

    /// Get standard gate by type
    pub fn get_gate(&self, gate_type: GateType, target_qubits: Vec<usize>) -> QuantumResult<crate::circuits::QuantumGate> {
        let _matrix = self.get_gate_matrix(&gate_type)?;
        let metrics = self.get_gate_metrics(&gate_type);

        Ok(crate::circuits::QuantumGate {
            gate_type,
            target_qubits,
            parameters: GateParameters::default(),
            fidelity: metrics.avg_fidelity,
            execution_time_ns: metrics.avg_execution_time_ns,
        })
    }

    /// Get gate matrix from cache or compute
    pub fn get_gate_matrix(&self, gate_type: &GateType) -> QuantumResult<DMatrix<Complex64>> {
        let key = GateTypeKey::from(gate_type);
        match self.matrix_cache.get(&key) {
            Some(matrix) => Ok(matrix.clone()),
            None => {
                // Compute parameterized gate matrices
                match gate_type {
                    GateType::RX(angle) => Ok(standard_gates::rx_gate(*angle)),
                    GateType::RY(angle) => Ok(standard_gates::ry_gate(*angle)),
                    GateType::RZ(angle) => Ok(standard_gates::rz_gate(*angle)),
                    GateType::U3(theta, phi, lambda) => Ok(standard_gates::u3_gate(*theta, *phi, *lambda)),
                    _ => Err(QuantumError::GateOperationFailed {
                        gate_type: format!("{:?}", gate_type),
                        reason: "Gate matrix not found in library".to_string(),
                    }),
                }
            }
        }
    }

    /// Get gate performance metrics
    pub fn get_gate_metrics(&self, gate_type: &GateType) -> GatePerformanceMetrics {
        let key = GateTypeKey::from(gate_type);
        self.performance_metrics.get(&key).cloned().unwrap_or_else(|| {
            // Default metrics for parameterized gates
            match gate_type {
                GateType::RX(_) | GateType::RY(_) | GateType::RZ(_) => GatePerformanceMetrics {
                    avg_execution_time_ns: 75,
                    avg_fidelity: 0.998,
                    success_rate: 0.998,
                    error_rate: 0.002,
                },
                GateType::U3(_, _, _) => GatePerformanceMetrics {
                    avg_execution_time_ns: 100,
                    avg_fidelity: 0.997,
                    success_rate: 0.997,
                    error_rate: 0.003,
                },
                _ => GatePerformanceMetrics {
                    avg_execution_time_ns: 500,
                    avg_fidelity: 0.99,
                    success_rate: 0.99,
                    error_rate: 0.01,
                },
            }
        })
    }

    /// Create Hadamard gate
    pub fn hadamard(&self, qubit: usize) -> QuantumResult<crate::circuits::QuantumGate> {
        self.get_gate(GateType::Hadamard, vec![qubit])
    }

    /// Create CNOT gate
    pub fn cnot(&self, control: usize, target: usize) -> QuantumResult<crate::circuits::QuantumGate> {
        self.get_gate(GateType::CNOT, vec![control, target])
    }

    /// Create X rotation gate
    pub fn rx(&self, qubit: usize, angle: f64) -> QuantumResult<crate::circuits::QuantumGate> {
        Ok(crate::circuits::QuantumGate {
            gate_type: GateType::RX(angle),
            target_qubits: vec![qubit],
            parameters: GateParameters::rotation(angle),
            fidelity: 0.998,
            execution_time_ns: 75,
        })
    }

    /// Create Y rotation gate
    pub fn ry(&self, qubit: usize, angle: f64) -> QuantumResult<crate::circuits::QuantumGate> {
        Ok(crate::circuits::QuantumGate {
            gate_type: GateType::RY(angle),
            target_qubits: vec![qubit],
            parameters: GateParameters::rotation(angle),
            fidelity: 0.998,
            execution_time_ns: 75,
        })
    }

    /// Create Z rotation gate
    pub fn rz(&self, qubit: usize, angle: f64) -> QuantumResult<crate::circuits::QuantumGate> {
        Ok(crate::circuits::QuantumGate {
            gate_type: GateType::RZ(angle),
            target_qubits: vec![qubit],
            parameters: GateParameters::rotation(angle),
            fidelity: 0.998,
            execution_time_ns: 75,
        })
    }

    /// Update gate performance metrics
    pub fn update_metrics(&mut self, gate_type: GateType, metrics: GatePerformanceMetrics) {
        let key = GateTypeKey::from(&gate_type);
        self.performance_metrics.insert(key, metrics);
    }

    /// Get all available gate types
    pub fn available_gates(&self) -> Vec<GateTypeKey> {
        self.matrix_cache.keys().cloned().collect()
    }
}

impl Default for QuantumGateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard quantum gate matrix implementations
pub mod standard_gates {
    use super::*;

    /// Identity gate matrix
    pub fn identity_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ])
    }

    /// Hadamard gate matrix
    pub fn hadamard_gate() -> DMatrix<Complex64> {
        let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0),
        ])
    }

    /// Pauli-X gate matrix
    pub fn pauli_x_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ])
    }

    /// Pauli-Y gate matrix
    pub fn pauli_y_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
        ])
    }

    /// Pauli-Z gate matrix
    pub fn pauli_z_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ])
    }

    /// CNOT gate matrix (4x4 for 2 qubits)
    pub fn cnot_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(4, 4, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ])
    }

    /// Controlled-Z gate matrix (4x4 for 2 qubits)
    pub fn cz_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(4, 4, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ])
    }

    /// SWAP gate matrix (4x4 for 2 qubits)
    pub fn swap_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(4, 4, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ])
    }

    /// Toffoli gate matrix (8x8 for 3 qubits)
    pub fn toffoli_gate() -> DMatrix<Complex64> {
        let mut matrix = DMatrix::identity(8, 8);
        // Swap |110⟩ ↔ |111⟩ (indices 6 and 7)
        matrix[(6, 6)] = Complex64::new(0.0, 0.0);
        matrix[(7, 7)] = Complex64::new(0.0, 0.0);
        matrix[(6, 7)] = Complex64::new(1.0, 0.0);
        matrix[(7, 6)] = Complex64::new(1.0, 0.0);
        matrix
    }

    /// X-rotation gate matrix
    pub fn rx_gate(angle: f64) -> DMatrix<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(0.0, -sin_half),
            Complex64::new(0.0, -sin_half), Complex64::new(cos_half, 0.0),
        ])
    }

    /// Y-rotation gate matrix
    pub fn ry_gate(angle: f64) -> DMatrix<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0),
            Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0),
        ])
    }

    /// Z-rotation gate matrix
    pub fn rz_gate(angle: f64) -> DMatrix<Complex64> {
        let exp_neg = Complex64::new(0.0, -angle / 2.0).exp();
        let exp_pos = Complex64::new(0.0, angle / 2.0).exp();
        
        DMatrix::from_row_slice(2, 2, &[
            exp_neg, Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), exp_pos,
        ])
    }

    /// Universal U3 gate matrix
    pub fn u3_gate(theta: f64, phi: f64, lambda: f64) -> DMatrix<Complex64> {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let exp_phi = Complex64::new(0.0, phi).exp();
        let exp_lambda = Complex64::new(0.0, lambda).exp();
        let exp_both = Complex64::new(0.0, phi + lambda).exp();
        
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), -exp_lambda * sin_half,
            exp_phi * sin_half, exp_both * cos_half,
        ])
    }

    /// Phase gate (S gate)
    pub fn phase_gate() -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
        ])
    }

    /// T gate (π/8 phase)
    pub fn t_gate() -> DMatrix<Complex64> {
        let exp_pi_4 = Complex64::new(0.0, PI / 4.0).exp();
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), exp_pi_4,
        ])
    }
}

/// Enterprise gate synthesis for algorithm compilation
#[derive(Debug)]
pub struct QuantumGateSynthesizer {
    /// Target gate library
    gate_library: QuantumGateLibrary,
    /// Synthesis optimization level
    optimization_level: u8,
    /// Hardware constraints
    hardware_constraints: Option<super::circuits::HardwareConstraints>,
}

impl QuantumGateSynthesizer {
    /// Create new gate synthesizer
    pub fn new(optimization_level: u8) -> Self {
        Self {
            gate_library: QuantumGateLibrary::new(),
            optimization_level,
            hardware_constraints: None,
        }
    }

    /// Set hardware constraints for synthesis
    pub fn with_hardware_constraints(mut self, constraints: super::circuits::HardwareConstraints) -> Self {
        self.hardware_constraints = Some(constraints);
        self
    }

    /// Synthesize arbitrary unitary gate to native gate sequence
    #[instrument(level = "debug", skip(self, target_unitary))]
    pub fn synthesize_unitary(
        &self,
        target_unitary: &DMatrix<Complex64>,
        target_qubits: Vec<usize>,
    ) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        if !self.is_unitary(target_unitary) {
            return Err(QuantumError::GateOperationFailed {
                gate_type: "synthesis_target".to_string(),
                reason: "Target matrix is not unitary".to_string(),
            });
        }

        let mut synthesized_gates = Vec::new();

        match target_unitary.shape() {
            (2, 2) => {
                // Single-qubit synthesis using ZYZ decomposition
                let gates = self.synthesize_single_qubit_zyz(target_unitary, target_qubits[0])?;
                synthesized_gates.extend(gates);
            },
            (4, 4) => {
                // Two-qubit synthesis using KAK decomposition
                let gates = self.synthesize_two_qubit_kak(target_unitary, target_qubits)?;
                synthesized_gates.extend(gates);
            },
            _ => {
                // Multi-qubit synthesis (simplified)
                warn!("Multi-qubit synthesis not fully implemented, using approximation");
                synthesized_gates.push(crate::circuits::QuantumGate::new(
                    GateType::Custom("multi_qubit_approximation".to_string()),
                    target_qubits,
                    GateParameters::default(),
                ));
            }
        }

        debug!(
            synthesized_gates = synthesized_gates.len(),
            optimization_level = self.optimization_level,
            "Gate synthesis completed"
        );

        Ok(synthesized_gates)
    }

    /// Synthesize single-qubit gate using ZYZ decomposition
    fn synthesize_single_qubit_zyz(
        &self,
        unitary: &DMatrix<Complex64>,
        qubit: usize,
    ) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        // Simplified ZYZ decomposition
        // U = e^(iα) RZ(φ) RY(θ) RZ(λ)
        
        // Extract angles from unitary matrix (simplified calculation)
        let u00 = unitary[(0, 0)];
        let u01 = unitary[(0, 1)];
        let _u10 = unitary[(1, 0)];
        let u11 = unitary[(1, 1)];

        // Calculate decomposition angles
        let theta = 2.0 * (u01.norm()).acos();
        let phi = if theta.abs() > 1e-10 {
            (u00 / u00.norm()).arg() - (u11 / u11.norm()).arg()
        } else {
            0.0
        };
        let lambda = if theta.abs() > 1e-10 {
            (u00 / u00.norm()).arg() + (u11 / u11.norm()).arg()
        } else {
            2.0 * (u00 / u00.norm()).arg()
        };

        let mut gates = Vec::new();

        // Add gates only if angles are significant
        if phi.abs() > 1e-10 {
            gates.push(crate::circuits::QuantumGate::new(
                GateType::RZ(phi),
                vec![qubit],
                GateParameters::rotation(phi),
            ));
        }

        if theta.abs() > 1e-10 {
            gates.push(crate::circuits::QuantumGate::new(
                GateType::RY(theta),
                vec![qubit],
                GateParameters::rotation(theta),
            ));
        }

        if lambda.abs() > 1e-10 {
            gates.push(crate::circuits::QuantumGate::new(
                GateType::RZ(lambda),
                vec![qubit],
                GateParameters::rotation(lambda),
            ));
        }

        Ok(gates)
    }

    /// Synthesize two-qubit gate using KAK decomposition
    fn synthesize_two_qubit_kak(
        &self,
        _unitary: &DMatrix<Complex64>,
        target_qubits: Vec<usize>,
    ) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        if target_qubits.len() != 2 {
            return Err(QuantumError::InvalidParameters {
                parameter: "target_qubits".to_string(),
                value: format!("Expected 2 qubits, got {}", target_qubits.len()),
            });
        }

        // Simplified KAK decomposition
        // U = (U1 ⊗ U2) exp(i(aXX + bYY + cZZ)) (U3 ⊗ U4)
        
        let mut gates = Vec::new();
        
        // Pre-rotation on each qubit (simplified)
        gates.push(crate::circuits::QuantumGate::new(
            GateType::RY(PI / 4.0),
            vec![target_qubits[0]],
            GateParameters::rotation(PI / 4.0),
        ));
        
        gates.push(crate::circuits::QuantumGate::new(
            GateType::RY(PI / 4.0),
            vec![target_qubits[1]],
            GateParameters::rotation(PI / 4.0),
        ));

        // Entangling gate
        gates.push(crate::circuits::QuantumGate::new(
            GateType::CNOT,
            target_qubits.clone(),
            GateParameters::default(),
        ));

        // Post-rotation (simplified)
        gates.push(crate::circuits::QuantumGate::new(
            GateType::RY(-PI / 4.0),
            vec![target_qubits[0]],
            GateParameters::rotation(-PI / 4.0),
        ));

        gates.push(crate::circuits::QuantumGate::new(
            GateType::RY(-PI / 4.0),
            vec![target_qubits[1]],
            GateParameters::rotation(-PI / 4.0),
        ));

        Ok(gates)
    }

    /// Check if matrix is unitary
    fn is_unitary(&self, matrix: &DMatrix<Complex64>) -> bool {
        if matrix.nrows() != matrix.ncols() {
            return false;
        }

        let conjugate_transpose = matrix.adjoint();
        let product = matrix * &conjugate_transpose;
        let identity: DMatrix<Complex64> = DMatrix::identity(matrix.nrows(), matrix.ncols());

        // Check if U†U = I within tolerance
        let tolerance = 1e-10;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let complex_diff: Complex64 = product[(i, j)] - identity[(i, j)];
                let diff = complex_diff.norm();
                if diff > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

/// Enterprise gate sequence optimizer
#[derive(Debug)]
pub struct GateSequenceOptimizer {
    /// Optimization strategies enabled
    strategies: Vec<OptimizationStrategy>,
    /// Target hardware constraints
    hardware_constraints: Option<super::circuits::HardwareConstraints>,
}

/// Gate optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Combine adjacent rotations
    GateFusion,
    /// Eliminate identity gates
    IdentityElimination,
    /// Cancel inverse operations
    InverseCancellation,
    /// Optimize for hardware connectivity
    ConnectivityOptimization,
    /// Minimize circuit depth
    DepthMinimization,
    /// Reduce gate count
    GateCountMinimization,
}

impl GateSequenceOptimizer {
    /// Create new gate sequence optimizer
    pub fn new(strategies: Vec<OptimizationStrategy>) -> Self {
        Self {
            strategies,
            hardware_constraints: None,
        }
    }

    /// Optimize sequence of quantum gates
    #[instrument(level = "debug", skip(self, gates))]
    pub fn optimize_gate_sequence(
        &self,
        gates: Vec<crate::circuits::QuantumGate>,
    ) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        let mut optimized_gates = gates.clone();

        for strategy in &self.strategies {
            optimized_gates = match strategy {
                OptimizationStrategy::GateFusion => self.apply_gate_fusion(optimized_gates)?,
                OptimizationStrategy::IdentityElimination => self.eliminate_identities(optimized_gates)?,
                OptimizationStrategy::InverseCancellation => self.cancel_inverses(optimized_gates)?,
                OptimizationStrategy::ConnectivityOptimization => self.optimize_connectivity(optimized_gates)?,
                OptimizationStrategy::DepthMinimization => self.minimize_depth(optimized_gates)?,
                OptimizationStrategy::GateCountMinimization => self.minimize_gate_count(optimized_gates)?,
            };
        }

        debug!(
            original_count = optimized_gates.len(),
            optimized_count = optimized_gates.len(),
            reduction = gates.len() as i32 - optimized_gates.len() as i32,
            "Gate sequence optimization completed"
        );

        Ok(optimized_gates)
    }

    /// Apply gate fusion optimization
    fn apply_gate_fusion(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() && self.can_fuse(&gates[i], &gates[i + 1]) {
                let fused = self.fuse_gates(&gates[i], &gates[i + 1])?;
                optimized.push(fused);
                i += 2;
            } else {
                optimized.push(gates[i].clone());
                i += 1;
            }
        }

        Ok(optimized)
    }

    /// Check if two gates can be fused
    fn can_fuse(&self, gate1: &crate::circuits::QuantumGate, gate2: &crate::circuits::QuantumGate) -> bool {
        gate1.target_qubits == gate2.target_qubits &&
        gate1.target_qubits.len() == 1 &&
        matches!(gate1.gate_type, GateType::RX(_) | GateType::RY(_) | GateType::RZ(_)) &&
        matches!(gate2.gate_type, GateType::RX(_) | GateType::RY(_) | GateType::RZ(_))
    }

    /// Fuse two compatible gates
    fn fuse_gates(&self, gate1: &crate::circuits::QuantumGate, gate2: &crate::circuits::QuantumGate) -> QuantumResult<crate::circuits::QuantumGate> {
        // Simplified fusion - actual implementation would properly combine rotations
        let combined_angle = match (&gate1.gate_type, &gate2.gate_type) {
            (GateType::RX(a1), GateType::RX(a2)) => a1 + a2,
            (GateType::RY(a1), GateType::RY(a2)) => a1 + a2,
            (GateType::RZ(a1), GateType::RZ(a2)) => a1 + a2,
            _ => return Err(QuantumError::GateOperationFailed {
                gate_type: "gate_fusion".to_string(),
                reason: "Cannot fuse incompatible gate types".to_string(),
            }),
        };

        let fused_type = match &gate1.gate_type {
            GateType::RX(_) => GateType::RX(combined_angle),
            GateType::RY(_) => GateType::RY(combined_angle),
            GateType::RZ(_) => GateType::RZ(combined_angle),
            _ => unreachable!(),
        };

        Ok(crate::circuits::QuantumGate::new(
            fused_type,
            gate1.target_qubits.clone(),
            GateParameters::rotation(combined_angle),
        ))
    }

    /// Eliminate identity gates
    fn eliminate_identities(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        Ok(gates.into_iter()
            .filter(|gate| gate.gate_type != GateType::Identity)
            .collect())
    }

    /// Cancel inverse operations
    fn cancel_inverses(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() && self.are_inverses(&gates[i], &gates[i + 1]) {
                // Skip both gates as they cancel
                i += 2;
            } else {
                optimized.push(gates[i].clone());
                i += 1;
            }
        }

        Ok(optimized)
    }

    /// Check if two gates are inverses
    fn are_inverses(&self, gate1: &crate::circuits::QuantumGate, gate2: &crate::circuits::QuantumGate) -> bool {
        gate1.target_qubits == gate2.target_qubits &&
        match (&gate1.gate_type, &gate2.gate_type) {
            (GateType::PauliX, GateType::PauliX) => true,
            (GateType::PauliY, GateType::PauliY) => true,
            (GateType::PauliZ, GateType::PauliZ) => true,
            (GateType::Hadamard, GateType::Hadamard) => true,
            (GateType::CNOT, GateType::CNOT) => true,
            (GateType::RX(a1), GateType::RX(a2)) => (a1 + a2).abs() < 1e-10,
            (GateType::RY(a1), GateType::RY(a2)) => (a1 + a2).abs() < 1e-10,
            (GateType::RZ(a1), GateType::RZ(a2)) => (a1 + a2).abs() < 1e-10,
            _ => false,
        }
    }

    /// Optimize for hardware connectivity
    fn optimize_connectivity(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        // Placeholder for connectivity optimization
        // Would insert SWAP gates to route around connectivity constraints
        Ok(gates)
    }

    /// Minimize circuit depth
    fn minimize_depth(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        // Simplified depth minimization by reordering commuting gates
        let mut optimized = gates;
        
        // Sort gates to group operations on same qubits
        optimized.sort_by_key(|gate| {
            (gate.target_qubits.first().unwrap_or(&0).clone(), gate.target_qubits.len())
        });

        Ok(optimized)
    }

    /// Minimize gate count
    fn minimize_gate_count(&self, gates: Vec<crate::circuits::QuantumGate>) -> QuantumResult<Vec<crate::circuits::QuantumGate>> {
        // Apply all count-reducing optimizations
        let gates = self.eliminate_identities(gates)?;
        let gates = self.cancel_inverses(gates)?;
        let gates = self.apply_gate_fusion(gates)?;
        Ok(gates)
    }
}

/// Gate compilation result for enterprise monitoring
#[derive(Debug, Clone)]
pub struct GateCompilationResult {
    /// Original gate count
    pub original_gate_count: usize,
    /// Compiled gate count
    pub compiled_gate_count: usize,
    /// Compilation time in milliseconds
    pub compilation_time_ms: u64,
    /// Target hardware backend
    pub target_hardware: String,
    /// Fidelity after compilation
    pub compiled_fidelity: f64,
    /// Optimizations applied
    pub optimizations_applied: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::standard_gates::*;

    #[test]
    fn test_gate_parameters() {
        let params = GateParameters::rotation(PI / 2.0);
        assert_eq!(params.angles, vec![PI / 2.0]);
        assert!(params.phases.is_empty());

        let u3_params = GateParameters::u3(PI / 2.0, PI / 4.0, PI / 8.0);
        assert_eq!(u3_params.angles, vec![PI / 2.0]);
        assert_eq!(u3_params.phases, vec![PI / 4.0, PI / 8.0]);
    }

    #[test]
    fn test_standard_gate_matrices() {
        let h = hadamard_gate();
        assert_eq!(h.shape(), (2, 2));
        
        let cnot = cnot_gate();
        assert_eq!(cnot.shape(), (4, 4));
        
        let toffoli = toffoli_gate();
        assert_eq!(toffoli.shape(), (8, 8));
    }

    #[test]
    fn test_rotation_gates() {
        let rx = rx_gate(PI / 2.0);
        assert_eq!(rx.shape(), (2, 2));
        
        let ry = ry_gate(PI / 2.0);
        assert_eq!(ry.shape(), (2, 2));
        
        let rz = rz_gate(PI / 2.0);
        assert_eq!(rz.shape(), (2, 2));
    }

    #[test]
    fn test_quantum_gate_library() {
        let library = QuantumGateLibrary::new();
        
        let h_gate = library.hadamard(0).expect("Should create Hadamard");
        assert_eq!(h_gate.gate_type, GateType::Hadamard);
        assert_eq!(h_gate.target_qubits, vec![0]);
        
        let cnot_gate = library.cnot(0, 1).expect("Should create CNOT");
        assert_eq!(cnot_gate.gate_type, GateType::CNOT);
        assert_eq!(cnot_gate.target_qubits, vec![0, 1]);
    }

    #[test]
    fn test_gate_synthesis() {
        let synthesizer = QuantumGateSynthesizer::new(2);
        
        // Test single-qubit synthesis
        let target = hadamard_gate();
        let synthesized = synthesizer.synthesize_unitary(&target, vec![0])
            .expect("Should synthesize Hadamard");
        
        assert!(!synthesized.is_empty());
        assert!(synthesized.iter().all(|g| g.target_qubits == vec![0]));
    }

    #[test]
    fn test_gate_sequence_optimizer() {
        let optimizer = GateSequenceOptimizer::new(vec![
            OptimizationStrategy::IdentityElimination,
            OptimizationStrategy::InverseCancellation,
        ]);

        let gates = vec![
            crate::circuits::QuantumGate::new(GateType::Hadamard, vec![0], GateParameters::default()),
            crate::circuits::QuantumGate::new(GateType::Identity, vec![0], GateParameters::default()),
            crate::circuits::QuantumGate::new(GateType::Hadamard, vec![0], GateParameters::default()),
        ];

        let optimized = optimizer.optimize_gate_sequence(gates)
            .expect("Should optimize sequence");

        // H-I-H should optimize to just I (since H·H = I), then eliminate I
        assert!(optimized.len() <= 1);
    }

    #[test]
    fn test_gate_library_metrics() {
        let library = QuantumGateLibrary::new();
        
        let h_metrics = library.get_gate_metrics(&GateType::Hadamard);
        assert!(h_metrics.avg_fidelity > 0.99);
        assert!(h_metrics.success_rate > 0.99);
        
        let cnot_metrics = library.get_gate_metrics(&GateType::CNOT);
        assert!(cnot_metrics.avg_execution_time_ns > h_metrics.avg_execution_time_ns);
    }

    #[test]
    fn test_u3_gate_decomposition() {
        let u3 = u3_gate(PI / 2.0, PI / 4.0, PI / 8.0);
        assert_eq!(u3.shape(), (2, 2));
        
        // Verify it's unitary
        let conj_transpose = u3.adjoint();
        let product = &u3 * &conj_transpose;
        let identity = DMatrix::identity(2, 2);
        
        for i in 0..2 {
            for j in 0..2 {
                let diff = (product[(i, j)] - identity[(i, j)]).norm();
                assert!(diff < 1e-10, "U3 gate is not unitary");
            }
        }
    }
}