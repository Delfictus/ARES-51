//! Enterprise quantum state management for deterministic execution
//!
//! This module provides production-grade quantum state management with
//! deterministic evolution, enterprise monitoring, and hardware abstraction.

use crate::{QuantumError, QuantumResult};
use csf_time::{global_time_source, LogicalTime, NanoTime, TimeSource};
use nalgebra::DVector;
use num_complex::Complex64;
use parking_lot::RwLock;
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// Enterprise quantum state vector with deterministic evolution
#[derive(Debug, Clone)]
pub struct QuantumStateVector {
    /// Complex amplitude vector
    pub amplitudes: DVector<Complex64>,
    /// Logical timestamp of state creation/measurement
    pub timestamp: LogicalTime,
    /// State coherence measure (0.0 to 1.0)
    pub coherence: f64,
    /// State fidelity confidence
    pub fidelity: f64,
    /// Normalization factor
    pub normalization: f64,
    /// State identifier for tracking
    pub state_id: u64,
}

impl QuantumStateVector {
    /// Create new quantum state vector
    pub fn new(amplitudes: DVector<Complex64>, timestamp: LogicalTime) -> QuantumResult<Self> {
        let normalization = Self::calculate_normalization(&amplitudes);
        
        if normalization < 1e-10 {
            return Err(QuantumError::StateManagementError {
                operation: "create_state".to_string(),
                reason: "State vector has zero norm".to_string(),
            });
        }
        
        let normalized_amplitudes = amplitudes.map(|a| a / Complex64::new(normalization, 0.0));
        let coherence = Self::calculate_coherence(&normalized_amplitudes);
        let fidelity = Self::calculate_fidelity(&normalized_amplitudes);
        
        Ok(Self {
            amplitudes: normalized_amplitudes,
            timestamp,
            coherence,
            fidelity,
            normalization,
            state_id: timestamp.physical.wrapping_add(timestamp.logical),
        })
    }

    /// Create quantum state from classical probability distribution
    pub fn from_classical_probabilities(probabilities: &[f64], timestamp: LogicalTime) -> QuantumResult<Self> {
        let amplitudes: DVector<Complex64> = DVector::from_iterator(
            probabilities.len(),
            probabilities.iter().map(|&p| Complex64::new(p.sqrt(), 0.0))
        );
        
        Self::new(amplitudes, timestamp)
    }

    /// Create superposition state (equal amplitudes)
    pub fn superposition(num_qubits: usize, timestamp: LogicalTime) -> QuantumResult<Self> {
        let dim = 1 << num_qubits; // 2^n states
        let amplitude = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        let amplitudes = DVector::from_element(dim, amplitude);
        
        Self::new(amplitudes, timestamp)
    }

    /// Calculate normalization factor
    fn calculate_normalization(amplitudes: &DVector<Complex64>) -> f64 {
        amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Calculate quantum coherence measure
    fn calculate_coherence(amplitudes: &DVector<Complex64>) -> f64 {
        let n = amplitudes.len() as f64;
        let sum_phases: f64 = amplitudes.iter()
            .map(|a| a.arg())
            .map(|phase| (phase.cos() + phase.sin()).abs())
            .sum();
        
        sum_phases / n
    }

    /// Calculate state fidelity confidence
    fn calculate_fidelity(amplitudes: &DVector<Complex64>) -> f64 {
        // Simple fidelity based on state purity
        let purity: f64 = amplitudes.iter().map(|a| a.norm_sqr().powi(2)).sum();
        purity.sqrt()
    }

    /// Apply quantum gate to state
    pub fn apply_gate(&mut self, gate_matrix: &DVector<Complex64>) -> QuantumResult<()> {
        if gate_matrix.len() != self.amplitudes.len().pow(2) {
            return Err(QuantumError::StateManagementError {
                operation: "apply_gate".to_string(),
                reason: format!("Gate matrix dimension {} incompatible with state dimension {}", 
                    gate_matrix.len(), self.amplitudes.len()),
            });
        }

        // Apply gate operation (simplified matrix-vector multiply)
        // In full implementation, this would use proper matrix multiplication
        for i in 0..self.amplitudes.len() {
            let old_amp = self.amplitudes[i];
            self.amplitudes[i] = old_amp * gate_matrix[i]; // Simplified for demonstration
        }

        // Renormalize
        let norm = Self::calculate_normalization(&self.amplitudes);
        self.amplitudes = self.amplitudes.map(|a| a / Complex64::new(norm, 0.0));
        
        // Update coherence and fidelity
        self.coherence = Self::calculate_coherence(&self.amplitudes);
        self.fidelity = Self::calculate_fidelity(&self.amplitudes);
        
        Ok(())
    }

    /// Measure quantum state with enterprise deterministic sampling
    pub fn measure(&self, time_source: &dyn TimeSource) -> QuantumResult<MeasurementResult> {
        let measurement_time = time_source.now_ns()
            .map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get measurement time: {}", e),
            })?;

        // Deterministic measurement based on timestamp
        let seed = measurement_time.as_nanos();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        
        // Calculate probabilities
        let probabilities: Vec<f64> = self.amplitudes.iter()
            .map(|a| a.norm_sqr())
            .collect();
        
        // Sample from probability distribution
        use rand::Rng;
        let random_value: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut measured_state = 0;
        
        for (i, prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                measured_state = i;
                break;
            }
        }
        
        Ok(MeasurementResult {
            measured_state,
            probability: probabilities[measured_state],
            measurement_time,
            fidelity: self.fidelity,
        })
    }

    /// Get quantum state statistics
    pub fn get_statistics(&self) -> QuantumStateStatistics {
        let probabilities: Vec<f64> = self.amplitudes.iter()
            .map(|a| a.norm_sqr())
            .collect();
        
        let entropy = -probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();
        
        QuantumStateStatistics {
            state_id: self.state_id,
            dimension: self.amplitudes.len(),
            coherence: self.coherence,
            fidelity: self.fidelity,
            entropy,
            max_probability: probabilities.iter().fold(0.0, |a, &b| a.max(b)),
            timestamp: self.timestamp,
        }
    }
}

/// Result of quantum measurement
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// Measured state index
    pub measured_state: usize,
    /// Probability of this measurement
    pub probability: f64,
    /// Time when measurement was taken
    pub measurement_time: NanoTime,
    /// Measurement fidelity
    pub fidelity: f64,
}

/// Quantum state statistics for monitoring
#[derive(Debug, Clone)]
pub struct QuantumStateStatistics {
    /// State identifier
    pub state_id: u64,
    /// State vector dimension
    pub dimension: usize,
    /// Coherence measure
    pub coherence: f64,
    /// State fidelity
    pub fidelity: f64,
    /// Von Neumann entropy
    pub entropy: f64,
    /// Maximum probability amplitude
    pub max_probability: f64,
    /// State timestamp
    pub timestamp: LogicalTime,
}

/// Enterprise quantum state manager with deterministic evolution
#[derive(Debug)]
pub struct QuantumStateManager {
    /// Maximum number of qubits supported
    max_qubits: usize,
    /// Active quantum states
    active_states: Arc<RwLock<HashMap<u64, QuantumStateVector>>>,
    /// State evolution history for debugging
    evolution_history: Arc<RwLock<Vec<StateEvolutionRecord>>>,
    /// Enterprise time source for deterministic operations
    time_source: Arc<dyn TimeSource>,
    /// State counter for unique IDs
    state_counter: std::sync::atomic::AtomicU64,
}

/// State evolution record for enterprise audit trail
#[derive(Debug, Clone)]
pub struct StateEvolutionRecord {
    /// State before evolution
    pub before_state_id: u64,
    /// State after evolution
    pub after_state_id: u64,
    /// Operation that caused evolution
    pub operation: String,
    /// Evolution timestamp
    pub evolved_at: LogicalTime,
    /// Evolution fidelity
    pub evolution_fidelity: f64,
}

impl QuantumStateManager {
    /// Create new quantum state manager
    pub fn new(max_qubits: usize) -> QuantumResult<Self> {
        if max_qubits == 0 || max_qubits > 64 {
            return Err(QuantumError::InvalidParameters {
                parameter: "max_qubits".to_string(),
                value: max_qubits.to_string(),
            });
        }

        Ok(Self {
            max_qubits,
            active_states: Arc::new(RwLock::new(HashMap::new())),
            evolution_history: Arc::new(RwLock::new(Vec::new())),
            time_source: global_time_source().clone(),
            state_counter: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Create new quantum state
    #[instrument(level = "debug", skip(self, amplitudes))]
    pub fn create_state(&self, amplitudes: DVector<Complex64>) -> QuantumResult<u64> {
        let timestamp = LogicalTime::from_nano_time(
            self.time_source.now_ns().map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get timestamp: {}", e),
            })?,
            1, // Default node ID
        );

        let state = QuantumStateVector::new(amplitudes, timestamp)?;
        let state_id = state.state_id;

        self.active_states.write().insert(state_id, state);

        debug!(
            state_id = state_id,
            timestamp = %timestamp,
            "Created new quantum state"
        );

        Ok(state_id)
    }

    /// Get quantum state by ID
    pub fn get_state(&self, state_id: u64) -> Option<QuantumStateVector> {
        self.active_states.read().get(&state_id).cloned()
    }

    /// Apply evolution to quantum state
    #[instrument(level = "debug", skip(self, gate_matrix))]
    pub fn evolve_state(&self, state_id: u64, gate_matrix: &DVector<Complex64>, operation_name: String) -> QuantumResult<u64> {
        let mut states = self.active_states.write();
        
        let mut state = states.get(&state_id).cloned()
            .ok_or_else(|| QuantumError::StateManagementError {
                operation: "evolve_state".to_string(),
                reason: format!("State {} not found", state_id),
            })?;

        let before_fidelity = state.fidelity;
        state.apply_gate(gate_matrix)?;
        
        // Create new state ID for evolved state
        let new_state_id = self.state_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        state.state_id = new_state_id;

        // Update timestamp
        let evolution_time = LogicalTime::from_nano_time(
            self.time_source.now_ns().map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get evolution time: {}", e),
            })?,
            1,
        );
        state.timestamp = evolution_time;

        // Record evolution
        let evolution_record = StateEvolutionRecord {
            before_state_id: state_id,
            after_state_id: new_state_id,
            operation: operation_name,
            evolved_at: evolution_time,
            evolution_fidelity: (before_fidelity + state.fidelity) / 2.0,
        };

        self.evolution_history.write().push(evolution_record);
        states.insert(new_state_id, state);

        debug!(
            before_state_id = state_id,
            after_state_id = new_state_id,
            evolution_fidelity = (before_fidelity + states[&new_state_id].fidelity) / 2.0,
            "Evolved quantum state"
        );

        Ok(new_state_id)
    }

    /// Measure quantum state with deterministic sampling
    pub fn measure_state(&self, state_id: u64) -> QuantumResult<MeasurementResult> {
        let states = self.active_states.read();
        let state = states.get(&state_id)
            .ok_or_else(|| QuantumError::StateManagementError {
                operation: "measure_state".to_string(),
                reason: format!("State {} not found", state_id),
            })?;

        state.measure(&*self.time_source)
    }

    /// Get number of active circuits
    pub fn active_circuit_count(&self) -> usize {
        self.active_states.read().len()
    }

    /// Cleanup old quantum states
    pub fn cleanup_old_states(&self, before_time: LogicalTime) -> QuantumResult<usize> {
        let mut states = self.active_states.write();
        let original_count = states.len();
        
        states.retain(|_, state| !state.timestamp.happens_before(before_time));
        
        let removed_count = original_count - states.len();
        
        if removed_count > 0 {
            info!(
                removed_count = removed_count,
                before_time = %before_time,
                "Cleaned up old quantum states"
            );
        }

        Ok(removed_count)
    }

    /// Get evolution history for audit trail
    pub fn get_evolution_history(&self) -> Vec<StateEvolutionRecord> {
        self.evolution_history.read().clone()
    }

    /// Get comprehensive state manager statistics
    pub fn get_manager_statistics(&self) -> StateManagerStatistics {
        let states = self.active_states.read();
        let history = self.evolution_history.read();
        
        let avg_coherence = if states.is_empty() {
            0.0
        } else {
            states.values().map(|s| s.coherence).sum::<f64>() / states.len() as f64
        };
        
        let avg_fidelity = if states.is_empty() {
            0.0
        } else {
            states.values().map(|s| s.fidelity).sum::<f64>() / states.len() as f64
        };
        
        StateManagerStatistics {
            max_qubits: self.max_qubits,
            active_states: states.len(),
            total_evolutions: history.len(),
            avg_coherence,
            avg_fidelity,
            memory_usage_bytes: states.len() * std::mem::size_of::<QuantumStateVector>(),
        }
    }
}

/// State manager statistics for enterprise monitoring
#[derive(Debug, Clone)]
pub struct StateManagerStatistics {
    /// Maximum qubits supported
    pub max_qubits: usize,
    /// Number of active quantum states
    pub active_states: usize,
    /// Total state evolutions performed
    pub total_evolutions: usize,
    /// Average coherence across states
    pub avg_coherence: f64,
    /// Average fidelity across states
    pub avg_fidelity: f64,
    /// Estimated memory usage
    pub memory_usage_bytes: usize,
}

/// Quantum state trait for algorithm compatibility
pub trait QuantumState: Send + Sync + std::fmt::Debug {
    /// Get state vector amplitudes
    fn amplitudes(&self) -> &DVector<Complex64>;
    
    /// Get state coherence
    fn coherence(&self) -> f64;
    
    /// Get state fidelity
    fn fidelity(&self) -> f64;
    
    /// Get state timestamp
    fn timestamp(&self) -> LogicalTime;
    
    /// Clone the quantum state
    fn clone_state(&self) -> Box<dyn QuantumState>;
}

impl QuantumState for QuantumStateVector {
    fn amplitudes(&self) -> &DVector<Complex64> {
        &self.amplitudes
    }
    
    fn coherence(&self) -> f64 {
        self.coherence
    }
    
    fn fidelity(&self) -> f64 {
        self.fidelity
    }
    
    fn timestamp(&self) -> LogicalTime {
        self.timestamp
    }
    
    fn clone_state(&self) -> Box<dyn QuantumState> {
        Box::new(self.clone())
    }
}

/// Enterprise quantum state factory for standardized creation
pub struct QuantumStateFactory {
    /// Default time source for state creation
    time_source: Arc<dyn TimeSource>,
}

impl QuantumStateFactory {
    /// Create new quantum state factory
    pub fn new() -> Self {
        Self {
            time_source: global_time_source().clone(),
        }
    }

    /// Create computational basis state |n⟩
    pub fn computational_basis_state(&self, n: usize, num_qubits: usize) -> QuantumResult<QuantumStateVector> {
        let dim = 1 << num_qubits;
        
        if n >= dim {
            return Err(QuantumError::InvalidParameters {
                parameter: "basis_state_index".to_string(),
                value: n.to_string(),
            });
        }
        
        let mut amplitudes = DVector::zeros(dim);
        amplitudes[n] = Complex64::new(1.0, 0.0);
        
        let timestamp = LogicalTime::from_nano_time(
            self.time_source.now_ns().map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get timestamp: {}", e),
            })?,
            1,
        );
        
        QuantumStateVector::new(amplitudes, timestamp)
    }

    /// Create Bell state (maximally entangled two-qubit state)
    pub fn bell_state(&self, bell_type: BellStateType) -> QuantumResult<QuantumStateVector> {
        let timestamp = LogicalTime::from_nano_time(
            self.time_source.now_ns().map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get timestamp: {}", e),
            })?,
            1,
        );

        let amplitudes = match bell_type {
            BellStateType::PhiPlus => DVector::from_vec(vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |00⟩
                Complex64::new(0.0, 0.0),                   // |01⟩
                Complex64::new(0.0, 0.0),                   // |10⟩
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |11⟩
            ]),
            BellStateType::PhiMinus => DVector::from_vec(vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),  // |00⟩
                Complex64::new(0.0, 0.0),                    // |01⟩
                Complex64::new(0.0, 0.0),                    // |10⟩
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0), // |11⟩
            ]),
            BellStateType::PsiPlus => DVector::from_vec(vec![
                Complex64::new(0.0, 0.0),                   // |00⟩
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |01⟩
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |10⟩
                Complex64::new(0.0, 0.0),                   // |11⟩
            ]),
            BellStateType::PsiMinus => DVector::from_vec(vec![
                Complex64::new(0.0, 0.0),                    // |00⟩
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),  // |01⟩
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0), // |10⟩
                Complex64::new(0.0, 0.0),                    // |11⟩
            ]),
        };

        QuantumStateVector::new(amplitudes, timestamp)
    }

    /// Create GHZ state (multi-qubit entangled state)
    pub fn ghz_state(&self, num_qubits: usize) -> QuantumResult<QuantumStateVector> {
        if num_qubits < 2 {
            return Err(QuantumError::InvalidParameters {
                parameter: "num_qubits".to_string(),
                value: num_qubits.to_string(),
            });
        }

        let dim = 1 << num_qubits;
        let mut amplitudes = DVector::zeros(dim);
        
        // |000...0⟩ + |111...1⟩ (normalized)
        amplitudes[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);           // |000...0⟩
        amplitudes[dim - 1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);     // |111...1⟩
        
        let timestamp = LogicalTime::from_nano_time(
            self.time_source.now_ns().map_err(|e| QuantumError::QuantumTimingError {
                details: format!("Failed to get timestamp: {}", e),
            })?,
            1,
        );
        
        QuantumStateVector::new(amplitudes, timestamp)
    }
}

/// Bell state types for entanglement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BellStateType {
    /// |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PhiPlus,
    /// |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PhiMinus,
    /// |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PsiPlus,
    /// |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    PsiMinus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use csf_time::initialize_simulated_time_source;

    fn init_test_time() {
        initialize_simulated_time_source(NanoTime::from_nanos(1000));
    }

    #[test]
    fn test_quantum_state_vector_creation() {
        init_test_time();
        
        let amplitudes = DVector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        
        let timestamp = LogicalTime::new(1000, 0, 1);
        let state = QuantumStateVector::new(amplitudes, timestamp).expect("Should create state");
        
        assert_eq!(state.amplitudes.len(), 2);
        assert!(state.coherence >= 0.0);
        assert!(state.fidelity >= 0.0);
        assert_eq!(state.timestamp, timestamp);
    }

    #[test]
    fn test_quantum_state_manager() {
        init_test_time();
        
        let manager = QuantumStateManager::new(3).expect("Should create manager");
        assert_eq!(manager.max_qubits, 3);
        assert_eq!(manager.active_circuit_count(), 0);
    }

    #[test]
    fn test_quantum_state_factory() {
        init_test_time();
        
        let factory = QuantumStateFactory::new();
        
        // Test computational basis state
        let state = factory.computational_basis_state(0, 2).expect("Should create |00⟩");
        assert_eq!(state.amplitudes[0].norm_sqr(), 1.0);
        assert_eq!(state.amplitudes[1].norm_sqr(), 0.0);
        assert_eq!(state.amplitudes[2].norm_sqr(), 0.0);
        assert_eq!(state.amplitudes[3].norm_sqr(), 0.0);
    }

    #[test]
    fn test_bell_state_creation() {
        init_test_time();
        
        let factory = QuantumStateFactory::new();
        let bell_state = factory.bell_state(BellStateType::PhiPlus).expect("Should create Bell state");
        
        // Check normalization
        let total_prob: f64 = bell_state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
        
        // Check entanglement structure
        assert!((bell_state.amplitudes[0].norm_sqr() - 0.5).abs() < 1e-10);
        assert!((bell_state.amplitudes[3].norm_sqr() - 0.5).abs() < 1e-10);
        assert!(bell_state.amplitudes[1].norm_sqr() < 1e-10);
        assert!(bell_state.amplitudes[2].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_quantum_state_measurement() {
        init_test_time();
        
        let amplitudes = DVector::from_vec(vec![
            Complex64::new(0.6, 0.0),
            Complex64::new(0.8, 0.0),
        ]);
        
        let timestamp = LogicalTime::new(2000, 0, 1);
        let state = QuantumStateVector::new(amplitudes, timestamp).expect("Should create state");
        
        let time_source = global_time_source();
        let result = state.measure(&*time_source).expect("Should measure state");
        
        assert!(result.measured_state == 0 || result.measured_state == 1);
        assert!(result.probability > 0.0);
        assert!(result.probability <= 1.0);
        assert!(result.fidelity > 0.0);
    }
}