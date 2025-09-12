//! Quantum state consistency for distributed temporal determinism
//!
//! This module implements enterprise-grade quantum state consistency protocols
//! for ensuring deterministic quantum state evolution across distributed nodes.

use crate::{clock::HlcClock, consensus::TemporalConsensusCoordinator, global_hlc, LogicalTime, TimeError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Quantum state vector for distributed consistency
#[derive(Debug, Clone)]
pub struct QuantumStateVector {
    /// Quantum state vector components
    pub state_vector: Vec<f64>,
    /// Energy eigenvalues for this state
    pub energy_levels: Vec<f64>,
    /// Coherence measure (0.0 to 1.0)
    pub coherence: f64,
    /// Logical time when state was measured
    pub measured_at: LogicalTime,
    /// Node that measured this state
    pub measuring_node: u64,
    /// Measurement confidence
    pub confidence: f64,
}

/// Quantum state transition for deterministic evolution
#[derive(Debug, Clone)]
pub struct QuantumTransition {
    /// Transition identifier
    pub transition_id: u64,
    /// Initial quantum state
    pub initial_state: QuantumStateVector,
    /// Final quantum state after transition
    pub final_state: QuantumStateVector,
    /// Transition unitary operator (flattened matrix)
    pub unitary_operator: Vec<f64>,
    /// Transition duration in logical time
    pub duration: u64,
    /// Nodes that must agree on this transition
    pub consensus_nodes: Vec<u64>,
}

/// Result of quantum state verification
#[derive(Debug, Clone)]
pub enum QuantumVerificationResult {
    /// State is consistent across all nodes
    Consistent {
        /// Verified quantum state
        verified_state: QuantumStateVector,
        /// Nodes that verified the state
        verifying_nodes: Vec<u64>,
    },
    /// State is inconsistent between nodes
    Inconsistent {
        /// Conflicting states from different nodes
        conflicting_states: HashMap<u64, QuantumStateVector>,
        /// Maximum deviation in coherence
        max_deviation: f64,
    },
    /// Insufficient data for verification
    InsufficientData {
        /// Available states for comparison
        available_states: Vec<QuantumStateVector>,
        /// Minimum required states for verification
        required_count: usize,
    },
}

/// Enterprise quantum consistency coordinator
#[derive(Debug)]
pub struct QuantumConsistencyCoordinator {
    /// Local node identifier
    node_id: u64,
    /// Current quantum state
    current_state: Arc<RwLock<Option<QuantumStateVector>>>,
    /// Quantum states from peer nodes
    peer_states: Arc<RwLock<HashMap<u64, QuantumStateVector>>>,
    /// Active quantum transitions
    active_transitions: Arc<RwLock<HashMap<u64, QuantumTransition>>>,
    /// Consensus coordinator for quantum decisions
    consensus_coordinator: Arc<TemporalConsensusCoordinator>,
    /// Coherence threshold for consistency checks
    coherence_threshold: f64,
    /// Maximum allowed state deviation
    max_state_deviation: f64,
}

impl QuantumConsistencyCoordinator {
    /// Create new quantum consistency coordinator
    pub fn new(
        node_id: u64,
        consensus_coordinator: Arc<TemporalConsensusCoordinator>,
        coherence_threshold: f64,
        max_state_deviation: f64,
    ) -> Self {
        Self {
            node_id,
            current_state: Arc::new(RwLock::new(None)),
            peer_states: Arc::new(RwLock::new(HashMap::new())),
            active_transitions: Arc::new(RwLock::new(HashMap::new())),
            consensus_coordinator,
            coherence_threshold,
            max_state_deviation,
        }
    }

    /// Initialize quantum state with enterprise deterministic measurement
    pub async fn initialize_quantum_state(&self, initial_state_vector: Vec<f64>) -> Result<QuantumStateVector, TimeError> {
        let hlc = global_hlc()?;
        let measured_at = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };

        let state = QuantumStateVector {
            state_vector: initial_state_vector.clone(),
            energy_levels: self.calculate_energy_eigenvalues(&initial_state_vector),
            coherence: self.calculate_coherence(&initial_state_vector),
            measured_at,
            measuring_node: self.node_id,
            confidence: 1.0,
        };

        // Store local state
        *self.current_state.write() = Some(state.clone());

        info!(
            node_id = self.node_id,
            measured_at = %measured_at,
            coherence = state.coherence,
            state_dimension = state.state_vector.len(),
            "Initialized enterprise quantum state"
        );

        Ok(state)
    }

    /// Update quantum state from peer node measurement
    pub async fn update_peer_quantum_state(&self, peer_node_id: u64, peer_state: QuantumStateVector) -> Result<(), TimeError> {
        self.peer_states.write().insert(peer_node_id, peer_state.clone());

        debug!(
            node_id = self.node_id,
            peer_node_id = peer_node_id,
            peer_coherence = peer_state.coherence,
            measured_at = %peer_state.measured_at,
            "Updated peer quantum state"
        );

        // Check if verification is needed
        self.check_consistency_and_verify().await?;

        Ok(())
    }

    /// Verify quantum state consistency across all nodes
    pub async fn verify_quantum_consistency(&self) -> Result<QuantumVerificationResult, TimeError> {
        let current_state = self.current_state.read().clone();
        let peer_states = self.peer_states.read().clone();

        let Some(ref local_state) = current_state else {
            return Ok(QuantumVerificationResult::InsufficientData {
                available_states: peer_states.values().cloned().collect(),
                required_count: 1,
            });
        };

        if peer_states.is_empty() {
            return Ok(QuantumVerificationResult::Consistent {
                verified_state: local_state.clone(),
                verifying_nodes: vec![self.node_id],
            });
        }

        // Check consistency with peer states
        let mut conflicting_states = HashMap::new();
        let mut verifying_nodes = vec![self.node_id];
        let mut max_deviation: f64 = 0.0;

        for (&peer_node_id, peer_state) in &peer_states {
            let deviation = self.calculate_state_deviation(local_state, peer_state);

            if deviation > self.max_state_deviation {
                conflicting_states.insert(peer_node_id, peer_state.clone());
                max_deviation = max_deviation.max(deviation);
            } else {
                verifying_nodes.push(peer_node_id);
            }
        }

        if conflicting_states.is_empty() {
            Ok(QuantumVerificationResult::Consistent {
                verified_state: local_state.clone(),
                verifying_nodes,
            })
        } else {
            conflicting_states.insert(self.node_id, local_state.clone());

            warn!(
                node_id = self.node_id,
                conflicting_nodes = ?conflicting_states.keys().collect::<Vec<_>>(),
                max_deviation = max_deviation,
                threshold = self.max_state_deviation,
                "Quantum state inconsistency detected"
            );

            Ok(QuantumVerificationResult::Inconsistent {
                conflicting_states,
                max_deviation,
            })
        }
    }

    /// Propose quantum state transition with distributed consensus
    pub async fn propose_quantum_transition(
        &self,
        target_state: Vec<f64>,
        unitary_operator: Vec<f64>,
        consensus_nodes: Vec<u64>,
    ) -> Result<u64, TimeError> {
        let current_state = self.current_state.read().clone()
            .ok_or_else(|| TimeError::SystemTimeError {
                details: "No current quantum state available for transition".to_string(),
            })?;

        let hlc = global_hlc()?;
        let transition_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };

        let final_state = QuantumStateVector {
            state_vector: target_state.clone(),
            energy_levels: self.calculate_energy_eigenvalues(&target_state),
            coherence: self.calculate_coherence(&target_state),
            measured_at: transition_time,
            measuring_node: self.node_id,
            confidence: 1.0,
        };

        let transition = QuantumTransition {
            transition_id: transition_time.physical.wrapping_add(transition_time.logical),
            initial_state: current_state,
            final_state,
            unitary_operator,
            duration: 1, // Single logical time step
            consensus_nodes: consensus_nodes.clone(),
        };

        // Store transition
        self.active_transitions.write().insert(transition.transition_id, transition.clone());

        // Propose consensus on the transition
        let proposal_id = self.consensus_coordinator
            .propose_consensus(transition_time, consensus_nodes, 5000) // 5 second timeout
            .await?;

        info!(
            node_id = self.node_id,
            transition_id = transition.transition_id,
            proposal_id = proposal_id,
            target_coherence = transition.final_state.coherence,
            "Proposed quantum state transition for consensus"
        );

        Ok(transition.transition_id)
    }

    /// Execute agreed quantum state transition
    pub async fn execute_quantum_transition(&self, transition_id: u64) -> Result<QuantumStateVector, TimeError> {
        let transition = {
            let transitions = self.active_transitions.read();
            transitions.get(&transition_id).cloned()
                .ok_or_else(|| TimeError::SystemTimeError {
                    details: format!("Quantum transition {} not found", transition_id),
                })?
        };

        // Apply the transition
        let new_state = transition.final_state.clone();

        // Update local quantum state
        *self.current_state.write() = Some(new_state.clone());

        // Remove completed transition
        self.active_transitions.write().remove(&transition_id);

        info!(
            node_id = self.node_id,
            transition_id = transition_id,
            new_coherence = new_state.coherence,
            state_dimension = new_state.state_vector.len(),
            "Executed quantum state transition"
        );

        Ok(new_state)
    }

    /// Check consistency and trigger verification if needed
    async fn check_consistency_and_verify(&self) -> Result<(), TimeError> {
        let verification_result = self.verify_quantum_consistency().await?;

        if let QuantumVerificationResult::Inconsistent { max_deviation, .. } = verification_result {
            if max_deviation > self.max_state_deviation * 2.0 {
                // Severe inconsistency - trigger emergency consensus
                warn!(
                        node_id = self.node_id,
                        max_deviation = max_deviation,
                        threshold = self.max_state_deviation,
                        "Severe quantum state inconsistency detected, triggering emergency consensus"
                    );

                // Propose emergency state consensus
                let local_state = self.current_state.read().clone();
                if let Some(local_state) = local_state {
                    let peer_nodes: Vec<u64> = self.peer_states.read().keys().copied().collect();
                    let _proposal_id = self.consensus_coordinator
                        .propose_consensus(local_state.measured_at, peer_nodes, 2000) // 2 second emergency timeout
                        .await?;
                }
            }
        }

        Ok(())
    }

    /// Calculate energy eigenvalues for quantum state
    fn calculate_energy_eigenvalues(&self, state_vector: &[f64]) -> Vec<f64> {
        // Simplified energy calculation - in real implementation this would use proper quantum mechanics
        let mut eigenvalues = vec![0.0; state_vector.len()];

        for i in 0..state_vector.len() {
            eigenvalues[i] = state_vector[i].powi(2) * (i as f64 + 0.5); // Harmonic oscillator approximation
        }

        eigenvalues
    }

    /// Calculate quantum coherence for state vector
    fn calculate_coherence(&self, state_vector: &[f64]) -> f64 {
        // Simple coherence measure based on state vector norm and spread
        let norm = state_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mean = state_vector.iter().sum::<f64>() / state_vector.len() as f64;
        let variance = state_vector.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / state_vector.len() as f64;

        if variance > 0.0 {
            norm / (1.0 + variance.sqrt()) * (1.0 - (mean - 0.5).abs())
        } else {
            norm
        }
    }

    /// Calculate deviation between two quantum states
    fn calculate_state_deviation(&self, state1: &QuantumStateVector, state2: &QuantumStateVector) -> f64 {
        if state1.state_vector.len() != state2.state_vector.len() {
            return 1.0; // Maximum deviation for incompatible dimensions
        }

        // Calculate normalized Euclidean distance
        let diff_norm = state1.state_vector.iter().zip(state2.state_vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>().sqrt();
        let deviation = diff_norm / state1.state_vector.len() as f64;

        // Also consider coherence difference
        let coherence_diff = (state1.coherence - state2.coherence).abs();

        deviation + coherence_diff
    }

    /// Get quantum consistency statistics
    pub fn get_quantum_stats(&self) -> QuantumConsistencyStats {
        let current_state = self.current_state.read().clone();
        let peer_states = self.peer_states.read();
        let active_transitions = self.active_transitions.read();

        let local_coherence = current_state.as_ref().map(|s| s.coherence).unwrap_or(0.0);
        let avg_peer_coherence = if peer_states.is_empty() {
            0.0
        } else {
            peer_states.values().map(|s| s.coherence).sum::<f64>() / peer_states.len() as f64
        };

        QuantumConsistencyStats {
            node_id: self.node_id,
            local_coherence,
            avg_peer_coherence,
            peer_state_count: peer_states.len(),
            active_transition_count: active_transitions.len(),
            coherence_threshold: self.coherence_threshold,
            max_allowed_deviation: self.max_state_deviation,
        }
    }

    /// Cleanup old quantum states and transitions
    pub async fn cleanup_old_quantum_data(&self, before_time: LogicalTime) -> Result<usize, TimeError> {
        let mut removed_count = 0;

        // Cleanup old peer states
        {
            let mut peer_states = self.peer_states.write();
            let original_count = peer_states.len();

            peer_states.retain(|_, state| !state.measured_at.happens_before(before_time));
            removed_count += original_count - peer_states.len();
        }

        // Cleanup old transitions
        {
            let mut transitions = self.active_transitions.write();
            let original_count = transitions.len();

            transitions.retain(|_, transition| !transition.initial_state.measured_at.happens_before(before_time));
            removed_count += original_count - transitions.len();
        }

        if removed_count > 0 {
            debug!(
                node_id = self.node_id,
                removed_count = removed_count,
                before_time = %before_time,
                "Cleaned up old quantum data"
            );
        }

        Ok(removed_count)
    }
}

/// Statistics for quantum consistency monitoring
#[derive(Debug, Clone)]
pub struct QuantumConsistencyStats {
    /// Local node identifier
    pub node_id: u64,
    /// Local quantum coherence measure
    pub local_coherence: f64,
    /// Average coherence across peer nodes
    pub avg_peer_coherence: f64,
    /// Number of peer quantum states tracked
    pub peer_state_count: usize,
    /// Number of active quantum transitions
    pub active_transition_count: usize,
    /// Coherence threshold for consistency
    pub coherence_threshold: f64,
    /// Maximum allowed state deviation
    pub max_allowed_deviation: f64,
}

/// Enterprise quantum determinism manager
pub struct QuantumDeterminismManager {
    /// Quantum consistency coordinator
    consistency_coordinator: Arc<QuantumConsistencyCoordinator>,
    /// Background monitoring task
    monitoring_task: Option<tokio::task::JoinHandle<()>>,
    /// Monitoring interval in milliseconds
    monitoring_interval_ms: u64,
}

impl QuantumDeterminismManager {
    /// Create new quantum determinism manager
    pub fn new(
        consistency_coordinator: Arc<QuantumConsistencyCoordinator>,
        monitoring_interval_ms: u64,
    ) -> Self {
        Self {
            consistency_coordinator,
            monitoring_task: None,
            monitoring_interval_ms,
        }
    }

    /// Start background quantum consistency monitoring
    pub async fn start_monitoring(&mut self) -> Result<(), TimeError> {
        if self.monitoring_task.is_some() {
            return Err(TimeError::InvalidOperation {
                operation: "start_monitoring".to_string(),
                reason: "Quantum monitoring already running".to_string(),
            });
        }

        let coordinator = Arc::clone(&self.consistency_coordinator);
        let interval_ms = self.monitoring_interval_ms;

        let handle = tokio::task::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));

            loop {
                interval.tick().await;

                // Verify quantum consistency
                if let Err(e) = coordinator.check_consistency_and_verify().await {
                    warn!(
                        node_id = coordinator.node_id,
                        error = %e,
                        "Quantum consistency verification failed"
                    );
                }

                // Cleanup old data
                let hlc = match global_hlc() {
                    Ok(hlc) => hlc,
                    Err(e) => {
                        warn!(error = %e, "Failed to get global HLC for cleanup");
                        continue;
                    }
                };

                let cleanup_before = match hlc.read().current_time() {
                    Ok(current) => LogicalTime::new(
                        current.physical.saturating_sub(3_600_000_000_000), // 1 hour ago
                        0,
                        current.node_id,
                    ),
                    Err(e) => {
                        warn!(error = %e, "Failed to get current time for cleanup");
                        continue;
                    }
                };

                if let Err(e) = coordinator.cleanup_old_quantum_data(cleanup_before).await {
                    warn!(
                        node_id = coordinator.node_id,
                        error = %e,
                        "Failed to cleanup old quantum data"
                    );
                }
            }
        });

        self.monitoring_task = Some(handle);

        info!(
            node_id = self.consistency_coordinator.node_id,
            interval_ms = interval_ms,
            "Started quantum consistency monitoring"
        );

        Ok(())
    }

    /// Stop quantum consistency monitoring
    pub async fn stop_monitoring(&mut self) {
        if let Some(handle) = self.monitoring_task.take() {
            handle.abort();

            info!(
                node_id = self.consistency_coordinator.node_id,
                "Stopped quantum consistency monitoring"
            );
        }
    }

    /// Get comprehensive quantum status
    pub fn get_quantum_status(&self) -> QuantumConsistencyStats {
        self.consistency_coordinator.get_quantum_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consensus::ConsensusAlgorithm, distributed::DistributedSynchronizer, initialize_simulated_time_source, NanoTime};

    #[tokio::test]
    async fn test_quantum_consistency_coordinator_creation() {
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let consensus = Arc::new(TemporalConsensusCoordinator::new(
            1,
            local_sync,
            ConsensusAlgorithm::EnterpriseHybrid,
        ));

        let coordinator = QuantumConsistencyCoordinator::new(1, consensus, 0.8, 0.1);

        assert_eq!(coordinator.node_id, 1);
        assert_eq!(coordinator.coherence_threshold, 0.8);
        assert_eq!(coordinator.max_state_deviation, 0.1);
    }

    #[tokio::test]
    async fn test_quantum_state_initialization() {
        initialize_simulated_time_source(NanoTime::from_nanos(6000));

        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let consensus = Arc::new(TemporalConsensusCoordinator::new(
            1,
            local_sync,
            ConsensusAlgorithm::EnterpriseHybrid,
        ));

        let coordinator = QuantumConsistencyCoordinator::new(1, consensus, 0.8, 0.1);

        let initial_state = vec![1.0, 0.0, 0.0];
        let quantum_state = coordinator
            .initialize_quantum_state(initial_state)
            .await
            .expect("Should initialize quantum state");

        assert_eq!(quantum_state.measuring_node, 1);
        assert_eq!(quantum_state.state_vector.len(), 3);
        assert!(quantum_state.coherence > 0.0);
    }

    #[tokio::test]
    async fn test_quantum_state_verification() {
        initialize_simulated_time_source(NanoTime::from_nanos(7000));

        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let consensus = Arc::new(TemporalConsensusCoordinator::new(
            1,
            local_sync,
            ConsensusAlgorithm::EnterpriseHybrid,
        ));

        let coordinator = QuantumConsistencyCoordinator::new(1, consensus, 0.8, 0.1);

        // Initialize local state
        let initial_state = vec![1.0, 0.0];
        coordinator.initialize_quantum_state(initial_state).await.expect("Should initialize");

        // Verify consistency (should be consistent with just local state)
        let result = coordinator.verify_quantum_consistency().await.expect("Should verify");

        match result {
            QuantumVerificationResult::Consistent { verifying_nodes, .. } => {
                assert_eq!(verifying_nodes, vec![1]);
            }
            _ => panic!("Expected consistent result"),
        }
    }

    #[test]
    fn test_quantum_determinism_manager_creation() {
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let consensus = Arc::new(TemporalConsensusCoordinator::new(
            1,
            local_sync,
            ConsensusAlgorithm::EnterpriseHybrid,
        ));

        let consistency_coordinator = Arc::new(QuantumConsistencyCoordinator::new(1, consensus, 0.8, 0.1));
        let manager = QuantumDeterminismManager::new(consistency_coordinator, 1000);

        assert_eq!(manager.monitoring_interval_ms, 1000);
        assert!(manager.monitoring_task.is_none());
    }
}
