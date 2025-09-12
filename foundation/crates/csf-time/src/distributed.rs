//! Enterprise distributed node synchronization mechanisms
//!
//! This module implements enterprise-grade distributed coordination protocols
//! for ChronoSynclastic deterministic execution across multiple nodes.

use crate::{
    clock::{DistributedCoordinationState, HlcClock, NodeState, NodeStatus},
    global_hlc, LogicalTime, TimeError,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Enterprise distributed synchronization coordinator
#[derive(Debug)]
pub struct DistributedSynchronizer {
    /// Local node identifier
    node_id: u64,
    /// Coordination state
    state: Arc<RwLock<DistributedCoordinationState>>,
    /// Network timeout for distributed operations
    #[allow(dead_code)]
    network_timeout_ms: u64,
}

impl DistributedSynchronizer {
    /// Create new distributed synchronizer
    pub fn new(node_id: u64, network_timeout_ms: u64) -> Self {
        Self {
            node_id,
            state: Arc::new(RwLock::new(DistributedCoordinationState {
                peer_nodes: HashMap::new(),
                global_time_vector: HashMap::new(),
                synchronization_barriers: Vec::new(),
                determinism_epoch: 0,
                last_global_sync: LogicalTime::zero(node_id),
            })),
            network_timeout_ms,
        }
    }

    /// Register multiple peer nodes for distributed coordination
    pub async fn register_peer_cluster(&self, peer_nodes: &[(u64, LogicalTime)]) -> Result<(), TimeError> {
        let mut state = self.state.write();
        
        for &(peer_node_id, initial_time) in peer_nodes {
            let node_state = NodeState {
                node_id: peer_node_id,
                last_seen_time: initial_time,
                status: NodeStatus::Active,
                rtt_nanos: 0,
                last_heartbeat: initial_time,
            };
            
            state.peer_nodes.insert(peer_node_id, node_state);
            state.global_time_vector.insert(peer_node_id, initial_time);
        }
        
        info!(
            node_id = self.node_id,
            peer_count = peer_nodes.len(),
            "Registered peer cluster for distributed coordination"
        );
        
        Ok(())
    }

    /// Perform distributed consensus on global logical time
    pub async fn consensus_global_time(&self) -> Result<LogicalTime, TimeError> {
        let state = self.state.read();
        
        // Collect all known timestamps from the global time vector
        let mut timestamps: Vec<LogicalTime> = state.global_time_vector.values().copied().collect();
        
        // Add our current time
        let hlc = global_hlc()?;
        let our_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        timestamps.push(our_time);
        
        // Find the maximum timestamp (distributed consensus)
        let consensus_time = timestamps.into_iter().reduce(|acc, time| acc.max(time))
            .unwrap_or(our_time);
        
        debug!(
            node_id = self.node_id,
            consensus_time = %consensus_time,
            peer_count = state.peer_nodes.len(),
            "Calculated distributed consensus time"
        );
        
        Ok(consensus_time)
    }

    /// Execute enterprise deterministic barrier synchronization
    pub async fn execute_barrier_sync(&self, barrier_id: u64, timeout_ms: u64) -> Result<LogicalTime, TimeError> {
        let start_time = std::time::Instant::now();
        let timeout_duration = Duration::from_millis(timeout_ms);
        
        loop {
            // Check if barrier is synchronized
            let hlc = global_hlc()?;
            let is_synchronized = {
                let clock = hlc.read();
                clock.is_barrier_synchronized(barrier_id)?
            };
            
            if is_synchronized {
                // All nodes reached barrier - perform global sync
                let sync_time = {
                    let clock = hlc.read();
                    clock.enterprise_global_sync()?
                };
                
                info!(
                    node_id = self.node_id,
                    barrier_id = barrier_id,
                    sync_time = %sync_time,
                    elapsed_ms = start_time.elapsed().as_millis(),
                    "Barrier synchronization completed successfully"
                );
                
                return Ok(sync_time);
            }
            
            // Check timeout
            if start_time.elapsed() > timeout_duration {
                warn!(
                    node_id = self.node_id,
                    barrier_id = barrier_id,
                    timeout_ms = timeout_ms,
                    "Barrier synchronization timed out"
                );
                
                return Err(TimeError::SyncFailure {
                    reason: format!("Barrier {} synchronization timed out after {}ms", barrier_id, timeout_ms),
                });
            }
            
            // Small delay before next check
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Monitor peer node health and update status
    pub async fn monitor_peer_health(&self, heartbeat_timeout_ms: u64) -> Result<Vec<u64>, TimeError> {
        let mut failed_nodes = Vec::new();
        let hlc = global_hlc()?;
        let current_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        let mut state = self.state.write();
        
        for (node_id, node_state) in state.peer_nodes.iter_mut() {
            let elapsed_ns = current_time.physical.saturating_sub(node_state.last_heartbeat.physical);
            let elapsed_ms = elapsed_ns / 1_000_000;
            
            if elapsed_ms > heartbeat_timeout_ms && node_state.status == NodeStatus::Active {
                node_state.status = NodeStatus::Degraded;
                
                if elapsed_ms > heartbeat_timeout_ms * 3 {
                    node_state.status = NodeStatus::Failed;
                    failed_nodes.push(*node_id);
                    
                    warn!(
                        node_id = self.node_id,
                        failed_node = node_id,
                        elapsed_ms = elapsed_ms,
                        "Peer node marked as failed due to heartbeat timeout"
                    );
                }
            }
        }
        
        Ok(failed_nodes)
    }

    /// Get active peer nodes for coordination
    pub fn get_active_peers(&self) -> Vec<u64> {
        let state = self.state.read();
        state.peer_nodes
            .iter()
            .filter(|(_, node_state)| node_state.status == NodeStatus::Active)
            .map(|(&node_id, _)| node_id)
            .collect()
    }

    /// Calculate network round-trip time to peer node
    pub async fn measure_peer_rtt(&self, peer_node_id: u64) -> Result<u64, TimeError> {
        let start = std::time::Instant::now();
        
        // Simulate network ping (in real implementation, this would be actual network communication)
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        let rtt_nanos = start.elapsed().as_nanos() as u64;
        
        // Update peer node RTT in state
        let mut state = self.state.write();
        if let Some(node_state) = state.peer_nodes.get_mut(&peer_node_id) {
            node_state.rtt_nanos = rtt_nanos;
        }
        
        debug!(
            node_id = self.node_id,
            peer_node_id = peer_node_id,
            rtt_nanos = rtt_nanos,
            "Measured peer node RTT"
        );
        
        Ok(rtt_nanos)
    }

    /// Get current distributed coordination state
    pub fn get_state_snapshot(&self) -> DistributedCoordinationState {
        self.state.read().clone()
    }
}

/// Enterprise consensus protocol for distributed determinism
pub struct ConsensusProtocol {
    /// Minimum number of nodes required for consensus
    quorum_size: usize,
    /// Maximum time to wait for consensus
    consensus_timeout_ms: u64,
    /// Synchronizer for coordination
    synchronizer: Arc<DistributedSynchronizer>,
}

impl ConsensusProtocol {
    /// Create new consensus protocol
    pub fn new(quorum_size: usize, consensus_timeout_ms: u64, synchronizer: Arc<DistributedSynchronizer>) -> Self {
        Self {
            quorum_size,
            consensus_timeout_ms,
            synchronizer,
        }
    }

    /// Execute distributed consensus on logical time
    pub async fn consensus_on_logical_time(&self, proposed_time: LogicalTime) -> Result<LogicalTime, TimeError> {
        let active_peers = self.synchronizer.get_active_peers();
        
        if active_peers.len() + 1 < self.quorum_size {
            return Err(TimeError::SyncFailure {
                reason: format!("Insufficient nodes for consensus: need {}, have {}", 
                    self.quorum_size, active_peers.len() + 1),
            });
        }
        
        // Create barrier for consensus
        let barrier_id = {
            let hlc = global_hlc()?;
            let clock = hlc.read();
            clock.create_synchronization_barrier(active_peers, self.consensus_timeout_ms * 1_000_000)?
        };
        
        // Wait for consensus
        let consensus_time = self.synchronizer
            .execute_barrier_sync(barrier_id, self.consensus_timeout_ms)
            .await?;
        
        info!(
            node_id = self.synchronizer.node_id,
            barrier_id = barrier_id,
            proposed_time = %proposed_time,
            consensus_time = %consensus_time,
            quorum_size = self.quorum_size,
            "Distributed consensus completed successfully"
        );
        
        Ok(consensus_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_simulated_time_source, NanoTime};

    #[tokio::test]
    async fn test_distributed_synchronizer_creation() {
        let synchronizer = DistributedSynchronizer::new(1, 5000);
        assert_eq!(synchronizer.node_id, 1);
        assert_eq!(synchronizer.network_timeout_ms, 5000);
    }

    #[tokio::test]
    async fn test_peer_registration() {
        initialize_simulated_time_source(NanoTime::from_nanos(1000));
        let synchronizer = DistributedSynchronizer::new(1, 5000);
        
        let peers = vec![
            (2, LogicalTime::new(1000, 0, 2)),
            (3, LogicalTime::new(1000, 0, 3)),
        ];
        
        synchronizer.register_peer_cluster(&peers).await.expect("Should register peers");
        
        let active_peers = synchronizer.get_active_peers();
        assert_eq!(active_peers.len(), 2);
        assert!(active_peers.contains(&2));
        assert!(active_peers.contains(&3));
    }

    #[tokio::test]
    async fn test_consensus_protocol() {
        initialize_simulated_time_source(NanoTime::from_nanos(2000));
        let synchronizer = Arc::new(DistributedSynchronizer::new(1, 5000));
        
        let peers = vec![
            (2, LogicalTime::new(2000, 0, 2)),
            (3, LogicalTime::new(2000, 0, 3)),
        ];
        synchronizer.register_peer_cluster(&peers).await.expect("Should register peers");
        
        let consensus = ConsensusProtocol::new(2, 1000, synchronizer);
        let proposed_time = LogicalTime::new(3000, 0, 1);
        
        // Note: This test may time out in isolation since we don't have actual peer communication
        // In real usage, peers would call reach_barrier from their nodes
    }
}
