//! Global HLC clock management for ChronoSynclastic Fabric
//!
//! This module provides utilities for managing a global HLC clock instance
//! across the distributed system, ensuring consistent causality tracking.

use crate::{global_time_source, HlcClock, HlcClockImpl, TimeError};
use parking_lot::RwLock;
use std::sync::{Arc, OnceLock};

/// Global HLC clock instance for the ChronoSynclastic Fabric
static GLOBAL_HLC: OnceLock<Arc<RwLock<HlcClockImpl>>> = OnceLock::new();

/// Initialize the global HLC clock with the specified node ID
///
/// This should be called once during application startup to establish
/// the node's identity in the distributed system.
///
/// # Arguments
/// * `node_id` - Unique identifier for this node in the distributed system
///
/// # Errors
/// Returns error if the clock has already been initialized or if
/// the underlying HLC implementation fails to initialize.
pub fn initialize_global_hlc(node_id: u64) -> Result<(), TimeError> {
    let time_source = global_time_source();
    let hlc_impl = HlcClockImpl::new(node_id, Arc::clone(time_source))?;

    GLOBAL_HLC
        .set(Arc::new(RwLock::new(hlc_impl)))
        .map_err(|_| TimeError::SystemTimeError {
            details: "Global HLC clock already initialized".to_string(),
        })?;

    tracing::info!(node_id = node_id, "Global HLC clock initialized");
    Ok(())
}

/// Get reference to the global HLC clock for read operations
///
/// # Panics
/// Panics if the global HLC clock has not been initialized.
/// Call `initialize_global_hlc()` first.
pub fn global_hlc() -> Result<Arc<RwLock<HlcClockImpl>>, TimeError> {
    GLOBAL_HLC.get().cloned().ok_or(TimeError::SystemTimeError {
        details: "Global HLC clock not initialized. Call initialize_global_hlc() first."
            .to_string(),
    })
}

/// Get current HLC timestamp from the global clock
///
/// This is a convenience function for getting the current logical time
/// from the global HLC clock.
pub fn global_hlc_now() -> Result<crate::LogicalTime, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    HlcClock::tick(&*clock)
}

/// Update global HLC clock with a remote timestamp
///
/// This should be called when receiving messages from remote nodes
/// to maintain causality across the distributed system.
///
/// # Arguments
/// * `remote_time` - Logical timestamp from a remote node
pub fn global_hlc_update(
    remote_time: crate::LogicalTime,
) -> Result<crate::CausalityResult, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    HlcClock::update(&*clock, remote_time)
}

/// Check if the global HLC clock has been initialized
pub fn is_global_hlc_initialized() -> bool {
    GLOBAL_HLC.get().is_some()
}

/// Enterprise distributed coordination functions for global HLC
/// Register a peer node with the global HLC for distributed coordination
pub fn global_hlc_register_peer(peer_node_id: u64, initial_time: crate::LogicalTime) -> Result<(), TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.register_peer_node(peer_node_id, initial_time)
}

/// Update peer node state in global HLC from received message
pub fn global_hlc_update_peer(peer_node_id: u64, peer_time: crate::LogicalTime) -> Result<(), TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.update_peer_node(peer_node_id, peer_time)
}

/// Create a distributed synchronization barrier across specified nodes
pub fn global_hlc_create_barrier(required_nodes: Vec<u64>, timeout_ns: u64) -> Result<u64, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.create_synchronization_barrier(required_nodes, timeout_ns)
}

/// Signal that this node has reached a synchronization barrier
pub fn global_hlc_reach_barrier(barrier_id: u64) -> Result<bool, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.reach_synchronization_barrier(barrier_id)
}

/// Check if all nodes have reached the specified barrier
pub fn global_hlc_is_barrier_synchronized(barrier_id: u64) -> Result<bool, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.is_barrier_synchronized(barrier_id)
}

/// Perform enterprise global time synchronization across all peers
pub fn global_hlc_enterprise_sync() -> Result<crate::LogicalTime, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.enterprise_global_sync()
}

/// Get distributed coordination state snapshot for monitoring
pub fn global_hlc_distributed_state() -> Result<crate::DistributedCoordinationState, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    Ok(clock.get_distributed_state_snapshot())
}

/// Cleanup expired synchronization barriers
pub fn global_hlc_cleanup_barriers() -> Result<usize, TimeError> {
    let hlc = global_hlc()?;
    let clock = hlc.read();
    clock.cleanup_expired_barriers()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_simulated_time_source, NanoTime};

    #[test]
    fn test_global_hlc_initialization() {
        // Reset any previous initialization for test isolation
        // Note: In real applications, this should only be called once

        // Initialize simulated time source for testing
        initialize_simulated_time_source(NanoTime::from_nanos(1000));

        // Initialize global HLC
        initialize_global_hlc(42).expect("Should initialize global HLC");

        // Verify it's initialized
        assert!(is_global_hlc_initialized());

        // Get current time
        let time = global_hlc_now().expect("Should get HLC time");
        assert_eq!(time.node_id, 42);
        assert!(time.physical > 0);
    }

    #[test]
    fn test_global_hlc_causality_update() {
        initialize_simulated_time_source(NanoTime::from_nanos(2000));

        // Note: This test assumes the global HLC hasn't been initialized yet
        // In a real test suite, you'd want proper test isolation
        if !is_global_hlc_initialized() {
            initialize_global_hlc(100).expect("Should initialize global HLC");
        }

        // Create a remote timestamp
        let remote_time = crate::LogicalTime::new(1500, 5, 200);

        // Update with remote time
        let result = global_hlc_update(remote_time).expect("Should update HLC");

        // Verify the result indicates causality handling
        match result {
            crate::CausalityResult::Valid { .. } | crate::CausalityResult::Concurrent { .. } => {
                // Both are acceptable outcomes
            }
            crate::CausalityResult::Violation { .. } => {
                panic!("Unexpected causality violation");
            }
            crate::CausalityResult::IncompleteDependencies { .. } => {
                // This is also acceptable for this test
            }
        }
    }
}