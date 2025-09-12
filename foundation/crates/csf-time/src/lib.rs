//! ChronoSynclastic Fabric Time Management
//!
//! This crate implements the Temporal Task Weaver (TTW) time abstraction for the NovaCore
//! ARES ChronoSynclastic Fabric. It provides deterministic time management with causality
//! tracking, quantum-inspired optimization, and sub-microsecond precision.
//!
//! # Core Components
//!
//! - [`TimeSource`]: Deterministic time source for ChronoSynclastic coherence
//! - [`HlcClock`]: Hybrid Logical Clock with causality tracking
//! - [`DeadlineScheduler`]: Predictive temporal analysis with quantum optimization
//! - [`NanoTime`]: High-precision time representation
//! - [`QuantumTimeOracle`]: Quantum-inspired optimization algorithms

#![warn(missing_docs)]

pub mod clock;
pub mod consensus;
pub mod cross_system;
pub mod deadline;
pub mod deadline_global;
pub mod distributed;
pub mod hlc_global;
pub mod quantum_consistency;
/// High-precision time representations
pub mod nano_time;
pub mod oracle;
pub mod precision;
pub mod source;
pub mod sync;

pub mod optimizer;

/// A prelude for conveniently importing the most common types from `csf-time`.
pub mod prelude {
    pub use super::clock::{HlcClock, LogicalTime};
    pub use super::deadline::DeadlineScheduler;
    pub use super::nano_time::{Duration, NanoTime};
    pub use super::precision::{
        PreciseDuration, PreciseQuantumOffset, PrecisionBound, PrecisionLevel,
    };
    pub use super::source::TimeSource;
    pub use super::{TimeError, TimeResult};
}

pub use clock::{
    CausalCheckpoint, CausalDependency, CausalityResult, DependencyType, DistributedBarrier,
    DistributedCoordinationState, Event, HlcClock, HlcClockImpl, LogicalTime, NodeState, NodeStatus,
};
pub use deadline::{DeadlineScheduler, DeadlineSchedulerImpl, OptimizationResult, ScheduleResult};
pub use deadline_global::{
    global_deadline_load, global_deadline_scheduler, global_schedule_after,
    global_schedule_with_deadline, initialize_global_deadline_scheduler,
    is_global_deadline_scheduler_initialized,
};
pub use hlc_global::{
    global_hlc, global_hlc_cleanup_barriers, global_hlc_create_barrier, 
    global_hlc_distributed_state, global_hlc_enterprise_sync, global_hlc_is_barrier_synchronized,
    global_hlc_now, global_hlc_reach_barrier, global_hlc_register_peer, global_hlc_update,
    global_hlc_update_peer, initialize_global_hlc, is_global_hlc_initialized,
};
pub use nano_time::{Duration, NanoTime, QuantumOffset};
pub use oracle::{OptimizationHint, OptimizationStrategy, QuantumState, QuantumTimeOracle};
pub use precision::{
    ErrorBounds, PreciseDuration, PreciseQuantumOffset, PrecisionBound, PrecisionLevel,
    PrecisionMetadata,
};
pub use consensus::{
    ConsensusAlgorithm, ConsensusProposal, ConsensusResult, ConsensusStats, ConsensusVote,
    TemporalConsensusCoordinator, VoteDecision, VoteTally,
};
pub use cross_system::{
    ComprehensiveTemporalStatus, CrossSystemSyncStats, CrossSystemSynchronizer,
    EnterpriseTemporalCoordinator, PeerSystemInfo, SyncProtocol, SyncSession,
    SyncSessionStatus, SystemStatus, SystemTimeOffset, SystemType,
};
pub use distributed::{ConsensusProtocol, DistributedSynchronizer};
pub use quantum_consistency::{
    QuantumConsistencyCoordinator, QuantumConsistencyStats, QuantumDeterminismManager,
    QuantumStateVector, QuantumTransition, QuantumVerificationResult,
};
pub use source::{
    get_or_init_test_time_source, initialize_global_time_source, initialize_simulated_time_source,
    SimulatedTimeSource, TimeCheckpoint, TimeSource, TimeSourceImpl,
};
pub use sync::{CausalityValidator, TemporalCoherence};

use thiserror::Error;

/// Errors that can occur in time management operations
#[derive(Error, Debug)]
pub enum TimeError {
    /// Causality violation detected with expected and actual logical times
    #[error("Causality violation: expected {expected:?}, got {actual:?}")]
    CausalityViolation {
        /// Expected logical time for causality
        expected: crate::LogicalTime,
        /// Actual logical time that caused violation
        actual: crate::LogicalTime,
    },

    /// Clock synchronization failed
    #[error("Clock sync failed: {reason}")]
    SyncFailure {
        /// Reason for synchronization failure
        reason: String,
    },

    /// Quantum optimization failed
    #[error("Quantum optimization error: {details}")]
    QuantumError {
        /// Details about the quantum optimization failure
        details: String,
    },

    /// Deadline scheduling failed for a specific task
    #[error("Deadline miss: task {task_id} missed deadline by {overage_ns}ns")]
    DeadlineMiss {
        /// Task identifier that missed deadline
        task_id: String,
        /// Nanoseconds by which deadline was missed
        overage_ns: u64,
    },

    /// Deadline scheduling failed
    #[error("Deadline scheduling failed: {task_id} - {reason}")]
    DeadlineFailure {
        /// Task identifier
        task_id: String,
        /// Reason for scheduling failure
        reason: String,
    },

    /// Quantum optimization failed during scheduling
    #[error("Quantum optimization failed: {details}")]
    QuantumOptimizationFailure {
        /// Details about optimization failure
        details: String,
    },

    /// System time error (e.g., clock went backwards, unavailable)
    #[error("System time error: {details}")]
    SystemTimeError {
        /// Details about system time error
        details: String,
    },

    /// Hardware timing features unavailable
    #[error("Hardware timing unavailable: {details}")]
    HardwareUnavailable {
        /// Details about unavailable hardware
        details: String,
    },

    /// Time arithmetic overflow
    #[error("Time arithmetic overflow")]
    ArithmeticOverflow,

    /// Invalid time value
    #[error("Invalid time value: {value}")]
    InvalidTime {
        /// Invalid time value
        value: i64,
    },

    /// Invalid operation attempted
    #[error("Invalid operation: {operation} - {reason}")]
    InvalidOperation {
        /// Operation that was attempted
        operation: String,
        /// Reason why operation is invalid
        reason: String,
    },
}

/// Result type for time operations
pub type TimeResult<T> = std::result::Result<T, TimeError>;

// Global time source re-exported from source module for consistency
pub use source::global_time_source;

/// Get current time from global source
///
/// Returns error if the global time source hasn't been initialized or if time retrieval fails
pub fn now() -> TimeResult<NanoTime> {
    global_time_source().now_ns()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_error_display() {
        let err = TimeError::CausalityViolation {
            expected: crate::LogicalTime::new(100, 5, 1),
            actual: crate::LogicalTime::new(99, 3, 1),
        };
        assert!(err.to_string().contains("Causality violation"));
    }

    #[test]
    fn test_global_time_source_init() {
        // Initialize simulated time source for testing
        initialize_simulated_time_source(NanoTime::from_nanos(1000));

        // Test that we can get time from the initialized source
        let time_result = global_time_source().now_ns();
        assert!(
            time_result.is_ok(),
            "Should be able to get time from global source"
        );

        // The actual time may vary depending on which source is active
        let time = time_result.unwrap();
        assert!(time >= NanoTime::ZERO, "Time should be non-negative");
    }
}
