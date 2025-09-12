//! Defines the ports for the hexagonal architecture.
//! These traits are the boundary of the core application logic.

use crate::envelope::Envelope;
use crate::error::Error;
use crate::types::NanoTime;
use async_trait::async_trait;
use bytes::Bytes;

/// A source for `ChronoSynclastic` Fabric time management.
///
/// Provides deterministic, high-precision time for the `NovaCore` architecture.
/// All time operations in CSF MUST use this trait for temporal coherence.
pub trait TimeSource: Send + Sync {
    /// Returns the current time with nanosecond precision.
    ///
    /// # Errors
    /// Returns error if time cannot be retrieved or temporal coherence is violated.
    fn now_ns(&self) -> Result<NanoTime, Error>;

    /// Returns monotonic time that never goes backwards.
    ///
    /// Essential for event ordering and causality tracking.
    ///
    /// # Errors
    /// Returns error if monotonic time cannot be retrieved.
    fn monotonic_ns(&self) -> Result<NanoTime, Error>;
}

/// A Hybrid Logical Clock (HLC) for causality-aware distributed timestamps.
///
/// Provides temporal ordering across distributed nodes with causality guarantees.
/// Essential for `ChronoSynclastic` coherence in the `NovaCore` architecture.
pub trait HlcClock: Send + Sync {
    /// Returns the current HLC timestamp with logical ordering.
    ///
    /// # Errors
    /// Returns error if clock state is invalid or causality violation detected.
    fn now_hlc(&self) -> Result<NanoTime, Error>;

    /// Updates the clock with a timestamp from a remote system.
    ///
    /// Maintains causality by advancing logical time when receiving
    /// timestamps from other nodes in the distributed system.
    ///
    /// # Errors  
    /// Returns error if remote timestamp would violate causality.
    fn update_hlc(&mut self, remote_time: NanoTime) -> Result<(), Error>;

    /// Get the current logical component of the clock.
    fn logical_time(&self) -> u64;
}

/// Quantum-inspired deadline scheduling for temporal task coordination.
///
/// Provides predictive scheduling with causality awareness and quantum optimization
/// for sub-microsecond precision in the `ChronoSynclastic` Fabric.
#[async_trait]
pub trait DeadlineScheduler: Send + Sync {
    /// The type of task to be scheduled.
    type Task: Send;

    /// Schedules a task to run before the given deadline with quantum optimization.
    ///
    /// # Arguments
    /// * `task` - The task to schedule
    /// * `deadline` - Absolute deadline in nanoseconds
    ///
    /// # Errors
    /// Returns error if deadline has passed or scheduling resources exhausted.
    async fn schedule_with_deadline(
        &self,
        task: Self::Task,
        deadline: NanoTime,
    ) -> Result<(), Error>;

    /// Schedules a task with relative deadline from now.
    ///
    /// # Arguments
    /// * `task` - The task to schedule  
    /// * `delay` - Relative delay from current time
    async fn schedule_after(&self, task: Self::Task, delay: NanoTime) -> Result<(), Error>;

    /// Get current scheduling load for backpressure control.
    fn current_load(&self) -> f64;
}

/// Provides a mechanism for reaching consensus on a value.
#[async_trait]
pub trait Consensus: Send + Sync {
    /// Proposes a value to the consensus algorithm.
    ///
    /// Returns the agreed-upon value.
    async fn propose(&self, value: Bytes) -> Result<Bytes, Error>;
}

/// A secure, append-only ledger for immutable data storage.
#[async_trait]
pub trait SecureImmutableLedger: Send + Sync {
    /// The type of identifier used for log entries.
    type LogId: Send;

    /// Appends a batch of data to the ledger.
    ///
    /// Returns a unique identifier for the appended batch.
    async fn append(&self, data: &[Bytes]) -> Result<Self::LogId, Error>;
}

/// The sending half of an event bus.
#[async_trait]
pub trait EventBusTx: Send + Sync {
    /// Sends an envelope on the event bus.
    async fn send(&self, event: Envelope) -> Result<(), Error>;
}

/// The receiving half of an event bus.
#[async_trait]
pub trait EventBusRx: Send + Sync {
    /// Receives an envelope from the event bus.
    async fn recv(&mut self) -> Result<Envelope, Error>;
}
