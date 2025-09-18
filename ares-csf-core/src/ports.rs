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

    /// Get the current time resolution in nanoseconds.
    fn resolution_ns(&self) -> u64 {
        1 // Default to nanosecond resolution
    }

    /// Check if the time source supports high-precision timing.
    fn is_high_precision(&self) -> bool {
        self.resolution_ns() <= 100 // Sub-microsecond precision
    }
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

    /// Get the physical component of the clock.
    fn physical_time(&self) -> NanoTime;

    /// Check if two timestamps are causally related.
    fn is_causal_order(&self, ts1: NanoTime, ts2: NanoTime) -> bool;

    /// Get the maximum drift allowed between physical and logical time.
    fn max_drift_ns(&self) -> u64 {
        1_000_000 // 1ms default
    }
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

    /// Get the number of pending tasks.
    fn pending_tasks(&self) -> usize;

    /// Cancel a scheduled task if possible.
    async fn cancel_task(&self, task_id: crate::types::TaskId) -> Result<bool, Error>;

    /// Get scheduling statistics.
    fn scheduler_stats(&self) -> SchedulerStats;
}

/// Statistics for deadline scheduler performance monitoring.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Total tasks scheduled
    pub total_scheduled: u64,
    /// Tasks completed on time
    pub on_time_completions: u64,
    /// Tasks that missed deadlines
    pub missed_deadlines: u64,
    /// Average scheduling latency in nanoseconds
    pub avg_scheduling_latency_ns: u64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Maximum queue depth seen
    pub max_queue_depth: usize,
}

impl SchedulerStats {
    /// Calculate the on-time completion rate.
    pub fn on_time_rate(&self) -> f64 {
        if self.total_scheduled > 0 {
            self.on_time_completions as f64 / self.total_scheduled as f64
        } else {
            0.0
        }
    }

    /// Calculate the deadline miss rate.
    pub fn miss_rate(&self) -> f64 {
        if self.total_scheduled > 0 {
            self.missed_deadlines as f64 / self.total_scheduled as f64
        } else {
            0.0
        }
    }
}

/// Provides a mechanism for reaching consensus on a value.
#[async_trait]
pub trait Consensus: Send + Sync {
    /// Proposes a value to the consensus algorithm.
    ///
    /// Returns the agreed-upon value.
    async fn propose(&self, value: Bytes) -> Result<Bytes, Error>;

    /// Get the current consensus state.
    async fn current_state(&self) -> Result<ConsensusState, Error>;

    /// Check if consensus has been reached for a specific proposal.
    async fn is_consensus_reached(&self, proposal_id: &str) -> Result<bool, Error>;

    /// Get the number of participants in the consensus protocol.
    fn participant_count(&self) -> usize;

    /// Get the minimum number of participants needed for consensus.
    fn quorum_size(&self) -> usize;
}

/// Current state of the consensus protocol.
#[derive(Debug, Clone)]
pub struct ConsensusState {
    /// Current round number
    pub round: u64,
    /// Number of active participants
    pub active_participants: usize,
    /// Whether consensus has been reached
    pub consensus_reached: bool,
    /// The agreed-upon value (if consensus reached)
    pub agreed_value: Option<Bytes>,
    /// Timestamp of last state change
    pub last_update: crate::types::Timestamp,
}

/// A secure, append-only ledger for immutable data storage.
#[async_trait]
pub trait SecureImmutableLedger: Send + Sync {
    /// The type of identifier used for log entries.
    type LogId: Send + Clone;

    /// Appends a batch of data to the ledger.
    ///
    /// Returns a unique identifier for the appended batch.
    async fn append(&self, data: &[Bytes]) -> Result<Self::LogId, Error>;

    /// Retrieves data by log identifier.
    async fn get(&self, log_id: &Self::LogId) -> Result<Option<Vec<Bytes>>, Error>;

    /// Get the current size of the ledger.
    async fn size(&self) -> Result<u64, Error>;

    /// Verify the integrity of the ledger.
    async fn verify_integrity(&self) -> Result<bool, Error>;

    /// Get ledger statistics.
    async fn stats(&self) -> Result<LedgerStats, Error>;
}

/// Statistics for ledger performance monitoring.
#[derive(Debug, Clone)]
pub struct LedgerStats {
    /// Total number of entries
    pub total_entries: u64,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Number of append operations
    pub append_operations: u64,
    /// Number of read operations
    pub read_operations: u64,
    /// Average append latency in nanoseconds
    pub avg_append_latency_ns: u64,
    /// Average read latency in nanoseconds
    pub avg_read_latency_ns: u64,
}

/// The sending half of an event bus.
#[async_trait]
pub trait EventBusTx: Send + Sync {
    /// Sends an envelope on the event bus.
    async fn send(&self, event: Envelope) -> Result<(), Error>;

    /// Sends multiple envelopes atomically.
    async fn send_batch(&self, events: Vec<Envelope>) -> Result<(), Error>;

    /// Check if the event bus is ready to accept new events.
    fn is_ready(&self) -> bool;

    /// Get the current queue depth.
    fn queue_depth(&self) -> usize;

    /// Get event bus transmission statistics.
    fn tx_stats(&self) -> BusStats;
}

/// The receiving half of an event bus.
#[async_trait]
pub trait EventBusRx: Send + Sync {
    /// Receives an envelope from the event bus.
    async fn recv(&mut self) -> Result<Envelope, Error>;

    /// Receives multiple envelopes at once (batching).
    async fn recv_batch(&mut self, max_events: usize) -> Result<Vec<Envelope>, Error>;

    /// Try to receive an envelope without blocking.
    fn try_recv(&mut self) -> Result<Option<Envelope>, Error>;

    /// Check if there are events available to receive.
    fn has_events(&self) -> bool;

    /// Get event bus reception statistics.
    fn rx_stats(&self) -> BusStats;
}

/// Statistics for event bus performance monitoring.
#[derive(Debug, Clone)]
pub struct BusStats {
    /// Total messages processed
    pub total_messages: u64,
    /// Messages processed in the last second
    pub messages_per_second: u64,
    /// Average message size in bytes
    pub avg_message_size_bytes: u64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Maximum queue depth seen
    pub max_queue_depth: usize,
    /// Number of dropped messages (if applicable)
    pub dropped_messages: u64,
    /// Average processing latency in nanoseconds
    pub avg_latency_ns: u64,
}

/// Generic data source trait for computational data streams.
#[async_trait]
pub trait DataSource: Send + Sync {
    /// The type of data this source provides.
    type Data: Send + Clone;

    /// Fetch data for the given time range.
    async fn fetch_range(
        &self,
        start: crate::types::Timestamp,
        end: crate::types::Timestamp,
    ) -> Result<Vec<Self::Data>, Error>;

    /// Fetch the latest available data.
    async fn fetch_latest(&self) -> Result<Option<Self::Data>, Error>;

    /// Check if the data source is currently available.
    async fn is_available(&self) -> bool;

    /// Get data source metadata.
    fn metadata(&self) -> DataSourceMetadata;
}

/// Metadata for a data source.
#[derive(Debug, Clone)]
pub struct DataSourceMetadata {
    /// Unique identifier for the data source
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of the data provided
    pub description: String,
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
    /// Data retention period
    pub retention_period: std::time::Duration,
    /// Whether the source provides real-time data
    pub real_time: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_stats() {
        let stats = SchedulerStats {
            total_scheduled: 100,
            on_time_completions: 95,
            missed_deadlines: 5,
            avg_scheduling_latency_ns: 1000,
            queue_depth: 10,
            max_queue_depth: 50,
        };

        assert_eq!(stats.on_time_rate(), 0.95);
        assert_eq!(stats.miss_rate(), 0.05);
    }

    #[test]
    fn test_consensus_state() {
        let state = ConsensusState {
            round: 1,
            active_participants: 5,
            consensus_reached: true,
            agreed_value: Some(Bytes::from("test")),
            last_update: crate::types::Timestamp::now(),
        };

        assert!(state.consensus_reached);
        assert!(state.agreed_value.is_some());
    }

    #[test]
    fn test_data_source_metadata() {
        let metadata = DataSourceMetadata {
            id: "test-source".to_string(),
            name: "Test Source".to_string(),
            description: "A test data source".to_string(),
            update_frequency_ms: 1000,
            retention_period: std::time::Duration::from_secs(86400),
            real_time: true,
        };

        assert_eq!(metadata.id, "test-source");
        assert!(metadata.real_time);
    }
}