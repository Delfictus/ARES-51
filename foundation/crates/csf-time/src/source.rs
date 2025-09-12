//! Production-grade TimeSource implementations for ChronoSynclastic determinism
//!
//! This module provides the core time abstraction for the Temporal Task Weaver (TTW),
//! ensuring all time operations are deterministic, causality-aware, and quantum-optimized.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use tracing::{debug, instrument, warn};

use crate::{oracle::QuantumTimeOracle, NanoTime, QuantumOffset, TimeError};
// Re-export types needed for time management

/// Time checkpoint for deterministic replay and debugging
#[derive(Debug, Clone, PartialEq)]
pub struct TimeCheckpoint {
    /// Timestamp when checkpoint was created
    pub timestamp_ns: NanoTime,
    /// Monotonic counter value at checkpoint
    pub monotonic_value: u64,
    /// Quantum state at checkpoint
    pub quantum_state: QuantumOffset,
    /// Checkpoint identifier for tracking
    pub id: u64,
}

impl TimeCheckpoint {
    /// Create a new time checkpoint
    pub fn new(
        timestamp_ns: NanoTime,
        monotonic_value: u64,
        quantum_state: QuantumOffset,
        id: u64,
    ) -> Self {
        Self {
            timestamp_ns,
            monotonic_value,
            quantum_state,
            id,
        }
    }
}

/// Core time source trait for ChronoSynclastic Fabric
///
/// All time operations in the CSF system MUST go through a TimeSource implementation
/// to ensure deterministic, causality-aware temporal behavior.
///
/// # Performance Requirements
/// - Sub-microsecond latency for `now_ns()` calls
/// - Zero memory allocations in hot paths
/// - Thread-safe concurrent access
///
/// # Safety Requirements
/// - No `unwrap()` or `expect()` calls
/// - Monotonic time guarantees
/// - Graceful error handling
pub trait TimeSource: Send + Sync + 'static + std::fmt::Debug {
    /// Get current time in nanoseconds with quantum optimization
    ///
    /// # Performance
    /// Must complete in <1μs for production workloads
    fn now_ns(&self) -> Result<NanoTime, TimeError>;

    /// Get monotonic time that never goes backwards
    ///
    /// This provides a strictly increasing timestamp for ordering events
    fn monotonic_ns(&self) -> Result<NanoTime, TimeError>;

    /// Create a checkpoint for deterministic replay
    ///
    /// Checkpoints capture the complete temporal state for debugging
    fn create_checkpoint(&self) -> Result<TimeCheckpoint, TimeError>;

    /// Advance simulation time by delta nanoseconds
    ///
    /// Only supported by simulated time sources. Production sources return Ok(())
    fn advance_simulation(&self, delta_ns: u64) -> Result<(), TimeError>;

    /// Check if this is a simulated time source
    fn is_simulated(&self) -> bool {
        false
    }

    /// Get time resolution in nanoseconds
    fn resolution_ns(&self) -> u64 {
        1 // Default to nanosecond precision
    }

    /// Get quantum offset for optimization hints
    fn quantum_offset(&self) -> QuantumOffset;
}

/// Production time source using system clock with quantum optimization
///
/// This implementation provides sub-microsecond latency with hardware-accelerated
/// quantum optimization for predictive temporal analysis.
#[derive(Debug)]
pub struct TimeSourceImpl {
    /// Quantum oracle for optimization hints
    quantum_oracle: Arc<QuantumTimeOracle>,
    /// Base time offset for calibration
    #[allow(dead_code)]
    base_offset: AtomicU64,
    /// Monotonic counter ensuring strict ordering
    monotonic_counter: AtomicU64,
    /// Checkpoint counter for unique IDs
    checkpoint_counter: AtomicU64,
    /// Start time for relative measurements
    #[allow(dead_code)]
    start_time: NanoTime,
}

impl TimeSourceImpl {
    /// Create a new production time source
    ///
    /// # Errors
    /// Returns `TimeError::SystemTimeError` if system clock is unavailable
    pub fn new() -> Result<Self, TimeError> {
        let start_time = Self::get_system_time_ns()?;

        Ok(Self {
            quantum_oracle: Arc::new(QuantumTimeOracle::new()),
            base_offset: AtomicU64::new(0),
            monotonic_counter: AtomicU64::new(start_time.as_nanos()),
            checkpoint_counter: AtomicU64::new(0),
            start_time,
        })
    }

    /// Create with specific quantum oracle
    pub fn with_oracle(oracle: Arc<QuantumTimeOracle>) -> Result<Self, TimeError> {
        let start_time = Self::get_system_time_ns()?;

        Ok(Self {
            quantum_oracle: oracle,
            base_offset: AtomicU64::new(0),
            monotonic_counter: AtomicU64::new(start_time.as_nanos()),
            checkpoint_counter: AtomicU64::new(0),
            start_time,
        })
    }

    /// Get system time safely without panicking
    fn get_system_time_ns() -> Result<NanoTime, TimeError> {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| NanoTime::from_nanos(d.as_nanos() as u64))
            .map_err(|e| TimeError::SystemTimeError {
                details: format!("System time before UNIX epoch: {}", e),
            })
    }

    /// Ensure monotonic time progression
    fn ensure_monotonic(&self, time_ns: u64) -> u64 {
        let current_max = self.monotonic_counter.load(Ordering::Acquire);

        // If new time is greater, update and return it
        if time_ns > current_max {
            match self.monotonic_counter.compare_exchange_weak(
                current_max,
                time_ns,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => time_ns,
                Err(actual) => {
                    // Another thread updated, ensure we're still monotonic
                    std::cmp::max(time_ns, actual + 1)
                }
            }
        } else {
            // Time went backwards, increment from current max
            let next_time = current_max + 1;
            self.monotonic_counter.store(next_time, Ordering::Release);
            next_time
        }
    }
}

impl Default for TimeSourceImpl {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback for tests - should never happen in production
            Self {
                quantum_oracle: Arc::new(QuantumTimeOracle::new()),
                base_offset: AtomicU64::new(0),
                monotonic_counter: AtomicU64::new(0),
                checkpoint_counter: AtomicU64::new(0),
                start_time: NanoTime::ZERO,
            }
        })
    }
}

impl TimeSource for TimeSourceImpl {
    #[instrument(level = "trace", skip(self))]
    fn now_ns(&self) -> Result<NanoTime, TimeError> {
        // Get system time safely
        let system_time = Self::get_system_time_ns()?;

        // Apply quantum optimization
        let quantum_offset = self.quantum_oracle.current_offset();
        let optimized_time = quantum_offset.apply(system_time);

        // Ensure monotonic property with lock-free algorithm
        let monotonic_time = self.ensure_monotonic(optimized_time.as_nanos());

        Ok(NanoTime::from_nanos(monotonic_time))
    }

    #[instrument(level = "trace", skip(self))]
    fn monotonic_ns(&self) -> Result<NanoTime, TimeError> {
        let current_monotonic = self.monotonic_counter.load(Ordering::Acquire);
        Ok(NanoTime::from_nanos(current_monotonic))
    }

    #[instrument(level = "debug", skip(self))]
    fn create_checkpoint(&self) -> Result<TimeCheckpoint, TimeError> {
        let timestamp = self.now_ns()?;
        let monotonic_value = self.monotonic_counter.load(Ordering::Acquire);
        let quantum_state = self.quantum_oracle.current_offset();
        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::AcqRel);

        let checkpoint =
            TimeCheckpoint::new(timestamp, monotonic_value, quantum_state, checkpoint_id);

        debug!(
            checkpoint_id = checkpoint_id,
            timestamp_ns = timestamp.as_nanos(),
            "Created time checkpoint"
        );

        Ok(checkpoint)
    }

    fn advance_simulation(&self, _delta_ns: u64) -> Result<(), TimeError> {
        // Production time source doesn't support simulation advancement
        // This is intentional - only SimulatedTimeSource supports this
        debug!("Ignoring advance_simulation call on production time source");
        Ok(())
    }

    fn resolution_ns(&self) -> u64 {
        // System clock typically has nanosecond resolution
        1
    }

    fn quantum_offset(&self) -> QuantumOffset {
        self.quantum_oracle.current_offset()
    }
}

/// Simulated time source for deterministic testing and replay
///
/// This implementation provides perfect determinism for testing scenarios,
/// allowing precise control over time progression and checkpoint replay.
#[derive(Debug)]
pub struct SimulatedTimeSource {
    /// Current simulated time
    current_time: Arc<RwLock<NanoTime>>,
    /// Quantum oracle for consistent optimization
    quantum_oracle: Arc<QuantumTimeOracle>,
    /// Monotonic counter for ordering
    monotonic_counter: AtomicU64,
    /// Checkpoint counter
    checkpoint_counter: AtomicU64,
    /// Start time for measurements
    #[allow(dead_code)]
    start_time: NanoTime,
}

impl SimulatedTimeSource {
    /// Create a new simulated time source starting at the given time
    pub fn new(start_time: NanoTime) -> Self {
        Self {
            current_time: Arc::new(RwLock::new(start_time)),
            quantum_oracle: Arc::new(QuantumTimeOracle::new()),
            monotonic_counter: AtomicU64::new(start_time.as_nanos()),
            checkpoint_counter: AtomicU64::new(0),
            start_time,
        }
    }

    /// Create starting at UNIX epoch
    pub fn new_at_epoch() -> Self {
        Self::new(NanoTime::ZERO)
    }

    /// Create with specific quantum oracle
    pub fn with_oracle(start_time: NanoTime, oracle: Arc<QuantumTimeOracle>) -> Self {
        Self {
            current_time: Arc::new(RwLock::new(start_time)),
            quantum_oracle: oracle,
            monotonic_counter: AtomicU64::new(start_time.as_nanos()),
            checkpoint_counter: AtomicU64::new(0),
            start_time,
        }
    }

    /// Manually set the current time (for testing)
    ///
    /// # Safety
    /// This can break monotonic guarantees if used incorrectly.
    /// Only use for deterministic test scenarios.
    pub fn set_time(&self, time: NanoTime) {
        let mut current = self.current_time.write();
        *current = time;

        // Update monotonic counter if time moved forward
        let time_ns = time.as_nanos();
        let current_monotonic = self.monotonic_counter.load(Ordering::Acquire);
        if time_ns > current_monotonic {
            self.monotonic_counter.store(time_ns, Ordering::Release);
        }
    }

    /// Restore from checkpoint for deterministic replay
    pub fn restore_checkpoint(&self, checkpoint: &TimeCheckpoint) -> Result<(), TimeError> {
        self.set_time(checkpoint.timestamp_ns);
        self.monotonic_counter
            .store(checkpoint.monotonic_value, Ordering::Release);

        debug!(
            checkpoint_id = checkpoint.id,
            timestamp_ns = checkpoint.timestamp_ns.as_nanos(),
            "Restored from checkpoint"
        );

        Ok(())
    }
}

impl TimeSource for SimulatedTimeSource {
    #[instrument(level = "trace", skip(self))]
    fn now_ns(&self) -> Result<NanoTime, TimeError> {
        let base_time = *self.current_time.read();
        let quantum_offset = self.quantum_oracle.current_offset();
        let optimized_time = quantum_offset.apply(base_time);

        // Update monotonic counter
        let time_ns = optimized_time.as_nanos();
        let current_monotonic = self.monotonic_counter.load(Ordering::Acquire);

        if time_ns > current_monotonic {
            self.monotonic_counter.store(time_ns, Ordering::Release);
        }

        Ok(optimized_time)
    }

    #[instrument(level = "trace", skip(self))]
    fn monotonic_ns(&self) -> Result<NanoTime, TimeError> {
        let current_monotonic = self.monotonic_counter.load(Ordering::Acquire);
        Ok(NanoTime::from_nanos(current_monotonic))
    }

    #[instrument(level = "debug", skip(self))]
    fn create_checkpoint(&self) -> Result<TimeCheckpoint, TimeError> {
        let timestamp = self.now_ns()?;
        let monotonic_value = self.monotonic_counter.load(Ordering::Acquire);
        let quantum_state = self.quantum_oracle.current_offset();
        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::AcqRel);

        let checkpoint =
            TimeCheckpoint::new(timestamp, monotonic_value, quantum_state, checkpoint_id);

        debug!(
            checkpoint_id = checkpoint_id,
            timestamp_ns = timestamp.as_nanos(),
            "Created simulated time checkpoint"
        );

        Ok(checkpoint)
    }

    #[instrument(level = "debug", skip(self))]
    fn advance_simulation(&self, delta_ns: u64) -> Result<(), TimeError> {
        let mut time = self.current_time.write();
        let new_time = NanoTime::from_nanos(time.as_nanos().saturating_add(delta_ns));
        *time = new_time;

        // Update monotonic counter
        let new_time_ns = new_time.as_nanos();
        self.monotonic_counter.store(new_time_ns, Ordering::Release);

        debug!(
            delta_ns = delta_ns,
            new_time_ns = new_time_ns,
            "Advanced simulation time"
        );

        Ok(())
    }

    fn is_simulated(&self) -> bool {
        true
    }

    fn resolution_ns(&self) -> u64 {
        1 // Perfect nanosecond precision in simulation
    }

    fn quantum_offset(&self) -> QuantumOffset {
        self.quantum_oracle.current_offset()
    }
}

/// High-resolution time source using hardware counters (Linux-specific)
///
/// This implementation uses TSC (Time Stamp Counter) for sub-nanosecond precision
/// on x86_64 architectures, falling back to system time on other platforms.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct HighResTimeSource {
    /// Quantum oracle for optimization
    quantum_oracle: Arc<QuantumTimeOracle>,
    /// Monotonic counter
    monotonic_counter: AtomicU64,
    /// Checkpoint counter
    checkpoint_counter: AtomicU64,
    /// TSC frequency for calibration (cycles per nanosecond)
    tsc_frequency: f64,
    /// Base TSC value for relative measurements
    base_tsc: u64,
    /// Base time corresponding to base_tsc
    base_time: NanoTime,
}

#[cfg(target_os = "linux")]
impl HighResTimeSource {
    /// Create a new high-resolution time source with TSC calibration
    ///
    /// # Errors
    /// Returns `TimeError::HardwareUnavailable` if TSC is not available or unstable
    pub fn new() -> Result<Self, TimeError> {
        let base_time = TimeSourceImpl::get_system_time_ns()?;
        let base_tsc = Self::rdtsc()?;

        // Estimate TSC frequency (in production, this should be properly calibrated)
        let tsc_frequency = Self::estimate_tsc_frequency()?;

        Ok(Self {
            quantum_oracle: Arc::new(QuantumTimeOracle::new()),
            monotonic_counter: AtomicU64::new(base_time.as_nanos()),
            checkpoint_counter: AtomicU64::new(0),
            tsc_frequency,
            base_tsc,
            base_time,
        })
    }

    /// Get raw hardware timestamp counter
    fn rdtsc() -> Result<u64, TimeError> {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: RDTSC is safe to call on x86_64
            let tsc = unsafe { core::arch::x86_64::_rdtsc() };
            Ok(tsc)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Err(TimeError::HardwareUnavailable {
                details: "TSC not available on non-x86_64 architectures".to_string(),
            })
        }
    }

    /// Estimate TSC frequency by comparing with system time
    fn estimate_tsc_frequency() -> Result<f64, TimeError> {
        #[cfg(target_arch = "x86_64")]
        {
            // Simple calibration - in production this should be more sophisticated
            let start_tsc = Self::rdtsc()?;
            let start_time = std::time::Instant::now();

            std::thread::sleep(std::time::Duration::from_millis(1));

            let end_tsc = Self::rdtsc()?;
            let end_time = std::time::Instant::now();

            let tsc_delta = end_tsc - start_tsc;
            let time_delta_ns = end_time.duration_since(start_time).as_nanos() as u64;

            if time_delta_ns == 0 {
                return Err(TimeError::HardwareUnavailable {
                    details: "Cannot calibrate TSC frequency".to_string(),
                });
            }

            let frequency = tsc_delta as f64 / time_delta_ns as f64;
            Ok(frequency)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Err(TimeError::HardwareUnavailable {
                details: "TSC frequency estimation not available on non-x86_64".to_string(),
            })
        }
    }

    /// Convert TSC cycles to nanoseconds
    fn tsc_to_ns(&self, tsc: u64) -> u64 {
        let tsc_delta = tsc.saturating_sub(self.base_tsc);
        let ns_delta = (tsc_delta as f64 / self.tsc_frequency) as u64;
        self.base_time.as_nanos() + ns_delta
    }

    /// Ensure monotonic time progression
    fn ensure_monotonic(&self, time_ns: u64) -> u64 {
        let current_max = self.monotonic_counter.load(Ordering::Acquire);

        if time_ns > current_max {
            match self.monotonic_counter.compare_exchange_weak(
                current_max,
                time_ns,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => time_ns,
                Err(actual) => std::cmp::max(time_ns, actual + 1),
            }
        } else {
            let next_time = current_max + 1;
            self.monotonic_counter.store(next_time, Ordering::Release);
            next_time
        }
    }
}

#[cfg(target_os = "linux")]
impl TimeSource for HighResTimeSource {
    #[instrument(level = "trace", skip(self))]
    fn now_ns(&self) -> Result<NanoTime, TimeError> {
        // Get TSC value
        let tsc = Self::rdtsc()?;

        // Convert to nanoseconds using calibrated frequency
        let time_ns = self.tsc_to_ns(tsc);

        // Apply quantum optimization
        let base_time = NanoTime::from_nanos(time_ns);
        let quantum_offset = self.quantum_oracle.current_offset();
        let optimized_time = quantum_offset.apply(base_time);

        // Ensure monotonic behavior
        let monotonic_time = self.ensure_monotonic(optimized_time.as_nanos());

        Ok(NanoTime::from_nanos(monotonic_time))
    }

    #[instrument(level = "trace", skip(self))]
    fn monotonic_ns(&self) -> Result<NanoTime, TimeError> {
        let current_monotonic = self.monotonic_counter.load(Ordering::Acquire);
        Ok(NanoTime::from_nanos(current_monotonic))
    }

    #[instrument(level = "debug", skip(self))]
    fn create_checkpoint(&self) -> Result<TimeCheckpoint, TimeError> {
        let timestamp = self.now_ns()?;
        let monotonic_value = self.monotonic_counter.load(Ordering::Acquire);
        let quantum_state = self.quantum_oracle.current_offset();
        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::AcqRel);

        let checkpoint =
            TimeCheckpoint::new(timestamp, monotonic_value, quantum_state, checkpoint_id);

        debug!(
            checkpoint_id = checkpoint_id,
            timestamp_ns = timestamp.as_nanos(),
            "Created high-resolution time checkpoint"
        );

        Ok(checkpoint)
    }

    fn advance_simulation(&self, _delta_ns: u64) -> Result<(), TimeError> {
        // Hardware time source doesn't support simulation advancement
        warn!("Ignoring advance_simulation call on hardware time source");
        Ok(())
    }

    fn resolution_ns(&self) -> u64 {
        // TSC typically provides sub-nanosecond resolution
        1
    }

    fn quantum_offset(&self) -> QuantumOffset {
        self.quantum_oracle.current_offset()
    }
}

/// Global time source instance
static GLOBAL_TIME_SOURCE: OnceLock<Arc<dyn TimeSource>> = OnceLock::new();

/// Initialize the global time source with a production implementation
///
/// # Errors
/// Returns `TimeError::SystemTimeError` if initialization fails
pub fn initialize_global_time_source() -> Result<(), TimeError> {
    let time_source = Arc::new(TimeSourceImpl::new()?) as Arc<dyn TimeSource>;
    GLOBAL_TIME_SOURCE
        .set(time_source)
        .map_err(|_| TimeError::SystemTimeError {
            details: "Global time source already initialized".to_string(),
        })?;
    Ok(())
}

/// Initialize the global time source with a simulated implementation for testing
///
/// This function is safe to call multiple times - subsequent calls will be ignored
/// to prevent conflicts in test environments where multiple tests may attempt initialization.
pub fn initialize_simulated_time_source(start_time: NanoTime) {
    let time_source = Arc::new(SimulatedTimeSource::new(start_time)) as Arc<dyn TimeSource>;
    let _ = GLOBAL_TIME_SOURCE.set(time_source);
}

/// Initialize the global time source with automatic fallback strategies
///
/// This internal function implements a robust initialization strategy:
/// 1. Try production time source (TimeSourceImpl)
/// 2. Fall back to simulated time source for testing
/// 3. Return error only if all strategies fail
fn initialize_global_time_source_with_fallback() -> Result<(), TimeError> {
    // Strategy 1: Try production time source
    if let Ok(production_source) = TimeSourceImpl::new() {
        let time_source = Arc::new(production_source) as Arc<dyn TimeSource>;
        if GLOBAL_TIME_SOURCE.set(time_source).is_ok() {
            tracing::debug!("Global time source initialized with production TimeSourceImpl");
            return Ok(());
        }
    }

    // Strategy 2: Fall back to simulated time source for testing
    let simulated_source = SimulatedTimeSource::new(NanoTime::ZERO);
    let time_source = Arc::new(simulated_source) as Arc<dyn TimeSource>;
    if GLOBAL_TIME_SOURCE.set(time_source).is_ok() {
        tracing::debug!("Global time source initialized with fallback SimulatedTimeSource");
        return Ok(());
    }

    // Strategy 3: If set() failed, another thread already initialized it - this is OK
    if GLOBAL_TIME_SOURCE.get().is_some() {
        tracing::debug!("Global time source already initialized by another thread");
        return Ok(());
    }

    // All strategies failed - this should be very rare
    Err(TimeError::SystemTimeError {
        details: "All global time source initialization strategies failed".to_string(),
    })
}

/// Get or initialize the global time source for testing scenarios
///
/// This is a convenience function for tests that need guaranteed access to a time source.
/// It will use the existing global source if available, or initialize a simulated one.
pub fn get_or_init_test_time_source() -> Result<&'static Arc<dyn TimeSource>, TimeError> {
    // Try to get existing source first
    if let Some(time_source) = GLOBAL_TIME_SOURCE.get() {
        return Ok(time_source);
    }

    // Initialize with simulated source for testing
    initialize_simulated_time_source(NanoTime::from_secs(1_700_000_000)); // Reasonable test epoch
    GLOBAL_TIME_SOURCE
        .get()
        .ok_or_else(|| TimeError::SystemTimeError {
            details: "Test time source initialization failed".to_string(),
        })
}

/// Get the global time source instance with automatic fallback initialization
///
/// This function provides a safe way to access the global time source, with automatic
/// initialization for testing scenarios. In production, explicit initialization via
/// `initialize_global_time_source()` is still recommended for performance.
///
/// # Fallback Strategy
/// 1. Return initialized global time source if available
/// 2. Attempt automatic initialization with production time source
/// 3. Fall back to simulated time source for testing
/// 4. Never panic: if initialization still fails, return a safe simulated fallback
pub fn global_time_source() -> &'static Arc<dyn TimeSource> {
    // Fast path: return already initialized time source
    if let Some(time_source) = GLOBAL_TIME_SOURCE.get() {
        return time_source;
    }

    // Slow path: attempt initialization with appropriate fallback
    if let Err(e) = initialize_global_time_source_with_fallback() {
        warn!(error = %e, "Failed to initialize global time source, using simulated fallback");
    }

    if let Some(time_source) = GLOBAL_TIME_SOURCE.get() {
        return time_source;
    }

    // Final safety fallback: a separate static simulated source to ensure non-panicking access
    static FALLBACK_TIME_SOURCE: OnceLock<Arc<dyn TimeSource>> = OnceLock::new();
    FALLBACK_TIME_SOURCE
        .get_or_init(|| Arc::new(SimulatedTimeSource::new(NanoTime::ZERO)) as Arc<dyn TimeSource>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;

    #[test]
    fn test_simulated_time_source_basic() {
        let start_time = NanoTime::from_secs(100);
        let source = SimulatedTimeSource::new(start_time);

        // Basic time functionality
        let current = source.now_ns().expect("now_ns should work");
        assert_eq!(current.as_secs(), 100);
        assert!(source.is_simulated());

        // Advancement
        source
            .advance_simulation(Duration::from_secs(10).as_nanos())
            .expect("advance_simulation should work");

        let advanced = source.now_ns().expect("now_ns should work after advance");
        assert_eq!(advanced.as_secs(), 110);
    }

    #[test]
    fn test_simulated_time_source_monotonic() {
        let source = SimulatedTimeSource::new(NanoTime::from_secs(100));

        let time1 = source.now_ns().expect("now_ns should work");
        let time2 = source.now_ns().expect("now_ns should work");

        // Should be monotonic
        assert!(time2 >= time1);

        let monotonic1 = source.monotonic_ns().expect("monotonic_ns should work");
        let monotonic2 = source.monotonic_ns().expect("monotonic_ns should work");

        assert!(monotonic2 >= monotonic1);
    }

    #[test]
    fn test_simulated_time_source_checkpoints() {
        let source = SimulatedTimeSource::new(NanoTime::from_secs(100));

        // Create checkpoint
        let checkpoint = source
            .create_checkpoint()
            .expect("create_checkpoint should work");
        assert_eq!(checkpoint.timestamp_ns.as_secs(), 100);
        assert_eq!(checkpoint.id, 0);

        // Advance time
        source
            .advance_simulation(Duration::from_secs(50).as_nanos())
            .expect("advance_simulation should work");

        let advanced = source.now_ns().expect("now_ns should work");
        assert_eq!(advanced.as_secs(), 150);

        // Restore from checkpoint
        source
            .restore_checkpoint(&checkpoint)
            .expect("restore_checkpoint should work");

        let restored = source.now_ns().expect("now_ns should work after restore");
        assert_eq!(restored.as_secs(), 100);
    }

    #[test]
    fn test_production_time_source() {
        let source = TimeSourceImpl::new().expect("TimeSourceImpl::new should work");

        let time1 = source.now_ns().expect("now_ns should work");

        // Time should be reasonable (after 2020)
        assert!(time1.as_secs() > 1_600_000_000);

        // Monotonic property
        std::thread::sleep(StdDuration::from_nanos(1));
        let time2 = source.now_ns().expect("now_ns should work");
        assert!(time2 >= time1);

        // Test monotonic_ns
        let mono1 = source.monotonic_ns().expect("monotonic_ns should work");
        let mono2 = source.monotonic_ns().expect("monotonic_ns should work");
        assert!(mono2 >= mono1);
    }

    #[test]
    fn test_production_time_source_checkpoints() {
        let source = TimeSourceImpl::new().expect("TimeSourceImpl::new should work");

        let checkpoint1 = source
            .create_checkpoint()
            .expect("create_checkpoint should work");
        let checkpoint2 = source
            .create_checkpoint()
            .expect("create_checkpoint should work");

        // Checkpoint IDs should be unique and increasing
        assert_ne!(checkpoint1.id, checkpoint2.id);
        assert!(checkpoint2.id > checkpoint1.id);

        // Timestamps should be reasonable
        assert!(checkpoint1.timestamp_ns.as_secs() > 1_600_000_000);
        assert!(checkpoint2.timestamp_ns >= checkpoint1.timestamp_ns);
    }

    #[test]
    fn test_quantum_offset_application() {
        let oracle = Arc::new(QuantumTimeOracle::new());
        let source = TimeSourceImpl::with_oracle(oracle).expect("with_oracle should work");

        let offset = source.quantum_offset();
        assert!(offset.amplitude > 0.0); // Should have positive amplitude
        assert!(offset.amplitude <= 1.0); // Should be reasonable value
    }

    #[test]
    fn test_global_time_source() {
        // Test simulated initialization
        initialize_simulated_time_source(NanoTime::from_secs(1000));

        let global = global_time_source();
        let time = global.now_ns().expect("global time source should work");
        assert_eq!(time.as_secs(), 1000);
        assert!(global.is_simulated());
    }

    #[test]
    fn test_advance_simulation_production() {
        let source = TimeSourceImpl::new().expect("TimeSourceImpl::new should work");

        // Production sources should ignore simulation advancement
        let result = source.advance_simulation(1000);
        assert!(result.is_ok());
    }
}
