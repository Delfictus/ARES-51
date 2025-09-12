//! Shared types for ARES ChronoFabric system
//!
//! This crate contains common types used across csf-core and csf-time
//! to break circular dependencies and enable proper modular architecture.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};
use uuid::Uuid;

/// Component identifier for distributed system coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(u64);

impl ComponentId {
    /// Create a new ComponentId from a u64
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Predefined component IDs
    /// DRPP - Dynamic Resource Pattern Processing component
    pub const DRPP: Self = Self(1);
    /// ADP - Adaptive Processing component  
    pub const ADP: Self = Self(2);
    /// EGC - Execution Governance Control component
    pub const EGC: Self = Self(3);
    /// EMS - Emotion Modeling System component
    pub const EMS: Self = Self(4);

    /// Create a custom component ID
    pub const fn custom(id: u64) -> Self {
        Self(id)
    }

    /// Generate a random component ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4().as_u128() as u64)
    }

    /// Get the inner ID value
    pub fn inner(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComponentId({})", self.0)
    }
}

/// High-precision time representation for ChronoSynclastic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NanoTime(u64);

impl NanoTime {
    /// Zero time constant
    pub const ZERO: Self = Self(0);

    /// Create a new NanoTime from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create a new NanoTime from microseconds  
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }

    /// Create a new NanoTime from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }

    /// Create a new NanoTime from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Get nanoseconds value
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Get microseconds value
    pub const fn as_micros(self) -> u64 {
        self.0 / 1_000
    }

    /// Get milliseconds value
    pub const fn as_millis(self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get seconds value
    pub const fn as_secs(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Saturating subtraction
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Zero time
    pub fn zero() -> Self {
        Self::ZERO
    }

    /// Current time from system monotonic clock  
    pub fn now() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        Self(now.as_nanos() as u64)
    }

}

impl Add for NanoTime {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for NanoTime {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl From<u64> for NanoTime {
    fn from(nanos: u64) -> Self {
        Self::from_nanos(nanos)
    }
}

impl From<NanoTime> for u64 {
    fn from(time: NanoTime) -> u64 {
        time.as_nanos()
    }
}

impl fmt::Display for NanoTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Common error types for temporal operations
#[derive(Debug, thiserror::Error)]
pub enum SharedError {
    #[error("Invalid time value: {value}")]
    InvalidTime { value: i64 },

    #[error("Arithmetic overflow in temporal calculation")]
    ArithmeticOverflow,

    #[error("System time error: {details}")]
    SystemTimeError { details: String },
}

/// Result type for shared operations
pub type SharedResult<T> = Result<T, SharedError>;

/// A unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(u64);

impl TaskId {
    /// Creates a new, random `TaskId`.
    pub fn new() -> Self {
        // This is a placeholder implementation. A real implementation might use a
        // counter, a UUID, or a combination of node ID and a local counter.
        Self(rand::random())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Precision level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Ultra-high precision (femtosecond level, 10^-15 seconds)
    Femtosecond,
    /// High precision (picosecond level, 10^-12 seconds)  
    Picosecond,
    /// Standard precision (nanosecond level, 10^-9 seconds)
    Nanosecond,
    /// Low precision (microsecond level, 10^-6 seconds)
    Microsecond,
}

/// Packet type enumeration for different kinds of data packets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PacketType {
    /// Control packets for system management
    Control,
    /// Data packets containing payload
    Data,
    /// Event packets for notifications
    Event,
    /// Stream packets for continuous data
    Stream,
}

impl PrecisionLevel {
    /// Get the epsilon value for this precision level
    pub const fn epsilon(self) -> f64 {
        match self {
            PrecisionLevel::Femtosecond => 1e-15,
            PrecisionLevel::Picosecond => 1e-12,
            PrecisionLevel::Nanosecond => 1e-9,
            PrecisionLevel::Microsecond => 1e-6,
        }
    }
}
