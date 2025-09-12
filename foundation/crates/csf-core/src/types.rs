#![allow(clippy::pedantic)]
//! Core data types used throughout the ARES Chronosynclastic Fabric.

use serde::{Deserialize, Serialize};

// Re-export shared types to avoid breaking existing code
pub use csf_shared_types::{ComponentId, NanoTime, PrecisionLevel, TaskId};
use csf_time::TimeSource;

/// Get current hardware timestamp in nanoseconds
pub fn hardware_timestamp() -> NanoTime {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    NanoTime::from_nanos(duration.as_nanos() as u64)
}

/// Get current hardware timestamp using enterprise TimeSource
pub fn hardware_timestamp_enterprise(time_source: &dyn TimeSource) -> NanoTime {
    time_source.now_ns().unwrap_or_else(|_| NanoTime::from_nanos(0))
}

/// The priority of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// High priority.
    High,
    /// Normal priority.
    Normal,
    /// Low priority.
    Low,
}

// NanoTime is now re-exported from csf-shared-types above

/// Phase value for quantum state representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub value: f64, // Phase in radians [0, 2π)
}

impl Phase {
    pub fn new(value: f64) -> Self {
        Self {
            value: value % (2.0 * std::f64::consts::PI),
        }
    }
}

/// Timestamp representation for temporal measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp {
    pub nanos: u64,
}

impl Timestamp {
    pub fn now() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self { nanos }
    }
    
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }
}

/// Phase state combining phase value with temporal and coherence information
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseState {
    pub phase: Phase,
    pub timestamp: Timestamp,
    pub coherence: f64, // Coherence factor [0, 1]
}
