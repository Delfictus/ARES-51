//! Core data types used throughout the ARES ChronoSynclastic Fabric.

use serde::{Deserialize, Serialize};

/// Component identifier for CSF components
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(String);

impl ComponentId {
    /// Create a new component ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ComponentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Nanosecond precision time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NanoTime(u64);

impl NanoTime {
    /// Create from nanoseconds
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Get nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Create from seconds
    pub fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Create from milliseconds
    pub fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }
}

/// Task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(String);

impl TaskId {
    /// Create a new task ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random task ID
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

/// Precision level for computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Low precision for fast computations
    Low,
    /// Standard precision for most applications
    Standard,
    /// High precision for critical computations
    High,
    /// Maximum precision for research applications
    Maximum,
}

/// Get current hardware timestamp in nanoseconds
pub fn hardware_timestamp() -> NanoTime {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    NanoTime::from_nanos(duration.as_nanos() as u64)
}

/// Get current hardware timestamp using enterprise TimeSource
#[cfg(feature = "enterprise")]
pub fn hardware_timestamp_enterprise(time_source: &dyn crate::ports::TimeSource) -> NanoTime {
    time_source.now_ns().unwrap_or_else(|_| NanoTime::from_nanos(0))
}

/// The priority of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Critical priority - system-level operations
    Critical,
    /// High priority - time-sensitive operations
    High,
    /// Normal priority - standard operations
    Normal,
    /// Low priority - background operations
    Low,
    /// Idle priority - lowest priority background tasks
    Idle,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Phase value for quantum state representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    /// Phase in radians [0, 2π)
    pub value: f64,
}

impl Phase {
    /// Create a new phase value, normalizing to [0, 2π)
    pub fn new(value: f64) -> Self {
        Self {
            value: value % (2.0 * std::f64::consts::PI),
        }
    }

    /// Create a phase from degrees
    pub fn from_degrees(degrees: f64) -> Self {
        Self::new(degrees.to_radians())
    }

    /// Get the phase value in degrees
    pub fn to_degrees(&self) -> f64 {
        self.value.to_degrees()
    }

    /// Get the normalized phase value [0, 2π)
    pub fn normalized(&self) -> f64 {
        self.value
    }

    /// Create a zero phase
    pub fn zero() -> Self {
        Self { value: 0.0 }
    }

    /// Create a π phase
    pub fn pi() -> Self {
        Self { value: std::f64::consts::PI }
    }

    /// Add two phases with proper modular arithmetic
    pub fn add(&self, other: &Phase) -> Phase {
        Phase::new(self.value + other.value)
    }

    /// Subtract two phases with proper modular arithmetic
    pub fn sub(&self, other: &Phase) -> Phase {
        Phase::new(self.value - other.value)
    }

    /// Calculate phase difference (shortest angular distance)
    pub fn difference(&self, other: &Phase) -> f64 {
        let diff = (self.value - other.value) % (2.0 * std::f64::consts::PI);
        if diff > std::f64::consts::PI {
            diff - 2.0 * std::f64::consts::PI
        } else if diff < -std::f64::consts::PI {
            diff + 2.0 * std::f64::consts::PI
        } else {
            diff
        }
    }
}

impl std::ops::Add for Phase {
    type Output = Phase;

    fn add(self, other: Phase) -> Phase {
        Phase::new(self.value + other.value)
    }
}

impl std::ops::Sub for Phase {
    type Output = Phase;

    fn sub(self, other: Phase) -> Phase {
        Phase::new(self.value - other.value)
    }
}

/// Timestamp representation for temporal measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp {
    /// Nanoseconds since Unix epoch
    pub nanos: u64,
}

impl Timestamp {
    /// Get current timestamp
    pub fn now() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self { nanos }
    }

    /// Create timestamp from nanoseconds
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Create timestamp from milliseconds
    pub fn from_millis(millis: u64) -> Self {
        Self { nanos: millis * 1_000_000 }
    }

    /// Create timestamp from seconds
    pub fn from_secs(secs: u64) -> Self {
        Self { nanos: secs * 1_000_000_000 }
    }

    /// Get duration since another timestamp
    pub fn duration_since(&self, earlier: Timestamp) -> std::time::Duration {
        std::time::Duration::from_nanos(self.nanos.saturating_sub(earlier.nanos))
    }

    /// Convert to standard Duration since Unix epoch
    pub fn as_duration(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.nanos)
    }

    /// Convert to SystemTime
    pub fn as_system_time(&self) -> std::time::SystemTime {
        std::time::UNIX_EPOCH + self.as_duration()
    }
}

impl From<std::time::SystemTime> for Timestamp {
    fn from(time: std::time::SystemTime) -> Self {
        let nanos = time
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self { nanos }
    }
}

impl From<NanoTime> for Timestamp {
    fn from(nano_time: NanoTime) -> Self {
        Self {
            nanos: nano_time.as_nanos(),
        }
    }
}

impl From<Timestamp> for NanoTime {
    fn from(timestamp: Timestamp) -> Self {
        NanoTime::from_nanos(timestamp.nanos)
    }
}

/// Phase state combining phase value with temporal and coherence information
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseState {
    /// Phase value in radians
    pub phase: Phase,
    /// Timestamp when phase was measured
    pub timestamp: Timestamp,
    /// Coherence factor [0, 1] where 1 = fully coherent
    pub coherence: f64,
}

impl PhaseState {
    /// Create a new phase state
    pub fn new(phase: Phase, timestamp: Timestamp, coherence: f64) -> Self {
        Self {
            phase,
            timestamp,
            coherence: coherence.clamp(0.0, 1.0),
        }
    }

    /// Create a phase state with current timestamp
    pub fn now(phase: Phase, coherence: f64) -> Self {
        Self::new(phase, Timestamp::now(), coherence)
    }

    /// Check if the phase state is coherent (coherence > threshold)
    pub fn is_coherent(&self, threshold: f64) -> bool {
        self.coherence > threshold
    }

    /// Calculate phase evolution over time with decoherence
    pub fn evolve(&self, target_time: Timestamp, frequency: f64, decoherence_rate: f64) -> PhaseState {
        let dt = target_time.duration_since(self.timestamp).as_secs_f64();
        let new_phase = Phase::new(self.phase.value + 2.0 * std::f64::consts::PI * frequency * dt);
        let new_coherence = self.coherence * (-decoherence_rate * dt).exp();
        PhaseState::new(new_phase, target_time, new_coherence)
    }
}

/// Quantum state representation for CSF computations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumState {
    /// Complex amplitude coefficients
    pub amplitudes: Vec<num_complex::Complex64>,
    /// Phase information
    pub phase: Phase,
    /// Coherence measures
    pub coherence: f64,
    /// Timestamp of state
    pub timestamp: Timestamp,
}

impl QuantumState {
    /// Create a new quantum state
    pub fn new(amplitudes: Vec<num_complex::Complex64>, phase: Phase, coherence: f64) -> Self {
        Self {
            amplitudes,
            phase,
            coherence: coherence.clamp(0.0, 1.0),
            timestamp: Timestamp::now(),
        }
    }

    /// Get the dimension of the quantum state
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// Calculate the probability of measuring a specific state
    pub fn probability(&self, index: usize) -> Option<f64> {
        self.amplitudes.get(index).map(|amp| amp.norm_sqr())
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm: f64 = self.amplitudes.iter().map(|amp| amp.norm_sqr()).sum();
        if norm > 0.0 {
            let norm_sqrt = norm.sqrt();
            for amp in &mut self.amplitudes {
                *amp /= norm_sqrt;
            }
        }
    }

    /// Calculate fidelity with another quantum state
    pub fn fidelity(&self, other: &QuantumState) -> f64 {
        if self.amplitudes.len() != other.amplitudes.len() {
            return 0.0;
        }

        let overlap: num_complex::Complex64 = self
            .amplitudes
            .iter()
            .zip(&other.amplitudes)
            .map(|(a, b)| a.conj() * b)
            .sum();

        overlap.norm_sqr()
    }
}

/// Configuration for computational precision requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputePrecision {
    /// Single precision (32-bit)
    F32,
    /// Double precision (64-bit) 
    F64,
    /// Extended precision (128-bit where available)
    F128,
    /// Arbitrary precision
    Arbitrary,
}

impl Default for ComputePrecision {
    fn default() -> Self {
        Self::F64
    }
}

/// Memory layout optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Optimal layout for current hardware
    Auto,
}

impl Default for MemoryLayout {
    fn default() -> Self {
        Self::Auto
    }
}

/// Hardware acceleration preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorType {
    /// CPU-only computation
    Cpu,
    /// GPU acceleration preferred
    Gpu,
    /// FPGA acceleration
    Fpga,
    /// Quantum processing unit
    Qpu,
    /// Automatic selection
    Auto,
}

impl Default for AcceleratorType {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_normalization() {
        let phase = Phase::new(3.0 * std::f64::consts::PI);
        assert!((phase.value - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_phase_arithmetic() {
        let phase1 = Phase::new(std::f64::consts::PI / 2.0);
        let phase2 = Phase::new(std::f64::consts::PI / 2.0);
        let sum = phase1 + phase2;
        assert!((sum.value - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_phase_difference() {
        let phase1 = Phase::new(0.1);
        let phase2 = Phase::new(2.0 * std::f64::consts::PI - 0.1);
        let diff = phase1.difference(&phase2);
        assert!((diff - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::now();
        assert!(ts.nanos > 0);
    }

    #[test]
    fn test_phase_state_evolution() {
        let initial_phase = Phase::new(0.0);
        let initial_time = Timestamp::now();
        let state = PhaseState::new(initial_phase, initial_time, 1.0);
        
        let target_time = Timestamp::from_nanos(initial_time.nanos + 1_000_000_000); // +1 second
        let evolved = state.evolve(target_time, 1.0, 0.1); // 1 Hz, 0.1 decoherence rate
        
        assert!(evolved.coherence < state.coherence);
        assert!(evolved.timestamp == target_time);
    }

    #[test]
    fn test_quantum_state_normalization() {
        let amplitudes = vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(1.0, 0.0),
        ];
        let mut state = QuantumState::new(amplitudes, Phase::zero(), 1.0);
        state.normalize();
        
        let norm: f64 = state.amplitudes.iter().map(|amp| amp.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_fidelity() {
        let amplitudes1 = vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
        ];
        let amplitudes2 = vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
        ];
        
        let state1 = QuantumState::new(amplitudes1, Phase::zero(), 1.0);
        let state2 = QuantumState::new(amplitudes2, Phase::zero(), 1.0);
        
        let fidelity = state1.fidelity(&state2);
        assert!((fidelity - 1.0).abs() < 1e-10);
    }
}