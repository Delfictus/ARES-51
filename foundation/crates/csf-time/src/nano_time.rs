pub use csf_shared_types::NanoTime;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Duration between two time points
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Duration(u64);

impl Duration {
    /// Create duration from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create duration from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }

    /// Create duration from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }

    /// Create duration from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Get nanoseconds
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Get microseconds
    pub const fn as_micros(self) -> u64 {
        self.0 / 1_000
    }

    /// Get milliseconds
    pub const fn as_millis(self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get seconds
    pub const fn as_secs(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Zero duration
    pub const ZERO: Self = Self(0);

    /// Maximum duration
    pub const MAX: Self = Self(u64::MAX);

    /// One nanosecond
    pub const NANOSECOND: Self = Self(1);

    /// One microsecond
    pub const MICROSECOND: Self = Self(1_000);

    /// One millisecond
    pub const MILLISECOND: Self = Self(1_000_000);

    /// One second
    pub const SECOND: Self = Self(1_000_000_000);
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 1_000 {
            write!(f, "{}ns", self.0)
        } else if self.0 < 1_000_000 {
            write!(f, "{:.1}µs", self.0 as f64 / 1_000.0)
        } else if self.0 < 1_000_000_000 {
            write!(f, "{:.1}ms", self.0 as f64 / 1_000_000.0)
        } else {
            write!(f, "{:.3}s", self.0 as f64 / 1_000_000_000.0)
        }
    }
}

impl Add for Duration {
    type Output = Duration;

    fn add(self, other: Duration) -> Duration {
        Duration(self.0.saturating_add(other.0))
    }
}

impl Sub for Duration {
    type Output = Duration;

    fn sub(self, other: Duration) -> Duration {
        Duration(self.0.saturating_sub(other.0))
    }
}

impl AddAssign for Duration {
    fn add_assign(&mut self, other: Duration) {
        self.0 = self.0.saturating_add(other.0);
    }
}

impl SubAssign for Duration {
    fn sub_assign(&mut self, other: Duration) {
        self.0 = self.0.saturating_sub(other.0);
    }
}

impl std::iter::Sum for Duration {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Duration::ZERO, |acc, duration| acc + duration)
    }
}

impl<'a> std::iter::Sum<&'a Duration> for Duration {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Duration::ZERO, |acc, duration| acc + *duration)
    }
}

// NanoTime arithmetic is now defined in csf_core. Do not reimplement here.

/// Quantum offset for optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantumOffset {
    /// Phase offset in nanoseconds
    pub phase: f64,
    /// Amplitude scaling factor
    pub amplitude: f64,
    /// Frequency adjustment
    pub frequency: f64,
}

impl QuantumOffset {
    /// Create a new quantum offset
    pub const fn new(phase: f64, amplitude: f64, frequency: f64) -> Self {
        Self {
            phase,
            amplitude,
            frequency,
        }
    }

    /// Zero offset (no quantum adjustment)
    pub const ZERO: Self = Self {
        phase: 0.0,
        amplitude: 1.0,
        frequency: 1.0,
    };

    /// Apply quantum offset to a time value
    pub fn apply(&self, base_time: NanoTime) -> NanoTime {
        let base_ns = base_time.as_nanos() as f64;
        let quantum_adjustment =
            self.amplitude * (self.frequency * base_ns / 1_000_000_000.0 + self.phase).sin();

        let adjusted_ns = base_ns + quantum_adjustment;
        NanoTime::from_nanos(adjusted_ns.max(0.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_time_creation() {
        let time = NanoTime::from_secs(42);
        assert_eq!(time.as_secs(), 42);
        assert_eq!(time.as_nanos(), 42_000_000_000);
    }

    #[test]
    fn test_duration_arithmetic() {
        let d1 = Duration::from_millis(100);
        let d2 = Duration::from_millis(50);
        let sum = d1 + d2;
        assert_eq!(sum.as_millis(), 150);
    }

    #[test]
    fn test_time_duration_ops() {
        let time = NanoTime::from_secs(10);
        let duration = Duration::from_secs(5);

        let later = NanoTime::from_nanos(time.as_nanos().saturating_add(duration.as_nanos()));
        assert_eq!(later.as_secs(), 15);

        let earlier = NanoTime::from_nanos(time.as_nanos().saturating_sub(duration.as_nanos()));
        assert_eq!(earlier.as_secs(), 5);
    }

    #[test]
    fn test_quantum_offset() {
        let offset = QuantumOffset::new(0.0, 1.0, 1.0);
        let base_time = NanoTime::from_secs(1);
        let adjusted = offset.apply(base_time);
        // Should be close to original time with small quantum adjustment
        assert!(adjusted.as_secs() <= 2);
    }

    #[test]
    fn test_display_format() {
        let time = NanoTime::from_nanos(1_234_567_890);
        assert_eq!(time.to_string(), "1.234567890s");

        let duration = Duration::from_micros(500);
        assert_eq!(duration.to_string(), "500.0µs");
    }
}
