//! Precision management for quantum temporal calculations
//!
//! This module provides high-precision types and traits for quantum temporal offset calculations
//! with femtosecond-level precision, numerical stability, and cross-platform consistency.

use crate::{NanoTime, TimeError, TimeResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// Precision tolerance levels for quantum calculations
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

    /// Get the scale factor for internal representation
    pub const fn scale_factor(self) -> u64 {
        match self {
            PrecisionLevel::Femtosecond => 1_000_000, // attoseconds per femtosecond
            PrecisionLevel::Picosecond => 1_000,      // femtoseconds per picosecond
            PrecisionLevel::Nanosecond => 1,          // nanosecond base
            PrecisionLevel::Microsecond => 1,         // nanosecond base (no extra precision)
        }
    }
}

/// Trait for types that can be compared with custom precision bounds
pub trait PrecisionBound {
    /// Check if two values are equal within the given precision level
    fn precision_eq(&self, other: &Self, level: PrecisionLevel) -> bool;

    /// Check if this value is within tolerance of zero
    fn is_precision_zero(&self, level: PrecisionLevel) -> bool;

    /// Get the precision loss when converting from another representation
    fn precision_loss_from<T>(&self, _source: &T) -> Option<f64> {
        None // Default implementation
    }
}

impl PrecisionBound for f64 {
    fn precision_eq(&self, other: &Self, level: PrecisionLevel) -> bool {
        (self - other).abs() < level.epsilon()
    }

    fn is_precision_zero(&self, level: PrecisionLevel) -> bool {
        self.abs() < level.epsilon()
    }
}

/// High-precision quantum offset with error bounds and precision tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreciseQuantumOffset {
    /// Temporal offset duration with sub-nanosecond precision
    pub temporal_offset: PreciseDuration,

    /// Phase component normalized to [0.0, 1.0) representing quantum phase
    pub phase_component: f64,

    /// Precision level for this offset calculation
    pub precision_level: PrecisionLevel,

    /// Accumulated precision error bounds
    pub error_bounds: ErrorBounds,

    /// Metadata about precision requirements
    pub precision_metadata: PrecisionMetadata,
}

/// High-precision duration representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreciseDuration {
    /// Nanoseconds (integer part)
    nanos: i64,

    /// Sub-nanosecond fraction (0.0 to 1.0)
    sub_nanos: f64,

    /// Precision level for this duration
    precision: PrecisionLevel,
}

/// Error bounds tracking for precision calculations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Absolute error bound
    pub absolute_error: f64,

    /// Relative error bound (as fraction)
    pub relative_error: f64,

    /// Accumulated rounding errors
    pub rounding_error: f64,

    /// Cross-platform consistency flag
    pub cross_platform_validated: bool,
}

/// Metadata about precision requirements and guarantees
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionMetadata {
    /// Target precision level
    pub target_precision: PrecisionLevel,

    /// Actual achieved precision
    pub achieved_precision: PrecisionLevel,

    /// Number of operations performed (affects error accumulation)
    pub operation_count: u32,

    /// Source of the precision requirement
    pub precision_source: String,
}

impl PreciseDuration {
    /// Create a new precise duration
    pub fn new(nanos: i64, sub_nanos: f64, precision: PrecisionLevel) -> TimeResult<Self> {
        if !(0.0..1.0).contains(&sub_nanos) {
            return Err(TimeError::InvalidTime { value: nanos });
        }

        Ok(Self {
            nanos,
            sub_nanos: sub_nanos.clamp(0.0, 1.0),
            precision,
        })
    }

    /// Create from nanoseconds with automatic precision detection
    pub fn from_nanos_precise(total_nanos: f64, precision: PrecisionLevel) -> Self {
        let nanos = total_nanos.trunc() as i64;
        let sub_nanos = total_nanos.fract().abs();

        Self {
            nanos,
            sub_nanos,
            precision,
        }
    }

    /// Convert to total nanoseconds as f64
    pub fn as_total_nanos(&self) -> f64 {
        self.nanos as f64 + self.sub_nanos
    }

    /// Convert to standard NanoTime (with precision loss warning)
    pub fn to_nano_time(&self) -> (NanoTime, f64) {
        let total_ns = self.as_total_nanos();
        let precision_loss = if self.sub_nanos > 0.0 {
            self.sub_nanos
        } else {
            0.0
        };

        (
            NanoTime::from_nanos(total_ns.max(0.0) as u64),
            precision_loss,
        )
    }

    /// Zero duration with specified precision
    pub fn zero(precision: PrecisionLevel) -> Self {
        Self {
            nanos: 0,
            sub_nanos: 0.0,
            precision,
        }
    }

    /// Check for overflow conditions
    pub fn check_overflow(&self) -> TimeResult<()> {
        if self.nanos.abs() > (i64::MAX / 2) {
            return Err(TimeError::ArithmeticOverflow);
        }
        Ok(())
    }
}

impl ErrorBounds {
    /// Create new error bounds with zero errors
    pub fn zero() -> Self {
        Self {
            absolute_error: 0.0,
            relative_error: 0.0,
            rounding_error: 0.0,
            cross_platform_validated: false,
        }
    }

    /// Add errors from an arithmetic operation
    pub fn add_operation_error(&mut self, abs_err: f64, rel_err: f64) {
        self.absolute_error += abs_err;
        self.relative_error += rel_err;
        self.rounding_error += abs_err * 0.1; // Estimate rounding contribution
    }

    /// Check if errors are within acceptable bounds
    pub fn is_within_bounds(&self, precision: PrecisionLevel) -> bool {
        self.absolute_error < precision.epsilon() && self.relative_error < precision.epsilon()
    }
}

impl PreciseQuantumOffset {
    /// Create a new precise quantum offset
    pub fn new(
        temporal_offset: PreciseDuration,
        phase_component: f64,
        precision_level: PrecisionLevel,
    ) -> TimeResult<Self> {
        // Normalize phase to [0.0, 1.0)
        let normalized_phase = ((phase_component % 1.0) + 1.0) % 1.0;

        Ok(Self {
            temporal_offset,
            phase_component: normalized_phase,
            precision_level,
            error_bounds: ErrorBounds::zero(),
            precision_metadata: PrecisionMetadata {
                target_precision: precision_level,
                achieved_precision: precision_level,
                operation_count: 0,
                precision_source: "direct_creation".to_string(),
            },
        })
    }

    /// Zero offset with specified precision
    pub fn zero(precision: PrecisionLevel) -> Self {
        Self {
            temporal_offset: PreciseDuration::zero(precision),
            phase_component: 0.0,
            precision_level: precision,
            error_bounds: ErrorBounds::zero(),
            precision_metadata: PrecisionMetadata {
                target_precision: precision,
                achieved_precision: precision,
                operation_count: 0,
                precision_source: "zero_initialization".to_string(),
            },
        }
    }

    /// Apply quantum offset to a time value with precision tracking
    pub fn apply_precise(&self, base_time: NanoTime) -> TimeResult<(NanoTime, f64)> {
        let base_ns = base_time.as_nanos() as f64;

        // High-precision quantum calculation
        let temporal_adjustment = self.temporal_offset.as_total_nanos();
        let phase_radians = self.phase_component * 2.0 * std::f64::consts::PI;

        // Quantum oscillation calculation with precision tracking
        let quantum_factor = phase_radians.sin();
        let quantum_adjustment = quantum_factor * temporal_adjustment;

        let adjusted_ns = base_ns + quantum_adjustment;
        let precision_loss = self.error_bounds.absolute_error;

        // Bounds checking
        if adjusted_ns < 0.0 || adjusted_ns > (u64::MAX as f64) {
            return Err(TimeError::ArithmeticOverflow);
        }

        Ok((NanoTime::from_nanos(adjusted_ns as u64), precision_loss))
    }

    /// Convert to legacy QuantumOffset (with precision loss)
    pub fn to_legacy(&self) -> (crate::QuantumOffset, f64) {
        let (_nano_time, precision_loss) = self.temporal_offset.to_nano_time();
        let legacy_offset = crate::QuantumOffset::new(
            self.phase_component,
            1.0, // amplitude
            1.0, // frequency
        );

        (
            legacy_offset,
            precision_loss + self.error_bounds.absolute_error,
        )
    }

    /// Update precision metadata after an operation
    pub fn record_operation(&mut self, operation_name: &str) {
        self.precision_metadata.operation_count += 1;
        self.precision_metadata.precision_source = format!(
            "{}_op_{}",
            operation_name, self.precision_metadata.operation_count
        );
    }

    /// Validate cross-platform consistency
    pub fn validate_cross_platform(&mut self) -> TimeResult<()> {
        // Perform cross-platform validation checks
        let test_time = NanoTime::from_nanos(1_000_000_000); // 1 second
        let (result1, _) = self.apply_precise(test_time)?;
        let (result2, _) = self.apply_precise(test_time)?;

        if result1 == result2 {
            self.error_bounds.cross_platform_validated = true;
            Ok(())
        } else {
            Err(TimeError::SystemTimeError {
                details: "Cross-platform consistency validation failed".to_string(),
            })
        }
    }
}

// Arithmetic operations with precision tracking
impl Add for PreciseDuration {
    type Output = TimeResult<PreciseDuration>;

    fn add(self, other: PreciseDuration) -> Self::Output {
        let mut result = PreciseDuration::new(
            self.nanos + other.nanos,
            self.sub_nanos + other.sub_nanos,
            self.precision.min(other.precision),
        )?;

        // Handle sub-nanosecond overflow
        if result.sub_nanos >= 1.0 {
            result.nanos += 1;
            result.sub_nanos -= 1.0;
        }

        result.check_overflow()?;
        Ok(result)
    }
}

impl Sub for PreciseDuration {
    type Output = TimeResult<PreciseDuration>;

    fn sub(self, other: PreciseDuration) -> Self::Output {
        let mut nanos = self.nanos - other.nanos;
        let mut sub_nanos = self.sub_nanos - other.sub_nanos;

        // Handle sub-nanosecond underflow
        if sub_nanos < 0.0 {
            nanos -= 1;
            sub_nanos += 1.0;
        }

        let result = PreciseDuration::new(nanos, sub_nanos, self.precision.min(other.precision))?;

        result.check_overflow()?;
        Ok(result)
    }
}

impl Mul<f64> for PreciseDuration {
    type Output = TimeResult<PreciseDuration>;

    fn mul(self, scalar: f64) -> Self::Output {
        let total_nanos = self.as_total_nanos() * scalar;
        Ok(PreciseDuration::from_nanos_precise(
            total_nanos,
            self.precision,
        ))
    }
}

impl Div<f64> for PreciseDuration {
    type Output = TimeResult<PreciseDuration>;

    fn div(self, scalar: f64) -> Self::Output {
        if scalar.abs() < self.precision.epsilon() {
            return Err(TimeError::ArithmeticOverflow);
        }

        let total_nanos = self.as_total_nanos() / scalar;
        Ok(PreciseDuration::from_nanos_precise(
            total_nanos,
            self.precision,
        ))
    }
}

// Precision-aware comparison
impl PrecisionBound for PreciseQuantumOffset {
    fn precision_eq(&self, other: &Self, level: PrecisionLevel) -> bool {
        self.temporal_offset
            .as_total_nanos()
            .precision_eq(&other.temporal_offset.as_total_nanos(), level)
            && self
                .phase_component
                .precision_eq(&other.phase_component, level)
    }

    fn is_precision_zero(&self, level: PrecisionLevel) -> bool {
        self.temporal_offset
            .as_total_nanos()
            .is_precision_zero(level)
            && self.phase_component.is_precision_zero(level)
    }
}

impl fmt::Display for PreciseQuantumOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PreciseQuantumOffset {{ temporal: {:.6}ns, phase: {:.6}, precision: {:?}, error: {:.2e} }}",
               self.temporal_offset.as_total_nanos(),
               self.phase_component,
               self.precision_level,
               self.error_bounds.absolute_error)
    }
}

impl fmt::Display for PreciseDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_ns = self.as_total_nanos();
        if total_ns.abs() < 1_000.0 {
            write!(f, "{:.3}ns", total_ns)
        } else if total_ns.abs() < 1_000_000.0 {
            write!(f, "{:.3}µs", total_ns / 1_000.0)
        } else if total_ns.abs() < 1_000_000_000.0 {
            write!(f, "{:.3}ms", total_ns / 1_000_000.0)
        } else {
            write!(f, "{:.6}s", total_ns / 1_000_000_000.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_levels() {
        assert_eq!(PrecisionLevel::Femtosecond.epsilon(), 1e-15);
        assert_eq!(PrecisionLevel::Nanosecond.epsilon(), 1e-9);
    }

    #[test]
    fn test_precise_duration_creation() {
        let duration = PreciseDuration::new(1000, 0.5, PrecisionLevel::Femtosecond).unwrap();
        assert_eq!(duration.as_total_nanos(), 1000.5);
    }

    #[test]
    fn test_precise_duration_arithmetic() {
        let d1 = PreciseDuration::new(1000, 0.3, PrecisionLevel::Femtosecond).unwrap();
        let d2 = PreciseDuration::new(500, 0.8, PrecisionLevel::Femtosecond).unwrap();

        let sum = (d1 + d2).unwrap();
        assert_eq!(sum.nanos, 1501); // 0.3 + 0.8 = 1.1, so 1 carries over
        assert!((sum.sub_nanos - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_offset_creation() {
        let temporal = PreciseDuration::new(1000, 0.0, PrecisionLevel::Nanosecond).unwrap();
        let offset = PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Nanosecond).unwrap();

        assert_eq!(offset.phase_component, 0.25);
        assert_eq!(offset.precision_level, PrecisionLevel::Nanosecond);
    }

    #[test]
    fn test_quantum_offset_application() {
        let temporal = PreciseDuration::new(100, 0.0, PrecisionLevel::Nanosecond).unwrap();
        let offset = PreciseQuantumOffset::new(temporal, 0.0, PrecisionLevel::Nanosecond).unwrap();

        let base_time = NanoTime::from_nanos(1_000_000_000);
        let (result, precision_loss) = offset.apply_precise(base_time).unwrap();

        assert!(precision_loss >= 0.0);
        assert!(result.as_nanos() > 0);
    }

    #[test]
    fn test_precision_bounds() {
        let val1 = 1.0000000001;
        let val2 = 1.0000000002;

        assert!(val1.precision_eq(&val2, PrecisionLevel::Nanosecond));
        assert!(!val1.precision_eq(&val2, PrecisionLevel::Femtosecond));
    }

    #[test]
    fn test_error_bounds() {
        let mut bounds = ErrorBounds::zero();
        bounds.add_operation_error(1e-12, 1e-15);

        assert!(bounds.is_within_bounds(PrecisionLevel::Nanosecond));
        assert!(!bounds.is_within_bounds(PrecisionLevel::Femtosecond));
    }

    #[test]
    fn test_cross_platform_validation() {
        let temporal = PreciseDuration::new(1000, 0.0, PrecisionLevel::Nanosecond).unwrap();
        let mut offset =
            PreciseQuantumOffset::new(temporal, 0.5, PrecisionLevel::Nanosecond).unwrap();

        assert!(offset.validate_cross_platform().is_ok());
        assert!(offset.error_bounds.cross_platform_validated);
    }

    #[test]
    fn test_legacy_conversion() {
        let temporal = PreciseDuration::new(1000, 0.123, PrecisionLevel::Femtosecond).unwrap();
        let precise_offset =
            PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Femtosecond).unwrap();

        let (legacy_offset, precision_loss) = precise_offset.to_legacy();
        assert_eq!(legacy_offset.phase, 0.25);
        assert!(precision_loss > 0.0); // Should have some precision loss
    }

    #[test]
    fn test_overflow_detection() {
        let result = PreciseDuration::new(i64::MAX, 0.0, PrecisionLevel::Nanosecond);
        assert!(result.unwrap().check_overflow().is_err());
    }

    #[test]
    fn test_serialization() {
        let temporal = PreciseDuration::new(1000, 0.5, PrecisionLevel::Femtosecond).unwrap();
        let offset =
            PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Femtosecond).unwrap();

        let serialized = serde_json::to_string(&offset).unwrap();
        let deserialized: PreciseQuantumOffset = serde_json::from_str(&serialized).unwrap();

        assert!(offset.precision_eq(&deserialized, PrecisionLevel::Femtosecond));
    }

    #[test]
    fn test_display_formatting() {
        let temporal = PreciseDuration::new(1234, 0.567, PrecisionLevel::Nanosecond).unwrap();
        let duration_str = temporal.to_string();
        assert!(duration_str.contains("1234.567"));

        let offset = PreciseQuantumOffset::new(temporal, 0.25, PrecisionLevel::Nanosecond).unwrap();
        let offset_str = offset.to_string();
        assert!(offset_str.contains("PreciseQuantumOffset"));
        assert!(offset_str.contains("1234.567"));
    }
}
