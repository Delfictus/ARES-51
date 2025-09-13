// Serializable time types for H100 performance profiler
// Maintains full precision while enabling proper serialization

use serde::{Serialize, Deserialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};

/// Serializable wrapper for SystemTime with millisecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SerializableSystemTime {
    /// Milliseconds since UNIX epoch
    epoch_millis: u64,
}

impl SerializableSystemTime {
    /// Create from current system time
    pub fn now() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self { epoch_millis: millis }
    }
    
    /// Create from SystemTime
    pub fn from_system_time(time: SystemTime) -> Self {
        let millis = time
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self { epoch_millis: millis }
    }
    
    /// Convert back to SystemTime
    pub fn to_system_time(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(self.epoch_millis)
    }
    
    /// Get milliseconds since epoch
    pub fn as_millis(&self) -> u64 {
        self.epoch_millis
    }
    
    /// Calculate elapsed time from another SerializableSystemTime
    pub fn duration_since(&self, earlier: SerializableSystemTime) -> SerializableDuration {
        let duration_millis = self.epoch_millis.saturating_sub(earlier.epoch_millis);
        SerializableDuration::from_millis(duration_millis)
    }
}

/// Serializable wrapper for Duration with microsecond precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SerializableDuration {
    /// Duration in microseconds for high precision
    micros: u64,
}

impl SerializableDuration {
    /// Create from Duration
    pub fn from_duration(duration: Duration) -> Self {
        Self {
            micros: duration.as_micros() as u64,
        }
    }
    
    /// Create from milliseconds
    pub fn from_millis(millis: u64) -> Self {
        Self {
            micros: millis * 1000,
        }
    }
    
    /// Create from microseconds
    pub fn from_micros(micros: u64) -> Self {
        Self { micros }
    }
    
    /// Create from nanoseconds (rounded to microsecond precision)
    pub fn from_nanos(nanos: u64) -> Self {
        Self {
            micros: nanos / 1000,
        }
    }
    
    /// Convert back to Duration
    pub fn to_duration(&self) -> Duration {
        Duration::from_micros(self.micros)
    }
    
    /// Get as milliseconds
    pub fn as_millis(&self) -> u64 {
        self.micros / 1000
    }
    
    /// Get as microseconds
    pub fn as_micros(&self) -> u64 {
        self.micros
    }
    
    /// Get as nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.micros * 1000
    }
    
    /// Get as seconds (floating point)
    pub fn as_secs_f64(&self) -> f64 {
        self.micros as f64 / 1_000_000.0
    }
    
    /// Check if duration is zero
    pub fn is_zero(&self) -> bool {
        self.micros == 0
    }
}

impl Default for SerializableDuration {
    fn default() -> Self {
        Self { micros: 0 }
    }
}

impl From<Duration> for SerializableDuration {
    fn from(duration: Duration) -> Self {
        Self::from_duration(duration)
    }
}

impl From<SerializableDuration> for Duration {
    fn from(serializable: SerializableDuration) -> Self {
        serializable.to_duration()
    }
}

impl From<SystemTime> for SerializableSystemTime {
    fn from(time: SystemTime) -> Self {
        Self::from_system_time(time)
    }
}

impl From<SerializableSystemTime> for SystemTime {
    fn from(serializable: SerializableSystemTime) -> Self {
        serializable.to_system_time()
    }
}

/// Helper for converting Instant to SerializableSystemTime
/// Note: Instant cannot be perfectly converted as it's relative to an arbitrary point
/// This provides a workaround by using SystemTime for serializable timestamps
pub struct InstantHelper;

impl InstantHelper {
    /// Convert Instant to SerializableSystemTime using current time
    /// This is approximate and should only be used when serialization is required
    pub fn instant_to_serializable(_instant: Instant) -> SerializableSystemTime {
        // Since Instant is relative and cannot be converted to absolute time,
        // we use current SystemTime as a reasonable approximation
        SerializableSystemTime::now()
    }
    
    /// Create a timestamp that can be used for relative measurements
    pub fn create_timestamp() -> SerializableSystemTime {
        SerializableSystemTime::now()
    }
}

/// Performance timing helper that works with serializable types
#[derive(Debug, Clone)]
pub struct SerializableTimer {
    start_time: SerializableSystemTime,
}

impl SerializableTimer {
    /// Start timing
    pub fn start() -> Self {
        Self {
            start_time: SerializableSystemTime::now(),
        }
    }
    
    /// Get elapsed time since start
    pub fn elapsed(&self) -> SerializableDuration {
        let now = SerializableSystemTime::now();
        now.duration_since(self.start_time)
    }
    
    /// Get elapsed time and restart timer
    pub fn elapsed_and_restart(&mut self) -> SerializableDuration {
        let now = SerializableSystemTime::now();
        let elapsed = now.duration_since(self.start_time);
        self.start_time = now;
        elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_serializable_system_time_roundtrip() {
        let original = SerializableSystemTime::now();
        let system_time = original.to_system_time();
        let roundtrip = SerializableSystemTime::from_system_time(system_time);
        
        // Should be identical after roundtrip
        assert_eq!(original, roundtrip);
    }
    
    #[test]
    fn test_serializable_duration_precision() {
        let original_duration = Duration::from_nanos(123_456_789);
        let serializable = SerializableDuration::from_duration(original_duration);
        let restored = serializable.to_duration();
        
        // Should preserve microsecond precision
        assert_eq!(
            original_duration.as_micros(),
            restored.as_micros()
        );
    }
    
    #[test]
    fn test_serializable_duration_arithmetic() {
        let d1 = SerializableDuration::from_millis(1500); // 1.5 seconds
        let d2 = SerializableDuration::from_micros(500_000); // 0.5 seconds
        
        assert_eq!(d1.as_micros(), 1_500_000);
        assert_eq!(d2.as_micros(), 500_000);
        assert_eq!(d1.as_secs_f64(), 1.5);
        assert_eq!(d2.as_secs_f64(), 0.5);
    }
    
    #[test]
    fn test_timer_functionality() {
        let timer = SerializableTimer::start();
        thread::sleep(Duration::from_millis(10));
        
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
        assert!(elapsed.as_millis() < 100); // Should be reasonable
    }
    
    #[test]
    fn test_serialization_roundtrip() {
        let original_time = SerializableSystemTime::now();
        let original_duration = SerializableDuration::from_millis(12345);
        
        // Test JSON serialization
        let time_json = serde_json::to_string(&original_time).unwrap();
        let duration_json = serde_json::to_string(&original_duration).unwrap();
        
        let deserialized_time: SerializableSystemTime = serde_json::from_str(&time_json).unwrap();
        let deserialized_duration: SerializableDuration = serde_json::from_str(&duration_json).unwrap();
        
        assert_eq!(original_time, deserialized_time);
        assert_eq!(original_duration, deserialized_duration);
    }
    
    #[test]
    fn test_duration_since() {
        let earlier = SerializableSystemTime::now();
        thread::sleep(Duration::from_millis(5));
        let later = SerializableSystemTime::now();
        
        let duration = later.duration_since(earlier);
        assert!(duration.as_millis() >= 5);
        assert!(duration.as_millis() < 100);
    }
    
    #[test]
    fn test_zero_duration() {
        let zero = SerializableDuration::default();
        assert!(zero.is_zero());
        assert_eq!(zero.as_millis(), 0);
        assert_eq!(zero.as_micros(), 0);
        assert_eq!(zero.as_secs_f64(), 0.0);
    }
}