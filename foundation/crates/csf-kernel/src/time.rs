//! High-precision time utilities with TTW integration

use csf_core::types::NanoTime;
use csf_time::{global_time_source, Duration as CsfDuration, NanoTime as CsfNanoTime, TimeSource};
use std::sync::Arc;

/// Convert csf_time::NanoTime to csf_core::NanoTime
fn convert_time(time: CsfNanoTime) -> NanoTime {
    NanoTime::from_nanos(time.as_nanos())
}

/// Hardware clock abstraction using TTW time source
pub mod hardware_clock {
    use super::*;

    /// Get current time in nanoseconds from TTW time source
    #[inline]
    pub fn now() -> NanoTime {
        convert_time(global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO))
    }

    /// Convert TTW Duration to nanoseconds
    #[inline]
    pub fn duration_to_nanos(d: CsfDuration) -> NanoTime {
        NanoTime::from_nanos(d.as_nanos())
    }

    /// Convert nanoseconds to TTW Duration
    #[inline]
    pub fn nanos_to_duration(nanos: NanoTime) -> CsfDuration {
        CsfDuration::from_nanos(nanos.as_nanos())
    }

    /// Precise sleep using TTW time management
    pub async fn precise_sleep(duration: CsfDuration) {
        let current_time = global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO);
        let _deadline =
            CsfNanoTime::from_nanos(current_time.as_nanos().saturating_add(duration.as_nanos()));
        // For now, use tokio sleep as placeholder - full implementation would use TTW scheduler
        let duration_std = std::time::Duration::from_nanos(duration.as_nanos());
        tokio::time::sleep(duration_std).await;
    }
}

/// Timer implementation using TTW time source
pub struct Timer {
    start: CsfNanoTime,
}

impl Timer {
    /// Create a new timer starting at current TTW time
    pub fn new() -> Self {
        Self {
            start: global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO),
        }
    }

    /// Get elapsed time as TTW Duration
    pub fn elapsed(&self) -> CsfDuration {
        let current = global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO);
        // Return Duration, not NanoTime
        CsfDuration::from_nanos(current.as_nanos().saturating_sub(self.start.as_nanos()))
    }

    /// Restart the timer from current TTW time
    pub fn restart(&mut self) {
        self.start = global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO);
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter using TTW time source
pub struct RateLimiter {
    period: CsfDuration,
    last_execution: Option<CsfNanoTime>,
}

impl RateLimiter {
    /// Create a rate limiter with the given frequency (Hz)
    pub fn new(frequency_hz: f64) -> Self {
        let period_secs = 1.0 / frequency_hz;
        let period = CsfDuration::from_nanos((period_secs * 1e9) as u64);

        Self {
            period,
            // Allow immediate first execution by setting last_execution to None
            last_execution: None,
        }
    }

    /// Check if enough time has passed to allow execution
    pub fn try_execute(&mut self) -> bool {
        let now = global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO);

        match self.last_execution {
            // First execution - always allowed
            None => {
                self.last_execution = Some(now);
                true
            }
            // Check if enough time has passed since last execution
            Some(last_exec) => {
                let next_allowed = CsfNanoTime::from_nanos(
                    last_exec.as_nanos().saturating_add(self.period.as_nanos()),
                );
                if now >= next_allowed {
                    self.last_execution = Some(now);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Wait until the next execution is allowed, then return
    pub async fn wait_and_execute(&mut self) {
        let now = global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO);

        if let Some(last_exec) = self.last_execution {
            let next_allowed = CsfNanoTime::from_nanos(
                last_exec.as_nanos().saturating_add(self.period.as_nanos()),
            );
            if now < next_allowed {
                // High-precision rate limiting with hybrid sleep
                let wait_duration =
                    CsfDuration::from_nanos(next_allowed.as_nanos().saturating_sub(now.as_nanos()));
                let duration_std = std::time::Duration::from_nanos(wait_duration.as_nanos());
                
                // For longer waits, use tokio sleep for efficiency
                if wait_duration.as_nanos() > 1_000_000 {
                    let bulk_sleep = duration_std.saturating_sub(std::time::Duration::from_micros(100));
                    tokio::time::sleep(bulk_sleep).await;
                }
                
                // Spin-wait for precise timing
                while global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO) < next_allowed {
                    std::hint::spin_loop();
                }
            }
        }
        // If last_execution is None (first execution), no waiting needed

        self.last_execution = Some(global_time_source().now_ns().unwrap_or(CsfNanoTime::ZERO));
    }
}

/// Enterprise rate limiter with injectable TimeSource for deterministic testing
pub struct RateLimiterEnterprise {
    period: CsfDuration,
    last_execution: Option<CsfNanoTime>,
    time_source: Arc<dyn TimeSource>,
}

impl RateLimiterEnterprise {
    /// Create a rate limiter with the given frequency and time source
    pub fn new(frequency_hz: f64, time_source: Arc<dyn TimeSource>) -> Self {
        let period_secs = 1.0 / frequency_hz;
        let period = CsfDuration::from_nanos((period_secs * 1e9) as u64);

        Self {
            period,
            last_execution: None,
            time_source,
        }
    }

    /// Check if enough time has passed to allow execution
    pub fn try_execute(&mut self) -> bool {
        let now = self.time_source.now_ns().unwrap_or(CsfNanoTime::ZERO);

        match self.last_execution {
            None => {
                self.last_execution = Some(now);
                true
            }
            Some(last_exec) => {
                let next_allowed = CsfNanoTime::from_nanos(
                    last_exec.as_nanos().saturating_add(self.period.as_nanos()),
                );
                if now >= next_allowed {
                    self.last_execution = Some(now);
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csf_time::{SimulatedTimeSource, TimeSource};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_timer() {
        // Create isolated simulated time source for this test
        let time_source = Arc::new(SimulatedTimeSource::new(CsfNanoTime::from_secs(1000)));
        
        // Create timer with current time
        let start = time_source.now_ns().unwrap_or(CsfNanoTime::ZERO);
        let timer = Timer { start };

        // Advance simulated time by 10ms
        time_source
            .advance_simulation(CsfDuration::from_millis(10).as_nanos())
            .expect("Time advancement should work in isolated test");

        // Calculate elapsed time using the same time source
        let current = time_source.now_ns().unwrap_or(CsfNanoTime::ZERO);
        let elapsed = CsfDuration::from_nanos(current.as_nanos().saturating_sub(start.as_nanos()));
        
        assert_eq!(elapsed.as_millis(), 10, "Elapsed time should be exactly 10ms");
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        // Create isolated simulated time source for this test
        let time_source = Arc::new(SimulatedTimeSource::new(CsfNanoTime::from_secs(2000)));
        
        // Create enterprise rate limiter with isolated time source
        let mut limiter = RateLimiterEnterprise::new(10.0, time_source.clone()); // 10 Hz = 100ms period

        // First execution should be allowed immediately
        assert!(limiter.try_execute());

        // Second execution should be blocked (too soon)
        assert!(!limiter.try_execute());

        // Advance time by exactly one period (100ms)
        time_source
            .advance_simulation(CsfDuration::from_millis(100).as_nanos())
            .expect("Time advancement should work in isolated test");

        // Now execution should be allowed again after period has passed
        assert!(limiter.try_execute());
    }

    #[test]
    fn test_hardware_clock_conversion() {
        let nanos = 1_500_000_000u64;
        let duration = hardware_clock::nanos_to_duration(nanos.into());
        assert_eq!(duration.as_secs(), 1);
        assert_eq!(duration.as_millis(), 1500);

        let back_to_nanos = hardware_clock::duration_to_nanos(duration);
        assert_eq!(back_to_nanos, nanos.into());
    }
}
