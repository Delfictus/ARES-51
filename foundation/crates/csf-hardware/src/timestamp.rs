//! Hardware timestamp implementation for nanosecond precision timing
//!
//! Uses TTW TimeSource for ChronoSynclastic determinism.

use csf_time::{global_time_source, NanoTime};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_lfence, _rdtsc};

/// Hardware timer abstraction
pub struct HardwareTimer {
    /// TSC frequency in Hz
    #[allow(dead_code)]
    tsc_frequency: u64,
    /// Reference time from TimeSource for calibration
    #[allow(dead_code)]
    reference_time: NanoTime,
    /// Reference TSC value
    #[allow(dead_code)]
    reference_tsc: u64,
}

static TSC_FREQUENCY: AtomicU64 = AtomicU64::new(0);
static TIMER_INITIALIZED: AtomicU64 = AtomicU64::new(0);

/// Initialize the hardware timer
pub fn init_timer() -> anyhow::Result<()> {
    if TIMER_INITIALIZED.load(Ordering::Acquire) != 0 {
        return Ok(());
    }

    let freq = calibrate_tsc_frequency()?;
    TSC_FREQUENCY.store(freq, Ordering::Release);
    TIMER_INITIALIZED.store(1, Ordering::Release);

    Ok(())
}

/// Get nanosecond timestamp using hardware counters
#[inline(always)]
pub fn hardware_timestamp() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if TSC_FREQUENCY.load(Ordering::Relaxed) == 0 {
            // Fallback to TTW TimeSource if not initialized
            return global_time_source()
                .now_ns()
                .unwrap_or(NanoTime::ZERO)
                .as_nanos();
        }

        unsafe {
            let tsc = _rdtsc();
            let freq = TSC_FREQUENCY.load(Ordering::Relaxed);
            // Convert TSC to nanoseconds
            (tsc as u128 * 1_000_000_000 / freq as u128) as u64
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64 architectures using TTW TimeSource
        global_time_source()
            .now_ns()
            .unwrap_or(NanoTime::ZERO)
            .as_nanos()
    }
}

/// Get hardware timestamp with CPU ID (for ordering guarantees)
#[inline(always)]
pub fn hardware_timestamp_with_cpu() -> (u64, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            // Use _rdtsc with lfence for serialization (stable fallback)
            _mm_lfence();
            let tsc = _rdtsc();
            _mm_lfence();
            let freq = TSC_FREQUENCY.load(Ordering::Relaxed);
            let nanos = (tsc as u128 * 1_000_000_000 / freq as u128) as u64;
            (nanos, 0) // Cannot get CPU ID from stable intrinsics
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        (hardware_timestamp(), 0)
    }
}

/// Calibrate TSC frequency by measuring against system time
fn calibrate_tsc_frequency() -> anyhow::Result<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::thread;
        use std::time::Instant;

        const CALIBRATION_MS: u64 = 100;

        // Warm up
        for _ in 0..10 {
            unsafe {
                _rdtsc();
            }
        }

        // Measure
        let start_instant = Instant::now();
        let start_tsc = unsafe { _rdtsc() };

        thread::sleep(Duration::from_millis(CALIBRATION_MS));

        let end_tsc = unsafe { _rdtsc() };
        let elapsed = start_instant.elapsed();

        let tsc_delta = end_tsc - start_tsc;
        let nanos_elapsed = elapsed.as_nanos() as u64;

        // Calculate frequency (TSC ticks per second)
        let frequency = (tsc_delta as u128 * 1_000_000_000 / nanos_elapsed as u128) as u64;

        // Sanity check - modern CPUs are typically 1-5 GHz
        if !(500_000_000..=10_000_000_000).contains(&frequency) {
            return Err(anyhow::anyhow!(
                "TSC frequency calibration failed: {} Hz",
                frequency
            ));
        }

        Ok(frequency)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Default frequency for fallback timer
        Ok(1_000_000_000) // 1 GHz
    }
}

/// High-precision timer for benchmarking using TTW TimeSource
pub struct PrecisionTimer {
    start_time: NanoTime,
}

impl PrecisionTimer {
    /// Start a new timer
    #[inline]
    pub fn start() -> Self {
        Self {
            start_time: global_time_source().now_ns().unwrap_or(NanoTime::ZERO),
        }
    }

    /// Get elapsed time in nanoseconds
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed().as_nanos().try_into().unwrap_or(u64::MAX)
    }

    /// Get elapsed time as Duration
    #[inline]
    pub fn elapsed(&self) -> Duration {
        let current_time = global_time_source().now_ns().unwrap_or(NanoTime::ZERO);
        if current_time > self.start_time {
            let diff = current_time - self.start_time;
            Duration::from_nanos(diff.as_nanos())
        } else {
            Duration::ZERO
        }
    }
}

/// Memory barrier for timestamp ordering
#[inline(always)]
pub fn timestamp_barrier() {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            _mm_lfence();
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_timestamp() {
        init_timer().unwrap();

        let t1 = hardware_timestamp();
        std::thread::sleep(Duration::from_millis(1));
        let t2 = hardware_timestamp();

        assert!(t2 > t1);
        let delta = t2 - t1;

        // Should be roughly 1ms
        assert!(delta > 900_000); // 0.9ms
        assert!(delta < 2_000_000); // 2ms
    }

    #[test]
    fn test_precision_timer() {
        init_timer().unwrap();

        let timer = PrecisionTimer::start();
        std::thread::sleep(Duration::from_micros(100));
        let elapsed = timer.elapsed_ns();

        // Should be roughly 100μs
        assert!(elapsed > 90_000); // 90μs
        assert!(elapsed < 200_000); // 200μs
    }
}
