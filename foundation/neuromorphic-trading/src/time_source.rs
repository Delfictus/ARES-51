//! Hardware TSC-based high-precision time source
//! 
//! Provides <10ns precision timing using x86_64 RDTSC instruction

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::Result;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_rdtsc, _mm_lfence};

/// Hardware clock using TSC
pub struct HardwareClock {
    tsc_frequency: AtomicU64,
    tsc_offset: AtomicI64,
    last_calibration: AtomicU64,
    calibration_interval_ns: u64,
    fallback_clock: Instant,
}

impl HardwareClock {
    /// Create new hardware clock
    pub fn new() -> Result<Self> {
        let mut clock = Self {
            tsc_frequency: AtomicU64::new(0),
            tsc_offset: AtomicI64::new(0),
            last_calibration: AtomicU64::new(0),
            calibration_interval_ns: 1_000_000_000, // 1 second
            fallback_clock: Instant::now(),
        };
        
        clock.calibrate()?;
        Ok(clock)
    }
    
    /// Get current time in nanoseconds
    #[inline(always)]
    pub fn now_ns(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                _mm_lfence(); // Serialize instruction stream
                let tsc = _rdtsc();
                self.tsc_to_nanos(tsc)
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86_64
            self.fallback_now_ns()
        }
    }
    
    /// Calibrate TSC frequency
    fn calibrate(&mut self) -> Result<f64> {
        #[cfg(target_arch = "x86_64")]
        {
            let start_time = SystemTime::now();
            let start_tsc = unsafe { _rdtsc() };
            
            // Calibration period
            std::thread::sleep(Duration::from_millis(100));
            
            let end_time = SystemTime::now();
            let end_tsc = unsafe { _rdtsc() };
            
            let elapsed_ns = end_time.duration_since(start_time)?.as_nanos() as u64;
            let tsc_diff = end_tsc - start_tsc;
            
            let frequency = (tsc_diff as f64 * 1_000_000_000.0) / elapsed_ns as f64;
            self.tsc_frequency.store(frequency as u64, Ordering::Release);
            
            // Store calibration time
            self.last_calibration.store(
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64,
                Ordering::Release
            );
            
            Ok(frequency)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Ok(1_000_000_000.0) // 1GHz fallback
        }
    }
    
    /// Convert TSC to nanoseconds
    #[inline(always)]
    fn tsc_to_nanos(&self, tsc: u64) -> u64 {
        let freq = self.tsc_frequency.load(Ordering::Relaxed);
        if freq == 0 {
            return self.fallback_now_ns();
        }
        
        let offset = self.tsc_offset.load(Ordering::Relaxed);
        let nanos = ((tsc as i128 * 1_000_000_000) / freq as i128) as i64;
        (nanos + offset) as u64
    }
    
    /// Fallback timing using Instant
    #[inline(always)]
    fn fallback_now_ns(&self) -> u64 {
        self.fallback_clock.elapsed().as_nanos() as u64
    }
    
    /// Check if recalibration is needed
    pub fn needs_calibration(&self) -> bool {
        let last = self.last_calibration.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        now - last > self.calibration_interval_ns
    }
    
    /// Get TSC frequency in Hz
    pub fn get_frequency(&self) -> u64 {
        self.tsc_frequency.load(Ordering::Relaxed)
    }
}

/// High-resolution timer for benchmarking
pub struct Timer {
    start: u64,
    clock: HardwareClock,
}

impl Timer {
    pub fn new() -> Result<Self> {
        let clock = HardwareClock::new()?;
        let start = clock.now_ns();
        Ok(Self { start, clock })
    }
    
    pub fn elapsed_ns(&self) -> u64 {
        self.clock.now_ns() - self.start
    }
    
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed_ns() as f64 / 1000.0
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns() as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_clock() {
        let clock = HardwareClock::new().unwrap();
        
        let t1 = clock.now_ns();
        std::thread::sleep(Duration::from_millis(1));
        let t2 = clock.now_ns();
        
        assert!(t2 > t1);
        let diff = t2 - t1;
        
        // Should be approximately 1ms
        assert!(diff > 900_000 && diff < 2_000_000);
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        
        let elapsed_ms = timer.elapsed_ms();
        assert!(elapsed_ms > 9.0 && elapsed_ms < 15.0);
    }
    
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_tsc_monotonic() {
        unsafe {
            let tsc1 = _rdtsc();
            let tsc2 = _rdtsc();
            let tsc3 = _rdtsc();
            
            assert!(tsc2 >= tsc1);
            assert!(tsc3 >= tsc2);
        }
    }
}