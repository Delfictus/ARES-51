//! Performance profiling utilities.

/// Performance profiler for HPC operations
pub struct PerformanceProfiler {
    start_time: std::time::Instant,
}

impl PerformanceProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}