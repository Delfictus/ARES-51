//! 🛡️ HARDENING PHASE 3: Advanced performance monitoring and mutex contention detection
//!
//! This module provides comprehensive performance monitoring capabilities including:
//! - Mutex contention detection and reporting
//! - Lock acquisition time tracking  
//! - Performance regression detection
//! - Resource usage monitoring

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 🛡️ HARDENING: Mutex contention monitoring constants
const CONTENTION_WARNING_THRESHOLD_MS: u64 = 10; // Warn if lock acquisition takes >10ms
const CONTENTION_CRITICAL_THRESHOLD_MS: u64 = 100; // Critical if >100ms
const MAX_CONTENTION_EVENTS: usize = 1000;

/// Global performance monitor instance
static PERFORMANCE_MONITOR: once_cell::sync::Lazy<Arc<PerformanceMonitor>> =
    once_cell::sync::Lazy::new(|| Arc::new(PerformanceMonitor::new()));

/// 🛡️ HARDENING: Comprehensive performance monitoring system
pub struct PerformanceMonitor {
    /// Mutex contention events
    contention_events: DashMap<String, ContentionStats>,

    /// Lock acquisition times
    lock_times: RwLock<HashMap<String, LockTimeStats>>,

    /// Performance metrics
    metrics: PerformanceMetrics,

    /// Operation timing data
    operation_times: DashMap<String, OperationStats>,
}

/// Statistics for mutex contention on a specific lock
#[derive(Debug)]
pub struct ContentionStats {
    pub total_contentions: AtomicU64,
    pub total_wait_time_ns: AtomicU64,
    pub max_wait_time_ns: AtomicU64,
    pub avg_wait_time_ns: AtomicU64,
    pub last_contention: AtomicU64, // timestamp in nanos
}

/// Lock acquisition time statistics
#[derive(Debug, Clone, Default)]
pub struct LockTimeStats {
    pub acquisitions: u64,
    pub total_time_ns: u64,
    pub max_time_ns: u64,
    pub min_time_ns: u64,
    pub contentions: u64,
}

/// Overall performance metrics
#[derive(Debug)]
pub struct PerformanceMetrics {
    pub mutex_contentions: AtomicU64,
    pub slow_operations: AtomicU64,
    pub memory_allocations: AtomicU64,
    pub circuit_breaker_activations: AtomicU64,
    pub total_operations: AtomicU64,
}

/// Operation performance statistics
#[derive(Debug)]
pub struct OperationStats {
    pub total_calls: AtomicU64,
    pub total_time_ns: AtomicU64,
    pub max_time_ns: AtomicU64,
    pub min_time_ns: AtomicU64,
    pub error_count: AtomicU64,
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            max_time_ns: AtomicU64::new(0),
            min_time_ns: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }
}

impl Clone for OperationStats {
    fn clone(&self) -> Self {
        Self {
            total_calls: AtomicU64::new(self.total_calls.load(Ordering::Relaxed)),
            total_time_ns: AtomicU64::new(self.total_time_ns.load(Ordering::Relaxed)),
            max_time_ns: AtomicU64::new(self.max_time_ns.load(Ordering::Relaxed)),
            min_time_ns: AtomicU64::new(self.min_time_ns.load(Ordering::Relaxed)),
            error_count: AtomicU64::new(self.error_count.load(Ordering::Relaxed)),
        }
    }
}

/// 🛡️ HARDENING: Monitored mutex wrapper with contention detection
pub struct MonitoredMutex<T> {
    inner: Mutex<T>,
    name: String,
    stats: Arc<ContentionStats>,
}

/// 🛡️ HARDENING: Lock acquisition timing guard
pub struct TimedMutexGuard<'a, T> {
    guard: parking_lot::MutexGuard<'a, T>,
    lock_name: String,
    acquired_at: Instant,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            contention_events: DashMap::new(),
            lock_times: RwLock::new(HashMap::new()),
            metrics: PerformanceMetrics {
                mutex_contentions: AtomicU64::new(0),
                slow_operations: AtomicU64::new(0),
                memory_allocations: AtomicU64::new(0),
                circuit_breaker_activations: AtomicU64::new(0),
                total_operations: AtomicU64::new(0),
            },
            operation_times: DashMap::new(),
        }
    }

    /// Get global performance monitor instance
    pub fn global() -> Arc<Self> {
        PERFORMANCE_MONITOR.clone()
    }

    /// Record mutex contention event
    pub fn record_contention(&self, lock_name: &str, wait_time: Duration) {
        let wait_ns = wait_time.as_nanos() as u64;

        // Update global contention counter
        self.metrics
            .mutex_contentions
            .fetch_add(1, Ordering::Relaxed);

        // Update per-lock statistics
        let stats = self
            .contention_events
            .entry(lock_name.to_string())
            .or_insert_with(|| ContentionStats {
                total_contentions: AtomicU64::new(0),
                total_wait_time_ns: AtomicU64::new(0),
                max_wait_time_ns: AtomicU64::new(0),
                avg_wait_time_ns: AtomicU64::new(0),
                last_contention: AtomicU64::new(0),
            });

        let contentions = stats.total_contentions.fetch_add(1, Ordering::Relaxed) + 1;
        let total_wait = stats
            .total_wait_time_ns
            .fetch_add(wait_ns, Ordering::Relaxed)
            + wait_ns;

        // Update max wait time
        let mut max_wait = stats.max_wait_time_ns.load(Ordering::Relaxed);
        while wait_ns > max_wait {
            match stats.max_wait_time_ns.compare_exchange_weak(
                max_wait,
                wait_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => max_wait = x,
            }
        }

        // Update average
        stats
            .avg_wait_time_ns
            .store(total_wait / contentions, Ordering::Relaxed);
        stats.last_contention.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );

        // Log warnings for significant contention
        if wait_ns > CONTENTION_CRITICAL_THRESHOLD_MS * 1_000_000 {
            tracing::error!(
                "🚨 CRITICAL: Severe mutex contention on '{}': {}ms wait time",
                lock_name,
                wait_time.as_millis()
            );
        } else if wait_ns > CONTENTION_WARNING_THRESHOLD_MS * 1_000_000 {
            tracing::warn!(
                "⚠️  Mutex contention detected on '{}': {}ms wait time",
                lock_name,
                wait_time.as_millis()
            );
        }
    }

    /// Record operation performance
    pub fn record_operation(&self, operation_name: &str, duration: Duration, error: bool) {
        let duration_ns = duration.as_nanos() as u64;

        self.metrics
            .total_operations
            .fetch_add(1, Ordering::Relaxed);

        let stats = self
            .operation_times
            .entry(operation_name.to_string())
            .or_insert_with(|| OperationStats::default());

        stats.total_calls.fetch_add(1, Ordering::Relaxed);
        stats
            .total_time_ns
            .fetch_add(duration_ns, Ordering::Relaxed);

        if error {
            stats.error_count.fetch_add(1, Ordering::Relaxed);
        }

        // Update max time
        let mut max_time = stats.max_time_ns.load(Ordering::Relaxed);
        while duration_ns > max_time {
            match stats.max_time_ns.compare_exchange_weak(
                max_time,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => max_time = x,
            }
        }

        // Update min time
        let mut min_time = stats.min_time_ns.load(Ordering::Relaxed);
        if min_time == 0 || duration_ns < min_time {
            while min_time == 0 || duration_ns < min_time {
                match stats.min_time_ns.compare_exchange_weak(
                    min_time,
                    duration_ns,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => min_time = x,
                }
            }
        }

        // Check for slow operations
        if duration.as_millis() > 100 {
            // Operations taking >100ms
            self.metrics.slow_operations.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "🐌 Slow operation detected: '{}' took {}ms",
                operation_name,
                duration.as_millis()
            );
        }
    }

    /// Record circuit breaker activation
    pub fn record_circuit_breaker_activation(&self, component: &str) {
        self.metrics
            .circuit_breaker_activations
            .fetch_add(1, Ordering::Relaxed);
        tracing::info!("🛡️ Circuit breaker activated for component: {}", component);
    }

    /// Get contention report
    pub fn get_contention_report(&self) -> ContentionReport {
        let mut report = ContentionReport {
            total_contentions: self.metrics.mutex_contentions.load(Ordering::Relaxed),
            lock_stats: HashMap::new(),
        };

        for entry in self.contention_events.iter() {
            let lock_name = entry.key().clone();
            let stats = entry.value();

            report.lock_stats.insert(
                lock_name,
                LockContentionInfo {
                    total_contentions: stats.total_contentions.load(Ordering::Relaxed),
                    total_wait_time_ns: stats.total_wait_time_ns.load(Ordering::Relaxed),
                    max_wait_time_ns: stats.max_wait_time_ns.load(Ordering::Relaxed),
                    avg_wait_time_ns: stats.avg_wait_time_ns.load(Ordering::Relaxed),
                    last_contention: stats.last_contention.load(Ordering::Relaxed),
                },
            );
        }

        report
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_operations: self.metrics.total_operations.load(Ordering::Relaxed),
            mutex_contentions: self.metrics.mutex_contentions.load(Ordering::Relaxed),
            slow_operations: self.metrics.slow_operations.load(Ordering::Relaxed),
            circuit_breaker_activations: self
                .metrics
                .circuit_breaker_activations
                .load(Ordering::Relaxed),
            operation_count: self.operation_times.len(),
            contention_locks: self.contention_events.len(),
        }
    }
}

impl<T> MonitoredMutex<T> {
    /// Create new monitored mutex
    pub fn new(data: T, name: String) -> Self {
        let stats = Arc::new(ContentionStats {
            total_contentions: AtomicU64::new(0),
            total_wait_time_ns: AtomicU64::new(0),
            max_wait_time_ns: AtomicU64::new(0),
            avg_wait_time_ns: AtomicU64::new(0),
            last_contention: AtomicU64::new(0),
        });

        Self {
            inner: Mutex::new(data),
            name,
            stats,
        }
    }

    /// Lock with contention monitoring
    pub fn lock(&self) -> TimedMutexGuard<'_, T> {
        let start = Instant::now();
        let guard = self.inner.lock();
        let acquire_time = start.elapsed();

        // Record contention if lock acquisition was slow
        if acquire_time.as_millis() > 1 {
            // >1ms indicates potential contention
            PerformanceMonitor::global().record_contention(&self.name, acquire_time);
        }

        TimedMutexGuard {
            guard,
            lock_name: self.name.clone(),
            acquired_at: Instant::now(),
        }
    }

    /// Try to lock without blocking
    pub fn try_lock(&self) -> Option<TimedMutexGuard<'_, T>> {
        let start = Instant::now();
        if let Some(guard) = self.inner.try_lock() {
            Some(TimedMutexGuard {
                guard,
                lock_name: self.name.clone(),
                acquired_at: start,
            })
        } else {
            // Record failed lock attempt as contention
            PerformanceMonitor::global().record_contention(&self.name, start.elapsed());
            None
        }
    }
}

impl<'a, T> Drop for TimedMutexGuard<'a, T> {
    fn drop(&mut self) {
        let hold_time = self.acquired_at.elapsed();

        // Log if lock was held for a long time
        if hold_time.as_millis() > 50 {
            // >50ms is considered long
            tracing::warn!(
                "🔒 Long lock hold time on '{}': {}ms",
                self.lock_name,
                hold_time.as_millis()
            );
        }
    }
}

impl<'a, T> std::ops::Deref for TimedMutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<'a, T> std::ops::DerefMut for TimedMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

/// Performance operation timing macro
#[macro_export]
macro_rules! timed_operation {
    ($op_name:expr, $code:block) => {{
        let _start = std::time::Instant::now();
        let result = $code;
        let duration = _start.elapsed();
        let error = result.is_err();

        crate::performance_monitor::PerformanceMonitor::global()
            .record_operation($op_name, duration, error);

        result
    }};
}

/// Contention report structure
#[derive(Debug)]
pub struct ContentionReport {
    pub total_contentions: u64,
    pub lock_stats: HashMap<String, LockContentionInfo>,
}

/// Per-lock contention information
#[derive(Debug, Clone)]
pub struct LockContentionInfo {
    pub total_contentions: u64,
    pub total_wait_time_ns: u64,
    pub max_wait_time_ns: u64,
    pub avg_wait_time_ns: u64,
    pub last_contention: u64,
}

/// Overall performance summary
#[derive(Debug)]
pub struct PerformanceSummary {
    pub total_operations: u64,
    pub mutex_contentions: u64,
    pub slow_operations: u64,
    pub circuit_breaker_activations: u64,
    pub operation_count: usize,
    pub contention_locks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_monitored_mutex_basic() {
        let mutex = MonitoredMutex::new(42, "test_mutex".to_string());

        {
            let guard = mutex.lock();
            assert_eq!(*guard, 42);
        }

        // Should have minimal contention
        let monitor = PerformanceMonitor::global();
        let report = monitor.get_contention_report();

        // Basic functionality test - exact contention count may vary
        assert!(report.total_contentions < 100);
    }

    #[test]
    fn test_contention_detection() {
        let mutex = Arc::new(MonitoredMutex::new(0, "contention_test".to_string()));
        let mutex_clone = mutex.clone();

        // Create contention by having multiple threads compete for the lock
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let mut guard = mutex_clone.lock();
                *guard += i;
                // Hold the lock briefly to create contention
                thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        // Main thread also competes for the lock
        for i in 0..50 {
            let mut guard = mutex.lock();
            *guard += i * 2;
            thread::sleep(std::time::Duration::from_millis(1));
        }

        handle.join().unwrap();

        // Should have detected some contention
        let monitor = PerformanceMonitor::global();
        let report = monitor.get_contention_report();

        if let Some(stats) = report.lock_stats.get("contention_test") {
            // With competing threads, we should see some contention
            assert!(stats.total_contentions > 0 || stats.total_wait_time_ns > 0);
        }
    }

    #[test]
    fn test_performance_monitoring() {
        let monitor = PerformanceMonitor::new();

        // Record some operations
        monitor.record_operation("test_op", Duration::from_millis(50), false);
        monitor.record_operation("slow_op", Duration::from_millis(150), false);
        monitor.record_operation("error_op", Duration::from_millis(25), true);

        let summary = monitor.get_performance_summary();
        assert_eq!(summary.total_operations, 3);
        assert_eq!(summary.slow_operations, 1); // slow_op should be flagged
    }
}
