//! Observability metrics for the Phase Coherence Bus
//!
//! Provides instrumentation for monitoring bus performance and health.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Metrics collector for the Phase Coherence Bus
#[derive(Debug, Default)]
pub struct BusMetrics {
    /// Total number of packets published to the bus
    pub publish_total: AtomicU64,
    /// Total number of subscriptions created
    pub subscribe_total: AtomicU64,
    /// Total publish-to-receive latency in nanoseconds
    pub latency_total_ns: AtomicU64,
    /// Number of latency measurements
    pub latency_count: AtomicU64,
    /// Peak latency observed in nanoseconds
    pub peak_latency_ns: AtomicU64,
    /// Total number of packets dropped
    pub packets_dropped: AtomicU64,
    /// Total number of active subscriptions
    pub active_subscriptions: AtomicU64,
}

impl BusMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a packet publish
    pub fn record_publish(&self) {
        self.publish_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a new subscription
    pub fn record_subscribe(&self) {
        self.subscribe_total.fetch_add(1, Ordering::Relaxed);
        self.active_subscriptions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a subscription cancellation
    pub fn record_unsubscribe(&self) {
        self.active_subscriptions.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record message latency
    pub fn record_latency(&self, latency_ns: u64) {
        self.latency_total_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);

        // Update peak latency
        let current_peak = self.peak_latency_ns.load(Ordering::Relaxed);
        if latency_ns > current_peak {
            let _ = self.peak_latency_ns.compare_exchange_weak(
                current_peak,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }

    /// Record a dropped packet
    pub fn record_drop(&self) {
        self.packets_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the average latency in nanoseconds
    pub fn average_latency_ns(&self) -> u64 {
        let total = self.latency_total_ns.load(Ordering::Relaxed);
        let count = self.latency_count.load(Ordering::Relaxed);
        if count > 0 {
            total / count
        } else {
            0
        }
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            publish_total: self.publish_total.load(Ordering::Relaxed),
            subscribe_total: self.subscribe_total.load(Ordering::Relaxed),
            average_latency_ns: self.average_latency_ns(),
            peak_latency_ns: self.peak_latency_ns.load(Ordering::Relaxed),
            packets_dropped: self.packets_dropped.load(Ordering::Relaxed),
            active_subscriptions: self.active_subscriptions.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total packets published
    pub publish_total: u64,
    /// Total subscriptions created
    pub subscribe_total: u64,
    /// Average latency in nanoseconds
    pub average_latency_ns: u64,
    /// Peak latency in nanoseconds
    pub peak_latency_ns: u64,
    /// Total packets dropped
    pub packets_dropped: u64,
    /// Current active subscriptions
    pub active_subscriptions: u64,
}

/// Shared metrics instance
static GLOBAL_METRICS: once_cell::sync::Lazy<Arc<BusMetrics>> =
    once_cell::sync::Lazy::new(|| Arc::new(BusMetrics::new()));

/// Get the global metrics instance
pub fn get_metrics() -> Arc<BusMetrics> {
    GLOBAL_METRICS.clone()
}

/// Initialize metrics collection
pub fn init_metrics() {
    once_cell::sync::Lazy::force(&GLOBAL_METRICS);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = BusMetrics::new();

        metrics.record_publish();
        metrics.record_subscribe();
        metrics.record_latency(1000);
        metrics.record_latency(2000);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.publish_total, 1);
        assert_eq!(snapshot.subscribe_total, 1);
        assert_eq!(snapshot.average_latency_ns, 1500);
        assert_eq!(snapshot.peak_latency_ns, 2000);
    }
}
