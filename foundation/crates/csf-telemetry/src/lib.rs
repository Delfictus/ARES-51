//! Telemetry and metrics collection for ARES CSF
//!
//! Provides comprehensive observability through metrics, tracing, and logging
//! with support for OpenTelemetry and Prometheus.

use csf_time::{global_time_source, NanoTime};
use parking_lot::RwLock;
use std::sync::{Arc, Mutex, OnceLock};

/// Telemetry error type
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Metrics error: {0}")]
    Metrics(String),
    #[error("Tracing error: {0}")]
    Tracing(String),
    #[error("Export error: {0}")]
    Export(String),
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Request error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Telemetry result type
pub type Result<T> = std::result::Result<T, TelemetryError>;

pub mod aggregator;
pub mod collector;
pub mod exporter;
pub mod metrics;
pub mod success_metrics;
pub mod tracing;

pub use collector::{Collector, CollectorConfig};
pub use exporter::{Exporter, ExporterConfig};
pub use metrics::{Metric, MetricType, MetricsRegistry};
pub use success_metrics::{CsfSuccessMetrics, LatencyMeasurement, LatencyTracker};
pub use tracing::{SpanContext, TracingConfig};

/// Monotonic-ish wallclock in nanoseconds for telemetry timestamps
/// Get current timestamp using TTW TimeSource for ChronoSynclastic determinism
#[inline]
fn now_nanos() -> u64 {
    global_time_source()
        .now_ns()
        .unwrap_or(NanoTime::ZERO)
        .as_nanos()
}

/// Telemetry configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TelemetryConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,

    /// Enable distributed tracing
    pub enable_tracing: bool,

    /// Enable system metrics
    pub enable_system_metrics: bool,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Tracing configuration
    pub tracing: TracingConfig,

    /// Collector configuration
    pub collector: CollectorConfig,

    /// Exporter configuration
    pub exporter: ExporterConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsConfig {
    /// Collection interval (ms)
    pub collection_interval_ms: u64,

    /// Histogram bounds
    pub histogram_bounds: Vec<f64>,

    /// Enable detailed metrics
    pub detailed_metrics: bool,

    /// Metric prefix
    pub prefix: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            enable_system_metrics: true,
            metrics: MetricsConfig {
                collection_interval_ms: 1000,
                histogram_bounds: vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                detailed_metrics: false,
                prefix: "csf".to_string(),
            },
            tracing: TracingConfig::default(),
            collector: CollectorConfig::default(),
            exporter: ExporterConfig::default(),
        }
    }
}

/// Telemetry system
pub struct TelemetrySystem {
    config: TelemetryConfig,
    metrics_registry: Arc<MetricsRegistry>,
    tracer: Arc<tracing::Tracer>,
    collector: Arc<Collector>,
    exporter: Arc<Exporter>,
    state: Arc<RwLock<TelemetryState>>,
}

#[derive(Debug, Default)]
struct TelemetryState {
    running: bool,
    metrics_collected: u64,
    traces_collected: u64,
    last_collection_time: u64,
}

impl TelemetrySystem {
    /// Create new telemetry system
    pub async fn new(config: TelemetryConfig) -> Result<Self> {
        // Initialize metrics registry
        let metrics_registry = Arc::new(MetricsRegistry::new(&config.metrics)?);

        // Initialize tracer
        let tracer = Arc::new(tracing::Tracer::new(&config.tracing).await?);

        // Initialize collector
        let collector = Arc::new(Collector::new(&config.collector, metrics_registry.clone())?);

        // Initialize exporter
        let exporter = Arc::new(Exporter::new(&config.exporter).await?);

        Ok(Self {
            config,
            metrics_registry,
            tracer,
            collector,
            exporter,
            state: Arc::new(RwLock::new(TelemetryState::default())),
        })
    }

    /// Start telemetry system
    pub async fn start(&self) -> Result<()> {
        {
            let state = self.state.read();
            if state.running {
                return Ok(());
            }
        }

        // Start collector
        self.collector.start().await?;

        // Start exporter
        self.exporter.start().await?;

        // Start collection loop
        let self_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            self_clone.collection_loop().await;
        });

        {
            let mut state = self.state.write();
            state.running = true;
        }
        Ok(())
    }

    /// Stop telemetry system
    pub async fn stop(&self) -> Result<()> {
        {
            let state = self.state.read();
            if !state.running {
                return Ok(());
            }
        }

        // Stop collector
        self.collector.stop().await?;

        // Stop exporter
        self.exporter.stop().await?;

        // Flush remaining data
        self.flush().await?;

        {
            let mut state = self.state.write();
            state.running = false;
        }
        Ok(())
    }

    /// Record a metric
    pub fn record_metric(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        self.metrics_registry.record(name, value, labels);

        let mut state = self.state.write();
        state.metrics_collected += 1;
    }

    /// Start a trace span
    pub fn start_span(&self, name: &str) -> Span {
        let span = self.tracer.start_span(name);

        let mut state = self.state.write();
        state.traces_collected += 1;

        span
    }

    /// Get system metrics
    pub async fn get_system_metrics(&self) -> SystemMetrics {
        self.collector.collect_system_metrics().await
    }

    /// Get telemetry statistics
    pub fn get_stats(&self) -> TelemetryStats {
        let state = self.state.read();
        let metrics_stats = self.metrics_registry.get_stats();

        TelemetryStats {
            metrics_collected: state.metrics_collected,
            traces_collected: state.traces_collected,
            active_metrics: metrics_stats.active_metrics,
            active_spans: self.tracer.active_spans(),
            last_export_time: self.exporter.last_export_time(),
        }
    }

    /// Flush all pending telemetry data
    pub async fn flush(&self) -> Result<()> {
        // Collect final metrics
        let metrics = self.collector.collect_all().await?;

        // Export metrics
        self.exporter.export_metrics(&metrics).await?;

        // Flush traces
        self.tracer.flush().await?;

        Ok(())
    }

    /// Collection loop
    async fn collection_loop(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(
            self.config.metrics.collection_interval_ms,
        ));

        loop {
            interval.tick().await;

            if !self.state.read().running {
                break;
            }

            // Collect metrics
            if let Ok(metrics) = self.collector.collect_all().await {
                // Export metrics
                if let Err(e) = self.exporter.export_metrics(&metrics).await {
                    ::tracing::error!("Failed to export metrics: {}", e);
                }
            }

            // Update state
            let mut state = self.state.write();
            state.last_collection_time = now_nanos();
        }
    }
}

impl Clone for TelemetrySystem {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_registry: self.metrics_registry.clone(),
            tracer: self.tracer.clone(),
            collector: self.collector.clone(),
            exporter: self.exporter.clone(),
            state: self.state.clone(),
        }
    }
}

/// System metrics
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,

    /// Memory usage in bytes
    pub memory_used: u64,

    /// Memory total in bytes
    pub memory_total: u64,

    /// Disk I/O read bytes/sec
    pub disk_read_bps: u64,

    /// Disk I/O write bytes/sec
    pub disk_write_bps: u64,

    /// Network receive bytes/sec
    pub network_rx_bps: u64,

    /// Network transmit bytes/sec
    pub network_tx_bps: u64,

    /// GPU metrics (if available)
    pub gpu_metrics: Option<GpuMetrics>,
}

#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization: f64,

    /// GPU memory used in bytes
    pub memory_used: u64,

    /// GPU memory total in bytes
    pub memory_total: u64,

    /// GPU temperature in Celsius
    pub temperature: f64,

    /// GPU power usage in watts
    pub power_watts: f64,
}

/// Telemetry statistics
#[derive(Debug, Clone)]
pub struct TelemetryStats {
    pub metrics_collected: u64,
    pub traces_collected: u64,
    pub active_metrics: usize,
    pub active_spans: usize,
    pub last_export_time: Option<u64>,
}

// ... existing code ...

/// Span for distributed tracing
pub struct Span {
    inner: opentelemetry::global::BoxedSpan,
    #[allow(dead_code)]
    start_time: u64,
}

impl Span {
    /// Set span attribute
    pub fn set_attribute(&mut self, key: String, value: opentelemetry::Value) {
        use opentelemetry::trace::Span as _;
        // BoxedSpan requires 'static; allocate owned key/value
        self.inner
            .set_attribute(opentelemetry::KeyValue::new(key, value));
    }

    /// Record an event
    pub fn record_event(&mut self, name: String, attributes: Vec<(String, String)>) {
        use opentelemetry::trace::Span as _;
        let attrs: Vec<opentelemetry::KeyValue> = attributes
            .into_iter()
            .map(|(k, v)| opentelemetry::KeyValue::new(k, v))
            .collect();

        self.inner.add_event(name, attrs);
    }

    /// End the span
    pub fn end(&mut self) {
        use opentelemetry::trace::Span as _;
        self.inner.end();
    }
}

/// Global telemetry instance
static TELEMETRY: OnceLock<Mutex<Option<Arc<TelemetrySystem>>>> = OnceLock::new();

/// Initialize global telemetry
pub async fn init(config: TelemetryConfig) -> Result<()> {
    // Check if already initialized without holding lock
    if let Some(mutex) = TELEMETRY.get() {
        let guard = match mutex.lock() {
            Ok(g) => g,
            Err(_) => return Err(TelemetryError::Config("Telemetry lock poisoned".into())),
        };
        if guard.is_some() {
            return Err(TelemetryError::Config(
                "Telemetry already initialized".into(),
            ));
        }
    }

    // Create telemetry system
    let telemetry = Arc::new(TelemetrySystem::new(config).await?);
    telemetry.start().await?;

    // Store in global
    let telemetry_mutex = TELEMETRY.get_or_init(|| Mutex::new(None));
    let mut telemetry_guard = match telemetry_mutex.lock() {
        Ok(g) => g,
        Err(_) => return Err(TelemetryError::Config("Telemetry lock poisoned".into())),
    };
    *telemetry_guard = Some(telemetry);

    Ok(())
}

/// Get global telemetry instance
pub fn telemetry() -> Option<Arc<TelemetrySystem>> {
    let mutex = TELEMETRY.get()?;
    match mutex.lock() {
        Ok(guard) => guard.clone(),
        Err(_) => None,
    }
}

/// Record a metric using global telemetry
pub fn record_metric(name: &str, value: f64, labels: &[(&str, &str)]) {
    if let Some(telemetry) = telemetry() {
        telemetry.record_metric(name, value, labels);
    }
}

/// Start a span using global telemetry
pub fn start_span(name: &str) -> Option<Span> {
    telemetry().map(|t| t.start_span(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_system() {
        let config = TelemetryConfig::default();
        let telemetry = TelemetrySystem::new(config).await.unwrap();

        telemetry.start().await.unwrap();

        // Record some metrics
        telemetry.record_metric("test_counter", 1.0, &[("label", "value")]);
        telemetry.record_metric("test_gauge", 42.0, &[]);

        // Start a span
        let mut span = telemetry.start_span("test_operation");
        span.set_attribute(
            "test_attr".to_string(),
            opentelemetry::Value::from("test_value"),
        );
        span.end();

        // Get stats
        let stats = telemetry.get_stats();
        assert!(stats.metrics_collected > 0);
        assert!(stats.traces_collected > 0);

        telemetry.stop().await.unwrap();
    }
}
