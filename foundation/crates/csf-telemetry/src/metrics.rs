//! Metrics registry and types

use super::*;
use parking_lot::RwLock;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramVec, Registry, TextEncoder,
};
use std::collections::HashMap;

/// Metrics registry
pub struct MetricsRegistry {
    config: MetricsConfig,
    registry: Registry,
    metrics: RwLock<HashMap<String, MetricHandle>>,
}

/// Metric handle
enum MetricHandle {
    Counter(Counter),
    CounterVec(CounterVec),
    Gauge(Gauge),
    GaugeVec(GaugeVec),
    Histogram(Histogram),
    HistogramVec(HistogramVec),
}

/// Metric type
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// Metric definition
#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub help: String,
    pub labels: Vec<String>,
}

impl MetricsRegistry {
    /// Create new metrics registry
    pub fn new(config: &MetricsConfig) -> Result<Self> {
        let registry = Registry::new();

        // Register default metrics
        let mut metrics_registry = Self {
            config: config.clone(),
            registry,
            metrics: RwLock::new(HashMap::new()),
        };

        metrics_registry.register_default_metrics()?;

        Ok(metrics_registry)
    }

    /// Register a metric
    pub fn register(&self, metric: Metric) -> Result<()> {
        let full_name = format!("{}_{}", self.config.prefix, metric.name);

        let handle = match metric.metric_type {
            MetricType::Counter => {
                if metric.labels.is_empty() {
                    let counter = Counter::new(full_name, metric.help)?;
                    self.registry.register(Box::new(counter.clone()))?;
                    MetricHandle::Counter(counter)
                } else {
                    let labels: Vec<&str> = metric.labels.iter().map(|s| s.as_str()).collect();
                    let counter =
                        CounterVec::new(prometheus::Opts::new(full_name, metric.help), &labels)?;
                    self.registry.register(Box::new(counter.clone()))?;
                    MetricHandle::CounterVec(counter)
                }
            }
            MetricType::Gauge => {
                if metric.labels.is_empty() {
                    let gauge = Gauge::new(full_name, metric.help)?;
                    self.registry.register(Box::new(gauge.clone()))?;
                    MetricHandle::Gauge(gauge)
                } else {
                    let labels: Vec<&str> = metric.labels.iter().map(|s| s.as_str()).collect();
                    let gauge =
                        GaugeVec::new(prometheus::Opts::new(full_name, metric.help), &labels)?;
                    self.registry.register(Box::new(gauge.clone()))?;
                    MetricHandle::GaugeVec(gauge)
                }
            }
            MetricType::Histogram => {
                let opts = prometheus::HistogramOpts::new(full_name, metric.help)
                    .buckets(self.config.histogram_bounds.clone());

                if metric.labels.is_empty() {
                    let histogram = Histogram::with_opts(opts)?;
                    self.registry.register(Box::new(histogram.clone()))?;
                    MetricHandle::Histogram(histogram)
                } else {
                    let labels: Vec<&str> = metric.labels.iter().map(|s| s.as_str()).collect();
                    let histogram = HistogramVec::new(opts, &labels)?;
                    self.registry.register(Box::new(histogram.clone()))?;
                    MetricHandle::HistogramVec(histogram)
                }
            }
        };

        self.metrics.write().insert(metric.name, handle);
        Ok(())
    }

    /// Record a metric value
    pub fn record(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let metrics = self.metrics.read();

        if let Some(handle) = metrics.get(name) {
            match handle {
                MetricHandle::Counter(counter) => {
                    counter.inc_by(value);
                }
                MetricHandle::CounterVec(counter) => {
                    let label_values: Vec<&str> = labels.iter().map(|(_, v)| *v).collect();
                    if let Ok(metric) = counter.get_metric_with_label_values(&label_values) {
                        metric.inc_by(value);
                    }
                }
                MetricHandle::Gauge(gauge) => {
                    gauge.set(value);
                }
                MetricHandle::GaugeVec(gauge) => {
                    let label_values: Vec<&str> = labels.iter().map(|(_, v)| *v).collect();
                    if let Ok(metric) = gauge.get_metric_with_label_values(&label_values) {
                        metric.set(value);
                    }
                }
                MetricHandle::Histogram(histogram) => {
                    histogram.observe(value);
                }
                MetricHandle::HistogramVec(histogram) => {
                    let label_values: Vec<&str> = labels.iter().map(|(_, v)| *v).collect();
                    if let Ok(metric) = histogram.get_metric_with_label_values(&label_values) {
                        metric.observe(value);
                    }
                }
            }
        }
    }

    /// Get metrics as Prometheus text format
    pub fn gather_prometheus(&self) -> Result<String> {
        let mut buffer = Vec::new();
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    /// Get metrics statistics
    pub fn get_stats(&self) -> MetricsStats {
        MetricsStats {
            active_metrics: self.metrics.read().len(),
        }
    }

    /// Register default CSF metrics
    fn register_default_metrics(&mut self) -> Result<()> {
        // Packet metrics
        self.register(Metric {
            name: "packets_processed_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total number of packets processed".to_string(),
            labels: vec!["type".to_string()],
        })?;

        self.register(Metric {
            name: "packet_processing_duration_seconds".to_string(),
            metric_type: MetricType::Histogram,
            help: "Packet processing duration in seconds".to_string(),
            labels: vec!["type".to_string()],
        })?;

        // Task metrics
        self.register(Metric {
            name: "tasks_scheduled_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total number of tasks scheduled".to_string(),
            labels: vec![],
        })?;

        self.register(Metric {
            name: "task_queue_size".to_string(),
            metric_type: MetricType::Gauge,
            help: "Current task queue size".to_string(),
            labels: vec!["priority".to_string()],
        })?;

        // C-LOGIC metrics
        self.register(Metric {
            name: "clogic_drpp_coherence".to_string(),
            metric_type: MetricType::Gauge,
            help: "DRPP coherence level".to_string(),
            labels: vec![],
        })?;

        self.register(Metric {
            name: "clogic_adp_load".to_string(),
            metric_type: MetricType::Gauge,
            help: "ADP processing load".to_string(),
            labels: vec![],
        })?;

        self.register(Metric {
            name: "clogic_egc_decisions_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total EGC decisions made".to_string(),
            labels: vec!["type".to_string()],
        })?;

        self.register(Metric {
            name: "clogic_ems_valence".to_string(),
            metric_type: MetricType::Gauge,
            help: "EMS emotional valence".to_string(),
            labels: vec![],
        })?;

        self.register(Metric {
            name: "clogic_ems_arousal".to_string(),
            metric_type: MetricType::Gauge,
            help: "EMS emotional arousal".to_string(),
            labels: vec![],
        })?;

        // Network metrics
        self.register(Metric {
            name: "network_bytes_sent_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total bytes sent over network".to_string(),
            labels: vec!["peer".to_string()],
        })?;

        self.register(Metric {
            name: "network_bytes_received_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total bytes received over network".to_string(),
            labels: vec!["peer".to_string()],
        })?;

        self.register(Metric {
            name: "network_connections_active".to_string(),
            metric_type: MetricType::Gauge,
            help: "Number of active network connections".to_string(),
            labels: vec![],
        })?;

        // MLIR metrics
        self.register(Metric {
            name: "mlir_compilations_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total MLIR compilations".to_string(),
            labels: vec!["backend".to_string()],
        })?;

        self.register(Metric {
            name: "mlir_compilation_duration_seconds".to_string(),
            metric_type: MetricType::Histogram,
            help: "MLIR compilation duration".to_string(),
            labels: vec!["backend".to_string()],
        })?;

        self.register(Metric {
            name: "mlir_executions_total".to_string(),
            metric_type: MetricType::Counter,
            help: "Total MLIR executions".to_string(),
            labels: vec!["module".to_string()],
        })?;

        self.register(Metric {
            name: "mlir_execution_duration_seconds".to_string(),
            metric_type: MetricType::Histogram,
            help: "MLIR execution duration".to_string(),
            labels: vec!["module".to_string()],
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetricsStats {
    pub active_metrics: usize,
}
