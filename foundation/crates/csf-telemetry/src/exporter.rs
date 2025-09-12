//! Metrics exporter implementation

use super::*;
use crate::collector::CollectedMetric;
use opentelemetry_prometheus::PrometheusExporter;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

/// Exporter configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExporterConfig {
    /// Export format
    pub format: ExportFormat,

    /// Export endpoint
    pub endpoint: String,

    /// Export interval (ms)
    pub export_interval_ms: u64,

    /// Batch size
    pub batch_size: usize,

    /// Enable compression
    pub enable_compression: bool,

    /// Retry configuration
    pub retry_config: RetryConfig,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    Prometheus,
    OpenTelemetry,
    Json,
    InfluxDB,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_interval_ms: u64,
    pub max_interval_ms: u64,
    pub multiplier: f64,
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Prometheus,
            endpoint: "http://localhost:9090/api/v1/write".to_string(),
            export_interval_ms: 10000,
            batch_size: 1000,
            enable_compression: true,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_interval_ms: 1000,
                max_interval_ms: 30000,
                multiplier: 2.0,
            },
        }
    }
}

/// Metrics exporter
pub struct Exporter {
    config: ExporterConfig,
    export_queue: mpsc::Sender<Vec<CollectedMetric>>,
    queue_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<Vec<CollectedMetric>>>>,
    last_export_time: AtomicU64,
    #[allow(dead_code)]
    prometheus_exporter: Option<PrometheusExporter>,
}

impl Exporter {
    /// Create new exporter
    pub async fn new(config: &ExporterConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel(100);

        let prometheus_exporter = if matches!(config.format, ExportFormat::Prometheus) {
            Some(
                opentelemetry_prometheus::exporter()
                    .build()
                    .map_err(|e| TelemetryError::Export(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            export_queue: tx,
            queue_receiver: Arc::new(tokio::sync::Mutex::new(rx)),
            last_export_time: AtomicU64::new(0),
            prometheus_exporter,
        })
    }

    /// Start exporter
    pub async fn start(&self) -> Result<()> {
        let config = self.config.clone();
        let receiver = self.queue_receiver.clone();

        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut rx = receiver.lock().await;

            while let Some(metrics) = rx.recv().await {
                batch.extend(metrics);

                if batch.len() >= config.batch_size {
                    if let Err(e) = export_batch(&config, &batch).await {
                        ::tracing::error!("Failed to export metrics: {}", e);
                    }
                    batch.clear();
                }
            }

            // Export remaining metrics
            if !batch.is_empty() {
                let _ = export_batch(&config, &batch).await;
            }
        });

        Ok(())
    }

    /// Stop exporter
    pub async fn stop(&self) -> Result<()> {
        // Exporter will stop when channel is closed
        Ok(())
    }

    /// Export metrics
    pub async fn export_metrics(&self, metrics: &[CollectedMetric]) -> Result<()> {
        self.export_queue
            .send(metrics.to_vec())
            .await
            .map_err(|e| anyhow::anyhow!("Export queue closed: {}", e))?;
        Ok(())
    }

    /// Get last export time
    pub fn last_export_time(&self) -> Option<u64> {
        let time = self.last_export_time.load(Ordering::Relaxed);
        if time > 0 {
            Some(time)
        } else {
            None
        }
    }
}

/// Export a batch of metrics
async fn export_batch(config: &ExporterConfig, metrics: &[CollectedMetric]) -> Result<()> {
    match config.format {
        ExportFormat::Prometheus => export_prometheus(config, metrics).await,
        ExportFormat::OpenTelemetry => export_otlp(config, metrics).await,
        ExportFormat::Json => export_json(config, metrics).await,
        ExportFormat::InfluxDB => export_influxdb(config, metrics).await,
    }
}

/// Export to Prometheus
async fn export_prometheus(config: &ExporterConfig, metrics: &[CollectedMetric]) -> Result<()> {
    use reqwest::header::{CONTENT_ENCODING, CONTENT_TYPE};

    // Convert to Prometheus format
    let mut lines = Vec::new();

    for metric in metrics {
        let labels = if metric.labels.is_empty() {
            String::new()
        } else {
            let label_pairs: Vec<String> = metric
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", label_pairs.join(","))
        };

        lines.push(format!(
            "{}{} {} {}",
            metric.name,
            labels,
            metric.value,
            metric.timestamp / 1_000_000 // Convert to milliseconds
        ));
    }

    let body = lines.join("\n");

    // Compress if enabled
    let (body, encoding) = if config.enable_compression {
        let compressed = zstd::encode_all(body.as_bytes(), 3)?;
        (compressed, Some("zstd"))
    } else {
        (body.into_bytes(), None)
    };

    // Send request with retries
    let client = reqwest::Client::new();
    let mut retry_count = 0;
    let mut interval = config.retry_config.initial_interval_ms;

    loop {
        let mut request = client
            .post(&config.endpoint)
            .header(CONTENT_TYPE, "text/plain")
            .body(body.clone());

        if let Some(enc) = encoding {
            request = request.header(CONTENT_ENCODING, enc);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();

                if retry_count >= config.retry_config.max_retries {
                    return Err(anyhow::anyhow!("Export failed: {} - {}", status, text).into());
                }
            }
            Err(e) => {
                if retry_count >= config.retry_config.max_retries {
                    return Err(e.into());
                }
            }
        }

        // Exponential backoff
        tokio::time::sleep(std::time::Duration::from_millis(interval)).await;
        interval = (interval as f64 * config.retry_config.multiplier) as u64;
        interval = interval.min(config.retry_config.max_interval_ms);
        retry_count += 1;
    }
}

/// Export to OpenTelemetry
async fn export_otlp(_config: &ExporterConfig, metrics: &[CollectedMetric]) -> Result<()> {
    // In a real implementation, this would use the OTLP protocol
    ::tracing::debug!("Exporting {} metrics via OTLP", metrics.len());
    Ok(())
}

/// Export as JSON
async fn export_json(config: &ExporterConfig, metrics: &[CollectedMetric]) -> Result<()> {
    let json = serde_json::to_string(metrics)?;

    let client = reqwest::Client::new();
    client
        .post(&config.endpoint)
        .header("Content-Type", "application/json")
        .body(json)
        .send()
        .await?;

    Ok(())
}

/// Export to InfluxDB
async fn export_influxdb(config: &ExporterConfig, metrics: &[CollectedMetric]) -> Result<()> {
    // Convert to InfluxDB line protocol
    let mut lines = Vec::new();

    for metric in metrics {
        let tags = metric
            .labels
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",");

        let line = if tags.is_empty() {
            format!(
                "{} value={} {}",
                metric.name, metric.value, metric.timestamp
            )
        } else {
            format!(
                "{},{} value={} {}",
                metric.name, tags, metric.value, metric.timestamp
            )
        };

        lines.push(line);
    }

    let body = lines.join("\n");

    let client = reqwest::Client::new();
    client
        .post(&config.endpoint)
        .header("Content-Type", "text/plain")
        .body(body)
        .send()
        .await?;

    Ok(())
}
