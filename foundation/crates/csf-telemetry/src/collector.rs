//! Metrics collector implementation

use super::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use sysinfo::System;

/// Collector configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollectorConfig {
    /// Enable CPU metrics
    pub collect_cpu: bool,

    /// Enable memory metrics
    pub collect_memory: bool,

    /// Enable disk metrics
    pub collect_disk: bool,

    /// Enable network metrics
    pub collect_network: bool,

    /// Enable GPU metrics
    pub collect_gpu: bool,

    /// Collection buffer size
    pub buffer_size: usize,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            collect_cpu: true,
            collect_memory: true,
            collect_disk: true,
            collect_network: true,
            collect_gpu: false,
            buffer_size: 10000,
        }
    }
}

/// Metrics collector
pub struct Collector {
    config: CollectorConfig,
    #[allow(dead_code)]
    metrics_registry: Arc<MetricsRegistry>,
    system: Arc<RwLock<System>>,
    collection_buffer: Arc<RwLock<CollectionBuffer>>,
    #[cfg(feature = "nvidia-gpu")]
    nvml: Option<nvml_wrapper::Nvml>,
}

struct CollectionBuffer {
    metrics: Vec<CollectedMetric>,
    capacity: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectedMetric {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp: u64,
}

impl Collector {
    pub fn new(config: &CollectorConfig, metrics_registry: Arc<MetricsRegistry>) -> Result<Self> {
        let system = System::new_all();

        #[cfg(feature = "nvidia-gpu")]
        let nvml = if config.collect_gpu {
            nvml_wrapper::Nvml::init().ok()
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            metrics_registry,
            system: Arc::new(RwLock::new(system)),
            collection_buffer: Arc::new(RwLock::new(CollectionBuffer {
                metrics: Vec::with_capacity(config.buffer_size),
                capacity: config.buffer_size,
            })),
            #[cfg(feature = "nvidia-gpu")]
            nvml,
        })
    }

    /// Start collector
    pub async fn start(&self) -> Result<()> {
        // Start background collection if needed
        Ok(())
    }

    /// Stop collector
    pub async fn stop(&self) -> Result<()> {
        Ok(())
    }

    /// Collect all metrics
    pub async fn collect_all(&self) -> Result<Vec<CollectedMetric>> {
        let mut metrics = Vec::new();

        // Collect system metrics
        if self.config.collect_cpu
            || self.config.collect_memory
            || self.config.collect_disk
            || self.config.collect_network
        {
            let system_metrics = self.collect_system_metrics().await;
            metrics.extend(self.system_metrics_to_collected(&system_metrics));
        }

        // Get buffered metrics
        let mut buffer = self.collection_buffer.write();
        metrics.append(&mut buffer.metrics);

        Ok(metrics)
    }

    /// Collect system metrics
    pub async fn collect_system_metrics(&self) -> SystemMetrics {
        let mut system = self.system.write();
        system.refresh_all();

        let cpu_usage = system.global_cpu_usage();
        let memory_used = system.used_memory();
        let memory_total = system.total_memory();

        // Calculate disk I/O - Using placeholder for sysinfo 0.31 compatibility
        // Full implementation requires proper disk and network metrics API
        let (disk_read_bps, disk_write_bps) = (0u64, 0u64);

        // Calculate network I/O - Using placeholder for sysinfo 0.31 compatibility
        // Full implementation requires proper disk and network metrics API
        let (network_rx_bps, network_tx_bps) = (0u64, 0u64);

        // Collect GPU metrics if available
        #[cfg(feature = "nvidia-gpu")]
        let gpu_metrics = if let Some(nvml) = &self.nvml {
            self.collect_gpu_metrics(nvml)
        } else {
            None
        };

        #[cfg(not(feature = "nvidia-gpu"))]
        let gpu_metrics = None;

        SystemMetrics {
            cpu_usage: cpu_usage as f64,
            memory_used: memory_used * 1024, // Convert to bytes
            memory_total: memory_total * 1024,
            disk_read_bps,
            disk_write_bps,
            network_rx_bps,
            network_tx_bps,
            gpu_metrics,
        }
    }

    #[cfg(feature = "nvidia-gpu")]
    fn collect_gpu_metrics(&self, nvml: &nvml_wrapper::Nvml) -> Option<GpuMetrics> {
        if let Ok(device) = nvml.device_by_index(0) {
            let utilization = device.utilization_rates().ok()?;
            let memory_info = device.memory_info().ok()?;
            let temperature = device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok()?;
            let power = device.power_usage().ok()? as f64 / 1000.0; // Convert to watts

            Some(GpuMetrics {
                utilization: utilization.gpu as f64,
                memory_used: memory_info.used,
                memory_total: memory_info.total,
                temperature: temperature as f64,
                power_watts: power,
            })
        } else {
            None
        }
    }

    /// Convert system metrics to collected metrics
    fn system_metrics_to_collected(&self, system: &SystemMetrics) -> Vec<CollectedMetric> {
        let mut metrics = Vec::new();
        let timestamp = super::now_nanos();

        if self.config.collect_cpu {
            metrics.push(CollectedMetric {
                name: "system_cpu_usage_percent".to_string(),
                value: system.cpu_usage,
                labels: HashMap::new(),
                timestamp,
            });
        }

        if self.config.collect_memory {
            metrics.push(CollectedMetric {
                name: "system_memory_used_bytes".to_string(),
                value: system.memory_used as f64,
                labels: HashMap::new(),
                timestamp,
            });

            metrics.push(CollectedMetric {
                name: "system_memory_total_bytes".to_string(),
                value: system.memory_total as f64,
                labels: HashMap::new(),
                timestamp,
            });
        }

        if self.config.collect_disk {
            metrics.push(CollectedMetric {
                name: "system_disk_read_bytes_per_second".to_string(),
                value: system.disk_read_bps as f64,
                labels: HashMap::new(),
                timestamp,
            });

            metrics.push(CollectedMetric {
                name: "system_disk_write_bytes_per_second".to_string(),
                value: system.disk_write_bps as f64,
                labels: HashMap::new(),
                timestamp,
            });
        }

        if self.config.collect_network {
            metrics.push(CollectedMetric {
                name: "system_network_receive_bytes_per_second".to_string(),
                value: system.network_rx_bps as f64,
                labels: HashMap::new(),
                timestamp,
            });
            metrics.push(CollectedMetric {
                name: "system_network_transmit_bytes_per_second".to_string(),
                value: system.network_tx_bps as f64,
                labels: HashMap::new(),
                timestamp,
            });
        }

        if let Some(gpu) = &system.gpu_metrics {
            if self.config.collect_gpu {
                metrics.push(CollectedMetric {
                    name: "system_gpu_utilization_percent".to_string(),
                    value: gpu.utilization,
                    labels: HashMap::new(),
                    timestamp,
                });

                metrics.push(CollectedMetric {
                    name: "system_gpu_memory_used_bytes".to_string(),
                    value: gpu.memory_used as f64,
                    labels: HashMap::new(),
                    timestamp,
                });

                metrics.push(CollectedMetric {
                    name: "system_gpu_temperature_celsius".to_string(),
                    value: gpu.temperature,
                    labels: HashMap::new(),
                    timestamp,
                });

                metrics.push(CollectedMetric {
                    name: "system_gpu_power_watts".to_string(),
                    value: gpu.power_watts,
                    labels: HashMap::new(),
                    timestamp,
                });
            }
        }

        metrics
    }

    /// Buffer a metric for collection
    pub fn buffer_metric(&self, metric: CollectedMetric) {
        let mut buffer = self.collection_buffer.write();

        if buffer.metrics.len() >= buffer.capacity {
            // Remove oldest metrics
            let capacity = buffer.capacity;
            buffer.metrics.drain(0..capacity / 2);
        }

        buffer.metrics.push(metric);
    }
}
