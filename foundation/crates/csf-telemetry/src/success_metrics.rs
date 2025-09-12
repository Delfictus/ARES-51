//! Success metrics implementation for ARES ChronoSynclastic Fabric
//!
//! This module provides concrete implementations of the success metrics framework
//! defined in docs/SUCCESS_METRICS.md, with specific focus on the critical KPIs
//! for NovaCore architecture validation.

use crate::{Result, TelemetryError};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Opts};
use std::sync::Arc;
use std::time::Instant;

/// Critical performance metrics for NovaCore ChronoSynclastic Fabric
pub struct CsfSuccessMetrics {
    // Core Latency Metrics (Critical KPIs)
    pub ttw_time_source_query_duration: Histogram,
    pub hlc_operation_duration: Histogram,
    pub pcb_message_routing_duration: Histogram,
    pub phase_packet_creation_duration: Histogram,
    pub quantum_oracle_duration: Histogram,
    pub task_scheduling_duration: Histogram,
    pub e2e_processing_duration: Histogram,

    // Throughput Metrics (Critical KPIs)
    pub pcb_messages_processed_total: Counter,
    pub time_operations_total: Counter,
    pub hlc_updates_total: Counter,
    pub scheduler_tasks_scheduled_total: Counter,
    pub clogic_inferences_total: Counter,
    pub mlir_kernels_executed_total: Counter,

    // Temporal Coherence Accuracy (Mission-Critical)
    pub causality_violations_total: Counter,
    pub hlc_drift_ns_per_hour: Gauge,
    pub quantum_time_deviation_ns: Gauge,
    pub distributed_coherence_ratio: Gauge,
    pub deadline_miss_ratio: Gauge,

    // Hardware Acceleration Efficiency
    pub gpu_utilization_percent: Gauge,
    pub mlir_backend_efficiency_ratio: Gauge,
    pub hardware_memory_usage_ratio: Gauge,
    pub kernel_launch_overhead_ratio: Gauge,
    pub data_transfer_efficiency_ratio: Gauge,

    // System Reliability
    pub uptime_ratio: Gauge,
    pub memory_violations_total: Counter,
    pub panic_events_total: Counter,
    pub error_rate_ratio: Gauge,
    pub recovery_duration_ms: Histogram,
    pub data_integrity_violations_total: Counter,

    // Operational Metrics
    pub service_availability_ratio: Gauge,
    pub component_health_score: Gauge,
    pub network_connectivity_ratio: Gauge,
    pub consensus_participation_ratio: Gauge,

    // Resource Utilization
    pub cpu_utilization_percent: Gauge,
    pub memory_utilization_ratio: Gauge,
    pub disk_io_utilization_ratio: Gauge,
    pub network_io_utilization_ratio: Gauge,
    pub thread_pool_utilization_ratio: Gauge,

    // C-LOGIC and Neuromorphic Integration
    pub clogic_accuracy_ratio: Gauge,
    pub drpp_detection_accuracy: Gauge,
    pub adp_adaptation_duration_ms: Histogram,
    pub egc_convergence_duration_ms: Histogram,
    pub ems_modeling_accuracy: Gauge,
}

impl CsfSuccessMetrics {
    /// Create new CSF success metrics collection
    pub fn new() -> Result<Self> {
        Ok(Self {
            // Core Latency Metrics - Sub-microsecond precision buckets
            ttw_time_source_query_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_time_source_query_duration_ns",
                    "TTW TimeSource query duration in nanoseconds",
                )
                .buckets(vec![
                    10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 800.0, 1000.0, 2000.0,
                ]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            hlc_operation_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_hlc_operation_duration_ns",
                    "HLC operation duration in nanoseconds",
                )
                .buckets(vec![10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 600.0, 1000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            pcb_message_routing_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_bus_routing_duration_ns",
                    "PCB message routing duration in nanoseconds",
                )
                .buckets(vec![5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            phase_packet_creation_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_bus_packet_creation_duration_ns",
                    "Phase packet creation duration in nanoseconds",
                )
                .buckets(vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            quantum_oracle_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_quantum_oracle_duration_ns",
                    "Quantum oracle query duration in nanoseconds",
                )
                .buckets(vec![10.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            task_scheduling_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_scheduler_schedule_duration_ns",
                    "Task scheduling duration in nanoseconds",
                )
                .buckets(vec![
                    100.0, 500.0, 1000.0, 2500.0, 5000.0, 8000.0, 15000.0, 30000.0,
                ]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            e2e_processing_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_e2e_processing_duration_ns",
                    "End-to-end processing duration in nanoseconds",
                )
                .buckets(vec![
                    1000.0, 2500.0, 5000.0, 10000.0, 15000.0, 25000.0, 50000.0,
                ]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // Throughput Metrics
            pcb_messages_processed_total: Counter::with_opts(Opts::new(
                "csf_bus_messages_processed_total",
                "Total PCB messages processed",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            time_operations_total: Counter::with_opts(Opts::new(
                "csf_time_operations_total",
                "Total time operations performed",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            hlc_updates_total: Counter::with_opts(Opts::new(
                "csf_hlc_updates_total",
                "Total HLC clock updates",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            scheduler_tasks_scheduled_total: Counter::with_opts(Opts::new(
                "csf_scheduler_tasks_scheduled_total",
                "Total tasks scheduled",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            clogic_inferences_total: Counter::with_opts(Opts::new(
                "csf_clogic_inferences_total",
                "Total C-LOGIC inferences performed",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            mlir_kernels_executed_total: Counter::with_opts(Opts::new(
                "csf_mlir_kernels_executed_total",
                "Total MLIR kernels executed",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // Temporal Coherence Accuracy (Mission-Critical)
            causality_violations_total: Counter::with_opts(Opts::new(
                "csf_causality_violations_total",
                "Total causality violations detected",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            hlc_drift_ns_per_hour: Gauge::with_opts(Opts::new(
                "csf_hlc_drift_ns_per_hour",
                "HLC clock drift in nanoseconds per hour",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            quantum_time_deviation_ns: Gauge::with_opts(Opts::new(
                "csf_quantum_time_deviation_ns",
                "Quantum time deviation in nanoseconds",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            distributed_coherence_ratio: Gauge::with_opts(Opts::new(
                "csf_distributed_coherence_ratio",
                "Distributed temporal coherence ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            deadline_miss_ratio: Gauge::with_opts(Opts::new(
                "csf_deadline_miss_ratio",
                "Task deadline miss ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // Hardware Acceleration Efficiency
            gpu_utilization_percent: Gauge::with_opts(Opts::new(
                "csf_gpu_utilization_percent",
                "GPU utilization percentage",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            mlir_backend_efficiency_ratio: Gauge::with_opts(Opts::new(
                "csf_mlir_backend_efficiency_ratio",
                "MLIR backend efficiency ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            hardware_memory_usage_ratio: Gauge::with_opts(Opts::new(
                "csf_hardware_memory_usage_ratio",
                "Hardware memory usage ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            kernel_launch_overhead_ratio: Gauge::with_opts(Opts::new(
                "csf_kernel_launch_overhead_ratio",
                "Kernel launch overhead ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            data_transfer_efficiency_ratio: Gauge::with_opts(Opts::new(
                "csf_data_transfer_efficiency_ratio",
                "Data transfer efficiency ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // System Reliability
            uptime_ratio: Gauge::with_opts(Opts::new("csf_uptime_ratio", "System uptime ratio"))
                .map_err(|e| TelemetryError::Prometheus(e))?,

            memory_violations_total: Counter::with_opts(Opts::new(
                "csf_memory_violations_total",
                "Total memory safety violations",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            panic_events_total: Counter::with_opts(Opts::new(
                "csf_panic_events_total",
                "Total panic events",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            error_rate_ratio: Gauge::with_opts(Opts::new(
                "csf_error_rate_ratio",
                "System error rate ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            recovery_duration_ms: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_recovery_duration_ms",
                    "System recovery duration in milliseconds",
                )
                .buckets(vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            data_integrity_violations_total: Counter::with_opts(Opts::new(
                "csf_data_integrity_violations_total",
                "Total data integrity violations",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // Operational Metrics
            service_availability_ratio: Gauge::with_opts(Opts::new(
                "csf_service_availability_ratio",
                "Service availability ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            component_health_score: Gauge::with_opts(Opts::new(
                "csf_component_health_score",
                "Overall component health score",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            network_connectivity_ratio: Gauge::with_opts(Opts::new(
                "csf_network_connectivity_ratio",
                "Network connectivity ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            consensus_participation_ratio: Gauge::with_opts(Opts::new(
                "csf_consensus_participation_ratio",
                "Consensus participation ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // Resource Utilization
            cpu_utilization_percent: Gauge::with_opts(Opts::new(
                "csf_cpu_utilization_percent",
                "CPU utilization percentage",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            memory_utilization_ratio: Gauge::with_opts(Opts::new(
                "csf_memory_utilization_ratio",
                "Memory utilization ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            disk_io_utilization_ratio: Gauge::with_opts(Opts::new(
                "csf_disk_io_utilization_ratio",
                "Disk I/O utilization ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            network_io_utilization_ratio: Gauge::with_opts(Opts::new(
                "csf_network_io_utilization_ratio",
                "Network I/O utilization ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            thread_pool_utilization_ratio: Gauge::with_opts(Opts::new(
                "csf_thread_pool_utilization_ratio",
                "Thread pool utilization ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            // C-LOGIC and Neuromorphic Integration
            clogic_accuracy_ratio: Gauge::with_opts(Opts::new(
                "csf_clogic_accuracy_ratio",
                "C-LOGIC module accuracy ratio",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            drpp_detection_accuracy: Gauge::with_opts(Opts::new(
                "csf_drpp_detection_accuracy",
                "DRPP pattern detection accuracy",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,

            adp_adaptation_duration_ms: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_adp_adaptation_duration_ms",
                    "ADP adaptation duration in milliseconds",
                )
                .buckets(vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            egc_convergence_duration_ms: Histogram::with_opts(
                HistogramOpts::new(
                    "csf_egc_convergence_duration_ms",
                    "EGC policy convergence duration in milliseconds",
                )
                .buckets(vec![100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]),
            )
            .map_err(|e| TelemetryError::Prometheus(e))?,

            ems_modeling_accuracy: Gauge::with_opts(Opts::new(
                "csf_ems_modeling_accuracy",
                "EMS emotion modeling accuracy",
            ))
            .map_err(|e| TelemetryError::Prometheus(e))?,
        })
    }

    /// Register all metrics with Prometheus registry
    pub fn register_all(&self, registry: &prometheus::Registry) -> Result<()> {
        // Register all histograms
        registry
            .register(Box::new(self.ttw_time_source_query_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.hlc_operation_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.pcb_message_routing_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.phase_packet_creation_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.quantum_oracle_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.task_scheduling_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.e2e_processing_duration.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.recovery_duration_ms.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.adp_adaptation_duration_ms.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.egc_convergence_duration_ms.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;

        // Register all counters
        registry
            .register(Box::new(self.pcb_messages_processed_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.time_operations_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.hlc_updates_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.scheduler_tasks_scheduled_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.clogic_inferences_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.mlir_kernels_executed_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.causality_violations_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.memory_violations_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.panic_events_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.data_integrity_violations_total.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;

        // Register all gauges
        registry
            .register(Box::new(self.hlc_drift_ns_per_hour.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.quantum_time_deviation_ns.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.distributed_coherence_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.deadline_miss_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.gpu_utilization_percent.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.mlir_backend_efficiency_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.hardware_memory_usage_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.kernel_launch_overhead_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.data_transfer_efficiency_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.uptime_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.error_rate_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.service_availability_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.component_health_score.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.network_connectivity_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.consensus_participation_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.cpu_utilization_percent.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.memory_utilization_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.disk_io_utilization_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.network_io_utilization_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.thread_pool_utilization_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.clogic_accuracy_ratio.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.drpp_detection_accuracy.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;
        registry
            .register(Box::new(self.ems_modeling_accuracy.clone()))
            .map_err(|e| TelemetryError::Prometheus(e))?;

        Ok(())
    }
}

/// Helper trait for measuring operation latency with automatic metric recording
pub trait LatencyMeasurement {
    /// Measure latency of an operation and record to the specified histogram
    fn measure_latency<F, R>(&self, histogram: &Histogram, operation: F) -> R
    where
        F: FnOnce() -> R;

    /// Measure latency with automatic validation against target thresholds
    fn measure_with_validation<F, R>(
        &self,
        histogram: &Histogram,
        target_ns: u64,
        alert_threshold_ns: u64,
        operation_name: &str,
        operation: F,
    ) -> R
    where
        F: FnOnce() -> R;
}

/// Implementation of latency measurement utilities
pub struct LatencyTracker;

impl LatencyMeasurement for LatencyTracker {
    fn measure_latency<F, R>(&self, histogram: &Histogram, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        histogram.observe(elapsed_ns);
        result
    }

    fn measure_with_validation<F, R>(
        &self,
        histogram: &Histogram,
        target_ns: u64,
        alert_threshold_ns: u64,
        operation_name: &str,
        operation: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as f64;

        histogram.observe(elapsed_ns);

        // Log performance violations
        if elapsed.as_nanos() > alert_threshold_ns as u128 {
            tracing::warn!(
                operation = operation_name,
                elapsed_ns = elapsed.as_nanos(),
                target_ns = target_ns,
                alert_threshold_ns = alert_threshold_ns,
                "Performance alert: Operation exceeded threshold"
            );
        }

        if elapsed.as_nanos() > target_ns as u128 {
            tracing::debug!(
                operation = operation_name,
                elapsed_ns = elapsed.as_nanos(),
                target_ns = target_ns,
                "Performance target missed"
            );
        }

        result
    }
}

/// Global success metrics instance
static SUCCESS_METRICS: std::sync::OnceLock<Arc<CsfSuccessMetrics>> = std::sync::OnceLock::new();

/// Initialize global success metrics
pub fn init_success_metrics() -> Result<()> {
    let metrics = Arc::new(CsfSuccessMetrics::new()?);
    SUCCESS_METRICS
        .set(metrics)
        .map_err(|_| TelemetryError::Config("Success metrics already initialized".into()))?;
    Ok(())
}

/// Get global success metrics instance
pub fn success_metrics() -> Option<Arc<CsfSuccessMetrics>> {
    SUCCESS_METRICS.get().cloned()
}

/// Convenience macros for metric recording
#[macro_export]
macro_rules! record_latency {
    ($metric:ident, $operation:expr) => {{
        if let Some(metrics) = $crate::success_metrics::success_metrics() {
            $crate::success_metrics::LatencyTracker.measure_latency(&metrics.$metric, || $operation)
        } else {
            $operation
        }
    }};
}

#[macro_export]
macro_rules! record_counter {
    ($metric:ident, $value:expr) => {{
        if let Some(metrics) = $crate::success_metrics::success_metrics() {
            metrics.$metric.inc_by($value as f64);
        }
    }};
}

#[macro_export]
macro_rules! record_gauge {
    ($metric:ident, $value:expr) => {{
        if let Some(metrics) = $crate::success_metrics::success_metrics() {
            metrics.$metric.set($value as f64);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_success_metrics_creation() {
        let metrics = CsfSuccessMetrics::new().expect("Failed to create success metrics");
        let registry = prometheus::Registry::new();
        metrics
            .register_all(&registry)
            .expect("Failed to register metrics");
    }

    #[tokio::test]
    async fn test_latency_measurement() {
        let metrics = CsfSuccessMetrics::new().expect("Failed to create metrics");
        let tracker = LatencyTracker;

        let result = tracker.measure_latency(&metrics.ttw_time_source_query_duration, || {
            std::thread::sleep(Duration::from_nanos(100));
            42
        });

        assert_eq!(result, 42);
        assert!(metrics.ttw_time_source_query_duration.get_sample_count() > 0);
    }

    #[tokio::test]
    async fn test_global_metrics_initialization() {
        // Note: This test can only run once due to global state
        if success_metrics().is_none() {
            init_success_metrics().expect("Failed to initialize global metrics");
            assert!(success_metrics().is_some());
        }
    }
}
