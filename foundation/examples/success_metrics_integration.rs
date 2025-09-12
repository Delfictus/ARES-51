//! Example integration of CSF Success Metrics Framework
//!
//! This example demonstrates how to integrate the comprehensive success metrics
//! framework into ARES ChronoSynclastic Fabric components for real-time
//! performance monitoring and validation.

use csf_core::prelude::*;
use csf_core::NanoTime;
use csf_telemetry::{
    record_counter, record_gauge, record_latency,
    success_metrics::{init_success_metrics, success_metrics, LatencyMeasurement, LatencyTracker},
    TelemetryConfig, TelemetrySystem,
};
use csf_time::{TimeError, TimeSource};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tokio::time::sleep;

/// Example TTW TimeSource implementation with integrated success metrics
pub struct InstrumentedTimeSource {
    inner: Arc<dyn TimeSource>,
    latency_tracker: LatencyTracker,
}

impl InstrumentedTimeSource {
    pub fn new(inner: Arc<dyn TimeSource>) -> Self {
        Self {
            inner,
            latency_tracker: LatencyTracker,
        }
    }
}

impl TimeSource for InstrumentedTimeSource {
    fn now_ns(&self) -> Result<NanoTime, TimeError> {
        // Record latency with sub-microsecond precision
        let result = record_latency!(ttw_time_source_query_duration, { self.inner.now_ns() });

        // Increment throughput counter
        record_counter!(time_operations_total, 1);

        result
    }

    fn monotonic_ns(&self) -> Result<NanoTime, TimeError> {
        // Also instrument monotonic time queries
        let result = record_latency!(ttw_time_source_query_duration, {
            self.inner.monotonic_ns()
        });

        record_counter!(time_operations_total, 1);
        result
    }
}

/// Example HLC Clock implementation with success metrics integration
pub struct InstrumentedHlcClock {
    inner: Arc<dyn HlcClock>,
}

impl InstrumentedHlcClock {
    pub fn new(inner: Arc<dyn HlcClock>) -> Self {
        Self { inner }
    }
}

impl HlcClock for InstrumentedHlcClock {
    fn now_hlc(&self) -> Result<NanoTime, Error> {
        let result = record_latency!(hlc_operation_duration, { self.inner.now_hlc() });

        record_counter!(hlc_updates_total, 1);
        result
    }

    fn update_hlc(&mut self, remote_time: NanoTime) -> Result<(), Error> {
        let result = record_latency!(hlc_operation_duration, {
            // Note: This is a simplified example - actual implementation would need
            // to handle the mutable reference properly
            Err(Error::Internal(
                "Mutable HLC update not implemented in example".into(),
            ))
        });

        record_counter!(hlc_updates_total, 1);
        result
    }

    fn logical_time(&self) -> u64 {
        self.inner.logical_time()
    }
}

/// Example Phase Coherence Bus message processing with metrics
pub struct InstrumentedMessageProcessor;

impl InstrumentedMessageProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Process a PCB message with comprehensive metrics collection
    pub async fn process_message(&self, message_data: &[u8]) -> Result<(), Error> {
        let start = Instant::now();

        // Simulate PCB packet creation
        let _packet = record_latency!(phase_packet_creation_duration, {
            // Simulate packet creation overhead
            std::thread::sleep(Duration::from_nanos(25));
            format!("packet_{}", message_data.len())
        });

        // Simulate PCB routing
        record_latency!(pcb_message_routing_duration, {
            // Simulate routing lookup and decision
            std::thread::sleep(Duration::from_nanos(50));
        });

        // Record successful message processing
        record_counter!(pcb_messages_processed_total, 1);

        // Measure end-to-end processing time
        let e2e_duration = start.elapsed();
        if let Some(metrics) = success_metrics() {
            metrics
                .e2e_processing_duration
                .observe(e2e_duration.as_nanos() as f64);
        }

        // Validate sub-microsecond target for critical path
        if e2e_duration.as_micros() > 10 {
            tracing::warn!(
                duration_us = e2e_duration.as_micros(),
                "E2E processing exceeded 10μs target"
            );
        }

        Ok(())
    }
}

/// Example C-LOGIC module with neuromorphic metrics
pub struct InstrumentedCLogicModule;

impl InstrumentedCLogicModule {
    pub fn new() -> Self {
        Self
    }

    /// Perform DRPP pattern detection with accuracy tracking
    pub async fn drpp_pattern_detection(&self, input: &[f64]) -> Result<bool, Error> {
        let start = Instant::now();

        // Simulate pattern detection processing
        sleep(Duration::from_micros(50)).await;

        // Simulate detection accuracy (in real implementation, this would be measured)
        let accuracy = 0.96; // 96% accuracy
        record_gauge!(drpp_detection_accuracy, accuracy);

        record_counter!(clogic_inferences_total, 1);

        let detection_result = input.iter().sum::<f64>() > 0.5;
        Ok(detection_result)
    }

    /// Perform ADP adaptive processing with timing metrics
    pub async fn adp_adaptive_processing(&self, parameters: &[f64]) -> Result<Vec<f64>, Error> {
        // Simulate adaptive processing
        std::thread::sleep(Duration::from_millis(75));
        record_counter!(clogic_inferences_total, 1);

        // Return adapted parameters
        Ok(parameters.iter().map(|x| x * 1.1).collect::<Vec<f64>>())
    }

    /// Perform EGC policy convergence with timing
    pub async fn egc_policy_convergence(&self, policies: &[String]) -> Result<String, Error> {
        // Simulate policy convergence
        std::thread::sleep(Duration::from_millis(800));
        record_counter!(clogic_inferences_total, 1);

        Ok(policies.first().unwrap_or(&"default".to_string()).clone())
    }
}

/// Example MLIR hardware acceleration with efficiency metrics
pub struct InstrumentedMlirRuntime;

impl InstrumentedMlirRuntime {
    pub fn new() -> Self {
        Self
    }

    /// Execute MLIR kernel with hardware acceleration metrics
    pub async fn execute_kernel(&self, kernel_name: &str, data: &[f32]) -> Result<Vec<f32>, Error> {
        let start = Instant::now();

        // Simulate kernel launch overhead
        let launch_overhead_start = Instant::now();
        sleep(Duration::from_micros(100)).await; // Simulate launch overhead
        let launch_overhead = launch_overhead_start.elapsed();

        // Simulate kernel execution
        sleep(Duration::from_micros(500)).await;

        // Calculate efficiency metrics
        let total_time = start.elapsed();
        let execution_time = total_time - launch_overhead;
        let overhead_ratio = launch_overhead.as_nanos() as f64 / total_time.as_nanos() as f64;
        let efficiency_ratio = execution_time.as_nanos() as f64 / total_time.as_nanos() as f64;

        // Record metrics
        record_gauge!(kernel_launch_overhead_ratio, overhead_ratio);
        record_gauge!(mlir_backend_efficiency_ratio, efficiency_ratio);
        record_counter!(mlir_kernels_executed_total, 1);

        // Simulate GPU utilization (would be measured from actual hardware)
        record_gauge!(gpu_utilization_percent, 82.5);

        // Return processed data
        let result = data.iter().map(|x| x * 2.0).collect();
        Ok(result)
    }
}

/// System health monitor that tracks operational metrics
pub struct SystemHealthMonitor {
    start_time: Instant,
    error_count: std::sync::atomic::AtomicU64,
    operation_count: std::sync::atomic::AtomicU64,
}

impl SystemHealthMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            error_count: std::sync::atomic::AtomicU64::new(0),
            operation_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Update system health metrics periodically
    pub async fn update_health_metrics(&self) {
        use std::sync::atomic::Ordering;

        // Calculate uptime ratio (simplified - assumes no downtime)
        let uptime_ratio = 0.9999; // 99.99% uptime
        record_gauge!(uptime_ratio, uptime_ratio);

        // Calculate error rate
        let total_operations = self.operation_count.load(Ordering::Relaxed);
        let total_errors = self.error_count.load(Ordering::Relaxed);
        let error_rate = if total_operations > 0 {
            total_errors as f64 / total_operations as f64
        } else {
            0.0
        };
        record_gauge!(error_rate_ratio, error_rate);

        // Simulate system resource utilization
        record_gauge!(cpu_utilization_percent, 72.3);
        record_gauge!(memory_utilization_ratio, 0.65);
        record_gauge!(network_io_utilization_ratio, 0.34);

        // Component health score (aggregated from various subsystems)
        let health_score = (uptime_ratio + (1.0 - error_rate) + 0.95 + 0.92) / 4.0 * 100.0;
        record_gauge!(component_health_score, health_score);

        // Service availability
        record_gauge!(service_availability_ratio, 0.9999);

        // Temporal coherence metrics (mission-critical)
        record_gauge!(distributed_coherence_ratio, 0.99999);
        record_gauge!(deadline_miss_ratio, 0.0001);
        record_gauge!(quantum_time_deviation_ns, 0.5);
    }

    /// Record an operation (for error rate calculation)
    pub fn record_operation(&self, success: bool) {
        use std::sync::atomic::Ordering;

        self.operation_count.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Example demonstrating comprehensive CSF metrics integration
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    tracing::info!("Starting ARES CSF Success Metrics Integration Example");

    // Initialize telemetry system
    let telemetry_config = TelemetryConfig::default();
    let _telemetry = TelemetrySystem::new(telemetry_config).await?;

    // Initialize success metrics
    init_success_metrics().map_err(|e| format!("Failed to initialize success metrics: {}", e))?;

    tracing::info!("Success metrics framework initialized");

    // Create instrumented components
    let health_monitor = Arc::new(SystemHealthMonitor::new());
    let message_processor = InstrumentedMessageProcessor::new();
    let clogic_module = InstrumentedCLogicModule::new();
    let mlir_runtime = InstrumentedMlirRuntime::new();

    // Simulation loop demonstrating metrics collection
    for iteration in 0..100 {
        tracing::debug!(iteration, "Running simulation iteration");

        // Simulate PCB message processing
        let message_data = vec![1u8, 2, 3, 4, 5];
        match message_processor.process_message(&message_data).await {
            Ok(_) => health_monitor.record_operation(true),
            Err(e) => {
                tracing::error!(error = %e, "Message processing failed");
                health_monitor.record_operation(false);
            }
        }

        // Simulate C-LOGIC operations
        if iteration % 5 == 0 {
            let input_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
            let _ = clogic_module.drpp_pattern_detection(&input_data).await;

            let parameters = vec![1.0, 2.0, 3.0];
            let _ = clogic_module.adp_adaptive_processing(&parameters).await;
        }

        // Simulate MLIR kernel execution
        if iteration % 10 == 0 {
            let kernel_data = vec![1.0, 2.0, 3.0, 4.0];
            let _ = mlir_runtime
                .execute_kernel("example_kernel", &kernel_data)
                .await;
        }

        // Simulate EGC policy convergence (less frequent)
        if iteration % 20 == 0 {
            let policies = vec!["policy_a".to_string(), "policy_b".to_string()];
            let _ = clogic_module.egc_policy_convergence(&policies).await;
        }

        // Update system health metrics periodically
        if iteration % 10 == 0 {
            health_monitor.update_health_metrics().await;
        }

        // Small delay between iterations
        sleep(Duration::from_millis(10)).await;
    }

    // Final metrics report
    tracing::info!("Simulation completed. Metrics have been recorded to Prometheus.");

    if let Some(metrics) = success_metrics() {
        tracing::info!(
            "Sample metrics recorded:",
            "PCB messages processed: {}",
            metrics.pcb_messages_processed_total.get()
        );

        tracing::info!("Time operations: {}", metrics.time_operations_total.get());

        tracing::info!(
            "C-LOGIC inferences: {}",
            metrics.clogic_inferences_total.get()
        );

        tracing::info!(
            "MLIR kernels executed: {}",
            metrics.mlir_kernels_executed_total.get()
        );
    }

    tracing::info!("Success metrics integration example completed successfully");

    // In a real application, the metrics would be:
    // 1. Exported to Prometheus via HTTP endpoint (default: :9090/metrics)
    // 2. Visualized in Grafana dashboards
    // 3. Monitored by alerting systems (PagerDuty, Slack)
    // 4. Used for automated performance regression detection
    // 5. Analyzed for capacity planning and optimization opportunities

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_processor_metrics() {
        // Initialize metrics for testing
        let _ = init_success_metrics();

        let processor = InstrumentedMessageProcessor::new();
        let message = vec![1, 2, 3, 4];

        let result = processor.process_message(&message).await;
        assert!(result.is_ok());

        // Verify metrics were recorded
        if let Some(metrics) = success_metrics() {
            assert!(metrics.pcb_messages_processed_total.get() > 0.0);
        }
    }

    #[tokio::test]
    async fn test_clogic_module_metrics() {
        let _ = init_success_metrics();

        let module = InstrumentedCLogicModule::new();
        let input = vec![0.1, 0.2, 0.3];

        let result = module.drpp_pattern_detection(&input).await;
        assert!(result.is_ok());

        // Verify C-LOGIC metrics were recorded
        if let Some(metrics) = success_metrics() {
            assert!(metrics.clogic_inferences_total.get() > 0.0);
            assert!(metrics.drpp_detection_accuracy.get() > 0.0);
        }
    }

    #[tokio::test]
    async fn test_mlir_runtime_metrics() {
        let _ = init_success_metrics();

        let runtime = InstrumentedMlirRuntime::new();
        let data = vec![1.0, 2.0, 3.0];

        let result = runtime.execute_kernel("test_kernel", &data).await;
        assert!(result.is_ok());

        // Verify MLIR metrics were recorded
        if let Some(metrics) = success_metrics() {
            assert!(metrics.mlir_kernels_executed_total.get() > 0.0);
            assert!(metrics.gpu_utilization_percent.get() > 0.0);
        }
    }

    #[tokio::test]
    async fn test_health_monitor_metrics() {
        let _ = init_success_metrics();

        let monitor = SystemHealthMonitor::new();

        // Record some operations
        monitor.record_operation(true);
        monitor.record_operation(true);
        monitor.record_operation(false); // One failure

        monitor.update_health_metrics().await;

        // Verify health metrics were updated
        if let Some(metrics) = success_metrics() {
            assert!(metrics.uptime_ratio.get() > 0.0);
            assert!(metrics.component_health_score.get() > 0.0);
        }
    }
}
