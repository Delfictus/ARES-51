---
sidebar_position: 1
title: "Performance Troubleshooting"
description: "Comprehensive performance optimization and troubleshooting guide for ARES ChronoFabric systems"
---

# ARES ChronoFabric Performance & Troubleshooting Guide

**Version**: 1.0  
**Last Updated**: August 26, 2025  
**Document Type**: Performance & Troubleshooting Guide

> **⚡ Performance-Critical Systems**: This guide covers optimization techniques and troubleshooting procedures for high-performance distributed systems using ARES ChronoFabric.

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Performance Monitoring](#performance-monitoring)
3. [Optimization Techniques](#optimization-techniques)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Advanced Diagnostics](#advanced-diagnostics)
6. [Production Troubleshooting](#production-troubleshooting)
7. [Performance Testing](#performance-testing)
8. [Best Practices](#best-practices)

---

## Performance Targets

### System Performance Goals

ARES ChronoFabric is designed to meet aggressive performance targets:

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| **Message Latency** | < 1μs | TSC-based timing |
| **Throughput** | > 1M msg/sec | Messages per second |
| **Memory Usage** | < 100MB baseline | RSS measurement |
| **CPU Utilization** | < 80% at peak load | System monitoring |
| **Temporal Accuracy** | ±10ns | HLC drift measurement |
| **Packet Loss** | < 0.01% | Drop rate monitoring |

### Hardware Requirements for Optimal Performance

```bash
# Minimum Requirements
CPU: x86_64 with TSC support, 4+ cores
RAM: 8GB minimum, 16GB+ recommended  
Storage: SSD with > 1000 IOPS
Network: 1Gbps+ for distributed deployments

# Optimal Performance
CPU: Intel Xeon or AMD EPYC, 8+ cores, 3GHz+
RAM: 32GB+ with ECC
Storage: NVMe SSD with > 10000 IOPS
Network: 10Gbps+ with SR-IOV support
```

---

## Performance Monitoring

### Built-in Metrics Collection

```rust
use csf_bus::{PhaseCoherenceBus, BusStats};
use csf_time::global_time_source;
use std::time::{Duration, Instant};

/// Comprehensive performance monitoring
pub struct PerformanceMonitor {
    bus: Arc<PhaseCoherenceBus>,
    start_time: Instant,
    measurement_window: Duration,
    metrics_history: Vec<PerformanceSnapshot>,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    timestamp: u64,
    bus_stats: BusStats,
    system_metrics: SystemMetrics,
    temporal_metrics: TemporalMetrics,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    cpu_usage_percent: f64,
    memory_usage_bytes: u64,
    network_rx_bytes: u64,
    network_tx_bytes: u64,
    disk_read_bytes: u64,
    disk_write_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct TemporalMetrics {
    hlc_drift_ns: i64,
    quantum_coherence_score: f64,
    scheduling_efficiency: f64,
    causality_violations: u64,
}

impl PerformanceMonitor {
    pub fn new(bus: Arc<PhaseCoherenceBus>) -> Self {
        Self {
            bus,
            start_time: Instant::now(),
            measurement_window: Duration::from_secs(1),
            metrics_history: Vec::new(),
        }
    }
    
    /// Collect current performance metrics
    pub async fn collect_snapshot(&self) -> Result<PerformanceSnapshot, Box<dyn std::error::Error>> {
        let timestamp = global_time_source().now_ns()?.as_nanos();
        let bus_stats = self.bus.get_stats();
        let system_metrics = self.collect_system_metrics().await?;
        let temporal_metrics = self.collect_temporal_metrics().await?;
        
        Ok(PerformanceSnapshot {
            timestamp,
            bus_stats,
            system_metrics,
            temporal_metrics,
        })
    }
    
    /// Check if system is meeting performance targets
    pub fn check_performance_targets(&self, snapshot: &PerformanceSnapshot) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();
        
        // Check latency target (< 1μs)
        if snapshot.bus_stats.avg_latency_ns > 1_000 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                metric: "avg_latency_ns".to_string(),
                current_value: snapshot.bus_stats.avg_latency_ns as f64,
                target_value: 1_000.0,
                description: "Average latency exceeds 1μs target".to_string(),
            });
        }
        
        // Check throughput target (> 1M msg/sec)
        if snapshot.bus_stats.throughput_mps < 1_000_000 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                metric: "throughput_mps".to_string(),
                current_value: snapshot.bus_stats.throughput_mps as f64,
                target_value: 1_000_000.0,
                description: "Throughput below 1M messages/sec target".to_string(),
            });
        }
        
        // Check packet loss (< 0.01%)
        let drop_rate = if snapshot.bus_stats.packets_published > 0 {
            (snapshot.bus_stats.packets_dropped as f64 / snapshot.bus_stats.packets_published as f64) * 100.0
        } else {
            0.0
        };
        
        if drop_rate > 0.01 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Critical,
                metric: "packet_drop_rate".to_string(),
                current_value: drop_rate,
                target_value: 0.01,
                description: "Packet drop rate exceeds 0.01% target".to_string(),
            });
        }
        
        // Check CPU utilization (< 80%)
        if snapshot.system_metrics.cpu_usage_percent > 80.0 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                metric: "cpu_usage_percent".to_string(),
                current_value: snapshot.system_metrics.cpu_usage_percent,
                target_value: 80.0,
                description: "CPU utilization exceeds 80% target".to_string(),
            });
        }
        
        // Check temporal coherence
        if snapshot.temporal_metrics.quantum_coherence_score < 0.7 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                metric: "quantum_coherence_score".to_string(),
                current_value: snapshot.temporal_metrics.quantum_coherence_score,
                target_value: 0.7,
                description: "Quantum coherence score below optimal threshold".to_string(),
            });
        }
        
        alerts
    }
    
    async fn collect_system_metrics(&self) -> Result<SystemMetrics, Box<dyn std::error::Error>> {
        use sysinfo::{System, SystemExt, CpuExt};
        
        let mut system = System::new_all();
        system.refresh_all();
        
        let cpu_usage = system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>() / system.cpus().len() as f32;
            
        Ok(SystemMetrics {
            cpu_usage_percent: cpu_usage as f64,
            memory_usage_bytes: system.used_memory() * 1024, // Convert KB to bytes
            network_rx_bytes: system.networks().iter()
                .map(|(_, data)| data.received())
                .sum(),
            network_tx_bytes: system.networks().iter()
                .map(|(_, data)| data.transmitted())
                .sum(),
            disk_read_bytes: 0, // Would need platform-specific implementation
            disk_write_bytes: 0,
        })
    }
    
    async fn collect_temporal_metrics(&self) -> Result<TemporalMetrics, Box<dyn std::error::Error>> {
        let coherence_metrics = self.bus.get_temporal_metrics();
        
        Ok(TemporalMetrics {
            hlc_drift_ns: 0, // Would measure against reference clock
            quantum_coherence_score: coherence_metrics.quantum_coherence_score,
            scheduling_efficiency: coherence_metrics.schedule_utilization,
            causality_violations: coherence_metrics.deadline_violations as u64,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    severity: AlertSeverity,
    metric: String,
    current_value: f64,
    target_value: f64,
    description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Real-time performance monitoring example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    csf_time::initialize_global_time_source()?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig::default())?);
    let monitor = PerformanceMonitor::new(bus.clone());
    
    // Start monitoring loop
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    
    loop {
        interval.tick().await;
        
        let snapshot = monitor.collect_snapshot().await?;
        let alerts = monitor.check_performance_targets(&snapshot);
        
        // Log performance metrics
        tracing::info!(
            "Performance: {}ns avg latency, {} msg/sec throughput, {:.1}% CPU",
            snapshot.bus_stats.avg_latency_ns,
            snapshot.bus_stats.throughput_mps,
            snapshot.system_metrics.cpu_usage_percent
        );
        
        // Handle alerts
        for alert in alerts {
            match alert.severity {
                AlertSeverity::Critical => {
                    tracing::error!(
                        "CRITICAL: {} = {:.2} (target: {:.2}) - {}",
                        alert.metric, alert.current_value, alert.target_value, alert.description
                    );
                }
                AlertSeverity::Warning => {
                    tracing::warn!(
                        "WARNING: {} = {:.2} (target: {:.2}) - {}",
                        alert.metric, alert.current_value, alert.target_value, alert.description
                    );
                }
                AlertSeverity::Info => {
                    tracing::info!(
                        "INFO: {} = {:.2} (target: {:.2}) - {}",
                        alert.metric, alert.current_value, alert.target_value, alert.description
                    );
                }
            }
        }
    }
}
```

### Prometheus Metrics Integration

```rust
use prometheus::{Counter, Gauge, Histogram, Registry, Opts, HistogramOpts};
use std::sync::Arc;

/// Prometheus metrics exporter for ChronoFabric
pub struct PrometheusExporter {
    registry: Registry,
    
    // Bus metrics
    messages_published_total: Counter,
    messages_delivered_total: Counter,
    messages_dropped_total: Counter,
    message_latency_seconds: Histogram,
    throughput_messages_per_second: Gauge,
    active_subscriptions: Gauge,
    
    // System metrics
    cpu_usage_percent: Gauge,
    memory_usage_bytes: Gauge,
    
    // Temporal metrics
    quantum_coherence_score: Gauge,
    hlc_drift_nanoseconds: Gauge,
    causality_violations_total: Counter,
}

impl PrometheusExporter {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        // Bus metrics
        let messages_published_total = Counter::with_opts(
            Opts::new("chronofabric_messages_published_total", 
                     "Total number of messages published")
        )?;
        
        let message_latency_seconds = Histogram::with_opts(
            HistogramOpts::new("chronofabric_message_latency_seconds",
                              "Message processing latency in seconds")
                .buckets(vec![0.000001, 0.000010, 0.000100, 0.001000, 0.010000]) // 1μs to 10ms
        )?;
        
        let throughput_messages_per_second = Gauge::with_opts(
            Opts::new("chronofabric_throughput_messages_per_second",
                     "Current message throughput")
        )?;
        
        // Register all metrics
        registry.register(Box::new(messages_published_total.clone()))?;
        registry.register(Box::new(message_latency_seconds.clone()))?;
        registry.register(Box::new(throughput_messages_per_second.clone()))?;
        
        Ok(Self {
            registry,
            messages_published_total,
            messages_delivered_total: Counter::default(),
            messages_dropped_total: Counter::default(),
            message_latency_seconds,
            throughput_messages_per_second,
            active_subscriptions: Gauge::default(),
            cpu_usage_percent: Gauge::default(),
            memory_usage_bytes: Gauge::default(),
            quantum_coherence_score: Gauge::default(),
            hlc_drift_nanoseconds: Gauge::default(),
            causality_violations_total: Counter::default(),
        })
    }
    
    pub fn update_metrics(&self, snapshot: &PerformanceSnapshot) {
        // Update bus metrics
        self.messages_published_total.inc_by(snapshot.bus_stats.packets_published);
        self.throughput_messages_per_second.set(snapshot.bus_stats.throughput_mps as f64);
        self.message_latency_seconds.observe(snapshot.bus_stats.avg_latency_ns as f64 / 1_000_000_000.0);
        self.active_subscriptions.set(snapshot.bus_stats.active_subscriptions as f64);
        
        // Update system metrics
        self.cpu_usage_percent.set(snapshot.system_metrics.cpu_usage_percent);
        self.memory_usage_bytes.set(snapshot.system_metrics.memory_usage_bytes as f64);
        
        // Update temporal metrics
        self.quantum_coherence_score.set(snapshot.temporal_metrics.quantum_coherence_score);
        self.hlc_drift_nanoseconds.set(snapshot.temporal_metrics.hlc_drift_ns as f64);
    }
    
    pub fn metrics_text(&self) -> Result<String, Box<dyn std::error::Error>> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}
```

---

## Optimization Techniques

### 1. Hardware-Level Optimizations

#### TSC Calibration Optimization

```rust
use csf_bus::routing::TscCalibration;
use std::sync::Arc;

/// Advanced TSC calibration for maximum timing accuracy
pub struct AdvancedTscCalibration {
    calibration: Arc<TscCalibration>,
    calibration_history: Vec<(u64, u64)>, // (frequency, timestamp)
    auto_recalibration: bool,
}

impl AdvancedTscCalibration {
    pub fn new() -> Self {
        let calibration = Arc::new(TscCalibration::new());
        
        Self {
            calibration,
            calibration_history: Vec::new(),
            auto_recalibration: true,
        }
    }
    
    /// Perform multiple calibration samples for increased accuracy
    pub fn calibrate_precise(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        const SAMPLE_COUNT: usize = 10;
        const SAMPLE_DURATION_MS: u64 = 100;
        
        let mut frequencies = Vec::new();
        
        for _ in 0..SAMPLE_COUNT {
            // Perform calibration
            self.calibration.calibrate();
            
            // Wait for settling
            std::thread::sleep(std::time::Duration::from_millis(SAMPLE_DURATION_MS));
            
            let frequency = self.calibration.frequency_hz.load(std::sync::atomic::Ordering::Relaxed);
            if frequency > 0 {
                frequencies.push(frequency);
            }
        }
        
        if frequencies.is_empty() {
            return Err("No valid calibration samples obtained".into());
        }
        
        // Use median frequency to avoid outliers
        frequencies.sort_unstable();
        let median_frequency = frequencies[frequencies.len() / 2];
        
        self.calibration.frequency_hz.store(median_frequency, std::sync::atomic::Ordering::Relaxed);
        
        tracing::info!(
            "TSC calibration complete: {}Hz (samples: {})",
            median_frequency, frequencies.len()
        );
        
        Ok(())
    }
    
    /// Automatic recalibration based on drift detection
    pub async fn start_auto_recalibration(&self) -> tokio::task::JoinHandle<()> {
        let calibration = self.calibration.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Check if recalibration is needed
                let last_calibration = calibration.calibrated_at.load(std::sync::atomic::Ordering::Relaxed);
                let current_tsc = TscCalibration::read_tsc();
                
                // If more than 5 minutes have passed, recalibrate
                if current_tsc.saturating_sub(last_calibration) > 5 * 60 * calibration.frequency_hz.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!("Performing automatic TSC recalibration");
                    calibration.calibrate();
                }
            }
        })
    }
}
```

### 2. Memory Optimization

#### Custom Memory Allocators

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// High-performance memory allocator with pool optimization
pub struct PerformanceAllocator {
    system: System,
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    bytes_allocated: AtomicUsize,
}

impl PerformanceAllocator {
    pub const fn new() -> Self {
        Self {
            system: System,
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
            bytes_allocated: AtomicUsize::new(0),
        }
    }
    
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.allocations.load(Ordering::Relaxed),
            self.deallocations.load(Ordering::Relaxed),
            self.bytes_allocated.load(Ordering::Relaxed),
        )
    }
}

unsafe impl GlobalAlloc for PerformanceAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.system.alloc(layout);
        if !ptr.is_null() {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.bytes_allocated.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.system.dealloc(ptr, layout);
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

// Global allocator (would be enabled in production)
// #[global_allocator]
// static ALLOCATOR: PerformanceAllocator = PerformanceAllocator::new();

/// Memory pool for high-frequency allocations
pub struct MessagePool<T> {
    pool: crossbeam::queue::SegQueue<Box<T>>,
    total_allocated: AtomicUsize,
    pool_hits: AtomicUsize,
    pool_misses: AtomicUsize,
}

impl<T: Default> MessagePool<T> {
    pub fn new() -> Self {
        Self {
            pool: crossbeam::queue::SegQueue::new(),
            total_allocated: AtomicUsize::new(0),
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
        }
    }
    
    /// Get object from pool or allocate new one
    pub fn acquire(&self) -> Box<T> {
        if let Some(item) = self.pool.pop() {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
            item
        } else {
            self.pool_misses.fetch_add(1, Ordering::Relaxed);
            self.total_allocated.fetch_add(1, Ordering::Relaxed);
            Box::new(T::default())
        }
    }
    
    /// Return object to pool
    pub fn release(&self, mut item: Box<T>) {
        // Reset object to default state
        *item = T::default();
        self.pool.push(item);
    }
    
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.total_allocated.load(Ordering::Relaxed),
            self.pool_hits.load(Ordering::Relaxed),
            self.pool_misses.load(Ordering::Relaxed),
        )
    }
}
```

---

## Common Issues & Solutions

### 1. High Latency Issues

#### Problem: Average latency > 1μs

**Diagnosis:**
```rust
/// Latency diagnosis tool
pub struct LatencyDiagnostics {
    bus: Arc<PhaseCoherenceBus>,
    router: Arc<HardwareRouter>,
}

impl LatencyDiagnostics {
    pub async fn diagnose_high_latency(&self) -> Vec<LatencyIssue> {
        let mut issues = Vec::new();
        
        let stats = self.bus.get_stats();
        let router_healthy = self.router.is_healthy();
        
        // Check TSC calibration
        if !self.router.tsc_calibration.is_valid() {
            issues.push(LatencyIssue {
                issue_type: "TSC_CALIBRATION".to_string(),
                severity: IssueSeverity::High,
                description: "TSC timing not calibrated properly".to_string(),
                solution: "Call router.tsc_calibration.calibrate()".to_string(),
            });
        }
        
        // Check quantum optimization
        let temporal_metrics = self.bus.get_temporal_metrics();
        if temporal_metrics.quantum_coherence_score < 0.5 {
            issues.push(LatencyIssue {
                issue_type: "QUANTUM_COHERENCE".to_string(),
                severity: IssueSeverity::Medium,
                description: "Low quantum coherence affecting routing efficiency".to_string(),
                solution: "Enable quantum optimization: bus.set_quantum_optimization(true)".to_string(),
            });
        }
        
        // Check buffer sizes
        let pending_count = temporal_metrics.pending_messages;
        if pending_count > 1000 {
            issues.push(LatencyIssue {
                issue_type: "BUFFER_OVERFLOW".to_string(),
                severity: IssueSeverity::High,
                description: format!("High pending message count: {}", pending_count),
                solution: "Increase channel_buffer_size or reduce message rate".to_string(),
            });
        }
        
        issues
    }
}

#[derive(Debug, Clone)]
pub struct LatencyIssue {
    issue_type: String,
    severity: IssueSeverity,
    description: String,
    solution: String,
}

#[derive(Debug, Clone, Copy)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}
```

**Solutions:**

1. **TSC Calibration Issues:**
```rust
// Force TSC recalibration
router.tsc_calibration.calibrate();

// Enable automatic recalibration
let calibration = AdvancedTscCalibration::new();
calibration.start_auto_recalibration().await;
```

2. **Quantum Optimization:**
```rust
// Enable quantum optimization
bus.set_quantum_optimization(true);

// Use quantum-optimized packets for critical messages
let quantum_offset = QuantumOffset::new(0.2, 1.5, 1800.0); // Optimized for latency
let packet = PhasePacket::with_quantum_optimization(data, quantum_offset);
```

3. **Buffer Size Optimization:**
```rust
// Increase buffer sizes
let config = BusConfig {
    channel_buffer_size: 4096, // Increase from default 1024
};

// Use non-blocking publish for high-frequency messages
match bus.try_publish(packet) {
    Ok(message_id) => { /* success */ }
    Err(BusError::ResourceExhausted { .. }) => {
        // Handle backpressure
        tokio::time::sleep(Duration::from_nanos(100)).await;
    }
}
```

### 2. Memory Issues

#### Problem: High memory usage or memory leaks

**Solutions:**

1. **Arc Reference Cycles:**
```rust
// Use weak references to break cycles
use std::sync::{Arc, Weak};

struct Service {
    bus: Arc<PhaseCoherenceBus>,
    // Use weak reference to avoid cycles
    parent: Weak<ParentService>,
}
```

2. **Subscription Management:**
```rust
// Properly unsubscribe when done
struct SubscriptionManager {
    subscriptions: Vec<SubscriptionId>,
    bus: Arc<PhaseCoherenceBus>,
}

impl Drop for SubscriptionManager {
    fn drop(&mut self) {
        for sub_id in &self.subscriptions {
            let _ = self.bus.unsubscribe::<MyMessage>(*sub_id);
        }
    }
}
```

3. **Message Pool Usage:**
```rust
// Use object pools for high-frequency allocations
lazy_static::lazy_static! {
    static ref MESSAGE_POOL: MessagePool<MyMessage> = MessagePool::new();
}

// Acquire from pool
let mut message = MESSAGE_POOL.acquire();
*message = MyMessage { /* ... */ };

// Use message...

// Return to pool when done
MESSAGE_POOL.release(message);
```

### 3. Temporal Coherence Violations

#### Problem: Causality violations or temporal inconsistencies

**Solutions:**

1. **Initialize HLC Properly:**
```rust
// Initialize global HLC with proper node ID
csf_time::initialize_global_hlc(
    Arc::new(csf_time::TimeSourceImpl::new()?),
    1 // unique node ID
)?;
```

2. **Add Causal Dependencies:**
```rust
// Track message dependencies
let mut packet = PhasePacket::new(data, source_id);
packet.add_temporal_correlation(vec![previous_message_id]);

// Use deadline scheduling for time-critical messages
bus.publish_with_deadline(packet, deadline_time).await?;
```

3. **Process Temporal Queue:**
```rust
// Regularly process pending temporal messages
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    loop {
        interval.tick().await;
        let processed = bus.process_temporal_queue();
        if processed > 0 {
            tracing::debug!("Processed {} temporal messages", processed);
        }
    }
});
```

---

## Advanced Diagnostics

### 1. Performance Profiling

```rust
use std::time::Instant;
use std::collections::HashMap;

/// Advanced performance profiler for message processing pipelines
pub struct PerformanceProfiler {
    samples: Vec<ProfileSample>,
    operation_stats: HashMap<String, OperationStats>,
}

#[derive(Debug, Clone)]
pub struct ProfileSample {
    timestamp: u64,
    operation: String,
    duration_ns: u64,
    success: bool,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct OperationStats {
    count: u64,
    total_duration_ns: u64,
    min_duration_ns: u64,
    max_duration_ns: u64,
    error_count: u64,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            operation_stats: HashMap::new(),
        }
    }
    
    /// Record operation timing
    pub fn record_operation<F, R>(&mut self, operation: &str, f: F) -> R 
    where F: FnOnce() -> Result<R, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let start_ns = csf_time::now().unwrap().as_nanos();
        
        let result = f();
        
        let duration = start.elapsed();
        let duration_ns = duration.as_nanos() as u64;
        let success = result.is_ok();
        
        // Record sample
        self.samples.push(ProfileSample {
            timestamp: start_ns,
            operation: operation.to_string(),
            duration_ns,
            success,
            metadata: HashMap::new(),
        });
        
        // Update stats
        let stats = self.operation_stats.entry(operation.to_string()).or_default();
        stats.count += 1;
        stats.total_duration_ns += duration_ns;
        
        if stats.count == 1 {
            stats.min_duration_ns = duration_ns;
            stats.max_duration_ns = duration_ns;
        } else {
            stats.min_duration_ns = stats.min_duration_ns.min(duration_ns);
            stats.max_duration_ns = stats.max_duration_ns.max(duration_ns);
        }
        
        if !success {
            stats.error_count += 1;
        }
        
        match result {
            Ok(value) => value,
            Err(e) => panic!("Operation {} failed: {}", operation, e),
        }
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut operation_reports = Vec::new();
        
        for (operation, stats) in &self.operation_stats {
            let avg_duration_ns = if stats.count > 0 {
                stats.total_duration_ns / stats.count
            } else {
                0
            };
            
            let error_rate = if stats.count > 0 {
                (stats.error_count as f64 / stats.count as f64) * 100.0
            } else {
                0.0
            };
            
            operation_reports.push(OperationReport {
                operation: operation.clone(),
                count: stats.count,
                avg_duration_ns,
                min_duration_ns: stats.min_duration_ns,
                max_duration_ns: stats.max_duration_ns,
                error_rate_percent: error_rate,
            });
        }
        
        // Sort by average duration (slowest first)
        operation_reports.sort_by(|a, b| b.avg_duration_ns.cmp(&a.avg_duration_ns));
        
        PerformanceReport {
            total_samples: self.samples.len(),
            operations: operation_reports,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceReport {
    total_samples: usize,
    operations: Vec<OperationReport>,
}

#[derive(Debug)]
pub struct OperationReport {
    operation: String,
    count: u64,
    avg_duration_ns: u64,
    min_duration_ns: u64,
    max_duration_ns: u64,
    error_rate_percent: f64,
}
```

This comprehensive performance and troubleshooting guide provides the tools and techniques necessary to optimize and maintain ARES ChronoFabric systems in production environments. The examples demonstrate real-world solutions to common performance challenges and provide detailed diagnostic capabilities for complex distributed systems.

**Document Information**:
- **Generated**: August 26, 2025
- **Project**: ARES ChronoFabric v0.1.0  
- **License**: Proprietary - ARES Systems