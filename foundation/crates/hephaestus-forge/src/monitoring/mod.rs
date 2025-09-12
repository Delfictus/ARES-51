//! Prometheus metrics and monitoring integration

use prometheus::{
    Encoder, TextEncoder, Counter, Gauge, Histogram, HistogramOpts,
    register_counter, register_gauge, register_histogram,
};
use warp::Filter;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Prometheus metrics for Hephaestus Forge
pub struct ForgeMetrics {
    // Resonance metrics
    pub resonance_computations: Counter,
    pub resonance_coherence: Gauge,
    pub resonance_frequency: Gauge,
    pub phase_lattice_energy: Gauge,
    
    // Performance metrics
    pub computation_duration: Histogram,
    pub optimization_latency: Histogram,
    pub synthesis_duration: Histogram,
    
    // System metrics
    pub active_optimizations: Gauge,
    pub total_optimizations: Counter,
    pub optimization_success_rate: Gauge,
    pub memory_usage_bytes: Gauge,
    pub cpu_usage_percent: Gauge,
    
    // Distributed metrics
    pub distributed_nodes: Gauge,
    pub consensus_latency: Histogram,
    pub network_throughput: Gauge,
}

impl ForgeMetrics {
    pub fn new() -> Self {
        Self {
            resonance_computations: register_counter!(
                "forge_resonance_computations_total",
                "Total number of resonance computations"
            ).unwrap(),
            
            resonance_coherence: register_gauge!(
                "forge_resonance_coherence",
                "Current resonance coherence level (0-1)"
            ).unwrap(),
            
            resonance_frequency: register_gauge!(
                "forge_resonance_frequency_hz",
                "Current dominant resonance frequency"
            ).unwrap(),
            
            phase_lattice_energy: register_gauge!(
                "forge_phase_lattice_energy",
                "Total energy in phase lattice"
            ).unwrap(),
            
            computation_duration: register_histogram!(
                HistogramOpts::new(
                    "forge_computation_duration_seconds",
                    "Duration of resonance computations"
                ).buckets(vec![0.001, 0.01, 0.1, 1.0, 10.0])
            ).unwrap(),
            
            optimization_latency: register_histogram!(
                HistogramOpts::new(
                    "forge_optimization_latency_seconds",
                    "End-to-end optimization latency"
                ).buckets(vec![0.1, 1.0, 10.0, 60.0, 300.0])
            ).unwrap(),
            
            synthesis_duration: register_histogram!(
                HistogramOpts::new(
                    "forge_synthesis_duration_seconds",
                    "Duration of code synthesis"
                ).buckets(vec![0.1, 1.0, 10.0, 60.0])
            ).unwrap(),
            
            active_optimizations: register_gauge!(
                "forge_active_optimizations",
                "Number of currently active optimizations"
            ).unwrap(),
            
            total_optimizations: register_counter!(
                "forge_total_optimizations",
                "Total number of optimizations performed"
            ).unwrap(),
            
            optimization_success_rate: register_gauge!(
                "forge_optimization_success_rate",
                "Success rate of optimizations (0-1)"
            ).unwrap(),
            
            memory_usage_bytes: register_gauge!(
                "forge_memory_usage_bytes",
                "Current memory usage in bytes"
            ).unwrap(),
            
            cpu_usage_percent: register_gauge!(
                "forge_cpu_usage_percent",
                "Current CPU usage percentage"
            ).unwrap(),
            
            distributed_nodes: register_gauge!(
                "forge_distributed_nodes",
                "Number of nodes in distributed lattice"
            ).unwrap(),
            
            consensus_latency: register_histogram!(
                HistogramOpts::new(
                    "forge_consensus_latency_seconds",
                    "Latency of distributed consensus"
                ).buckets(vec![0.01, 0.1, 1.0, 5.0])
            ).unwrap(),
            
            network_throughput: register_gauge!(
                "forge_network_throughput_mbps",
                "Network throughput in Mbps"
            ).unwrap(),
        }
    }
    
    /// Update resonance metrics
    pub fn update_resonance(&self, coherence: f64, frequency: f64, energy: f64) {
        self.resonance_coherence.set(coherence);
        self.resonance_frequency.set(frequency);
        self.phase_lattice_energy.set(energy);
        self.resonance_computations.inc();
    }
    
    /// Record computation timing
    pub fn record_computation(&self, duration_secs: f64) {
        self.computation_duration.observe(duration_secs);
    }
    
    /// Export metrics in Prometheus format
    pub fn export(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// Metrics server for Prometheus scraping
pub struct MetricsServer {
    metrics: Arc<ForgeMetrics>,
    port: u16,
}

impl MetricsServer {
    pub fn new(metrics: Arc<ForgeMetrics>, port: u16) -> Self {
        Self { metrics, port }
    }
    
    /// Start metrics HTTP server
    pub async fn start(self) {
        let metrics = self.metrics.clone();
        
        let metrics_route = warp::path("metrics")
            .map(move || {
                let metrics = metrics.export();
                warp::reply::with_header(metrics, "content-type", "text/plain")
            });
        
        let health = warp::path("health")
            .map(|| warp::reply::json(&serde_json::json!({ "status": "healthy" })));
        
        let routes = metrics_route.or(health);
        
        println!("Metrics server listening on :{}", self.port);
        warp::serve(routes)
            .run(([0, 0, 0, 0], self.port))
            .await;
    }
}

/// Grafana dashboard configuration
pub fn generate_grafana_dashboard() -> serde_json::Value {
    serde_json::json!({
        "dashboard": {
            "title": "Hephaestus Forge - Resonance Monitoring",
            "panels": [
                {
                    "title": "Resonance Coherence",
                    "type": "graph",
                    "targets": [{
                        "expr": "forge_resonance_coherence",
                        "legendFormat": "Coherence"
                    }],
                    "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 }
                },
                {
                    "title": "Phase Lattice Energy",
                    "type": "graph",
                    "targets": [{
                        "expr": "forge_phase_lattice_energy",
                        "legendFormat": "Energy"
                    }],
                    "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 }
                },
                {
                    "title": "Optimization Latency",
                    "type": "heatmap",
                    "targets": [{
                        "expr": "rate(forge_optimization_latency_seconds_bucket[5m])",
                        "format": "heatmap"
                    }],
                    "gridPos": { "x": 0, "y": 8, "w": 12, "h": 8 }
                },
                {
                    "title": "Success Rate",
                    "type": "stat",
                    "targets": [{
                        "expr": "forge_optimization_success_rate * 100",
                        "legendFormat": "Success %"
                    }],
                    "gridPos": { "x": 12, "y": 8, "w": 12, "h": 8 }
                },
                {
                    "title": "Distributed Nodes",
                    "type": "graph",
                    "targets": [{
                        "expr": "forge_distributed_nodes",
                        "legendFormat": "Active Nodes"
                    }],
                    "gridPos": { "x": 0, "y": 16, "w": 12, "h": 8 }
                },
                {
                    "title": "System Resources",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "forge_memory_usage_bytes / 1024 / 1024",
                            "legendFormat": "Memory (MB)"
                        },
                        {
                            "expr": "forge_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        }
                    ],
                    "gridPos": { "x": 12, "y": 16, "w": 12, "h": 8 }
                }
            ]
        }
    })
}