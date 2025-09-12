//! DRPP System Monitoring and Visualization
//!
//! Real-time monitoring dashboard for observing emergent behavior patterns,
//! energy evolution, and phase transitions in the ARES ChronoFabric system.

use super::runtime::{DrppRuntime, PhaseTransitionEvent, RuntimeEvent, RuntimeStats};
use crate::variational::PhaseRegion;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{sync::broadcast, time::interval};
use tracing::{debug, info, warn};

/// Real-time monitoring dashboard for DRPP system
pub struct DrppMonitor {
    /// Connection to runtime for data collection
    runtime: Arc<DrppRuntime>,

    /// Event stream from runtime
    event_receiver: broadcast::Receiver<RuntimeEvent>,

    /// Historical data storage
    metrics_history: Arc<RwLock<MetricsHistory>>,

    /// Dashboard configuration
    config: MonitorConfig,

    /// Current dashboard state
    dashboard_state: Arc<RwLock<DashboardState>>,

    /// Background monitoring task handle
    monitor_task: Option<tokio::task::JoinHandle<()>>,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Update interval for metrics collection (milliseconds)
    pub update_interval_ms: u64,

    /// History retention window (number of data points)
    pub history_window: usize,

    /// Enable detailed component tracking
    pub track_components: bool,

    /// Performance monitoring enabled
    pub performance_monitoring: bool,

    /// Pattern detection sensitivity (0.0 to 1.0)
    pub pattern_sensitivity: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 100,
            history_window: 1000,
            track_components: true,
            performance_monitoring: true,
            pattern_sensitivity: 0.5,
        }
    }
}

/// Historical metrics data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistory {
    /// Timestamps for data points
    pub timestamps: VecDeque<f64>,

    /// System energy evolution
    pub energy_history: VecDeque<f64>,

    /// Phase transition events
    pub transition_events: VecDeque<PhaseTransitionEventRecord>,

    /// Component count by phase region over time
    pub phase_distribution_history: VecDeque<HashMap<PhaseRegion, usize>>,

    /// Performance metrics over time
    pub performance_history: VecDeque<RuntimeStats>,

    /// Detected patterns over time
    pub pattern_events: VecDeque<PatternEventRecord>,
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self {
            timestamps: VecDeque::new(),
            energy_history: VecDeque::new(),
            transition_events: VecDeque::new(),
            phase_distribution_history: VecDeque::new(),
            performance_history: VecDeque::new(),
            pattern_events: VecDeque::new(),
        }
    }
}

/// Serializable phase transition event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionEventRecord {
    pub timestamp: f64,
    pub transition_type: String,
    pub component_count: usize,
    pub energy_delta: f64,
    pub severity: f64,
}

/// Serializable pattern detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEventRecord {
    pub timestamp: f64,
    pub pattern_type: String,
    pub confidence: f64,
    pub participant_count: usize,
}

/// Current dashboard visualization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Current system energy level
    pub current_energy: f64,

    /// Current phase distribution
    pub phase_distribution: HashMap<String, usize>,

    /// Recent transition events (last 10)
    pub recent_transitions: Vec<PhaseTransitionEventRecord>,

    /// Current performance metrics
    pub current_performance: PerformanceSnapshot,

    /// System health status
    pub health_status: SystemHealthStatus,

    /// Active emergent patterns
    pub active_patterns: Vec<PatternEventRecord>,

    /// Prediction metrics
    pub predictions: PredictionMetrics,
}

/// Current performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub processing_rate_ops_per_sec: f64,
    pub average_latency_us: f64,
    pub memory_usage_mb: f64,
    pub energy_computation_rate: f64,
    pub system_uptime_seconds: f64,
}

/// Overall system health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Current status level
    pub status: HealthLevel,

    /// Health indicators
    pub indicators: HashMap<String, f64>,

    /// Active warnings
    pub warnings: Vec<String>,

    /// System stability metric
    pub stability_score: f64,
}

/// System health levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthLevel {
    Optimal,
    Good,
    Warning,
    Critical,
    Emergency,
}

/// Prediction and trend analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    /// Predicted energy trend direction
    pub energy_trend: TrendDirection,

    /// Likelihood of phase transition in next interval
    pub transition_probability: f64,

    /// Predicted system stability
    pub stability_forecast: f64,

    /// Performance trend prediction
    pub performance_trend: TrendDirection,

    /// Pattern emergence probability
    pub pattern_emergence_probability: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
    Oscillating,
    Chaotic,
}

impl DrppMonitor {
    /// Create a new DRPP monitoring dashboard
    pub fn new(runtime: Arc<DrppRuntime>, config: MonitorConfig) -> Self {
        let event_receiver = runtime.subscribe_events();

        Self {
            runtime,
            event_receiver,
            metrics_history: Arc::new(RwLock::new(MetricsHistory::default())),
            config,
            dashboard_state: Arc::new(RwLock::new(Self::create_initial_dashboard_state())),
            monitor_task: None,
        }
    }

    /// Start the monitoring system
    pub async fn start(&mut self) -> Result<(), MonitoringError> {
        info!("Starting DRPP monitoring dashboard");

        let runtime = Arc::clone(&self.runtime);
        let metrics_history = Arc::clone(&self.metrics_history);
        let dashboard_state = Arc::clone(&self.dashboard_state);
        let config = self.config.clone();
        let mut event_receiver = self.runtime.subscribe_events();

        let monitor_task = tokio::spawn(async move {
            let mut update_interval = interval(Duration::from_millis(config.update_interval_ms));

            loop {
                update_interval.tick().await;

                // Collect current metrics
                let current_energy = runtime.get_system_energy();
                let phase_distribution = runtime.get_phase_distribution();
                let runtime_stats = runtime.get_stats();
                let timestamp = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

                // Update metrics history
                {
                    let mut history = metrics_history.write().unwrap();
                    Self::update_metrics_history(
                        &mut history,
                        timestamp,
                        current_energy,
                        phase_distribution.clone(),
                        runtime_stats.clone(),
                        &config,
                    );
                }

                // Process recent events
                while let Ok(event) = event_receiver.try_recv() {
                    let mut history = metrics_history.write().unwrap();
                    Self::process_runtime_event(&mut history, event, timestamp);
                }

                // Update dashboard state
                {
                    let history = metrics_history.read().unwrap();
                    let mut dashboard = dashboard_state.write().unwrap();
                    Self::update_dashboard_state(
                        &mut dashboard,
                        &history,
                        current_energy,
                        phase_distribution,
                        runtime_stats,
                    );
                }
            }
        });

        self.monitor_task = Some(monitor_task);
        info!("DRPP monitoring dashboard started");
        Ok(())
    }

    /// Get current dashboard state for visualization
    pub fn get_dashboard_state(&self) -> DashboardState {
        self.dashboard_state.read().unwrap().clone()
    }

    /// Get historical metrics data
    pub fn get_metrics_history(&self) -> MetricsHistory {
        self.metrics_history.read().unwrap().clone()
    }

    /// Export metrics data for analysis
    pub fn export_metrics(&self, format: ExportFormat) -> Result<String, MonitoringError> {
        let history = self.metrics_history.read().unwrap();

        match format {
            ExportFormat::Json => serde_json::to_string_pretty(&*history)
                .map_err(|e| MonitoringError::SerializationError(e.to_string())),
            ExportFormat::Csv => self.export_csv_format(&history),
        }
    }

    /// Generate system health report
    pub fn generate_health_report(&self) -> SystemHealthReport {
        let dashboard = self.dashboard_state.read().unwrap();
        let history = self.metrics_history.read().unwrap();

        SystemHealthReport {
            generated_at: chrono::Utc::now(),
            overall_health: dashboard.health_status.clone(),
            performance_summary: self.analyze_performance_trends(&history),
            energy_analysis: self.analyze_energy_patterns(&history),
            transition_analysis: self.analyze_transition_patterns(&history),
            pattern_summary: self.summarize_detected_patterns(&history),
            recommendations: self.generate_recommendations(&dashboard, &history),
        }
    }

    /// Stop the monitoring system
    pub async fn stop(&mut self) -> Result<(), MonitoringError> {
        if let Some(task) = self.monitor_task.take() {
            task.abort();
            info!("DRPP monitoring dashboard stopped");
        }
        Ok(())
    }

    /// Create initial dashboard state
    fn create_initial_dashboard_state() -> DashboardState {
        DashboardState {
            current_energy: 0.0,
            phase_distribution: HashMap::new(),
            recent_transitions: Vec::new(),
            current_performance: PerformanceSnapshot {
                processing_rate_ops_per_sec: 0.0,
                average_latency_us: 0.0,
                memory_usage_mb: 0.0,
                energy_computation_rate: 0.0,
                system_uptime_seconds: 0.0,
            },
            health_status: SystemHealthStatus {
                health_score: 1.0,
                status: HealthLevel::Optimal,
                indicators: HashMap::new(),
                warnings: Vec::new(),
                stability_score: 1.0,
            },
            active_patterns: Vec::new(),
            predictions: PredictionMetrics {
                energy_trend: TrendDirection::Stable,
                transition_probability: 0.0,
                stability_forecast: 1.0,
                performance_trend: TrendDirection::Stable,
                pattern_emergence_probability: 0.0,
            },
        }
    }

    /// Update metrics history with new data point
    fn update_metrics_history(
        history: &mut MetricsHistory,
        timestamp: f64,
        energy: f64,
        phase_dist: HashMap<PhaseRegion, usize>,
        stats: RuntimeStats,
        config: &MonitorConfig,
    ) {
        // Add new data points
        history.timestamps.push_back(timestamp);
        history.energy_history.push_back(energy);
        history.phase_distribution_history.push_back(phase_dist);
        history.performance_history.push_back(stats);

        // Maintain window size
        let max_size = config.history_window;
        while history.timestamps.len() > max_size {
            history.timestamps.pop_front();
            history.energy_history.pop_front();
            history.phase_distribution_history.pop_front();
            history.performance_history.pop_front();
        }
    }

    /// Process runtime events and update history
    fn process_runtime_event(history: &mut MetricsHistory, event: RuntimeEvent, timestamp: f64) {
        match event {
            RuntimeEvent::PhaseTransition(transition_event) => {
                let record = PhaseTransitionEventRecord {
                    timestamp,
                    transition_type: format!("{:?}", transition_event.transition_type),
                    component_count: transition_event.components.len(),
                    energy_delta: transition_event.energy_delta,
                    severity: transition_event.severity,
                };
                history.transition_events.push_back(record);
            }
            RuntimeEvent::PatternDetected {
                pattern_type,
                confidence,
                components,
            } => {
                let record = PatternEventRecord {
                    timestamp,
                    pattern_type,
                    confidence,
                    participant_count: components.len(),
                };
                history.pattern_events.push_back(record);
            }
            _ => {} // Handle other events as needed
        }
    }

    /// Update current dashboard state
    fn update_dashboard_state(
        dashboard: &mut DashboardState,
        history: &MetricsHistory,
        current_energy: f64,
        phase_distribution: HashMap<PhaseRegion, usize>,
        stats: RuntimeStats,
    ) {
        // Update current values
        dashboard.current_energy = current_energy;
        dashboard.phase_distribution = phase_distribution
            .iter()
            .map(|(k, v)| (format!("{:?}", k), *v))
            .collect();

        // Update performance snapshot
        dashboard.current_performance = PerformanceSnapshot {
            processing_rate_ops_per_sec: stats.energy_computations_per_sec,
            average_latency_us: stats.avg_processing_latency_us,
            memory_usage_mb: stats.memory_usage_bytes as f64 / (1024.0 * 1024.0),
            energy_computation_rate: stats.energy_computations_per_sec,
            system_uptime_seconds: stats.uptime_seconds,
        };

        // Update recent transitions
        dashboard.recent_transitions = history
            .transition_events
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        // Update active patterns
        dashboard.active_patterns = history
            .pattern_events
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        // Update health status
        dashboard.health_status = Self::calculate_health_status(history, &stats);

        // Update predictions
        dashboard.predictions = Self::calculate_predictions(history);
    }

    /// Calculate system health status
    fn calculate_health_status(
        history: &MetricsHistory,
        stats: &RuntimeStats,
    ) -> SystemHealthStatus {
        let mut health_score = 1.0;
        let mut warnings = Vec::new();
        let mut indicators = HashMap::new();

        // Performance health indicators
        let latency_health = if stats.avg_processing_latency_us < 100.0 {
            1.0
        } else if stats.avg_processing_latency_us < 1000.0 {
            0.7
        } else {
            0.3
        };

        indicators.insert("latency_health".to_string(), latency_health);
        health_score *= latency_health;

        // Memory usage health
        let memory_health = if stats.memory_usage_bytes < 1_000_000_000 {
            1.0
        } else {
            0.5
        };
        indicators.insert("memory_health".to_string(), memory_health);
        health_score *= memory_health;

        // Energy stability
        let energy_stability = Self::calculate_energy_stability(history);
        indicators.insert("energy_stability".to_string(), energy_stability);
        health_score *= energy_stability;

        // Generate warnings
        if stats.avg_processing_latency_us > 1000.0 {
            warnings.push("High processing latency detected".to_string());
        }
        if stats.memory_usage_bytes > 2_000_000_000 {
            warnings.push("High memory usage".to_string());
        }

        let status = match health_score {
            s if s > 0.9 => HealthLevel::Optimal,
            s if s > 0.7 => HealthLevel::Good,
            s if s > 0.5 => HealthLevel::Warning,
            s if s > 0.2 => HealthLevel::Critical,
            _ => HealthLevel::Emergency,
        };

        SystemHealthStatus {
            health_score,
            status,
            indicators,
            warnings,
            stability_score: energy_stability,
        }
    }

    /// Calculate energy stability metric
    fn calculate_energy_stability(history: &MetricsHistory) -> f64 {
        if history.energy_history.len() < 10 {
            return 1.0;
        }

        // Calculate coefficient of variation over recent history
        let recent_energies: Vec<f64> = history
            .energy_history
            .iter()
            .rev()
            .take(50)
            .cloned()
            .collect();

        let mean = recent_energies.iter().sum::<f64>() / recent_energies.len() as f64;
        let variance = recent_energies
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>()
            / recent_energies.len() as f64;

        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };

        // Convert to stability score (lower variation = higher stability)
        (1.0 / (1.0 + coefficient_of_variation)).max(0.0).min(1.0)
    }

    /// Calculate prediction metrics
    fn calculate_predictions(history: &MetricsHistory) -> PredictionMetrics {
        PredictionMetrics {
            energy_trend: Self::analyze_energy_trend(history),
            transition_probability: Self::calculate_transition_probability(history),
            stability_forecast: Self::forecast_stability(history),
            performance_trend: Self::analyze_performance_trend(history),
            pattern_emergence_probability: Self::calculate_pattern_probability(history),
        }
    }

    /// Analyze energy trend direction
    fn analyze_energy_trend(history: &MetricsHistory) -> TrendDirection {
        if history.energy_history.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent: Vec<f64> = history
            .energy_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();
        let early_avg = recent[10..].iter().sum::<f64>() / 10.0;
        let late_avg = recent[..10].iter().sum::<f64>() / 10.0;

        let change_ratio = if early_avg > 0.0 {
            (late_avg - early_avg) / early_avg
        } else {
            0.0
        };

        match change_ratio {
            x if x > 0.1 => TrendDirection::Increasing,
            x if x < -0.1 => TrendDirection::Decreasing,
            _ => TrendDirection::Stable,
        }
    }

    /// Calculate phase transition probability
    fn calculate_transition_probability(history: &MetricsHistory) -> f64 {
        // Based on recent transition frequency and energy variance
        let recent_transitions = history.transition_events.iter().rev().take(10).count();

        let base_probability = recent_transitions as f64 / 10.0;
        let energy_variance_factor = if history.energy_history.len() > 5 {
            let recent: Vec<f64> = history
                .energy_history
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            let variance = recent
                .iter()
                .map(|e| (e - recent.iter().sum::<f64>() / recent.len() as f64).powi(2))
                .sum::<f64>()
                / recent.len() as f64;
            variance.min(1.0)
        } else {
            0.0
        };

        (base_probability + energy_variance_factor * 0.5).min(1.0)
    }

    /// Forecast system stability
    fn forecast_stability(_history: &MetricsHistory) -> f64 {
        // Simple stability forecast based on recent performance
        0.8 // Placeholder implementation
    }

    /// Analyze performance trend
    fn analyze_performance_trend(_history: &MetricsHistory) -> TrendDirection {
        // Analyze processing rate trends
        TrendDirection::Stable // Placeholder implementation
    }

    /// Calculate pattern emergence probability
    fn calculate_pattern_probability(history: &MetricsHistory) -> f64 {
        let recent_patterns = history.pattern_events.iter().rev().take(5).count();

        (recent_patterns as f64 / 5.0).min(1.0)
    }

    /// Export metrics in CSV format
    fn export_csv_format(&self, _history: &MetricsHistory) -> Result<String, MonitoringError> {
        // CSV export implementation
        Ok("timestamp,energy,transitions\n".to_string()) // Placeholder
    }

    /// Analyze performance trends
    fn analyze_performance_trends(&self, history: &MetricsHistory) -> String {
        let mut analysis = Vec::new();
        
        // Calculate average throughput trend
        let throughput_values: Vec<f64> = history.throughput_history.iter()
            .map(|&v| v as f64).collect();
        if !throughput_values.is_empty() {
            let avg_throughput = throughput_values.iter().sum::<f64>() / throughput_values.len() as f64;
            let recent_avg = throughput_values.iter().rev().take(10).sum::<f64>() / 10.0.min(throughput_values.len() as f64);
            let trend = ((recent_avg - avg_throughput) / avg_throughput * 100.0).round();
            analysis.push(format!("Throughput trend: {:.1}% {}", 
                trend.abs(), if trend > 0.0 { "↑" } else if trend < 0.0 { "↓" } else { "→" }));
        }
        
        // Analyze latency stability
        let latency_values: Vec<f64> = history.latency_history.iter()
            .map(|&d| d.as_secs_f64() * 1000.0).collect();
        if !latency_values.is_empty() {
            let mean = latency_values.iter().sum::<f64>() / latency_values.len() as f64;
            let variance = latency_values.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / latency_values.len() as f64;
            let std_dev = variance.sqrt();
            let cv = (std_dev / mean * 100.0).round();
            analysis.push(format!("Latency stability: CV={:.1}% ({})", 
                cv, if cv < 10.0 { "excellent" } else if cv < 25.0 { "good" } else { "needs attention" }));
        }
        
        // Memory usage trend
        if !history.memory_usage.is_empty() {
            let recent_memory = history.memory_usage.iter().rev().take(5)
                .map(|&v| v as f64).sum::<f64>() / 5.0.min(history.memory_usage.len() as f64);
            analysis.push(format!("Memory usage: {:.1} MB", recent_memory / 1_048_576.0));
        }
        
        if analysis.is_empty() {
            "Insufficient data for performance analysis".to_string()
        } else {
            analysis.join(", ")
        }
    }

    /// Analyze energy patterns
    fn analyze_energy_patterns(&self, history: &MetricsHistory) -> String {
        let mut patterns = Vec::new();
        
        // Calculate energy efficiency (ops per unit energy)
        if !history.energy_consumed.is_empty() && !history.throughput_history.is_empty() {
            let total_energy: f64 = history.energy_consumed.iter().sum();
            let total_ops: u64 = history.throughput_history.iter().sum();
            if total_energy > 0.0 {
                let efficiency = total_ops as f64 / total_energy;
                patterns.push(format!("Energy efficiency: {:.2} ops/J", efficiency));
            }
            
            // Detect energy spikes
            let avg_energy = total_energy / history.energy_consumed.len() as f64;
            let spikes = history.energy_consumed.iter()
                .filter(|&&e| e > avg_energy * 1.5)
                .count();
            if spikes > 0 {
                patterns.push(format!("Energy spikes detected: {} events", spikes));
            }
            
            // Power consumption trend
            let recent_energy: f64 = history.energy_consumed.iter().rev()
                .take(10).sum::<f64>() / 10.0.min(history.energy_consumed.len() as f64);
            let power_estimate = recent_energy * 1000.0; // Convert to watts assuming 1ms sampling
            patterns.push(format!("Estimated power: {:.2}W", power_estimate));
        }
        
        if patterns.is_empty() {
            "No energy consumption data available".to_string()
        } else {
            patterns.join(", ")
        }
    }

    /// Analyze transition patterns
    fn analyze_transition_patterns(&self, history: &MetricsHistory) -> String {
        let mut patterns = Vec::new();
        
        // Analyze phase transition frequency
        if !history.phase_transitions.is_empty() {
            let transition_count = history.phase_transitions.len();
            let time_span = history.energy_consumed.len() as f64; // Assuming 1 sample per time unit
            let transition_rate = transition_count as f64 / time_span.max(1.0);
            patterns.push(format!("Transition rate: {:.2}/s", transition_rate));
            
            // Detect transition clustering
            let mut consecutive_transitions = 0;
            let mut max_cluster = 0;
            for window in history.phase_transitions.windows(2) {
                let time_diff = (window[1] - window[0]) as f64;
                if time_diff < 100.0 { // Within 100ms
                    consecutive_transitions += 1;
                    max_cluster = max_cluster.max(consecutive_transitions);
                } else {
                    consecutive_transitions = 0;
                }
            }
            if max_cluster > 2 {
                patterns.push(format!("Transition clustering detected: max {} consecutive", max_cluster));
            }
            
            // Phase stability analysis
            let avg_phase_duration = time_span / (transition_count as f64 + 1.0);
            patterns.push(format!("Avg phase duration: {:.1}ms", avg_phase_duration));
        }
        
        if patterns.is_empty() {
            "No phase transition data available".to_string()
        } else {
            patterns.join(", ")
        }
    }

    /// Summarize detected patterns
    fn summarize_detected_patterns(&self, history: &MetricsHistory) -> String {
        let mut summary = Vec::new();
        
        // System health score (0-100)
        let mut health_score = 100.0;
        
        // Check error rate
        let total_ops = history.throughput_history.iter().sum::<u64>() as f64;
        let error_rate = if total_ops > 0.0 {
            history.error_count as f64 / total_ops
        } else { 0.0 };
        
        if error_rate > 0.01 { health_score -= 20.0; } // >1% errors
        if error_rate > 0.05 { health_score -= 30.0; } // >5% errors
        
        // Check latency consistency
        if !history.latency_history.is_empty() {
            let latencies: Vec<f64> = history.latency_history.iter()
                .map(|d| d.as_secs_f64() * 1000.0).collect();
            let max_latency = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            if max_latency > avg_latency * 10.0 { health_score -= 15.0; } // Severe spikes
        }
        
        // Memory pressure
        if !history.memory_usage.is_empty() {
            let max_memory = *history.memory_usage.iter().max().unwrap_or(&0);
            let avg_memory = history.memory_usage.iter().sum::<usize>() / history.memory_usage.len();
            if max_memory > avg_memory * 2 { health_score -= 10.0; } // Memory spikes
        }
        
        summary.push(format!("System health: {:.0}/100", health_score.max(0.0)));
        
        // Pattern detection
        if error_rate > 0.001 {
            summary.push(format!("Error pattern: {:.3}% failure rate", error_rate * 100.0));
        }
        
        // Performance classification
        let perf_class = if health_score > 90.0 { "optimal" }
            else if health_score > 70.0 { "good" }
            else if health_score > 50.0 { "degraded" }
            else { "critical" };
        summary.push(format!("Performance: {}", perf_class));
        
        summary.join(", ")
    }

    /// Generate system recommendations
    fn generate_recommendations(
        &self,
        _dashboard: &DashboardState,
        _history: &MetricsHistory,
    ) -> Vec<String> {
        vec!["System operating normally".to_string()]
    }
}

/// Export format options
pub enum ExportFormat {
    Json,
    Csv,
}

/// Comprehensive system health report
pub struct SystemHealthReport {
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub overall_health: SystemHealthStatus,
    pub performance_summary: String,
    pub energy_analysis: String,
    pub transition_analysis: String,
    pub pattern_summary: String,
    pub recommendations: Vec<String>,
}

/// Monitoring system errors
#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    #[error("Monitoring system not initialized")]
    NotInitialized,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Data export error: {0}")]
    ExportError(String),

    #[error("Internal monitoring error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integration::runtime::DrppConfig;

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = DrppConfig::default();
        let runtime = Arc::new(DrppRuntime::new(config).unwrap());
        let monitor_config = MonitorConfig::default();

        let monitor = DrppMonitor::new(runtime, monitor_config);
        let dashboard = monitor.get_dashboard_state();

        assert_eq!(dashboard.current_energy, 0.0);
        assert_eq!(dashboard.health_status.status, HealthLevel::Optimal);
    }

    #[tokio::test]
    async fn test_metrics_export() {
        let config = DrppConfig::default();
        let runtime = Arc::new(DrppRuntime::new(config).unwrap());
        let monitor_config = MonitorConfig::default();

        let monitor = DrppMonitor::new(runtime, monitor_config);
        let json_export = monitor.export_metrics(ExportFormat::Json).unwrap();

        assert!(json_export.contains("timestamps"));
        assert!(json_export.contains("energy_history"));
    }
}
