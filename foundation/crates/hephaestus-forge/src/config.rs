//! Configuration system for Hephaestus Forge
//! Enterprise-grade configuration with builder pattern

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::ops::Range;
use crate::validation::hardening::RateLimiterConfig;

/// Core configuration for Hephaestus Forge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeConfig {
    /// Operational mode
    pub mode: OperationalMode,
    
    /// Risk configuration
    pub risk_config: RiskConfig,
    
    /// Resource limits
    pub resource_limits: ResourceLimits,
    
    /// Synthesis configuration
    pub synthesis_config: SynthesisConfig,
    
    /// Testing configuration
    pub testing_config: TestingConfig,
    
    /// Consensus parameters
    pub consensus_config: ConsensusConfig,
    
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
    
    /// HITL configuration
    pub hitl_config: Option<HitlConfig>,

    /// Rate limiter configuration
    pub rate_limiter_config: RateLimiterConfig,
}

/// Operational modes for the Forge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalMode {
    /// Fully autonomous operation
    FullyAutonomous,
    
    /// Alias for FullyAutonomous
    Autonomous,
    
    /// Supervised with automatic approval for low risk
    Supervised,
    
    /// All changes require human approval
    Manual,
    
    /// Read-only mode for analysis
    Observer,
}

/// Risk configuration and tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum allowed risk score (0.0 - 1.0)
    pub max_risk_score: f64,
    
    /// Risk tolerance level
    pub risk_tolerance: f64,
    
    /// Autonomy levels for different risk tiers
    pub autonomy_levels: Vec<(RiskTier, AutonomyLevel)>,
    
    /// Emergency stop threshold
    pub emergency_stop_threshold: f64,
    
    /// Risk factor weights
    pub risk_factors: RiskFactorWeights,
}

/// Risk tiers with ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskTier {
    Low(Range<f64>),
    Medium(Range<f64>),
    High(Range<f64>),
    Critical(Range<f64>),
}

/// Autonomy levels for decision making
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyLevel {
    FullyAutomated,
    Supervised,
    HumanRequired,
    Blocked,
}

/// Risk factor weights for assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactorWeights {
    pub complexity: f64,
    pub criticality: f64,
    pub confidence: f64,
}

/// Resource limits for Forge operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores for synthesis
    pub synthesis_cpu_cores: usize,
    
    /// Maximum memory for testing (GB)
    pub testing_memory_gb: usize,
    
    /// Maximum concurrent optimizations
    pub max_concurrent_optimizations: usize,
    
    /// Synthesis timeout
    pub synthesis_timeout: Duration,
    
    /// Maximum sandbox instances
    pub max_sandbox_instances: usize,
}

/// Testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingConfig {
    /// Enable chaos engineering
    pub chaos_engineering: bool,
    
    /// Enable differential testing
    pub differential_testing: bool,
    
    /// Test timeout per module
    pub test_timeout: Duration,
    
    /// Traffic replay duration
    pub replay_duration: Duration,
    
    /// Shadow traffic percentage
    pub shadow_traffic_percent: f64,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus nodes
    pub consensus_nodes: Vec<String>,
    
    /// PBFT enabled
    pub pbft_enabled: bool,
    
    /// Quantum resistant cryptography
    pub quantum_resistant: bool,
    
    /// Consensus timeout
    pub consensus_timeout: Duration,
    
    /// Minimum validators
    pub min_validators: usize,
    
    /// Approval quorum percentage
    pub approval_quorum: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics port
    pub metrics_port: u16,
    
    /// Dashboard port
    pub dashboard_port: u16,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Metrics export interval
    pub export_interval: Duration,
}

/// Human-in-the-loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitlConfig {
    /// Webhook for approval requests
    pub approval_webhook: String,
    
    /// Approval timeout
    pub timeout: Duration,
    
    /// Escalation path
    pub escalation_path: Vec<String>,
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics: MetricsConfig,
    pub tracing: TracingConfig,
    pub logging: LoggingConfig,
    pub alerting: AlertingConfig,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub prometheus_port: u16,
    pub custom_metrics: Vec<CustomMetric>,
}

/// Custom metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

impl CustomMetric {
    pub fn new(name: impl Into<String>, metric_type: MetricType) -> Self {
        Self {
            name: name.into(),
            metric_type,
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub jaeger_endpoint: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub structured: bool,
    pub destinations: Vec<LogDestination>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogDestination {
    Stdout,
    File(String),
    Elasticsearch(String),
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub rules: Vec<AlertRule>,
    pub destinations: Vec<AlertDestination>,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub severity: Severity,
}

impl AlertRule {
    pub fn new(name: impl Into<String>, condition: impl Into<String>, severity: Severity) -> Self {
        Self {
            name: name.into(),
            condition: condition.into(),
            severity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertDestination {
    Slack(String),
    PagerDuty(String),
    Email(String),
}

// Builder pattern implementation
pub struct ForgeConfigBuilder {
    mode: OperationalMode,
    risk_tolerance: f64,
    max_concurrent_optimizations: usize,
    synthesis_timeout: Duration,
    consensus_nodes: Vec<String>,
    enable_resonance: bool,
    energy_threshold: f64,
    detection_sensitivity: f64,
    rate_limiter_config: RateLimiterConfig,
}

impl ForgeConfigBuilder {
    pub fn new() -> Self {
        Self {
            mode: OperationalMode::Supervised,
            risk_tolerance: 0.5,
            max_concurrent_optimizations: 4,
            synthesis_timeout: Duration::from_secs(300),
            consensus_nodes: Vec::new(),
            enable_resonance: true,
            energy_threshold: 0.1,
            detection_sensitivity: 0.8,
            rate_limiter_config: RateLimiterConfig {
                max_rps: 100.0,
                burst_capacity: 200,
                refill_interval: Duration::from_secs(1),
            },
        }
    }
    
    pub fn mode(mut self, mode: OperationalMode) -> Self {
        self.mode = mode;
        self
    }
    
    pub fn enable_resonance_processing(mut self, enabled: bool) -> Self {
        self.enable_resonance = enabled;
        self
    }
    
    pub fn energy_threshold(mut self, threshold: f64) -> Self {
        self.energy_threshold = threshold;
        self
    }
    
    pub fn detection_sensitivity(mut self, sensitivity: f64) -> Self {
        self.detection_sensitivity = sensitivity;
        self
    }
    
    pub fn risk_tolerance(mut self, tolerance: f64) -> Self {
        self.risk_tolerance = tolerance;
        self
    }
    
    pub fn max_concurrent_optimizations(mut self, max: usize) -> Self {
        self.max_concurrent_optimizations = max;
        self
    }
    
    pub fn synthesis_timeout(mut self, timeout: Duration) -> Self {
        self.synthesis_timeout = timeout;
        self
    }
    
    pub fn consensus_nodes(mut self, nodes: Vec<impl Into<String>>) -> Self {
        self.consensus_nodes = nodes.into_iter().map(Into::into).collect();
        self
    }
    
    pub fn risk_config(self, config: RiskConfig) -> Self {
        // Custom risk config
        self
    }

    pub fn rate_limit(mut self, max_rps: f64, burst_capacity: usize) -> Self {
        self.rate_limiter_config = RateLimiterConfig {
            max_rps,
            burst_capacity,
            refill_interval: Duration::from_secs(1),
        };
        self
    }
    
    pub fn build(self) -> Result<ForgeConfig, ForgeError> {
        Ok(ForgeConfig {
            mode: self.mode,
            risk_config: RiskConfig {
                max_risk_score: 0.8,
                risk_tolerance: self.risk_tolerance,
                autonomy_levels: vec![
                    (RiskTier::Low(0.0..0.3), AutonomyLevel::FullyAutomated),
                    (RiskTier::Medium(0.3..0.6), AutonomyLevel::Supervised),
                    (RiskTier::High(0.6..0.8), AutonomyLevel::HumanRequired),
                    (RiskTier::Critical(0.8..1.0), AutonomyLevel::Blocked),
                ],
                emergency_stop_threshold: 0.9,
                risk_factors: RiskFactorWeights {
                    complexity: 0.3,
                    criticality: 0.4,
                    confidence: 0.3,
                },
            },
            resource_limits: ResourceLimits {
                synthesis_cpu_cores: 8,
                testing_memory_gb: 16,
                max_concurrent_optimizations: self.max_concurrent_optimizations,
                synthesis_timeout: self.synthesis_timeout,
                max_sandbox_instances: 10,
            },
            synthesis_config: SynthesisConfig::default(),
            testing_config: TestingConfig {
                chaos_engineering: true,
                differential_testing: true,
                test_timeout: Duration::from_secs(60),
                replay_duration: Duration::from_secs(300),
                shadow_traffic_percent: 0.01,
            },
            consensus_config: ConsensusConfig {
                consensus_nodes: self.consensus_nodes,
                pbft_enabled: true,
                quantum_resistant: false,
                consensus_timeout: Duration::from_secs(30),
                min_validators: 3,
                approval_quorum: 0.67,
            },
            monitoring_config: MonitoringConfig {
                metrics_port: 9090,
                dashboard_port: 8080,
                enable_tracing: true,
                export_interval: Duration::from_secs(10),
            },
            hitl_config: None,
            rate_limiter_config: self.rate_limiter_config,
        })
    }
}

impl ForgeConfig {
    pub fn builder() -> ForgeConfigBuilder {
        ForgeConfigBuilder::new()
    }
}

impl Default for ForgeConfig {
    fn default() -> Self {
        ForgeConfigBuilder::new().build().unwrap()
    }
}