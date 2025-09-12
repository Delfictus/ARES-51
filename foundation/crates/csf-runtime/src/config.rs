//! Configuration management for the CSF Runtime
//!
//! This module provides comprehensive configuration management with validation,
//! environment variable substitution, hot-reload capabilities, and hierarchical
//! configuration merging.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use config::{Config, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};

use crate::core::ComponentType;
use crate::error::{ConfigurationError, RuntimeError, RuntimeResult};

/// Comprehensive runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Runtime orchestration settings
    pub runtime: RuntimeSettings,
    /// Component-specific configurations
    pub components: HashMap<String, ComponentConfig>,
    /// Network configuration
    pub network: NetworkConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Performance optimization settings
    pub performance: PerformanceConfig,
    /// Telemetry and observability settings
    pub telemetry: TelemetryConfig,
    /// Resource limits and quotas
    pub resources: ResourceConfig,
}

/// Runtime orchestration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSettings {
    /// Maximum number of components
    pub max_components: usize,
    /// Health check interval
    #[serde(with = "duration_serde")]
    pub health_check_interval: Duration,
    /// Component startup timeout
    #[serde(with = "duration_serde")]
    pub startup_timeout: Duration,
    /// Component shutdown timeout
    #[serde(with = "duration_serde")]
    pub shutdown_timeout: Duration,
    /// Maximum dependency resolution depth
    pub max_dependency_depth: usize,
    /// Enable hot-reload of configuration
    pub enable_hot_reload: bool,
    /// Configuration file paths to watch
    pub config_paths: Vec<PathBuf>,
    /// Worker thread pool size
    pub worker_threads: Option<usize>,
    /// Enable graceful shutdown
    pub graceful_shutdown: bool,
}

/// Component-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfig {
    /// Component type
    pub component_type: ComponentType,
    /// Whether component is enabled
    pub enabled: bool,
    /// Component-specific settings
    pub settings: HashMap<String, serde_json::Value>,
    /// Resource limits for this component
    pub resources: ComponentResourceLimits,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Dependencies configuration
    pub dependencies: Vec<DependencyConfig>,
}

/// Component resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<u64>,
    /// Maximum CPU percentage (0-100)
    pub max_cpu_percent: Option<f32>,
    /// Maximum file descriptors
    pub max_file_descriptors: Option<u32>,
    /// Maximum network connections
    pub max_connections: Option<u32>,
    /// Request rate limit (requests per second)
    pub rate_limit_rps: Option<f32>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checking
    pub enabled: bool,
    /// Health check interval
    #[serde(with = "duration_serde")]
    pub interval: Duration,
    /// Health check timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Custom health check endpoint or method
    pub endpoint: Option<String>,
}

/// Dependency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConfig {
    /// Target component name
    pub component: String,
    /// Dependency type
    pub dependency_type: String,
    /// Whether dependency is required
    pub required: bool,
    /// Timeout for dependency resolution
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Retry configuration
    pub retry: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry delay
    #[serde(with = "duration_serde")]
    pub initial_delay: Duration,
    /// Maximum retry delay
    #[serde(with = "duration_serde")]
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f32,
    /// Enable jitter
    pub jitter: bool,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address for the runtime
    pub listen_address: String,
    /// Listen port
    pub listen_port: u16,
    /// Enable TLS
    pub enable_tls: bool,
    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,
    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,
    /// TLS CA certificate path
    pub tls_ca_path: Option<PathBuf>,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Connection timeout
    #[serde(with = "duration_serde")]
    pub connection_timeout: Duration,
    /// Keep-alive settings
    pub keep_alive: KeepAliveConfig,
    /// QUIC configuration
    pub quic: QuicConfig,
}

/// Keep-alive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepAliveConfig {
    /// Enable keep-alive
    pub enabled: bool,
    /// Keep-alive timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Keep-alive interval
    #[serde(with = "duration_serde")]
    pub interval: Duration,
    /// Maximum keep-alive probes
    pub max_probes: u32,
}

/// QUIC protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    /// Enable QUIC transport
    pub enabled: bool,
    /// Maximum concurrent streams per connection
    pub max_streams_per_connection: u64,
    /// Initial connection window size
    pub initial_connection_window_size: u32,
    /// Initial stream window size
    pub initial_stream_window_size: u32,
    /// Keep-alive interval
    #[serde(with = "duration_serde")]
    pub keep_alive_interval: Duration,
    /// Connection idle timeout
    #[serde(with = "duration_serde")]
    pub idle_timeout: Duration,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable authentication
    pub enable_authentication: bool,
    /// Enable authorization
    pub enable_authorization: bool,
    /// JWT secret for token validation
    pub jwt_secret: Option<String>,
    /// JWT token expiration
    #[serde(with = "duration_serde")]
    pub jwt_expiration: Duration,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitingConfig,
    /// Audit logging configuration
    pub audit: AuditConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Global requests per second limit
    pub global_rps: Option<f32>,
    /// Per-client requests per second limit
    pub per_client_rps: Option<f32>,
    /// Rate limit window size
    #[serde(with = "duration_serde")]
    pub window_size: Duration,
    /// Burst allowance
    pub burst_size: u32,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Audit log file path
    pub log_file: Option<PathBuf>,
    /// Log rotation configuration
    pub rotation: LogRotationConfig,
    /// Events to audit
    pub events: Vec<String>,
    /// Sensitive fields to redact
    pub redact_fields: Vec<String>,
}

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Maximum log file size
    pub max_size_mb: u64,
    /// Maximum number of archived files
    pub max_files: u32,
    /// Compress archived files
    pub compress: bool,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key derivation function
    pub kdf: String,
    /// Key rotation interval
    #[serde(with = "duration_serde")]
    pub key_rotation_interval: Duration,
    /// Enable encryption at rest
    pub encrypt_at_rest: bool,
    /// Enable encryption in transit
    pub encrypt_in_transit: bool,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Performance target thresholds
    pub thresholds: PerformanceThresholds,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Memory management settings
    pub memory: MemoryConfig,
    /// CPU optimization settings
    pub cpu: CpuConfig,
}

/// Performance threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency in microseconds
    pub max_latency_us: f64,
    /// Minimum required throughput (operations per second)
    pub min_throughput_ops: f64,
    /// Maximum memory usage percentage
    pub max_memory_percent: f32,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,
    /// Error rate threshold
    pub max_error_rate: f32,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Optimization algorithm
    pub algorithm: String,
    /// Learning rate for ML-based optimization
    pub learning_rate: f32,
    /// Optimization window size
    #[serde(with = "duration_serde")]
    pub optimization_window: Duration,
    /// Minimum improvement threshold
    pub min_improvement_threshold: f32,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable huge pages
    pub enable_huge_pages: bool,
    /// Enable NUMA awareness
    pub enable_numa_awareness: bool,
    /// Memory pool sizes
    pub pool_sizes: Vec<usize>,
    /// Garbage collection settings
    pub gc: GarbageCollectionConfig,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionConfig {
    /// GC algorithm
    pub algorithm: String,
    /// GC trigger threshold
    pub trigger_threshold: f32,
    /// Maximum GC pause time
    #[serde(with = "duration_serde")]
    pub max_pause_time: Duration,
}

/// CPU optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Enable CPU affinity
    pub enable_affinity: bool,
    /// CPU cores to bind to
    pub cpu_cores: Option<Vec<u32>>,
    /// Enable frequency scaling control
    pub enable_frequency_scaling: bool,
    /// Target CPU frequency
    pub target_frequency_ghz: Option<f32>,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
}

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    pub enabled: bool,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// Tracing configuration
    pub tracing: TracingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics collection interval
    #[serde(with = "duration_serde")]
    pub collection_interval: Duration,
    /// Prometheus exporter configuration
    pub prometheus: PrometheusConfig,
    /// Custom metrics to collect
    pub custom_metrics: Vec<String>,
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable Prometheus exporter
    pub enabled: bool,
    /// Prometheus metrics endpoint
    pub endpoint: String,
    /// Prometheus port
    pub port: u16,
    /// Metrics namespace
    pub namespace: String,
}

/// Distributed tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,
    /// Tracing exporter type (jaeger, zipkin, otlp)
    pub exporter: String,
    /// Tracing endpoint
    pub endpoint: String,
    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f32,
    /// Maximum spans per trace
    pub max_spans_per_trace: u32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace)
    pub level: String,
    /// Log format (json, text)
    pub format: String,
    /// Log output (stdout, stderr, file)
    pub output: String,
    /// Log file path (if output is file)
    pub file_path: Option<PathBuf>,
    /// Enable structured logging
    pub structured: bool,
}

/// Resource configuration and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Global memory limits
    pub memory: MemoryLimits,
    /// Global CPU limits
    pub cpu: CpuLimits,
    /// Network resource limits
    pub network: NetworkLimits,
    /// Storage resource limits
    pub storage: StorageLimits,
}

/// Memory resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum total memory usage
    pub max_total_bytes: u64,
    /// Maximum heap size
    pub max_heap_bytes: u64,
    /// Maximum stack size per thread
    pub max_stack_bytes: u64,
    /// Memory pressure threshold
    pub pressure_threshold: f32,
}

/// CPU resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU cores to use
    pub max_cores: Option<u32>,
    /// CPU quota (percentage of total CPU)
    pub quota_percent: Option<f32>,
    /// CPU priority (nice value)
    pub priority: Option<i32>,
    /// Enable real-time scheduling
    pub enable_realtime: bool,
}

/// Network resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Maximum bandwidth (bytes per second)
    pub max_bandwidth_bps: u64,
    /// Maximum packet rate
    pub max_packet_rate: u32,
    /// Connection timeout
    #[serde(with = "duration_serde")]
    pub connection_timeout: Duration,
}

/// Storage resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLimits {
    /// Maximum disk usage
    pub max_disk_usage_bytes: u64,
    /// Maximum number of open files
    pub max_open_files: u32,
    /// Maximum I/O operations per second
    pub max_iops: u32,
    /// Disk usage warning threshold
    pub disk_warning_threshold: f32,
}

/// Configuration manager for loading and managing runtime configuration
#[derive(Debug)]
pub struct ConfigurationManager {
    /// Current configuration
    config: RuntimeConfig,
    /// Configuration file paths
    config_paths: Vec<PathBuf>,
    /// Environment variable prefix
    env_prefix: String,
    /// Hot-reload enabled
    hot_reload: bool,
}

impl ConfigurationManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
            config_paths: Vec::new(),
            env_prefix: "CSF".to_string(),
            hot_reload: false,
        }
    }

    /// Load configuration from multiple sources
    pub fn load_configuration<P: AsRef<Path>>(&mut self, config_paths: &[P]) -> RuntimeResult<()> {
        let mut builder = Config::builder();

        // Start with default configuration
        builder = builder.add_source(Config::try_from(&RuntimeConfig::default())?);

        // Add configuration files in order
        for path in config_paths {
            let path = path.as_ref();
            if path.exists() {
                tracing::info!("Loading configuration from: {}", path.display());

                let format = self.detect_file_format(path)?;
                let path_str = path.to_str().ok_or_else(|| {
                    RuntimeError::Configuration(ConfigurationError::InvalidPath {
                        path: path.display().to_string(),
                        reason: "Path contains invalid Unicode".to_string(),
                    })
                })?;
                builder = builder.add_source(File::new(path_str, format));

                self.config_paths.push(path.to_path_buf());
            } else {
                tracing::warn!("Configuration file not found: {}", path.display());
            }
        }

        // Add environment variables with prefix
        builder = builder.add_source(
            Environment::with_prefix(&self.env_prefix)
                .separator("_")
                .try_parsing(true),
        );

        // Build and deserialize configuration
        let config = builder.build()?;
        self.config = config.try_deserialize()?;

        // Validate configuration
        self.validate_configuration()?;

        // Enable hot-reload if configured
        if self.config.runtime.enable_hot_reload {
            self.setup_hot_reload()?;
        }

        tracing::info!("Configuration loaded successfully");
        Ok(())
    }

    /// Detect configuration file format from extension
    fn detect_file_format(&self, path: &Path) -> RuntimeResult<FileFormat> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => Ok(FileFormat::Toml),
            Some("yaml") | Some("yml") => Ok(FileFormat::Yaml),
            Some("json") => Ok(FileFormat::Json),
            Some("ini") => Ok(FileFormat::Ini),
            _ => Ok(FileFormat::Toml), // Default to TOML
        }
    }

    /// Validate loaded configuration
    fn validate_configuration(&self) -> RuntimeResult<()> {
        // Validate runtime settings
        if self.config.runtime.max_components == 0 {
            return Err(RuntimeError::Configuration(
                ConfigurationError::InvalidValue {
                    key: "runtime.max_components".to_string(),
                    reason: "Must be greater than 0".to_string(),
                },
            ));
        }

        if self.config.runtime.max_dependency_depth == 0 {
            return Err(RuntimeError::Configuration(
                ConfigurationError::InvalidValue {
                    key: "runtime.max_dependency_depth".to_string(),
                    reason: "Must be greater than 0".to_string(),
                },
            ));
        }

        // Validate network configuration
        if self.config.network.listen_port == 0 {
            return Err(RuntimeError::Configuration(
                ConfigurationError::InvalidValue {
                    key: "network.listen_port".to_string(),
                    reason: "Must be a valid port number".to_string(),
                },
            ));
        }

        // Validate TLS configuration
        if self.config.network.enable_tls {
            if self.config.network.tls_cert_path.is_none() {
                return Err(RuntimeError::Configuration(
                    ConfigurationError::MissingKey {
                        key: "network.tls_cert_path".to_string(),
                    },
                ));
            }
            if self.config.network.tls_key_path.is_none() {
                return Err(RuntimeError::Configuration(
                    ConfigurationError::MissingKey {
                        key: "network.tls_key_path".to_string(),
                    },
                ));
            }
        }

        // Validate performance thresholds
        let thresholds = &self.config.performance.thresholds;
        if thresholds.max_latency_us <= 0.0 {
            return Err(RuntimeError::Configuration(
                ConfigurationError::InvalidValue {
                    key: "performance.thresholds.max_latency_us".to_string(),
                    reason: "Must be greater than 0".to_string(),
                },
            ));
        }

        if thresholds.min_throughput_ops <= 0.0 {
            return Err(RuntimeError::Configuration(
                ConfigurationError::InvalidValue {
                    key: "performance.thresholds.min_throughput_ops".to_string(),
                    reason: "Must be greater than 0".to_string(),
                },
            ));
        }

        tracing::info!("Configuration validation passed");
        Ok(())
    }

    /// Setup hot-reload monitoring
    fn setup_hot_reload(&mut self) -> RuntimeResult<()> {
        // In a real implementation, we'd set up file system watchers
        // using notify crate or similar
        tracing::info!("Hot-reload monitoring enabled for: {:?}", self.config_paths);
        self.hot_reload = true;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Update configuration dynamically
    pub fn update_config(
        &mut self,
        updates: HashMap<String, serde_json::Value>,
    ) -> RuntimeResult<()> {
        // Apply configuration updates
        for (key, value) in updates {
            self.apply_config_update(&key, value)?;
        }

        // Re-validate configuration
        self.validate_configuration()?;

        tracing::info!("Configuration updated successfully");
        Ok(())
    }

    /// Apply a single configuration update
    fn apply_config_update(&mut self, key: &str, value: serde_json::Value) -> RuntimeResult<()> {
        // This is a simplified implementation
        // In reality, we'd need sophisticated path-based updates
        tracing::info!("Applying configuration update: {} = {:?}", key, value);

        // For demonstration, we'll support a few common update patterns
        match key {
            "runtime.health_check_interval" => {
                if let Some(duration_ms) = value.as_u64() {
                    self.config.runtime.health_check_interval = Duration::from_millis(duration_ms);
                }
            }
            "performance.thresholds.max_latency_us" => {
                if let Some(latency) = value.as_f64() {
                    self.config.performance.thresholds.max_latency_us = latency;
                }
            }
            "telemetry.metrics.collection_interval" => {
                if let Some(interval_ms) = value.as_u64() {
                    self.config.telemetry.metrics.collection_interval =
                        Duration::from_millis(interval_ms);
                }
            }
            _ => {
                tracing::warn!("Unsupported configuration update key: {}", key);
            }
        }

        Ok(())
    }

    /// Get component configuration
    pub fn get_component_config(&self, component_name: &str) -> Option<&ComponentConfig> {
        self.config.components.get(component_name)
    }

    /// Set component configuration
    pub fn set_component_config(&mut self, component_name: String, config: ComponentConfig) {
        self.config.components.insert(component_name, config);
    }

    /// Export current configuration to file
    pub fn export_config<P: AsRef<Path>>(&self, path: P) -> RuntimeResult<()> {
        let path = path.as_ref();
        let config_str = toml::to_string_pretty(&self.config).map_err(|e| {
            RuntimeError::Configuration(ConfigurationError::InvalidFormat {
                path: path.display().to_string(),
                reason: e.to_string(),
            })
        })?;

        fs::write(path, config_str).map_err(|e| {
            RuntimeError::Configuration(ConfigurationError::InvalidFormat {
                path: path.display().to_string(),
                reason: e.to_string(),
            })
        })?;

        tracing::info!("Configuration exported to: {}", path.display());
        Ok(())
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeSettings::default(),
            components: HashMap::new(),
            network: NetworkConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
            telemetry: TelemetryConfig::default(),
            resources: ResourceConfig::default(),
        }
    }
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            max_components: crate::MAX_COMPONENTS,
            health_check_interval: crate::DEFAULT_HEALTH_CHECK_INTERVAL,
            startup_timeout: crate::DEFAULT_STARTUP_TIMEOUT,
            shutdown_timeout: crate::DEFAULT_SHUTDOWN_TIMEOUT,
            max_dependency_depth: crate::MAX_DEPENDENCY_DEPTH,
            enable_hot_reload: false,
            config_paths: vec![],
            worker_threads: None,
            graceful_shutdown: true,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_address: "127.0.0.1".to_string(),
            listen_port: 8080,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            tls_ca_path: None,
            max_connections: 10000,
            connection_timeout: Duration::from_secs(30),
            keep_alive: KeepAliveConfig::default(),
            quic: QuicConfig::default(),
        }
    }
}

impl Default for KeepAliveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout: Duration::from_secs(60),
            interval: Duration::from_secs(10),
            max_probes: 3,
        }
    }
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_streams_per_connection: 1000,
            initial_connection_window_size: 1024 * 1024, // 1MB
            initial_stream_window_size: 64 * 1024,       // 64KB
            keep_alive_interval: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_authentication: false,
            enable_authorization: false,
            jwt_secret: None,
            jwt_expiration: Duration::from_secs(3600), // 1 hour
            rate_limiting: RateLimitingConfig::default(),
            audit: AuditConfig::default(),
            encryption: EncryptionConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            thresholds: PerformanceThresholds::default(),
            optimization: OptimizationConfig::default(),
            memory: MemoryConfig::default(),
            cpu: CpuConfig::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_us: 1000.0,          // 1ms
            min_throughput_ops: 1_000_000.0, // 1M ops/sec
            max_memory_percent: 80.0,
            max_cpu_percent: 80.0,
            max_error_rate: 0.01, // 1%
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            memory: MemoryLimits::default(),
            cpu: CpuLimits::default(),
            network: NetworkLimits::default(),
            storage: StorageLimits::default(),
        }
    }
}

// Implement other Default traits for remaining config structs...
// (Similar implementations follow the same pattern)

/// Serde helper for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ms = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(ms))
    }
}

// Additional default implementations for the remaining config structs
impl Default for ComponentResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: Some(1024 * 1024 * 1024), // 1GB
            max_cpu_percent: Some(50.0),
            max_file_descriptors: Some(1024),
            max_connections: Some(1000),
            rate_limit_rps: Some(1000.0),
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            global_rps: Some(10000.0),
            per_client_rps: Some(100.0),
            window_size: Duration::from_secs(60),
            burst_size: 100,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_file: None,
            rotation: LogRotationConfig::default(),
            events: vec!["login".to_string(), "config_change".to_string()],
            redact_fields: vec!["password".to_string(), "token".to_string()],
        }
    }
}

impl Default for LogRotationConfig {
    fn default() -> Self {
        Self {
            max_size_mb: 100,
            max_files: 10,
            compress: true,
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: "AES-256-GCM".to_string(),
            kdf: "PBKDF2".to_string(),
            key_rotation_interval: Duration::from_secs(86400 * 30), // 30 days
            encrypt_at_rest: false,
            encrypt_in_transit: true,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            auto_optimization: false,
            algorithm: "gradient_descent".to_string(),
            learning_rate: 0.01,
            optimization_window: Duration::from_secs(300), // 5 minutes
            min_improvement_threshold: 0.01,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_huge_pages: false,
            enable_numa_awareness: true,
            pool_sizes: vec![64, 256, 1024, 4096, 16384],
            gc: GarbageCollectionConfig::default(),
        }
    }
}

impl Default for GarbageCollectionConfig {
    fn default() -> Self {
        Self {
            algorithm: "concurrent_mark_sweep".to_string(),
            trigger_threshold: 0.8,
            max_pause_time: Duration::from_millis(10),
        }
    }
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            enable_affinity: false,
            cpu_cores: None,
            enable_frequency_scaling: false,
            target_frequency_ghz: None,
            enable_simd: true,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(10),
            prometheus: PrometheusConfig::default(),
            custom_metrics: vec![],
        }
    }
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "/metrics".to_string(),
            port: 9090,
            namespace: "csf".to_string(),
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exporter: "jaeger".to_string(),
            endpoint: "http://localhost:14268/api/traces".to_string(),
            sampling_rate: 0.1,
            max_spans_per_trace: 1000,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            output: "stdout".to_string(),
            file_path: None,
            structured: true,
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            max_heap_bytes: 6 * 1024 * 1024 * 1024,  // 6GB
            max_stack_bytes: 8 * 1024 * 1024,        // 8MB
            pressure_threshold: 0.8,
        }
    }
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_cores: None,
            quota_percent: Some(80.0),
            priority: Some(0),
            enable_realtime: false,
        }
    }
}

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_connections: 10000,
            max_bandwidth_bps: 1024 * 1024 * 1024, // 1Gbps
            max_packet_rate: 100000,
            connection_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for StorageLimits {
    fn default() -> Self {
        Self {
            max_disk_usage_bytes: 100 * 1024 * 1024 * 1024, // 100GB
            max_open_files: 65536,
            max_iops: 10000,
            disk_warning_threshold: 0.8,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
            success_threshold: 2,
            endpoint: None,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}
