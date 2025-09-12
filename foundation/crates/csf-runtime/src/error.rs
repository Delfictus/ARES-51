//! Error types for the CSF Runtime system
//!
//! This module defines a comprehensive error hierarchy for all runtime operations,
//! providing detailed error information for debugging and operational monitoring.

use crate::{ComponentId, PortId};
use std::fmt;
use thiserror::Error;

/// Result type for runtime operations
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Comprehensive error type for CSF Runtime operations
#[derive(Error, Debug)]
pub enum RuntimeError {
    /// Component-related errors
    #[error("Component error: {0}")]
    Component(#[from] ComponentError),

    /// Port and adapter registry errors
    #[error("Registry error: {0}")]
    Registry(#[from] RegistryError),

    /// Dependency resolution errors
    #[error("Dependency error: {0}")]
    Dependency(#[from] DependencyError),

    /// Lifecycle management errors
    #[error("Lifecycle error: {0}")]
    Lifecycle(#[from] LifecycleError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),

    /// Config library errors
    #[error("Config error: {0}")]
    Config(#[from] config::ConfigError),

    /// Health monitoring errors
    #[error("Health error: {0}")]
    Health(#[from] HealthError),

    /// Performance orchestration errors
    #[error("Performance error: {0}")]
    Performance(#[from] PerformanceError),

    /// System-level errors
    #[error("System error: {0}")]
    System(#[from] SystemError),
}

/// Component-specific errors
#[derive(Error, Debug, Clone)]
pub enum ComponentError {
    /// Component not found
    #[error("Component '{id}' not found")]
    NotFound { id: ComponentId },

    /// Component already exists
    #[error("Component '{id}' already exists")]
    AlreadyExists { id: ComponentId },

    /// Component initialization failed
    #[error("Failed to initialize component '{id}': {reason}")]
    InitializationFailed { id: ComponentId, reason: String },

    /// Component startup failed
    #[error("Failed to start component '{id}': {reason}")]
    StartupFailed { id: ComponentId, reason: String },

    /// Component shutdown failed
    #[error("Failed to shutdown component '{id}': {reason}")]
    ShutdownFailed { id: ComponentId, reason: String },

    /// Component in invalid state
    #[error("Component '{id}' is in invalid state: {state}")]
    InvalidState { id: ComponentId, state: String },

    /// Component timeout
    #[error("Component '{id}' operation timed out after {timeout_ms}ms")]
    Timeout { id: ComponentId, timeout_ms: u64 },
}

/// Registry-specific errors
#[derive(Error, Debug, Clone)]
pub enum RegistryError {
    /// Port not found
    #[error("Port '{port_id}' not found")]
    PortNotFound { port_id: PortId },

    /// Port already bound
    #[error("Port '{port_id}' is already bound to adapter '{adapter_id}'")]
    PortAlreadyBound { port_id: PortId, adapter_id: String },

    /// Multiple adapters for single port (violates hexagonal architecture)
    #[error("Multiple adapters detected for port '{port_id}': {adapters:?}")]
    MultipleAdapters {
        port_id: PortId,
        adapters: Vec<String>,
    },

    /// Adapter not found
    #[error("Adapter '{adapter_id}' not found")]
    AdapterNotFound { adapter_id: String },

    /// Adapter validation failed
    #[error("Adapter '{adapter_id}' validation failed: {reason}")]
    AdapterValidationFailed { adapter_id: String, reason: String },

    /// Port type mismatch
    #[error("Port '{port_id}' type mismatch: expected '{expected}', found '{actual}'")]
    PortTypeMismatch {
        port_id: PortId,
        expected: String,
        actual: String,
    },
}

/// Dependency resolution errors
#[derive(Error, Debug, Clone)]
pub enum DependencyError {
    /// Circular dependency detected
    #[error("Circular dependency detected in chain: {chain:?}")]
    CircularDependency { chain: Vec<ComponentId> },

    /// Unresolvable dependency
    #[error("Cannot resolve dependency '{dependency}' for component '{component}'")]
    UnresolvableDependency {
        component: ComponentId,
        dependency: ComponentId,
    },

    /// Missing required dependency
    #[error("Component '{component}' requires missing dependency '{dependency}'")]
    MissingDependency {
        component: ComponentId,
        dependency: ComponentId,
    },

    /// Dependency depth exceeded
    #[error("Dependency resolution depth exceeded maximum of {max_depth}")]
    DepthExceeded { max_depth: usize },

    /// Conflicting dependencies
    #[error("Conflicting dependencies detected: {conflicts:?}")]
    ConflictingDependencies { conflicts: Vec<String> },
}

/// Lifecycle management errors
#[derive(Error, Debug, Clone)]
pub enum LifecycleError {
    /// Invalid lifecycle transition
    #[error("Invalid lifecycle transition from '{from}' to '{to}' for component '{component}'")]
    InvalidTransition {
        component: ComponentId,
        from: String,
        to: String,
    },

    /// Startup sequence failed
    #[error("Startup sequence failed at step {step}: {reason}")]
    StartupSequenceFailed { step: usize, reason: String },

    /// Shutdown sequence failed
    #[error("Shutdown sequence failed at step {step}: {reason}")]
    ShutdownSequenceFailed { step: usize, reason: String },

    /// Component stuck in transitional state
    #[error("Component '{component}' stuck in transitional state '{state}' for {duration_ms}ms")]
    StuckInTransition {
        component: ComponentId,
        state: String,
        duration_ms: u64,
    },

    /// Graceful shutdown timeout
    #[error("Graceful shutdown timed out after {timeout_ms}ms")]
    GracefulShutdownTimeout { timeout_ms: u64 },
}

/// Configuration errors
#[derive(Error, Debug, Clone)]
pub enum ConfigurationError {
    /// Configuration file not found
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    /// Invalid configuration format
    #[error("Invalid configuration format in {path}: {reason}")]
    InvalidFormat { path: String, reason: String },

    /// Missing required configuration
    #[error("Missing required configuration key: {key}")]
    MissingKey { key: String },

    /// Invalid configuration value
    #[error("Invalid configuration value for key '{key}': {reason}")]
    InvalidValue { key: String, reason: String },

    /// Invalid file path
    #[error("Invalid configuration file path '{path}': {reason}")]
    InvalidPath { path: String, reason: String },

    /// Configuration validation failed
    #[error("Configuration validation failed: {reason}")]
    ValidationFailed { reason: String },

    /// Environment variable not found
    #[error("Required environment variable not found: {var}")]
    EnvironmentVariableNotFound { var: String },
}

/// Health monitoring errors
#[derive(Error, Debug, Clone)]
pub enum HealthError {
    /// Health check failed
    #[error("Health check failed for component '{component}': {reason}")]
    HealthCheckFailed {
        component: ComponentId,
        reason: String,
    },

    /// Health monitor not responding
    #[error("Health monitor for component '{component}' not responding")]
    MonitorNotResponding { component: ComponentId },

    /// Health metrics collection failed
    #[error("Failed to collect health metrics: {reason}")]
    MetricsCollectionFailed { reason: String },

    /// Health threshold exceeded
    #[error("Health threshold exceeded for metric '{metric}': {value} > {threshold}")]
    ThresholdExceeded {
        metric: String,
        value: f64,
        threshold: f64,
    },
}

/// Performance orchestration errors
#[derive(Error, Debug, Clone)]
pub enum PerformanceError {
    /// Performance target not met
    #[error("Performance target not met for '{metric}': {actual} < {target}")]
    TargetNotMet {
        metric: String,
        actual: f64,
        target: f64,
    },

    /// Optimization failed
    #[error("Performance optimization failed: {reason}")]
    OptimizationFailed { reason: String },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    /// Performance regression detected
    #[error("Performance regression detected in '{metric}': {current} vs {baseline}")]
    RegressionDetected {
        metric: String,
        current: f64,
        baseline: f64,
    },

    /// Performance monitoring failed
    #[error("Performance monitoring failed: {reason}")]
    MonitoringFailed { reason: String },
}

/// System-level errors
#[derive(Error, Debug, Clone)]
pub enum SystemError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Timeout error
    #[error("Operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} ({current}/{limit})")]
    ResourceLimitExceeded {
        resource: String,
        current: usize,
        limit: usize,
    },

    /// Internal error
    #[error("Internal error: {reason}")]
    Internal { reason: String },

    /// External service error
    #[error("External service error: {service} - {reason}")]
    ExternalService { service: String, reason: String },
}

impl RuntimeError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            RuntimeError::Component(err) => err.is_recoverable(),
            RuntimeError::Registry(err) => err.is_recoverable(),
            RuntimeError::Dependency(err) => err.is_recoverable(),
            RuntimeError::Lifecycle(err) => err.is_recoverable(),
            RuntimeError::Configuration(err) => err.is_recoverable(),
            RuntimeError::Health(err) => err.is_recoverable(),
            RuntimeError::Performance(err) => err.is_recoverable(),
            RuntimeError::System(err) => err.is_recoverable(),
            RuntimeError::Config(_) => false, // Config errors are typically not recoverable
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            RuntimeError::Component(err) => err.severity(),
            RuntimeError::Registry(err) => err.severity(),
            RuntimeError::Dependency(err) => err.severity(),
            RuntimeError::Lifecycle(err) => err.severity(),
            RuntimeError::Configuration(err) => err.severity(),
            RuntimeError::Health(err) => err.severity(),
            RuntimeError::Performance(err) => err.severity(),
            RuntimeError::System(err) => err.severity(),
            RuntimeError::Config(_) => ErrorSeverity::Error, // Config errors are high severity
        }
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            RuntimeError::Component(_) => "component",
            RuntimeError::Registry(_) => "registry",
            RuntimeError::Dependency(_) => "dependency",
            RuntimeError::Lifecycle(_) => "lifecycle",
            RuntimeError::Configuration(_) => "configuration",
            RuntimeError::Health(_) => "health",
            RuntimeError::Performance(_) => "performance",
            RuntimeError::System(_) => "system",
            RuntimeError::Config(_) => "config",
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - informational
    Info,
    /// Medium severity - warning
    Warning,
    /// High severity - error
    Error,
    /// Critical severity - system failure
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARN"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// Trait implementations for error categories
impl ComponentError {
    fn is_recoverable(&self) -> bool {
        match self {
            ComponentError::NotFound { .. } => false,
            ComponentError::AlreadyExists { .. } => false,
            ComponentError::InitializationFailed { .. } => false,
            ComponentError::StartupFailed { .. } => true,
            ComponentError::ShutdownFailed { .. } => true,
            ComponentError::InvalidState { .. } => true,
            ComponentError::Timeout { .. } => true,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            ComponentError::NotFound { .. } => ErrorSeverity::Error,
            ComponentError::AlreadyExists { .. } => ErrorSeverity::Warning,
            ComponentError::InitializationFailed { .. } => ErrorSeverity::Critical,
            ComponentError::StartupFailed { .. } => ErrorSeverity::Error,
            ComponentError::ShutdownFailed { .. } => ErrorSeverity::Warning,
            ComponentError::InvalidState { .. } => ErrorSeverity::Error,
            ComponentError::Timeout { .. } => ErrorSeverity::Warning,
        }
    }
}

impl RegistryError {
    fn is_recoverable(&self) -> bool {
        match self {
            RegistryError::PortNotFound { .. } => false,
            RegistryError::PortAlreadyBound { .. } => false,
            RegistryError::MultipleAdapters { .. } => false,
            RegistryError::AdapterNotFound { .. } => false,
            RegistryError::AdapterValidationFailed { .. } => false,
            RegistryError::PortTypeMismatch { .. } => false,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            RegistryError::PortNotFound { .. } => ErrorSeverity::Error,
            RegistryError::PortAlreadyBound { .. } => ErrorSeverity::Critical,
            RegistryError::MultipleAdapters { .. } => ErrorSeverity::Critical,
            RegistryError::AdapterNotFound { .. } => ErrorSeverity::Error,
            RegistryError::AdapterValidationFailed { .. } => ErrorSeverity::Error,
            RegistryError::PortTypeMismatch { .. } => ErrorSeverity::Critical,
        }
    }
}

impl DependencyError {
    fn is_recoverable(&self) -> bool {
        match self {
            DependencyError::CircularDependency { .. } => false,
            DependencyError::UnresolvableDependency { .. } => false,
            DependencyError::MissingDependency { .. } => false,
            DependencyError::DepthExceeded { .. } => false,
            DependencyError::ConflictingDependencies { .. } => false,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            DependencyError::CircularDependency { .. } => ErrorSeverity::Critical,
            DependencyError::UnresolvableDependency { .. } => ErrorSeverity::Critical,
            DependencyError::MissingDependency { .. } => ErrorSeverity::Error,
            DependencyError::DepthExceeded { .. } => ErrorSeverity::Error,
            DependencyError::ConflictingDependencies { .. } => ErrorSeverity::Critical,
        }
    }
}

impl LifecycleError {
    fn is_recoverable(&self) -> bool {
        match self {
            LifecycleError::InvalidTransition { .. } => false,
            LifecycleError::StartupSequenceFailed { .. } => true,
            LifecycleError::ShutdownSequenceFailed { .. } => true,
            LifecycleError::StuckInTransition { .. } => true,
            LifecycleError::GracefulShutdownTimeout { .. } => true,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            LifecycleError::InvalidTransition { .. } => ErrorSeverity::Critical,
            LifecycleError::StartupSequenceFailed { .. } => ErrorSeverity::Error,
            LifecycleError::ShutdownSequenceFailed { .. } => ErrorSeverity::Warning,
            LifecycleError::StuckInTransition { .. } => ErrorSeverity::Error,
            LifecycleError::GracefulShutdownTimeout { .. } => ErrorSeverity::Warning,
        }
    }
}

impl ConfigurationError {
    fn is_recoverable(&self) -> bool {
        match self {
            ConfigurationError::FileNotFound { .. } => false,
            ConfigurationError::InvalidFormat { .. } => false,
            ConfigurationError::MissingKey { .. } => false,
            ConfigurationError::InvalidValue { .. } => false,
            ConfigurationError::InvalidPath { .. } => false,
            ConfigurationError::ValidationFailed { .. } => false,
            ConfigurationError::EnvironmentVariableNotFound { .. } => false,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            ConfigurationError::FileNotFound { .. } => ErrorSeverity::Critical,
            ConfigurationError::InvalidFormat { .. } => ErrorSeverity::Critical,
            ConfigurationError::MissingKey { .. } => ErrorSeverity::Error,
            ConfigurationError::InvalidValue { .. } => ErrorSeverity::Error,
            ConfigurationError::InvalidPath { .. } => ErrorSeverity::Critical,
            ConfigurationError::ValidationFailed { .. } => ErrorSeverity::Error,
            ConfigurationError::EnvironmentVariableNotFound { .. } => ErrorSeverity::Warning,
        }
    }
}

impl HealthError {
    fn is_recoverable(&self) -> bool {
        match self {
            HealthError::HealthCheckFailed { .. } => true,
            HealthError::MonitorNotResponding { .. } => true,
            HealthError::MetricsCollectionFailed { .. } => true,
            HealthError::ThresholdExceeded { .. } => true,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            HealthError::HealthCheckFailed { .. } => ErrorSeverity::Warning,
            HealthError::MonitorNotResponding { .. } => ErrorSeverity::Error,
            HealthError::MetricsCollectionFailed { .. } => ErrorSeverity::Warning,
            HealthError::ThresholdExceeded { .. } => ErrorSeverity::Warning,
        }
    }
}

impl PerformanceError {
    fn is_recoverable(&self) -> bool {
        match self {
            PerformanceError::TargetNotMet { .. } => true,
            PerformanceError::OptimizationFailed { .. } => true,
            PerformanceError::ResourceExhausted { .. } => true,
            PerformanceError::RegressionDetected { .. } => true,
            PerformanceError::MonitoringFailed { .. } => true,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            PerformanceError::TargetNotMet { .. } => ErrorSeverity::Warning,
            PerformanceError::OptimizationFailed { .. } => ErrorSeverity::Warning,
            PerformanceError::ResourceExhausted { .. } => ErrorSeverity::Error,
            PerformanceError::RegressionDetected { .. } => ErrorSeverity::Warning,
            PerformanceError::MonitoringFailed { .. } => ErrorSeverity::Warning,
        }
    }
}

impl SystemError {
    fn is_recoverable(&self) -> bool {
        match self {
            SystemError::Io(_) => true,
            SystemError::Network(_) => true,
            SystemError::Timeout { .. } => true,
            SystemError::ResourceLimitExceeded { .. } => true,
            SystemError::Internal { .. } => false,
            SystemError::ExternalService { .. } => true,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            SystemError::Io(_) => ErrorSeverity::Error,
            SystemError::Network(_) => ErrorSeverity::Error,
            SystemError::Timeout { .. } => ErrorSeverity::Warning,
            SystemError::ResourceLimitExceeded { .. } => ErrorSeverity::Error,
            SystemError::Internal { .. } => ErrorSeverity::Critical,
            SystemError::ExternalService { .. } => ErrorSeverity::Warning,
        }
    }
}
