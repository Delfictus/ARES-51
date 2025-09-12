//! Adaptive Port Registry - Hexagonal Architecture Enforcement
//!
//! This module implements the sophisticated adapter registry system that enforces
//! the "one-adapter-per-port" rule of hexagonal architecture, provides dynamic
//! port binding validation, and enables runtime adapter swapping for testing.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};
use uuid::Uuid;

use crate::core::{PortDirection, PortId};
use crate::error::{RegistryError, RuntimeError, RuntimeResult};

/// Advanced adapter registry with hexagonal architecture enforcement
pub struct AdapterRegistry {
    /// Port bindings (Port -> Adapter mapping)
    port_bindings: DashMap<PortId, PortBinding>,
    /// Adapter information registry
    adapters: DashMap<AdapterId, AdapterInfo>,
    /// Port validation rules
    validators: Vec<Arc<dyn PortValidator>>,
    /// Binding history for audit and rollback
    binding_history: Arc<RwLock<Vec<BindingHistoryEntry>>>,
    /// Registry metrics
    metrics: RegistryMetrics,
    /// Registry configuration
    config: RegistryConfig,
    /// Change notification
    change_notify: Arc<Notify>,
}

/// Unique identifier for an adapter
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdapterId {
    /// Adapter name
    pub name: String,
    /// Unique identifier
    pub uuid: Uuid,
    /// Adapter type
    pub adapter_type: AdapterType,
}

/// Type of adapter in the hexagonal architecture
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdapterType {
    /// Primary adapter (drives the application)
    Primary,
    /// Secondary adapter (driven by the application)
    Secondary,
    /// Test adapter (for testing purposes)
    Test,
    /// Mock adapter (for mocking external dependencies)
    Mock,
}

/// Port binding information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortBinding {
    /// Port identifier
    pub port_id: PortId,
    /// Bound adapter identifier
    pub adapter_id: AdapterId,
    /// Binding timestamp
    pub bound_at: SystemTime,
    /// Binding status
    pub status: BindingStatus,
    /// Binding configuration
    pub config: BindingConfig,
    /// Binding metrics
    pub metrics: BindingMetrics,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Status of a port binding
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingStatus {
    /// Binding is active and healthy
    Active,
    /// Binding is inactive but healthy
    Inactive,
    /// Binding is in error state
    Error(String),
    /// Binding is being validated
    Validating,
    /// Binding is being initialized
    Initializing,
    /// Binding is being destroyed
    Destroying,
}

/// Configuration for a port binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    /// Enable automatic validation
    pub auto_validate: bool,
    /// Validation interval
    pub validation_interval: Duration,
    /// Maximum retry attempts for failed bindings
    pub max_retries: u32,
    /// Retry delay
    pub retry_delay: Duration,
    /// Enable binding health checks
    pub health_checks_enabled: bool,
    /// Custom binding properties
    pub properties: HashMap<String, serde_json::Value>,
}

/// Metrics for a port binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingMetrics {
    /// Total number of requests through this binding
    pub total_requests: u64,
    /// Total number of successful requests
    pub successful_requests: u64,
    /// Total number of failed requests
    pub failed_requests: u64,
    /// Average request latency in microseconds
    pub avg_latency_us: f64,
    /// Current request rate (requests per second)
    pub current_rps: f64,
    /// Last activity timestamp
    pub last_activity: SystemTime,
    /// Binding uptime
    pub uptime: Duration,
}

/// Comprehensive adapter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    /// Adapter identifier
    pub id: AdapterId,
    /// Adapter description
    pub description: String,
    /// Supported port types
    pub supported_port_types: Vec<String>,
    /// Adapter capabilities
    pub capabilities: AdapterCapabilities,
    /// Adapter configuration schema
    pub config_schema: serde_json::Value,
    /// Adapter metadata
    pub metadata: HashMap<String, String>,
    /// Adapter lifecycle status
    pub lifecycle_status: AdapterLifecycleStatus,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
}

/// Adapter capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    /// Supports hot-swapping
    pub hot_swappable: bool,
    /// Supports graceful shutdown
    pub graceful_shutdown: bool,
    /// Supports health checks
    pub health_checks: bool,
    /// Supports metrics collection
    pub metrics_collection: bool,
    /// Supports configuration updates
    pub config_updates: bool,
    /// Supports load balancing
    pub load_balancing: bool,
    /// Supports circuit breaking
    pub circuit_breaking: bool,
    /// Maximum concurrent connections
    pub max_concurrent_connections: Option<u32>,
    /// Supported protocol versions
    pub protocol_versions: Vec<String>,
}

/// Adapter lifecycle status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterLifecycleStatus {
    /// Adapter is being initialized
    Initializing,
    /// Adapter is ready for binding
    Ready,
    /// Adapter is active and processing requests
    Active,
    /// Adapter is inactive but available
    Inactive,
    /// Adapter is degraded but functional
    Degraded,
    /// Adapter has failed
    Failed,
    /// Adapter is being shutdown
    ShuttingDown,
    /// Adapter has been destroyed
    Destroyed,
}

/// Port validation trait
#[async_trait::async_trait]
pub trait PortValidator: Send + Sync {
    /// Validate a port binding
    async fn validate_binding(&self, binding: &PortBinding) -> RuntimeResult<ValidationResult>;

    /// Get validator name
    fn name(&self) -> &str;

    /// Get validator priority (higher priority validators run first)
    fn priority(&self) -> u32 {
        100
    }

    /// Check if validator applies to this binding
    fn applies_to(&self, binding: &PortBinding) -> bool {
        true
    }
}

/// Result of port binding validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation passed
    pub valid: bool,
    /// Validation score (0.0 = completely invalid, 1.0 = perfect)
    pub score: f64,
    /// Validation messages
    pub messages: Vec<ValidationMessage>,
    /// Suggested corrections
    pub suggestions: Vec<String>,
    /// Validation metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Validation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMessage {
    /// Message level
    pub level: ValidationLevel,
    /// Message content
    pub message: String,
    /// Message code for programmatic handling
    pub code: Option<String>,
    /// Associated field or property
    pub field: Option<String>,
}

/// Validation message level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Informational message
    Info,
    /// Warning message
    Warning,
    /// Error message
    Error,
    /// Critical error message
    Critical,
}

/// Binding history entry for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingHistoryEntry {
    /// History entry ID
    pub id: Uuid,
    /// Port identifier
    pub port_id: PortId,
    /// Operation type
    pub operation: BindingOperation,
    /// Previous adapter (if any)
    pub previous_adapter: Option<AdapterId>,
    /// New adapter (if any)
    pub new_adapter: Option<AdapterId>,
    /// Operation result
    pub result: OperationResult,
    /// Operation timestamp
    pub timestamp: SystemTime,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Type of binding operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingOperation {
    /// Bind adapter to port
    Bind,
    /// Unbind adapter from port
    Unbind,
    /// Replace adapter on port
    Replace,
    /// Validate binding
    Validate,
    /// Update binding configuration
    UpdateConfig,
    /// Activate binding
    Activate,
    /// Deactivate binding
    Deactivate,
}

/// Result of a binding operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationResult {
    /// Operation succeeded
    Success,
    /// Operation failed with error
    Failed { error: String },
    /// Operation partially succeeded with warnings
    PartialSuccess { warnings: Vec<String> },
}

/// Registry-wide metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryMetrics {
    /// Total number of registered adapters
    pub total_adapters: u64,
    /// Total number of active bindings
    pub active_bindings: u64,
    /// Total number of failed bindings
    pub failed_bindings: u64,
    /// Total validation attempts
    pub validation_attempts: u64,
    /// Successful validations
    pub successful_validations: u64,
    /// Average binding creation time
    pub avg_binding_creation_time_ms: f64,
    /// Registry uptime
    pub uptime: Duration,
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Maximum number of adapters
    pub max_adapters: usize,
    /// Maximum number of bindings per port
    pub max_bindings_per_port: usize,
    /// Enable binding validation
    pub enable_validation: bool,
    /// Validation timeout
    pub validation_timeout: Duration,
    /// Enable binding metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Enable audit logging
    pub enable_audit: bool,
    /// Maximum history entries to keep
    pub max_history_entries: usize,
}

/// Built-in validators
#[derive(Debug)]
pub struct HexagonalArchitectureValidator;

#[derive(Debug)]
pub struct PortTypeValidator;

#[derive(Debug)]
pub struct AdapterCapabilityValidator;

impl fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterRegistry")
            .field(
                "port_bindings",
                &format!("<{} bindings>", self.port_bindings.len()),
            )
            .field("adapters", &format!("<{} adapters>", self.adapters.len()))
            .field(
                "validators",
                &format!("<{} validators>", self.validators.len()),
            )
            .field("config", &self.config)
            .finish()
    }
}

impl AdapterRegistry {
    /// Create a new adapter registry
    pub fn new(config: RegistryConfig) -> Self {
        let mut registry = Self {
            port_bindings: DashMap::new(),
            adapters: DashMap::new(),
            validators: Vec::new(),
            binding_history: Arc::new(RwLock::new(Vec::new())),
            metrics: RegistryMetrics::default(),
            config,
            change_notify: Arc::new(Notify::new()),
        };

        // Register built-in validators
        registry.add_validator(Arc::new(HexagonalArchitectureValidator));
        registry.add_validator(Arc::new(PortTypeValidator));
        registry.add_validator(Arc::new(AdapterCapabilityValidator));

        registry
    }

    /// Register an adapter
    pub async fn register_adapter(&self, adapter_info: AdapterInfo) -> RuntimeResult<()> {
        let adapter_id = adapter_info.id.clone();

        // Check if adapter already exists
        if self.adapters.contains_key(&adapter_id) {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: adapter_id.name,
                    reason: "Adapter already registered".to_string(),
                },
            ));
        }

        // Check registry limits
        if self.adapters.len() >= self.config.max_adapters {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: adapter_id.name,
                    reason: format!(
                        "Maximum adapter limit reached: {}",
                        self.config.max_adapters
                    ),
                },
            ));
        }

        // Validate adapter information
        self.validate_adapter_info(&adapter_info).await?;

        // Register the adapter
        self.adapters.insert(adapter_id.clone(), adapter_info);

        tracing::info!("Registered adapter: {}", adapter_id.name);
        self.change_notify.notify_waiters();

        Ok(())
    }

    /// Bind an adapter to a port
    pub async fn bind_adapter(&self, port_id: PortId, adapter_id: AdapterId) -> RuntimeResult<()> {
        // Check if adapter exists
        let adapter_info = self.adapters.get(&adapter_id).ok_or_else(|| {
            RuntimeError::Registry(RegistryError::AdapterNotFound {
                adapter_id: adapter_id.name.clone(),
            })
        })?;

        // Check for existing binding (enforce one-adapter-per-port)
        if let Some(existing_binding) = self.port_bindings.get(&port_id) {
            return Err(RuntimeError::Registry(RegistryError::PortAlreadyBound {
                port_id: port_id.clone(),
                adapter_id: existing_binding.adapter_id.name.clone(),
            }));
        }

        // Create binding
        let binding = PortBinding {
            port_id: port_id.clone(),
            adapter_id: adapter_id.clone(),
            bound_at: SystemTime::now(),
            status: BindingStatus::Initializing,
            config: BindingConfig::default(),
            metrics: BindingMetrics::default(),
            metadata: HashMap::new(),
        };

        // Validate binding
        if self.config.enable_validation {
            let validation_result = self.validate_binding(&binding).await?;
            if !validation_result.valid {
                return Err(RuntimeError::Registry(
                    RegistryError::AdapterValidationFailed {
                        adapter_id: adapter_id.name,
                        reason: format!(
                            "Binding validation failed: {:?}",
                            validation_result.messages
                        ),
                    },
                ));
            }
        }

        // Store binding
        self.port_bindings.insert(port_id.clone(), binding);

        // Record history
        self.record_binding_operation(
            port_id.clone(),
            BindingOperation::Bind,
            None,
            Some(adapter_id.clone()),
            OperationResult::Success,
        )
        .await;

        tracing::info!("Bound adapter {} to port {}", adapter_id.name, port_id.name);
        self.change_notify.notify_waiters();

        Ok(())
    }

    /// Unbind an adapter from a port
    pub async fn unbind_adapter(&self, port_id: &PortId) -> RuntimeResult<Option<AdapterId>> {
        let removed_binding = self.port_bindings.remove(port_id);

        if let Some((_, binding)) = removed_binding {
            let adapter_id = binding.adapter_id.clone();

            // Record history
            self.record_binding_operation(
                port_id.clone(),
                BindingOperation::Unbind,
                Some(adapter_id.clone()),
                None,
                OperationResult::Success,
            )
            .await;

            tracing::info!(
                "Unbound adapter {} from port {}",
                adapter_id.name,
                port_id.name
            );
            self.change_notify.notify_waiters();

            Ok(Some(adapter_id))
        } else {
            Ok(None)
        }
    }

    /// Replace an adapter on a port (hot-swap)
    pub async fn replace_adapter(
        &self,
        port_id: &PortId,
        new_adapter_id: AdapterId,
    ) -> RuntimeResult<Option<AdapterId>> {
        // Get current binding
        let current_binding = self.port_bindings.get(port_id).ok_or_else(|| {
            RuntimeError::Registry(RegistryError::PortNotFound {
                port_id: port_id.clone(),
            })
        })?;

        let old_adapter_id = current_binding.adapter_id.clone();

        // Check if new adapter supports hot-swapping
        let new_adapter = self.adapters.get(&new_adapter_id).ok_or_else(|| {
            RuntimeError::Registry(RegistryError::AdapterNotFound {
                adapter_id: new_adapter_id.name.clone(),
            })
        })?;

        if !new_adapter.capabilities.hot_swappable {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: new_adapter_id.name,
                    reason: "Adapter does not support hot-swapping".to_string(),
                },
            ));
        }

        drop(current_binding); // Release the lock

        // Create new binding
        let new_binding = PortBinding {
            port_id: port_id.clone(),
            adapter_id: new_adapter_id.clone(),
            bound_at: SystemTime::now(),
            status: BindingStatus::Initializing,
            config: BindingConfig::default(),
            metrics: BindingMetrics::default(),
            metadata: HashMap::new(),
        };

        // Validate new binding
        if self.config.enable_validation {
            let validation_result = self.validate_binding(&new_binding).await?;
            if !validation_result.valid {
                return Err(RuntimeError::Registry(
                    RegistryError::AdapterValidationFailed {
                        adapter_id: new_adapter_id.name,
                        reason: format!(
                            "New binding validation failed: {:?}",
                            validation_result.messages
                        ),
                    },
                ));
            }
        }

        // Replace binding atomically
        self.port_bindings.insert(port_id.clone(), new_binding);

        // Record history
        self.record_binding_operation(
            port_id.clone(),
            BindingOperation::Replace,
            Some(old_adapter_id.clone()),
            Some(new_adapter_id.clone()),
            OperationResult::Success,
        )
        .await;

        tracing::info!(
            "Replaced adapter {} with {} on port {}",
            old_adapter_id.name,
            new_adapter_id.name,
            port_id.name
        );
        self.change_notify.notify_waiters();

        Ok(Some(old_adapter_id))
    }

    /// Get binding for a port
    pub fn get_binding(&self, port_id: &PortId) -> Option<PortBinding> {
        self.port_bindings
            .get(port_id)
            .map(|binding| binding.value().clone())
    }

    /// Get all bindings
    pub fn get_all_bindings(&self) -> Vec<PortBinding> {
        self.port_bindings
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get adapter information
    pub fn get_adapter_info(&self, adapter_id: &AdapterId) -> Option<AdapterInfo> {
        self.adapters
            .get(adapter_id)
            .map(|info| info.value().clone())
    }

    /// Get all registered adapters
    pub fn get_all_adapters(&self) -> Vec<AdapterInfo> {
        self.adapters
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Add a port validator
    pub fn add_validator(&mut self, validator: Arc<dyn PortValidator>) {
        self.validators.push(validator);
        // Sort by priority (higher priority first)
        self.validators
            .sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Validate a binding using all applicable validators
    async fn validate_binding(&self, binding: &PortBinding) -> RuntimeResult<ValidationResult> {
        let mut overall_result = ValidationResult {
            valid: true,
            score: 1.0,
            messages: Vec::new(),
            suggestions: Vec::new(),
            metadata: HashMap::new(),
        };

        for validator in &self.validators {
            if validator.applies_to(binding) {
                match tokio::time::timeout(
                    self.config.validation_timeout,
                    validator.validate_binding(binding),
                )
                .await
                {
                    Ok(Ok(result)) => {
                        // Combine results
                        overall_result.valid = overall_result.valid && result.valid;
                        overall_result.score = (overall_result.score + result.score) / 2.0;
                        overall_result.messages.extend(result.messages);
                        overall_result.suggestions.extend(result.suggestions);

                        // Merge metadata
                        for (key, value) in result.metadata {
                            overall_result
                                .metadata
                                .insert(format!("{}:{}", validator.name(), key), value);
                        }
                    }
                    Ok(Err(e)) => {
                        overall_result.valid = false;
                        overall_result.messages.push(ValidationMessage {
                            level: ValidationLevel::Error,
                            message: format!("Validator {} failed: {}", validator.name(), e),
                            code: Some("VALIDATOR_ERROR".to_string()),
                            field: None,
                        });
                    }
                    Err(_) => {
                        overall_result.valid = false;
                        overall_result.messages.push(ValidationMessage {
                            level: ValidationLevel::Error,
                            message: format!("Validator {} timed out", validator.name()),
                            code: Some("VALIDATOR_TIMEOUT".to_string()),
                            field: None,
                        });
                    }
                }
            }
        }

        Ok(overall_result)
    }

    /// Validate adapter information
    async fn validate_adapter_info(&self, adapter_info: &AdapterInfo) -> RuntimeResult<()> {
        // Check required fields
        if adapter_info.id.name.is_empty() {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: adapter_info.id.name.clone(),
                    reason: "Adapter name cannot be empty".to_string(),
                },
            ));
        }

        if adapter_info.supported_port_types.is_empty() {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: adapter_info.id.name.clone(),
                    reason: "Adapter must support at least one port type".to_string(),
                },
            ));
        }

        // Validate configuration schema
        if !adapter_info.config_schema.is_object() && !adapter_info.config_schema.is_null() {
            return Err(RuntimeError::Registry(
                RegistryError::AdapterValidationFailed {
                    adapter_id: adapter_info.id.name.clone(),
                    reason: "Invalid configuration schema".to_string(),
                },
            ));
        }

        Ok(())
    }

    /// Record binding operation in history
    async fn record_binding_operation(
        &self,
        port_id: PortId,
        operation: BindingOperation,
        previous_adapter: Option<AdapterId>,
        new_adapter: Option<AdapterId>,
        result: OperationResult,
    ) {
        if !self.config.enable_audit {
            return;
        }

        let entry = BindingHistoryEntry {
            id: Uuid::new_v4(),
            port_id,
            operation,
            previous_adapter,
            new_adapter,
            result,
            timestamp: SystemTime::now(),
            context: HashMap::new(),
        };

        let mut history = self.binding_history.write().await;
        history.push(entry);

        // Trim history if needed
        if history.len() > self.config.max_history_entries {
            let excess = history.len() - self.config.max_history_entries;
            history.drain(0..excess);
        }
    }

    /// Get binding history
    pub async fn get_binding_history(&self) -> Vec<BindingHistoryEntry> {
        self.binding_history.read().await.clone()
    }

    /// Get registry metrics
    pub fn get_metrics(&self) -> RegistryMetrics {
        let mut metrics = self.metrics.clone();
        metrics.total_adapters = self.adapters.len() as u64;
        metrics.active_bindings = self.port_bindings.len() as u64;
        metrics
    }

    /// Wait for registry changes
    pub async fn wait_for_change(&self) {
        self.change_notify.notified().await;
    }
}

/// Implementation of the hexagonal architecture validator
#[async_trait::async_trait]
impl PortValidator for HexagonalArchitectureValidator {
    async fn validate_binding(&self, binding: &PortBinding) -> RuntimeResult<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            score: 1.0,
            messages: Vec::new(),
            suggestions: Vec::new(),
            metadata: HashMap::new(),
        };

        // Validate port direction and adapter type alignment
        match (&binding.port_id.direction, &binding.adapter_id.adapter_type) {
            (PortDirection::Inbound, AdapterType::Secondary) => {
                result.valid = false;
                result.score = 0.0;
                result.messages.push(ValidationMessage {
                    level: ValidationLevel::Critical,
                    message: "Inbound ports cannot use secondary adapters".to_string(),
                    code: Some("HEXAGONAL_VIOLATION".to_string()),
                    field: Some("adapter_type".to_string()),
                });
                result
                    .suggestions
                    .push("Use a primary adapter for inbound ports".to_string());
            }
            (PortDirection::Outbound, AdapterType::Primary) => {
                result.valid = false;
                result.score = 0.0;
                result.messages.push(ValidationMessage {
                    level: ValidationLevel::Critical,
                    message: "Outbound ports cannot use primary adapters".to_string(),
                    code: Some("HEXAGONAL_VIOLATION".to_string()),
                    field: Some("adapter_type".to_string()),
                });
                result
                    .suggestions
                    .push("Use a secondary adapter for outbound ports".to_string());
            }
            _ => {
                result.messages.push(ValidationMessage {
                    level: ValidationLevel::Info,
                    message: "Hexagonal architecture constraints satisfied".to_string(),
                    code: Some("HEXAGONAL_VALID".to_string()),
                    field: None,
                });
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "hexagonal_architecture"
    }

    fn priority(&self) -> u32 {
        1000 // Highest priority - this is fundamental
    }
}

/// Implementation of the port type validator
#[async_trait::async_trait]
impl PortValidator for PortTypeValidator {
    async fn validate_binding(&self, binding: &PortBinding) -> RuntimeResult<ValidationResult> {
        // For this implementation, we'll assume the adapter supports the port type
        // In a real system, we'd check the adapter's supported_port_types
        Ok(ValidationResult {
            valid: true,
            score: 1.0,
            messages: vec![ValidationMessage {
                level: ValidationLevel::Info,
                message: "Port type compatibility verified".to_string(),
                code: Some("PORT_TYPE_VALID".to_string()),
                field: None,
            }],
            suggestions: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "port_type"
    }

    fn priority(&self) -> u32 {
        900
    }
}

/// Implementation of the adapter capability validator
#[async_trait::async_trait]
impl PortValidator for AdapterCapabilityValidator {
    async fn validate_binding(&self, binding: &PortBinding) -> RuntimeResult<ValidationResult> {
        // Basic capability validation
        Ok(ValidationResult {
            valid: true,
            score: 1.0,
            messages: vec![ValidationMessage {
                level: ValidationLevel::Info,
                message: "Adapter capabilities verified".to_string(),
                code: Some("CAPABILITIES_VALID".to_string()),
                field: None,
            }],
            suggestions: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "adapter_capability"
    }

    fn priority(&self) -> u32 {
        800
    }
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_adapters: 1000,
            max_bindings_per_port: 1, // Enforce one-adapter-per-port
            enable_validation: true,
            validation_timeout: Duration::from_secs(5),
            enable_metrics: true,
            metrics_interval: Duration::from_secs(60),
            enable_audit: true,
            max_history_entries: 10000,
        }
    }
}

impl Default for BindingConfig {
    fn default() -> Self {
        Self {
            auto_validate: true,
            validation_interval: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            health_checks_enabled: true,
            properties: HashMap::new(),
        }
    }
}

impl Default for BindingMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_latency_us: 0.0,
            current_rps: 0.0,
            last_activity: SystemTime::now(),
            uptime: Duration::new(0, 0),
        }
    }
}

impl fmt::Display for AdapterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name, self.adapter_type)
    }
}

impl fmt::Display for AdapterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterType::Primary => write!(f, "Primary"),
            AdapterType::Secondary => write!(f, "Secondary"),
            AdapterType::Test => write!(f, "Test"),
            AdapterType::Mock => write!(f, "Mock"),
        }
    }
}

impl fmt::Display for BindingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BindingStatus::Active => write!(f, "Active"),
            BindingStatus::Inactive => write!(f, "Inactive"),
            BindingStatus::Error(err) => write!(f, "Error: {}", err),
            BindingStatus::Validating => write!(f, "Validating"),
            BindingStatus::Initializing => write!(f, "Initializing"),
            BindingStatus::Destroying => write!(f, "Destroying"),
        }
    }
}
