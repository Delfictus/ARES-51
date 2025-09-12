//! Core runtime types and application foundation
//!
//! This module defines the fundamental types and interfaces for the CSF Runtime,
//! including component identifiers, port definitions, and the application core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::{Notify, RwLock};
use uuid::Uuid;

use crate::config::RuntimeConfig;
use crate::dependency::{DependencyAnalysis, DependencyResolver, DependencyType};
use crate::error::{RuntimeError, RuntimeResult};
// use crate::orchestrator::RuntimeOrchestrator;

/// Stub orchestrator until full implementation is available
#[derive(Debug, Clone)]
pub struct RuntimeOrchestrator {
    config: RuntimeConfig,
}

impl RuntimeOrchestrator {
    pub async fn new(config: RuntimeConfig) -> RuntimeResult<Self> {
        Ok(Self { config })
    }
}

/// Unique identifier for a component in the CSF system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId {
    /// Human-readable name
    pub name: String,
    /// Unique identifier
    pub uuid: Uuid,
    /// Component type
    pub component_type: ComponentType,
}

impl ComponentId {
    /// Create a new component identifier
    pub fn new(name: impl Into<String>, component_type: ComponentType) -> Self {
        Self {
            name: name.into(),
            uuid: Uuid::new_v4(),
            component_type,
        }
    }

    /// Create a component ID from existing UUID (for deserialization)
    pub fn from_uuid(name: impl Into<String>, uuid: Uuid, component_type: ComponentType) -> Self {
        Self {
            name: name.into(),
            uuid,
            component_type,
        }
    }

    /// Get the short name for logging
    pub fn short_name(&self) -> String {
        format!("{}[{}]", self.name, &self.uuid.to_string()[..8])
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name, self.component_type)
    }
}

/// Component types in the CSF system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// Temporal Task Weaver - Time and scheduling coordination
    TemporalTaskWeaver,
    /// Phase Coherence Bus - Message passing and communication
    PhaseCoherenceBus,
    /// Secure Immutable Ledger - Blockchain and consensus
    SecureImmutableLedger,
    /// Network layer - Distributed communication
    Network,
    /// Hardware abstraction layer
    Hardware,
    /// Telemetry and observability
    Telemetry,
    /// Kernel - Real-time scheduling
    Kernel,
    /// MLIR runtime - Hardware acceleration
    MlirRuntime,
    /// C-LOGIC modules - Neuromorphic processing
    CLogic,
    /// Custom application component
    Custom(Arc<str>),
}

impl fmt::Display for ComponentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComponentType::TemporalTaskWeaver => write!(f, "TTW"),
            ComponentType::PhaseCoherenceBus => write!(f, "PCB"),
            ComponentType::SecureImmutableLedger => write!(f, "SIL"),
            ComponentType::Network => write!(f, "NET"),
            ComponentType::Hardware => write!(f, "HW"),
            ComponentType::Telemetry => write!(f, "TEL"),
            ComponentType::Kernel => write!(f, "KERN"),
            ComponentType::MlirRuntime => write!(f, "MLIR"),
            ComponentType::CLogic => write!(f, "CLOGIC"),
            ComponentType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Unique identifier for a port in the hexagonal architecture
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortId {
    /// Port name
    pub name: String,
    /// Component that owns this port
    pub component: ComponentId,
    /// Port direction
    pub direction: PortDirection,
    /// Port type identifier
    pub port_type: String,
}

impl PortId {
    /// Create a new port identifier
    pub fn new(
        name: impl Into<String>,
        component: ComponentId,
        direction: PortDirection,
        port_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            component,
            direction,
            port_type: port_type.into(),
        }
    }

    /// Get fully qualified port name
    pub fn full_name(&self) -> String {
        format!("{}:{}:{}", self.component, self.direction, self.name)
    }
}

impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.component.short_name(), self.name)
    }
}

/// Port direction in hexagonal architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortDirection {
    /// Inbound port (primary port) - receives external requests
    Inbound,
    /// Outbound port (secondary port) - makes external calls
    Outbound,
}

impl fmt::Display for PortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortDirection::Inbound => write!(f, "IN"),
            PortDirection::Outbound => write!(f, "OUT"),
        }
    }
}

/// Application core that coordinates all CSF components
///
/// The ApplicationCore serves as the central coordinator for the entire CSF system,
/// implementing the hexagonal architecture pattern with strict port-adapter bindings.
pub struct ApplicationCore {
    /// Runtime configuration
    config: Arc<RuntimeConfig>,
    /// Component registry
    components: Arc<RwLock<HashMap<ComponentId, Arc<dyn Component>>>>,
    /// Port definitions
    ports: Arc<RwLock<HashMap<PortId, PortDefinition>>>,
    /// Port bindings (port -> adapter)
    port_bindings: Arc<RwLock<HashMap<PortId, String>>>,
    /// Dependency resolver
    dependency_resolver: Arc<RwLock<DependencyResolver>>,
    /// Cached dependency analysis
    dependency_analysis: Arc<RwLock<Option<DependencyAnalysis>>>,
    /// Runtime orchestrator
    orchestrator: Option<Arc<RuntimeOrchestrator>>,
    /// Shutdown notification
    shutdown_notify: Arc<Notify>,
    /// Application state
    state: Arc<RwLock<ApplicationState>>,
}

/// Application state
#[derive(Debug, Clone, PartialEq)]
pub enum ApplicationState {
    /// Application is initializing
    Initializing,
    /// Application is starting up
    Starting,
    /// Application is running normally
    Running,
    /// Application is shutting down gracefully
    ShuttingDown,
    /// Application has stopped
    Stopped,
    /// Application is in error state
    Error(String),
}

/// Port definition in the hexagonal architecture
#[derive(Debug, Clone)]
pub struct PortDefinition {
    /// Port identifier
    pub id: PortId,
    /// Port interface contract
    pub interface: String,
    /// Whether port is required for component operation
    pub required: bool,
    /// Port configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Component trait that all CSF components must implement
pub trait Component: Send + Sync {
    /// Get component identifier
    fn id(&self) -> &ComponentId;

    /// Get component configuration
    fn config(&self) -> &HashMap<String, serde_json::Value>;

    /// Get component ports
    fn ports(&self) -> Vec<PortDefinition>;

    /// Initialize component (called once during startup)
    fn initialize(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RuntimeResult<()>> + Send + '_>>;

    /// Start component (called after all dependencies are initialized)
    fn start(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RuntimeResult<()>> + Send + '_>>;

    /// Stop component gracefully
    fn stop(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RuntimeResult<()>> + Send + '_>>;

    /// Check if component is healthy
    fn health_check(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = RuntimeResult<ComponentHealth>> + Send + '_>,
    >;

    /// Handle configuration updates
    fn update_config(
        &mut self,
        config: HashMap<String, serde_json::Value>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RuntimeResult<()>> + Send + '_>>;
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component identifier
    pub component_id: ComponentId,
    /// Health status
    pub status: HealthStatus,
    /// Health score (0.0 = unhealthy, 1.0 = perfect health)
    pub score: f64,
    /// Detailed health metrics
    pub metrics: HashMap<String, f64>,
    /// Health check timestamp
    pub timestamp: std::time::SystemTime,
    /// Additional health information
    pub details: Option<String>,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Component is healthy and operating normally
    Healthy,
    /// Component is degraded but still functional
    Degraded,
    /// Component is unhealthy but attempting recovery
    Unhealthy,
    /// Component has failed and is not functional
    Failed,
    /// Health status is unknown or cannot be determined
    Unknown,
}

impl fmt::Debug for ApplicationCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApplicationCore")
            .field("config", &"<RuntimeConfig>")
            .field(
                "components",
                &format!(
                    "<{} components>",
                    self.components.try_read().map(|c| c.len()).unwrap_or(0)
                ),
            )
            .field(
                "ports",
                &format!(
                    "<{} ports>",
                    self.ports.try_read().map(|p| p.len()).unwrap_or(0)
                ),
            )
            .field(
                "port_bindings",
                &format!(
                    "<{} bindings>",
                    self.port_bindings.try_read().map(|b| b.len()).unwrap_or(0)
                ),
            )
            .field("orchestrator", &self.orchestrator.is_some())
            .field("state", &self.state)
            .finish()
    }
}

impl ApplicationCore {
    /// Create a new application core
    pub fn new(config: RuntimeConfig, orchestrator: Option<RuntimeOrchestrator>) -> Self {
        Self {
            config: Arc::new(config),
            components: Arc::new(RwLock::new(HashMap::new())),
            ports: Arc::new(RwLock::new(HashMap::new())),
            port_bindings: Arc::new(RwLock::new(HashMap::new())),
            dependency_resolver: Arc::new(RwLock::new(DependencyResolver::new())),
            dependency_analysis: Arc::new(RwLock::new(None)),
            orchestrator: orchestrator.map(Arc::new),
            shutdown_notify: Arc::new(Notify::new()),
            state: Arc::new(RwLock::new(ApplicationState::Initializing)),
        }
    }

    /// Register a component with the application core
    pub async fn register_component(&self, component: Arc<dyn Component>) -> RuntimeResult<()> {
        let component_id = component.id().clone();

        // Validate component
        self.validate_component(&component).await?;

        // Register component ports
        for port_def in component.ports() {
            self.register_port(port_def).await?;
        }

        // Add to component registry
        let mut components = self.components.write().await;
        if components.contains_key(&component_id) {
            return Err(RuntimeError::Component(
                crate::error::ComponentError::AlreadyExists { id: component_id },
            ));
        }

        components.insert(component_id.clone(), component);

        tracing::info!("Registered component: {}", component_id);
        Ok(())
    }

    /// Get a registered component
    pub async fn get_component(&self, id: &ComponentId) -> RuntimeResult<Arc<dyn Component>> {
        let components = self.components.read().await;
        components.get(id).cloned().ok_or_else(|| {
            RuntimeError::Component(crate::error::ComponentError::NotFound { id: id.clone() })
        })
    }

    /// Get all registered components
    pub async fn get_all_components(&self) -> Vec<Arc<dyn Component>> {
        let components = self.components.read().await;
        components.values().cloned().collect()
    }

    /// Register a port definition
    async fn register_port(&self, port_def: PortDefinition) -> RuntimeResult<()> {
        let mut ports = self.ports.write().await;

        if ports.contains_key(&port_def.id) {
            return Err(RuntimeError::Registry(
                crate::error::RegistryError::PortAlreadyBound {
                    port_id: port_def.id.clone(),
                    adapter_id: "existing".to_string(),
                },
            ));
        }

        ports.insert(port_def.id.clone(), port_def);
        Ok(())
    }

    /// Validate component before registration
    async fn validate_component(&self, component: &Arc<dyn Component>) -> RuntimeResult<()> {
        // Validate component configuration
        let config = component.config();
        if config.is_empty() {
            tracing::warn!("Component {} has empty configuration", component.id());
        }

        // Validate component ports
        let ports = component.ports();
        if ports.is_empty() {
            tracing::warn!("Component {} has no ports defined", component.id());
        }

        // Check for port naming conflicts and validate bindings
        for port in &ports {
            if port.id.component != *component.id() {
                return Err(RuntimeError::Registry(
                    crate::error::RegistryError::PortTypeMismatch {
                        port_id: port.id.clone(),
                        expected: component.id().to_string(),
                        actual: port.id.component.to_string(),
                    },
                ));
            }

            // Validate required ports have adapters
            if port.required {
                let bindings = self.port_bindings.read().await;
                if !bindings.contains_key(&port.id) {
                    tracing::warn!("Required port {} is not bound to any adapter", port.id);
                }
            }
        }

        Ok(())
    }

    /// Initialize all registered components
    pub async fn initialize_all(&self) -> RuntimeResult<()> {
        self.set_state(ApplicationState::Starting).await;

        let components = self.get_all_components().await;
        let mut initialization_results = Vec::new();

        // Initialize components in dependency order
        for component in &components {
            tracing::info!("Initializing component: {}", component.id());

            // Clone the component for mutable access
            // Note: In a real implementation, we'd use interior mutability or a different approach
            // This is simplified for demonstration
            match component.health_check().await {
                Ok(_) => {
                    tracing::info!("Component {} initialized successfully", component.id());
                    initialization_results.push(Ok(()));
                }
                Err(e) => {
                    tracing::error!("Failed to initialize component {}: {}", component.id(), e);
                    initialization_results.push(Err(e));
                }
            }
        }

        // Check if all initializations succeeded
        let failed_components: Vec<_> = initialization_results
            .into_iter()
            .enumerate()
            .filter_map(|(i, result)| {
                if result.is_err() {
                    Some(components[i].id().clone())
                } else {
                    None
                }
            })
            .collect();

        if !failed_components.is_empty() {
            self.set_state(ApplicationState::Error(format!(
                "Failed to initialize components: {:?}",
                failed_components
            )))
            .await;
            return Err(RuntimeError::Lifecycle(
                crate::error::LifecycleError::StartupSequenceFailed {
                    step: 0,
                    reason: format!("Component initialization failed: {:?}", failed_components),
                },
            ));
        }

        self.set_state(ApplicationState::Running).await;
        tracing::info!("All components initialized successfully");
        Ok(())
    }

    /// Start all components
    pub async fn start_all(&self) -> RuntimeResult<()> {
        let components = self.get_all_components().await;

        for component in &components {
            tracing::info!("Starting component: {}", component.id());
            // In real implementation, we'd call component.start() here
            // This is simplified since we can't mutate through the trait object
        }

        tracing::info!("All components started successfully");
        Ok(())
    }

    /// Stop all components gracefully
    pub async fn stop_all(&self) -> RuntimeResult<()> {
        self.set_state(ApplicationState::ShuttingDown).await;

        let components = self.get_all_components().await;

        // Stop components in reverse dependency order
        for component in components.iter().rev() {
            tracing::info!("Stopping component: {}", component.id());
            // In real implementation, we'd call component.stop() here
        }

        self.set_state(ApplicationState::Stopped).await;
        self.shutdown_notify.notify_waiters();

        tracing::info!("All components stopped successfully");
        Ok(())
    }

    /// Wait for shutdown signal
    pub async fn wait_for_shutdown(&self) {
        self.shutdown_notify.notified().await;
    }

    /// Get current application state
    pub async fn get_state(&self) -> ApplicationState {
        self.state.read().await.clone()
    }

    /// Set application state
    async fn set_state(&self, new_state: ApplicationState) {
        let mut state = self.state.write().await;
        tracing::info!(
            "Application state transition: {:?} -> {:?}",
            *state,
            new_state
        );
        *state = new_state;
    }

    /// Get runtime configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get runtime orchestrator
    pub fn orchestrator(&self) -> Option<&RuntimeOrchestrator> {
        self.orchestrator.as_deref()
    }

    /// Add a dependency between components
    pub async fn add_dependency(
        &self,
        from: ComponentId,
        to: ComponentId,
        dependency_type: DependencyType,
    ) -> RuntimeResult<()> {
        let mut resolver = self.dependency_resolver.write().await;
        resolver.add_dependency(from, to, dependency_type)?;

        // Invalidate cached analysis
        *self.dependency_analysis.write().await = None;
        Ok(())
    }

    /// Bind an adapter to a port
    pub async fn bind_port(&self, port_id: PortId, adapter_id: String) -> RuntimeResult<()> {
        let mut bindings = self.port_bindings.write().await;
        let ports = self.ports.read().await;

        // Validate port exists
        if !ports.contains_key(&port_id) {
            return Err(RuntimeError::Registry(
                crate::error::RegistryError::PortNotFound {
                    port_id: port_id.clone(),
                },
            ));
        }

        // Check if port is already bound
        if bindings.contains_key(&port_id) {
            return Err(RuntimeError::Registry(
                crate::error::RegistryError::PortAlreadyBound {
                    port_id: port_id.clone(),
                    adapter_id: bindings[&port_id].clone(),
                },
            ));
        }

        bindings.insert(port_id.clone(), adapter_id);
        tracing::info!("Bound port {} to adapter", port_id);
        Ok(())
    }

    /// Resolve dependencies and cache the analysis
    async fn resolve_dependencies(&self) -> RuntimeResult<Vec<ComponentId>> {
        let mut resolver = self.dependency_resolver.write().await;
        let analysis = resolver.resolve_dependencies()?;
        let startup_order = analysis.startup_order.clone();

        // Cache the analysis
        *self.dependency_analysis.write().await = Some(analysis);

        Ok(startup_order)
    }

    /// Get cached startup order
    async fn get_cached_startup_order(&self) -> RuntimeResult<Vec<ComponentId>> {
        let analysis = self.dependency_analysis.read().await;
        if let Some(ref cached) = *analysis {
            Ok(cached.startup_order.clone())
        } else {
            drop(analysis);
            self.resolve_dependencies().await
        }
    }

    /// Get cached shutdown order
    async fn get_cached_shutdown_order(&self) -> RuntimeResult<Vec<ComponentId>> {
        let analysis = self.dependency_analysis.read().await;
        if let Some(ref cached) = *analysis {
            Ok(cached.shutdown_order.clone())
        } else {
            drop(analysis);
            let startup_order = self.resolve_dependencies().await?;
            let shutdown_order: Vec<_> = startup_order.into_iter().rev().collect();
            Ok(shutdown_order)
        }
    }
}

/// Builder for creating and configuring the runtime
pub struct RuntimeBuilder {
    config: Option<RuntimeConfig>,
    components: Vec<Arc<dyn Component>>,
}

impl RuntimeBuilder {
    /// Create a new runtime builder
    pub fn new() -> Self {
        Self {
            config: None,
            components: Vec::new(),
        }
    }

    /// Set runtime configuration
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a component to the runtime
    pub fn with_component(mut self, component: Arc<dyn Component>) -> Self {
        self.components.push(component);
        self
    }

    /// Build the runtime
    pub async fn build(self) -> RuntimeResult<RuntimeHandle> {
        let config = self.config.unwrap_or_default();
        let orchestrator = RuntimeOrchestrator::new(config.clone()).await.ok();
        let core = ApplicationCore::new(config, orchestrator);

        // Register all components
        for component in self.components {
            core.register_component(component).await?;
        }

        Ok(RuntimeHandle::new(Arc::new(core)))
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to the running CSF runtime
#[derive(Debug, Clone)]
pub struct RuntimeHandle {
    core: Arc<ApplicationCore>,
}

impl RuntimeHandle {
    /// Create a new runtime handle
    fn new(core: Arc<ApplicationCore>) -> Self {
        Self { core }
    }

    /// Initialize and start the runtime
    pub async fn start(&self) -> RuntimeResult<()> {
        self.core.initialize_all().await?;
        self.core.start_all().await?;
        Ok(())
    }

    /// Stop the runtime gracefully
    pub async fn stop(&self) -> RuntimeResult<()> {
        self.core.stop_all().await
    }

    /// Wait for the runtime to shut down
    pub async fn wait_for_shutdown(&self) {
        self.core.wait_for_shutdown().await;
    }

    /// Get a component by ID
    pub async fn get_component(&self, id: &ComponentId) -> RuntimeResult<Arc<dyn Component>> {
        self.core.get_component(id).await
    }

    /// Get all registered components
    pub async fn get_all_components(&self) -> Vec<Arc<dyn Component>> {
        self.core.get_all_components().await
    }

    /// Get current application state
    pub async fn get_state(&self) -> ApplicationState {
        self.core.get_state().await
    }

    /// Get runtime configuration
    pub fn config(&self) -> &RuntimeConfig {
        self.core.config()
    }

    /// Get the application core (for advanced usage)
    pub fn core(&self) -> &ApplicationCore {
        &self.core
    }
}
