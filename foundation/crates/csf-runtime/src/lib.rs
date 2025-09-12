//! NovaCore ChronoSynclastic Fabric Runtime Orchestrator
//!
//! The CSF Runtime serves as the central coordinator for the hexagonal architecture,
//! enforcing the "one-adapter-per-port" rule and providing comprehensive system
//! orchestration, lifecycle management, and dependency resolution.
//!
//! ## Core Responsibilities
//!
//! - **Application Assembly**: Coordinate all CSF components into a cohesive system
//! - **Hexagonal Architecture Enforcement**: Ensure architectural constraints
//! - **Dependency Resolution**: Advanced topological sorting with cycle detection
//! - **Lifecycle Management**: Component startup, shutdown, and health monitoring
//! - **Configuration Management**: Centralized configuration with validation
//! - **Performance Orchestration**: Real-time performance optimization coordination
//!
//! ## Architecture
//!
//! The runtime implements a sophisticated component orchestration system:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   CSF Runtime Orchestrator                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │   Adapter   │  │ Dependency  │  │     Lifecycle       │  │
//! │  │  Registry   │  │  Resolver   │  │     Manager        │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────────┐  │
//! │  │ TTW │  │ PCB │  │ SIL │  │ NET │  │ HW  │  │ TELNET │  │
//! │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![allow(unsafe_code)] // Required for high-performance coordination

pub mod config;
pub mod core;
pub mod dependency;
pub mod error;
pub mod health;
pub mod lifecycle;
pub mod orchestrator;
pub mod performance;
pub mod registry;

// Re-export core types for convenience
pub use crate::config::{ComponentConfig, ConfigurationManager, RuntimeConfig};
pub use crate::core::{
    ApplicationCore, Component, ComponentId, ComponentType, PortId, RuntimeBuilder, RuntimeHandle,
};
pub use crate::dependency::{CircularDependencyError, DependencyGraph, DependencyResolver};
pub use crate::error::{RuntimeError, RuntimeResult};
pub use crate::health::{ComponentHealthState, HealthEvent, HealthMonitor, HealthStatus};
pub use crate::lifecycle::{ComponentState, LifecycleManager, StartupSequence};
pub use crate::orchestrator::{OrchestrationPlan, RuntimeOrchestrator, SystemCoordinator};
pub use crate::performance::{
    GlobalPerformanceState, OptimizationStrategy, PerformanceEvent, PerformanceOrchestrator,
};
pub use crate::registry::{AdapterInfo, AdapterRegistry, PortBinding, PortValidator};

/// Current version of the CSF Runtime
pub const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum number of components that can be registered
pub const MAX_COMPONENTS: usize = 1000;

/// Maximum depth for dependency resolution
pub const MAX_DEPENDENCY_DEPTH: usize = 100;

/// Default health check interval
pub const DEFAULT_HEALTH_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Default startup timeout
pub const DEFAULT_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Default shutdown timeout  
pub const DEFAULT_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
