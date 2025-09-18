//! # ARES ChronoSynclastic Fabric (CSF) Core
//!
//! High-performance distributed computation fabric for quantum-temporal processing,
//! variational optimization, and enterprise-scale tensor operations.
//!
//! ## Overview
//!
//! The CSF Core provides the foundational infrastructure for building distributed,
//! quantum-aware computational systems. It combines traditional high-performance
//! computing with quantum-inspired algorithms and temporal synchronization.
//!
//! ## Features
//!
//! - **Quantum-Temporal Processing**: Hybrid quantum-classical computation with temporal coherence
//! - **High-Performance Tensors**: Enterprise-grade tensor operations with GPU acceleration
//! - **Distributed Computing**: Fault-tolerant distributed computation with consensus
//! - **Variational Optimization**: Advanced optimization algorithms for complex search spaces
//! - **Streaming Processing**: Real-time data processing with zero-copy operations
//! - **Enterprise Integration**: Production-ready systems with monitoring and observability
//!
//! ## Quick Start
//!
//! ```rust
//! use ares_csf_core::prelude::*;
//!
//! // Create a basic CSF computation context
//! let config = CSFConfig::default();
//! let context = CSFContext::new(config)?;
//!
//! // Perform quantum-inspired optimization
//! let result = context.optimize_variational(parameters).await?;
//! # Ok::<(), ares_csf_core::Error>(())
//! ```

#![allow(clippy::pedantic, warnings)]
#![cfg_attr(not(feature = "std"), no_std)]

// Core modules
pub mod error;
pub mod types;
pub mod ports;

// Core messaging and communication
pub mod envelope;
pub mod phase_packet;

// High-performance computing
pub mod tensor;
pub mod tensor_real;
pub mod tensor_verification;
pub mod hpc;

// Quantum and variational computing
pub mod energy_functional;
pub mod variational;

// Enterprise and integration
pub mod integration;
pub mod demo;

// Re-export the primary error type for convenience
pub use error::Error;

// Re-export all port traits to define the core API
pub use ports::{
    Consensus, DeadlineScheduler, EventBusRx, EventBusTx, HlcClock, SecureImmutableLedger,
    TimeSource,
};

// Re-export all core types
pub use types::{
    hardware_timestamp, ComponentId, NanoTime, Priority, TaskId, Phase, PhaseState, Timestamp,
};

// Conditional re-exports based on features
#[cfg(feature = "csf-protocol")]
pub use csf_protocol::{
    PacketFlags, PacketHeader, PacketId, PacketPayload, PacketType, PhasePacket,
};

/// Result type alias for CSF operations
pub type Result<T> = std::result::Result<T, Error>;

/// Prelude module that re-exports commonly used types and traits
pub mod prelude {
    pub use crate::error::Error;
    pub use crate::ports::*;
    pub use crate::types::*;
    pub use crate::Result;

    // Re-export protocol types if available
    #[cfg(feature = "csf-protocol")]
    pub use csf_protocol::{
        PacketCodec, PacketFlags, PacketHeader, PacketId, PacketPayload, PacketType,
        PacketValidator, PhasePacket, ValidationError,
    };

    // Re-export variational module types
    pub use crate::variational::{
        AdvancedOptimizer, OptimizationAlgorithm, PhaseTransitionOperator,
        RelationalPhaseEnergyFunctional, StructuralModification,
    };

    // Re-export integration framework types
    pub use crate::integration::{
        ComponentState, DashboardState, DrppConfig, DrppRuntime, DrppTestScenario,
        EmergentPattern, HealthLevel, MonitorConfig, PhaseTransitionEvent, RuntimeEvent,
        RuntimeStats, SystemHealthStatus, TestResults,
    };

    // Re-export HPC module types
    pub use crate::hpc::{
        DistributedCompute, GPULinearAlgebra, HPCConfiguration, HardwareCapabilities, MemoryPool,
        OptimizedMatrix, PerformanceProfiler, SIMDLinearAlgebra, StreamingBuffer,
        StreamingProcessor,
    };

    // Re-export demo module types
    pub use crate::demo::{
        AresProofOfPowerDemo, CertificationLevel, NetworkBenchmark, ProofOfPowerResults,
        QuantumBenchmark, TemporalCoherence, run_proof_of_power_demo,
    };

    // Re-export tensor operations for high-performance computing
    #[cfg(feature = "tensor-ops")]
    pub use crate::tensor::{
        TensorOps, ComplexTensor, RealTensor, TensorResult, tensor_multiply, tensor_transpose,
        tensor_inverse, eigenvalue_decomposition, svd_decomposition,
    };
}

/// Main CSF computation context
pub struct CSFContext {
    config: CSFConfig,
    #[cfg(feature = "hpc")]
    hpc_runtime: Option<crate::hpc::HPCRuntime>,
    #[cfg(feature = "quantum")]
    quantum_backend: Option<crate::variational::QuantumBackend>,
}

/// Configuration for CSF computation context
#[derive(Debug, Clone)]
pub struct CSFConfig {
    /// Number of worker threads for parallel computation
    pub worker_threads: usize,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Memory pool size for tensor operations (in MB)
    pub memory_pool_size: usize,
    /// Enable quantum-inspired optimizations
    pub enable_quantum: bool,
    /// Temporal precision requirements
    pub temporal_precision: crate::types::PrecisionLevel,
}

impl Default for CSFConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            enable_gpu: false, // Disabled by default for compatibility
            memory_pool_size: 1024, // 1GB default
            enable_quantum: false, // Disabled by default
            temporal_precision: crate::types::PrecisionLevel::Standard,
        }
    }
}

impl CSFContext {
    /// Create a new CSF computation context
    pub fn new(config: CSFConfig) -> Result<Self> {
        Ok(Self {
            config,
            #[cfg(feature = "hpc")]
            hpc_runtime: None,
            #[cfg(feature = "quantum")]
            quantum_backend: None,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &CSFConfig {
        &self.config
    }

    /// Initialize HPC runtime if available
    #[cfg(feature = "hpc")]
    pub async fn initialize_hpc(&mut self) -> Result<()> {
        self.hpc_runtime = Some(crate::hpc::HPCRuntime::new(&self.config).await?);
        Ok(())
    }

    /// Initialize quantum backend if available
    #[cfg(feature = "quantum")]
    pub async fn initialize_quantum(&mut self) -> Result<()> {
        self.quantum_backend = Some(crate::variational::QuantumBackend::new(&self.config).await?);
        Ok(())
    }
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Get version information
pub fn version_info() -> String {
    format!("{} v{} by {}", DESCRIPTION, VERSION, AUTHORS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(info.contains("0.1.0"));
        assert!(info.contains("CSF"));
    }

    #[test]
    fn test_default_config() {
        let config = CSFConfig::default();
        assert!(config.worker_threads > 0);
        assert_eq!(config.memory_pool_size, 1024);
    }

    #[tokio::test]
    async fn test_context_creation() {
        let config = CSFConfig::default();
        let context = CSFContext::new(config).unwrap();
        assert_eq!(context.config().worker_threads, num_cpus::get());
    }
}