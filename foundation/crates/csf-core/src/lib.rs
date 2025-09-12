//! The core crate for the Compute and Sensor Fusion (CSF) platform.
//!
//! This crate defines the essential traits (ports) and types for the hexagonal
//! architecture. It provides the vocabulary for building the domain logic and
//! adapters.

// Relaxed linting for integration development
#![allow(warnings, missing_docs, clippy::pedantic, unsafe_code)]

pub mod data;
pub mod demo;
pub mod energy_functional;
pub mod envelope;
pub mod error;
pub mod hpc;
pub mod integration;
pub mod phase_packet;
pub mod ports;
pub mod tensor;
pub mod tensor_real;
pub mod tensor_verification;
pub mod trading;
pub mod types;
pub mod variational;

// Re-export the primary error type for convenience.
pub use error::Error;

// Re-export all port traits to define the core API.
pub use ports::{
    Consensus, DeadlineScheduler, EventBusRx, EventBusTx, HlcClock, SecureImmutableLedger,
    TimeSource,
};

// Re-export all core types.
pub use types::{hardware_timestamp, ComponentId, NanoTime, Priority, TaskId};

// Re-export protocol types at the crate level
pub use csf_protocol::{
    PacketFlags, PacketHeader, PacketId, PacketPayload, PacketType, PhasePacket,
};

/// Prelude module that re-exports commonly used types and traits
pub mod prelude {
    pub use crate::error::Error;
    pub use crate::ports::*;
    pub use crate::types::*;

    // Re-export canonical protocol types
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
        ComponentState, DashboardState, DrppConfig, DrppMonitor, DrppRuntime, DrppTestScenario,
        EmergentPattern, HealthLevel, MonitorConfig, PhaseTransitionEvent, RuntimeEvent,
        RuntimeStats, SystemHealthStatus, TestResults,
    };

    // Re-export hardware timestamp function
    pub use crate::types::hardware_timestamp;

    // Re-export HPC module types
    pub use crate::hpc::{
        DistributedCompute, GPULinearAlgebra, HPCConfiguration, HardwareCapabilities, MemoryPool,
        OptimizedMatrix, PerformanceProfiler, SIMDLinearAlgebra, StreamingBuffer,
        StreamingProcessor,
    };

    // Re-export data module types
    pub use crate::data::{
        DataSource, HistoricalDataConfig, HistoricalDataError, HistoricalDataFetcher,
        HistoricalDataPoint, TimeInterval,
    };

    // Re-export demo module types
    pub use crate::demo::{
        AresProofOfPowerDemo, CertificationLevel, NetworkBenchmark, ProofOfPowerResults,
        QuantumBenchmark, TemporalCoherence, TradingBenchmark, run_proof_of_power_demo,
    };

    // Re-export trading engine types
    pub use crate::trading::{
        AresQuantumTradingEngine, InstrumentId, KellyCriterionOptimizer, MarketDataPoint,
        MarketSimulator, Order, OrderId, OrderSide, OrderStatus, OrderType, Position,
        QuantumSignalGenerator, TradingSignal, TradingStats, run_trading_demo,
    };
}
