//! Hephaestus Forge - Metamorphic Execution Substrate (MES)
//!
//! An always-on component of csf-runtime that orchestrates self-modification
//! across three parallel operational vectors:
//! 1. Cognition & Generation: Problem identification and solution synthesis
//! 2. Simulation & Evolution: Testing and selection in secure environment
//! 3. Governance & Integration: Consensus, auditing, and zero-downtime integration

pub mod api;
pub mod config;
pub mod intent;
pub mod storage;
pub mod orchestrator;
pub mod sandbox;
pub mod ledger;
pub mod temporal;
pub mod resonance;
pub mod drpp;
pub mod synthesis;
pub mod validation;
pub mod monitor;
pub mod adapters;
pub mod types;
pub mod tpra;
pub mod accelerator;
pub mod distributed;
pub mod profiling;
#[cfg(feature = "monitoring")]
pub mod monitoring;
pub mod mlir_synthesis;
pub mod autonomous;
pub mod ares_bridge;
pub mod workload;
pub mod quantum_integration;
#[cfg(feature = "abi-verification")]
pub mod abi;

// Re-export main public API
pub use api::{HephaestusForge, ForgeStatus, SynthesisStrategy, SynthesisSpec, MlirModule, SynthesisError, MetricsClient};
pub use config::{
    ForgeConfig, ForgeConfigBuilder, OperationalMode, RiskConfig, MonitoringConfig, 
    ObservabilityConfig, HitlConfig, TestingConfig, ResourceLimits, AutonomyLevel,
    Severity, AlertRule, AlertDestination, LogLevel, LogDestination
};
pub use intent::{
    OptimizationIntent, IntentId, IntentStatus, Objective, Constraint, Priority,
    OptimizationOpportunity, OpportunityType, OptimizationDomain
};

// Re-export core components for advanced usage
pub use orchestrator::MetamorphicRuntimeOrchestrator;
pub use sandbox::HardenedSandbox;
pub use ledger::MetamorphicLedger;
pub use temporal::TemporalSwapCoordinator;
pub use drpp::EnhancedDRPP;
pub use synthesis::ProgramSynthesizer;
pub use validation::MetamorphicTestSuite;
pub use monitor::ForgeMonitor;

// Re-export resonance components for testing
pub use resonance::{DynamicResonanceProcessor, ComputationTensor, StabilizedPattern};
pub use resonance::{HarmonicInducer, PhaseLattice, ResonantMode};
pub use resonance::{AdaptiveDissipativeProcessor, DissipationStrategy, InterferencePatterns};
pub use resonance::{TopologicalAnalyzer};

// Re-export TPRA components
pub use tpra::{
    TopologicalPhaseResonanceAnalyzer,
    PhaseTopologyMapper,
    PersistentHomologyEngine,
    ResonanceModeTracker,
    TopologicalInvariantDetector,
    PhaseTransitionAnalyzer,
    BettiNumberTracker,
    TPRAConfig
};

// Re-export types
pub use types::*;

// Re-export ARES bridge for integration
pub use ares_bridge::AresSystemBridge;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_forge_initialization() {
        let config = ForgeConfig::default();
        let forge = HephaestusForge::new_async_public(config).await;
        assert!(forge.is_ok());
    }
    
    #[tokio::test]
    async fn test_config_builder() {
        let config = ForgeConfig::builder()
            .mode(OperationalMode::Supervised)
            .risk_tolerance(0.6)
            .max_concurrent_optimizations(4)
            .synthesis_timeout(std::time::Duration::from_secs(300))
            .consensus_nodes(vec!["node1:8080", "node2:8080"])
            .build();
        
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.mode, OperationalMode::Supervised);
    }
    
    #[tokio::test]
    async fn test_intent_builder() {
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(Objective::MinimizeLatency { 
                percentile: 99.0,
                target_ms: 10.0 
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(Priority::High)
            .build();
        
        assert!(intent.is_ok());
        let intent = intent.unwrap();
        assert_eq!(intent.priority, Priority::High);
        assert_eq!(intent.objectives.len(), 1);
        assert_eq!(intent.constraints.len(), 1);
    }
}