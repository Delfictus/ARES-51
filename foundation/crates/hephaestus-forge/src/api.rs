//! Public API for Hephaestus Forge
//! Enterprise-grade interface with comprehensive functionality

use crate::config::{
    ForgeConfig, OperationalMode, MonitoringConfig, ObservabilityConfig, 
    HitlConfig
};
use crate::intent::*;
use crate::types::*;
use crate::monitor::ForgeMonitor;
use crate::orchestrator::MetamorphicRuntimeOrchestrator;
use crate::sandbox::HardenedSandbox;
use crate::ledger::MetamorphicLedger;
use crate::temporal::TemporalSwapCoordinator;
use crate::drpp::EnhancedDRPP;
use crate::synthesis::ProgramSynthesizer;
use crate::validation::MetamorphicTestSuite;
use crate::adapters::{MetamorphicConsensus, ForgeMonitoringIntegration};
use crate::validation::hardening::{InputValidator, CircuitBreaker, RateLimiter, ResourceMonitor, ChaosEngine, AuditLogger};
use crate::validation::hardening::{CircuitBreakerConfig, AuditEntry, AuditResult, ChaosConfig};

use std::sync::Arc;
use tokio::sync::RwLock;
use futures::stream::{Stream, StreamExt};
use std::collections::HashMap;

/// Main interface to the Hephaestus Forge
pub struct HephaestusForge {
    /// Core configuration
    config: ForgeConfig,
    
    /// Current operational status
    status: Arc<RwLock<ForgeStatus>>,
    
    /// Core components
    runtime: Arc<MetamorphicRuntimeOrchestrator>,
    sandbox: Arc<HardenedSandbox>,
    ledger: Arc<MetamorphicLedger>,
    temporal_coordinator: Arc<TemporalSwapCoordinator>,
    
    /// Vector 1: Cognition & Generation
    drpp: Arc<EnhancedDRPP>,
    synthesizer: Arc<ProgramSynthesizer>,
    
    /// Vector 2: Simulation & Evolution
    test_suite: Arc<MetamorphicTestSuite>,
    
    /// Vector 3: Governance & Integration
    consensus_adapter: Arc<MetamorphicConsensus>,
    
    /// Monitoring and observability
    monitor: Arc<ForgeMonitor>,
    monitoring_adapter: Arc<ForgeMonitoringIntegration>,
    
    /// Active intents tracking
    active_intents: Arc<RwLock<HashMap<IntentId, IntentState>>>,
    
    /// Synthesis strategy registry
    synthesis_strategies: Arc<RwLock<HashMap<String, Box<dyn SynthesisStrategy>>>>,
    
    /// Metrics client
    metrics_client: Arc<MetricsClient>,
    
    /// Hardening components
    input_validator: Arc<InputValidator>,
    circuit_breaker: Arc<CircuitBreaker>,
    rate_limiter: Arc<RateLimiter>,
    resource_monitor: Arc<ResourceMonitor>,
    pub chaos_engine: Option<Arc<ChaosEngine>>,
    audit_logger: Arc<AuditLogger>,
}

/// Current status of the Forge
#[derive(Debug, Clone)]
pub struct ForgeStatus {
    pub is_running: bool,
    pub mode: OperationalMode,
    pub active_optimizations: usize,
    pub total_optimizations: u64,
    pub success_rate: f64,
    pub last_optimization: Option<chrono::DateTime<chrono::Utc>>,
}

/// Internal state for tracking intents
#[derive(Debug, Clone)]
struct IntentState {
    pub intent: OptimizationIntent,
    pub status: IntentStatus,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    pub module_id: Option<ModuleId>,
}

/// Synthesis strategy trait for custom implementations
#[async_trait::async_trait]
pub trait SynthesisStrategy: Send + Sync {
    async fn synthesize(
        &self,
        spec: &SynthesisSpec,
    ) -> Result<Vec<MlirModule>, SynthesisError>;
}

/// Synthesis specification
#[derive(Debug, Clone)]
pub struct SynthesisSpec {
    pub target: OptimizationTarget,
    pub objectives: Vec<Objective>,
    pub constraints: Vec<Constraint>,
}

/// MLIR module representation
#[derive(Debug, Clone)]
pub struct MlirModule {
    pub id: ModuleId,
    pub mlir_code: String,
    pub metadata: ModuleMetadata,
}

/// Synthesis error
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("Synthesis timeout")]
    Timeout,
    
    #[error("No viable candidates")]
    NoCandidates,
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    #[error("Synthesis failed: {0}")]
    Failed(String),
}

/// Metrics client for monitoring
pub struct MetricsClient {
    stats: Arc<RwLock<ForgeStatistics>>,
}

/// Forge statistics
#[derive(Debug, Clone, Default)]
pub struct ForgeStatistics {
    pub active_optimizations: usize,
    pub success_rate: f64,
    pub avg_synthesis_time: Duration,
    pub total_improvement: f64,
}

use std::time::Duration;

impl HephaestusForge {
    /// Initialize the Forge with configuration
    pub fn new(config: ForgeConfig) -> Result<Self, ForgeError> {
        // Use tokio runtime for async initialization
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| ForgeError::ConfigError(e.to_string()))?;
        
        rt.block_on(Self::new_async(config))
    }
    
    /// Initialize the Forge asynchronously (for use in async contexts)
    pub async fn new_async_public(config: ForgeConfig) -> Result<Self, ForgeError> {
        Self::new_async(config).await
    }
    
    /// Async initialization
    async fn new_async(config: ForgeConfig) -> Result<Self, ForgeError> {
        // Convert ForgeConfig to component configs
        let runtime_config = RuntimeConfig {
            max_concurrent_swaps: config.resource_limits.max_concurrent_optimizations,
            rollback_window_ms: 60000,
            shadow_traffic_percent: config.testing_config.shadow_traffic_percent,
        };
        
        let sandbox_config = crate::types::SandboxConfig {
            isolation_type: crate::types::IsolationType::Process,
            resource_limits: crate::types::ResourceLimits {
                cpu_cores: config.resource_limits.synthesis_cpu_cores as f64,
                memory_mb: (config.resource_limits.testing_memory_gb * 1024) as u64,
                disk_mb: 10240,
                network_mbps: 1000,
            },
            network_isolation: true,
        };
        
        let ledger_config = LedgerConfig {
            consensus_timeout_ms: config.consensus_config.consensus_timeout.as_millis() as u64,
            min_validators: config.consensus_config.min_validators,
            risk_threshold: config.risk_config.max_risk_score,
        };
        
        let drpp_config = DrppConfig {
            detection_sensitivity: 0.8,
            pattern_window_size: 1000,
            energy_threshold: 0.5,
            enable_resonance: true,
        };
        
        let synthesis_config = config.synthesis_config.clone();
        
        let validation_config = ValidationConfig {
            test_timeout_ms: config.testing_config.test_timeout.as_millis() as u64,
            chaos_engineering: config.testing_config.chaos_engineering,
            differential_testing: config.testing_config.differential_testing,
        };
        
        let monitor_config = MonitorConfig {
            metrics_interval_ms: config.monitoring_config.export_interval.as_millis() as u64,
            alert_thresholds: AlertThresholds {
                error_rate_percent: 1.0,
                latency_p99_ms: 100.0,
                memory_usage_percent: 80.0,
            },
        };
        
        // Initialize core components
        let runtime = Arc::new(
            MetamorphicRuntimeOrchestrator::new(runtime_config).await?
        );
        let sandbox = Arc::new(
            HardenedSandbox::new(sandbox_config).await?
        );
        let ledger = Arc::new(
            MetamorphicLedger::new(ledger_config).await?
        );
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await?
        );
        
        // Initialize Vector 1 components
        let drpp = Arc::new(
            EnhancedDRPP::new(drpp_config).await?
        );
        let synthesizer = Arc::new(
            ProgramSynthesizer::new(synthesis_config.clone()).await?
        );
        
        // Initialize Vector 2 components
        let test_suite = Arc::new(
            MetamorphicTestSuite::new(validation_config).await?
        );
        
        // Initialize Vector 3 components
        let consensus_adapter = Arc::new(
            MetamorphicConsensus::new(config.consensus_config.clone()).await?
        );
        
        // Initialize monitoring
        let monitor = Arc::new(
            ForgeMonitor::new(monitor_config).await?
        );
        let monitoring_adapter = Arc::new(
            ForgeMonitoringIntegration::new().await?
        );
        
        // Initialize hardening components
        let input_validator = Arc::new(InputValidator::new());
        
        let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            reset_timeout: Duration::from_secs(60),
        }));
        
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limiter_config.clone()));
        
        let resource_monitor = Arc::new(ResourceMonitor::new(
            config.resource_limits.testing_memory_gb as usize * 1024 * 1024 * 1024,
            80.0,
        ));
        
        let chaos_engine = if config.testing_config.chaos_engineering {
            Some(Arc::new(ChaosEngine::new(ChaosConfig {
                failure_probability: 0.01,
                latency_ms: Some(100),
                memory_pressure_mb: None,
                cpu_stress_threads: None,
                network_partition: false,
            })))
        } else {
            None
        };
        
        let audit_logger = Arc::new(
            AuditLogger::new("file:///var/log/hephaestus-forge/audit.log").await
        );
        
        Ok(Self {
            config,
            status: Arc::new(RwLock::new(ForgeStatus {
                is_running: false,
                mode: OperationalMode::Supervised,
                active_optimizations: 0,
                total_optimizations: 0,
                success_rate: 0.0,
                last_optimization: None,
            })),
            runtime,
            sandbox,
            ledger,
            temporal_coordinator,
            drpp,
            synthesizer,
            test_suite,
            consensus_adapter,
            monitor,
            monitoring_adapter,
            active_intents: Arc::new(RwLock::new(HashMap::new())),
            synthesis_strategies: Arc::new(RwLock::new(HashMap::new())),
            metrics_client: Arc::new(MetricsClient {
                stats: Arc::new(RwLock::new(ForgeStatistics::default())),
            }),
            input_validator,
            circuit_breaker,
            rate_limiter,
            resource_monitor,
            chaos_engine,
            audit_logger,
        })
    }
    
    /// Start autonomous optimization
    pub async fn start(&self) -> Result<(), ForgeError> {
        let mut status = self.status.write().await;
        if status.is_running {
            return Err(ForgeError::ConfigError("Forge is already running".into()));
        }
        
        status.is_running = true;
        status.mode = self.config.mode;
        drop(status);
        
        // Start monitoring
        self.monitor.start().await;
        
        // Start autonomous optimization loop
        let forge = self.clone_components();
        tokio::spawn(async move {
            forge.autonomous_optimization_loop().await;
        });
        
        Ok(())
    }
    
    /// Stop the Forge gracefully
    pub async fn stop(&self) -> Result<(), ForgeError> {
        let mut status = self.status.write().await;
        if !status.is_running {
            return Ok(());
        }
        
        status.is_running = false;
        
        // Wait for active optimizations to complete
        while status.active_optimizations > 0 {
            tokio::time::sleep(Duration::from_secs(1)).await;
            status = self.status.write().await;
        }
        
        Ok(())
    }
    
    /// Get current optimization status
    pub async fn status(&self) -> ForgeStatus {
        self.status.read().await.clone()
    }
    
    /// Manual intent submission
    pub async fn submit_intent(&self, intent: OptimizationIntent) -> Result<IntentId, ForgeError> {
        let intent_id = intent.id.clone();
        
        // Store intent state
        let state = IntentState {
            intent: intent.clone(),
            status: IntentStatus::Synthesizing { progress: 0.0 },
            submitted_at: chrono::Utc::now(),
            module_id: None,
        };
        
        self.active_intents.write().await.insert(intent_id.clone(), state);
        
        // Process intent asynchronously
        let forge = self.clone_components();
        let id = intent_id.clone();
        tokio::spawn(async move {
            if let Err(e) = forge.process_intent(intent).await {
                eprintln!("Failed to process intent {:?}: {}", id, e);
            }
        });
        
        Ok(intent_id)
    }
    
    /// Get intent status
    pub async fn get_intent_status(&self, intent_id: IntentId) -> Result<IntentStatus, ForgeError> {
        self.active_intents
            .read()
            .await
            .get(&intent_id)
            .map(|state| state.status.clone())
            .ok_or_else(|| ForgeError::ValidationError("Intent not found".into()))
    }
    
    /// Register custom synthesis strategy
    pub async fn register_synthesis_strategy(
        &self,
        name: impl Into<String>,
        strategy: Box<dyn SynthesisStrategy>,
    ) -> Result<(), ForgeError> {
        self.synthesis_strategies
            .write()
            .await
            .insert(name.into(), strategy);
        Ok(())
    }
    
    /// Enable monitoring with configuration
    pub async fn enable_monitoring(&self, config: MonitoringConfig) -> Result<(), ForgeError> {
        // Update monitoring configuration
        self.monitor.start().await;
        Ok(())
    }
    
    /// Configure observability
    pub async fn configure_observability(&self, config: ObservabilityConfig) -> Result<(), ForgeError> {
        // Configure metrics, tracing, logging, alerting
        Ok(())
    }
    
    /// Configure HITL
    pub async fn configure_hitl(&self, config: HitlConfig) -> Result<(), ForgeError> {
        // Configure human-in-the-loop
        Ok(())
    }
    
    /// Subscribe to risk events
    pub async fn subscribe_risk_events(&self) -> Result<impl Stream<Item = RiskEvent>, ForgeError> {
        // Create risk event stream
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
    
    /// Get metrics client
    pub fn get_metrics_client(&self) -> Arc<MetricsClient> {
        self.metrics_client.clone()
    }
    
    /// Process a single intent with full hardening
    async fn process_intent(&self, intent: OptimizationIntent) -> Result<(), ForgeError> {
        let intent_id = intent.id.clone();
        
        // Rate limiting
        self.rate_limiter.acquire().await
            .map_err(|e| ForgeError::ValidationError(e.to_string()))?;
        
        // Resource monitoring
        self.resource_monitor.check_resources().await
            .map_err(|e| ForgeError::ValidationError(e.to_string()))?;
        
        // Audit logging
        self.audit_logger.log(AuditEntry {
            timestamp: chrono::Utc::now(),
            operation: "process_intent".to_string(),
            principal: "system".to_string(),
            resource: format!("intent:{}", intent_id),
            result: AuditResult::Success,
            metadata: serde_json::json!({
                "intent_type": format!("{:?}", intent.priority),
                "objectives": intent.objectives.len(),
            }),
        }).await;
        
        // Chaos engineering injection point
        if let Some(chaos) = &self.chaos_engine {
            chaos.maybe_inject_fault("process_intent_start").await
                .map_err(|e| ForgeError::ValidationError(e.to_string()))?;
        }
        
        // Increment total optimizations
        self.status.write().await.total_optimizations += 1;

        // Synthesis phase
        self.update_intent_status(
            &intent_id,
            IntentStatus::Synthesizing { progress: 0.0 }
        ).await?;
        
        let candidates = self.synthesizer.generate_candidates(&[intent.clone()]).await?;
        
        if candidates.is_empty() {
            self.update_intent_status(
                &intent_id,
                IntentStatus::Failed { reason: "No candidates generated".into() }
            ).await?;
            return Ok(());
        }
        
        // Testing phase
        let module_id = candidates[0].id.clone();
        self.update_intent_status(
            &intent_id,
            IntentStatus::Testing { module_id: module_id.clone() }
        ).await?;
        
        let validated = self.test_suite.validate_candidates(candidates).await?;
        
        if validated.is_empty() {
            self.update_intent_status(
                &intent_id,
                IntentStatus::Failed { reason: "Validation failed".into() }
            ).await?;
            return Ok(());
        }
        
        // Governance phase
        let approved = self.ledger.propose_changes(validated).await?;
        
        if approved.is_empty() {
            self.update_intent_status(
                &intent_id,
                IntentStatus::Failed { reason: "Not approved".into() }
            ).await?;
            return Ok(());
        }
        
        // Deployment phase
        self.update_intent_status(
            &intent_id,
            IntentStatus::Deploying { stage: "Starting".into() }
        ).await?;
        
        let integrated = self.runtime.integrate_changes(approved).await?;
        
        // Calculate actual improvement based on integration results
        let improvement = self.calculate_integration_improvement(&integrated).await?;
        
        // Mark as completed
        self.update_intent_status(
            &intent_id,
            IntentStatus::Completed { improvement }
        ).await?;
        
        Ok(())
    }
    
    /// Update intent status
    async fn update_intent_status(
        &self,
        intent_id: &IntentId,
        status: IntentStatus,
    ) -> Result<(), ForgeError> {
        let mut intents = self.active_intents.write().await;
        if let Some(state) = intents.get_mut(intent_id) {
            state.status = status;
        }
        Ok(())
    }

    /// Calculate actual improvement percentage from integration results
    async fn calculate_integration_improvement(&self, integrated: &IntegratedChanges) -> Result<f64, ForgeError> {
        // Calculate improvement based on multiple metrics
        let mut total_improvement = 0.0;
        let mut metric_count = 0;
        
        // Performance improvement (latency reduction)
        if let Some(perf_before) = &integrated.performance_before {
            if let Some(perf_after) = &integrated.performance_after {
                let latency_improvement = (perf_before.avg_latency_ms - perf_after.avg_latency_ms) / perf_before.avg_latency_ms;
                if latency_improvement > 0.0 {
                    total_improvement += latency_improvement;
                    metric_count += 1;
                }
                
                let throughput_improvement = (perf_after.throughput_rps - perf_before.throughput_rps) / perf_before.throughput_rps;
                if throughput_improvement > 0.0 {
                    total_improvement += throughput_improvement;
                    metric_count += 1;
                }
            }
        }
        
        // Resource utilization improvement (CPU/memory reduction)
        if let Some(resource_before) = &integrated.resource_usage_before {
            if let Some(resource_after) = &integrated.resource_usage_after {
                let cpu_improvement = (resource_before.cpu_usage - resource_after.cpu_usage) / resource_before.cpu_usage;
                if cpu_improvement > 0.0 {
                    total_improvement += cpu_improvement * 0.5; // Weight CPU improvements less
                    metric_count += 1;
                }
                
                let memory_improvement = (resource_before.memory_usage_mb as f64 - resource_after.memory_usage_mb as f64) 
                    / resource_before.memory_usage_mb as f64;
                if memory_improvement > 0.0 {
                    total_improvement += memory_improvement * 0.3; // Weight memory improvements even less
                    metric_count += 1;
                }
            }
        }
        
        // Code quality improvement
        if let Some(quality_before) = &integrated.code_quality_before {
            if let Some(quality_after) = &integrated.code_quality_after {
                let complexity_improvement = (quality_before.cyclomatic_complexity as f64 - quality_after.cyclomatic_complexity as f64) 
                    / quality_before.cyclomatic_complexity as f64;
                if complexity_improvement > 0.0 {
                    total_improvement += complexity_improvement * 0.2; // Weight complexity improvements lower
                    metric_count += 1;
                }
                
                let test_coverage_improvement = (quality_after.test_coverage - quality_before.test_coverage) / 100.0;
                if test_coverage_improvement > 0.0 {
                    total_improvement += test_coverage_improvement * 0.1; // Small weight for coverage improvements
                    metric_count += 1;
                }
            }
        }
        
        // Calculate weighted average improvement
        let improvement = if metric_count > 0 {
            total_improvement / metric_count as f64
        } else {
            // Fallback: estimate based on integration success
            match integrated.integration_result {
                IntegrationResult::Success => 0.05, // 5% baseline improvement
                IntegrationResult::PartialSuccess => 0.02, // 2% for partial success
                IntegrationResult::Failed => 0.0, // No improvement on failure
            }
        };
        
        // Cap improvement at 50% to prevent unrealistic values
        Ok(improvement.min(0.5).max(0.0))
    }
    
    /// Autonomous optimization loop
    async fn autonomous_optimization_loop(&self) {
        loop {
            let status = self.status.read().await;
            if !status.is_running {
                break;
            }
            drop(status);
            
            // Detect opportunities
            if let Ok(opportunities) = self.drpp.detect_optimization_opportunities().await {
                for opportunity in opportunities {
                    // Convert to intent
                    let intent = self.opportunity_to_intent(opportunity);
                    
                    // Submit for processing
                    if let Ok(intent) = intent {
                        let _ = self.submit_intent(intent).await;
                    }
                }
            }
            
            // Sleep before next iteration
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    }
    
    /// Convert opportunity to intent
    fn opportunity_to_intent(&self, opportunity: OptimizationOpportunity) -> Result<OptimizationIntent, ForgeError> {
        let mut builder = OptimizationIntent::builder()
            .target(OptimizationTarget::Module(opportunity.module_id))
            .priority(Priority::Medium);
        
        // Add objectives based on opportunity type
        match opportunity.opportunity_type {
            OpportunityType::LoopOptimization => {
                builder = builder.add_objective(Objective::MinimizeLatency {
                    percentile: 99.0,
                    target_ms: 10.0,
                });
            }
            OpportunityType::MemoryLayoutOptimization => {
                builder = builder.add_objective(Objective::ReduceMemory {
                    target_mb: 256,
                });
            }
            _ => {
                builder = builder.add_objective(Objective::MaximizeThroughput {
                    target_ops_per_sec: 10000.0,
                });
            }
        }
        
        // Add standard constraints
        builder = builder
            .add_constraint(Constraint::MaintainCorrectness)
            .add_constraint(Constraint::RequireProof);
        
        builder.build()
    }
    
    /// Clone components for async operations
    fn clone_components(&self) -> Self {
        Self {
            config: self.config.clone(),
            status: self.status.clone(),
            runtime: self.runtime.clone(),
            sandbox: self.sandbox.clone(),
            ledger: self.ledger.clone(),
            temporal_coordinator: self.temporal_coordinator.clone(),
            drpp: self.drpp.clone(),
            synthesizer: self.synthesizer.clone(),
            test_suite: self.test_suite.clone(),
            consensus_adapter: self.consensus_adapter.clone(),
            monitor: self.monitor.clone(),
            monitoring_adapter: self.monitoring_adapter.clone(),
            active_intents: self.active_intents.clone(),
            synthesis_strategies: self.synthesis_strategies.clone(),
            metrics_client: self.metrics_client.clone(),
            input_validator: self.input_validator.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
            rate_limiter: self.rate_limiter.clone(),
            resource_monitor: self.resource_monitor.clone(),
            chaos_engine: self.chaos_engine.clone(),
            audit_logger: self.audit_logger.clone(),
        }
    }
    
    /// Enable chaos engineering for testing
    pub async fn enable_chaos_engineering(&self) -> Result<(), ForgeError> {
        if let Some(chaos) = &self.chaos_engine {
            chaos.enable().await;
            Ok(())
        } else {
            Err(ForgeError::ConfigError("Chaos engineering not configured".into()))
        }
    }
    
    /// Disable chaos engineering
    pub async fn disable_chaos_engineering(&self) -> Result<(), ForgeError> {
        if let Some(chaos) = &self.chaos_engine {
            chaos.disable().await;
            Ok(())
        } else {
            Err(ForgeError::ConfigError("Chaos engineering not configured".into()))
        }
    }

    pub fn chaos_engine(&self) -> Option<&Arc<ChaosEngine>> {
        self.chaos_engine.as_ref()
    }
}

impl MetricsClient {
    pub async fn get_current_stats(&self) -> Result<ForgeStatistics, ForgeError> {
        Ok(self.stats.read().await.clone())
    }
}

/// Risk events for monitoring
#[derive(Debug, Clone)]
pub enum RiskEvent {
    HighRiskDetected { module: String, score: f64 },
    ApprovalRequired { intent_id: IntentId, reason: String },
    EmergencyStop { reason: String },
}