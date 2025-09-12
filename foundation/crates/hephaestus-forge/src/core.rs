// Core Hephaestus Forge Implementation with Phase Lattice Integration

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;

use crate::{
    ForgeError, ForgeResult, IntentId, ModuleId, Priority,
    drpp::{DRPPEngine, ResonancePattern},
    adp::AdaptiveDissipativeProcessor,
    synthesis::{SynthesisEngine, SynthesizedModule},
    sandbox::HardenedSandbox,
    ledger::{MetamorphicLedger, ConsensusDecision},
    orchestrator::{RuntimeOrchestrator, DeploymentStrategy},
    monitoring::ForgeMetrics,
    storage::{IntentStorage, StorageConfig},
    intent::OptimizationIntent,
};

/// Risk assessment score with detailed breakdown
#[derive(Debug, Clone)]
pub struct RiskScore {
    pub value: f64,
    pub factors: Vec<RiskFactor>,
    pub explanation: String,
}

/// Individual risk factors contributing to overall risk
#[derive(Debug, Clone)]
pub enum RiskFactor {
    Complexity(f64),
    Criticality(f64),
    Confidence(f64),
    SecurityImpact(f64),
    PerformanceImpact(f64),
    SystemIntegrationRisk(f64),
}

/// Synthesized module with complete metadata for risk assessment
#[derive(Debug, Clone)]
pub struct SynthesizedModule {
    pub id: ModuleId,
    pub complexity_score: f64,
    pub metadata: ModuleMetadata,
    pub code: Vec<u8>,
    pub dependencies: Vec<ModuleId>,
    pub security_classification: SecurityClassification,
    pub performance_characteristics: PerformanceCharacteristics,
    pub integration_points: Vec<IntegrationPoint>,
}

/// Security classification levels
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityClassification {
    Public,
    Internal,
    Confidential,
    Critical,
}

/// Performance characteristics for risk assessment
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    pub cpu_intensive: bool,
    pub memory_intensive: bool,
    pub io_intensive: bool,
    pub network_intensive: bool,
    pub real_time_constraints: bool,
}

/// Integration points that affect criticality
#[derive(Debug, Clone)]
pub struct IntegrationPoint {
    pub target_system: String,
    pub interaction_type: InteractionType,
    pub criticality_level: CriticalityLevel,
}

#[derive(Debug, Clone)]
pub enum InteractionType {
    Database,
    ExternalService,
    CoreSystem,
    UserInterface,
    SecuritySystem,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum CriticalityLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Enhanced module metadata with criticality assessment
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub module_type: ModuleType,
    pub business_domain: BusinessDomain,
    pub error_handling_completeness: f64,
    pub test_coverage: f64,
    pub documentation_completeness: f64,
    pub code_quality_metrics: CodeQualityMetrics,
    pub operational_history: OperationalHistory,
}

#[derive(Debug, Clone)]
pub enum ModuleType {
    CoreLogic,
    DataAccess,
    UserInterface,
    Integration,
    Security,
    Monitoring,
    Configuration,
}

#[derive(Debug, Clone)]
pub enum BusinessDomain {
    Finance,
    UserManagement,
    DataProcessing,
    Security,
    Infrastructure,
    Analytics,
    Communication,
}

#[derive(Debug, Clone)]
pub struct CodeQualityMetrics {
    pub cyclomatic_complexity: f64,
    pub maintainability_index: f64,
    pub technical_debt_ratio: f64,
    pub code_duplication: f64,
}

#[derive(Debug, Clone)]
pub struct OperationalHistory {
    pub deployment_frequency: f64,
    pub failure_rate: f64,
    pub recovery_time: Duration,
    pub performance_degradation_incidents: u32,
    pub security_incidents: u32,
}

/// Risk calculation weights for enterprise risk assessment
#[derive(Debug, Clone)]
struct RiskWeights {
    pub complexity: f64,
    pub criticality: f64,
    pub confidence: f64,
    pub security_impact: f64,
    pub performance_impact: f64,
    pub integration_risk: f64,
}

/// Main Hephaestus Forge orchestrator
pub struct HephaestusForge {
    /// Configuration
    config: Arc<ForgeConfig>,
    
    /// Vector 1: Cognition & Generation
    drpp_engine: Arc<RwLock<DRPPEngine>>,
    synthesis_engine: Arc<SynthesisEngine>,
    
    /// Vector 2: Simulation & Evolution  
    sandbox: Arc<HardenedSandbox>,
    chaos_engine: Option<Arc<crate::chaos::ChaosEngine>>,
    
    /// Vector 3: Governance & Integration
    ledger: Arc<MetamorphicLedger>,
    orchestrator: Arc<RuntimeOrchestrator>,
    
    /// Energy management via ADP
    adp_processor: Arc<RwLock<AdaptiveDissipativeProcessor>>,
    
    /// State management
    state: Arc<RwLock<ForgeState>>,
    active_intents: Arc<DashMap<IntentId, IntentState>>,
    
    /// High-performance persistent intent storage
    intent_storage: Arc<IntentStorage>,
    
    /// Metrics and monitoring
    metrics: Arc<ForgeMetrics>,
    
    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<tokio::sync::broadcast::Sender<()>>>>,
}

/// Forge configuration
#[derive(Debug, Clone)]
pub struct ForgeConfig {
    /// Operational mode
    pub mode: OperationalMode,
    
    /// Risk management
    pub risk_tolerance: f64,
    pub max_risk_score: f64,
    pub require_proof: bool,
    
    /// Resource limits
    pub max_concurrent_optimizations: usize,
    pub synthesis_timeout: Duration,
    pub validation_timeout: Duration,
    pub max_memory_gb: usize,
    pub max_cpu_percent: f64,
    
    /// Phase Lattice parameters
    pub lattice_dimensions: (usize, usize, usize),
    pub resonance_threshold: f64,
    pub dissipation_rate: f64,
    
    /// Consensus configuration
    pub consensus_nodes: Vec<String>,
    pub consensus_timeout: Duration,
    pub min_consensus_nodes: usize,
    
    /// Monitoring
    pub metrics_port: u16,
    pub dashboard_port: u16,
    pub enable_tracing: bool,
    pub telemetry_endpoint: Option<String>,
    
    /// Safety
    pub sandbox_memory_mb: usize,
    pub enable_chaos_testing: bool,
    pub rollback_on_regression: bool,
}

impl Default for ForgeConfig {
    fn default() -> Self {
        Self {
            mode: OperationalMode::Supervised,
            risk_tolerance: 0.5,
            max_risk_score: 0.8,
            require_proof: true,
            max_concurrent_optimizations: 4,
            synthesis_timeout: Duration::from_secs(300),
            validation_timeout: Duration::from_secs(600),
            max_memory_gb: 32,
            max_cpu_percent: 80.0,
            lattice_dimensions: (10, 10, 10),
            resonance_threshold: 0.7,
            dissipation_rate: 0.1,
            consensus_nodes: vec![],
            consensus_timeout: Duration::from_secs(30),
            min_consensus_nodes: 3,
            metrics_port: 9090,
            dashboard_port: 8080,
            enable_tracing: true,
            telemetry_endpoint: None,
            sandbox_memory_mb: 2048,
            enable_chaos_testing: false,
            rollback_on_regression: true,
        }
    }
}

/// Operational modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalMode {
    /// Forge is disabled
    Disabled,
    /// Requires human approval for all changes
    Manual,
    /// Notifies humans but can proceed
    Supervised,
    /// Fully autonomous operation
    Autonomous,
    /// Emergency mode - critical fixes only
    Emergency,
}

/// Forge state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgeState {
    /// Not yet initialized
    Uninitialized,
    /// Initialized but not running
    Initialized,
    /// Starting up
    Starting,
    /// Running normally
    Running,
    /// Degraded performance
    Degraded,
    /// Stopping
    Stopping,
    /// Stopped
    Stopped,
    /// Error state
    Error,
}

/// State of an optimization intent
#[derive(Debug, Clone)]
pub struct IntentState {
    pub intent_id: IntentId,
    pub status: IntentStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub module_id: Option<ModuleId>,
    pub risk_score: Option<RiskScore>,
    pub performance_gain: Option<f64>,
    pub error: Option<String>,
}

/// Intent processing status
#[derive(Debug, Clone, PartialEq)]
pub enum IntentStatus {
    /// Queued for processing
    Queued,
    /// Analyzing optimization opportunity
    Analyzing { progress: f64 },
    /// Synthesizing solution
    Synthesizing { progress: f64 },
    /// Validating in sandbox
    Validating { test_suite: String },
    /// Awaiting consensus
    AwaitingConsensus { votes: usize, required: usize },
    /// Awaiting human approval
    AwaitingApproval { risk_score: f64, reason: String },
    /// Deploying to production
    Deploying { strategy: String, stage: String },
    /// Successfully completed
    Completed { improvement: f64, deployed_at: chrono::DateTime<chrono::Utc> },
    /// Failed
    Failed { reason: String, failed_at: chrono::DateTime<chrono::Utc> },
    /// Rolled back
    RolledBack { reason: String, rolled_back_at: chrono::DateTime<chrono::Utc> },
}

impl HephaestusForge {
    /// Create new Forge instance (async version with storage)
    pub async fn new_async(config: ForgeConfig) -> ForgeResult<Self> {
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
        
        // Initialize Phase Lattice based DRPP engine
        let drpp_engine = Arc::new(RwLock::new(
            DRPPEngine::new(
                config.lattice_dimensions,
                config.resonance_threshold,
            )?
        ));
        
        // Initialize ADP for energy management
        let adp_processor = Arc::new(RwLock::new(
            AdaptiveDissipativeProcessor::new(
                config.dissipation_rate,
                config.max_memory_gb,
            )?
        ));
        
        // Initialize synthesis engine
        let synthesis_engine = Arc::new(SynthesisEngine::new(
            config.synthesis_timeout,
            config.require_proof,
        )?);
        
        // Initialize sandbox
        let sandbox = Arc::new(HardenedSandbox::new(
            config.sandbox_memory_mb,
            config.validation_timeout,
        )?);
        
        // Initialize chaos engine if enabled
        let chaos_engine = if config.enable_chaos_testing {
            Some(Arc::new(crate::chaos::ChaosEngine::new()?))
        } else {
            None
        };
        
        // Initialize ledger
        let ledger = Arc::new(MetamorphicLedger::new(
            config.consensus_nodes.clone(),
            config.consensus_timeout,
            config.min_consensus_nodes,
        )?);
        
        // Initialize orchestrator
        let orchestrator = Arc::new(RuntimeOrchestrator::new(
            config.rollback_on_regression,
        )?);
        
        // Initialize metrics
        let metrics = Arc::new(ForgeMetrics::new(
            config.metrics_port,
            config.enable_tracing,
        )?);
        
        // Initialize high-performance persistent intent storage
        let storage_config = StorageConfig {
            db_path: format!("./data/forge_intents_{}", std::process::id()),
            max_concurrent_transactions: config.max_concurrent_optimizations * 2,
            cache_size_mb: (config.max_memory_gb * 1024) / 4, // Use 1/4 of memory for cache
            ..Default::default()
        };
        
        let intent_storage = Arc::new(IntentStorage::new(storage_config).await
            .map_err(|e| ForgeError::Generic(anyhow::anyhow!("Failed to initialize intent storage: {}", e)))?);
        
        Ok(Self {
            config: Arc::new(config),
            drpp_engine,
            synthesis_engine,
            sandbox,
            chaos_engine,
            ledger,
            orchestrator,
            adp_processor,
            state: Arc::new(RwLock::new(ForgeState::Initialized)),
            active_intents: Arc::new(DashMap::new()),
            intent_storage,
            metrics,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
        })
    }
    
    /// Create new Forge instance (sync version - blocks on async initialization)
    pub fn new(config: ForgeConfig) -> ForgeResult<Self> {
        tokio::task::block_in_place(|| {
            let runtime = tokio::runtime::Handle::try_current()
                .or_else(|_| {
                    // Create a new runtime if none exists
                    tokio::runtime::Runtime::new()
                        .map(|rt| {
                            let handle = rt.handle().clone();
                            std::mem::forget(rt); // Keep runtime alive
                            handle
                        })
                        .map_err(|e| ForgeError::Generic(anyhow::anyhow!("Failed to create async runtime: {}", e)))
                })?;
                
            runtime.block_on(Self::new_async(config))
        })
    }
    
    /// Start the Forge
    pub async fn start(&self) -> ForgeResult<()> {
        // Update state
        {
            let mut state = self.state.write().await;
            if *state != ForgeState::Initialized {
                return Err(ForgeError::Generic(anyhow::anyhow!(
                    "Cannot start Forge from state: {:?}", *state
                )));
            }
            *state = ForgeState::Starting;
        }
        
        tracing::info!("Starting Hephaestus Forge in {:?} mode", self.config.mode);
        
        // Start monitoring
        self.metrics.start().await?;
        
        // Start DRPP monitoring loop
        self.start_drpp_monitoring().await?;
        
        // Start ADP balancing loop
        self.start_adp_balancing().await?;
        
        // Start intent processing loop
        self.start_intent_processing().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = ForgeState::Running;
        }
        
        tracing::info!("Hephaestus Forge started successfully");
        Ok(())
    }
    
    /// Stop the Forge gracefully
    pub async fn stop(&self) -> ForgeResult<()> {
        tracing::info!("Stopping Hephaestus Forge");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = ForgeState::Stopping;
        }
        
        // Send shutdown signal
        if let Some(tx) = self.shutdown_tx.lock().as_ref() {
            let _ = tx.send(());
        }
        
        // Wait for active intents to complete or timeout
        let timeout = Duration::from_secs(30);
        let start = std::time::Instant::now();
        
        while !self.active_intents.is_empty() && start.elapsed() < timeout {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        // Force-fail remaining intents
        for mut entry in self.active_intents.iter_mut() {
            entry.status = IntentStatus::Failed {
                reason: "Forge shutdown".to_string(),
                failed_at: chrono::Utc::now(),
            };
        }
        
        // Stop metrics
        self.metrics.stop().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = ForgeState::Stopped;
        }
        
        tracing::info!("Hephaestus Forge stopped");
        Ok(())
    }
    
    /// Start DRPP monitoring for optimization opportunities
    async fn start_drpp_monitoring(&self) -> ForgeResult<()> {
        let drpp = self.drpp_engine.clone();
        let active_intents = self.active_intents.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        
        let shutdown_rx = self.shutdown_tx.lock()
            .as_ref()
            .ok_or_else(|| ForgeError::Generic(anyhow::anyhow!("No shutdown channel")))?
            .subscribe();
        
        tokio::spawn(async move {
            let mut shutdown_rx = shutdown_rx;
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("DRPP monitoring loop shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        // Get resonance patterns from Phase Lattice
                        let patterns = {
                            let mut drpp = drpp.write().await;
                            match drpp.detect_patterns().await {
                                Ok(p) => p,
                                Err(e) => {
                                    tracing::error!("DRPP pattern detection failed: {}", e);
                                    continue;
                                }
                            }
                        };
                        
                        // Convert significant patterns to optimization intents
                        for pattern in patterns {
                            if pattern.resonance_strength > config.resonance_threshold {
                                // Create optimization intent from pattern
                                let intent = match Self::pattern_to_intent(pattern).await {
                                    Ok(i) => i,
                                    Err(e) => {
                                        tracing::error!("Failed to create intent from pattern: {}", e);
                                        continue;
                                    }
                                };
                                
                                // Queue intent if not at capacity
                                if active_intents.len() < config.max_concurrent_optimizations {
                                    let intent_state = IntentState {
                                        intent_id: intent.id.clone(),
                                        status: IntentStatus::Queued,
                                        created_at: chrono::Utc::now(),
                                        updated_at: chrono::Utc::now(),
                                        module_id: None,
                                        risk_score: None,
                                        performance_gain: None,
                                        error: None,
                                    };
                                    
                                    active_intents.insert(intent.id.clone(), intent_state);
                                    metrics.record_intent_created(&intent.id);
                                    
                                    tracing::info!("Created optimization intent {:?} from resonance pattern", intent.id);
                                }
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start ADP energy balancing loop
    async fn start_adp_balancing(&self) -> ForgeResult<()> {
        let adp = self.adp_processor.clone();
        let drpp = self.drpp_engine.clone();
        let metrics = self.metrics.clone();
        
        let shutdown_rx = self.shutdown_tx.lock()
            .as_ref()
            .ok_or_else(|| ForgeError::Generic(anyhow::anyhow!("No shutdown channel")))?
            .subscribe();
        
        tokio::spawn(async move {
            let mut shutdown_rx = shutdown_rx;
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("ADP balancing loop shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        // Get current energy distribution from Phase Lattice
                        let energy_state = {
                            let drpp = drpp.read().await;
                            drpp.get_energy_distribution().await
                        };
                        
                        // Perform dissipative balancing
                        let mut adp = adp.write().await;
                        if let Err(e) = adp.balance_energy(energy_state).await {
                            tracing::error!("ADP energy balancing failed: {}", e);
                            continue;
                        }
                        
                        // Record metrics
                        let entropy = adp.calculate_entropy();
                        metrics.record_system_entropy(entropy);
                        
                        // Check for system stability
                        if entropy > 0.9 {
                            tracing::warn!("System entropy high: {:.2}, may need intervention", entropy);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start intent processing loop
    async fn start_intent_processing(&self) -> ForgeResult<()> {
        let active_intents = self.active_intents.clone();
        let synthesis_engine = self.synthesis_engine.clone();
        let sandbox = self.sandbox.clone();
        let ledger = self.ledger.clone();
        let orchestrator = self.orchestrator.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let intent_storage = self.intent_storage.clone();
        
        let shutdown_rx = self.shutdown_tx.lock()
            .as_ref()
            .ok_or_else(|| ForgeError::Generic(anyhow::anyhow!("No shutdown channel")))?
            .subscribe();
        
        tokio::spawn(async move {
            let mut shutdown_rx = shutdown_rx;
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Intent processing loop shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        // Process queued intents
                        let queued_intents: Vec<IntentId> = active_intents
                            .iter()
                            .filter(|entry| matches!(entry.status, IntentStatus::Queued))
                            .map(|entry| entry.intent_id.clone())
                            .collect();
                        
                        for intent_id in queued_intents {
                            // Process intent through full pipeline
                            tokio::spawn({
                                let intent_id = intent_id.clone();
                                let active_intents = active_intents.clone();
                                let synthesis_engine = synthesis_engine.clone();
                                let sandbox = sandbox.clone();
                                let ledger = ledger.clone();
                                let orchestrator = orchestrator.clone();
                                let config = config.clone();
                                let metrics = metrics.clone();
                                let intent_storage = intent_storage.clone();
                                
                                async move {
                                    if let Err(e) = Self::process_intent(
                                        intent_id,
                                        active_intents,
                                        synthesis_engine,
                                        sandbox,
                                        ledger,
                                        orchestrator,
                                        config,
                                        metrics,
                                        intent_storage,
                                    ).await {
                                        tracing::error!("Intent processing failed: {}", e);
                                    }
                                }
                            });
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Process a single optimization intent through the full pipeline
    async fn process_intent(
        intent_id: IntentId,
        active_intents: Arc<DashMap<IntentId, IntentState>>,
        synthesis_engine: Arc<SynthesisEngine>,
        sandbox: Arc<HardenedSandbox>,
        ledger: Arc<MetamorphicLedger>,
        orchestrator: Arc<RuntimeOrchestrator>,
        config: Arc<ForgeConfig>,
        metrics: Arc<ForgeMetrics>,
        intent_storage: Arc<IntentStorage>,
    ) -> ForgeResult<()> {
        // Update status to analyzing
        active_intents.alter(&intent_id, |_, mut state| {
            state.status = IntentStatus::Analyzing { progress: 0.0 };
            state.updated_at = chrono::Utc::now();
            state
        });
        
        // Load actual intent from high-performance persistent storage
        let intent = match intent_storage.get_intent(&intent_id).await? {
            Some(intent) => {
                tracing::debug!("Successfully loaded intent {} from storage", intent_id);
                metrics.record_storage_hit(&intent_id);
                intent
            }
            None => {
                tracing::error!("Intent {} not found in storage", intent_id);
                metrics.record_storage_miss(&intent_id);
                
                // Update status to failed
                active_intents.alter(&intent_id, |_, mut state| {
                    state.status = IntentStatus::Failed {
                        reason: "Intent not found in storage".to_string(),
                        failed_at: chrono::Utc::now(),
                    };
                    state.updated_at = chrono::Utc::now();
                    state
                });
                
                return Err(ForgeError::NotFound(format!("Intent {} not found in storage", intent_id)));
            }
        };
        
        tracing::info!(
            "Processing intent {} with {} objectives, {} constraints, priority {:?}",
            intent.id,
            intent.objectives.len(),
            intent.constraints.len(),
            intent.priority
        );
        
        // Vector 1: Synthesis
        active_intents.alter(&intent_id, |_, mut state| {
            state.status = IntentStatus::Synthesizing { progress: 0.0 };
            state.updated_at = chrono::Utc::now();
            state
        });
        
        metrics.record_synthesis_started(&intent_id);
        let synthesized = synthesis_engine.synthesize(intent.clone()).await?;
        metrics.record_synthesis_completed(&intent_id, true);
        
        // Vector 2: Validation
        active_intents.alter(&intent_id, |_, mut state| {
            state.status = IntentStatus::Validating { 
                test_suite: "comprehensive".to_string() 
            };
            state.updated_at = chrono::Utc::now();
            state
        });
        
        metrics.record_validation_started(&intent_id);
        let validation = sandbox.validate(&synthesized).await?;
        metrics.record_validation_completed(&intent_id, validation.passed);
        
        if !validation.passed {
            active_intents.alter(&intent_id, |_, mut state| {
                state.status = IntentStatus::Failed {
                    reason: format!("Validation failed: {}", validation.reason),
                    failed_at: chrono::Utc::now(),
                };
                state
            });
            return Ok(());
        }
        
        // Calculate risk score
        let risk_score = Self::calculate_risk(&synthesized, &validation);
        
        // Check risk threshold
        if risk_score.value > config.max_risk_score {
            active_intents.alter(&intent_id, |_, mut state| {
                state.status = IntentStatus::Failed {
                    reason: format!("Risk too high: {:.2}", risk_score.value),
                    failed_at: chrono::Utc::now(),
                };
                state.risk_score = Some(risk_score);
                state
            });
            return Err(ForgeError::RiskThresholdExceeded {
                risk_score: risk_score.value,
                threshold: config.max_risk_score,
            });
        }
        
        // Vector 3: Consensus
        active_intents.alter(&intent_id, |_, mut state| {
            state.status = IntentStatus::AwaitingConsensus {
                votes: 0,
                required: config.min_consensus_nodes,
            };
            state.risk_score = Some(risk_score.clone());
            state.updated_at = chrono::Utc::now();
            state
        });
        
        let decision = ledger.propose_change(synthesized.clone(), validation, risk_score).await?;
        
        match decision {
            ConsensusDecision::Approved => {
                // Deploy
                active_intents.alter(&intent_id, |_, mut state| {
                    state.status = IntentStatus::Deploying {
                        strategy: "canary".to_string(),
                        stage: "0%".to_string(),
                    };
                    state.updated_at = chrono::Utc::now();
                    state
                });
                
                let deployment = orchestrator.deploy(
                    synthesized,
                    DeploymentStrategy::Canary {
                        stages: vec![0.01, 0.05, 0.25, 0.50, 1.0],
                    },
                ).await?;
                
                active_intents.alter(&intent_id, |_, mut state| {
                    state.status = IntentStatus::Completed {
                        improvement: deployment.performance_gain,
                        deployed_at: chrono::Utc::now(),
                    };
                    state.performance_gain = Some(deployment.performance_gain);
                    state
                });
                
                metrics.record_deployment_completed(&intent_id, true);
            }
            ConsensusDecision::Rejected(reason) => {
                active_intents.alter(&intent_id, |_, mut state| {
                    state.status = IntentStatus::Failed {
                        reason: format!("Consensus rejected: {}", reason),
                        failed_at: chrono::Utc::now(),
                    };
                    state
                });
            }
            ConsensusDecision::RequiresApproval(reason) => {
                active_intents.alter(&intent_id, |_, mut state| {
                    state.status = IntentStatus::AwaitingApproval {
                        risk_score: state.risk_score.as_ref().map(|r| r.value).unwrap_or(0.0),
                        reason,
                    };
                    state
                });
            }
        }
        
        Ok(())
    }
    
    /// Convert resonance pattern to optimization intent
    async fn pattern_to_intent(pattern: ResonancePattern) -> Result<crate::synthesis::OptimizationIntent> {
        // Extract optimization opportunity from pattern topology
        let objectives = pattern.topological_features
            .persistence_diagram
            .iter()
            .map(|(birth, death)| {
                if death - birth > 0.5 {
                    crate::Objective::MinimizeLatency {
                        percentile: 99.0,
                        target_ms: 10.0,
                    }
                } else {
                    crate::Objective::MaximizeThroughput {
                        target_ops_per_sec: 10000.0,
                    }
                }
            })
            .collect();
        
        Ok(crate::synthesis::OptimizationIntent {
            id: IntentId::new(),
            target_module: ModuleId(pattern.participating_nodes[0].clone()),
            objectives,
            constraints: vec![crate::Constraint::MaintainCorrectness],
            priority: if pattern.resonance_strength > 0.9 {
                Priority::High
            } else {
                Priority::Medium
            },
            deadline: None,
        })
    }
    
    /// Calculate comprehensive risk score for a synthesized module
    /// Uses multi-dimensional analysis of module metadata for accurate criticality assessment
    fn calculate_risk(
        module: &SynthesizedModule,
        validation: &crate::sandbox::ValidationReport,
    ) -> RiskScore {
        // Base factors
        let complexity = module.complexity_score;
        let confidence = validation.confidence_score;
        
        // Comprehensive criticality calculation from module metadata
        let criticality = Self::calculate_module_criticality(module);
        
        // Advanced risk factors
        let security_impact = Self::calculate_security_impact(module);
        let performance_impact = Self::calculate_performance_impact(module);
        let integration_risk = Self::calculate_integration_risk(module);
        
        // Risk weights based on enterprise requirements
        let weights = RiskWeights {
            complexity: 0.20,
            criticality: 0.35,
            confidence: 0.15,
            security_impact: 0.15,
            performance_impact: 0.10,
            integration_risk: 0.05,
        };
        
        // Calculate weighted risk score
        let value = (
            complexity * weights.complexity +
            criticality * weights.criticality +
            (1.0 - confidence) * weights.confidence +
            security_impact * weights.security_impact +
            performance_impact * weights.performance_impact +
            integration_risk * weights.integration_risk
        ).min(1.0).max(0.0);
        
        // Create comprehensive risk factors
        let factors = vec![
            RiskFactor::Complexity(complexity),
            RiskFactor::Criticality(criticality),
            RiskFactor::Confidence(confidence),
            RiskFactor::SecurityImpact(security_impact),
            RiskFactor::PerformanceImpact(performance_impact),
            RiskFactor::SystemIntegrationRisk(integration_risk),
        ];
        
        let explanation = format!(
            "Multi-dimensional risk assessment: complexity={:.3}, criticality={:.3}, confidence={:.3}, security_impact={:.3}, performance_impact={:.3}, integration_risk={:.3} | Overall risk: {:.3}",
            complexity, criticality, confidence, security_impact, performance_impact, integration_risk, value
        );
        
        RiskScore {
            value,
            factors,
            explanation,
        }
    }
    
    /// Calculate module criticality based on comprehensive metadata analysis
    fn calculate_module_criticality(module: &SynthesizedModule) -> f64 {
        let mut criticality_score = 0.0;
        let mut weight_sum = 0.0;
        
        // 1. Business Domain Impact (Weight: 0.25)
        let domain_weight = 0.25;
        let domain_criticality = match module.metadata.business_domain {
            BusinessDomain::Finance => 0.95,          // Highest criticality - financial systems
            BusinessDomain::Security => 0.90,         // Critical security systems
            BusinessDomain::UserManagement => 0.80,   // User-facing systems
            BusinessDomain::DataProcessing => 0.75,   // Data integrity systems
            BusinessDomain::Infrastructure => 0.70,   // Core infrastructure
            BusinessDomain::Analytics => 0.50,        // Analytics systems
            BusinessDomain::Communication => 0.45,    // Communication systems
        };
        criticality_score += domain_criticality * domain_weight;
        weight_sum += domain_weight;
        
        // 2. Module Type Impact (Weight: 0.20)
        let type_weight = 0.20;
        let type_criticality = match module.metadata.module_type {
            ModuleType::Security => 0.95,        // Security modules are critical
            ModuleType::CoreLogic => 0.85,       // Core business logic
            ModuleType::DataAccess => 0.80,      // Data access layer
            ModuleType::Integration => 0.75,     // Integration points
            ModuleType::UserInterface => 0.60,   // UI components
            ModuleType::Monitoring => 0.55,      // Monitoring systems
            ModuleType::Configuration => 0.50,   // Configuration modules
        };
        criticality_score += type_criticality * type_weight;
        weight_sum += type_weight;
        
        // 3. Security Classification Impact (Weight: 0.20)
        let security_weight = 0.20;
        let security_criticality = match module.security_classification {
            SecurityClassification::Critical => 0.95,
            SecurityClassification::Confidential => 0.80,
            SecurityClassification::Internal => 0.60,
            SecurityClassification::Public => 0.30,
        };
        criticality_score += security_criticality * security_weight;
        weight_sum += security_weight;
        
        // 4. Integration Points Impact (Weight: 0.15)
        let integration_weight = 0.15;
        let max_integration_criticality = module.integration_points
            .iter()
            .map(|point| match point.criticality_level {
                CriticalityLevel::Critical => 0.95,
                CriticalityLevel::High => 0.80,
                CriticalityLevel::Medium => 0.60,
                CriticalityLevel::Low => 0.30,
            })
            .fold(0.0, f64::max);
        criticality_score += max_integration_criticality * integration_weight;
        weight_sum += integration_weight;
        
        // 5. Operational History Impact (Weight: 0.10)
        let history_weight = 0.10;
        let history = &module.metadata.operational_history;
        let history_criticality = Self::calculate_operational_criticality(history);
        criticality_score += history_criticality * history_weight;
        weight_sum += history_weight;
        
        // 6. Performance Constraints Impact (Weight: 0.10)
        let perf_weight = 0.10;
        let perf_criticality = if module.performance_characteristics.real_time_constraints {
            0.90 // Real-time systems are highly critical
        } else {
            let mut perf_score = 0.4;
            if module.performance_characteristics.cpu_intensive { perf_score += 0.15; }
            if module.performance_characteristics.memory_intensive { perf_score += 0.15; }
            if module.performance_characteristics.io_intensive { perf_score += 0.10; }
            if module.performance_characteristics.network_intensive { perf_score += 0.10; }
            perf_score.min(0.85)
        };
        criticality_score += perf_criticality * perf_weight;
        weight_sum += perf_weight;
        
        // Normalize and return
        (criticality_score / weight_sum).min(1.0).max(0.0)
    }
    
    /// Calculate operational criticality from historical data
    fn calculate_operational_criticality(history: &OperationalHistory) -> f64 {
        let mut criticality = 0.5; // Baseline
        
        // Higher failure rate increases criticality (need more careful changes)
        criticality += (history.failure_rate * 0.3).min(0.3);
        
        // Security incidents significantly increase criticality
        let security_impact = (history.security_incidents as f64 * 0.1).min(0.3);
        criticality += security_impact;
        
        // Frequent performance issues increase criticality
        let perf_impact = (history.performance_degradation_incidents as f64 * 0.05).min(0.2);
        criticality += perf_impact;
        
        criticality.min(0.95).max(0.1)
    }
    
    /// Calculate security impact score
    fn calculate_security_impact(module: &SynthesizedModule) -> f64 {
        let mut security_score = match module.security_classification {
            SecurityClassification::Critical => 0.95,
            SecurityClassification::Confidential => 0.80,
            SecurityClassification::Internal => 0.50,
            SecurityClassification::Public => 0.20,
        };
        
        // Security-related modules have higher impact
        if matches!(module.metadata.module_type, ModuleType::Security) {
            security_score = (security_score * 1.2).min(1.0);
        }
        
        // Security integration points increase impact
        let has_security_integration = module.integration_points
            .iter()
            .any(|point| matches!(point.interaction_type, InteractionType::SecuritySystem));
        
        if has_security_integration {
            security_score = (security_score * 1.15).min(1.0);
        }
        
        // Historical security incidents increase impact
        if module.metadata.operational_history.security_incidents > 0 {
            let incident_impact = (module.metadata.operational_history.security_incidents as f64 * 0.1).min(0.2);
            security_score = (security_score + incident_impact).min(1.0);
        }
        
        security_score
    }
    
    /// Calculate performance impact score
    fn calculate_performance_impact(module: &SynthesizedModule) -> f64 {
        let perf_chars = &module.performance_characteristics;
        let mut impact = 0.3; // Baseline
        
        // Real-time constraints have highest impact
        if perf_chars.real_time_constraints {
            return 0.90;
        }
        
        // Accumulate impact from different performance characteristics
        if perf_chars.cpu_intensive { impact += 0.20; }
        if perf_chars.memory_intensive { impact += 0.15; }
        if perf_chars.io_intensive { impact += 0.15; }
        if perf_chars.network_intensive { impact += 0.10; }
        
        // Code quality affects performance impact
        let quality = &module.metadata.code_quality_metrics;
        let quality_impact = (quality.cyclomatic_complexity / 20.0).min(0.2);
        impact += quality_impact;
        
        // Technical debt affects performance
        let debt_impact = quality.technical_debt_ratio * 0.15;
        impact += debt_impact;
        
        impact.min(1.0)
    }
    
    /// Calculate system integration risk
    fn calculate_integration_risk(module: &SynthesizedModule) -> f64 {
        if module.integration_points.is_empty() {
            return 0.1; // Minimal risk for isolated modules
        }
        
        let mut risk = 0.0;
        let mut max_individual_risk = 0.0;
        
        for integration_point in &module.integration_points {
            let point_risk = match integration_point.interaction_type {
                InteractionType::Database => 0.70,        // Database integrations are risky
                InteractionType::ExternalService => 0.85,  // External services highest risk
                InteractionType::CoreSystem => 0.75,       // Core system integration
                InteractionType::SecuritySystem => 0.90,   // Security system integration
                InteractionType::UserInterface => 0.40,    // UI integration lower risk
            };
            
            let criticality_multiplier = match integration_point.criticality_level {
                CriticalityLevel::Critical => 1.0,
                CriticalityLevel::High => 0.85,
                CriticalityLevel::Medium => 0.65,
                CriticalityLevel::Low => 0.40,
            };
            
            let adjusted_risk = point_risk * criticality_multiplier;
            risk += adjusted_risk * 0.2; // Each integration contributes
            max_individual_risk = max_individual_risk.max(adjusted_risk);
        }
        
        // Take the maximum of accumulated risk and highest individual risk
        // This ensures that even one high-risk integration significantly impacts the score
        risk.max(max_individual_risk).min(1.0)
    }
    
    /// Get current Forge status
    pub async fn get_status(&self) -> ForgeStatus {
        let state = *self.state.read().await;
        let active_count = self.active_intents.len();
        let metrics = self.metrics.get_current_stats().await;
        
        ForgeStatus {
            state,
            mode: self.config.mode,
            active_intents: active_count,
            total_processed: metrics.total_intents_processed,
            success_rate: metrics.success_rate,
            average_synthesis_time: metrics.avg_synthesis_time,
            average_deployment_time: metrics.avg_deployment_time,
            system_entropy: metrics.system_entropy,
        }
    }
    
    /// Submit an optimization intent manually
    pub async fn submit_intent(&self, intent: OptimizationIntent) -> ForgeResult<IntentId> {
        // Check capacity
        if self.active_intents.len() >= self.config.max_concurrent_optimizations {
            return Err(ForgeError::ResourceLimitExceeded(
                format!("Maximum concurrent optimizations ({}) reached", 
                    self.config.max_concurrent_optimizations)
            ));
        }
        
        let intent_id = intent.id.clone();
        
        // Store intent in persistent storage with versioning and ACID guarantees
        let version = self.intent_storage.store_intent(intent.clone()).await
            .map_err(|e| ForgeError::StorageError(format!("Failed to persist intent: {}", e)))?;
        
        tracing::info!("Stored intent {} with version {} in persistent storage", intent_id, version.version);
        
        // Extract target module based on intent target
        let module_id = match &intent.target {
            crate::intent::OptimizationTarget::Module(mid) => Some(mid.clone()),
            crate::intent::OptimizationTarget::ModuleName(name) => Some(ModuleId(name.clone())),
            _ => None,
        };
        
        // Create intent state for active processing
        let intent_state = IntentState {
            intent_id: intent_id.clone(),
            status: IntentStatus::Queued,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            module_id,
            risk_score: None,
            performance_gain: None,
            error: None,
        };
        
        // Queue intent for processing
        self.active_intents.insert(intent_id.clone(), intent_state);
        self.metrics.record_intent_created(&intent_id);
        
        tracing::info!(
            "Successfully submitted optimization intent {} with {} objectives, {} constraints, priority {:?}",
            intent_id,
            intent.objectives.len(),
            intent.constraints.len(),
            intent.priority
        );
        
        Ok(intent_id)
    }
    
    /// Get intent status
    pub async fn get_intent_status(&self, intent_id: &IntentId) -> Option<IntentStatus> {
        self.active_intents
            .get(intent_id)
            .map(|entry| entry.status.clone())
    }
}

/// Forge status information
#[derive(Debug, Clone)]
pub struct ForgeStatus {
    pub state: ForgeState,
    pub mode: OperationalMode,
    pub active_intents: usize,
    pub total_processed: u64,
    pub success_rate: f64,
    pub average_synthesis_time: Duration,
    pub average_deployment_time: Duration,
    pub system_entropy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_forge_creation() {
        let config = ForgeConfig::default();
        let forge = HephaestusForge::new(config);
        assert!(forge.is_ok());
    }

    #[tokio::test]
    async fn test_forge_lifecycle() {
        let config = ForgeConfig {
            mode: OperationalMode::Manual,
            ..Default::default()
        };
        
        let forge = HephaestusForge::new(config).unwrap();
        
        // Start
        assert!(forge.start().await.is_ok());
        
        // Check status
        let status = forge.get_status().await;
        assert_eq!(status.state, ForgeState::Running);
        
        // Stop
        assert!(forge.stop().await.is_ok());
        
        // Check status
        let status = forge.get_status().await;
        assert_eq!(status.state, ForgeState::Stopped);
    }
    
    /// Test comprehensive criticality calculation for financial security module
    #[test]
    fn test_criticality_calculation_financial_security() {
        let module = create_test_module(
            BusinessDomain::Finance,
            ModuleType::Security,
            SecurityClassification::Critical,
            vec![IntegrationPoint {
                target_system: "payment_processor".to_string(),
                interaction_type: InteractionType::SecuritySystem,
                criticality_level: CriticalityLevel::Critical,
            }],
            true, // real_time_constraints
            OperationalHistory {
                deployment_frequency: 0.1,
                failure_rate: 0.02,
                recovery_time: Duration::from_secs(30),
                performance_degradation_incidents: 1,
                security_incidents: 0,
            },
        );
        
        let criticality = HephaestusForge::calculate_module_criticality(&module);
        
        // Financial + Security + Critical classification + Critical integration + Real-time
        // Should result in very high criticality (> 0.85)
        assert!(criticality > 0.85, "Financial security module should have very high criticality: {:.3}", criticality);
        assert!(criticality <= 1.0, "Criticality should not exceed 1.0");
    }
    
    /// Test criticality calculation for low-risk analytics module
    #[test]
    fn test_criticality_calculation_analytics_ui() {
        let module = create_test_module(
            BusinessDomain::Analytics,
            ModuleType::UserInterface,
            SecurityClassification::Internal,
            vec![IntegrationPoint {
                target_system: "analytics_dashboard".to_string(),
                interaction_type: InteractionType::UserInterface,
                criticality_level: CriticalityLevel::Low,
            }],
            false, // real_time_constraints
            OperationalHistory {
                deployment_frequency: 2.0,
                failure_rate: 0.001,
                recovery_time: Duration::from_secs(300),
                performance_degradation_incidents: 0,
                security_incidents: 0,
            },
        );
        
        let criticality = HephaestusForge::calculate_module_criticality(&module);
        
        // Analytics + UI + Internal + Low integration + No real-time
        // Should result in low-medium criticality (< 0.65)
        assert!(criticality < 0.65, "Analytics UI module should have low-medium criticality: {:.3}", criticality);
        assert!(criticality >= 0.0, "Criticality should not be negative");
    }
    
    /// Test comprehensive risk calculation with all factors
    #[test]
    fn test_comprehensive_risk_calculation() {
        let module = create_test_module(
            BusinessDomain::Finance,
            ModuleType::CoreLogic,
            SecurityClassification::Confidential,
            vec![
                IntegrationPoint {
                    target_system: "database".to_string(),
                    interaction_type: InteractionType::Database,
                    criticality_level: CriticalityLevel::High,
                },
                IntegrationPoint {
                    target_system: "external_api".to_string(),
                    interaction_type: InteractionType::ExternalService,
                    criticality_level: CriticalityLevel::Medium,
                },
            ],
            false,
            OperationalHistory {
                deployment_frequency: 0.5,
                failure_rate: 0.05,
                recovery_time: Duration::from_secs(120),
                performance_degradation_incidents: 2,
                security_incidents: 1,
            },
        );
        
        // Create validation report
        let validation = create_test_validation_report(0.85);
        
        let risk_score = HephaestusForge::calculate_risk(&module, &validation);
        
        // Verify risk score is reasonable
        assert!(risk_score.value >= 0.0 && risk_score.value <= 1.0, 
                "Risk score should be between 0 and 1: {:.3}", risk_score.value);
        
        // Verify all risk factors are present
        assert_eq!(risk_score.factors.len(), 6, "Should have all 6 risk factors");
        
        // Check that explanation contains key metrics
        assert!(risk_score.explanation.contains("complexity="));
        assert!(risk_score.explanation.contains("criticality="));
        assert!(risk_score.explanation.contains("confidence="));
        assert!(risk_score.explanation.contains("security_impact="));
        assert!(risk_score.explanation.contains("performance_impact="));
        assert!(risk_score.explanation.contains("integration_risk="));
        
        // Financial core logic with security incidents should have elevated risk
        assert!(risk_score.value > 0.5, "Financial module with security incidents should have elevated risk: {:.3}", risk_score.value);
    }
    
    /// Test security impact calculation
    #[test]
    fn test_security_impact_calculation() {
        let high_security_module = create_test_module(
            BusinessDomain::Security,
            ModuleType::Security,
            SecurityClassification::Critical,
            vec![IntegrationPoint {
                target_system: "auth_system".to_string(),
                interaction_type: InteractionType::SecuritySystem,
                criticality_level: CriticalityLevel::Critical,
            }],
            false,
            OperationalHistory {
                deployment_frequency: 0.1,
                failure_rate: 0.001,
                recovery_time: Duration::from_secs(60),
                performance_degradation_incidents: 0,
                security_incidents: 2,
            },
        );
        
        let low_security_module = create_test_module(
            BusinessDomain::Analytics,
            ModuleType::UserInterface,
            SecurityClassification::Public,
            vec![],
            false,
            OperationalHistory {
                deployment_frequency: 1.0,
                failure_rate: 0.001,
                recovery_time: Duration::from_secs(60),
                performance_degradation_incidents: 0,
                security_incidents: 0,
            },
        );
        
        let high_impact = HephaestusForge::calculate_security_impact(&high_security_module);
        let low_impact = HephaestusForge::calculate_security_impact(&low_security_module);
        
        assert!(high_impact > 0.85, "Security module with critical classification should have high security impact: {:.3}", high_impact);
        assert!(low_impact < 0.5, "Public analytics UI should have low security impact: {:.3}", low_impact);
        assert!(high_impact > low_impact, "High security module should have higher impact than low security module");
    }
    
    /// Test performance impact calculation
    #[test]
    fn test_performance_impact_calculation() {
        let real_time_module = create_test_module(
            BusinessDomain::Infrastructure,
            ModuleType::CoreLogic,
            SecurityClassification::Internal,
            vec![],
            true, // real_time_constraints
            create_default_history(),
        );
        
        let normal_module = create_test_module(
            BusinessDomain::Analytics,
            ModuleType::UserInterface,
            SecurityClassification::Internal,
            vec![],
            false, // real_time_constraints
            create_default_history(),
        );
        
        let real_time_impact = HephaestusForge::calculate_performance_impact(&real_time_module);
        let normal_impact = HephaestusForge::calculate_performance_impact(&normal_module);
        
        assert!(real_time_impact >= 0.90, "Real-time module should have very high performance impact: {:.3}", real_time_impact);
        assert!(normal_impact < 0.70, "Normal module should have lower performance impact: {:.3}", normal_impact);
        assert!(real_time_impact > normal_impact, "Real-time module should have higher impact than normal module");
    }
    
    /// Test integration risk calculation
    #[test]
    fn test_integration_risk_calculation() {
        let high_risk_module = create_test_module(
            BusinessDomain::Finance,
            ModuleType::Integration,
            SecurityClassification::Internal,
            vec![
                IntegrationPoint {
                    target_system: "external_payment_gateway".to_string(),
                    interaction_type: InteractionType::ExternalService,
                    criticality_level: CriticalityLevel::Critical,
                },
                IntegrationPoint {
                    target_system: "security_system".to_string(),
                    interaction_type: InteractionType::SecuritySystem,
                    criticality_level: CriticalityLevel::High,
                },
            ],
            false,
            create_default_history(),
        );
        
        let isolated_module = create_test_module(
            BusinessDomain::Analytics,
            ModuleType::UserInterface,
            SecurityClassification::Internal,
            vec![], // No integrations
            false,
            create_default_history(),
        );
        
        let high_risk = HephaestusForge::calculate_integration_risk(&high_risk_module);
        let low_risk = HephaestusForge::calculate_integration_risk(&isolated_module);
        
        assert!(high_risk > 0.70, "Module with critical external service integration should have high risk: {:.3}", high_risk);
        assert!(low_risk <= 0.1, "Isolated module should have minimal integration risk: {:.3}", low_risk);
        assert!(high_risk > low_risk, "Highly integrated module should have higher risk than isolated module");
    }
    
    /// Test operational history impact on criticality
    #[test]
    fn test_operational_history_criticality() {
        let problematic_history = OperationalHistory {
            deployment_frequency: 0.1,
            failure_rate: 0.15, // High failure rate
            recovery_time: Duration::from_secs(600),
            performance_degradation_incidents: 5,
            security_incidents: 3, // Multiple security incidents
        };
        
        let stable_history = OperationalHistory {
            deployment_frequency: 1.0,
            failure_rate: 0.001, // Very low failure rate
            recovery_time: Duration::from_secs(30),
            performance_degradation_incidents: 0,
            security_incidents: 0,
        };
        
        let problematic_criticality = HephaestusForge::calculate_operational_criticality(&problematic_history);
        let stable_criticality = HephaestusForge::calculate_operational_criticality(&stable_history);
        
        assert!(problematic_criticality > 0.70, "Problematic operational history should result in high criticality: {:.3}", problematic_criticality);
        assert!(stable_criticality < 0.60, "Stable operational history should result in moderate criticality: {:.3}", stable_criticality);
        assert!(problematic_criticality > stable_criticality, "Problematic history should have higher criticality than stable history");
    }
    
    /// Performance test for real-time criticality calculation
    #[test]
    fn test_criticality_calculation_performance() {
        let module = create_test_module(
            BusinessDomain::Finance,
            ModuleType::CoreLogic,
            SecurityClassification::Confidential,
            vec![
                IntegrationPoint {
                    target_system: "system1".to_string(),
                    interaction_type: InteractionType::Database,
                    criticality_level: CriticalityLevel::High,
                },
                IntegrationPoint {
                    target_system: "system2".to_string(),
                    interaction_type: InteractionType::ExternalService,
                    criticality_level: CriticalityLevel::Medium,
                },
            ],
            false,
            create_default_history(),
        );
        
        let start = std::time::Instant::now();
        
        // Perform multiple criticality calculations to test performance
        for _ in 0..1000 {
            let _ = HephaestusForge::calculate_module_criticality(&module);
        }
        
        let elapsed = start.elapsed();
        let per_calculation = elapsed.as_nanos() / 1000;
        
        // Should be able to calculate criticality in under 10 microseconds per calculation
        assert!(per_calculation < 10_000, "Criticality calculation should be fast: {}ns per calculation", per_calculation);
    }
    
    // Helper functions for creating test data
    
    fn create_test_module(
        business_domain: BusinessDomain,
        module_type: ModuleType,
        security_classification: SecurityClassification,
        integration_points: Vec<IntegrationPoint>,
        real_time_constraints: bool,
        operational_history: OperationalHistory,
    ) -> SynthesizedModule {
        SynthesizedModule {
            id: ModuleId("test_module".to_string()),
            complexity_score: 0.6,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                module_type,
                business_domain,
                error_handling_completeness: 0.8,
                test_coverage: 0.85,
                documentation_completeness: 0.7,
                code_quality_metrics: CodeQualityMetrics {
                    cyclomatic_complexity: 8.0,
                    maintainability_index: 75.0,
                    technical_debt_ratio: 0.15,
                    code_duplication: 0.05,
                },
                operational_history,
            },
            code: vec![],
            dependencies: vec![],
            security_classification,
            performance_characteristics: PerformanceCharacteristics {
                cpu_intensive: true,
                memory_intensive: false,
                io_intensive: false,
                network_intensive: false,
                real_time_constraints,
            },
            integration_points,
        }
    }
    
    fn create_test_validation_report(confidence_score: f64) -> crate::sandbox::ValidationReport {
        // This would need to be implemented based on the actual ValidationReport structure
        // For now, we'll create a mock structure
        crate::sandbox::ValidationReport {
            passed: true,
            confidence_score,
            reason: "Mock validation report for testing".to_string(),
        }
    }
    
    fn create_default_history() -> OperationalHistory {
        OperationalHistory {
            deployment_frequency: 1.0,
            failure_rate: 0.01,
            recovery_time: Duration::from_secs(60),
            performance_degradation_incidents: 0,
            security_incidents: 0,
        }
    }
}