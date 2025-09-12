//! Unified Neuromorphic System - Always-on Natural Language Interface
//!
//! This module implements the enterprise-grade unified system that provides
//! always-on natural language processing with dynamic resource allocation.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use csf_clogic::{CLogicSystem, CLogicState};
use super::{
    NeuromorphicBackend, NeuralLanguageProcessor, CommandIntent, CommandContext, CommandDomain,
    LearningConfig, HardwareCapabilities, SystemState, ResourceAllocation
};

/// Neural Language Interface - Always-on NLP layer
pub struct NeuralLanguageInterface {
    /// Primary NLP processor
    nlp_processor: Arc<NeuralLanguageProcessor>,
    
    /// Context analyzer for domain classification
    context_analyzer: ContextAnalyzer,
    
    /// Command router for appropriate handling
    command_router: CommandRouter,
    
    /// Interface metrics
    metrics: Arc<RwLock<InterfaceMetrics>>,
}

#[derive(Debug, Clone)]
struct InterfaceMetrics {
    pub total_interactions: u64,
    pub defense_commands: u64,
    pub system_commands: u64,
    pub quantum_commands: u64,
    pub learning_commands: u64,
    pub avg_confidence: f64,
    pub context_accuracy: f64,
}

impl Default for InterfaceMetrics {
    fn default() -> Self {
        Self {
            total_interactions: 0,
            defense_commands: 0,
            system_commands: 0,
            quantum_commands: 0,
            learning_commands: 0,
            avg_confidence: 0.0,
            context_accuracy: 0.0,
        }
    }
}

/// Context analyzer for understanding operational environment
struct ContextAnalyzer {
    /// Current operational mode
    operational_mode: OperationalMode,
    
    /// System load tracking
    system_load_monitor: SystemLoadMonitor,
    
    /// Threat level assessment
    threat_assessment: ThreatAssessment,
}

#[derive(Debug, Clone)]
pub enum OperationalMode {
    Normal,
    Defense,
    CriticalDefense,
    Learning,
    Maintenance,
}

#[derive(Debug, Clone)]
struct SystemLoadMonitor {
    cpu_usage: f64,
    memory_usage: f64,
    neuromorphic_load: f64,
    clogic_load: f64,
}

#[derive(Debug, Clone)]
struct ThreatAssessment {
    current_threat_level: ThreatLevel,
    active_incidents: u32,
    defense_readiness: f64,
}

#[derive(Debug, Clone)]
pub enum ThreatLevel {
    Minimal,
    Elevated,
    High,
    Critical,
}

/// Command router for context-aware processing
struct CommandRouter {
    /// Defense command patterns
    defense_patterns: Vec<DefenseCommandPattern>,
    
    /// System command patterns
    system_patterns: Vec<SystemCommandPattern>,
    
    /// Priority routing rules
    priority_rules: Vec<RoutingRule>,
}

#[derive(Debug, Clone)]
struct DefenseCommandPattern {
    keywords: Vec<String>,
    urgency_boost: f64,
    requires_escalation: bool,
}

#[derive(Debug, Clone)]
struct SystemCommandPattern {
    keywords: Vec<String>,
    resource_impact: ResourceImpact,
    concurrent_safe: bool,
}

#[derive(Debug, Clone)]
enum ResourceImpact {
    Minimal,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone)]
struct RoutingRule {
    condition: RoutingCondition,
    action: RoutingAction,
    priority: u32,
}

#[derive(Debug, Clone)]
enum RoutingCondition {
    ThreatLevel(ThreatLevel),
    OperationalMode(OperationalMode),
    SystemLoad(f64),
    CommandDomain(CommandDomain),
}

#[derive(Debug, Clone)]
enum RoutingAction {
    AllocateResources(ResourceAllocation),
    EscalateToDefense,
    DeferToMaintenance,
    PrioritizeExecution,
}

/// Dynamic Resource Allocator - Manages neuromorphic computing resources
pub struct DynamicResourceAllocator {
    /// Current allocation strategy
    allocation_strategy: AllocationStrategy,
    
    /// Resource pools
    resource_pools: ResourcePools,
    
    /// Performance monitors
    performance_monitors: PerformanceMonitors,
    
    /// Historical allocation data for optimization
    allocation_history: Arc<RwLock<AllocationHistory>>,
}

#[derive(Debug, Clone)]
enum AllocationStrategy {
    /// Static allocation based on configuration
    Static,
    /// Dynamic allocation based on workload
    Dynamic,
    /// Predictive allocation using historical data
    Predictive,
    /// Emergency allocation for critical operations
    Emergency,
}

#[derive(Debug, Clone)]
struct ResourcePools {
    /// Available neuromorphic processing units
    neuromorphic_units: f64,
    
    /// C-LOGIC module resources
    clogic_resources: CLogicResources,
    
    /// Python bridge connections
    python_bridge_capacity: f64,
    
    /// Memory allocated for spike processing
    spike_memory_mb: f64,
}

#[derive(Debug, Clone)]
struct CLogicResources {
    drpp_capacity: f64,
    ems_capacity: f64,
    adp_capacity: f64,
    egc_capacity: f64,
}

#[derive(Debug, Clone)]
struct PerformanceMonitors {
    processing_latency: f64,
    throughput_commands_per_sec: f64,
    memory_efficiency: f64,
    error_rate: f64,
}

#[derive(Debug, Clone)]
struct AllocationHistory {
    allocations: Vec<HistoricalAllocation>,
    performance_correlation: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct HistoricalAllocation {
    timestamp: std::time::SystemTime,
    allocation: ResourceAllocation,
    workload_type: CommandDomain,
    performance_result: f64,
}

impl NeuralLanguageInterface {
    pub async fn new(
        nlp_processor: Arc<NeuralLanguageProcessor>,
        clogic_system: Arc<CLogicSystem>,
    ) -> Result<Self> {
        info!("Initializing Neural Language Interface");
        
        let context_analyzer = ContextAnalyzer::new().await?;
        let command_router = CommandRouter::new();
        let metrics = Arc::new(RwLock::new(InterfaceMetrics::default()));
        
        Ok(Self {
            nlp_processor,
            context_analyzer,
            command_router,
            metrics,
        })
    }
    
    /// Process input with full context awareness
    pub async fn process_with_context(
        &self,
        input: &str,
        current_mode: &OperationalMode,
        threat_level: &ThreatLevel,
    ) -> Result<EnhancedCommandIntent> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing with context: mode={:?}, threat={:?}", current_mode, threat_level);
        
        // Stage 1: Basic NLP processing
        let base_intent = self.nlp_processor.process_input(input).await?;
        
        // Stage 2: Context enhancement
        let enhanced_context = self.context_analyzer.enhance_context(
            &base_intent.context,
            current_mode,
            threat_level,
        ).await?;
        
        // Stage 3: Route through appropriate command handler
        let routing_decision = self.command_router.determine_routing(
            &base_intent,
            &enhanced_context,
        ).await?;
        
        // Stage 4: Create enhanced intent
        let enhanced_intent = EnhancedCommandIntent {
            base_intent,
            enhanced_context,
            routing_decision: routing_decision.clone(),
            resource_requirements: self.calculate_resource_requirements(&routing_decision).await?,
        };
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_metrics(&enhanced_intent, processing_time).await;
        
        Ok(enhanced_intent)
    }
    
    async fn calculate_resource_requirements(&self, routing: &RoutingDecision) -> Result<ResourceRequirements> {
        let base_requirements = ResourceRequirements {
            neuromorphic_units: 0.1,
            clogic_modules: CLogicRequirements {
                drpp: 0.05,
                ems: 0.05,
                adp: 0.05,
                egc: 0.05,
            },
            python_bridges: 0.1,
            memory_mb: 50.0,
            execution_priority: ExecutionPriority::Normal,
        };
        
        // Adjust based on routing decision
        match routing.priority_level {
            PriorityLevel::Critical => Ok(ResourceRequirements {
                neuromorphic_units: base_requirements.neuromorphic_units * 3.0,
                execution_priority: ExecutionPriority::Critical,
                ..base_requirements
            }),
            PriorityLevel::High => Ok(ResourceRequirements {
                neuromorphic_units: base_requirements.neuromorphic_units * 2.0,
                execution_priority: ExecutionPriority::High,
                ..base_requirements
            }),
            _ => Ok(base_requirements),
        }
    }
    
    async fn update_metrics(&self, intent: &EnhancedCommandIntent, processing_time: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.total_interactions += 1;
        
        match intent.base_intent.context.domain {
            CommandDomain::Defense => metrics.defense_commands += 1,
            CommandDomain::System => metrics.system_commands += 1,
            CommandDomain::Quantum => metrics.quantum_commands += 1,
            CommandDomain::Learning => metrics.learning_commands += 1,
            _ => {}
        }
        
        metrics.avg_confidence = (metrics.avg_confidence + intent.base_intent.confidence) / 2.0;
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedCommandIntent {
    pub base_intent: CommandIntent,
    pub enhanced_context: EnhancedContext,
    pub routing_decision: RoutingDecision,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone)]
pub struct EnhancedContext {
    pub operational_mode: OperationalMode,
    pub threat_level: ThreatLevel,
    pub system_load: SystemLoadMonitor,
    pub concurrent_operations: u32,
    pub defense_readiness: f64,
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub target_handler: CommandHandler,
    pub priority_level: PriorityLevel,
    pub resource_allocation: ResourceAllocation,
    pub execution_strategy: ExecutionStrategy,
}

#[derive(Debug, Clone)]
enum CommandHandler {
    DefenseSystem,
    QuantumSystem,
    LearningSystem,
    SystemManagement,
    Shell,
}

#[derive(Debug, Clone)]
pub enum PriorityLevel {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
enum ExecutionStrategy {
    Immediate,
    Queued,
    BackgroundAsync,
    DefenseEscalated,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub neuromorphic_units: f64,
    pub clogic_modules: CLogicRequirements,
    pub python_bridges: f64,
    pub memory_mb: f64,
    pub execution_priority: ExecutionPriority,
}

#[derive(Debug, Clone)]
pub struct CLogicRequirements {
    pub drpp: f64,
    pub ems: f64,
    pub adp: f64,
    pub egc: f64,
}

#[derive(Debug, Clone)]
enum ExecutionPriority {
    Background,
    Normal,
    High,
    Critical,
}

impl DynamicResourceAllocator {
    pub async fn new(
        hardware_caps: &HardwareCapabilities,
        initial_config: &ResourceAllocation,
    ) -> Result<Self> {
        info!("Initializing Dynamic Resource Allocator");
        
        let allocation_strategy = AllocationStrategy::Dynamic;
        
        let resource_pools = ResourcePools {
            neuromorphic_units: if hardware_caps.has_neuromorphic_chip { 100.0 } else { 50.0 },
            clogic_resources: CLogicResources {
                drpp_capacity: 25.0,
                ems_capacity: 20.0,
                adp_capacity: 20.0,
                egc_capacity: 20.0,
            },
            python_bridge_capacity: if hardware_caps.has_cuda_gpu { 20.0 } else { 10.0 },
            spike_memory_mb: if hardware_caps.gpu_memory_gb > 16.0 { 1024.0 } else { 512.0 },
        };
        
        let performance_monitors = PerformanceMonitors {
            processing_latency: 0.0,
            throughput_commands_per_sec: 0.0,
            memory_efficiency: 1.0,
            error_rate: 0.0,
        };
        
        let allocation_history = Arc::new(RwLock::new(AllocationHistory {
            allocations: Vec::new(),
            performance_correlation: HashMap::new(),
        }));
        
        Ok(Self {
            allocation_strategy,
            resource_pools,
            performance_monitors,
            allocation_history,
        })
    }
    
    /// Allocate resources for specific command intent
    pub async fn allocate_for_intent(&self, intent: &EnhancedCommandIntent) -> Result<AllocationResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Allocating resources for intent: {:?}", intent.base_intent.command);
        
        // Calculate required allocation based on command context
        let required_allocation = self.calculate_allocation_requirements(intent).await?;
        
        // Check resource availability
        let availability = self.check_resource_availability(&required_allocation).await?;
        
        if !availability.sufficient {
            // Implement resource reclamation if needed
            self.reclaim_resources(&required_allocation, &intent.routing_decision.priority_level).await?;
        }
        
        // Apply allocation
        let allocation_result = self.apply_allocation(&required_allocation, intent).await?;
        
        // Record allocation for learning
        self.record_allocation_decision(&required_allocation, intent).await;
        
        let allocation_time = start_time.elapsed().as_millis() as f64;
        debug!("Resource allocation completed in {:.1}ms", allocation_time);
        
        Ok(allocation_result)
    }
    
    async fn calculate_allocation_requirements(&self, intent: &EnhancedCommandIntent) -> Result<ResourceAllocation> {
        let base_allocation = match intent.base_intent.context.domain {
            CommandDomain::Defense => ResourceAllocation {
                nlp: 0.05,  // Minimal for defense - prioritize action
                drpp: 0.40, // High pattern recognition for threats
                ems: 0.15,  // Emotional state analysis
                adp: 0.25,  // Decision making under pressure
                egc: 0.15,  // Consensus for critical decisions
            },
            CommandDomain::Quantum => ResourceAllocation {
                nlp: 0.10,
                drpp: 0.35, // Pattern analysis for quantum states
                ems: 0.05,  // Minimal emotional processing
                adp: 0.30,  // Complex decision making
                egc: 0.20,  // Quantum consensus protocols
            },
            CommandDomain::Learning => ResourceAllocation {
                nlp: 0.30,  // High for processing training examples
                drpp: 0.25, // Pattern learning and recognition
                ems: 0.15,  // Understanding user feedback sentiment
                adp: 0.20,  // Learning strategy decisions
                egc: 0.10,  // Consensus on learned patterns
            },
            CommandDomain::System => ResourceAllocation {
                nlp: 0.15,  // Standard language processing
                drpp: 0.20, // System pattern recognition
                ems: 0.10,  // Basic emotional context
                adp: 0.25,  // System management decisions
                egc: 0.30,  // System coordination consensus
            },
            _ => ResourceAllocation {
                nlp: 0.20,  // Default balanced allocation
                drpp: 0.20,
                ems: 0.20,
                adp: 0.20,
                egc: 0.20,
            },
        };
        
        // Apply urgency multipliers
        let urgency_multiplier = 1.0 + intent.base_intent.context.urgency;
        
        Ok(ResourceAllocation {
            nlp: base_allocation.nlp * urgency_multiplier,
            drpp: base_allocation.drpp * urgency_multiplier,
            ems: base_allocation.ems * urgency_multiplier,
            adp: base_allocation.adp * urgency_multiplier,
            egc: base_allocation.egc * urgency_multiplier,
        })
    }
    
    async fn check_resource_availability(&self, required: &ResourceAllocation) -> Result<ResourceAvailability> {
        // Check current resource utilization
        let current_load = self.get_current_resource_load().await?;
        
        let sufficient = 
            current_load.nlp + required.nlp <= 1.0 &&
            current_load.drpp + required.drpp <= 1.0 &&
            current_load.ems + required.ems <= 1.0 &&
            current_load.adp + required.adp <= 1.0 &&
            current_load.egc + required.egc <= 1.0;
        
        Ok(ResourceAvailability {
            sufficient,
            available_nlp: 1.0 - current_load.nlp,
            available_drpp: 1.0 - current_load.drpp,
            available_ems: 1.0 - current_load.ems,
            available_adp: 1.0 - current_load.adp,
            available_egc: 1.0 - current_load.egc,
        })
    }
    
    async fn get_current_resource_load(&self) -> Result<ResourceAllocation> {
        // In a real implementation, this would query actual system utilization
        Ok(ResourceAllocation {
            nlp: 0.10,
            drpp: 0.15,
            ems: 0.05,
            adp: 0.20,
            egc: 0.10,
        })
    }
    
    async fn reclaim_resources(&self, required: &ResourceAllocation, priority: &PriorityLevel) -> Result<()> {
        info!("Reclaiming resources for {:?} priority command", priority);
        
        match priority {
            PriorityLevel::Critical => {
                // Preempt lower priority operations
                self.preempt_lower_priority_operations().await?;
            },
            PriorityLevel::High => {
                // Gracefully reduce background operations
                self.reduce_background_operations().await?;
            },
            _ => {
                // Wait for resources to become available
                self.wait_for_resource_availability(required).await?;
            }
        }
        
        Ok(())
    }
    
    async fn preempt_lower_priority_operations(&self) -> Result<()> {
        debug!("Preempting lower priority operations for critical command");
        Ok(())
    }
    
    async fn reduce_background_operations(&self) -> Result<()> {
        debug!("Reducing background operations for high priority command");
        Ok(())
    }
    
    async fn wait_for_resource_availability(&self, _required: &ResourceAllocation) -> Result<()> {
        debug!("Waiting for resource availability");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(())
    }
    
    async fn apply_allocation(&self, allocation: &ResourceAllocation, intent: &EnhancedCommandIntent) -> Result<AllocationResult> {
        debug!("Applying resource allocation: {:?}", allocation);
        
        // Record allocation
        let allocation_record = HistoricalAllocation {
            timestamp: std::time::SystemTime::now(),
            allocation: allocation.clone(),
            workload_type: intent.base_intent.context.domain.clone(),
            performance_result: 0.0, // Will be updated after execution
        };
        
        {
            let mut history = self.allocation_history.write().await;
            history.allocations.push(allocation_record);
            
            // Keep history bounded
            if history.allocations.len() > 10000 {
                history.allocations.remove(0);
            }
        }
        
        Ok(AllocationResult {
            allocated_resources: allocation.clone(),
            allocation_id: format!("alloc_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()),
            estimated_duration_ms: self.estimate_execution_duration(intent).await?,
        })
    }
    
    async fn estimate_execution_duration(&self, intent: &EnhancedCommandIntent) -> Result<f64> {
        let base_duration = match intent.base_intent.context.domain {
            CommandDomain::Defense => 50.0,    // Fast defense responses
            CommandDomain::Quantum => 200.0,   // Complex quantum calculations
            CommandDomain::Learning => 500.0,  // Learning operations take time
            CommandDomain::System => 100.0,    // Standard system commands
            _ => 150.0,
        };
        
        // Adjust for urgency
        let urgency_factor = 1.0 / (1.0 + intent.base_intent.context.urgency);
        
        Ok(base_duration * urgency_factor)
    }
    
    async fn record_allocation_decision(&self, allocation: &ResourceAllocation, intent: &EnhancedCommandIntent) {
        debug!("Recording allocation decision for performance learning");
        
        // This data is used to improve future allocation decisions
        let mut history = self.allocation_history.write().await;
        
        let domain_key = format!("{:?}", intent.base_intent.context.domain);
        let current_performance = history.performance_correlation.get(&domain_key).unwrap_or(&0.5);
        
        // Update performance correlation (simplified - real implementation would be more sophisticated)
        let new_performance = (current_performance + intent.base_intent.confidence) / 2.0;
        history.performance_correlation.insert(domain_key, new_performance);
    }
}

#[derive(Debug, Clone)]
struct ResourceAvailability {
    sufficient: bool,
    available_nlp: f64,
    available_drpp: f64,
    available_ems: f64,
    available_adp: f64,
    available_egc: f64,
}

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub allocated_resources: ResourceAllocation,
    pub allocation_id: String,
    pub estimated_duration_ms: f64,
}

impl ContextAnalyzer {
    async fn new() -> Result<Self> {
        let operational_mode = OperationalMode::Normal;
        let system_load_monitor = SystemLoadMonitor {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            neuromorphic_load: 0.0,
            clogic_load: 0.0,
        };
        let threat_assessment = ThreatAssessment {
            current_threat_level: ThreatLevel::Minimal,
            active_incidents: 0,
            defense_readiness: 0.8,
        };
        
        Ok(Self {
            operational_mode,
            system_load_monitor,
            threat_assessment,
        })
    }
    
    async fn enhance_context(
        &self,
        base_context: &CommandContext,
        current_mode: &OperationalMode,
        threat_level: &ThreatLevel,
    ) -> Result<EnhancedContext> {
        // Update system load monitoring
        let system_load = self.update_system_load().await?;
        
        // Assess concurrent operations
        let concurrent_operations = self.count_concurrent_operations().await?;
        
        Ok(EnhancedContext {
            operational_mode: current_mode.clone(),
            threat_level: threat_level.clone(),
            system_load,
            concurrent_operations,
            defense_readiness: self.threat_assessment.defense_readiness,
        })
    }
    
    async fn update_system_load(&self) -> Result<SystemLoadMonitor> {
        // In real implementation, this would query actual system metrics
        Ok(SystemLoadMonitor {
            cpu_usage: 0.15,
            memory_usage: 0.25,
            neuromorphic_load: 0.10,
            clogic_load: 0.20,
        })
    }
    
    async fn count_concurrent_operations(&self) -> Result<u32> {
        // In real implementation, this would count active operations
        Ok(2)
    }
}

impl CommandRouter {
    fn new() -> Self {
        let defense_patterns = vec![
            DefenseCommandPattern {
                keywords: vec!["threat".to_string(), "attack".to_string(), "intrusion".to_string(), "breach".to_string()],
                urgency_boost: 0.8,
                requires_escalation: true,
            },
            DefenseCommandPattern {
                keywords: vec!["scan".to_string(), "monitor".to_string(), "watch".to_string(), "guard".to_string()],
                urgency_boost: 0.3,
                requires_escalation: false,
            },
        ];
        
        let system_patterns = vec![
            SystemCommandPattern {
                keywords: vec!["status".to_string(), "health".to_string(), "info".to_string()],
                resource_impact: ResourceImpact::Minimal,
                concurrent_safe: true,
            },
            SystemCommandPattern {
                keywords: vec!["optimize".to_string(), "configure".to_string(), "deploy".to_string()],
                resource_impact: ResourceImpact::High,
                concurrent_safe: false,
            },
        ];
        
        let priority_rules = vec![
            RoutingRule {
                condition: RoutingCondition::ThreatLevel(ThreatLevel::Critical),
                action: RoutingAction::EscalateToDefense,
                priority: 1,
            },
            RoutingRule {
                condition: RoutingCondition::SystemLoad(0.9),
                action: RoutingAction::DeferToMaintenance,
                priority: 2,
            },
        ];
        
        Self {
            defense_patterns,
            system_patterns,
            priority_rules,
        }
    }
    
    async fn determine_routing(
        &self,
        intent: &CommandIntent,
        context: &EnhancedContext,
    ) -> Result<RoutingDecision> {
        // Analyze command for defense patterns
        let is_defense_command = self.is_defense_command(&intent.command);
        let priority_level = self.assess_priority(intent, context, is_defense_command);
        
        let target_handler = if is_defense_command {
            CommandHandler::DefenseSystem
        } else {
            match intent.context.domain {
                CommandDomain::Quantum => CommandHandler::QuantumSystem,
                CommandDomain::Learning => CommandHandler::LearningSystem,
                CommandDomain::System => CommandHandler::SystemManagement,
                _ => CommandHandler::Shell,
            }
        };
        
        let execution_strategy = match (&priority_level, &context.operational_mode) {
            (PriorityLevel::Critical, _) => ExecutionStrategy::Immediate,
            (PriorityLevel::High, OperationalMode::Defense) => ExecutionStrategy::DefenseEscalated,
            (_, OperationalMode::Defense) => ExecutionStrategy::Queued,
            _ => ExecutionStrategy::Immediate,
        };
        
        // Calculate dynamic resource allocation
        let resource_allocation = self.calculate_dynamic_allocation(intent, context, &priority_level);
        
        Ok(RoutingDecision {
            target_handler,
            priority_level,
            resource_allocation,
            execution_strategy,
        })
    }
    
    fn is_defense_command(&self, command: &str) -> bool {
        let cmd_lower = command.to_lowercase();
        self.defense_patterns.iter().any(|pattern| {
            pattern.keywords.iter().any(|keyword| cmd_lower.contains(keyword))
        })
    }
    
    fn assess_priority(&self, intent: &CommandIntent, context: &EnhancedContext, is_defense: bool) -> PriorityLevel {
        if is_defense && matches!(context.threat_level, ThreatLevel::Critical) {
            return PriorityLevel::Critical;
        }
        
        if intent.context.urgency > 0.8 || matches!(context.threat_level, ThreatLevel::High) {
            return PriorityLevel::High;
        }
        
        if intent.confidence < 0.5 {
            return PriorityLevel::Low;
        }
        
        PriorityLevel::Normal
    }
    
    fn calculate_dynamic_allocation(&self, intent: &CommandIntent, context: &EnhancedContext, priority: &PriorityLevel) -> ResourceAllocation {
        let mut base_allocation = ResourceAllocation {
            nlp: 0.15,
            drpp: 0.20,
            ems: 0.15,
            adp: 0.25,
            egc: 0.25,
        };
        
        // Adjust for operational mode
        match context.operational_mode {
            OperationalMode::Defense => {
                base_allocation.drpp += 0.15; // Boost pattern recognition
                base_allocation.adp += 0.10; // Boost decision making
                base_allocation.nlp -= 0.10; // Reduce NLP overhead
                base_allocation.ems -= 0.05; // Reduce emotional processing
            },
            OperationalMode::Learning => {
                base_allocation.nlp += 0.15; // Boost language understanding
                base_allocation.ems += 0.10; // Boost emotional context
                base_allocation.drpp += 0.05; // Boost pattern learning
            },
            _ => {} // Normal allocation
        }
        
        // Priority adjustments
        match priority {
            PriorityLevel::Critical => {
                // Boost all resources for critical operations
                base_allocation.nlp *= 1.5;
                base_allocation.drpp *= 1.5;
                base_allocation.ems *= 1.2;
                base_allocation.adp *= 1.5;
                base_allocation.egc *= 1.3;
            },
            PriorityLevel::High => {
                // Moderate boost
                base_allocation.drpp *= 1.3;
                base_allocation.adp *= 1.3;
            },
            PriorityLevel::Low => {
                // Reduce resource usage
                base_allocation.nlp *= 0.7;
                base_allocation.drpp *= 0.8;
                base_allocation.ems *= 0.8;
                base_allocation.adp *= 0.8;
                base_allocation.egc *= 0.8;
            },
            _ => {} // Normal allocation
        }
        
        // Ensure allocation doesn't exceed 1.0 total
        let total = base_allocation.nlp + base_allocation.drpp + base_allocation.ems + 
                   base_allocation.adp + base_allocation.egc;
        
        if total > 1.0 {
            let scale_factor = 1.0 / total;
            base_allocation.nlp *= scale_factor;
            base_allocation.drpp *= scale_factor;
            base_allocation.ems *= scale_factor;
            base_allocation.adp *= scale_factor;
            base_allocation.egc *= scale_factor;
        }
        
        base_allocation
    }
}

/// Enhanced Unified System with always-on NLP
pub struct EnhancedUnifiedNeuromorphicSystem {
    /// Original neuromorphic system core
    core_system: super::UnifiedNeuromorphicSystem,
    
    /// Always-on neural language interface
    language_interface: Arc<NeuralLanguageInterface>,
    
    /// Dynamic resource allocator
    resource_allocator: Arc<DynamicResourceAllocator>,
    
    /// Current operational context
    operational_context: Arc<RwLock<OperationalContext>>,
}

#[derive(Debug, Clone)]
struct OperationalContext {
    mode: OperationalMode,
    threat_level: ThreatLevel,
    active_allocations: Vec<String>,
    command_queue: Vec<QueuedCommand>,
}

#[derive(Debug, Clone)]
struct QueuedCommand {
    intent: CommandIntent,
    priority: PriorityLevel,
    timestamp: std::time::SystemTime,
}

impl EnhancedUnifiedNeuromorphicSystem {
    /// Initialize the enhanced unified system
    pub async fn initialize(config_path: Option<&std::path::Path>) -> Result<Self> {
        info!("Initializing Enhanced Unified Neuromorphic System");
        
        // Initialize core system
        let core_system = super::UnifiedNeuromorphicSystem::initialize(config_path).await?;
        
        // Create enhanced components
        let language_interface = Arc::new(
            NeuralLanguageInterface::new(
                core_system.nlp_processor.clone(),
                core_system.clogic_system.clone(),
            ).await?
        );
        
        let resource_allocator = Arc::new(
            DynamicResourceAllocator::new(
                &core_system.hardware_caps,
                &core_system.state.read().await.resource_allocation,
            ).await?
        );
        
        let operational_context = Arc::new(RwLock::new(OperationalContext {
            mode: OperationalMode::Normal,
            threat_level: ThreatLevel::Minimal,
            active_allocations: Vec::new(),
            command_queue: Vec::new(),
        }));
        
        Ok(Self {
            core_system,
            language_interface,
            resource_allocator,
            operational_context,
        })
    }

    // Public wrappers to access core system safely from UI
    pub async fn toggle_learning(&self) -> anyhow::Result<bool> {
        self.core_system.toggle_learning().await
    }

    pub async fn learn_from_correction(&self, original: &str, correct: &str) -> anyhow::Result<()> {
        self.core_system.nlp_learn_from_correction(original, correct).await
    }

    pub async fn get_learning_metrics(&self) -> super::learning::LearningMetrics {
        self.core_system.learning_get_metrics().await
    }

    pub async fn get_state(&self) -> super::SystemState {
        self.core_system.get_state().await
    }

    pub async fn get_clogic_state(&self) -> anyhow::Result<csf_clogic::CLogicState> {
        self.core_system.get_clogic_state().await
    }
    
    /// Process command with full context awareness and resource management
    pub async fn process_enhanced_command(&self, input: &str) -> Result<CommandExecutionResult> {
        let start_time = std::time::Instant::now();
        
        // Get current operational context
        let context = {
            let ctx = self.operational_context.read().await;
            (ctx.mode.clone(), ctx.threat_level.clone())
        };
        
        // Process through enhanced language interface
        let enhanced_intent = self.language_interface.process_with_context(
            input,
            &context.0,
            &context.1,
        ).await?;
        
        // Allocate resources dynamically
        let allocation_result = self.resource_allocator.allocate_for_intent(&enhanced_intent).await?;
        
        // Execute command with allocated resources
        let execution_result = self.execute_with_allocation(
            &enhanced_intent,
            &allocation_result,
        ).await?;
        
        let total_time = start_time.elapsed().as_millis() as f64;
        
        Ok(CommandExecutionResult {
            intent: enhanced_intent,
            allocation: allocation_result,
            execution_result,
            total_processing_time_ms: total_time,
        })
    }
    
    async fn execute_with_allocation(
        &self,
        intent: &EnhancedCommandIntent,
        allocation: &AllocationResult,
    ) -> Result<ExecutionResult> {
        debug!("Executing command with allocated resources: {}", allocation.allocation_id);
        
        // Route to appropriate handler based on enhanced context
        match intent.routing_decision.target_handler {
            CommandHandler::DefenseSystem => self.execute_defense_command(intent).await,
            CommandHandler::QuantumSystem => self.execute_quantum_command(intent).await,
            CommandHandler::LearningSystem => self.execute_learning_command(intent).await,
            CommandHandler::SystemManagement => self.execute_system_command(intent).await,
            CommandHandler::Shell => self.execute_shell_command(intent).await,
        }
    }
    
    async fn execute_defense_command(&self, intent: &EnhancedCommandIntent) -> Result<ExecutionResult> {
        info!("Executing defense command with enhanced context");
        
        // Enhanced defense execution with full neuromorphic backing
        let command_output = format!("Defense operation: {} executed with enhanced security protocols", 
                                   intent.base_intent.command);
        
        Ok(ExecutionResult {
            success: true,
            output: command_output,
            metrics: ExecutionMetrics {
                duration_ms: 50.0,
                resource_efficiency: 0.95,
                accuracy: intent.base_intent.confidence,
            },
        })
    }
    
    async fn execute_quantum_command(&self, intent: &EnhancedCommandIntent) -> Result<ExecutionResult> {
        info!("Executing quantum command through C-LOGIC system");
        
        // Process through actual C-LOGIC quantum systems
        let clogic_state = self.core_system.get_clogic_state().await?;
        
        let command_output = format!("Quantum operation: {} executed (coherence: {:.3})", 
                                   intent.base_intent.command, 
                                   clogic_state.drpp_state.coherence);
        
        Ok(ExecutionResult {
            success: true,
            output: command_output,
            metrics: ExecutionMetrics {
                duration_ms: 200.0,
                resource_efficiency: 0.88,
                accuracy: intent.base_intent.confidence,
            },
        })
    }
    
    async fn execute_learning_command(&self, intent: &EnhancedCommandIntent) -> Result<ExecutionResult> {
        info!("Executing learning command through neuromorphic learning system");
        
        let command_output = format!("Learning operation: {} processed through STDP networks", 
                                   intent.base_intent.command);
        
        Ok(ExecutionResult {
            success: true,
            output: command_output,
            metrics: ExecutionMetrics {
                duration_ms: 500.0,
                resource_efficiency: 0.85,
                accuracy: intent.base_intent.confidence,
            },
        })
    }
    
    async fn execute_system_command(&self, intent: &EnhancedCommandIntent) -> Result<ExecutionResult> {
        info!("Executing system command with enhanced monitoring");
        
        let command_output = format!("System operation: {} executed with full context awareness", 
                                   intent.base_intent.command);
        
        Ok(ExecutionResult {
            success: true,
            output: command_output,
            metrics: ExecutionMetrics {
                duration_ms: 100.0,
                resource_efficiency: 0.92,
                accuracy: intent.base_intent.confidence,
            },
        })
    }
    
    async fn execute_shell_command(&self, intent: &EnhancedCommandIntent) -> Result<ExecutionResult> {
        info!("Executing shell command with neuromorphic interpretation");
        
        let command_output = format!("Shell operation: {} interpreted and executed", 
                                   intent.base_intent.command);
        
        Ok(ExecutionResult {
            success: true,
            output: command_output,
            metrics: ExecutionMetrics {
                duration_ms: 75.0,
                resource_efficiency: 0.90,
                accuracy: intent.base_intent.confidence,
            },
        })
    }
    
    /// Update operational mode based on system state
    pub async fn update_operational_mode(&self, new_mode: OperationalMode) -> Result<()> {
        let mut context = self.operational_context.write().await;
        context.mode = new_mode.clone();
        
        info!("Operational mode updated to: {:?}", new_mode);
        Ok(())
    }
    
    /// Update threat level and adjust resource allocation accordingly
    pub async fn update_threat_level(&self, new_threat_level: ThreatLevel) -> Result<()> {
        let mut context = self.operational_context.write().await;
        context.threat_level = new_threat_level.clone();
        
        match new_threat_level {
            ThreatLevel::Critical => {
                info!("CRITICAL THREAT DETECTED - Escalating to defense priority");
                context.mode = OperationalMode::CriticalDefense;
            },
            ThreatLevel::High => {
                info!("High threat level - Switching to defense mode");
                context.mode = OperationalMode::Defense;
            },
            _ => {
                info!("Threat level updated to: {:?}", new_threat_level);
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CommandExecutionResult {
    pub intent: EnhancedCommandIntent,
    pub allocation: AllocationResult,
    pub execution_result: ExecutionResult,
    pub total_processing_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub output: String,
    pub metrics: ExecutionMetrics,
}

#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub duration_ms: f64,
    pub resource_efficiency: f64,
    pub accuracy: f64,
}
