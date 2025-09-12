//! Optimization intent system for the Forge

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Unique identifier for optimization intents
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntentId(pub Uuid);

impl IntentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for IntentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of an optimization intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentStatus {
    /// Intent is being synthesized
    Synthesizing { progress: f64 },
    
    /// Module is being tested
    Testing { module_id: ModuleId },
    
    /// Awaiting human approval
    AwaitingApproval { risk_score: f64 },
    
    /// Module is being deployed
    Deploying { stage: String },
    
    /// Optimization completed successfully
    Completed { improvement: f64 },
    
    /// Optimization failed
    Failed { reason: String },
}

/// Formal optimization intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationIntent {
    /// Unique identifier
    pub id: IntentId,
    
    /// Target module or component
    pub target: OptimizationTarget,
    
    /// Optimization objectives
    pub objectives: Vec<Objective>,
    
    /// Constraints to satisfy
    pub constraints: Vec<Constraint>,
    
    /// Priority level
    pub priority: Priority,
    
    /// Optional deadline
    pub deadline: Option<Duration>,
    
    /// Optional synthesis strategy
    pub synthesis_strategy: Option<String>,
}

/// Target for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Specific module by ID
    Module(ModuleId),
    
    /// Module by name
    ModuleName(String),
    
    /// Component group
    ComponentGroup(String),
    
    /// System-wide optimization
    System,
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Objective {
    /// Minimize latency
    MinimizeLatency {
        percentile: f64,
        target_ms: f64,
    },
    
    /// Maximize throughput
    MaximizeThroughput {
        target_ops_per_sec: f64,
    },
    
    /// Reduce memory usage
    ReduceMemory {
        target_mb: usize,
    },
    
    /// Improve energy efficiency
    ImproveEnergy {
        reduction_percent: f64,
    },
    
    /// Custom objective
    Custom {
        name: String,
        weight: f64,
    },
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Maintain correctness
    MaintainCorrectness,
    
    /// Preserve specific invariant
    MaintainInvariant(String),
    
    /// Maximum complexity allowed
    MaxComplexity(f64),
    
    /// Maximum memory usage (MB)
    MaxMemoryMB(usize),
    
    /// Require formal proof
    RequireProof,
    
    /// Custom constraint
    Custom(String),
}

/// Priority levels for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Builder for OptimizationIntent
pub struct OptimizationIntentBuilder {
    target: Option<OptimizationTarget>,
    objectives: Vec<Objective>,
    constraints: Vec<Constraint>,
    priority: Priority,
    deadline: Option<Duration>,
    synthesis_strategy: Option<String>,
}

impl OptimizationIntentBuilder {
    pub fn new() -> Self {
        Self {
            target: None,
            objectives: Vec::new(),
            constraints: Vec::new(),
            priority: Priority::Medium,
            deadline: None,
            synthesis_strategy: None,
        }
    }
    
    pub fn target_module(mut self, module: impl Into<String>) -> Self {
        self.target = Some(OptimizationTarget::ModuleName(module.into()));
        self
    }
    
    pub fn target(mut self, target: OptimizationTarget) -> Self {
        self.target = Some(target);
        self
    }
    
    pub fn add_objective(mut self, objective: Objective) -> Self {
        self.objectives.push(objective);
        self
    }
    
    pub fn add_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }
    
    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
    
    pub fn synthesis_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.synthesis_strategy = Some(strategy.into());
        self
    }
    
    pub fn build(self) -> Result<OptimizationIntent, ForgeError> {
        let target = self.target.ok_or_else(|| {
            ForgeError::ConfigError("OptimizationIntent requires a target".into())
        })?;
        
        if self.objectives.is_empty() {
            return Err(ForgeError::ConfigError(
                "OptimizationIntent requires at least one objective".into()
            ));
        }
        
        Ok(OptimizationIntent {
            id: IntentId::new(),
            target,
            objectives: self.objectives,
            constraints: self.constraints,
            priority: self.priority,
            deadline: self.deadline,
            synthesis_strategy: self.synthesis_strategy,
        })
    }
}

impl OptimizationIntent {
    pub fn builder() -> OptimizationIntentBuilder {
        OptimizationIntentBuilder::new()
    }
}

/// Optimization opportunity detected by DRPP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub module_id: ModuleId,
    pub opportunity_type: OpportunityType,
    pub expected_improvement: f64,
    pub confidence: f64,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Types of optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunityType {
    LoopOptimization,
    MemoryLayoutOptimization,
    AlgorithmSubstitution,
    ParallelizationOpportunity,
    CacheOptimization,
    VectorizationOpportunity,
    DeadCodeElimination,
}

/// Domain for specialized optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDomain {
    Cryptography,
    MachineLearning,
    Networking,
    DataProcessing,
    General,
}