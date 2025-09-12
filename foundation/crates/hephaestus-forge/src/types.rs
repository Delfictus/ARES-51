//! Core types for the Hephaestus Forge MES

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Module identifier for tracking versions
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModuleId(pub String);

/// Versioned module with metadata
#[derive(Debug, Clone)]
pub struct VersionedModule {
    pub metadata: ModuleMetadata,
    pub bytecode: Vec<u8>,
    pub dependencies: Vec<ModuleId>,
    pub performance_profile: PerformanceProfile,
    pub safety_invariants: Vec<SafetyInvariant>,
    pub proof_certificate: Option<ProofCertificate>,
}

/// Proof certificate for Proof-Carrying Code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub smt_proof: Vec<u8>,
    pub invariants: Vec<SafetyInvariant>,
    pub solver_used: String,
    pub verification_time_ms: u64,
}

/// Module metadata for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub id: ModuleId,
    pub version: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub risk_score: f64,
    pub complexity_score: f64,
}

/// Performance characteristics of a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub cpu_usage_percent: f64,
    pub memory_mb: u64,
    pub latency_p99_ms: f64,
    pub throughput_ops_per_sec: u64,
}

/// Safety invariant that must be maintained
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyInvariant {
    pub id: String,
    pub description: String,
    pub smt_formula: String,
    pub criticality: InvariantCriticality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantCriticality {
    Low,
    Medium,
    High,
    Critical,
}

/// Safety constraint for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraint {
    pub constraint_type: ConstraintType,
    pub specification: String,
    pub enforcement: EnforcementLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MemoryBounds,
    ExecutionTime,
    ResourceUsage,
    SecurityPolicy,
    ComplianceRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Advisory,
    Required,
    Mandatory,
}

/// Deployment strategy for module integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    Immediate,
    Canary { stages: Vec<f64> },
    Shadow { duration_ms: u64 },
    BlueGreen,
}

/// Configuration for the Forge
#[derive(Debug, Clone, Default)]
pub struct ForgeConfig {
    pub runtime_config: RuntimeConfig,
    pub sandbox_config: SandboxConfig,
    pub ledger_config: LedgerConfig,
    pub drpp_config: DrppConfig,
    pub synthesis_config: SynthesisConfig,
    pub validation_config: ValidationConfig,
    pub consensus_config: ConsensusConfig,
    pub executor_config: ExecutorConfig,
    pub monitor_config: MonitorConfig,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    pub max_concurrent_swaps: usize,
    pub rollback_window_ms: u64,
    pub shadow_traffic_percent: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SandboxConfig {
    pub isolation_type: IsolationType,
    pub resource_limits: ResourceLimits,
    pub network_isolation: bool,
}

#[derive(Debug, Clone, Default)]
pub enum IsolationType {
    #[default]
    Process,
    Container,
    FirecrackerVM,
    HardwareSecure,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceLimits {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub disk_mb: u64,
    pub network_mbps: u64,
}

#[derive(Debug, Clone, Default)]
pub struct LedgerConfig {
    pub consensus_timeout_ms: u64,
    pub min_validators: usize,
    pub risk_threshold: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DrppConfig {
    pub detection_sensitivity: f64,
    pub pattern_window_size: usize,
    pub energy_threshold: f64,
    pub enable_resonance: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisConfig {
    pub max_synthesis_time_ms: u64,
    pub smt_solver: SmtSolver,
    pub search_strategy: SearchStrategy,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SmtSolver {
    #[default]
    Z3,
    CVC5,
    Yices,
    Multi,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SearchStrategy {
    #[default]
    MCTS,
    BeamSearch,
    GeneticAlgorithm,
    HybridNeuralSymbolic,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub test_timeout_ms: u64,
    pub chaos_engineering: bool,
    pub differential_testing: bool,
    pub property_testing: bool,
    pub parallel_threads: usize,
    pub max_test_cases_per_property: usize,
    pub memory_limit_mb: usize,
    pub regression_threshold: f64,
    pub shrinking_enabled: bool,
    pub max_shrinking_attempts: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ConsensusConfig {
    pub pbft_enabled: bool,
    pub quantum_resistant: bool,
    pub approval_quorum: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorConfig {
    pub synthesis_threads: usize,
    pub validation_threads: usize,
    pub integration_threads: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MonitorConfig {
    pub metrics_interval_ms: u64,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub error_rate_percent: f64,
    pub latency_p99_ms: f64,
    pub memory_usage_percent: f64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub response_time_ms: u64,
}

/// Errors for the Forge
#[derive(Debug, thiserror::Error)]
pub enum ForgeError {
    #[error("Synthesis failed: {0}")]
    SynthesisError(String),
    
    #[error("Validation failed: {0}")]
    ValidationError(String),
    
    #[error("Consensus failed: {0}")]
    ConsensusError(String),
    
    #[error("Integration failed: {0}")]
    IntegrationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    
    #[error("Deployment error: {0}")]
    DeploymentError(String),
    
    #[error("Code generation error: {0}")]
    CodeGenerationError(String),
    
    #[error("Safety violation: {0}")]
    SafetyViolation(String),
    
    #[error("Execution timeout")]
    ExecutionTimeout,
    
    #[error("Resource exhaustion")]
    ResourceExhaustion,
    
    #[error("Network unavailable")]
    NetworkUnavailable,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    // Approval workflow errors
    #[error("Unauthorized approver")]
    UnauthorizedApprover,
    
    #[error("Approval request not found")]
    ApprovalNotFound,
    
    #[error("Insufficient permissions")]
    InsufficientPermissions,
    
    #[error("Invalid cryptographic signature")]
    InvalidSignature,
    
    #[error("Cryptographic operation failed")]
    CryptographicError,
    
    #[error("Approval timeout")]
    ApprovalTimeout,
    
    #[error("Approval rejected")]
    ApprovalRejected,
}

/// Result type for Forge operations
pub type ForgeResult<T> = Result<T, ForgeError>;

/// Metamorphic transaction for ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphicTransaction {
    pub id: Uuid,
    pub module_id: ModuleId,
    pub change_type: ChangeType,
    pub risk_score: f64,
    pub proof: Option<ProofCertificate>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    ModuleOptimization,
    ArchitecturalRefactor,
    SecurityPatch,
    PerformanceEnhancement,
}

/// Approval levels for changes
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalLevel {
    FullyAutomated,
    LowRiskAutomated,
    MediumRiskReview,
    HighRiskReview,
    HumanRequired,
}

/// Risk assessment for changes
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub score: f64,
    pub factors: Vec<RiskFactor>,
    pub mitigation_strategy: MitigationStrategy,
}

#[derive(Debug, Clone)]
pub struct RiskFactor {
    pub factor_type: RiskFactorType,
    pub weight: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum RiskFactorType {
    ModuleCriticality,
    ChangeComplexity,
    VerificationConfidence,
    SystemState,
}

#[derive(Debug, Clone)]
pub enum MitigationStrategy {
    None,
    Canary,
    Shadow,
    Rollback,
    Manual,
}

/// Human approval workflow types
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
    Escalated,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalRole {
    TechnicalLead,
    SecurityOfficer,
    ExecutiveOfficer,
    SystemAdministrator,
    QualityAssurance,
}

#[derive(Debug, Clone)]
pub struct ApprovalSignature {
    pub signer_id: String,
    pub role: ApprovalRole,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub nonce: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ApprovalAuditEntry {
    pub id: uuid::Uuid,
    pub transaction_id: uuid::Uuid,
    pub action: ApprovalAction,
    pub actor: String,
    pub role: ApprovalRole,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: ApprovalSignature,
    pub previous_hash: Vec<u8>,
    pub entry_hash: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum ApprovalAction {
    Requested,
    Approved,
    Rejected,
    Escalated,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct ApprovalWorkflowConfig {
    pub timeout_seconds: u64,
    pub escalation_timeout_seconds: u64,
    pub required_roles: Vec<ApprovalRole>,
    pub minimum_approvals: usize,
    pub allow_self_approval: bool,
    pub require_unanimous: bool,
}

#[derive(Debug, Clone)]
pub struct ApprovalNotification {
    pub recipient: String,
    pub role: ApprovalRole,
    pub transaction_id: uuid::Uuid,
    pub urgency: NotificationUrgency,
    pub message: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NotificationUrgency {
    Low,
    Medium,
    High,
    Critical,
}