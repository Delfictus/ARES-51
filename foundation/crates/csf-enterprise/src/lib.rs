//! ARES ChronoFabric Enterprise Intake and Management System
//!
//! Provides enterprise-grade onboarding, data intake, format conversion,
//! intent validation, and live phase lattice monitoring capabilities.

pub mod cli;
pub mod conversion;
pub mod intake;
pub mod intent;
pub mod lattice;
pub mod roe;
pub mod web;

use csf_core::prelude::*;
use csf_sil::SilCore;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Enterprise system configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnterpriseConfig {
    /// Web server configuration
    pub web: WebConfig,
    
    /// File upload configuration
    pub upload: UploadConfig,
    
    /// Data conversion settings
    pub conversion: ConversionConfig,
    
    /// Intent validation settings
    pub intent: IntentConfig,
    
    /// Phase lattice monitoring
    pub lattice: LatticeConfig,
    
    /// Rules of Engagement settings
    pub roe: RoeConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebConfig {
    pub host: String,
    pub port: u16,
    pub max_upload_size: usize,
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UploadConfig {
    pub temp_dir: String,
    pub max_file_size: usize,
    pub allowed_formats: Vec<String>,
    pub batch_size: usize,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversionConfig {
    pub auto_detect_schema: bool,
    pub validation_threshold: f64,
    pub max_conversion_attempts: usize,
    pub preserve_metadata: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentConfig {
    pub confirmation_required: bool,
    pub max_questions: usize,
    pub confidence_threshold: f64,
    pub auto_approve_threshold: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatticeConfig {
    pub update_interval_ms: u64,
    pub history_retention_hours: u64,
    pub alert_thresholds: AlertThresholds,
    pub visualization_depth: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlertThresholds {
    pub coherence_loss: f64,
    pub phase_deviation: f64,
    pub temporal_drift: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoeConfig {
    pub default_engagement_level: String,
    pub audit_retention_days: u32,
    pub approval_workflows: Vec<String>,
    pub escalation_thresholds: Vec<f64>,
}

impl Default for EnterpriseConfig {
    fn default() -> Self {
        Self {
            web: WebConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                max_upload_size: 100 * 1024 * 1024, // 100MB
                cors_origins: vec!["*".to_string()],
            },
            upload: UploadConfig {
                temp_dir: "/tmp/ares-uploads".to_string(),
                max_file_size: 50 * 1024 * 1024, // 50MB
                allowed_formats: vec![
                    "json".to_string(),
                    "csv".to_string(),
                    "xml".to_string(),
                    "yaml".to_string(),
                    "parquet".to_string(),
                ],
                batch_size: 1000,
                timeout_seconds: 300,
            },
            conversion: ConversionConfig {
                auto_detect_schema: true,
                validation_threshold: 0.95,
                max_conversion_attempts: 3,
                preserve_metadata: true,
            },
            intent: IntentConfig {
                confirmation_required: true,
                max_questions: 5,
                confidence_threshold: 0.8,
                auto_approve_threshold: 0.95,
            },
            lattice: LatticeConfig {
                update_interval_ms: 100,
                history_retention_hours: 24,
                alert_thresholds: AlertThresholds {
                    coherence_loss: 0.05,
                    phase_deviation: 0.1,
                    temporal_drift: 1000.0, // nanoseconds
                },
                visualization_depth: 10,
            },
            roe: RoeConfig {
                default_engagement_level: "standard".to_string(),
                audit_retention_days: 365,
                approval_workflows: vec!["auto".to_string(), "manual".to_string(), "escalated".to_string()],
                escalation_thresholds: vec![0.7, 0.9, 0.99],
            },
        }
    }
}

/// Main enterprise system orchestrator
pub struct EnterpriseSystem {
    config: EnterpriseConfig,
    intake_service: Arc<intake::IntakeService>,
    conversion_engine: Arc<conversion::ConversionEngine>,
    intent_validator: Arc<intent::IntentValidator>,
    lattice_monitor: Arc<lattice::LatticeMonitor>,
    sil_core: Arc<SilCore>,
    roe_manager: Arc<roe::RoeManager>,
}

impl EnterpriseSystem {
    /// Create new enterprise system
    pub async fn new(
        config: EnterpriseConfig,
        sil_core: Arc<SilCore>,
    ) -> anyhow::Result<Self> {
        let intake_service = Arc::new(
            intake::IntakeService::new(config.upload.clone()).await?
        );
        
        let conversion_engine = Arc::new(
            conversion::ConversionEngine::new(config.conversion.clone())?
        );
        
        let intent_validator = Arc::new(
            intent::IntentValidator::new(config.intent.clone())?
        );
        
        let lattice_monitor = Arc::new(
            lattice::LatticeMonitor::new(config.lattice.clone()).await?
        );
        
        let roe_manager = Arc::new(
            roe::RoeManager::new(config.roe.clone(), sil_core.clone()).await?
        );

        Ok(Self {
            config,
            intake_service,
            conversion_engine,
            intent_validator,
            lattice_monitor,
            sil_core,
            roe_manager,
        })
    }

    /// Start the enterprise system
    pub async fn start(&self) -> anyhow::Result<()> {
        tracing::info!("Starting ARES Enterprise System");
        
        // Start all subsystems
        self.intake_service.start().await?;
        self.lattice_monitor.start().await?;
        self.roe_manager.start().await?;
        
        tracing::info!("Enterprise system started successfully");
        Ok(())
    }

    /// Stop the enterprise system
    pub async fn stop(&self) -> anyhow::Result<()> {
        tracing::info!("Stopping ARES Enterprise System");
        
        self.intake_service.stop().await?;
        self.lattice_monitor.stop().await?;
        self.roe_manager.stop().await?;
        
        tracing::info!("Enterprise system stopped");
        Ok(())
    }
}

/// Enterprise system errors
#[derive(Debug, thiserror::Error)]
pub enum EnterpriseError {
    #[error("Configuration error: {details}")]
    Configuration { details: String },
    
    #[error("File processing error: {reason}")]
    FileProcessing { reason: String },
    
    #[error("Data conversion failed: {format} -> PhasePacket")]
    ConversionFailed { format: String },
    
    #[error("Intent validation failed: {reason}")]
    IntentValidation { reason: String },
    
    #[error("Execution error: {operation}")]
    ExecutionFailed { operation: String },
    
    #[error("Lattice monitoring error: {details}")]
    LatticeError { details: String },
    
    #[error("ROE violation: {rule} - {details}")]
    RoeViolation { rule: String, details: String },
    
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },
    
    #[error("Authorization denied: {resource}")]
    AuthorizationDenied { resource: String },
    
    #[error("Internal system error: {details}")]
    Internal { details: String },
}

pub type EnterpriseResult<T> = Result<T, EnterpriseError>;