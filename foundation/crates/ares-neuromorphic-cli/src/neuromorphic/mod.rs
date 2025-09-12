//! Unified Neuromorphic System - Self-contained AI with Brian2/Lava integration
//!
//! This module provides the core neuromorphic computing layer that enables
//! natural language command processing without external AI dependencies.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use csf_clogic::{CLogicSystem, CLogicConfig};
use csf_hardware::neuromorphic::{detect_loihi_chips, NeuromorphicConfig};
use csf_bus::PhaseCoherenceBus;

pub mod backend;
pub mod nlp;
pub mod hardware;
pub mod learning;
// Python bridge is enabled only when `python-bridge` feature is set and native-only is off.
// For now, route to a minimal PyO3-backed bridge that compiles cleanly.
#[cfg(all(feature = "python-bridge", not(feature = "native-only")))]
pub mod python_bridge_min;
#[cfg(all(feature = "python-bridge", not(feature = "native-only")))]
pub use python_bridge_min as python_bridge;
// Otherwise, use a lightweight stub.
#[cfg(any(feature = "native-only", not(feature = "python-bridge")))]
pub mod python_bridge { pub use super::python_bridge_stub::*; }
#[cfg(any(feature = "native-only", not(feature = "python-bridge")))]
mod python_bridge_stub;
pub mod unified_system;
pub mod performance;

pub use backend::{NeuromorphicBackend, BackendCapabilities, BackendMetrics, ProcessingSpeed};
pub use nlp::{NeuralLanguageProcessor, CommandIntent, CommandContext, CommandDomain};
pub use hardware::{HardwareDetector, HardwareCapabilities};
pub use learning::{LearnMode, TrainingSample, LearningMetrics};
pub use unified_system::{
    EnhancedUnifiedNeuromorphicSystem, NeuralLanguageInterface, DynamicResourceAllocator,
    CommandExecutionResult, EnhancedCommandIntent
};
pub use performance::{PerformanceOptimizer, PerformanceMetrics, MemoryManager, OptimizationConfig};

/// Configuration for the unified neuromorphic system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuromorphicSystemConfig {
    /// Backend selection strategy
    pub backend_strategy: BackendStrategy,
    
    /// Resource allocation for NLP (0.0 - 1.0)
    pub nlp_allocation: f64,
    
    /// Brian2 configuration
    pub brian2: Brian2Config,
    
    /// Lava configuration
    pub lava: LavaConfig,
    
    /// Learning configuration
    pub learning: LearningConfig,
    
    /// Hardware preferences
    pub hardware: HardwarePreferences,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BackendStrategy {
    /// Auto-detect optimal backend
    Auto,
    /// Force specific backend
    Force(crate::cli::NeuromorphicBackend),
    /// Use multiple backends for verification
    Hybrid { primary: crate::cli::NeuromorphicBackend, fallback: crate::cli::NeuromorphicBackend },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Brian2Config {
    /// Preferred device (auto, cpu, cuda)
    pub device: String,
    /// Number of CPU threads for simulation
    pub threads: Option<usize>,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
    /// Optimization level
    pub optimization: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LavaConfig {
    /// Prefer hardware when available
    pub prefer_hardware: bool,
    /// Simulation precision (fp16, fp32)
    pub precision: String,
    /// Enable GPU simulation
    pub gpu_simulation: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearningConfig {
    /// STDP learning rate
    pub learning_rate: f64,
    /// Pattern confidence threshold
    pub confidence_threshold: f64,
    /// Maximum patterns to store
    pub max_patterns: usize,
    /// Enable online learning
    pub online_learning: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwarePreferences {
    /// Preferred GPU device ID
    pub gpu_device: Option<u32>,
    /// Minimum GPU memory (GB)
    pub min_gpu_memory: f32,
    /// Enable neuromorphic chip detection
    pub detect_neuromorphic_chips: bool,
}

impl Default for NeuromorphicSystemConfig {
    fn default() -> Self {
        Self {
            backend_strategy: BackendStrategy::Auto,
            nlp_allocation: 0.15,
            brian2: Brian2Config {
                device: "auto".to_string(),
                threads: None,
                use_gpu: true,
                optimization: "O3".to_string(),
            },
            lava: LavaConfig {
                prefer_hardware: true,
                precision: "fp32".to_string(),
                gpu_simulation: true,
            },
            learning: LearningConfig {
                learning_rate: 0.01,
                confidence_threshold: 0.85,
                max_patterns: 10000,
                online_learning: true,
            },
            hardware: HardwarePreferences {
                gpu_device: None,
                min_gpu_memory: 4.0,
                detect_neuromorphic_chips: true,
            },
        }
    }
}

/// Main neuromorphic system that coordinates all subsystems
#[derive(Clone)]
pub struct UnifiedNeuromorphicSystem {
    /// Current active backend
    backend: Arc<RwLock<backend::NeuromorphicBackend>>,
    
    /// Natural language processor
    nlp_processor: Arc<nlp::NeuralLanguageProcessor>,
    
    /// Hardware capabilities
    hardware_caps: hardware::HardwareCapabilities,
    
    /// C-LOGIC system integration
    clogic_system: Arc<CLogicSystem>,
    
    /// Learning subsystem
    learning_system: Arc<learning::LearnMode>,
    
    /// Configuration
    config: NeuromorphicSystemConfig,
    
    /// System state
    state: Arc<RwLock<SystemState>>,
}

#[derive(Debug, Clone)]
pub struct SystemState {
    /// Backend information
    pub backend_info: String,
    /// Active learning status
    pub learning_active: bool,
    /// Processed commands count
    pub commands_processed: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Current resource allocation
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub nlp: f64,
    pub drpp: f64,
    pub ems: f64,
    pub adp: f64,
    pub egc: f64,
}

impl UnifiedNeuromorphicSystem {
    /// Initialize the neuromorphic system with optimal backend
    pub async fn initialize(config_path: Option<&std::path::Path>) -> Result<Self> {
        info!("Initializing ARES Neuromorphic System");
        
        // Load configuration
        let config = if let Some(path) = config_path {
            Self::load_config(path).await?
        } else {
            NeuromorphicSystemConfig::default()
        };
        
        debug!("Using neuromorphic config: {:?}", config);
        
        // Detect hardware capabilities
        let hardware_caps = hardware::HardwareDetector::detect().await?;
        info!("Detected hardware: {:?}", hardware_caps);
        
        // Initialize C-LOGIC system
        let bus = Arc::new(PhaseCoherenceBus::new(Default::default())?);
        let clogic_config = CLogicConfig::default();
        let clogic_system = Arc::new(CLogicSystem::new(bus.clone(), clogic_config).await?);
        
        // Select optimal neuromorphic backend
        let backend = Self::select_backend(&config, &hardware_caps).await?;
        let backend_info = backend.info();
        info!("Selected neuromorphic backend: {}", backend_info);
        
        // Wrap backend in Arc<RwLock<>> for shared access
        let backend_arc = Arc::new(RwLock::new(backend));
        
        // Initialize NLP processor
        let nlp_processor = Arc::new(
            nlp::NeuralLanguageProcessor::new(
                backend_arc.clone(),
                clogic_system.clone(),
                config.learning.clone()
            ).await?
        );
        
        // Initialize learning system
        let learning_system = Arc::new(
            learning::LearnMode::new(
                backend_arc.clone(),
                config.learning.clone()
            ).await?
        );
        
        // Initialize system state
        let state = Arc::new(RwLock::new(SystemState {
            backend_info,
            learning_active: false,
            commands_processed: 0,
            avg_processing_time_ms: 0.0,
            resource_allocation: ResourceAllocation {
                nlp: config.nlp_allocation,
                drpp: 0.25,
                ems: 0.20,
                adp: 0.20,
                egc: 0.20,
            },
        }));
        
        let system = Self {
            backend: backend_arc,
            nlp_processor,
            hardware_caps,
            clogic_system,
            learning_system,
            config,
            state,
        };
        
        // Start C-LOGIC modules
        system.clogic_system.start().await?;
        
        info!("ARES Neuromorphic System initialized successfully");
        Ok(system)
    }
    
    /// Process natural language input through neuromorphic network
    pub async fn process_natural_language(&self, input: &str) -> Result<CommandIntent> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing natural language input: '{}'", input);
        
        // Process through NLP layer
        let intent = self.nlp_processor.process_input(input).await?;
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        {
            let mut state = self.state.write().await;
            state.commands_processed += 1;
            state.avg_processing_time_ms = 
                (state.avg_processing_time_ms + processing_time) / 2.0;
        }
        
        debug!("Processed command intent: {:?}", intent);
        Ok(intent)
    }
    
    /// Get current system information
    pub fn backend_info(&self) -> String {
        format!(
            "Backend: {}, Hardware: {:?}, Learning: {}",
            self.config.backend_strategy,
            self.hardware_caps.summary(),
            if self.learning_system.is_active() { "Active" } else { "Inactive" }
        )
    }
    
    /// Toggle learning mode
    pub async fn toggle_learning(&self) -> Result<bool> {
        let is_active = self.learning_system.toggle().await?;
        
        let mut state = self.state.write().await;
        state.learning_active = is_active;
        
        if is_active {
            info!("🧠 Learning mode ACTIVATED - System will learn from your commands");
        } else {
            info!("📚 Learning mode DEACTIVATED - Knowledge consolidated");
        }
        
        Ok(is_active)
    }

    // ---- Convenience wrappers to encapsulate internals ----
    pub async fn nlp_learn_from_correction(&self, original_input: &str, correct_command: &str) -> Result<()> {
        self.nlp_processor.learn_from_correction(original_input, correct_command).await
    }

    pub async fn learning_get_metrics(&self) -> learning::LearningMetrics {
        self.learning_system.get_metrics().await
    }

    pub async fn learning_export_patterns(&self) -> Result<Vec<learning::TrainingSample>> {
        self.learning_system.export_patterns().await
    }

    pub async fn learning_record_sample(
        &self,
        input: &str,
        command: &str,
        success: bool,
        context: learning::TrainingContext,
    ) -> Result<()> {
        self.learning_system
            .record_training_sample(input, command, success, context)
            .await
    }
    
    /// Get current system state
    pub async fn get_state(&self) -> SystemState {
        self.state.read().await.clone()
    }
    
    /// Get C-LOGIC system state
    pub async fn get_clogic_state(&self) -> Result<csf_clogic::CLogicState> {
        Ok(self.clogic_system.get_state().await)
    }
    
    async fn load_config(path: &std::path::Path) -> Result<NeuromorphicSystemConfig> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: NeuromorphicSystemConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    async fn select_backend(
        config: &NeuromorphicSystemConfig,
        hardware_caps: &hardware::HardwareCapabilities,
    ) -> Result<backend::NeuromorphicBackend> {
        match &config.backend_strategy {
            BackendStrategy::Auto => {
                backend::auto_select_backend(hardware_caps, config).await
            },
            BackendStrategy::Force(backend_type) => {
                backend::create_backend(backend_type.clone(), hardware_caps, config).await
            },
            BackendStrategy::Hybrid { primary, fallback } => {
                backend::create_hybrid_backend(
                    primary.clone(),
                    fallback.clone(),
                    hardware_caps,
                    config
                ).await
            },
        }
    }
}

impl std::fmt::Display for BackendStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendStrategy::Auto => write!(f, "Auto-detect"),
            BackendStrategy::Force(backend) => write!(f, "Force {:?}", backend),
            BackendStrategy::Hybrid { primary, fallback } => {
                write!(f, "Hybrid ({:?} -> {:?})", primary, fallback)
            },
        }
    }
}
