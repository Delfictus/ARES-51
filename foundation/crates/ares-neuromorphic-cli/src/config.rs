//! Configuration management for the neuromorphic CLI

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, info};

// In normal builds, use the full neuromorphic config type.
#[cfg(not(feature = "status-only"))]
use crate::neuromorphic::NeuromorphicSystemConfig;

// In status-only builds, provide a minimal local stand-in so the CLI can compile
// without the full neuromorphic stack.
#[cfg(feature = "status-only")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicSystemConfig {
    /// Resource allocation for NLP (0.0 - 1.0)
    pub nlp_allocation: f64,
    /// Minimal learning configuration
    pub learning: LearningConfig,
}

#[cfg(feature = "status-only")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// STDP learning rate (minimal field for validation)
    pub learning_rate: f64,
}

#[cfg(feature = "status-only")]
impl Default for NeuromorphicSystemConfig {
    fn default() -> Self {
        Self {
            nlp_allocation: 0.15,
            learning: LearningConfig { learning_rate: 0.01 },
        }
    }
}

/// Complete CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Neuromorphic system configuration
    pub neuromorphic: NeuromorphicSystemConfig,
    
    /// CLI interface settings
    pub interface: InterfaceConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Security settings
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    /// Default output format
    pub default_format: String,
    
    /// Enable colored output
    pub colored_output: bool,
    
    /// Auto-confirm low-risk commands
    pub auto_confirm_threshold: f64,
    
    /// Show processing time in output
    pub show_timing: bool,
    
    /// Interactive mode settings
    pub interactive: InteractiveConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Show welcome message
    pub show_welcome: bool,
    
    /// Enable command history
    pub enable_history: bool,
    
    /// History file path
    pub history_file: Option<PathBuf>,
    
    /// Maximum history entries
    pub max_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Default log level
    pub level: String,
    
    /// Log to file
    pub log_file: Option<PathBuf>,
    
    /// Enable structured logging (JSON)
    pub structured: bool,
    
    /// Log neuromorphic operations
    pub log_neuromorphic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Require confirmation for destructive commands
    pub confirm_destructive: bool,
    
    /// Sandbox unknown commands
    pub sandbox_unknown: bool,
    
    /// Maximum command execution time (seconds)
    pub max_execution_time: u64,
    
    /// Enable audit logging
    pub audit_logging: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            neuromorphic: NeuromorphicSystemConfig::default(),
            interface: InterfaceConfig {
                default_format: "human".to_string(),
                colored_output: true,
                auto_confirm_threshold: 0.9,
                show_timing: false,
                interactive: InteractiveConfig {
                    show_welcome: true,
                    enable_history: true,
                    history_file: None, // Will use default location
                    max_history: 1000,
                },
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                log_file: None,
                structured: false,
                log_neuromorphic: true,
            },
            security: SecurityConfig {
                confirm_destructive: true,
                sandbox_unknown: true,
                max_execution_time: 300, // 5 minutes
                audit_logging: true,
            },
        }
    }
}

impl CliConfig {
    /// Load configuration from file
    pub async fn load(path: &std::path::Path) -> Result<Self> {
        debug!("Loading configuration from: {:?}", path);
        
        let content = tokio::fs::read_to_string(path).await?;
        let config: CliConfig = toml::from_str(&content)?;
        
        info!("Configuration loaded successfully");
        Ok(config)
    }
    
    /// Save configuration to file
    pub async fn save(&self, path: &std::path::Path) -> Result<()> {
        debug!("Saving configuration to: {:?}", path);
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        let content = toml::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        
        info!("Configuration saved successfully");
        Ok(())
    }
    
    /// Get default configuration file path
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;
        
        Ok(config_dir.join("ares").join("neuromorphic-cli.toml"))
    }
    
    /// Load configuration with fallback to default
    pub async fn load_or_default(path: Option<&std::path::Path>) -> Result<Self> {
        let config_path = if let Some(path) = path {
            path.to_path_buf()
        } else {
            Self::default_config_path()?
        };
        
        match Self::load(&config_path).await {
            Ok(config) => {
                debug!("Loaded configuration from: {:?}", config_path);
                Ok(config)
            },
            Err(e) => {
                debug!("Failed to load config ({}), using defaults", e);
                
                // Save default configuration for future use
                let default_config = Self::default();
                if let Err(save_err) = default_config.save(&config_path).await {
                    debug!("Failed to save default config: {}", save_err);
                }
                
                Ok(default_config)
            }
        }
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate neuromorphic configuration
        if self.neuromorphic.nlp_allocation < 0.1 || self.neuromorphic.nlp_allocation > 0.9 {
            return Err(anyhow::anyhow!(
                "Invalid NLP allocation: {} (must be 0.1-0.9)",
                self.neuromorphic.nlp_allocation
            ));
        }
        
        if self.neuromorphic.learning.learning_rate <= 0.0 || self.neuromorphic.learning.learning_rate > 1.0 {
            return Err(anyhow::anyhow!(
                "Invalid learning rate: {} (must be 0.0-1.0)",
                self.neuromorphic.learning.learning_rate
            ));
        }
        
        // Validate interface configuration
        if self.interface.auto_confirm_threshold < 0.0 || self.interface.auto_confirm_threshold > 1.0 {
            return Err(anyhow::anyhow!(
                "Invalid auto-confirm threshold: {} (must be 0.0-1.0)",
                self.interface.auto_confirm_threshold
            ));
        }
        
        // Validate security configuration
        if self.security.max_execution_time == 0 {
            return Err(anyhow::anyhow!("Max execution time must be greater than 0"));
        }
        
        Ok(())
    }
}
