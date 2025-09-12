//! Configuration management for ChronoFabric

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronoFabricConfig {
    /// System configuration
    pub system: SystemConfig,

    /// Bus configuration
    pub bus: BusConfig,

    /// CLogic configuration
    pub clogic: CLogicConfig,

    /// MLIR configuration
    pub mlir: MlirConfig,

    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Number of worker threads
    pub worker_threads: usize,

    /// Realtime scheduling
    pub realtime: bool,

    /// Memory pool (MB)
    pub memory_pool_mb: usize,

    /// Hardware acceleration enabled
    pub hardware_accel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusConfig {
    /// Channel buffer size for subscribers
    pub channel_buffer_size: usize,

    /// Maximum packet size in bytes
    pub max_packet_size: usize,

    /// Estimated routing table size
    pub routing_table_size: usize,

    /// Enable phase coherence
    pub phase_coherence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLogicConfig {
    pub drpp: DrppConfig,
    pub adp: AdpConfig,
    pub egc: EgcConfig,
    pub ems: EmsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrppConfig {
    pub num_variables: usize,
    pub history_size: usize,
    pub update_interval_ms: u64,
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdpConfig {
    pub nn_layers: Vec<usize>,
    pub learning_rate: f64,
    pub decision_timeout_ms: u64,
    pub quantum_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgcConfig {
    pub consensus_threshold: f64,
    pub max_policies: usize,
    pub evaluation_interval_ms: u64,
    pub auto_rules: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmsConfig {
    pub dimensions: usize,
    pub update_frequency: f64,
    pub decay_rate: f64,
    pub social_modeling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlirConfig {
    pub jit_enabled: bool,
    pub opt_level: u8,
    pub backends: Vec<String>,
    pub memory_pool_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub metrics_enabled: bool,
    pub metrics_endpoint: String,
    pub collection_interval_ms: u64,
    pub tracing_enabled: bool,
}

impl Default for ChronoFabricConfig {
    fn default() -> Self {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            system: SystemConfig {
                worker_threads: num_cores,
                realtime: false,
                memory_pool_mb: 1024,
                hardware_accel: true,
            },
            bus: BusConfig {
                channel_buffer_size: 1024,
                max_packet_size: 64 * 1024,
                routing_table_size: 10000,
                phase_coherence: true,
            },
            clogic: CLogicConfig {
                drpp: DrppConfig {
                    num_variables: 64,
                    history_size: 1000,
                    update_interval_ms: 10,
                    use_gpu: true,
                },
                adp: AdpConfig {
                    nn_layers: vec![128, 64, 32, 16],
                    learning_rate: 0.001,
                    decision_timeout_ms: 50,
                    quantum_enabled: true,
                },
                egc: EgcConfig {
                    consensus_threshold: 0.66,
                    max_policies: 1000,
                    evaluation_interval_ms: 100,
                    auto_rules: true,
                },
                ems: EmsConfig {
                    dimensions: 8,
                    update_frequency: 10.0,
                    decay_rate: 0.95,
                    social_modeling: true,
                },
            },
            mlir: MlirConfig {
                jit_enabled: true,
                opt_level: 2,
                backends: vec!["cpu".to_string(), "gpu".to_string()],
                memory_pool_mb: 512,
            },
            telemetry: TelemetryConfig {
                metrics_enabled: true,
                metrics_endpoint: "127.0.0.1:9090".to_string(),
                collection_interval_ms: 1000,
                tracing_enabled: true,
            },
        }
    }
}

impl ChronoFabricConfig {
    /// Load configuration from file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // System validation
        if self.system.worker_threads == 0 {
            anyhow::bail!("Worker threads must be > 0");
        }

        // Bus validation
        if self.bus.max_packet_size > 16 * 1024 * 1024 {
            anyhow::bail!("Max packet size too large (> 16MB)");
        }

        // DRPP validation
        if self.clogic.drpp.num_variables == 0 {
            anyhow::bail!("DRPP variables must be > 0");
        }

        // ADP validation
        if self.clogic.adp.nn_layers.is_empty() {
            anyhow::bail!("ADP neural network must have layers");
        }

        Ok(())
    }
}
