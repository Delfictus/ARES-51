//! ARES Neuromorphic CLI - Natural Language Processing for Quantum Systems
//!
//! This library provides a self-contained neuromorphic command interface that
//! processes natural language input through Brian2/Lava neuromorphic simulation
//! integrated with the ARES C-LOGIC cognitive modules.

pub mod cli;
pub mod commands;
#[cfg(not(feature = "status-only"))]
pub mod neuromorphic;
#[cfg(feature = "status-only")]
pub mod neuromorphic_status_only;
pub mod config;
pub mod error;
pub mod utils;

pub use cli::Cli;
pub use config::CliConfig;
pub use error::{CliError, Result};
#[cfg(not(feature = "status-only"))]
pub use neuromorphic::UnifiedNeuromorphicSystem;
#[cfg(feature = "status-only")]
pub use neuromorphic_status_only::UnifiedNeuromorphicSystem;

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Initialize the ARES neuromorphic CLI library
pub async fn init() -> Result<()> {
    // Initialize logging
    utils::init_logging();
    
    // In minimal status-only mode, skip heavy environment checks
    #[cfg(not(feature = "status-only"))]
    {
        validate_system_requirements().await?;
    }
    
    Ok(())
}

/// Validate that the system meets requirements for neuromorphic processing
#[cfg(not(feature = "status-only"))]
async fn validate_system_requirements() -> Result<()> {
    // Check Python availability (required for Brian2/Lava)
    match std::process::Command::new("python3").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            tracing::debug!("Python3 available: {}", version.trim());
        },
        _ => {
            return Err(CliError::config(
                "Python3 not found. Required for Brian2/Lava neuromorphic simulation."
            ));
        }
    }
    
    // Check memory requirements (minimum 4GB)
    let memory_gb = get_system_memory_gb();
    if memory_gb < 4.0 {
        tracing::warn!("Low system memory: {:.1}GB (recommended: 8GB+)", memory_gb);
    }
    
    Ok(())
}

fn get_system_memory_gb() -> f32 {
    use sysinfo::System;
    
    let mut sys = System::new_all();
    sys.refresh_memory();
    
    let total_memory_kb = sys.total_memory();
    total_memory_kb as f32 / (1024.0 * 1024.0)
}
