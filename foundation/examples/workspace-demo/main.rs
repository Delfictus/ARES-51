//! ARES ChronoFabric - Main Entry Point
//!
//! High-performance temporal computing system with Phase Coherence Bus,
//! C-LOGIC modules, and MLIR-based hardware acceleration.

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

mod config;
mod runtime;

use config::ChronoFabricConfig;
use runtime::ChronoFabricRuntime;

#[derive(Parser)]
#[command(name = "chronofabric")]
#[command(about = "ARES ChronoFabric - Temporal Computing System")]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable development mode
    #[arg(long)]
    dev: bool,

    /// Bind address
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: String,

    /// Number of worker threads
    #[arg(long)]
    threads: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    init_tracing(&args.log_level)?;

    info!("Starting ARES ChronoFabric v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration with fallback to default
    let config = if std::path::Path::new(&args.config).exists() {
        ChronoFabricConfig::load(&args.config).await?
    } else {
        info!("Configuration file not found, using defaults");
        ChronoFabricConfig::default()
    };

    // Override with CLI args if provided
    let mut config = config;
    if let Some(threads) = args.threads {
        config.system.worker_threads = threads;
    }

    info!("Loaded configuration");

    // Create and start runtime
    let runtime = Arc::new(ChronoFabricRuntime::new(config).await?);
    info!("ChronoFabric runtime initialized");

    // Start the runtime
    runtime.start().await?;
    info!("ChronoFabric runtime started on {}", args.bind);

    // Wait for shutdown signal or runtime completion
    let shutdown = async {
        let _ = signal::ctrl_c().await;
        info!("Received shutdown signal");
    };

    tokio::select! {
        _ = shutdown => {
            info!("Shutting down ChronoFabric...");
            runtime.shutdown().await?;
            info!("ChronoFabric shutdown complete");
        }
        result = runtime.run() => {
            if let Err(e) = result {
                error!("Runtime error: {}", e);
                runtime.shutdown().await?;
                return Err(e);
            }
        }
    }

    Ok(())
}

fn init_tracing(level: &str) -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let level = level.parse().unwrap_or(tracing::Level::INFO);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("chronofabric={}", level).into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    Ok(())
}
