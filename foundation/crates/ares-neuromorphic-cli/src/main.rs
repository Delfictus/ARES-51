use anyhow::Result;
use clap::Parser;
use std::process;
use tracing::{debug, error, info};

mod cli;
mod commands;
#[cfg(not(feature = "status-only"))]
mod neuromorphic;
#[cfg(feature = "status-only")]
mod neuromorphic_status_only;
mod config;
mod error;
mod utils;

use cli::Cli;
use utils::{logging, signals};

#[tokio::main]
async fn main() {
    // Set up human-readable panic messages
    human_panic::setup_panic!();
    
    // Initialize logging early
    logging::init();
    
    info!("Starting ARES Neuromorphic CLI Interface");
    
    // Set up graceful shutdown handling
    let shutdown_handler = signals::setup_shutdown_handler();
    
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Initialize logging level from verbosity
    logging::set_level_from_verbosity(cli.verbose);
    
    debug!("CLI arguments parsed: {:?}", cli);
    
    // Execute main application logic with graceful shutdown
    let result = tokio::select! {
        result = run_app(cli) => result,
        _ = shutdown_handler => {
            info!("Received shutdown signal, cleaning up neuromorphic systems...");
            Ok(())
        }
    };
    
    if let Err(err) = result {
        error!("Application error: {:?}", err);
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}

async fn run_app(cli: Cli) -> Result<()> {
    debug!("Initializing ARES neuromorphic systems");
    
    // Initialize neuromorphic backend
    #[cfg(not(feature = "status-only"))]
    let neuromorphic_system = neuromorphic::UnifiedNeuromorphicSystem::initialize(
        cli.config.as_deref()
    ).await?;
    #[cfg(feature = "status-only")]
    let neuromorphic_system = neuromorphic_status_only::UnifiedNeuromorphicSystem::initialize(
        cli.config.as_deref()
    ).await?;
    
    info!("Neuromorphic backend initialized: {}", neuromorphic_system.backend_info());
    
    // Process command through neuromorphic interface
    match cli.command {
        #[cfg(not(feature = "status-only"))]
        commands::Commands::Interactive => {
            commands::interactive::run_interactive_mode(neuromorphic_system).await
        },
        #[cfg(not(feature = "status-only"))]
        commands::Commands::Enhanced => {
            // Initialize enhanced system for always-on NLP
            let enhanced_system = neuromorphic::EnhancedUnifiedNeuromorphicSystem::initialize(
                cli.config.as_deref()
            ).await?;
            commands::enhanced_interactive::run_enhanced_interactive_mode(enhanced_system).await
        },
        commands::Commands::Status(args) => {
            commands::status::execute(args, neuromorphic_system).await
        },
        #[cfg(not(feature = "status-only"))]
        commands::Commands::Learn(args) => {
            commands::learn::execute(args, neuromorphic_system).await
        },
        #[cfg(not(feature = "status-only"))]
        commands::Commands::Query { input } => {
            commands::query::execute_natural_language_query(input, neuromorphic_system).await
        },
    }
}
