//! Logging utilities for the neuromorphic CLI

use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize tracing for the application
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_timer(tracing_subscriber::fmt::time::uptime())
                .with_level(true)
                .with_ansi(true)
        )
        .with(filter)
        .init();
}

/// Set logging level based on verbosity flags
pub fn set_level_from_verbosity(verbosity: u8) {
    let level = match verbosity {
        0 => Level::WARN,
        1 => Level::INFO, 
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };
    
    // Create filter with the specified level
    let filter = EnvFilter::new(level.to_string().to_lowercase());
    
    // Set the global subscriber with the new filter
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_timer(tracing_subscriber::fmt::time::uptime())
                .with_level(true)
                .with_ansi(true)
        )
        .with(filter)
        .try_init()
        .ok(); // Ignore error if already initialized
}