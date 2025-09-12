//! Signal handling for graceful shutdown

use anyhow::Result;
use tracing::info;

/// Set up graceful shutdown signal handlers
pub async fn setup_shutdown_handler() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        
        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sigint = signal(SignalKind::interrupt())?;
        let mut sigquit = signal(SignalKind::quit())?;
        
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM - shutting down neuromorphic systems");
            },
            _ = sigint.recv() => {
                info!("Received SIGINT (Ctrl+C) - shutting down neuromorphic systems");
            },
            _ = sigquit.recv() => {
                info!("Received SIGQUIT - shutting down neuromorphic systems");
            },
        }
    }
    
    #[cfg(windows)]
    {
        use tokio::signal;
        
        signal::ctrl_c().await?;
        info!("Received Ctrl+C - shutting down neuromorphic systems");
    }
    
    Ok(())
}