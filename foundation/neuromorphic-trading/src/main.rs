//! Main entry point for neuromorphic trading system

use neuromorphic_trading::integration::{IntegratedTradingSystem, SystemConfig};
use neuromorphic_trading::exchanges::{Symbol, Exchange};
use anyhow::Result;
use std::time::Duration;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    
    println!("=== NEUROMORPHIC TRADING SYSTEM ===");
    println!("Initializing hybrid neuromorphic-deterministic trading engine...\n");
    
    // Configure system
    let config = SystemConfig {
        neuron_count: 10000,
        reservoir_size: 1000,
        initial_capital: 100000.0,
        exchanges: vec![Exchange::Binance, Exchange::Coinbase],
        symbols: vec![
            Symbol::new("BTC-USD"),
            Symbol::new("ETH-USD"),
            Symbol::new("SOL-USD"),
        ],
        enable_paper_trading: true,
        enable_live_trading: false,
        update_interval: Duration::from_millis(100),
    };
    
    // Create integrated system
    let mut system = IntegratedTradingSystem::new(config)?;
    
    // Start system
    println!("Starting system...");
    system.start().await?;
    
    // Run until interrupted
    println!("\nSystem is running. Press Ctrl+C to stop.\n");
    
    // Set up graceful shutdown
    let shutdown = signal::ctrl_c();
    
    tokio::select! {
        _ = shutdown => {
            println!("\nShutdown signal received...");
        }
        _ = tokio::time::sleep(Duration::from_secs(3600)) => {
            println!("\nMax runtime reached...");
        }
    }
    
    // Stop system
    system.stop().await?;
    
    // Print final statistics
    let stats = system.get_statistics();
    println!("\n=== FINAL STATISTICS ===");
    println!("Final Capital: ${:.2}", stats.current_capital);
    println!("Total P&L: ${:.2} ({:.2}%)", stats.total_pnl, stats.total_return_pct);
    println!("Total Trades: {}", stats.total_trades);
    println!("Win Rate: {:.1}%", stats.win_rate);
    println!("Sharpe Ratio: {:.3}", stats.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", stats.max_drawdown * 100.0);
    println!("Signals Processed: {}", stats.signals_processed);
    println!("Signals Executed: {}", stats.signals_executed);
    println!("========================\n");
    
    println!("System shutdown complete.");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuromorphic_trading::integration::BacktestResults;
    
    #[tokio::test]
    async fn test_backtest() {
        let config = SystemConfig {
            neuron_count: 1000,
            reservoir_size: 100,
            initial_capital: 10000.0,
            exchanges: vec![Exchange::Binance],
            symbols: vec![Symbol::new("BTC-USD")],
            enable_paper_trading: true,
            enable_live_trading: false,
            update_interval: Duration::from_millis(10),
        };
        
        let mut system = IntegratedTradingSystem::new(config).unwrap();
        
        // Run short backtest
        let results = system.run_backtest(Duration::from_secs(1)).await.unwrap();
        
        // Print results
        results.print_summary();
        
        assert!(results.statistics.current_capital > 0.0);
    }
}