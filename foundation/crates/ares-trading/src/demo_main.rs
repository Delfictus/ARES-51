use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use clap::{Parser, Subcommand};
use std::sync::Arc;
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use ares_trading::{
    market_data::SimulatedMarketDataProvider,
    market_demo::{MarketDemo, DemoConfig},
};

#[derive(Parser)]
#[command(name = "market-demo")]
#[command(about = "ARES Market Analysis & Prediction Demo")]
#[command(long_about = "
ARES ChronoFabric Market Demo

This demo showcases the quantum temporal correlation system for market analysis and prediction.
It can analyze historical market data and make predictions using advanced quantum resonance 
techniques with animated visualizations.

Features:
- Minute-level historical data analysis
- Quantum temporal correlation prediction model
- Animated training and prediction visualization
- Performance metrics and accuracy analysis
- Real-time prediction vs actual comparison
")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a market prediction demo
    Run {
        /// Stock symbol to analyze
        #[arg(short, long, default_value = "AAPL")]
        symbol: String,

        /// Start date (YYYY-MM-DD format)
        #[arg(short = 's', long)]
        start_date: String,

        /// End date (YYYY-MM-DD format)
        #[arg(short = 'e', long)]
        end_date: String,

        /// Training ratio (0.0 - 1.0, default: 0.8)
        #[arg(short = 't', long, default_value = "0.8")]
        training_ratio: f64,

        /// Prediction horizon in minutes (default: 1)
        #[arg(short = 'h', long, default_value = "1")]
        prediction_horizon: i64,

        /// Enable visualization (default: true)
        #[arg(short = 'v', long, default_value = "true")]
        visualization: bool,

        /// Animation speed multiplier (default: 1.0)
        #[arg(short = 'a', long, default_value = "1.0")]
        animation_speed: f64,
    },
    
    /// Run a quick demo with default parameters
    Quick {
        /// Stock symbol to analyze
        #[arg(short, long, default_value = "AAPL")]
        symbol: String,
    },

    /// Show available symbols and their descriptions
    Symbols,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "market_demo=info,ares_trading=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    print_banner();

    match cli.command {
        Some(Commands::Run { 
            symbol, 
            start_date, 
            end_date, 
            training_ratio, 
            prediction_horizon,
            visualization,
            animation_speed,
        }) => {
            run_demo(
                symbol, 
                start_date, 
                end_date, 
                training_ratio, 
                prediction_horizon,
                visualization,
                animation_speed,
            ).await?;
        },
        
        Some(Commands::Quick { symbol }) => {
            run_quick_demo(symbol).await?;
        },
        
        Some(Commands::Symbols) => {
            show_available_symbols();
        },
        
        None => {
            // Default: run quick demo with AAPL
            run_quick_demo("AAPL".to_string()).await?;
        }
    }

    Ok(())
}

async fn run_demo(
    symbol: String,
    start_date: String,
    end_date: String,
    training_ratio: f64,
    prediction_horizon: i64,
    visualization: bool,
    animation_speed: f64,
) -> Result<()> {
    info!("🚀 Starting Market Analysis & Prediction Demo");
    info!("==============================================");

    // Parse dates
    let start = parse_date(&start_date)?;
    let end = parse_date(&end_date)?;

    // Validate parameters
    if training_ratio < 0.1 || training_ratio > 0.9 {
        return Err(anyhow::anyhow!("Training ratio must be between 0.1 and 0.9"));
    }

    if prediction_horizon < 1 || prediction_horizon > 60 {
        return Err(anyhow::anyhow!("Prediction horizon must be between 1 and 60 minutes"));
    }

    if end <= start {
        return Err(anyhow::anyhow!("End date must be after start date"));
    }

    let total_days = (end - start).num_days();
    if total_days > 30 {
        warn!("Demo period is {} days. This might take a while...", total_days);
    }

    // Create market data provider
    info!("🔌 Initializing quantum temporal correlation system");
    let provider = Arc::new(SimulatedMarketDataProvider::new());
    
    // Create and configure the demo
    let mut demo = MarketDemo::new(provider);
    let config = DemoConfig {
        symbol: symbol.clone(),
        start_date: start,
        end_date: end,
        training_ratio,
        prediction_horizon_minutes: prediction_horizon,
        visualization_enabled: visualization,
        animation_speed,
    };

    info!("📊 Demo Configuration:");
    info!("   Symbol: {}", config.symbol);
    info!("   Period: {} to {}", config.start_date, config.end_date);
    info!("   Training Ratio: {:.1}%", config.training_ratio * 100.0);
    info!("   Prediction Horizon: {} minutes", config.prediction_horizon_minutes);
    info!("   Visualization: {}", if config.visualization_enabled { "Enabled" } else { "Disabled" });
    info!("   Animation Speed: {}x", config.animation_speed);
    info!("");

    if visualization {
        info!("🎬 Visualization enabled - watch the quantum temporal correlations unfold!");
        info!("   Phase 1: Training the quantum model with historical data");
        info!("   Phase 2: Making predictions and comparing with actual results");
        info!("");
    }

    // Run the demo
    demo.run_demo(config).await?;

    info!("✨ Demo completed successfully!");
    Ok(())
}

async fn run_quick_demo(symbol: String) -> Result<()> {
    info!("🚀 Starting Quick Market Demo for {}", symbol);
    info!("=====================================");

    // Use last 7 days of data
    let end_date = Utc::now();
    let start_date = end_date - Duration::days(7);

    let provider = Arc::new(SimulatedMarketDataProvider::new());
    let mut demo = MarketDemo::new(provider);
    
    let config = DemoConfig {
        symbol: symbol.clone(),
        start_date,
        end_date,
        training_ratio: 0.8,
        prediction_horizon_minutes: 1,
        visualization_enabled: true,
        animation_speed: 2.0, // Faster for quick demo
    };

    info!("📊 Quick Demo - Last 7 days of {} data", symbol);
    info!("🎬 Visualization enabled at 2x speed");
    info!("");

    demo.run_demo(config).await?;

    info!("✨ Quick demo completed!");
    Ok(())
}

fn show_available_symbols() {
    println!("📈 Available Symbols for Analysis");
    println!("=================================");
    println!();
    
    println!("📊 Popular Stocks:");
    println!("   AAPL  - Apple Inc.");
    println!("   GOOGL - Alphabet Inc. (Google)");
    println!("   MSFT  - Microsoft Corporation");
    println!("   AMZN  - Amazon.com Inc.");
    println!("   TSLA  - Tesla Inc.");
    println!("   META  - Meta Platforms Inc. (Facebook)");
    println!("   NVDA  - NVIDIA Corporation");
    println!();
    
    println!("📈 Market Indices:");
    println!("   SPY   - SPDR S&P 500 ETF Trust");
    println!("   QQQ   - Invesco QQQ Trust (Nasdaq-100)");
    println!();
    
    println!("💰 Cryptocurrencies:");
    println!("   BTC-USD - Bitcoin");
    println!("   ETH-USD - Ethereum");
    println!();
    
    println!("Usage: market-demo run --symbol <SYMBOL>");
    println!("Example: market-demo run --symbol AAPL -s 2024-01-01 -e 2024-01-07");
}

fn parse_date(date_str: &str) -> Result<DateTime<Utc>> {
    use chrono::NaiveDate;
    
    let naive_date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map_err(|_| anyhow::anyhow!("Invalid date format. Use YYYY-MM-DD"))?;
    
    let naive_datetime = naive_date.and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("Failed to create datetime"))?;
    
    Ok(DateTime::from_naive_utc_and_offset(naive_datetime, Utc))
}

fn print_banner() {
    println!(r#"
╔══════════════════════════════════════════════════════════════╗
║                    ARES ChronoFabric                         ║
║              Market Analysis & Prediction Demo               ║
║                                                              ║
║   🌌 Quantum Temporal Correlation System                     ║
║   📊 Minute-Level Historical Analysis                        ║
║   🔮 Advanced Prediction Models                              ║
║   🎬 Real-Time Visualization                                 ║
║                                                              ║
║                  Author: Ididia Serfaty                      ║
╚══════════════════════════════════════════════════════════════╝
"#);
    println!();
}