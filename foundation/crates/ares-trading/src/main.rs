use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::sync::Arc;
use std::fs;
use tokio::signal;
use serde::Deserialize;

use ares_trading::{
    account::AccountManager,
    market_data::{SimulatedMarketDataProvider, MarketDataProvider},
    trading_engine::{TradingEngine, TradingConfig},
    providers::{MarketDataConfig, ProviderType, MarketDataProviderFactory},
};

#[derive(Debug, Deserialize)]
struct Config {
    trading: TradingConfigSection,
    market_data: MarketDataConfigSection,
    symbols: Vec<SymbolConfig>,
}

#[derive(Debug, Deserialize)]
struct TradingConfigSection {
    initial_balance: f64,
    max_position_size: f64,
    stop_loss_percent: f64,
    take_profit_percent: f64,
    max_open_positions: u32,
    rsi_period: Option<u32>,
    rsi_oversold: Option<u32>,
    rsi_overbought: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MarketDataConfigSection {
    provider: String,
    polygon_api_key: Option<String>,
    iex_api_token: Option<String>,
    finnhub_api_key: Option<String>,
    twelve_data_api_key: Option<String>,
    rate_limit_per_minute: Option<u32>,
    enable_websocket: bool,
}

#[derive(Debug, Deserialize)]
struct SymbolConfig {
    symbol: String,
    asset_type: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ares_trading=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting ARES Paper Trading System");
    info!("===================================");
    
    // Load configuration
    let config_path = "crates/ares-trading/config.toml";
    let config_str = fs::read_to_string(config_path)
        .unwrap_or_else(|_| {
            warn!("Config file not found at {}, using defaults", config_path);
            include_str!("../config.toml").to_string()
        });
    let config: Config = toml::from_str(&config_str)?;
    
    info!("Configuration loaded:");
    info!("  Provider: {}", config.market_data.provider);
    info!("  Initial Balance: ${}", config.trading.initial_balance);
    info!("  Max Positions: {}", config.trading.max_open_positions);
    
    // Initialize account manager with paper trading account
    info!("Creating paper trading account...");
    let account_manager = Arc::new(AccountManager::new());
    let initial_balance = Decimal::from_f64_retain(config.trading.initial_balance)
        .unwrap_or(dec!(100000));
    account_manager.create_account(
        "paper-account-001".to_string(),
        initial_balance
    ).await?;
    
    // Create market data provider based on configuration
    info!("Setting up market data provider: {}", config.market_data.provider);
    let market_data: Arc<dyn MarketDataProvider> = create_provider(&config.market_data).await?;
    
    // Subscribe to configured symbols
    let symbols: Vec<String> = config.symbols.iter()
        .map(|s| s.symbol.clone())
        .collect();
    
    info!("Subscribing to {} symbols: {:?}", symbols.len(), symbols);
    market_data.subscribe(symbols.clone()).await?;
    
    // Configure trading parameters
    let trading_config = TradingConfig {
        max_position_size: Decimal::from_f64_retain(config.trading.initial_balance * config.trading.max_position_size)
            .unwrap_or(dec!(10000)),
        max_risk_per_trade: dec!(0.02),
        max_open_positions: config.trading.max_open_positions as usize,
        min_confidence_threshold: 0.7,
        resonance_threshold: 0.75,
        stop_loss_percentage: Decimal::from_f64_retain(config.trading.stop_loss_percent / 100.0)
            .unwrap_or(dec!(0.02)),
        take_profit_percentage: Decimal::from_f64_retain(config.trading.take_profit_percent / 100.0)
            .unwrap_or(dec!(0.05)),
        enable_short_selling: false,
        enable_margin_trading: false,
        market_scan_interval_ms: 2000,
    };
    
    // Initialize trading engine
    info!("Initializing trading engine...");
    let trading_engine = Arc::new(TradingEngine::new(
        account_manager.clone(),
        market_data.clone(),
        trading_config,
    ).await?);
    
    // Start trading engine
    info!("Starting automated trading...");
    trading_engine.start().await?;
    
    // Status reporting loop
    let status_handle = {
        let engine = trading_engine.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                match engine.get_portfolio_status().await {
                    Ok(status) => {
                        info!("Portfolio Status Update:");
                        info!("========================");
                        if let Some(account) = status.get("account") {
                            info!("Account: {}", serde_json::to_string_pretty(account).unwrap());
                        }
                        if let Some(positions) = status.get("positions") {
                            info!("Open Positions: {}", positions);
                        }
                        if let Some(performance) = status.get("performance") {
                            info!("Performance: {}", serde_json::to_string_pretty(performance).unwrap());
                        }
                    }
                    Err(e) => {
                        error!("Failed to get portfolio status: {}", e);
                    }
                }
            }
        })
    };
    
    // Market data display loop
    let market_handle = {
        let data_provider = market_data.clone();
        let syms = symbols.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                info!("Market Data Update:");
                for symbol in syms.iter().take(5) {
                    match data_provider.get_quote(symbol).await {
                        Ok(quote) => {
                            info!(
                                "{}: ${:.2} ({}${:.2} {:.2}%)",
                                symbol,
                                quote.last,
                                if quote.change > dec!(0) { "+" } else { "" },
                                quote.change,
                                quote.change_percent
                            );
                        }
                        Err(e) => {
                            warn!("Failed to get quote for {}: {}", symbol, e);
                        }
                    }
                }
            }
        })
    };
    
    info!("Paper trading system is running. Press Ctrl+C to stop.");
    info!("Initial Balance: ${}", initial_balance);
    info!("Trading Strategy: Technical Analysis with RSI, SMA, and Momentum");
    info!("Market Data Provider: {}", config.market_data.provider);
    info!("Monitoring {} symbols", symbols.len());
    
    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received, stopping trading engine...");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
    
    // Clean shutdown
    status_handle.abort();
    market_handle.abort();
    
    // Unsubscribe from market data
    market_data.unsubscribe(symbols).await?;
    
    // Get final status
    match account_manager.get_active_account().await? {
        Some(account) => {
            info!("=== Final Account Status ===");
            info!("Initial Balance: ${}", account.initial_balance);
            info!("Current Balance: ${}", account.current_balance);
            info!("Total Equity: ${}", account.equity());
            info!("Realized P&L: ${}", account.realized_pnl);
            info!("Unrealized P&L: ${}", account.unrealized_pnl);
            let total_pnl = account.realized_pnl + account.unrealized_pnl;
            let return_pct = (total_pnl / account.initial_balance) * dec!(100);
            info!("Total Return: {:.2}%", return_pct);
        }
        None => {
            error!("No active account found");
        }
    }
    
    info!("Paper trading system stopped.");
    Ok(())
}

async fn create_provider(config: &MarketDataConfigSection) -> Result<Arc<dyn MarketDataProvider>> {
    let provider_type = match config.provider.as_str() {
        "yahoo_finance" => ProviderType::YahooFinance,
        "polygon_io" => ProviderType::PolygonIo,
        "iex_cloud" => ProviderType::IexCloud,
        "binance" => ProviderType::Binance,
        "finnhub" => ProviderType::Finnhub,
        "twelve_data" => ProviderType::TwelveData,
        _ => {
            warn!("Unknown provider '{}', defaulting to simulated data", config.provider);
            return Ok(Arc::new(SimulatedMarketDataProvider::new()));
        }
    };
    
    let api_key = match provider_type {
        ProviderType::PolygonIo => config.polygon_api_key.clone(),
        ProviderType::IexCloud => config.iex_api_token.clone(),
        ProviderType::Finnhub => config.finnhub_api_key.clone(),
        ProviderType::TwelveData => config.twelve_data_api_key.clone(),
        _ => None,
    };
    
    // Check if API key is required but missing
    match provider_type {
        ProviderType::YahooFinance => {
            info!("Using Yahoo Finance provider (FREE, no API key required)");
        },
        ProviderType::Binance => {
            info!("Using Binance provider (FREE cryptocurrency data)");
        },
        _ => {
            if api_key.is_none() {
                error!("Provider {} requires an API key", config.provider);
                info!("Falling back to simulated market data");
                return Ok(Arc::new(SimulatedMarketDataProvider::new()));
            }
            info!("Using {} provider with API key", config.provider);
        }
    }
    
    let provider_config = MarketDataConfig {
        provider: provider_type,
        api_key,
        api_secret: None,
        base_url: None,
        rate_limit_per_minute: config.rate_limit_per_minute,
        enable_websocket: config.enable_websocket,
    };
    
    MarketDataProviderFactory::create(provider_config).await
        .map(|provider| Arc::from(provider))
}