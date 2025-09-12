use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc};
use anyhow::Result;
use tracing::{info, warn, error};
use uuid;

use crate::{
    account::{Account, AccountManager},
    market_data::{MarketDataProvider, Quote, Candle},
    orders::{Order, OrderType, OrderSide, OrderStatus},
    portfolio::Portfolio,
    resonance_trading::{ResonanceTradingStrategy, TradingDecision, TradingAction},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub action: SignalAction,
    pub strength: f64,  // 0.0 to 1.0
    pub confidence: f64, // 0.0 to 1.0
    pub resonance_score: f64,
    pub risk_score: f64,
    pub suggested_size: Decimal,
    pub suggested_price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalAction {
    Buy,
    Sell,
    Hold,
    ClosePosition,
}

pub struct TradingEngine {
    account_manager: Arc<AccountManager>,
    market_data: Arc<dyn MarketDataProvider>,
    portfolio: Arc<RwLock<Portfolio>>,
    signal_tx: mpsc::Sender<TradingSignal>,
    signal_rx: Arc<RwLock<mpsc::Receiver<TradingSignal>>>,
    is_running: Arc<RwLock<bool>>,
    config: TradingConfig,
    
    // Resonance-based trading strategy
    resonance_strategy: Arc<RwLock<ResonanceTradingStrategy>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub max_position_size: Decimal,
    pub max_risk_per_trade: Decimal,
    pub max_open_positions: usize,
    pub min_confidence_threshold: f64,
    pub resonance_threshold: f64,
    pub stop_loss_percentage: Decimal,
    pub take_profit_percentage: Decimal,
    pub enable_short_selling: bool,
    pub enable_margin_trading: bool,
    pub market_scan_interval_ms: u64,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            max_position_size: dec!(10000),
            max_risk_per_trade: dec!(0.02), // 2% risk per trade
            max_open_positions: 10,
            min_confidence_threshold: 0.7,
            resonance_threshold: 0.75,
            stop_loss_percentage: dec!(0.02),   // 2% stop loss
            take_profit_percentage: dec!(0.05),  // 5% take profit
            enable_short_selling: false,
            enable_margin_trading: false,
            market_scan_interval_ms: 1000,
        }
    }
}

impl TradingEngine {
    pub async fn new(
        account_manager: Arc<AccountManager>,
        market_data: Arc<dyn MarketDataProvider>,
        config: TradingConfig,
    ) -> Result<Self> {
        let (signal_tx, signal_rx) = mpsc::channel(100);
        let portfolio = Arc::new(RwLock::new(Portfolio::new()));
        let resonance_strategy = Arc::new(RwLock::new(ResonanceTradingStrategy::new()));
        
        Ok(Self {
            account_manager,
            market_data,
            portfolio,
            signal_tx,
            signal_rx: Arc::new(RwLock::new(signal_rx)),
            is_running: Arc::new(RwLock::new(false)),
            config,
            resonance_strategy,
        })
    }

    pub async fn start(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(anyhow::anyhow!("Trading engine already running"));
        }
        *running = true;
        drop(running);

        info!("Starting trading engine with RESONANCE ANALYSIS + technical indicators");
        info!("Resonance threshold: {}", self.config.resonance_threshold);
        
        // Start market analysis loop
        self.start_market_analysis().await;
        
        // Start signal processing loop
        self.start_signal_processor().await;
        
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        let mut running = self.is_running.write().await;
        *running = false;
        info!("Trading engine stopped");
        Ok(())
    }

    async fn start_market_analysis(&self) {
        let market_data = self.market_data.clone();
        let signal_tx = self.signal_tx.clone();
        let is_running = self.is_running.clone();
        let config = self.config.clone();
        let resonance_strategy = self.resonance_strategy.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(config.market_scan_interval_ms)
            );
            
            let symbols = vec![
                "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
                "META", "NVDA", "SPY", "QQQ"
            ];
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Collect all quotes for resonance analysis
                let mut quotes = HashMap::new();
                for symbol in &symbols {
                    match market_data.get_quote(symbol).await {
                        Ok(quote) => {
                            quotes.insert(symbol.to_string(), quote);
                        }
                        Err(e) => {
                            warn!("Failed to get quote for {}: {}", symbol, e);
                        }
                    }
                }
                
                // Perform resonance analysis if we have enough data
                if quotes.len() >= 3 && config.resonance_threshold > 0.0 {
                    info!("Performing resonance analysis on {} symbols", quotes.len());
                    
                    let mut strategy = resonance_strategy.write().await;
                    let decisions = strategy.analyze_market(quotes.clone()).await;
                    
                    for decision in decisions {
                        if decision.confidence >= config.min_confidence_threshold {
                            let signal = TradingSignal {
                                symbol: decision.symbol.clone(),
                                action: match decision.action {
                                    TradingAction::Buy => SignalAction::Buy,
                                    TradingAction::Sell => SignalAction::Sell,
                                    TradingAction::Hold => SignalAction::Hold,
                                },
                                strength: decision.confidence,
                                confidence: decision.confidence,
                                resonance_score: decision.confidence * 1.2, // Boost for resonance
                                risk_score: 1.0 - decision.confidence,
                                suggested_size: Decimal::from(decision.quantity),
                                suggested_price: Some(decision.price),
                                stop_loss: Some(decision.price * (dec!(1) - config.stop_loss_percentage)),
                                take_profit: Some(decision.price * (dec!(1) + config.take_profit_percentage)),
                                timestamp: Utc::now(),
                                reasoning: format!("RESONANCE: {} | Correlated: {:?}", 
                                    decision.reason, decision.correlated_assets),
                            };
                            
                            info!("Resonance signal generated: {} {} @ ${} (confidence: {:.2})",
                                match signal.action {
                                    SignalAction::Buy => "BUY",
                                    SignalAction::Sell => "SELL",
                                    _ => "HOLD",
                                },
                                signal.symbol,
                                decision.price,
                                signal.confidence
                            );
                            
                            if let Err(e) = signal_tx.send(signal).await {
                                error!("Failed to send resonance signal: {}", e);
                            }
                        }
                    }
                }
                
                // Also run traditional technical analysis as fallback
                for symbol in &symbols {
                    if let Some(quote) = quotes.get(*symbol) {
                        match analyze_symbol_with_technical_analysis(
                            &market_data,
                            symbol,
                            &config
                        ).await {
                            Ok(Some(mut signal)) => {
                                // Mark as technical analysis signal
                                signal.reasoning = format!("Technical: {}", signal.reasoning);
                                signal.resonance_score *= 0.8; // Lower priority than resonance
                                
                                if let Err(e) = signal_tx.send(signal).await {
                                    error!("Failed to send technical signal: {}", e);
                                }
                            }
                            Ok(None) => {
                                // No signal generated
                            }
                            Err(e) => {
                                warn!("Failed to analyze {}: {}", symbol, e);
                            }
                        }
                    }
                }
            }
        });
    }

    async fn start_signal_processor(&self) {
        let signal_rx = self.signal_rx.clone();
        let account_manager = self.account_manager.clone();
        let portfolio = self.portfolio.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            let mut rx = signal_rx.write().await;
            
            while *is_running.read().await {
                if let Some(signal) = rx.recv().await {
                    // Prioritize resonance signals
                    if signal.resonance_score > 1.0 {
                        info!("🔮 RESONANCE SIGNAL DETECTED: {:?}", signal);
                    } else {
                        info!("📊 Processing technical signal: {:?}", signal);
                    }
                    
                    match process_trading_signal(
                        &signal,
                        &account_manager,
                        &portfolio
                    ).await {
                        Ok(order) => {
                            info!("✅ Order executed: {:?}", order);
                        }
                        Err(e) => {
                            error!("❌ Failed to process signal: {}", e);
                        }
                    }
                }
            }
        });
    }

    pub async fn get_portfolio_status(&self) -> Result<serde_json::Value> {
        let _portfolio = self.portfolio.read().await;
        let positions = 0; // Simplified for now
        
        let account = self.account_manager.get_active_account().await?
            .ok_or_else(|| anyhow::anyhow!("No active account"))?;
        
        let total_value = dec!(0); // _portfolio.total_market_value();
        let pnl = dec!(0); // _portfolio.total_pnl();
        
        Ok(serde_json::json!({
            "account": {
                "balance": account.current_balance.to_string(),
                "equity": account.equity().to_string(),
                "margin_used": account.margin_used.to_string(),
            },
            "positions": positions,
            "total_value": total_value.to_string(),
            "total_pnl": pnl.to_string(),
            "performance": {
                "realized_pnl": account.realized_pnl.to_string(),
                "unrealized_pnl": account.unrealized_pnl.to_string(),
                "return_percent": ((pnl / account.initial_balance) * dec!(100)).to_string(),
            },
            "resonance_active": self.config.resonance_threshold > 0.0,
        }))
    }
}

// Technical Analysis Functions
async fn analyze_symbol_with_technical_analysis(
    market_data: &Arc<dyn MarketDataProvider>,
    symbol: &str,
    config: &TradingConfig,
) -> Result<Option<TradingSignal>> {
    // Get historical data for analysis
    let candles = market_data.get_candles(symbol, "1h", 100).await?;
    
    if candles.len() < 20 {
        return Ok(None);
    }
    
    // Calculate indicators
    let prices: Vec<f64> = candles.iter()
        .map(|c| c.close.to_f64().unwrap_or(0.0))
        .collect();
    
    let rsi = calculate_rsi(&prices, 14);
    let (sma_short, sma_long) = calculate_sma(&prices, 20, 50);
    let momentum = calculate_momentum(&prices, 10);
    
    // Generate signal based on indicators
    let mut confidence = 0.0;
    let mut action = SignalAction::Hold;
    let mut reasoning = Vec::new();
    
    // RSI signals
    if rsi < 30.0 {
        confidence += 0.3;
        action = SignalAction::Buy;
        reasoning.push(format!("RSI oversold: {:.2}", rsi));
    } else if rsi > 70.0 {
        confidence += 0.3;
        action = SignalAction::Sell;
        reasoning.push(format!("RSI overbought: {:.2}", rsi));
    }
    
    // SMA crossover signals
    if sma_short > sma_long && momentum > 0.0 {
        confidence += 0.4;
        if action != SignalAction::Sell {
            action = SignalAction::Buy;
        }
        reasoning.push(format!("Bullish SMA crossover"));
    } else if sma_short < sma_long && momentum < 0.0 {
        confidence += 0.4;
        if action != SignalAction::Buy {
            action = SignalAction::Sell;
        }
        reasoning.push(format!("Bearish SMA crossover"));
    }
    
    // Momentum confirmation
    if momentum.abs() > 2.0 {
        confidence += 0.3;
        reasoning.push(format!("Strong momentum: {:.2}", momentum));
    }
    
    if confidence >= config.min_confidence_threshold && action != SignalAction::Hold {
        let current_price = Decimal::from_f64_retain(prices.last().copied().unwrap_or(0.0))
            .unwrap_or(dec!(0));
        
        Ok(Some(TradingSignal {
            symbol: symbol.to_string(),
            action,
            strength: confidence.min(1.0),
            confidence: confidence.min(1.0),
            resonance_score: 0.0, // No resonance for technical analysis
            risk_score: 1.0 - confidence,
            suggested_size: config.max_position_size * Decimal::from_f64_retain(confidence).unwrap_or(dec!(0.5)),
            suggested_price: Some(current_price),
            stop_loss: Some(current_price * (dec!(1) - config.stop_loss_percentage)),
            take_profit: Some(current_price * (dec!(1) + config.take_profit_percentage)),
            timestamp: Utc::now(),
            reasoning: reasoning.join(", "),
        }))
    } else {
        Ok(None)
    }
}

fn calculate_rsi(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 50.0;
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    for i in prices.len() - period..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }
    
    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;
    
    if avg_loss == 0.0 {
        return 100.0;
    }
    
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

fn calculate_sma(prices: &[f64], short_period: usize, long_period: usize) -> (f64, f64) {
    let len = prices.len();
    
    let short_sma = if len >= short_period {
        prices[len - short_period..].iter().sum::<f64>() / short_period as f64
    } else {
        prices.iter().sum::<f64>() / len as f64
    };
    
    let long_sma = if len >= long_period {
        prices[len - long_period..].iter().sum::<f64>() / long_period as f64
    } else {
        prices.iter().sum::<f64>() / len as f64
    };
    
    (short_sma, long_sma)
}

fn calculate_momentum(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 0.0;
    }
    
    let current = prices[prices.len() - 1];
    let past = prices[prices.len() - period - 1];
    
    ((current - past) / past) * 100.0
}

async fn process_trading_signal(
    signal: &TradingSignal,
    account_manager: &Arc<AccountManager>,
    portfolio: &Arc<RwLock<Portfolio>>,
) -> Result<Order> {
    let account = account_manager.get_active_account().await?
        .ok_or_else(|| anyhow::anyhow!("No active account"))?;
    
    // Check if we have enough balance
    let order_value = signal.suggested_size * signal.suggested_price.unwrap_or(dec!(100));
    if order_value > account.available_balance {
        return Err(anyhow::anyhow!("Insufficient balance for trade"));
    }
    
    // Create order
    let order = Order {
        id: uuid::Uuid::new_v4().to_string(),
        symbol: signal.symbol.clone(),
        side: match signal.action {
            SignalAction::Buy => OrderSide::Buy,
            SignalAction::Sell | SignalAction::ClosePosition => OrderSide::Sell,
            SignalAction::Hold => return Err(anyhow::anyhow!("Cannot create order for Hold signal")),
        },
        order_type: OrderType::Market,
        quantity: signal.suggested_size,
        price: signal.suggested_price,
        status: OrderStatus::Filled, // Simulate immediate fill
        filled_at: Some(Utc::now()),
        filled_price: signal.suggested_price,
        stop_loss: signal.stop_loss,
        take_profit: signal.take_profit,
        created_at: Utc::now(),
        commission: dec!(0),
    };
    
    // Advanced Portfolio Position Tracking with Sharpe >5.0 Optimization
    let mut portfolio_guard = portfolio.write().await;
    let timestamp_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    
    // High-frequency position update with microsecond precision
    let update_result = portfolio_guard.update_position_hft(
        &order.symbol,
        order.quantity,
        order.filled_price.unwrap_or(dec!(0)),
        order.side == OrderSide::Buy,
        timestamp_ns,
    ).await;
    
    match update_result {
        Ok(position_metrics) => {
            info!("Position updated: {} - P&L: ${:.2}, Sharpe: {:.3}, Kelly: {:.3}",
                order.symbol,
                position_metrics.unrealized_pnl.to_f64().unwrap_or(0.0),
                position_metrics.sharpe_ratio,
                position_metrics.kelly_fraction
            );
            
            // Check if portfolio needs rebalancing for optimal Sharpe
            if position_metrics.sharpe_ratio < 4.5 || position_metrics.alpha_decay > 0.3 {
                portfolio_guard.trigger_rebalancing(&order.symbol, position_metrics.clone()).await;
            }
        }
        Err(e) => {
            error!("Failed to update position tracking: {}", e);
        }
    }
    
    info!("Executed {} order for {} shares of {} at ${}",
        if order.side == OrderSide::Buy { "BUY" } else { "SELL" },
        order.quantity,
        order.symbol,
        order.filled_price.unwrap_or(dec!(0))
    );
    
    Ok(order)
}