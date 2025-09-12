//! Market state tracking and regime detection
//! 
//! Maintains real-time market conditions including:
//! - Volatility (realized and implied)
//! - Spread dynamics
//! - Volume patterns
//! - Momentum indicators
//! - Market regime classification

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::event_bus::{TradeData, QuoteData, OrderBookData};

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,      // Strong directional movement
    RangeBoound,   // Oscillating within range
    Volatile,      // High volatility, unclear direction
    Quiet,         // Low volume, tight spreads
    Transitioning, // Regime change in progress
}

/// Rolling window for calculations
struct RollingWindow<T> {
    data: VecDeque<T>,
    max_size: usize,
}

impl<T: Clone> RollingWindow<T> {
    fn new(max_size: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_size),
            max_size,
        }
    }
    
    fn push(&mut self, value: T) {
        if self.data.len() >= self.max_size {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }
    
    fn values(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn is_full(&self) -> bool {
        self.data.len() >= self.max_size
    }
}

/// Price level for volatility calculation
#[derive(Clone, Debug)]
struct PricePoint {
    price: f64,
    timestamp_ns: u64,
    volume: u64,
}

/// Market state with comprehensive tracking
#[derive(Clone)]
pub struct MarketState {
    // Core metrics
    pub volatility: f64,           // Realized volatility (annualized)
    pub spread: f64,               // Current bid-ask spread
    pub volume: u64,               // Recent volume (decaying)
    pub momentum: f64,             // Price momentum (-1 to 1)
    pub liquidity_score: f64,      // Market depth score (0 to 1)
    
    // Regime detection
    pub regime: MarketRegime,
    pub regime_confidence: f64,
    
    // Advanced metrics
    pub volume_imbalance: f64,     // Buy vs sell pressure
    pub tick_rate: f64,            // Ticks per second
    pub spread_volatility: f64,    // Spread stability
    pub order_flow_toxicity: f64,  // Adverse selection measure
    
    // Internal tracking
    price_window: Arc<RwLock<RollingWindow<PricePoint>>>,
    spread_window: Arc<RwLock<RollingWindow<f64>>>,
    volume_accumulator: Arc<RwLock<VolumeAccumulator>>,
    momentum_calculator: Arc<RwLock<MomentumCalculator>>,
    last_update_ns: u64,
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            volatility: 0.16,  // 16% annualized (typical equity vol)
            spread: 0.01,      // 1 cent default
            volume: 0,
            momentum: 0.0,
            liquidity_score: 0.5,
            regime: MarketRegime::Quiet,
            regime_confidence: 0.5,
            volume_imbalance: 0.0,
            tick_rate: 0.0,
            spread_volatility: 0.0,
            order_flow_toxicity: 0.0,
            price_window: Arc::new(RwLock::new(RollingWindow::new(100))),
            spread_window: Arc::new(RwLock::new(RollingWindow::new(50))),
            volume_accumulator: Arc::new(RwLock::new(VolumeAccumulator::new())),
            momentum_calculator: Arc::new(RwLock::new(MomentumCalculator::new())),
            last_update_ns: 0,
        }
    }
}

impl MarketState {
    /// Create new market state with custom parameters
    pub fn new(volatility_window: usize, spread_window: usize) -> Self {
        Self {
            price_window: Arc::new(RwLock::new(RollingWindow::new(volatility_window))),
            spread_window: Arc::new(RwLock::new(RollingWindow::new(spread_window))),
            ..Default::default()
        }
    }
    
    /// Update state with trade data
    pub fn update_with_trade(&mut self, trade: &TradeData) {
        // Add to price window
        self.price_window.write().push(PricePoint {
            price: trade.price,
            timestamp_ns: trade.timestamp_ns,
            volume: trade.quantity,
        });
        
        // Update volume with decay
        self.volume_accumulator.write().add_volume(trade.quantity, trade.timestamp_ns);
        self.volume = self.volume_accumulator.read().get_current_volume();
        
        // Update momentum
        self.momentum_calculator.write().update(trade.price, trade.timestamp_ns);
        self.momentum = self.momentum_calculator.read().get_momentum();
        
        // Update volume imbalance
        match trade.aggressor_side {
            crate::event_bus::Side::Buy => {
                self.volume_imbalance = (self.volume_imbalance * 0.95 + 0.05).min(1.0);
            }
            crate::event_bus::Side::Sell => {
                self.volume_imbalance = (self.volume_imbalance * 0.95 - 0.05).max(-1.0);
            }
        }
        
        // Calculate volatility if we have enough data
        if self.price_window.read().is_full() {
            self.volatility = self.calculate_volatility();
        }
        
        // Update tick rate
        if self.last_update_ns > 0 {
            let time_diff_sec = (trade.timestamp_ns - self.last_update_ns) as f64 / 1e9;
            if time_diff_sec > 0.0 {
                self.tick_rate = self.tick_rate * 0.9 + 0.1 / time_diff_sec;
            }
        }
        
        self.last_update_ns = trade.timestamp_ns;
        
        // Detect regime
        self.detect_regime();
    }
    
    /// Update state with quote data
    pub fn update_with_quote(&mut self, quote: &QuoteData) {
        let spread = quote.ask_price - quote.bid_price;
        
        // Update spread tracking
        self.spread_window.write().push(spread);
        self.spread = spread;
        
        // Calculate spread volatility
        if self.spread_window.read().is_full() {
            let spreads = self.spread_window.read().values();
            let mean = spreads.iter().sum::<f64>() / spreads.len() as f64;
            let variance = spreads.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / spreads.len() as f64;
            self.spread_volatility = variance.sqrt();
        }
        
        // Update liquidity score based on spread
        let normalized_spread = (spread / quote.bid_price).min(0.01);
        self.liquidity_score = 1.0 - normalized_spread * 100.0;
        
        // Estimate toxicity (wide spreads = higher toxicity)
        self.order_flow_toxicity = (spread / quote.bid_price * 10000.0).min(100.0) / 100.0;
    }
    
    /// Update state with order book data
    pub fn update_with_orderbook(&mut self, book: &OrderBookData) {
        // Calculate weighted mid price
        let best_bid = book.bids[0].0;
        let best_ask = book.asks[0].0;
        let bid_size = book.bids[0].1 as f64;
        let ask_size = book.asks[0].1 as f64;
        
        let weighted_mid = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size);
        
        // Update price for volatility calculation
        self.price_window.write().push(PricePoint {
            price: weighted_mid,
            timestamp_ns: book.timestamp_ns,
            volume: (bid_size + ask_size) as u64,
        });
        
        // Calculate order book imbalance
        let total_bid_volume: u64 = book.bids.iter().map(|&(_, size)| size).sum();
        let total_ask_volume: u64 = book.asks.iter().map(|&(_, size)| size).sum();
        
        if total_bid_volume + total_ask_volume > 0 {
            self.volume_imbalance = (total_bid_volume as f64 - total_ask_volume as f64) / 
                                   (total_bid_volume + total_ask_volume) as f64;
        }
        
        // Update liquidity score based on book depth
        let depth_score = ((total_bid_volume + total_ask_volume) as f64 / 100000.0).min(1.0);
        self.liquidity_score = self.liquidity_score * 0.7 + depth_score * 0.3;
        
        // Detect regime
        self.detect_regime();
    }
    
    /// Calculate realized volatility
    fn calculate_volatility(&self) -> f64 {
        let prices = self.price_window.read();
        let price_points = prices.values();
        
        if price_points.len() < 2 {
            return 0.16;  // Default volatility
        }
        
        // Calculate returns
        let mut returns = Vec::new();
        for i in 1..price_points.len() {
            let return_val = (price_points[i].price / price_points[i-1].price).ln();
            returns.push(return_val);
        }
        
        // Calculate standard deviation
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        // Annualize (assuming 252 trading days, 6.5 hours per day, data in seconds)
        let periods_per_year = 252.0 * 6.5 * 3600.0;
        let sample_period = if price_points.len() > 1 {
            let time_diff = price_points.last().unwrap().timestamp_ns - 
                          price_points.first().unwrap().timestamp_ns;
            (time_diff as f64 / 1e9) / price_points.len() as f64
        } else {
            1.0
        };
        
        let annualization_factor = (periods_per_year / sample_period).sqrt();
        
        variance.sqrt() * annualization_factor
    }
    
    /// Detect current market regime
    fn detect_regime(&mut self) {
        let vol_threshold_high = 0.25;  // 25% annualized
        let vol_threshold_low = 0.10;   // 10% annualized
        let momentum_threshold = 0.3;
        let volume_threshold = self.volume_accumulator.read().get_average_volume() * 1.5;
        
        let prev_regime = self.regime;
        
        // Determine new regime
        self.regime = if self.volatility > vol_threshold_high {
            MarketRegime::Volatile
        } else if self.volatility < vol_threshold_low && self.volume < volume_threshold {
            MarketRegime::Quiet
        } else if self.momentum.abs() > momentum_threshold {
            MarketRegime::Trending
        } else if self.spread_volatility < self.spread * 0.2 {
            MarketRegime::RangeBoound
        } else {
            MarketRegime::Transitioning
        };
        
        // Update confidence
        if self.regime == prev_regime {
            self.regime_confidence = (self.regime_confidence * 1.05).min(1.0);
        } else {
            self.regime_confidence = 0.5;  // Reset on regime change
        }
    }
    
    /// Get risk-adjusted metrics
    pub fn get_risk_metrics(&self) -> RiskMetrics {
        RiskMetrics {
            volatility_percentile: self.get_volatility_percentile(),
            spread_cost_bps: (self.spread / self.get_mid_price() * 10000.0) as f32,
            toxicity_score: self.order_flow_toxicity as f32,
            liquidity_risk: 1.0 - self.liquidity_score as f32,
            regime_risk: self.get_regime_risk(),
        }
    }
    
    fn get_volatility_percentile(&self) -> f32 {
        // Map volatility to percentile (rough approximation)
        match self.volatility {
            v if v < 0.10 => 0.1,
            v if v < 0.15 => 0.25,
            v if v < 0.20 => 0.50,
            v if v < 0.30 => 0.75,
            v if v < 0.40 => 0.90,
            _ => 0.99,
        }
    }
    
    fn get_mid_price(&self) -> f64 {
        self.price_window.read()
            .values()
            .last()
            .map(|p| p.price)
            .unwrap_or(150.0)
    }
    
    fn get_regime_risk(&self) -> f32 {
        match self.regime {
            MarketRegime::Quiet => 0.1,
            MarketRegime::RangeBoound => 0.3,
            MarketRegime::Trending => 0.5,
            MarketRegime::Transitioning => 0.7,
            MarketRegime::Volatile => 0.9,
        }
    }
}

/// Volume accumulator with exponential decay
struct VolumeAccumulator {
    current_volume: u64,
    decay_factor: f64,
    last_update_ns: u64,
    volume_history: VecDeque<u64>,
}

impl VolumeAccumulator {
    fn new() -> Self {
        Self {
            current_volume: 0,
            decay_factor: 0.9999,  // Decay per nanosecond
            last_update_ns: 0,
            volume_history: VecDeque::with_capacity(100),
        }
    }
    
    fn add_volume(&mut self, volume: u64, timestamp_ns: u64) {
        // Apply decay
        if self.last_update_ns > 0 {
            let time_diff = timestamp_ns - self.last_update_ns;
            let decay = self.decay_factor.powi(time_diff as i32 / 1_000_000); // Per millisecond
            self.current_volume = (self.current_volume as f64 * decay) as u64;
        }
        
        // Add new volume
        self.current_volume += volume;
        
        // Track history
        self.volume_history.push_back(volume);
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }
        
        self.last_update_ns = timestamp_ns;
    }
    
    fn get_current_volume(&self) -> u64 {
        self.current_volume
    }
    
    fn get_average_volume(&self) -> u64 {
        if self.volume_history.is_empty() {
            return 1000;  // Default
        }
        
        let sum: u64 = self.volume_history.iter().sum();
        sum / self.volume_history.len() as u64
    }
}

/// Momentum calculator
struct MomentumCalculator {
    prices: VecDeque<(f64, u64)>,  // (price, timestamp)
    momentum: f64,
}

impl MomentumCalculator {
    fn new() -> Self {
        Self {
            prices: VecDeque::with_capacity(50),
            momentum: 0.0,
        }
    }
    
    fn update(&mut self, price: f64, timestamp_ns: u64) {
        self.prices.push_back((price, timestamp_ns));
        
        if self.prices.len() > 50 {
            self.prices.pop_front();
        }
        
        if self.prices.len() >= 10 {
            // Calculate short-term vs long-term momentum
            let recent_prices: Vec<f64> = self.prices.iter()
                .rev()
                .take(5)
                .map(|(p, _)| *p)
                .collect();
            
            let older_prices: Vec<f64> = self.prices.iter()
                .skip(self.prices.len() - 10)
                .take(5)
                .map(|(p, _)| *p)
                .collect();
            
            let recent_avg = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            let older_avg = older_prices.iter().sum::<f64>() / older_prices.len() as f64;
            
            // Normalize momentum to [-1, 1]
            let raw_momentum = (recent_avg - older_avg) / older_avg;
            self.momentum = raw_momentum.max(-1.0).min(1.0);
        }
    }
    
    fn get_momentum(&self) -> f64 {
        self.momentum
    }
}

/// Risk metrics for trading decisions
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub volatility_percentile: f32,
    pub spread_cost_bps: f32,
    pub toxicity_score: f32,
    pub liquidity_risk: f32,
    pub regime_risk: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::Side;
    
    #[test]
    fn test_market_state_update() {
        let mut state = MarketState::default();
        
        let trade = TradeData {
            symbol: crate::event_bus::Symbol::AAPL,
            price: 150.0,
            quantity: 1000,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
            aggressor_side: Side::Buy,
            trade_id: 1,
        };
        
        state.update_with_trade(&trade);
        
        assert_eq!(state.volume, 1000);
        assert!(state.volume_imbalance > 0.0);
    }
    
    #[test]
    fn test_volatility_calculation() {
        let mut state = MarketState::new(20, 10);
        
        // Add price points with increasing volatility
        for i in 0..30 {
            let price = 150.0 + (i as f64).sin() * (i as f64 / 5.0);
            state.price_window.write().push(PricePoint {
                price,
                timestamp_ns: i * 1_000_000_000,
                volume: 1000,
            });
        }
        
        let vol = state.calculate_volatility();
        assert!(vol > 0.0);
        assert!(vol < 1.0);  // Should be reasonable
    }
    
    #[test]
    fn test_regime_detection() {
        let mut state = MarketState::default();
        
        // Simulate high volatility
        state.volatility = 0.35;
        state.detect_regime();
        assert_eq!(state.regime, MarketRegime::Volatile);
        
        // Simulate trending market
        state.volatility = 0.15;
        state.momentum = 0.5;
        state.detect_regime();
        assert_eq!(state.regime, MarketRegime::Trending);
        
        // Simulate quiet market
        state.volatility = 0.08;
        state.momentum = 0.0;
        state.volume = 100;
        state.detect_regime();
        assert_eq!(state.regime, MarketRegime::Quiet);
    }
}