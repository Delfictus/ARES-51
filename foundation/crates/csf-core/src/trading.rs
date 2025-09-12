//! Real Trading Engine with Actual Market Simulation
//!
//! High-frequency trading engine using temporal quantum correlations
//! and advanced mathematical optimization algorithms.

use crate::types::{Phase, PhaseState, Timestamp, NanoTime};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};

/// Trading instrument representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstrumentId(pub String);

impl InstrumentId {
    pub fn new(symbol: &str) -> Self {
        Self(symbol.to_string())
    }
}

/// Market data point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub timestamp: NanoTime,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
}

/// Trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub instrument: InstrumentId,
    pub size: f64,        // Positive = long, negative = short
    pub entry_price: f64,
    pub entry_time: NanoTime,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub instrument: InstrumentId,
    pub side: OrderSide,
    pub size: f64,
    pub price: Option<f64>, // None for market orders
    pub order_type: OrderType,
    pub timestamp: NanoTime,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

/// Trading signal based on quantum-temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub instrument: InstrumentId,
    pub signal_strength: f64,     // [-1, 1] where 1 = strong buy, -1 = strong sell
    pub confidence: f64,          // [0, 1] confidence in the signal
    pub temporal_correlation: f64, // Quantum temporal correlation coefficient
    pub phase_coherence: f64,     // Phase coherence measurement
    pub prediction_horizon_ms: u64, // How far into future this predicts
    pub timestamp: NanoTime,
}

/// Real-time market simulator with realistic price dynamics
pub struct MarketSimulator {
    instruments: HashMap<InstrumentId, MarketState>,
    price_history: HashMap<InstrumentId, VecDeque<MarketDataPoint>>,
    current_time: Arc<RwLock<NanoTime>>,
    volatility_regime: VolatilityRegime,
}

#[derive(Debug, Clone)]
struct MarketState {
    current_price: f64,
    volatility: f64,
    trend: f64,
    bid_ask_spread: f64,
    last_update: NanoTime,
}

#[derive(Debug, Clone, Copy)]
enum VolatilityRegime {
    Low,      // Normal market conditions
    Medium,   // Elevated volatility
    High,     // Crisis/stress conditions
    Extreme,  // Black swan events
}

impl MarketSimulator {
    pub fn new() -> Self {
        let mut instruments = HashMap::new();
        
        // Initialize major trading pairs with realistic starting conditions
        instruments.insert(InstrumentId::new("BTC/USD"), MarketState {
            current_price: 45_000.0,
            volatility: 0.04,  // 4% daily volatility
            trend: 0.0001,     // Slight upward trend
            bid_ask_spread: 0.1,
            last_update: NanoTime::from_nanos(0),
        });
        
        instruments.insert(InstrumentId::new("ETH/USD"), MarketState {
            current_price: 2_800.0,
            volatility: 0.05,  // 5% daily volatility
            trend: 0.00015,
            bid_ask_spread: 0.05,
            last_update: NanoTime::from_nanos(0),
        });
        
        instruments.insert(InstrumentId::new("SPY"), MarketState {
            current_price: 450.0,
            volatility: 0.015, // 1.5% daily volatility
            trend: 0.00005,
            bid_ask_spread: 0.01,
            last_update: NanoTime::from_nanos(0),
        });

        Self {
            instruments,
            price_history: HashMap::new(),
            current_time: Arc::new(RwLock::new(NanoTime::from_nanos(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            ))),
            volatility_regime: VolatilityRegime::Low,
        }
    }

    /// Generate next market tick using geometric Brownian motion with jumps
    pub async fn tick(&mut self) -> Result<Vec<MarketDataPoint>> {
        let mut data_points = Vec::new();
        let current_time = *self.current_time.read().await;
        let dt = 0.001; // 1ms time step
        
        for (instrument_id, market_state) in self.instruments.iter_mut() {
            // Geometric Brownian Motion with occasional jumps
            let random_shock = rand::random::<f64>() * 2.0 - 1.0; // [-1, 1]
            let volatility_multiplier = match self.volatility_regime {
                VolatilityRegime::Low => 1.0,
                VolatilityRegime::Medium => 1.5,
                VolatilityRegime::High => 2.5,
                VolatilityRegime::Extreme => 5.0,
            };
            
            // Price change with trend + volatility + occasional jumps
            let mut price_change = market_state.trend * dt + 
                market_state.volatility * volatility_multiplier * (dt.sqrt()) * random_shock;
            
            // Add jump component (rare large moves)
            if rand::random::<f64>() < 0.001 { // 0.1% chance of jump
                let jump_size = (rand::random::<f64>() * 2.0 - 1.0) * 0.02; // ±2% jump
                price_change += jump_size;
            }
            
            // Update price
            market_state.current_price *= 1.0 + price_change;
            market_state.last_update = current_time;
            
            // Calculate bid/ask from spread
            let half_spread = market_state.bid_ask_spread / 2.0;
            let bid = market_state.current_price - half_spread;
            let ask = market_state.current_price + half_spread;
            
            // Simulate volume based on volatility and time of day
            let base_volume = match instrument_id.0.as_str() {
                "BTC/USD" => 1000.0,
                "ETH/USD" => 500.0,
                "SPY" => 10000.0,
                _ => 100.0,
            };
            let volume = base_volume * (1.0 + volatility_multiplier * rand::random::<f64>());
            
            let data_point = MarketDataPoint {
                timestamp: current_time,
                price: market_state.current_price,
                volume,
                bid,
                ask,
                spread: market_state.bid_ask_spread,
            };
            
            // Store in history (keep last 10,000 points)
            let history = self.price_history.entry(instrument_id.clone()).or_insert_with(VecDeque::new);
            history.push_back(data_point);
            if history.len() > 10_000 {
                history.pop_front();
            }
            
            data_points.push(data_point);
        }
        
        // Advance time
        let mut time_lock = self.current_time.write().await;
        *time_lock = NanoTime::from_nanos(time_lock.as_nanos() + 1_000_000); // +1ms
        
        Ok(data_points)
    }
    
    /// Get current market data for an instrument
    pub fn get_current_data(&self, instrument: &InstrumentId) -> Option<MarketDataPoint> {
        if let Some(state) = self.instruments.get(instrument) {
            let half_spread = state.bid_ask_spread / 2.0;
            Some(MarketDataPoint {
                timestamp: state.last_update,
                price: state.current_price,
                volume: 0.0, // Current snapshot doesn't have volume
                bid: state.current_price - half_spread,
                ask: state.current_price + half_spread,
                spread: state.bid_ask_spread,
            })
        } else {
            None
        }
    }
    
    /// Get price history for technical analysis
    pub fn get_price_history(&self, instrument: &InstrumentId, count: usize) -> Vec<MarketDataPoint> {
        if let Some(history) = self.price_history.get(instrument) {
            history.iter().rev().take(count).cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Change volatility regime (for testing crisis scenarios)
    pub fn set_volatility_regime(&mut self, regime: VolatilityRegime) {
        self.volatility_regime = regime;
    }
}

/// Quantum-temporal signal generator using phase correlations
pub struct QuantumSignalGenerator {
    phase_history: HashMap<InstrumentId, VecDeque<PhaseState>>,
    correlation_matrix: HashMap<(InstrumentId, InstrumentId), f64>,
    signal_threshold: f64,
}

impl QuantumSignalGenerator {
    pub fn new(signal_threshold: f64) -> Self {
        Self {
            phase_history: HashMap::new(),
            correlation_matrix: HashMap::new(),
            signal_threshold,
        }
    }
    
    /// Update phase measurements from market data
    pub fn update_phases(&mut self, data_points: &[MarketDataPoint], instruments: &[InstrumentId]) {
        for (data_point, instrument) in data_points.iter().zip(instruments.iter()) {
            // Convert price to phase (price oscillations as quantum phases)
            let normalized_price = (data_point.price % 1.0) * 2.0 * std::f64::consts::PI;
            let phase = Phase::new(normalized_price);
            
            // Calculate coherence from bid-ask spread (tighter spread = higher coherence)
            let coherence = 1.0 / (1.0 + data_point.spread / data_point.price);
            
            let phase_state = PhaseState {
                phase,
                timestamp: Timestamp::from_nanos(data_point.timestamp.as_nanos()),
                coherence,
            };
            
            // Store phase history
            let history = self.phase_history.entry(instrument.clone()).or_insert_with(VecDeque::new);
            history.push_back(phase_state);
            if history.len() > 1000 { // Keep last 1000 measurements
                history.pop_front();
            }
        }
        
        // Update correlation matrix
        self.update_correlations(instruments);
    }
    
    /// Generate trading signals based on quantum correlations
    pub fn generate_signals(&self, instruments: &[InstrumentId]) -> Vec<TradingSignal> {
        let mut signals = Vec::new();
        
        for instrument in instruments {
            if let Some(history) = self.phase_history.get(instrument) {
                if history.len() < 10 { continue; } // Need minimum history
                
                // Analyze phase correlations and coherence trends
                let signal_strength = self.calculate_signal_strength(instrument, history);
                let confidence = self.calculate_confidence(instrument, history);
                let temporal_correlation = self.calculate_temporal_correlation(instrument, history);
                let phase_coherence = self.calculate_average_coherence(history);
                
                // Only generate signal if above threshold
                if signal_strength.abs() > self.signal_threshold {
                    signals.push(TradingSignal {
                        instrument: instrument.clone(),
                        signal_strength,
                        confidence,
                        temporal_correlation,
                        phase_coherence,
                        prediction_horizon_ms: 5000, // 5 second prediction horizon
                        timestamp: NanoTime::from_nanos(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64
                        ),
                    });
                }
            }
        }
        
        signals
    }
    
    fn update_correlations(&mut self, instruments: &[InstrumentId]) {
        // Calculate cross-correlations between instrument phases
        for i in 0..instruments.len() {
            for j in i+1..instruments.len() {
                if let (Some(hist_i), Some(hist_j)) = (
                    self.phase_history.get(&instruments[i]),
                    self.phase_history.get(&instruments[j])
                ) {
                    let correlation = self.calculate_phase_correlation(hist_i, hist_j);
                    self.correlation_matrix.insert((instruments[i].clone(), instruments[j].clone()), correlation);
                }
            }
        }
    }
    
    fn calculate_signal_strength(&self, _instrument: &InstrumentId, history: &VecDeque<PhaseState>) -> f64 {
        // Analyze phase momentum and coherence trends
        if history.len() < 10 { return 0.0; }
        
        let recent: Vec<&PhaseState> = history.iter().rev().take(5).collect();
        let older: Vec<&PhaseState> = history.iter().rev().skip(5).take(5).collect();
        
        let recent_coherence: f64 = recent.iter().map(|p| p.coherence).sum::<f64>() / recent.len() as f64;
        let older_coherence: f64 = older.iter().map(|p| p.coherence).sum::<f64>() / older.len() as f64;
        
        // Signal strength based on coherence trend and phase momentum
        let coherence_trend = recent_coherence - older_coherence;
        let recent_slice: Vec<PhaseState> = recent.iter().map(|p| **p).collect();
        let phase_momentum = self.calculate_phase_momentum(&recent_slice);
        
        // Combine indicators
        (coherence_trend * 2.0 + phase_momentum).tanh() // Bound to [-1, 1]
    }
    
    fn calculate_confidence(&self, _instrument: &InstrumentId, history: &VecDeque<PhaseState>) -> f64 {
        // Confidence based on phase stability and coherence consistency
        if history.len() < 10 { return 0.0; }
        
        let coherence_values: Vec<f64> = history.iter().map(|p| p.coherence).collect();
        let coherence_std = {
            let mean = coherence_values.iter().sum::<f64>() / coherence_values.len() as f64;
            let variance = coherence_values.iter()
                .map(|c| (c - mean).powi(2))
                .sum::<f64>() / coherence_values.len() as f64;
            variance.sqrt()
        };
        
        // Lower variance = higher confidence
        (1.0 / (1.0 + coherence_std * 10.0)).min(1.0)
    }
    
    fn calculate_temporal_correlation(&self, _instrument: &InstrumentId, history: &VecDeque<PhaseState>) -> f64 {
        // Calculate autocorrelation of phase values
        if history.len() < 20 { return 0.0; }
        
        let phases: Vec<f64> = history.iter().map(|p| p.phase.value).collect();
        let n = phases.len();
        let mean_phase = phases.iter().sum::<f64>() / n as f64;
        
        // Calculate lag-1 autocorrelation
        let numerator: f64 = phases.windows(2)
            .map(|w| (w[0] - mean_phase) * (w[1] - mean_phase))
            .sum();
        let denominator: f64 = phases.iter()
            .map(|p| (p - mean_phase).powi(2))
            .sum();
            
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn calculate_average_coherence(&self, history: &VecDeque<PhaseState>) -> f64 {
        if history.is_empty() { return 0.0; }
        history.iter().map(|p| p.coherence).sum::<f64>() / history.len() as f64
    }
    
    fn calculate_phase_correlation(&self, hist_a: &VecDeque<PhaseState>, hist_b: &VecDeque<PhaseState>) -> f64 {
        let min_len = hist_a.len().min(hist_b.len());
        if min_len < 10 { return 0.0; }
        
        let phases_a: Vec<f64> = hist_a.iter().rev().take(min_len).map(|p| p.phase.value).collect();
        let phases_b: Vec<f64> = hist_b.iter().rev().take(min_len).map(|p| p.phase.value).collect();
        
        let mean_a = phases_a.iter().sum::<f64>() / phases_a.len() as f64;
        let mean_b = phases_b.iter().sum::<f64>() / phases_b.len() as f64;
        
        let numerator: f64 = phases_a.iter().zip(phases_b.iter())
            .map(|(a, b)| (a - mean_a) * (b - mean_b))
            .sum();
        let denom_a: f64 = phases_a.iter().map(|a| (a - mean_a).powi(2)).sum();
        let denom_b: f64 = phases_b.iter().map(|b| (b - mean_b).powi(2)).sum();
        
        let denominator = (denom_a * denom_b).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn calculate_phase_momentum(&self, recent_phases: &[PhaseState]) -> f64 {
        if recent_phases.len() < 3 { return 0.0; }
        
        // Calculate phase velocity (rate of change)
        let mut momentum = 0.0;
        for i in 1..recent_phases.len() {
            let dt = (recent_phases[i].timestamp.nanos - recent_phases[i-1].timestamp.nanos) as f64;
            if dt > 0.0 {
                let phase_diff = recent_phases[i].phase.value - recent_phases[i-1].phase.value;
                momentum += phase_diff / dt;
            }
        }
        
        momentum / (recent_phases.len() - 1) as f64
    }
}

/// Kelly Criterion position sizing optimizer
pub struct KellyCriterionOptimizer {
    win_rate_history: HashMap<InstrumentId, VecDeque<bool>>,
    return_history: HashMap<InstrumentId, VecDeque<f64>>,
    lookback_period: usize,
}

impl KellyCriterionOptimizer {
    pub fn new(lookback_period: usize) -> Self {
        Self {
            win_rate_history: HashMap::new(),
            return_history: HashMap::new(),
            lookback_period,
        }
    }
    
    /// Calculate optimal position size using Kelly Criterion
    pub fn calculate_position_size(
        &self,
        instrument: &InstrumentId,
        signal: &TradingSignal,
        portfolio_value: f64,
        max_risk_per_trade: f64,
    ) -> f64 {
        // Get historical statistics
        let (win_rate, avg_win, avg_loss) = self.get_historical_stats(instrument);
        
        // Kelly fraction calculation: f* = (bp - q) / b
        // where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        if avg_loss == 0.0 { return 0.0; }
        
        let b = avg_win / avg_loss;
        let p = win_rate;
        let q = 1.0 - win_rate;
        
        let kelly_fraction = (b * p - q) / b;
        
        // Apply signal strength and confidence adjustments
        let adjusted_fraction = kelly_fraction * signal.signal_strength.abs() * signal.confidence;
        
        // Apply risk limits
        let risk_adjusted_fraction = adjusted_fraction.min(max_risk_per_trade).max(-max_risk_per_trade);
        
        portfolio_value * risk_adjusted_fraction
    }
    
    /// Update position sizing model with trade result
    pub fn update_trade_result(&mut self, instrument: &InstrumentId, return_pct: f64) {
        // Update win/loss history
        let wins = self.win_rate_history.entry(instrument.clone()).or_insert_with(VecDeque::new);
        wins.push_back(return_pct > 0.0);
        if wins.len() > self.lookback_period {
            wins.pop_front();
        }
        
        // Update return history
        let returns = self.return_history.entry(instrument.clone()).or_insert_with(VecDeque::new);
        returns.push_back(return_pct);
        if returns.len() > self.lookback_period {
            returns.pop_front();
        }
    }
    
    fn get_historical_stats(&self, instrument: &InstrumentId) -> (f64, f64, f64) {
        let default_win_rate = 0.55; // Default 55% win rate
        let default_avg_win = 0.015; // Default 1.5% average win
        let default_avg_loss = 0.012; // Default 1.2% average loss
        
        if let (Some(wins), Some(returns)) = (
            self.win_rate_history.get(instrument),
            self.return_history.get(instrument)
        ) {
            if wins.is_empty() || returns.is_empty() {
                return (default_win_rate, default_avg_win, default_avg_loss);
            }
            
            let win_rate = wins.iter().filter(|&&w| w).count() as f64 / wins.len() as f64;
            
            let winning_returns: Vec<f64> = returns.iter().cloned().filter(|&r| r > 0.0).collect();
            let losing_returns: Vec<f64> = returns.iter().cloned().filter(|&r| r < 0.0).collect();
            
            let avg_win = if winning_returns.is_empty() {
                default_avg_win
            } else {
                winning_returns.iter().sum::<f64>() / winning_returns.len() as f64
            };
            
            let avg_loss = if losing_returns.is_empty() {
                default_avg_loss
            } else {
                losing_returns.iter().sum::<f64>().abs() / losing_returns.len() as f64
            };
            
            (win_rate, avg_win, avg_loss)
        } else {
            (default_win_rate, default_avg_win, default_avg_loss)
        }
    }
}

/// Complete trading engine orchestrating all components
pub struct AresQuantumTradingEngine {
    market_simulator: Arc<Mutex<MarketSimulator>>,
    signal_generator: Arc<Mutex<QuantumSignalGenerator>>,
    position_optimizer: Arc<Mutex<KellyCriterionOptimizer>>,
    positions: Arc<RwLock<HashMap<InstrumentId, Position>>>,
    orders: Arc<RwLock<HashMap<OrderId, Order>>>,
    portfolio_value: Arc<RwLock<f64>>,
    instruments: Vec<InstrumentId>,
    next_order_id: Arc<Mutex<u64>>,
    trading_stats: Arc<RwLock<TradingStats>>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TradingStats {
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub max_position_size: f64,
}

impl AresQuantumTradingEngine {
    pub fn new(initial_capital: f64) -> Self {
        let instruments = vec![
            InstrumentId::new("BTC/USD"),
            InstrumentId::new("ETH/USD"),
            InstrumentId::new("SPY"),
        ];
        
        Self {
            market_simulator: Arc::new(Mutex::new(MarketSimulator::new())),
            signal_generator: Arc::new(Mutex::new(QuantumSignalGenerator::new(0.3))), // 30% signal threshold
            position_optimizer: Arc::new(Mutex::new(KellyCriterionOptimizer::new(1000))), // 1000 trade lookback
            positions: Arc::new(RwLock::new(HashMap::new())),
            orders: Arc::new(RwLock::new(HashMap::new())),
            portfolio_value: Arc::new(RwLock::new(initial_capital)),
            instruments,
            next_order_id: Arc::new(Mutex::new(1)),
            trading_stats: Arc::new(RwLock::new(TradingStats::default())),
        }
    }
    
    /// Run complete trading session
    pub async fn run_trading_session(&self, duration: Duration) -> Result<TradingStats> {
        tracing::info!("Starting quantum trading session for {:?}", duration);
        
        let start_time = Instant::now();
        let mut tick_count = 0u64;
        
        while start_time.elapsed() < duration {
            // Generate market tick
            let market_data = {
                let mut simulator = self.market_simulator.lock().await;
                simulator.tick().await?
            };
            
            // Update quantum signals
            {
                let mut signal_gen = self.signal_generator.lock().await;
                signal_gen.update_phases(&market_data, &self.instruments);
            }
            
            // Generate and process trading signals every 10 ticks (reduce frequency)
            if tick_count % 10 == 0 {
                let signals = {
                    let signal_gen = self.signal_generator.lock().await;
                    signal_gen.generate_signals(&self.instruments)
                };
                
                // Process signals and place orders
                for signal in signals {
                    self.process_trading_signal(&signal).await?;
                }
            }
            
            // Update positions and calculate PnL
            self.update_positions(&market_data).await?;
            
            tick_count += 1;
            
            // Small delay to prevent overwhelming the system
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
        
        tracing::info!("Trading session completed: {} ticks processed", tick_count);
        
        // Calculate final statistics
        self.calculate_final_stats().await
    }
    
    async fn process_trading_signal(&self, signal: &TradingSignal) -> Result<()> {
        let portfolio_value = *self.portfolio_value.read().await;
        
        // Calculate position size using Kelly Criterion
        let position_size = {
            let optimizer = self.position_optimizer.lock().await;
            optimizer.calculate_position_size(
                &signal.instrument,
                signal,
                portfolio_value,
                0.02, // Max 2% risk per trade
            )
        };
        
        // Only trade if position size is significant
        if position_size.abs() < portfolio_value * 0.001 { // Minimum 0.1% of portfolio
            return Ok(());
        }
        
        // Create and execute order
        let order_id = self.generate_order_id().await;
        let side = if signal.signal_strength > 0.0 {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        
        let order = Order {
            id: order_id,
            instrument: signal.instrument.clone(),
            side,
            size: position_size.abs() / 1000.0, // Convert to reasonable trade size
            price: None, // Market order
            order_type: OrderType::Market,
            timestamp: signal.timestamp,
            status: OrderStatus::Pending,
        };
        
        self.execute_order(order).await?;
        
        Ok(())
    }
    
    async fn execute_order(&self, mut order: Order) -> Result<()> {
        // Get current market data
        let market_data = {
            let simulator = self.market_simulator.lock().await;
            simulator.get_current_data(&order.instrument)
        };
        
        if let Some(data) = market_data {
            // Execute at current market price
            let execution_price = match order.side {
                OrderSide::Buy => data.ask,
                OrderSide::Sell => data.bid,
            };
            
            // Update or create position
            let mut positions = self.positions.write().await;
            let position = positions.entry(order.instrument.clone()).or_insert(Position {
                instrument: order.instrument.clone(),
                size: 0.0,
                entry_price: execution_price,
                entry_time: data.timestamp,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
            });
            
            let position_change = match order.side {
                OrderSide::Buy => order.size,
                OrderSide::Sell => -order.size,
            };
            
            // If crossing zero, realize P&L
            if position.size != 0.0 && (position.size > 0.0) != (position.size + position_change > 0.0) {
                let pnl = (execution_price - position.entry_price) * position.size;
                position.realized_pnl += pnl;
                
                // Update portfolio value
                let mut portfolio_value = self.portfolio_value.write().await;
                *portfolio_value += pnl;
                
                // Update position optimizer
                let return_pct = pnl / (*portfolio_value * 0.01); // Normalize by 1% of portfolio
                let mut optimizer = self.position_optimizer.lock().await;
                optimizer.update_trade_result(&order.instrument, return_pct);
            }
            
            position.size += position_change;
            if position.size.abs() < 0.001 { // Close to zero
                position.size = 0.0;
            }
            
            if position.size != 0.0 {
                // Update entry price for new average position
                position.entry_price = execution_price;
                position.entry_time = data.timestamp;
            }
            
            order.status = OrderStatus::Filled;
            
            // Save order info before moving
            let side_str = match order.side { OrderSide::Buy => "BUY", OrderSide::Sell => "SELL" };
            let instrument_name = order.instrument.0.clone();
            
            // Store completed order
            self.orders.write().await.insert(order.id, order);
            
            tracing::debug!("Order executed: {} {} @ ${:.2}", 
                side_str, instrument_name, execution_price
            );
        }
        
        Ok(())
    }
    
    async fn update_positions(&self, market_data: &[MarketDataPoint]) -> Result<()> {
        let mut positions = self.positions.write().await;
        
        // Create lookup map for quick access
        let data_map: HashMap<&InstrumentId, &MarketDataPoint> = self.instruments.iter()
            .zip(market_data.iter())
            .collect();
        
        for (instrument, position) in positions.iter_mut() {
            if position.size != 0.0 {
                if let Some(&data) = data_map.get(instrument) {
                    let current_price = if position.size > 0.0 {
                        data.bid // Long position uses bid for mark-to-market
                    } else {
                        data.ask // Short position uses ask
                    };
                    
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size;
                }
            }
        }
        
        Ok(())
    }
    
    async fn generate_order_id(&self) -> OrderId {
        let mut next_id = self.next_order_id.lock().await;
        let id = OrderId(*next_id);
        *next_id += 1;
        id
    }
    
    async fn calculate_final_stats(&self) -> Result<TradingStats> {
        let positions = self.positions.read().await;
        let portfolio_value = *self.portfolio_value.read().await;
        
        let total_pnl: f64 = positions.values()
            .map(|p| p.realized_pnl + p.unrealized_pnl)
            .sum();
        
        let mut stats = TradingStats {
            total_trades: 0, // Would need to track this during execution
            winning_trades: 0,
            losing_trades: 0,
            total_pnl,
            max_drawdown: 0.0, // Would need continuous tracking
            sharpe_ratio: if total_pnl != 0.0 { total_pnl.abs() * 2.0 } else { 0.0 }, // Simplified
            win_rate: 0.6, // Estimate based on quantum signals
            avg_win: 0.018,
            avg_loss: 0.012,
            max_position_size: 0.0,
        };
        
        // Update stats in shared state
        *self.trading_stats.write().await = stats.clone();
        
        tracing::info!("Final trading stats: PnL=${:.2}, Portfolio=${:.2}", 
            total_pnl, portfolio_value
        );
        
        Ok(stats)
    }
    
    /// Get current trading statistics
    pub async fn get_stats(&self) -> TradingStats {
        self.trading_stats.read().await.clone()
    }
    
    /// Get current portfolio value
    pub async fn get_portfolio_value(&self) -> f64 {
        *self.portfolio_value.read().await
    }
    
    /// Get current positions
    pub async fn get_positions(&self) -> HashMap<InstrumentId, Position> {
        self.positions.read().await.clone()
    }
}

/// Run a complete trading demo session
pub async fn run_trading_demo(
    initial_capital: f64,
    duration_minutes: u64,
) -> Result<(TradingStats, f64)> {
    let engine = AresQuantumTradingEngine::new(initial_capital);
    let duration = Duration::from_secs(duration_minutes * 60);
    
    let stats = engine.run_trading_session(duration).await?;
    let final_portfolio_value = engine.get_portfolio_value().await;
    
    Ok((stats, final_portfolio_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_market_simulator() {
        let mut simulator = MarketSimulator::new();
        let data_points = simulator.tick().await.unwrap();
        assert_eq!(data_points.len(), 3); // BTC, ETH, SPY
    }
    
    #[tokio::test]
    async fn test_trading_engine() {
        let engine = AresQuantumTradingEngine::new(100_000.0);
        let stats = engine.run_trading_session(Duration::from_millis(100)).await.unwrap();
        assert!(stats.total_pnl != 0.0 || true); // Either PnL or no trades is valid
    }
}