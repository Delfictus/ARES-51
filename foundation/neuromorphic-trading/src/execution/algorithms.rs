//! Execution algorithms (TWAP, VWAP, Iceberg, etc.)

use crate::exchanges::{Symbol, Exchange, Side};
use crate::paper_trading::{Order, OrderType};
use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use dashmap::DashMap;

/// Execution algorithm type
#[derive(Clone, Debug)]
pub enum AlgorithmType {
    TWAP,     // Time-Weighted Average Price
    VWAP,     // Volume-Weighted Average Price
    ICEBERG,  // Iceberg orders
    POV,      // Percentage of Volume
    SNIPER,   // Opportunistic execution
    DCA,      // Dollar Cost Averaging
}

/// Execution parameters
#[derive(Clone, Debug)]
pub struct ExecutionParams {
    pub algorithm: AlgorithmType,
    pub total_quantity: f64,
    pub duration: Duration,
    pub urgency: f64,
    pub price_limit: Option<f64>,
    pub min_slice_size: f64,
    pub max_slice_size: f64,
    pub participation_rate: f64,  // For POV
    pub show_quantity: Option<f64>, // For Iceberg
}

impl Default for ExecutionParams {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmType::TWAP,
            total_quantity: 1.0,
            duration: Duration::from_secs(300),
            urgency: 0.5,
            price_limit: None,
            min_slice_size: 0.01,
            max_slice_size: 0.1,
            participation_rate: 0.1,
            show_quantity: None,
        }
    }
}

/// Execution progress
#[derive(Clone, Debug)]
pub struct ExecutionProgress {
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub slices_executed: usize,
    pub start_time: Instant,
    pub elapsed: Duration,
    pub estimated_completion: Duration,
}

/// Base trait for execution algorithms
pub trait ExecutionAlgorithm: Send + Sync {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> Option<OrderSlice>;
    
    fn should_execute(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> bool;
    
    fn adjust_for_market(
        &self,
        slice: &mut OrderSlice,
        market_data: &MarketContext,
    );
}

/// Market context for execution decisions
#[derive(Clone, Debug)]
pub struct MarketContext {
    pub current_price: f64,
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
    pub volume: f64,
    pub volatility: f64,
    pub trend: f64,
}

/// Order slice for execution
#[derive(Clone, Debug)]
pub struct OrderSlice {
    pub quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub time_in_force: Duration,
}

/// TWAP Algorithm
pub struct TWAPAlgorithm {
    slice_interval: Duration,
}

impl TWAPAlgorithm {
    pub fn new(duration: Duration, num_slices: usize) -> Self {
        Self {
            slice_interval: duration / num_slices as u32,
        }
    }
}

impl ExecutionAlgorithm for TWAPAlgorithm {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        _market_data: &MarketContext,
    ) -> Option<OrderSlice> {
        if progress.remaining_quantity <= 0.0 {
            return None;
        }
        
        // Calculate time-based slice size
        let time_remaining = params.duration.saturating_sub(progress.elapsed);
        let slices_remaining = (time_remaining.as_secs_f64() / self.slice_interval.as_secs_f64()).ceil() as f64;
        
        let slice_size = if slices_remaining > 0.0 {
            (progress.remaining_quantity / slices_remaining)
                .max(params.min_slice_size)
                .min(params.max_slice_size)
        } else {
            progress.remaining_quantity
        };
        
        Some(OrderSlice {
            quantity: slice_size,
            order_type: OrderType::Market,
            price: params.price_limit,
            time_in_force: self.slice_interval,
        })
    }
    
    fn should_execute(
        &self,
        _params: &ExecutionParams,
        progress: &ExecutionProgress,
        _market_data: &MarketContext,
    ) -> bool {
        // Execute at regular intervals
        let slices_done = progress.slices_executed as f64;
        let expected_slices = progress.elapsed.as_secs_f64() / self.slice_interval.as_secs_f64();
        
        slices_done < expected_slices
    }
    
    fn adjust_for_market(&self, _slice: &mut OrderSlice, _market_data: &MarketContext) {
        // TWAP doesn't adjust for market conditions
    }
}

/// VWAP Algorithm
pub struct VWAPAlgorithm {
    volume_profile: Vec<f64>,
    current_bucket: usize,
}

impl VWAPAlgorithm {
    pub fn new() -> Self {
        // Typical intraday volume profile (U-shaped)
        let volume_profile = vec![
            0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15,
        ];
        
        Self {
            volume_profile,
            current_bucket: 0,
        }
    }
}

impl ExecutionAlgorithm for VWAPAlgorithm {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> Option<OrderSlice> {
        if progress.remaining_quantity <= 0.0 {
            return None;
        }
        
        // Get expected volume percentage for current time bucket
        let bucket_idx = self.current_bucket.min(self.volume_profile.len() - 1);
        let volume_pct = self.volume_profile[bucket_idx];
        
        // Calculate slice based on volume profile
        let expected_quantity = params.total_quantity * volume_pct;
        let slice_size = expected_quantity
            .min(progress.remaining_quantity)
            .max(params.min_slice_size)
            .min(params.max_slice_size);
        
        // Use limit orders near the spread
        let price = if market_data.spread > 0.001 {
            Some(market_data.bid + market_data.spread * 0.3)
        } else {
            None
        };
        
        Some(OrderSlice {
            quantity: slice_size,
            order_type: if price.is_some() { OrderType::Limit } else { OrderType::Market },
            price,
            time_in_force: Duration::from_secs(30),
        })
    }
    
    fn should_execute(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> bool {
        // Execute based on volume participation
        let market_volume_rate = market_data.volume / 60.0; // Per minute
        let our_rate = progress.executed_quantity / progress.elapsed.as_secs_f64() * 60.0;
        
        our_rate < market_volume_rate * params.participation_rate
    }
    
    fn adjust_for_market(&self, slice: &mut OrderSlice, market_data: &MarketContext) {
        // Adjust for high volatility
        if market_data.volatility > 0.02 {
            slice.quantity *= 0.8; // Reduce size in volatile markets
        }
        
        // Adjust price for trend
        if let Some(price) = &mut slice.price {
            *price += market_data.trend * 0.0001; // Slight adjustment for trend
        }
    }
}

/// Iceberg Algorithm
pub struct IcebergAlgorithm {
    visible_ratio: f64,
}

impl IcebergAlgorithm {
    pub fn new(visible_ratio: f64) -> Self {
        Self { visible_ratio }
    }
}

impl ExecutionAlgorithm for IcebergAlgorithm {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> Option<OrderSlice> {
        if progress.remaining_quantity <= 0.0 {
            return None;
        }
        
        // Show only a portion of the total order
        let show_qty = params.show_quantity.unwrap_or(params.total_quantity * self.visible_ratio);
        let slice_size = show_qty.min(progress.remaining_quantity);
        
        // Place limit order at or near the spread
        let price = Some(market_data.bid + market_data.spread * 0.1);
        
        Some(OrderSlice {
            quantity: slice_size,
            order_type: OrderType::Limit,
            price,
            time_in_force: Duration::from_secs(60),
        })
    }
    
    fn should_execute(
        &self,
        _params: &ExecutionParams,
        progress: &ExecutionProgress,
        _market_data: &MarketContext,
    ) -> bool {
        // Execute when previous slice is filled
        progress.slices_executed == 0 || progress.remaining_quantity > 0.0
    }
    
    fn adjust_for_market(&self, slice: &mut OrderSlice, market_data: &MarketContext) {
        // Hide more in thin markets
        if market_data.volume < 1000.0 {
            slice.quantity *= 0.5;
        }
    }
}

/// POV (Percentage of Volume) Algorithm
pub struct POVAlgorithm {
    target_rate: f64,
    max_participation: f64,
}

impl POVAlgorithm {
    pub fn new(target_rate: f64) -> Self {
        Self {
            target_rate,
            max_participation: target_rate * 1.5,
        }
    }
}

impl ExecutionAlgorithm for POVAlgorithm {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> Option<OrderSlice> {
        if progress.remaining_quantity <= 0.0 {
            return None;
        }
        
        // Calculate slice based on market volume
        let market_rate = market_data.volume / 60.0; // Per minute
        let target_quantity = market_rate * self.target_rate;
        
        let slice_size = target_quantity
            .min(progress.remaining_quantity)
            .max(params.min_slice_size)
            .min(params.max_slice_size);
        
        Some(OrderSlice {
            quantity: slice_size,
            order_type: OrderType::Market,
            price: params.price_limit,
            time_in_force: Duration::from_secs(10),
        })
    }
    
    fn should_execute(
        &self,
        _params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> bool {
        // Check participation rate
        let our_rate = progress.executed_quantity / progress.elapsed.as_secs_f64();
        let market_rate = market_data.volume / 60.0;
        
        our_rate / market_rate < self.max_participation
    }
    
    fn adjust_for_market(&self, slice: &mut OrderSlice, market_data: &MarketContext) {
        // Increase participation in liquid markets
        if market_data.volume > 10000.0 {
            slice.quantity *= 1.2;
        }
    }
}

/// Sniper Algorithm for opportunistic execution
pub struct SniperAlgorithm {
    price_improvement_threshold: f64,
    size_threshold: f64,
}

impl SniperAlgorithm {
    pub fn new() -> Self {
        Self {
            price_improvement_threshold: 0.001, // 0.1%
            size_threshold: 0.5,
        }
    }
}

impl ExecutionAlgorithm for SniperAlgorithm {
    fn calculate_next_slice(
        &self,
        params: &ExecutionParams,
        progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> Option<OrderSlice> {
        if progress.remaining_quantity <= 0.0 {
            return None;
        }
        
        // Aggressive sizing when opportunity detected
        let slice_size = if self.detect_opportunity(market_data) {
            params.max_slice_size.min(progress.remaining_quantity)
        } else {
            params.min_slice_size
        };
        
        Some(OrderSlice {
            quantity: slice_size,
            order_type: OrderType::Market,
            price: None,
            time_in_force: Duration::from_secs(1),
        })
    }
    
    fn should_execute(
        &self,
        params: &ExecutionParams,
        _progress: &ExecutionProgress,
        market_data: &MarketContext,
    ) -> bool {
        // Execute when price is favorable
        if let Some(limit) = params.price_limit {
            market_data.current_price < limit * (1.0 - self.price_improvement_threshold)
        } else {
            self.detect_opportunity(market_data)
        }
    }
    
    fn adjust_for_market(&self, slice: &mut OrderSlice, market_data: &MarketContext) {
        // Be more aggressive in favorable conditions
        if market_data.trend < -0.001 {
            slice.quantity *= 1.5;
        }
    }
}

impl SniperAlgorithm {
    fn detect_opportunity(&self, market_data: &MarketContext) -> bool {
        // Detect price dips or high liquidity
        market_data.spread < 0.0005 || 
        market_data.volume > 5000.0 ||
        market_data.trend < -0.002
    }
}

/// Execution engine that runs algorithms
pub struct ExecutionEngine {
    algorithms: DashMap<String, Arc<dyn ExecutionAlgorithm>>,
    active_executions: DashMap<String, ExecutionState>,
    order_sender: mpsc::UnboundedSender<Order>,
}

/// Execution state tracking
struct ExecutionState {
    params: ExecutionParams,
    progress: ExecutionProgress,
    algorithm: Arc<dyn ExecutionAlgorithm>,
    symbol: Symbol,
    exchange: Exchange,
    side: Side,
}

impl ExecutionEngine {
    pub fn new(order_sender: mpsc::UnboundedSender<Order>) -> Self {
        let mut engine = Self {
            algorithms: DashMap::new(),
            active_executions: DashMap::new(),
            order_sender,
        };
        
        // Register default algorithms
        engine.register_algorithm("TWAP", Arc::new(TWAPAlgorithm::new(
            Duration::from_secs(300), 10
        )));
        engine.register_algorithm("VWAP", Arc::new(VWAPAlgorithm::new()));
        engine.register_algorithm("Iceberg", Arc::new(IcebergAlgorithm::new(0.2)));
        engine.register_algorithm("POV", Arc::new(POVAlgorithm::new(0.1)));
        engine.register_algorithm("Sniper", Arc::new(SniperAlgorithm::new()));
        
        engine
    }
    
    /// Register a custom algorithm
    pub fn register_algorithm(&mut self, name: &str, algorithm: Arc<dyn ExecutionAlgorithm>) {
        self.algorithms.insert(name.to_string(), algorithm);
    }
    
    /// Start execution with specified algorithm
    pub fn start_execution(
        &self,
        execution_id: String,
        symbol: Symbol,
        exchange: Exchange,
        side: Side,
        params: ExecutionParams,
    ) -> Result<()> {
        let algorithm = match params.algorithm {
            AlgorithmType::TWAP => self.algorithms.get("TWAP"),
            AlgorithmType::VWAP => self.algorithms.get("VWAP"),
            AlgorithmType::ICEBERG => self.algorithms.get("Iceberg"),
            AlgorithmType::POV => self.algorithms.get("POV"),
            AlgorithmType::SNIPER => self.algorithms.get("Sniper"),
            AlgorithmType::DCA => self.algorithms.get("TWAP"), // Use TWAP for DCA
        }.ok_or_else(|| anyhow::anyhow!("Algorithm not found"))?;
        
        let state = ExecutionState {
            params: params.clone(),
            progress: ExecutionProgress {
                executed_quantity: 0.0,
                remaining_quantity: params.total_quantity,
                average_price: 0.0,
                slices_executed: 0,
                start_time: Instant::now(),
                elapsed: Duration::ZERO,
                estimated_completion: params.duration,
            },
            algorithm: algorithm.clone(),
            symbol,
            exchange,
            side,
        };
        
        self.active_executions.insert(execution_id, state);
        
        Ok(())
    }
    
    /// Process active executions
    pub async fn process_executions(&self, market_data: &DashMap<Symbol, MarketContext>) -> Result<()> {
        for mut entry in self.active_executions.iter_mut() {
            let execution_id = entry.key().clone();
            let state = entry.value_mut();
            
            // Update progress
            state.progress.elapsed = state.progress.start_time.elapsed();
            
            // Get market context
            let context = market_data
                .get(&state.symbol)
                .map(|c| c.clone())
                .unwrap_or_else(|| MarketContext {
                    current_price: 0.0,
                    bid: 0.0,
                    ask: 0.0,
                    spread: 0.0,
                    volume: 0.0,
                    volatility: 0.0,
                    trend: 0.0,
                });
            
            // Check if should execute
            if !state.algorithm.should_execute(&state.params, &state.progress, &context) {
                continue;
            }
            
            // Calculate next slice
            if let Some(mut slice) = state.algorithm.calculate_next_slice(
                &state.params,
                &state.progress,
                &context
            ) {
                // Adjust for market conditions
                state.algorithm.adjust_for_market(&mut slice, &context);
                
                // Create and send order
                let order = match slice.order_type {
                    OrderType::Market => Order::market(
                        state.symbol.clone(),
                        state.exchange,
                        state.side,
                        slice.quantity
                    ),
                    OrderType::Limit => Order::limit(
                        state.symbol.clone(),
                        state.exchange,
                        state.side,
                        slice.quantity,
                        slice.price.unwrap_or(context.current_price)
                    ),
                    _ => continue,
                };
                
                self.order_sender.send(order)?;
                
                // Update progress
                state.progress.slices_executed += 1;
            }
        }
        
        Ok(())
    }
    
    /// Update execution progress with fill
    pub fn update_execution(&self, execution_id: &str, fill_quantity: f64, fill_price: f64) {
        if let Some(mut state) = self.active_executions.get_mut(execution_id) {
            let prev_executed = state.progress.executed_quantity;
            state.progress.executed_quantity += fill_quantity;
            state.progress.remaining_quantity -= fill_quantity;
            
            // Update average price
            if prev_executed > 0.0 {
                state.progress.average_price = 
                    (state.progress.average_price * prev_executed + fill_price * fill_quantity) /
                    state.progress.executed_quantity;
            } else {
                state.progress.average_price = fill_price;
            }
        }
    }
    
    /// Get execution progress
    pub fn get_progress(&self, execution_id: &str) -> Option<ExecutionProgress> {
        self.active_executions
            .get(execution_id)
            .map(|s| s.progress.clone())
    }
    
    /// Cancel execution
    pub fn cancel_execution(&self, execution_id: &str) -> bool {
        self.active_executions.remove(execution_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_twap_algorithm() {
        let algo = TWAPAlgorithm::new(Duration::from_secs(300), 10);
        let params = ExecutionParams::default();
        let progress = ExecutionProgress {
            executed_quantity: 0.0,
            remaining_quantity: 1.0,
            average_price: 0.0,
            slices_executed: 0,
            start_time: Instant::now(),
            elapsed: Duration::ZERO,
            estimated_completion: Duration::from_secs(300),
        };
        let market = MarketContext {
            current_price: 100.0,
            bid: 99.9,
            ask: 100.1,
            spread: 0.2,
            volume: 1000.0,
            volatility: 0.01,
            trend: 0.0,
        };
        
        let slice = algo.calculate_next_slice(&params, &progress, &market);
        assert!(slice.is_some());
    }
}