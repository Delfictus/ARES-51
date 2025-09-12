//! Neuromorphic Trading System
//! 
//! Hybrid neuromorphic-deterministic trading system combining:
//! - Spiking neural networks for pattern detection
//! - Reservoir computing for temporal analysis  
//! - Deterministic algorithms for fast execution
//! - Signal fusion for optimal decision making

#![allow(dead_code)] // During development

// Core modules
pub mod spike_encoding;
pub mod spike_encoding_enhanced;
pub mod spike_encoding_simd;
pub mod brian2_bridge;
pub mod event_bus;
pub mod event_bus_enhanced;
pub mod reservoir;
pub mod reservoir_stdp;
pub mod reservoir_parallel;
pub mod pattern_training;
pub mod signal_fusion;
pub mod market_state;
pub mod time_source;
pub mod time_source_calibrated;
pub mod memory_pool;
pub mod memory_pool_numa;

// Exchange connectivity
pub mod exchanges;
pub mod market_data;
pub mod trading;

// Paper trading and execution
pub mod paper_trading;
pub mod execution;

// Full system integration
pub mod integration;

// Revolutionary neuromorphic components with DRPP/ADP
pub mod neuromorphic;
pub mod drpp;
pub mod adp;
pub mod test_drpp_integration;
pub mod test_adp_quantum_quality;
pub mod drpp_adp_bridge;
pub mod test_drpp_adp_integration;
pub mod test_drpp_resonance_analyzer;
pub mod transfer_entropy;
pub mod multi_timeframe;
pub mod test_pattern_performance;
pub mod test_phase_2b_integration;
pub mod coupling_adaptation;
pub mod pattern_routing;
pub mod cross_module_optimizer;
pub mod realtime_adaptation;
pub mod test_phase_2c_integration;

// Re-exports
pub use spike_encoding::{SpikeEncoder, Spike, NeuronType};
pub use event_bus::{MarketDataBus, TradeData, QuoteData, OrderBookData};
pub use reservoir::{LiquidStateMachine, PatternType};
pub use signal_fusion::{SignalFusion, TradingSignal, TradeAction};

#[cfg(feature = "brian2_available")]
pub use brian2_bridge::{Brian2Bridge, NetworkConfig};

use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;

/// Main neuromorphic trading system
pub struct NeuromorphicTradingSystem {
    // Fast path (deterministic)
    deterministic_engine: DeterministicEngine,
    
    // Smart path (neuromorphic)
    spike_encoder: SpikeEncoder,
    #[cfg(feature = "brian2_available")]
    brian2_network: Option<Brian2Bridge>,
    reservoir: LiquidStateMachine,
    
    // Signal fusion
    signal_fusion: SignalFusion,
    
    // Infrastructure
    event_bus: Arc<MarketDataBus>,
    memory_pool: Arc<memory_pool::MemoryPool>,
    
    // Metrics
    metrics: SystemMetrics,
}

impl NeuromorphicTradingSystem {
    /// Create new neuromorphic trading system
    pub fn new(config: SystemConfig) -> Result<Self> {
        // Initialize memory pool
        let memory_pool = Arc::new(memory_pool::MemoryPool::new(
            memory_pool::PoolConfig::default()
        )?);
        
        // Initialize event bus
        let event_bus = Arc::new(MarketDataBus::new(
            event_bus::BusConfig::default()
        ));
        
        // Initialize spike encoder
        let spike_encoder = SpikeEncoder::new(
            spike_encoding::EncoderConfig {
                num_neurons: config.num_neurons,
                encoding_schemes: vec![
                    spike_encoding::EncodingScheme::RateCoding,
                    spike_encoding::EncodingScheme::TemporalCoding,
                    spike_encoding::EncodingScheme::PopulationCoding,
                ],
                window_size_ms: 1000,
            }
        );
        
        // Initialize Brian2 if available
        #[cfg(feature = "brian2_available")]
        let brian2_network = if config.enable_brian2 {
            Some(Brian2Bridge::initialize()?)
        } else {
            None
        };
        
        // Initialize reservoir
        let reservoir = LiquidStateMachine::new(
            reservoir::ReservoirConfig {
                size: config.reservoir_size,
                spectral_radius: 0.95,
                connection_probability: 0.2,
                leak_rate: 0.1,
            }
        );
        
        // Initialize deterministic engine
        let deterministic_engine = DeterministicEngine::new();
        
        // Initialize signal fusion
        let signal_fusion = SignalFusion::new();
        
        Ok(Self {
            deterministic_engine,
            spike_encoder,
            #[cfg(feature = "brian2_available")]
            brian2_network,
            reservoir,
            signal_fusion,
            event_bus,
            memory_pool,
            metrics: SystemMetrics::default(),
        })
    }
    
    /// Process market data and generate trading signals
    pub async fn process_market_data(&mut self, data: MarketData) -> Result<TradingSignal> {
        // Update metrics
        self.metrics.events_processed += 1;
        let start = std::time::Instant::now();
        
        // Parallel processing of both paths
        let (fast_signal, smart_signal) = tokio::join!(
            self.process_deterministic(&data),
            self.process_neuromorphic(&data)
        );
        
        // Fuse signals
        let signal = self.signal_fusion.fuse(
            fast_signal.unwrap_or_else(|_| DeterministicSignal {
                action: TradeAction::Hold,
                confidence: 0.0,
                latency_ns: 0,
                strategy: "error".to_string(),
            }),
            smart_signal.unwrap_or_else(|_| NeuromorphicSignal {
                patterns: vec![],
                confidence: 0.0,
                novelty_score: 0.0,
            }),
            &self.get_market_state()
        );
        
        // Update metrics
        self.metrics.total_latency_ns += start.elapsed().as_nanos() as u64;
        
        Ok(signal)
    }
    
    /// Process through deterministic path
    async fn process_deterministic(&self, data: &MarketData) -> Result<DeterministicSignal> {
        self.deterministic_engine.process(data).await
    }
    
    /// Process through neuromorphic path
    async fn process_neuromorphic(&mut self, data: &MarketData) -> Result<NeuromorphicSignal> {
        // Encode to spikes
        let spikes = self.spike_encoder.encode(data);
        
        // Process through Brian2 if available
        #[cfg(feature = "brian2_available")]
        let brian_output = if let Some(ref mut brian2) = self.brian2_network {
            brian2.process(spikes.clone()).await?
        } else {
            spikes.clone()
        };
        
        #[cfg(not(feature = "brian2_available"))]
        let brian_output = spikes.clone();
        
        // Process through reservoir
        let reservoir_state = self.reservoir.process(&brian_output);
        
        // Detect patterns
        let patterns = self.reservoir.detect_patterns();
        
        Ok(NeuromorphicSignal {
            patterns,
            confidence: reservoir_state.confidence,
            novelty_score: reservoir_state.novelty,
        })
    }
    
    fn get_market_state(&self) -> market_state::MarketState {
        // Return current market state from tracking
        // In production, this would be continuously updated
        let mut state = market_state::MarketState::default();
        
        // Update with latest metrics if available
        if let Some(last_trade) = self.get_last_trade() {
            state.update_with_trade(&last_trade);
        }
        
        state
    }
    
    fn get_last_trade(&self) -> Option<TradeData> {
        // In production, this would retrieve from event bus
        None
    }
}

/// System configuration
pub struct SystemConfig {
    pub num_neurons: usize,
    pub reservoir_size: usize,
    pub enable_brian2: bool,
    pub enable_cuda: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            num_neurons: 10_000,
            reservoir_size: 5_000,
            enable_brian2: cfg!(feature = "brian2_available"),
            enable_cuda: cfg!(feature = "cuda_available"),
        }
    }
}

/// Market data wrapper
pub struct MarketData {
    pub trade: Option<TradeData>,
    pub quote: Option<QuoteData>,
    pub order_book: Option<OrderBookData>,
    pub timestamp_ns: u64,
}

/// Deterministic engine for fast path
struct DeterministicEngine {
    // Strategies would go here
}

impl DeterministicEngine {
    fn new() -> Self {
        Self {}
    }
    
    async fn process(&self, _data: &MarketData) -> Result<DeterministicSignal> {
        // Fast deterministic processing
        Ok(DeterministicSignal {
            action: TradeAction::Hold,
            confidence: 0.5,
            latency_ns: 1000,
            strategy: "momentum".to_string(),
        })
    }
}

/// Deterministic signal
pub struct DeterministicSignal {
    pub action: TradeAction,
    pub confidence: f32,
    pub latency_ns: u64,
    pub strategy: String,
}

/// Neuromorphic signal
pub struct NeuromorphicSignal {
    pub patterns: Vec<(PatternType, f32)>,
    pub confidence: f32,
    pub novelty_score: f32,
}

// Market state is now in market_state module
pub use market_state::MarketState;

/// System metrics
#[derive(Default)]
struct SystemMetrics {
    events_processed: u64,
    total_latency_ns: u64,
    patterns_detected: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_creation() {
        let system = NeuromorphicTradingSystem::new(SystemConfig::default());
        assert!(system.is_ok());
    }
}