//! # ARES Neuromorphic Core
//!
//! High-performance neuromorphic prediction engine for trading systems.
//!
//! This crate provides the core neuromorphic computing capabilities for the ARES
//! trading system, including spiking neural networks, reservoir computing, and
//! signal processing algorithms optimized for financial market prediction.
//!
//! ## Features
//!
//! - **Spike Encoding**: Convert market data to spike trains using various encoding schemes
//! - **Reservoir Computing**: Liquid State Machines for temporal pattern recognition
//! - **Signal Processing**: Advanced neuromorphic signal processing algorithms
//! - **Pattern Detection**: Real-time pattern recognition in market data
//! - **High Performance**: Optimized for low-latency financial applications
//!
//! ## Quick Start
//!
//! ```rust
//! use ares_neuromorphic_core::{NeuromorphicEngine, EngineConfig, MarketData};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create engine configuration
//! let config = EngineConfig {
//!     spike_neurons: 10000,
//!     reservoir_size: 5000,
//!     encoding_window_ms: 1000,
//!     ..Default::default()
//! };
//!
//! // Initialize the neuromorphic engine
//! let mut engine = NeuromorphicEngine::new(config)?;
//!
//! // Process market data
//! let market_data = MarketData {
//!     price: 50000.0,
//!     volume: 1.5,
//!     timestamp: chrono::Utc::now(),
//!     symbol: "BTC-USD".to_string(),
//! };
//!
//! let prediction = engine.process(market_data).await?;
//! println!("Prediction confidence: {:.2}", prediction.confidence);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The neuromorphic engine follows a multi-stage processing pipeline:
//!
//! 1. **Input Encoding**: Market data → Spike trains
//! 2. **Spike Processing**: Neuromorphic computation on spike patterns
//! 3. **Reservoir Dynamics**: Temporal pattern extraction
//! 4. **Pattern Recognition**: Classification and prediction
//! 5. **Signal Generation**: Trading signal output
//!

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core modules
pub mod core;
pub mod spike;
pub mod reservoir;
pub mod signals;
pub mod types;

// Re-exports for easy access
pub use core::{NeuromorphicEngine, EngineConfig, ProcessingError};
pub use types::{MarketData, Prediction, Pattern, Spike, SpikePattern};
pub use signals::{TradingSignal, SignalStrength, PatternType};

// Feature-gated re-exports
#[cfg(feature = "async")]
pub use core::AsyncNeuromorphicEngine;


/// Main result type for the neuromorphic engine
pub type Result<T> = std::result::Result<T, ProcessingError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Engine capabilities and metadata
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Engine version
    pub version: String,
    /// Supported features
    pub features: Vec<String>,
    /// Performance characteristics
    pub performance: PerformanceInfo,
}

/// Performance characteristics of the engine
#[derive(Debug, Clone)]
pub struct PerformanceInfo {
    /// Maximum processing rate (samples/second)
    pub max_sample_rate: f64,
    /// Typical latency (microseconds)
    pub typical_latency_us: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
}

impl Default for EngineInfo {
    fn default() -> Self {
        Self {
            version: VERSION.to_string(),
            features: vec![
                "spike-encoding".to_string(),
                "reservoir-computing".to_string(),
                "pattern-recognition".to_string(),
                #[cfg(feature = "async")]
                "async-processing".to_string(),
                #[cfg(feature = "brian2")]
                "brian2-integration".to_string(),
                #[cfg(feature = "cuda")]
                "cuda-acceleration".to_string(),
            ],
            performance: PerformanceInfo {
                max_sample_rate: 1_000_000.0, // 1M samples/second
                typical_latency_us: 10.0,     // 10 microseconds
                memory_usage_bytes: 100 * 1024 * 1024, // 100MB
            },
        }
    }
}

/// Get engine information and capabilities
pub fn engine_info() -> EngineInfo {
    EngineInfo::default()
}

/// Initialize the neuromorphic engine with default configuration
pub fn init() -> Result<NeuromorphicEngine> {
    NeuromorphicEngine::new(EngineConfig::default())
}

/// Initialize the neuromorphic engine with custom configuration
pub fn init_with_config(config: EngineConfig) -> Result<NeuromorphicEngine> {
    NeuromorphicEngine::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_info() {
        let info = engine_info();
        assert_eq!(info.version, VERSION);
        assert!(!info.features.is_empty());
        assert!(info.performance.max_sample_rate > 0.0);
    }

    #[test]
    fn test_init_default() {
        let engine = init();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_init_with_config() {
        let config = EngineConfig {
            spike_neurons: 1000,
            reservoir_size: 500,
            ..Default::default()
        };
        let engine = init_with_config(config);
        assert!(engine.is_ok());
    }
}