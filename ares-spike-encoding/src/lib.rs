//! # ARES Spike Encoding
//!
//! High-performance neuromorphic spike encoding algorithms for converting market data 
//! and other time-series data into neural spike trains.
//!
//! This crate provides various spike encoding methods used in neuromorphic computing
//! and spiking neural networks, optimized for financial market data processing.
//!
//! ## Features
//!
//! - **Multiple Encoding Methods**: Rate, temporal, population, and phase coding
//! - **High Performance**: Optimized algorithms with SIMD support
//! - **Flexible Input**: Support for various data types and structures  
//! - **Configurable**: Extensive parameter customization
//! - **Real-time**: Designed for low-latency streaming applications
//!
//! ## Quick Start
//!
//! ```rust
//! use ares_spike_encoding::{SpikeEncoder, EncodingMethod, MarketData};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a rate-based spike encoder
//! let mut encoder = SpikeEncoder::new(1000, 1000.0)?
//!     .with_method(EncodingMethod::Rate);
//!
//! // Create market data
//! let market_data = MarketData::new("BTC-USD", 50000.0, 1.5);
//!
//! // Encode to spike pattern
//! let spike_pattern = encoder.encode(&market_data)?;
//!
//! println!("Generated {} spikes", spike_pattern.spike_count());
//! # Ok(())
//! # }
//! ```
//!
//! ## Encoding Methods
//!
//! ### Rate Coding
//! Encodes values as spike frequencies - higher values produce more spikes per unit time.
//!
//! ### Temporal Coding  
//! Encodes values as spike timing - earlier spikes represent higher values.
//!
//! ### Population Coding
//! Uses multiple neurons with overlapping tuning curves to represent values.
//!
//! ### Phase Coding
//! Encodes values as the phase relationship between spike trains.
//!
//! ## Architecture
//!
//! ```text
//! Input Data → Feature Extraction → Encoding Algorithm → Spike Pattern
//! ```
//!
//! The encoding process consists of:
//! 1. **Feature Extraction**: Convert input data to normalized feature vectors
//! 2. **Method Selection**: Choose appropriate encoding algorithm
//! 3. **Spike Generation**: Generate spike trains based on features
//! 4. **Pattern Assembly**: Combine spikes into structured patterns

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core modules
pub mod encoding;
pub mod types;
pub mod presets;

// Re-exports for easy access
pub use encoding::{SpikeEncoder, EncodingConfig, EncodingError};
pub use types::{
    Spike, SpikePattern, PatternMetadata, MarketData, 
    EncodingMethod, NeuronType, FeatureVector
};
pub use presets::{EncodingPresets, FinancialPresets};

// Feature-gated re-exports
#[cfg(feature = "ndarray-support")]
pub use encoding::NDArrayEncoder;


/// Main result type for spike encoding operations
pub type Result<T> = std::result::Result<T, EncodingError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Encoder capabilities and metadata
#[derive(Debug, Clone)]
pub struct EncoderInfo {
    /// Encoder version
    pub version: String,
    /// Supported encoding methods
    pub methods: Vec<EncodingMethod>,
    /// Supported features
    pub features: Vec<String>,
    /// Performance characteristics
    pub performance: PerformanceInfo,
}

/// Performance characteristics of encoders
#[derive(Debug, Clone)]
pub struct PerformanceInfo {
    /// Maximum encoding rate (samples/second)
    pub max_sample_rate: f64,
    /// Typical latency (microseconds)
    pub typical_latency_us: f64,
    /// Memory usage per encoder (bytes)
    pub memory_usage_bytes: usize,
    /// Supported neuron count range
    pub neuron_range: (usize, usize),
}

impl Default for EncoderInfo {
    fn default() -> Self {
        Self {
            version: VERSION.to_string(),
            methods: vec![
                EncodingMethod::Rate,
                EncodingMethod::Temporal, 
                EncodingMethod::Population,
                EncodingMethod::Phase,
            ],
            features: vec![
                "rate-coding".to_string(),
                "temporal-coding".to_string(),
                "population-coding".to_string(),
                "phase-coding".to_string(),
                #[cfg(feature = "ndarray-support")]
                "ndarray-support".to_string(),
                #[cfg(feature = "simd")]
                "simd-acceleration".to_string(),
                #[cfg(feature = "brian2")]
                "brian2-integration".to_string(),
            ],
            performance: PerformanceInfo {
                max_sample_rate: 10_000_000.0, // 10M samples/second
                typical_latency_us: 1.0,       // 1 microsecond
                memory_usage_bytes: 1024 * 1024, // 1MB per encoder
                neuron_range: (1, 1_000_000),   // 1 to 1M neurons
            },
        }
    }
}

/// Get encoder information and capabilities
pub fn encoder_info() -> EncoderInfo {
    EncoderInfo::default()
}

/// Create a rate-based spike encoder with default settings
pub fn rate_encoder(neuron_count: usize, window_ms: f64) -> Result<SpikeEncoder> {
    SpikeEncoder::new(neuron_count, window_ms)?
        .with_method(EncodingMethod::Rate)
}

/// Create a temporal spike encoder with default settings
pub fn temporal_encoder(neuron_count: usize, window_ms: f64) -> Result<SpikeEncoder> {
    SpikeEncoder::new(neuron_count, window_ms)?
        .with_method(EncodingMethod::Temporal)
}

/// Create a population spike encoder with default settings
pub fn population_encoder(neuron_count: usize, window_ms: f64) -> Result<SpikeEncoder> {
    SpikeEncoder::new(neuron_count, window_ms)?
        .with_method(EncodingMethod::Population)
}

/// Create a phase spike encoder with default settings
pub fn phase_encoder(neuron_count: usize, window_ms: f64) -> Result<SpikeEncoder> {
    SpikeEncoder::new(neuron_count, window_ms)?
        .with_method(EncodingMethod::Phase)
}

/// Create an encoder optimized for financial market data
pub fn financial_encoder(_neuron_count: usize, _window_ms: f64) -> Result<SpikeEncoder> {
    let config = FinancialPresets::default_config();
    SpikeEncoder::with_config(config)
}

/// Batch encode multiple data points efficiently
pub fn batch_encode<T>(
    encoder: &mut SpikeEncoder,
    data_points: &[T],
) -> Result<Vec<SpikePattern>>
where
    T: Into<MarketData> + Clone,
{
    let mut patterns = Vec::with_capacity(data_points.len());
    
    for data_point in data_points {
        let market_data = data_point.clone().into();
        let pattern = encoder.encode(&market_data)?;
        patterns.push(pattern);
    }
    
    Ok(patterns)
}

/// Encode data with automatic method selection based on characteristics
pub fn adaptive_encode(
    data: &MarketData,
    neuron_count: usize,
    window_ms: f64,
) -> Result<SpikePattern> {
    // Analyze data characteristics to choose best encoding method
    let method = if data.has_high_frequency_components() {
        EncodingMethod::Temporal
    } else if data.has_multiple_features() {
        EncodingMethod::Population  
    } else if data.has_periodic_patterns() {
        EncodingMethod::Phase
    } else {
        EncodingMethod::Rate // Default fallback
    };
    
    let mut encoder = SpikeEncoder::new(neuron_count, window_ms)?
        .with_method(method)?;
    
    encoder.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_info() {
        let info = encoder_info();
        assert_eq!(info.version, VERSION);
        assert!(!info.methods.is_empty());
        assert!(!info.features.is_empty());
        assert!(info.performance.max_sample_rate > 0.0);
    }

    #[test]
    fn test_rate_encoder_creation() {
        let encoder = rate_encoder(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_temporal_encoder_creation() {
        let encoder = temporal_encoder(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_population_encoder_creation() {
        let encoder = population_encoder(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_phase_encoder_creation() {
        let encoder = phase_encoder(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_financial_encoder_creation() {
        let encoder = financial_encoder(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_batch_encoding() {
        let mut encoder = rate_encoder(100, 1000.0).unwrap();
        let data_points = vec![
            MarketData::new("BTC-USD", 50000.0, 1.0),
            MarketData::new("ETH-USD", 3000.0, 2.0),
        ];
        
        let patterns = batch_encode(&mut encoder, &data_points);
        assert!(patterns.is_ok());
        assert_eq!(patterns.unwrap().len(), 2);
    }

    #[test]
    fn test_adaptive_encoding() {
        let data = MarketData::new("BTC-USD", 50000.0, 1.5);
        let pattern = adaptive_encode(&data, 100, 1000.0);
        assert!(pattern.is_ok());
    }
}