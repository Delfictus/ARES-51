//! Core types for the neuromorphic engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market data input for the neuromorphic engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Asset price
    pub price: f64,
    /// Trading volume
    pub volume: f64,
    /// Timestamp of the data point
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Symbol identifier (e.g., "BTC-USD")
    pub symbol: String,
    /// Additional market metrics
    pub metrics: Option<HashMap<String, f64>>,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: String, price: f64, volume: f64) -> Self {
        Self {
            price,
            volume,
            timestamp: chrono::Utc::now(),
            symbol,
            metrics: None,
        }
    }

    /// Add a custom metric
    pub fn with_metric(mut self, key: String, value: f64) -> Self {
        self.metrics.get_or_insert_with(HashMap::new).insert(key, value);
        self
    }

    /// Get a metric value
    pub fn get_metric(&self, key: &str) -> Option<f64> {
        self.metrics.as_ref()?.get(key).copied()
    }
}

/// Individual spike in a neural network
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Spike {
    /// Neuron ID that fired
    pub neuron_id: usize,
    /// Time of the spike (milliseconds)
    pub time_ms: f64,
    /// Spike amplitude (optional)
    pub amplitude: Option<f32>,
}

impl Spike {
    /// Create a new spike
    pub fn new(neuron_id: usize, time_ms: f64) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: None,
        }
    }

    /// Create a spike with amplitude
    pub fn with_amplitude(neuron_id: usize, time_ms: f64, amplitude: f32) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: Some(amplitude),
        }
    }
}

/// Collection of spikes forming a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePattern {
    /// Individual spikes in the pattern
    pub spikes: Vec<Spike>,
    /// Duration of the pattern (milliseconds)
    pub duration_ms: f64,
    /// Pattern metadata
    pub metadata: PatternMetadata,
}

impl SpikePattern {
    /// Create a new spike pattern
    pub fn new(spikes: Vec<Spike>, duration_ms: f64) -> Self {
        Self {
            spikes,
            duration_ms,
            metadata: PatternMetadata::default(),
        }
    }

    /// Get the number of spikes in the pattern
    pub fn spike_count(&self) -> usize {
        self.spikes.len()
    }

    /// Get spikes within a time window
    pub fn spikes_in_window(&self, start_ms: f64, end_ms: f64) -> Vec<&Spike> {
        self.spikes
            .iter()
            .filter(|spike| spike.time_ms >= start_ms && spike.time_ms <= end_ms)
            .collect()
    }

    /// Calculate spike rate (spikes per second)
    pub fn spike_rate(&self) -> f64 {
        if self.duration_ms == 0.0 {
            0.0
        } else {
            (self.spikes.len() as f64) / (self.duration_ms / 1000.0)
        }
    }
}

/// Metadata associated with a spike pattern
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Pattern strength/confidence
    pub strength: f32,
    /// Pattern classification
    pub pattern_type: Option<String>,
    /// Source market data
    pub source_symbol: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, f64>,
}

/// Recognized pattern types in market data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pattern {
    /// Trend patterns
    UpTrend,
    DownTrend,
    Sideways,
    
    /// Reversal patterns
    Reversal,
    Breakout,
    
    /// Volatility patterns
    HighVolatility,
    LowVolatility,
    
    /// Custom pattern with name
    Custom(String),
}

impl Pattern {
    /// Get pattern as string
    pub fn as_str(&self) -> &str {
        match self {
            Pattern::UpTrend => "uptrend",
            Pattern::DownTrend => "downtrend",
            Pattern::Sideways => "sideways",
            Pattern::Reversal => "reversal",
            Pattern::Breakout => "breakout",
            Pattern::HighVolatility => "high_volatility",
            Pattern::LowVolatility => "low_volatility",
            Pattern::Custom(name) => name,
        }
    }
}

/// Prediction output from the neuromorphic engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted direction/action
    pub direction: PredictionDirection,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Predicted price change magnitude
    pub magnitude: Option<f32>,
    /// Time horizon for the prediction
    pub time_horizon_ms: f64,
    /// Detected patterns that led to this prediction
    pub patterns: Vec<Pattern>,
    /// Prediction metadata
    pub metadata: PredictionMetadata,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(
        direction: PredictionDirection,
        confidence: f32,
        time_horizon_ms: f64,
    ) -> Self {
        Self {
            direction,
            confidence: confidence.clamp(0.0, 1.0),
            magnitude: None,
            time_horizon_ms,
            patterns: Vec::new(),
            metadata: PredictionMetadata::default(),
        }
    }

    /// Add a detected pattern
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Set magnitude
    pub fn with_magnitude(mut self, magnitude: f32) -> Self {
        self.magnitude = Some(magnitude);
        self
    }

    /// Check if prediction is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(self.direction, PredictionDirection::Up)
    }

    /// Check if prediction is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(self.direction, PredictionDirection::Down)
    }

    /// Check if prediction suggests holding
    pub fn is_neutral(&self) -> bool {
        matches!(self.direction, PredictionDirection::Hold)
    }
}

/// Direction of a prediction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionDirection {
    /// Upward price movement expected
    Up,
    /// Downward price movement expected
    Down,
    /// No significant movement expected
    Hold,
}

/// Metadata for predictions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Model version used
    pub model_version: Option<String>,
    /// Processing latency in microseconds
    pub latency_us: Option<f64>,
    /// Number of spikes processed
    pub spike_count: Option<usize>,
    /// Reservoir state information
    pub reservoir_state: Option<HashMap<String, f64>>,
}

/// Neuron types in the neuromorphic network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    /// Input neuron (receives external stimuli)
    Input,
    /// Hidden/reservoir neuron
    Hidden,
    /// Output neuron (produces final signals)
    Output,
    /// Inhibitory neuron
    Inhibitory,
}

/// Configuration for spike encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Encoding method to use
    pub method: EncodingMethod,
    /// Time window for encoding (milliseconds)
    pub window_ms: f64,
    /// Number of neurons to use for encoding
    pub neuron_count: usize,
    /// Encoding parameters
    pub parameters: HashMap<String, f64>,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            method: EncodingMethod::Rate,
            window_ms: 1000.0,
            neuron_count: 1000,
            parameters: HashMap::new(),
        }
    }
}

/// Spike encoding methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingMethod {
    /// Rate coding (spike frequency represents value)
    Rate,
    /// Temporal coding (spike timing represents value)
    Temporal,
    /// Population coding (multiple neurons represent value)
    Population,
    /// Phase coding (spike phase represents value)
    Phase,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_creation() {
        let data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5)
            .with_metric("volatility".to_string(), 0.02);
        
        assert_eq!(data.symbol, "BTC-USD");
        assert_eq!(data.price, 50000.0);
        assert_eq!(data.volume, 1.5);
        assert_eq!(data.get_metric("volatility"), Some(0.02));
    }

    #[test]
    fn test_spike_pattern() {
        let spikes = vec![
            Spike::new(0, 10.0),
            Spike::new(1, 20.0),
            Spike::new(2, 30.0),
        ];
        let pattern = SpikePattern::new(spikes, 100.0);
        
        assert_eq!(pattern.spike_count(), 3);
        assert_eq!(pattern.spike_rate(), 30.0); // 3 spikes in 0.1 seconds
    }

    #[test]
    fn test_prediction() {
        let prediction = Prediction::new(
            PredictionDirection::Up,
            0.85,
            5000.0,
        )
        .with_pattern(Pattern::UpTrend)
        .with_magnitude(0.05);
        
        assert!(prediction.is_bullish());
        assert!(!prediction.is_bearish());
        assert_eq!(prediction.confidence, 0.85);
        assert_eq!(prediction.magnitude, Some(0.05));
        assert_eq!(prediction.patterns.len(), 1);
    }

    #[test]
    fn test_pattern_as_str() {
        assert_eq!(Pattern::UpTrend.as_str(), "uptrend");
        assert_eq!(Pattern::Custom("my_pattern".to_string()).as_str(), "my_pattern");
    }
}