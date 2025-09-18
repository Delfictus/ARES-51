//! Spike encoding for converting market data to neural spikes

use crate::types::{MarketData, Spike, SpikePattern, PatternMetadata, EncodingMethod};
use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use chrono::{Timelike, Datelike};

/// Spike encoder for converting market data to spike trains
#[derive(Debug)]
pub struct SpikeEncoder {
    neuron_count: usize,
    window_ms: f64,
    encoding_method: EncodingMethod,
    parameters: EncodingParameters,
    rng: rand::rngs::ThreadRng,
}

/// Parameters for different encoding methods
#[derive(Debug, Clone)]
struct EncodingParameters {
    // Rate coding parameters
    max_rate: f64,        // Maximum spike rate (Hz)
    min_rate: f64,        // Minimum spike rate (Hz)
    
    // Temporal coding parameters
    delay_range_ms: f64,  // Time delay range for temporal coding
    
    // Population coding parameters
    neurons_per_feature: usize, // Neurons per feature dimension
    
    // Phase coding parameters
    base_frequency: f64,  // Base oscillation frequency (Hz)
    phase_range: f64,     // Phase range in radians
}

impl Default for EncodingParameters {
    fn default() -> Self {
        Self {
            max_rate: 100.0,      // 100 Hz max
            min_rate: 1.0,        // 1 Hz min
            delay_range_ms: 50.0, // 50ms delay range
            neurons_per_feature: 10,
            base_frequency: 40.0, // 40 Hz gamma-like frequency
            phase_range: 2.0 * std::f64::consts::PI,
        }
    }
}

impl SpikeEncoder {
    /// Create a new spike encoder
    pub fn new(neuron_count: usize, window_ms: f64) -> Result<Self> {
        if neuron_count == 0 {
            return Err(anyhow::anyhow!("Neuron count must be greater than 0"));
        }
        if window_ms <= 0.0 {
            return Err(anyhow::anyhow!("Window duration must be positive"));
        }
        
        Ok(Self {
            neuron_count,
            window_ms,
            encoding_method: EncodingMethod::Rate,
            parameters: EncodingParameters::default(),
            rng: rand::thread_rng(),
        })
    }
    
    /// Set encoding method
    pub fn with_encoding_method(mut self, method: EncodingMethod) -> Self {
        self.encoding_method = method;
        self
    }
    
    /// Set encoding parameters
    pub fn with_parameters(mut self, params: EncodingParameters) -> Self {
        self.parameters = params;
        self
    }
    
    /// Encode market data to spike pattern
    pub fn encode(&mut self, market_data: &MarketData) -> Result<SpikePattern> {
        // Extract features from market data
        let features = self.extract_features(market_data);
        
        // Generate spikes based on encoding method
        let spikes = match self.encoding_method {
            EncodingMethod::Rate => self.rate_encode(&features)?,
            EncodingMethod::Temporal => self.temporal_encode(&features)?,
            EncodingMethod::Population => self.population_encode(&features)?,
            EncodingMethod::Phase => self.phase_encode(&features)?,
        };
        
        // Create spike pattern with metadata
        let mut pattern = SpikePattern::new(spikes, self.window_ms);
        pattern.metadata = PatternMetadata {
            strength: self.calculate_pattern_strength(&features),
            pattern_type: Some(format!("{:?}", self.encoding_method)),
            source_symbol: Some(market_data.symbol.clone()),
            custom: self.create_custom_metadata(&features),
        };
        
        Ok(pattern)
    }
    
    /// Extract numerical features from market data
    fn extract_features(&self, market_data: &MarketData) -> FeatureVector {
        let mut features = FeatureVector::new();
        
        // Normalize price (simple log transform)
        let normalized_price = (market_data.price / 1000.0).ln();
        features.add("price", normalized_price);
        
        // Normalize volume (log transform)
        let normalized_volume = (market_data.volume + 1.0).ln();
        features.add("volume", normalized_volume);
        
        // Add time-based features
        let hour = market_data.timestamp.hour() as f64 / 24.0;
        features.add("hour", hour);
        
        let day_of_week = market_data.timestamp.weekday().num_days_from_sunday() as f64 / 7.0;
        features.add("day_of_week", day_of_week);
        
        // Add custom metrics if available
        if let Some(metrics) = &market_data.metrics {
            for (key, value) in metrics {
                features.add(&format!("metric_{}", key), self.normalize_value(*value));
            }
        }
        
        features
    }
    
    /// Normalize a value to [0, 1] range using tanh
    fn normalize_value(&self, value: f64) -> f64 {
        (value.tanh() + 1.0) / 2.0
    }
    
    /// Rate coding: encode values as spike rates
    fn rate_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.neuron_count / features.len().max(1);
        
        for (i, (_, value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let spike_rate = self.value_to_rate(*value);
            
            // Generate spikes for this feature
            let feature_spikes = self.generate_poisson_spikes(
                neuron_start,
                neurons_per_feature.min(self.neuron_count - neuron_start),
                spike_rate,
            );
            
            spikes.extend(feature_spikes);
        }
        
        Ok(spikes)
    }
    
    /// Temporal coding: encode values as spike timing
    fn temporal_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.neuron_count / features.len().max(1);
        
        for (i, (_, value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            
            // Map value to delay time
            let delay_ms = value * self.parameters.delay_range_ms;
            
            // Generate spikes at specific times
            for j in 0..neurons_per_feature.min(self.neuron_count - neuron_start) {
                let neuron_id = neuron_start + j;
                let spike_time = delay_ms + (j as f64) * 2.0; // Slight offset per neuron
                
                if spike_time < self.window_ms {
                    spikes.push(Spike::new(neuron_id, spike_time));
                }
            }
        }
        
        Ok(spikes)
    }
    
    /// Population coding: use multiple neurons per feature
    fn population_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let total_groups = features.len();
        let neurons_per_group = self.neuron_count / total_groups.max(1);
        
        for (i, (_, value)) in features.iter().enumerate() {
            let group_start = i * neurons_per_group;
            
            // Create Gaussian-like activation pattern
            let peak_neuron = (*value * neurons_per_group as f64) as usize;
            let sigma = neurons_per_group as f64 / 6.0; // Standard deviation
            
            for j in 0..neurons_per_group.min(self.neuron_count - group_start) {
                let neuron_id = group_start + j;
                
                // Calculate activation based on distance from peak
                let distance = (j as f64 - peak_neuron as f64).abs();
                let activation = (-0.5 * (distance / sigma).powi(2)).exp();
                
                // Convert activation to spike probability
                let spike_prob = activation * 0.8; // Max 80% spike probability
                
                if self.rng.gen::<f64>() < spike_prob {
                    let spike_time = self.rng.gen::<f64>() * self.window_ms;
                    spikes.push(Spike::new(neuron_id, spike_time));
                }
            }
        }
        
        Ok(spikes)
    }
    
    /// Phase coding: encode values as spike phases
    fn phase_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.neuron_count / features.len().max(1);
        let period_ms = 1000.0 / self.parameters.base_frequency;
        
        for (i, (_, value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            
            // Map value to phase
            let phase = value * self.parameters.phase_range;
            
            // Generate spikes at phase-shifted times
            let mut current_time = (phase / (2.0 * std::f64::consts::PI)) * period_ms;
            
            while current_time < self.window_ms {
                for j in 0..neurons_per_feature.min(self.neuron_count - neuron_start) {
                    let neuron_id = neuron_start + j;
                    let jitter = self.rng.gen::<f64>() * 2.0 - 1.0; // ±1ms jitter
                    let spike_time = current_time + jitter;
                    
                    if spike_time >= 0.0 && spike_time < self.window_ms {
                        spikes.push(Spike::new(neuron_id, spike_time));
                    }
                }
                current_time += period_ms;
            }
        }
        
        Ok(spikes)
    }
    
    /// Convert normalized value to spike rate
    fn value_to_rate(&self, value: f64) -> f64 {
        self.parameters.min_rate + value * (self.parameters.max_rate - self.parameters.min_rate)
    }
    
    /// Generate Poisson-distributed spikes
    fn generate_poisson_spikes(
        &mut self,
        neuron_start: usize,
        neuron_count: usize,
        rate_hz: f64,
    ) -> Vec<Spike> {
        let mut spikes = Vec::new();
        let window_sec = self.window_ms / 1000.0;
        let _expected_spikes = rate_hz * window_sec;
        
        for i in 0..neuron_count {
            let neuron_id = neuron_start + i;
            
            // Poisson process: inter-spike intervals are exponentially distributed
            let mut current_time = 0.0;
            
            while current_time < self.window_ms {
                // Generate next inter-spike interval
                let lambda = rate_hz / 1000.0; // Convert to per-ms
                let interval = -lambda.recip() * self.rng.gen::<f64>().ln();
                current_time += interval;
                
                if current_time < self.window_ms {
                    spikes.push(Spike::new(neuron_id, current_time));
                }
            }
        }
        
        spikes
    }
    
    /// Calculate pattern strength based on features
    fn calculate_pattern_strength(&self, features: &FeatureVector) -> f32 {
        if features.is_empty() {
            return 0.0;
        }
        
        // Calculate variance as a measure of pattern strength
        let mean: f64 = features.values().sum::<f64>() / features.len() as f64;
        let variance: f64 = features.values()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / features.len() as f64;
        
        // Normalize variance to [0, 1] range
        (variance.sqrt() * 2.0).min(1.0) as f32
    }
    
    /// Create custom metadata from features
    fn create_custom_metadata(&self, features: &FeatureVector) -> HashMap<String, f64> {
        let mut metadata = HashMap::new();
        
        // Add feature statistics
        if !features.is_empty() {
            let values: Vec<f64> = features.values().cloned().collect();
            metadata.insert("feature_mean".to_string(), 
                           values.iter().sum::<f64>() / values.len() as f64);
            metadata.insert("feature_min".to_string(), 
                           values.iter().cloned().fold(f64::INFINITY, f64::min));
            metadata.insert("feature_max".to_string(), 
                           values.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        }
        
        metadata.insert("encoding_window_ms".to_string(), self.window_ms);
        metadata.insert("neuron_count".to_string(), self.neuron_count as f64);
        
        metadata
    }
}

/// Container for extracted features
#[derive(Debug, Clone)]
struct FeatureVector {
    features: HashMap<String, f64>,
}

impl FeatureVector {
    fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }
    
    fn add(&mut self, name: &str, value: f64) {
        self.features.insert(name.to_string(), value);
    }
    
    fn len(&self) -> usize {
        self.features.len()
    }
    
    fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
    
    fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.features.iter()
    }
    
    fn values(&self) -> impl Iterator<Item = &f64> {
        self.features.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_encoder_creation() {
        let encoder = SpikeEncoder::new(1000, 1000.0);
        assert!(encoder.is_ok());
        
        let invalid_encoder = SpikeEncoder::new(0, 1000.0);
        assert!(invalid_encoder.is_err());
    }

    #[test]
    fn test_feature_extraction() {
        let mut encoder = SpikeEncoder::new(1000, 1000.0).unwrap();
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        
        let features = encoder.extract_features(&market_data);
        assert!(!features.is_empty());
        assert!(features.features.contains_key("price"));
        assert!(features.features.contains_key("volume"));
    }

    #[test]
    fn test_rate_encoding() {
        let mut encoder = SpikeEncoder::new(100, 1000.0).unwrap();
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        
        let pattern = encoder.encode(&market_data);
        assert!(pattern.is_ok());
        
        let pattern = pattern.unwrap();
        assert!(pattern.spike_count() > 0);
        assert_eq!(pattern.duration_ms, 1000.0);
    }

    #[test]
    fn test_different_encoding_methods() {
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        
        for method in [
            EncodingMethod::Rate,
            EncodingMethod::Temporal,
            EncodingMethod::Population,
            EncodingMethod::Phase,
        ] {
            let mut encoder = SpikeEncoder::new(100, 1000.0)
                .unwrap()
                .with_encoding_method(method);
            
            let pattern = encoder.encode(&market_data);
            assert!(pattern.is_ok(), "Failed for method {:?}", method);
        }
    }

    #[test]
    fn test_pattern_metadata() {
        let mut encoder = SpikeEncoder::new(100, 1000.0).unwrap();
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        
        let pattern = encoder.encode(&market_data).unwrap();
        assert!(pattern.metadata.strength >= 0.0 && pattern.metadata.strength <= 1.0);
        assert_eq!(pattern.metadata.source_symbol, Some("BTC-USD".to_string()));
        assert!(!pattern.metadata.custom.is_empty());
    }
}