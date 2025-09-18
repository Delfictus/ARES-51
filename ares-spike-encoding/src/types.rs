//! Core types for spike encoding

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::Timelike;

/// Individual spike in a neural network
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Spike {
    /// Neuron ID that fired
    pub neuron_id: usize,
    /// Time of the spike (milliseconds)
    pub time_ms: f64,
    /// Spike amplitude (optional, normalized 0.0-1.0)
    pub amplitude: Option<f32>,
    /// Neuron type that generated this spike
    pub neuron_type: NeuronType,
}

impl Spike {
    /// Create a new spike
    pub fn new(neuron_id: usize, time_ms: f64) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: None,
            neuron_type: NeuronType::Excitatory,
        }
    }

    /// Create a spike with amplitude
    pub fn with_amplitude(neuron_id: usize, time_ms: f64, amplitude: f32) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: Some(amplitude.clamp(0.0, 1.0)),
            neuron_type: NeuronType::Excitatory,
        }
    }

    /// Create a spike with specific neuron type
    pub fn with_type(neuron_id: usize, time_ms: f64, neuron_type: NeuronType) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: None,
            neuron_type,
        }
    }

    /// Create a spike with all properties
    pub fn new_full(
        neuron_id: usize,
        time_ms: f64,
        amplitude: Option<f32>,
        neuron_type: NeuronType,
    ) -> Self {
        Self {
            neuron_id,
            time_ms,
            amplitude: amplitude.map(|a| a.clamp(0.0, 1.0)),
            neuron_type,
        }
    }

    /// Get effective amplitude (1.0 if none specified)
    pub fn effective_amplitude(&self) -> f32 {
        self.amplitude.unwrap_or(1.0)
    }

    /// Check if this is an inhibitory spike
    pub fn is_inhibitory(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Inhibitory)
    }

    /// Check if this is an excitatory spike
    pub fn is_excitatory(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Excitatory)
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
    /// Source encoding method
    pub encoding_method: Option<EncodingMethod>,
}

impl SpikePattern {
    /// Create a new spike pattern
    pub fn new(spikes: Vec<Spike>, duration_ms: f64) -> Self {
        Self {
            spikes,
            duration_ms,
            metadata: PatternMetadata::default(),
            encoding_method: None,
        }
    }

    /// Create spike pattern with encoding method
    pub fn with_method(
        spikes: Vec<Spike>,
        duration_ms: f64,
        method: EncodingMethod,
    ) -> Self {
        Self {
            spikes,
            duration_ms,
            metadata: PatternMetadata::default(),
            encoding_method: Some(method),
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

    /// Get spikes from specific neurons
    pub fn spikes_from_neurons(&self, neuron_ids: &[usize]) -> Vec<&Spike> {
        self.spikes
            .iter()
            .filter(|spike| neuron_ids.contains(&spike.neuron_id))
            .collect()
    }

    /// Get spikes by neuron type
    pub fn spikes_by_type(&self, neuron_type: NeuronType) -> Vec<&Spike> {
        self.spikes
            .iter()
            .filter(|spike| spike.neuron_type == neuron_type)
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

    /// Calculate average inter-spike interval
    pub fn average_isi(&self) -> Option<f64> {
        if self.spikes.len() < 2 {
            return None;
        }

        let mut times: Vec<f64> = self.spikes.iter().map(|s| s.time_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total_interval: f64 = times.windows(2)
            .map(|window| window[1] - window[0])
            .sum();

        Some(total_interval / (times.len() - 1) as f64)
    }

    /// Get unique neuron IDs that fired
    pub fn active_neurons(&self) -> Vec<usize> {
        let mut neurons: Vec<usize> = self.spikes
            .iter()
            .map(|spike| spike.neuron_id)
            .collect();
        neurons.sort_unstable();
        neurons.dedup();
        neurons
    }

    /// Calculate coefficient of variation (measure of regularity)
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        let isi_values: Vec<f64> = self.inter_spike_intervals();
        if isi_values.len() < 2 {
            return None;
        }

        let mean = isi_values.iter().sum::<f64>() / isi_values.len() as f64;
        let variance = isi_values.iter()
            .map(|&isi| (isi - mean).powi(2))
            .sum::<f64>() / isi_values.len() as f64;
        let std_dev = variance.sqrt();

        if mean != 0.0 {
            Some(std_dev / mean)
        } else {
            None
        }
    }

    /// Get all inter-spike intervals
    pub fn inter_spike_intervals(&self) -> Vec<f64> {
        if self.spikes.len() < 2 {
            return Vec::new();
        }

        let mut times: Vec<f64> = self.spikes.iter().map(|s| s.time_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        times.windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }

    /// Calculate firing rate for each neuron
    pub fn neuron_firing_rates(&self) -> HashMap<usize, f64> {
        let mut rates = HashMap::new();
        let duration_sec = self.duration_ms / 1000.0;

        if duration_sec == 0.0 {
            return rates;
        }

        for spike in &self.spikes {
            *rates.entry(spike.neuron_id).or_insert(0.0) += 1.0 / duration_sec;
        }

        rates
    }

    /// Add metadata to the pattern
    pub fn with_metadata(mut self, metadata: PatternMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a custom metadata field
    pub fn add_metadata(mut self, key: String, value: f64) -> Self {
        self.metadata.custom.insert(key, value);
        self
    }
}

/// Metadata associated with a spike pattern
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// Pattern strength/confidence
    pub strength: f32,
    /// Pattern classification
    pub pattern_type: Option<String>,
    /// Source data identifier
    pub source_symbol: Option<String>,
    /// Encoding parameters used
    pub encoding_params: HashMap<String, f64>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Additional custom metadata
    pub custom: HashMap<String, f64>,
}

/// Quality metrics for spike patterns
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio
    pub snr: Option<f32>,
    /// Information content (bits)
    pub information_content: Option<f32>,
    /// Temporal coherence measure
    pub temporal_coherence: Option<f32>,
    /// Spatial distribution measure
    pub spatial_distribution: Option<f32>,
}

/// Market data input for spike encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    /// Asset symbol (e.g., "BTC-USD")
    pub symbol: String,
    /// Asset price
    pub price: f64,
    /// Trading volume
    pub volume: f64,
    /// Timestamp of the data point
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional market metrics
    pub metrics: Option<HashMap<String, f64>>,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: &str, price: f64, volume: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            price,
            volume,
            timestamp: chrono::Utc::now(),
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

    /// Check if data has high-frequency components
    pub fn has_high_frequency_components(&self) -> bool {
        // Simple heuristic: high volume suggests high frequency
        self.volume > 10.0
    }

    /// Check if data has multiple features
    pub fn has_multiple_features(&self) -> bool {
        self.metrics.as_ref().map_or(false, |m| m.len() > 2)
    }

    /// Check if data has periodic patterns
    pub fn has_periodic_patterns(&self) -> bool {
        // Simple heuristic: look for RSI or other oscillator indicators
        self.get_metric("rsi").is_some() || 
        self.get_metric("macd").is_some() ||
        self.get_metric("oscillator").is_some()
    }

    /// Extract all numeric features as a vector
    pub fn to_feature_vector(&self) -> FeatureVector {
        let mut features = FeatureVector::new();
        
        // Add basic features
        features.add("price", self.price);
        features.add("volume", self.volume);
        
        // Add timestamp features
        let hour = self.timestamp.hour() as f64 / 24.0;
        features.add("hour", hour);
        
        // Add custom metrics
        if let Some(metrics) = &self.metrics {
            for (key, value) in metrics {
                features.add(key, *value);
            }
        }
        
        features
    }
}

/// Feature vector for encoding
#[derive(Debug, Clone)]
pub struct FeatureVector {
    features: HashMap<String, f64>,
}

impl FeatureVector {
    /// Create new feature vector
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    /// Add a feature
    pub fn add(&mut self, name: &str, value: f64) {
        self.features.insert(name.to_string(), value);
    }

    /// Get feature value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.features.get(name).copied()
    }

    /// Get number of features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Iterate over features
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.features.iter()
    }

    /// Get all values
    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.features.values()
    }

    /// Get feature names
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.features.keys()
    }

    /// Convert to vector in consistent order
    pub fn to_vec(&self) -> Vec<f64> {
        let mut pairs: Vec<_> = self.features.iter().collect();
        pairs.sort_by_key(|(name, _)| *name);
        pairs.into_iter().map(|(_, value)| *value).collect()
    }

    /// Normalize all features to [0, 1] range
    pub fn normalize(&mut self) {
        let values: Vec<f64> = self.features.values().cloned().collect();
        if values.is_empty() {
            return;
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range != 0.0 {
            for value in self.features.values_mut() {
                *value = (*value - min_val) / range;
            }
        }
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Spike encoding methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncodingMethod {
    /// Rate coding (spike frequency represents value)
    Rate,
    /// Temporal coding (spike timing represents value)
    Temporal,
    /// Population coding (multiple neurons represent value)
    Population,
    /// Phase coding (spike phase represents value)
    Phase,
    /// Latency coding (first-spike time represents value)
    Latency,
    /// Burst coding (spike bursts represent values)
    Burst,
}

impl EncodingMethod {
    /// Get method name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            EncodingMethod::Rate => "rate",
            EncodingMethod::Temporal => "temporal",
            EncodingMethod::Population => "population",
            EncodingMethod::Phase => "phase",
            EncodingMethod::Latency => "latency",
            EncodingMethod::Burst => "burst",
        }
    }

    /// Get all available methods
    pub fn all() -> Vec<EncodingMethod> {
        vec![
            EncodingMethod::Rate,
            EncodingMethod::Temporal,
            EncodingMethod::Population,
            EncodingMethod::Phase,
            EncodingMethod::Latency,
            EncodingMethod::Burst,
        ]
    }
}

/// Neuron types in the encoding network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuronType {
    /// Excitatory neuron (positive contribution)
    Excitatory,
    /// Inhibitory neuron (negative contribution)
    Inhibitory,
    /// Input neuron (receives external stimuli)
    Input,
    /// Output neuron (produces final signals)
    Output,
}

impl NeuronType {
    /// Get neuron type name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            NeuronType::Excitatory => "excitatory",
            NeuronType::Inhibitory => "inhibitory",
            NeuronType::Input => "input",
            NeuronType::Output => "output",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_creation() {
        let spike = Spike::new(42, 123.45);
        assert_eq!(spike.neuron_id, 42);
        assert_eq!(spike.time_ms, 123.45);
        assert_eq!(spike.amplitude, None);
        assert_eq!(spike.neuron_type, NeuronType::Excitatory);
    }

    #[test]
    fn test_spike_with_amplitude() {
        let spike = Spike::with_amplitude(0, 100.0, 0.75);
        assert_eq!(spike.amplitude, Some(0.75));
        assert_eq!(spike.effective_amplitude(), 0.75);
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
        assert_eq!(pattern.duration_ms, 100.0);
        assert_eq!(pattern.spike_rate(), 30.0); // 3 spikes in 0.1 seconds
    }

    #[test]
    fn test_spike_pattern_time_window() {
        let spikes = vec![
            Spike::new(0, 5.0),
            Spike::new(1, 15.0),
            Spike::new(2, 25.0),
            Spike::new(3, 35.0),
        ];
        let pattern = SpikePattern::new(spikes, 100.0);
        
        let window_spikes = pattern.spikes_in_window(10.0, 30.0);
        assert_eq!(window_spikes.len(), 2); // Spikes at 15.0 and 25.0
    }

    #[test]
    fn test_market_data() {
        let data = MarketData::new("BTC-USD", 50000.0, 1.5)
            .with_metric("rsi".to_string(), 65.0)
            .with_metric("volume_ma".to_string(), 2.1);
        
        assert_eq!(data.symbol, "BTC-USD");
        assert_eq!(data.price, 50000.0);
        assert_eq!(data.volume, 1.5);
        assert_eq!(data.get_metric("rsi"), Some(65.0));
        assert!(data.has_periodic_patterns());
    }

    #[test]
    fn test_feature_vector() {
        let mut features = FeatureVector::new();
        features.add("price", 100.0);
        features.add("volume", 50.0);
        features.add("rsi", 75.0);
        
        assert_eq!(features.len(), 3);
        assert_eq!(features.get("price"), Some(100.0));
        
        let values = features.to_vec();
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn test_feature_vector_normalization() {
        let mut features = FeatureVector::new();
        features.add("a", 10.0);
        features.add("b", 20.0);
        features.add("c", 30.0);
        
        features.normalize();
        
        assert_eq!(features.get("a"), Some(0.0));
        assert_eq!(features.get("b"), Some(0.5));
        assert_eq!(features.get("c"), Some(1.0));
    }

    #[test]
    fn test_encoding_method_string() {
        assert_eq!(EncodingMethod::Rate.as_str(), "rate");
        assert_eq!(EncodingMethod::Temporal.as_str(), "temporal");
        assert_eq!(EncodingMethod::Population.as_str(), "population");
        assert_eq!(EncodingMethod::Phase.as_str(), "phase");
    }

    #[test]
    fn test_neuron_type_string() {
        assert_eq!(NeuronType::Excitatory.as_str(), "excitatory");
        assert_eq!(NeuronType::Inhibitory.as_str(), "inhibitory");
        assert_eq!(NeuronType::Input.as_str(), "input");
        assert_eq!(NeuronType::Output.as_str(), "output");
    }

    #[test]
    fn test_spike_pattern_statistics() {
        let spikes = vec![
            Spike::new(0, 10.0),
            Spike::new(1, 20.0),
            Spike::new(0, 30.0), // Same neuron fires again
            Spike::new(2, 40.0),
        ];
        let pattern = SpikePattern::new(spikes, 100.0);
        
        let active_neurons = pattern.active_neurons();
        assert_eq!(active_neurons, vec![0, 1, 2]);
        
        let isi = pattern.inter_spike_intervals();
        assert_eq!(isi, vec![10.0, 10.0, 10.0]);
        
        let avg_isi = pattern.average_isi();
        assert_eq!(avg_isi, Some(10.0));
        
        let rates = pattern.neuron_firing_rates();
        assert_eq!(rates.get(&0), Some(&20.0)); // Neuron 0 fired twice in 0.1s
        assert_eq!(rates.get(&1), Some(&10.0)); // Neuron 1 fired once
    }
}