//! Spike encoding algorithms and implementations

use crate::types::{
    MarketData, Spike, SpikePattern, PatternMetadata, FeatureVector, 
    EncodingMethod
};
use std::collections::HashMap;
use rand::Rng;

/// Main spike encoder
#[derive(Debug)]
pub struct SpikeEncoder {
    config: EncodingConfig,
    rng: rand::rngs::ThreadRng,
    cache: EncodingCache,
    statistics: EncodingStatistics,
}

/// Configuration for spike encoding
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    /// Number of neurons for encoding
    pub neuron_count: usize,
    /// Time window for encoding (milliseconds)
    pub window_ms: f64,
    /// Encoding method to use
    pub method: EncodingMethod,
    /// Rate coding parameters
    pub rate_params: RateParams,
    /// Temporal coding parameters  
    pub temporal_params: TemporalParams,
    /// Population coding parameters
    pub population_params: PopulationParams,
    /// Phase coding parameters
    pub phase_params: PhaseParams,
    /// Noise parameters
    pub noise_params: NoiseParams,
    /// Quality control settings
    pub quality_control: QualityControl,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            neuron_count: 1000,
            window_ms: 1000.0,
            method: EncodingMethod::Rate,
            rate_params: RateParams::default(),
            temporal_params: TemporalParams::default(),
            population_params: PopulationParams::default(),
            phase_params: PhaseParams::default(),
            noise_params: NoiseParams::default(),
            quality_control: QualityControl::default(),
        }
    }
}

/// Rate coding parameters
#[derive(Debug, Clone)]
pub struct RateParams {
    /// Maximum spike rate (Hz)
    pub max_rate: f64,
    /// Minimum spike rate (Hz)
    pub min_rate: f64,
    /// Rate scaling function
    pub scaling: RateScaling,
}

impl Default for RateParams {
    fn default() -> Self {
        Self {
            max_rate: 100.0,
            min_rate: 1.0,
            scaling: RateScaling::Linear,
        }
    }
}

/// Rate scaling functions
#[derive(Debug, Clone, Copy)]
pub enum RateScaling {
    /// Linear scaling
    Linear,
    /// Logarithmic scaling
    Logarithmic,
    /// Exponential scaling
    Exponential,
    /// Sigmoid scaling
    Sigmoid,
}

/// Temporal coding parameters
#[derive(Debug, Clone)]
pub struct TemporalParams {
    /// Maximum delay (milliseconds)
    pub max_delay_ms: f64,
    /// Minimum delay (milliseconds) 
    pub min_delay_ms: f64,
    /// Temporal resolution (milliseconds)
    pub resolution_ms: f64,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            max_delay_ms: 50.0,
            min_delay_ms: 0.0,
            resolution_ms: 0.1,
        }
    }
}

/// Population coding parameters
#[derive(Debug, Clone)]
pub struct PopulationParams {
    /// Number of neurons per feature
    pub neurons_per_feature: usize,
    /// Tuning curve width (standard deviation)
    pub tuning_width: f64,
    /// Overlap between tuning curves
    pub overlap: f64,
}

impl Default for PopulationParams {
    fn default() -> Self {
        Self {
            neurons_per_feature: 10,
            tuning_width: 0.2,
            overlap: 0.5,
        }
    }
}

/// Phase coding parameters
#[derive(Debug, Clone)]
pub struct PhaseParams {
    /// Base oscillation frequency (Hz)
    pub base_frequency: f64,
    /// Phase range (radians)
    pub phase_range: f64,
    /// Number of cycles per window
    pub cycles_per_window: f64,
}

impl Default for PhaseParams {
    fn default() -> Self {
        Self {
            base_frequency: 40.0, // Gamma frequency
            phase_range: 2.0 * std::f64::consts::PI,
            cycles_per_window: 1.0,
        }
    }
}

/// Noise parameters
#[derive(Debug, Clone)]
pub struct NoiseParams {
    /// Noise level (0.0 = no noise, 1.0 = maximum noise)
    pub level: f64,
    /// Noise type
    pub noise_type: NoiseType,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            level: 0.01,
            noise_type: NoiseType::Gaussian,
            seed: None,
        }
    }
}

/// Types of noise
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    /// Gaussian (normal) noise
    Gaussian,
    /// Uniform noise
    Uniform,
    /// Poisson noise
    Poisson,
}

/// Quality control settings
#[derive(Debug, Clone)]
pub struct QualityControl {
    /// Enable spike validation
    pub validate_spikes: bool,
    /// Maximum spikes per pattern
    pub max_spikes: Option<usize>,
    /// Minimum information content
    pub min_information: Option<f32>,
}

impl Default for QualityControl {
    fn default() -> Self {
        Self {
            validate_spikes: true,
            max_spikes: Some(10000),
            min_information: None,
        }
    }
}

/// Encoding cache for performance
#[derive(Debug, Default)]
struct EncodingCache {
    feature_cache: HashMap<String, FeatureVector>,
    pattern_cache: HashMap<u64, SpikePattern>,
}

/// Encoding statistics
#[derive(Debug, Default)]
pub struct EncodingStatistics {
    patterns_encoded: u64,
    total_spikes: u64,
    cache_hits: u64,
    cache_misses: u64,
    average_encoding_time_us: f64,
}

/// Encoding errors
#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Input validation error
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Encoding algorithm error
    #[error("Encoding failed: {0}")]
    EncodingFailed(String),
    
    /// Quality control error
    #[error("Quality control failed: {0}")]
    QualityFailed(String),
    
    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
}

impl SpikeEncoder {
    /// Create a new spike encoder
    pub fn new(neuron_count: usize, window_ms: f64) -> Result<Self, EncodingError> {
        if neuron_count == 0 {
            return Err(EncodingError::InvalidConfig(
                "Neuron count must be greater than 0".to_string()
            ));
        }
        if window_ms <= 0.0 {
            return Err(EncodingError::InvalidConfig(
                "Window duration must be positive".to_string()
            ));
        }

        let config = EncodingConfig {
            neuron_count,
            window_ms,
            ..Default::default()
        };

        Ok(Self {
            config,
            rng: rand::thread_rng(),
            cache: EncodingCache::default(),
            statistics: EncodingStatistics::default(),
        })
    }

    /// Create encoder with custom configuration
    pub fn with_config(config: EncodingConfig) -> Result<Self, EncodingError> {
        Self::validate_config(&config)?;

        Ok(Self {
            config,
            rng: rand::thread_rng(),
            cache: EncodingCache::default(),
            statistics: EncodingStatistics::default(),
        })
    }

    /// Set encoding method
    pub fn with_method(mut self, method: EncodingMethod) -> Result<Self, EncodingError> {
        self.config.method = method;
        Ok(self)
    }

    /// Set rate parameters
    pub fn with_rate_params(mut self, params: RateParams) -> Self {
        self.config.rate_params = params;
        self
    }

    /// Set temporal parameters
    pub fn with_temporal_params(mut self, params: TemporalParams) -> Self {
        self.config.temporal_params = params;
        self
    }

    /// Set population parameters
    pub fn with_population_params(mut self, params: PopulationParams) -> Self {
        self.config.population_params = params;
        self
    }

    /// Set phase parameters
    pub fn with_phase_params(mut self, params: PhaseParams) -> Self {
        self.config.phase_params = params;
        self
    }

    /// Encode market data to spike pattern
    pub fn encode(&mut self, market_data: &MarketData) -> Result<SpikePattern, EncodingError> {
        let start_time = std::time::Instant::now();

        // Validate input
        self.validate_input(market_data)?;

        // Extract features
        let features = self.extract_features(market_data)?;

        // Check cache
        let cache_key = self.calculate_cache_key(&features);
        if let Some(cached_pattern) = self.cache.pattern_cache.get(&cache_key) {
            self.statistics.cache_hits += 1;
            return Ok(cached_pattern.clone());
        }
        self.statistics.cache_misses += 1;

        // Generate spikes based on method
        let spikes = match self.config.method {
            EncodingMethod::Rate => self.rate_encode(&features)?,
            EncodingMethod::Temporal => self.temporal_encode(&features)?,
            EncodingMethod::Population => self.population_encode(&features)?,
            EncodingMethod::Phase => self.phase_encode(&features)?,
            EncodingMethod::Latency => self.latency_encode(&features)?,
            EncodingMethod::Burst => self.burst_encode(&features)?,
        };

        // Create pattern with metadata
        let mut pattern = SpikePattern::with_method(
            spikes,
            self.config.window_ms,
            self.config.method,
        );

        // Add metadata
        pattern.metadata = self.create_metadata(&features, market_data);

        // Quality control
        self.validate_pattern(&pattern)?;

        // Update statistics
        let encoding_time = start_time.elapsed().as_micros() as f64;
        self.update_statistics(&pattern, encoding_time);

        // Cache result
        self.cache.pattern_cache.insert(cache_key, pattern.clone());

        Ok(pattern)
    }

    /// Validate configuration
    fn validate_config(config: &EncodingConfig) -> Result<(), EncodingError> {
        if config.neuron_count == 0 {
            return Err(EncodingError::InvalidConfig(
                "Neuron count must be greater than 0".to_string()
            ));
        }

        if config.window_ms <= 0.0 {
            return Err(EncodingError::InvalidConfig(
                "Window duration must be positive".to_string()
            ));
        }

        if config.rate_params.max_rate <= config.rate_params.min_rate {
            return Err(EncodingError::InvalidConfig(
                "Max rate must be greater than min rate".to_string()
            ));
        }

        if config.temporal_params.max_delay_ms <= config.temporal_params.min_delay_ms {
            return Err(EncodingError::InvalidConfig(
                "Max delay must be greater than min delay".to_string()
            ));
        }

        if config.population_params.neurons_per_feature == 0 {
            return Err(EncodingError::InvalidConfig(
                "Neurons per feature must be greater than 0".to_string()
            ));
        }

        Ok(())
    }

    /// Validate input data
    fn validate_input(&self, market_data: &MarketData) -> Result<(), EncodingError> {
        if market_data.price <= 0.0 {
            return Err(EncodingError::InvalidInput(
                "Price must be positive".to_string()
            ));
        }

        if market_data.volume < 0.0 {
            return Err(EncodingError::InvalidInput(
                "Volume cannot be negative".to_string()
            ));
        }

        Ok(())
    }

    /// Extract features from market data
    fn extract_features(&mut self, market_data: &MarketData) -> Result<FeatureVector, EncodingError> {
        // Check feature cache
        let cache_key = format!("{}_{:.2}_{:.2}", 
                               market_data.symbol, market_data.price, market_data.volume);
        
        if let Some(cached_features) = self.cache.feature_cache.get(&cache_key) {
            return Ok(cached_features.clone());
        }

        let mut features = market_data.to_feature_vector();
        
        // Add derived features
        self.add_derived_features(&mut features, market_data);
        
        // Normalize features
        features.normalize();
        
        // Cache features
        self.cache.feature_cache.insert(cache_key, features.clone());
        
        Ok(features)
    }

    /// Add derived features
    fn add_derived_features(&self, features: &mut FeatureVector, market_data: &MarketData) {
        // Price momentum (simplified)
        if let Some(prev_price) = features.get("prev_price") {
            let momentum = (market_data.price - prev_price) / prev_price;
            features.add("momentum", momentum);
        }

        // Volume-weighted price
        let vwap = market_data.price * market_data.volume;
        features.add("vwap", vwap);

        // Log transforms for better normalization
        features.add("log_price", market_data.price.ln());
        features.add("log_volume", (market_data.volume + 1.0).ln());
    }

    /// Rate encoding implementation
    fn rate_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.neuron_count / features.len().max(1);

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let neuron_count = neurons_per_feature.min(self.config.neuron_count - neuron_start);
            
            // Convert value to spike rate
            let spike_rate = self.value_to_rate(value);
            
            // Generate Poisson spikes
            let feature_spikes = self.generate_poisson_spikes(
                neuron_start,
                neuron_count,
                spike_rate,
            )?;
            
            spikes.extend(feature_spikes);
        }

        Ok(spikes)
    }

    /// Temporal encoding implementation
    fn temporal_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.neuron_count / features.len().max(1);

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let neuron_count = neurons_per_feature.min(self.config.neuron_count - neuron_start);
            
            // Map value to delay time
            let delay_ms = self.value_to_delay(value);
            
            // Generate spikes at specific times
            for j in 0..neuron_count {
                let neuron_id = neuron_start + j;
                let jitter = self.rng.gen::<f64>() * self.config.temporal_params.resolution_ms;
                let spike_time = delay_ms + jitter;
                
                if spike_time >= 0.0 && spike_time < self.config.window_ms {
                    spikes.push(Spike::new(neuron_id, spike_time));
                }
            }
        }

        Ok(spikes)
    }

    /// Population encoding implementation
    fn population_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.population_params.neurons_per_feature;
        let total_feature_neurons = features.len() * neurons_per_feature;

        if total_feature_neurons > self.config.neuron_count {
            return Err(EncodingError::EncodingFailed(
                "Not enough neurons for population encoding".to_string()
            ));
        }

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            
            // Create Gaussian tuning curves
            for j in 0..neurons_per_feature {
                let neuron_id = neuron_start + j;
                let preferred_value = j as f64 / (neurons_per_feature - 1) as f64;
                
                // Calculate neuron activation
                let distance = (value - preferred_value).abs();
                let activation = (-0.5 * (distance / self.config.population_params.tuning_width).powi(2)).exp();
                
                // Convert activation to spike probability
                let spike_prob = activation * 0.8; // Max 80% probability
                
                if self.rng.gen::<f64>() < spike_prob {
                    let spike_time = self.rng.gen::<f64>() * self.config.window_ms;
                    spikes.push(Spike::new(neuron_id, spike_time));
                }
            }
        }

        Ok(spikes)
    }

    /// Phase encoding implementation
    fn phase_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.neuron_count / features.len().max(1);
        let period_ms = 1000.0 / self.config.phase_params.base_frequency;

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let neuron_count = neurons_per_feature.min(self.config.neuron_count - neuron_start);
            
            // Map value to phase
            let phase = value * self.config.phase_params.phase_range;
            
            // Generate phase-locked spikes
            let mut current_time = (phase / (2.0 * std::f64::consts::PI)) * period_ms;
            
            while current_time < self.config.window_ms {
                for j in 0..neuron_count {
                    let neuron_id = neuron_start + j;
                    let jitter = (self.rng.gen::<f64>() - 0.5) * 2.0; // ±2ms jitter
                    let spike_time = current_time + jitter;
                    
                    if spike_time >= 0.0 && spike_time < self.config.window_ms {
                        spikes.push(Spike::new(neuron_id, spike_time));
                    }
                }
                current_time += period_ms;
            }
        }

        Ok(spikes)
    }

    /// Latency encoding implementation
    fn latency_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.neuron_count / features.len().max(1);

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let neuron_count = neurons_per_feature.min(self.config.neuron_count - neuron_start);
            
            // First spike time encodes the value (earlier = higher value)
            let latency = (1.0 - value) * self.config.temporal_params.max_delay_ms;
            
            // Only one spike per neuron at the calculated latency
            for j in 0..neuron_count {
                let neuron_id = neuron_start + j;
                let spike_time = latency + j as f64 * 0.1; // Small offset per neuron
                
                if spike_time < self.config.window_ms {
                    spikes.push(Spike::new(neuron_id, spike_time));
                }
            }
        }

        Ok(spikes)
    }

    /// Burst encoding implementation
    fn burst_encode(&mut self, features: &FeatureVector) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let neurons_per_feature = self.config.neuron_count / features.len().max(1);

        for (i, (_, &value)) in features.iter().enumerate() {
            let neuron_start = i * neurons_per_feature;
            let neuron_count = neurons_per_feature.min(self.config.neuron_count - neuron_start);
            
            // Burst length encodes the value
            let burst_length = (value * 10.0) as usize + 1; // 1-11 spikes
            let burst_interval = 2.0; // 2ms between spikes in burst
            
            for j in 0..neuron_count {
                let neuron_id = neuron_start + j;
                let burst_start = self.rng.gen::<f64>() * (self.config.window_ms - burst_length as f64 * burst_interval);
                
                for k in 0..burst_length {
                    let spike_time = burst_start + k as f64 * burst_interval;
                    if spike_time < self.config.window_ms {
                        spikes.push(Spike::new(neuron_id, spike_time));
                    }
                }
            }
        }

        Ok(spikes)
    }

    /// Convert normalized value to spike rate
    fn value_to_rate(&self, value: f64) -> f64 {
        let normalized = value.clamp(0.0, 1.0);
        
        match self.config.rate_params.scaling {
            RateScaling::Linear => {
                self.config.rate_params.min_rate + 
                normalized * (self.config.rate_params.max_rate - self.config.rate_params.min_rate)
            }
            RateScaling::Logarithmic => {
                let log_min = self.config.rate_params.min_rate.ln();
                let log_max = self.config.rate_params.max_rate.ln();
                (log_min + normalized * (log_max - log_min)).exp()
            }
            RateScaling::Exponential => {
                self.config.rate_params.min_rate * 
                (self.config.rate_params.max_rate / self.config.rate_params.min_rate).powf(normalized)
            }
            RateScaling::Sigmoid => {
                let x = (normalized - 0.5) * 10.0; // Scale to steeper sigmoid
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                self.config.rate_params.min_rate + 
                sigmoid * (self.config.rate_params.max_rate - self.config.rate_params.min_rate)
            }
        }
    }

    /// Convert normalized value to delay time
    fn value_to_delay(&self, value: f64) -> f64 {
        let normalized = value.clamp(0.0, 1.0);
        self.config.temporal_params.min_delay_ms + 
        (1.0 - normalized) * (self.config.temporal_params.max_delay_ms - self.config.temporal_params.min_delay_ms)
    }

    /// Generate Poisson-distributed spikes
    fn generate_poisson_spikes(
        &mut self,
        neuron_start: usize,
        neuron_count: usize,
        rate_hz: f64,
    ) -> Result<Vec<Spike>, EncodingError> {
        let mut spikes = Vec::new();
        let _window_sec = self.config.window_ms / 1000.0;

        for i in 0..neuron_count {
            let neuron_id = neuron_start + i;
            let mut current_time = 0.0;

            while current_time < self.config.window_ms {
                // Generate next inter-spike interval (exponentially distributed)
                let lambda = rate_hz / 1000.0; // Convert to per-ms
                if lambda <= 0.0 {
                    break;
                }
                
                let interval = -lambda.recip() * self.rng.gen::<f64>().ln();
                current_time += interval;

                if current_time < self.config.window_ms {
                    spikes.push(Spike::new(neuron_id, current_time));
                }
            }
        }

        Ok(spikes)
    }

    /// Create pattern metadata
    fn create_metadata(&self, features: &FeatureVector, market_data: &MarketData) -> PatternMetadata {
        let mut metadata = PatternMetadata {
            source_symbol: Some(market_data.symbol.clone()),
            pattern_type: Some(self.config.method.as_str().to_string()),
            ..Default::default()
        };

        // Add encoding parameters
        metadata.encoding_params.insert("neuron_count".to_string(), self.config.neuron_count as f64);
        metadata.encoding_params.insert("window_ms".to_string(), self.config.window_ms);
        metadata.encoding_params.insert("feature_count".to_string(), features.len() as f64);

        // Calculate pattern strength
        if !features.is_empty() {
            let values: Vec<f64> = features.values().cloned().collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            metadata.strength = variance.sqrt().min(1.0) as f32;
        }

        metadata
    }

    /// Validate generated pattern
    fn validate_pattern(&self, pattern: &SpikePattern) -> Result<(), EncodingError> {
        if !self.config.quality_control.validate_spikes {
            return Ok(());
        }

        // Check spike count limits
        if let Some(max_spikes) = self.config.quality_control.max_spikes {
            if pattern.spike_count() > max_spikes {
                return Err(EncodingError::QualityFailed(
                    format!("Too many spikes: {} > {}", pattern.spike_count(), max_spikes)
                ));
            }
        }

        // Validate spike timing
        for spike in &pattern.spikes {
            if spike.time_ms < 0.0 || spike.time_ms >= pattern.duration_ms {
                return Err(EncodingError::QualityFailed(
                    format!("Spike time out of bounds: {:.2}ms", spike.time_ms)
                ));
            }

            if spike.neuron_id >= self.config.neuron_count {
                return Err(EncodingError::QualityFailed(
                    format!("Neuron ID out of bounds: {}", spike.neuron_id)
                ));
            }
        }

        Ok(())
    }

    /// Calculate cache key for features
    fn calculate_cache_key(&self, features: &FeatureVector) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash feature values (simplified)
        let values = features.to_vec();
        for value in values {
            ((value * 1000.0) as i64).hash(&mut hasher);
        }
        
        // Hash encoding method
        (self.config.method as u8).hash(&mut hasher);
        
        hasher.finish()
    }

    /// Update encoding statistics
    fn update_statistics(&mut self, pattern: &SpikePattern, encoding_time_us: f64) {
        self.statistics.patterns_encoded += 1;
        self.statistics.total_spikes += pattern.spike_count() as u64;
        
        // Update average encoding time (exponential moving average)
        let alpha = 0.1;
        self.statistics.average_encoding_time_us = alpha * encoding_time_us + 
            (1.0 - alpha) * self.statistics.average_encoding_time_us;
    }

    /// Get encoding statistics
    pub fn get_statistics(&self) -> &EncodingStatistics {
        &self.statistics
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.cache.feature_cache.clear();
        self.cache.pattern_cache.clear();
    }

    /// Get configuration
    pub fn get_config(&self) -> &EncodingConfig {
        &self.config
    }
}

/// N-dimensional array encoder (optional feature)
#[cfg(feature = "ndarray-support")]
pub struct NDArrayEncoder {
    inner: SpikeEncoder,
}

#[cfg(feature = "ndarray-support")]
impl NDArrayEncoder {
    /// Create new N-dimensional array encoder
    pub fn new(neuron_count: usize, window_ms: f64) -> Result<Self, EncodingError> {
        Ok(Self {
            inner: SpikeEncoder::new(neuron_count, window_ms)?,
        })
    }

    /// Encode ndarray data
    pub fn encode_array(&mut self, data: &ndarray::Array1<f64>) -> Result<SpikePattern, EncodingError> {
        // Convert ndarray to feature vector
        let mut features = FeatureVector::new();
        for (i, &value) in data.iter().enumerate() {
            features.add(&format!("feature_{}", i), value);
        }

        // Create temporary market data
        let market_data = MarketData {
            symbol: "ARRAY_DATA".to_string(),
            price: data.mean().unwrap_or(0.0),
            volume: data.len() as f64,
            timestamp: chrono::Utc::now(),
            metrics: None,
        };

        self.inner.encode(&market_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = SpikeEncoder::new(1000, 1000.0);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let encoder = SpikeEncoder::new(0, 1000.0);
        assert!(encoder.is_err());

        let encoder = SpikeEncoder::new(1000, 0.0);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_rate_encoding() {
        let mut encoder = SpikeEncoder::new(100, 1000.0).unwrap();
        let market_data = MarketData::new("BTC-USD", 50000.0, 1.5);
        
        let pattern = encoder.encode(&market_data);
        assert!(pattern.is_ok());
        
        let pattern = pattern.unwrap();
        assert!(pattern.spike_count() > 0);
        assert_eq!(pattern.duration_ms, 1000.0);
        assert_eq!(pattern.encoding_method, Some(EncodingMethod::Rate));
    }

    #[test]
    fn test_different_encoding_methods() {
        let market_data = MarketData::new("BTC-USD", 50000.0, 1.5);
        
        for method in [
            EncodingMethod::Rate,
            EncodingMethod::Temporal,
            EncodingMethod::Population,
            EncodingMethod::Phase,
            EncodingMethod::Latency,
            EncodingMethod::Burst,
        ] {
            let mut encoder = SpikeEncoder::new(100, 1000.0)
                .unwrap()
                .with_method(method)
                .unwrap();
            
            let pattern = encoder.encode(&market_data);
            assert!(pattern.is_ok(), "Failed for method {:?}", method);
        }
    }

    #[test]
    fn test_rate_scaling() {
        let mut encoder = SpikeEncoder::new(100, 1000.0).unwrap();
        
        for scaling in [
            RateScaling::Linear,
            RateScaling::Logarithmic,
            RateScaling::Exponential,
            RateScaling::Sigmoid,
        ] {
            encoder.config.rate_params.scaling = scaling;
            let rate = encoder.value_to_rate(0.5);
            assert!(rate > 0.0);
        }
    }

    #[test]
    fn test_statistics() {
        let mut encoder = SpikeEncoder::new(100, 1000.0).unwrap();
        let market_data = MarketData::new("BTC-USD", 50000.0, 1.5);
        
        encoder.encode(&market_data).unwrap();
        encoder.encode(&market_data).unwrap();
        
        let stats = encoder.get_statistics();
        assert_eq!(stats.patterns_encoded, 2);
        assert!(stats.total_spikes > 0);
    }

    #[cfg(feature = "ndarray-support")]
    #[test]
    fn test_ndarray_encoder() {
        let mut encoder = NDArrayEncoder::new(100, 1000.0).unwrap();
        let data = ndarray::Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let pattern = encoder.encode_array(&data);
        assert!(pattern.is_ok());
    }
}