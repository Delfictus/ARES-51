//! Core neuromorphic engine implementation

use crate::types::{MarketData, Prediction, PredictionDirection, Pattern};
use crate::spike::SpikeEncoder;
use crate::reservoir::ReservoirComputer;
use crate::signals::PatternDetector;

use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Main neuromorphic engine for processing market data
#[derive(Debug)]
pub struct NeuromorphicEngine {
    config: EngineConfig,
    spike_encoder: SpikeEncoder,
    reservoir: ReservoirComputer,
    pattern_detector: PatternDetector,
    state: Arc<RwLock<EngineState>>,
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Number of neurons for spike encoding
    pub spike_neurons: usize,
    /// Size of the reservoir network
    pub reservoir_size: usize,
    /// Encoding time window in milliseconds
    pub encoding_window_ms: f64,
    /// Reservoir spectral radius
    pub spectral_radius: f64,
    /// Connection probability in reservoir
    pub connection_prob: f64,
    /// Leak rate for reservoir neurons
    pub leak_rate: f64,
    /// Minimum confidence threshold for predictions
    pub confidence_threshold: f32,
    /// Enable pattern caching
    pub enable_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            spike_neurons: 10000,
            reservoir_size: 5000,
            encoding_window_ms: 1000.0,
            spectral_radius: 0.95,
            connection_prob: 0.1,
            leak_rate: 0.3,
            confidence_threshold: 0.6,
            enable_caching: true,
            max_cache_size: 10000,
        }
    }
}

/// Internal engine state
#[derive(Debug)]
struct EngineState {
    /// Processing statistics
    samples_processed: u64,
    predictions_made: u64,
    average_latency_us: f64,
    /// Pattern cache
    pattern_cache: HashMap<u64, CachedPattern>,
    /// Last prediction
    last_prediction: Option<Prediction>,
}

/// Cached pattern information
#[derive(Debug, Clone)]
struct CachedPattern {
    pattern: Pattern,
    confidence: f32,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            predictions_made: 0,
            average_latency_us: 0.0,
            pattern_cache: HashMap::new(),
            last_prediction: None,
        }
    }
}

/// Processing errors
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Spike encoding error
    #[error("Spike encoding failed: {0}")]
    SpikeEncodingError(String),
    
    /// Reservoir computation error
    #[error("Reservoir computation failed: {0}")]
    ReservoirError(String),
    
    /// Pattern detection error
    #[error("Pattern detection failed: {0}")]
    PatternError(String),
    
    /// Invalid input data
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
}

impl From<anyhow::Error> for ProcessingError {
    fn from(err: anyhow::Error) -> Self {
        ProcessingError::InvalidConfig(err.to_string())
    }
}

impl NeuromorphicEngine {
    /// Create a new neuromorphic engine
    pub fn new(config: EngineConfig) -> Result<Self, ProcessingError> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Initialize components
        let spike_encoder = SpikeEncoder::new(
            config.spike_neurons,
            config.encoding_window_ms,
        )?;
        
        let reservoir = ReservoirComputer::new(
            config.reservoir_size,
            config.spike_neurons,
            config.spectral_radius,
            config.connection_prob,
            config.leak_rate,
        )?;
        
        let pattern_detector = PatternDetector::new(
            config.confidence_threshold,
        );
        
        let state = Arc::new(RwLock::new(EngineState::default()));
        
        Ok(Self {
            config,
            spike_encoder,
            reservoir,
            pattern_detector,
            state,
        })
    }
    
    /// Process market data and generate prediction
    pub fn process(&mut self, market_data: MarketData) -> Result<Prediction, ProcessingError> {
        let start_time = std::time::Instant::now();
        
        // 1. Encode market data to spikes
        let spike_pattern = self.spike_encoder.encode(&market_data)
            .map_err(|e| ProcessingError::SpikeEncodingError(e.to_string()))?;
        
        // 2. Process spikes through reservoir
        let reservoir_state = self.reservoir.process(&spike_pattern)
            .map_err(|e| ProcessingError::ReservoirError(e.to_string()))?;
        
        // 3. Detect patterns
        let detected_patterns = self.pattern_detector.detect(&reservoir_state, &spike_pattern)
            .map_err(|e| ProcessingError::PatternError(e.to_string()))?;
        
        // 4. Generate prediction
        let prediction = self.generate_prediction(
            &detected_patterns,
            &reservoir_state,
            &market_data,
        )?;
        
        // Update statistics
        let latency = start_time.elapsed().as_micros() as f64;
        self.update_statistics(latency, &prediction);
        
        Ok(prediction)
    }
    
    /// Get engine statistics
    pub fn get_statistics(&self) -> EngineStatistics {
        let state = self.state.read();
        EngineStatistics {
            samples_processed: state.samples_processed,
            predictions_made: state.predictions_made,
            average_latency_us: state.average_latency_us,
            cache_size: state.pattern_cache.len(),
            config: self.config.clone(),
        }
    }
    
    /// Reset engine state
    pub fn reset(&mut self) {
        let mut state = self.state.write();
        *state = EngineState::default();
        self.reservoir.reset();
    }
    
    /// Get last prediction
    pub fn last_prediction(&self) -> Option<Prediction> {
        self.state.read().last_prediction.clone()
    }
    
    /// Validate configuration
    fn validate_config(config: &EngineConfig) -> Result<(), ProcessingError> {
        if config.spike_neurons == 0 {
            return Err(ProcessingError::InvalidConfig(
                "spike_neurons must be greater than 0".to_string()
            ));
        }
        
        if config.reservoir_size == 0 {
            return Err(ProcessingError::InvalidConfig(
                "reservoir_size must be greater than 0".to_string()
            ));
        }
        
        if config.encoding_window_ms <= 0.0 {
            return Err(ProcessingError::InvalidConfig(
                "encoding_window_ms must be positive".to_string()
            ));
        }
        
        if config.spectral_radius <= 0.0 || config.spectral_radius >= 1.0 {
            return Err(ProcessingError::InvalidConfig(
                "spectral_radius must be between 0 and 1".to_string()
            ));
        }
        
        if config.connection_prob < 0.0 || config.connection_prob > 1.0 {
            return Err(ProcessingError::InvalidConfig(
                "connection_prob must be between 0 and 1".to_string()
            ));
        }
        
        if config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0 {
            return Err(ProcessingError::InvalidConfig(
                "confidence_threshold must be between 0 and 1".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Generate prediction from detected patterns
    fn generate_prediction(
        &self,
        patterns: &[Pattern],
        reservoir_state: &crate::reservoir::ReservoirState,
        _market_data: &MarketData,
    ) -> Result<Prediction, ProcessingError> {
        if patterns.is_empty() {
            return Ok(Prediction::new(
                PredictionDirection::Hold,
                0.0,
                self.config.encoding_window_ms,
            ));
        }
        
        // Determine direction based on patterns
        let direction = self.determine_direction(patterns);
        
        // Calculate confidence based on reservoir state and patterns
        let confidence = self.calculate_confidence(patterns, reservoir_state);
        
        // Create prediction
        let mut prediction = Prediction::new(
            direction,
            confidence,
            self.config.encoding_window_ms,
        );
        
        // Add detected patterns
        for pattern in patterns {
            prediction = prediction.with_pattern(pattern.clone());
        }
        
        // Add metadata
        prediction.metadata.model_version = Some("ares-neuromorphic-v0.1.0".to_string());
        prediction.metadata.spike_count = Some(reservoir_state.last_spike_count);
        
        Ok(prediction)
    }
    
    /// Determine prediction direction from patterns
    fn determine_direction(&self, patterns: &[Pattern]) -> PredictionDirection {
        let mut bullish_score = 0.0;
        let mut bearish_score = 0.0;
        
        for pattern in patterns {
            match pattern {
                Pattern::UpTrend | Pattern::Breakout => bullish_score += 1.0,
                Pattern::DownTrend | Pattern::Reversal => bearish_score += 1.0,
                Pattern::HighVolatility => {
                    // High volatility could go either way
                    bullish_score += 0.5;
                    bearish_score += 0.5;
                }
                Pattern::LowVolatility | Pattern::Sideways => {
                    // Low volatility suggests holding
                }
                Pattern::Custom(_) => {
                    // Custom patterns are neutral by default
                }
            }
        }
        
        if bullish_score > bearish_score {
            PredictionDirection::Up
        } else if bearish_score > bullish_score {
            PredictionDirection::Down
        } else {
            PredictionDirection::Hold
        }
    }
    
    /// Calculate prediction confidence
    fn calculate_confidence(
        &self,
        patterns: &[Pattern],
        reservoir_state: &crate::reservoir::ReservoirState,
    ) -> f32 {
        if patterns.is_empty() {
            return 0.0;
        }
        
        // Base confidence from number of patterns
        let pattern_confidence = (patterns.len() as f32).min(3.0) / 3.0;
        
        // Reservoir state confidence (based on activation levels)
        let reservoir_confidence = reservoir_state.average_activation.min(1.0);
        
        // Combine confidences
        let combined = (pattern_confidence + reservoir_confidence) / 2.0;
        combined.clamp(0.0, 1.0)
    }
    
    /// Update engine statistics
    fn update_statistics(&self, latency_us: f64, prediction: &Prediction) {
        let mut state = self.state.write();
        
        state.samples_processed += 1;
        if prediction.confidence >= self.config.confidence_threshold {
            state.predictions_made += 1;
        }
        
        // Update average latency (exponential moving average)
        let alpha = 0.1;
        state.average_latency_us = alpha * latency_us + (1.0 - alpha) * state.average_latency_us;
        
        // Cache prediction
        state.last_prediction = Some(prediction.clone());
        
        // Update pattern cache
        if self.config.enable_caching {
            self.update_pattern_cache(&mut state, prediction);
        }
    }
    
    /// Update pattern cache
    fn update_pattern_cache(&self, state: &mut EngineState, prediction: &Prediction) {
        for pattern in &prediction.patterns {
            let key = self.pattern_hash(pattern);
            let cached = CachedPattern {
                pattern: pattern.clone(),
                confidence: prediction.confidence,
                timestamp: chrono::Utc::now(),
            };
            
            state.pattern_cache.insert(key, cached);
            
            // Evict old entries if cache is too large
            if state.pattern_cache.len() > self.config.max_cache_size {
                // Remove oldest entry (simple LRU approximation)
                if let Some(oldest_key) = state.pattern_cache.keys().next().copied() {
                    state.pattern_cache.remove(&oldest_key);
                }
            }
        }
    }
    
    /// Generate hash for pattern (for caching)
    fn pattern_hash(&self, pattern: &Pattern) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        hasher.finish()
    }
}

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatistics {
    /// Total samples processed
    pub samples_processed: u64,
    /// Total predictions made (above confidence threshold)
    pub predictions_made: u64,
    /// Average processing latency in microseconds
    pub average_latency_us: f64,
    /// Current cache size
    pub cache_size: usize,
    /// Engine configuration
    pub config: EngineConfig,
}

/// Async version of the neuromorphic engine
#[cfg(feature = "async")]
#[derive(Debug)]
pub struct AsyncNeuromorphicEngine {
    inner: Arc<tokio::sync::Mutex<NeuromorphicEngine>>,
}

#[cfg(feature = "async")]
impl AsyncNeuromorphicEngine {
    /// Create new async engine
    pub fn new(config: EngineConfig) -> Result<Self, ProcessingError> {
        let engine = NeuromorphicEngine::new(config)?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(engine)),
        })
    }
    
    /// Process market data asynchronously
    pub async fn process(&self, market_data: MarketData) -> Result<Prediction, ProcessingError> {
        let mut engine = self.inner.lock().await;
        engine.process(market_data)
    }
    
    /// Get statistics asynchronously
    pub async fn get_statistics(&self) -> EngineStatistics {
        let engine = self.inner.lock().await;
        engine.get_statistics()
    }
    
    /// Reset engine state asynchronously
    pub async fn reset(&self) {
        let mut engine = self.inner.lock().await;
        engine.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = NeuromorphicEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = EngineConfig {
            spike_neurons: 0,
            ..Default::default()
        };
        let engine = NeuromorphicEngine::new(config);
        assert!(engine.is_err());
    }

    #[test]
    fn test_process_market_data() {
        let config = EngineConfig {
            spike_neurons: 100,
            reservoir_size: 50,
            ..Default::default()
        };
        let mut engine = NeuromorphicEngine::new(config).unwrap();
        
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        let result = engine.process(market_data);
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn test_statistics() {
        let config = EngineConfig::default();
        let engine = NeuromorphicEngine::new(config).unwrap();
        
        let stats = engine.get_statistics();
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.predictions_made, 0);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_engine() {
        let config = EngineConfig {
            spike_neurons: 100,
            reservoir_size: 50,
            ..Default::default()
        };
        let engine = AsyncNeuromorphicEngine::new(config).unwrap();
        
        let market_data = MarketData::new("BTC-USD".to_string(), 50000.0, 1.5);
        let result = engine.process(market_data).await;
        assert!(result.is_ok());
    }
}