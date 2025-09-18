//! Signal processing and pattern detection for neuromorphic systems

use crate::types::{Pattern, SpikePattern};
use crate::reservoir::ReservoirState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pattern detector for identifying market patterns from reservoir states
#[derive(Debug)]
pub struct PatternDetector {
    config: DetectorConfig,
    pattern_templates: HashMap<PatternType, PatternTemplate>,
    detection_history: Vec<DetectionEvent>,
}

/// Configuration for pattern detection
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Minimum confidence threshold for pattern detection
    pub confidence_threshold: f32,
    /// Maximum number of patterns to detect simultaneously
    pub max_patterns: usize,
    /// Enable temporal pattern analysis
    pub enable_temporal: bool,
    /// Window size for temporal analysis
    pub temporal_window: usize,
    /// Sensitivity setting (0.0 = conservative, 1.0 = aggressive)
    pub sensitivity: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            max_patterns: 5,
            enable_temporal: true,
            temporal_window: 10,
            sensitivity: 0.5,
        }
    }
}

/// Pattern template for matching
#[derive(Debug, Clone)]
struct PatternTemplate {
    /// Expected activation pattern
    activation_signature: Vec<f64>,
    /// Pattern type
    pattern_type: PatternType,
    /// Matching threshold
    threshold: f32,
    /// Weight for this pattern
    weight: f32,
}

/// Detection event
#[derive(Debug, Clone)]
struct DetectionEvent {
    pattern: Pattern,
    confidence: f32,
    timestamp: chrono::DateTime<chrono::Utc>,
    reservoir_state_hash: u64,
}

/// Types of patterns that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Trending patterns
    UpTrend,
    DownTrend,
    Sideways,
    
    /// Momentum patterns
    Acceleration,
    Deceleration,
    
    /// Volatility patterns
    VolatilitySpike,
    VolatilityDrop,
    
    /// Reversal patterns
    BullishReversal,
    BearishReversal,
    
    /// Breakout patterns
    UpwardBreakout,
    DownwardBreakout,
    
    /// Complex patterns
    WavePattern,
    CyclePattern,
    
    /// Anomaly patterns
    Anomaly,
    NoisePattern,
}

/// Trading signal generated from patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal direction
    pub direction: SignalDirection,
    /// Signal strength
    pub strength: SignalStrength,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Time horizon for the signal
    pub time_horizon_ms: f64,
    /// Detected patterns contributing to this signal
    pub patterns: Vec<Pattern>,
    /// Additional metadata
    pub metadata: SignalMetadata,
}

/// Direction of trading signal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold signal
    Hold,
}

/// Strength of trading signal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    /// Weak signal
    Weak,
    /// Moderate signal
    Moderate,
    /// Strong signal
    Strong,
    /// Very strong signal
    VeryStrong,
}

/// Signal metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalMetadata {
    /// Number of patterns detected
    pub pattern_count: usize,
    /// Average pattern confidence
    pub avg_pattern_confidence: f32,
    /// Reservoir activation level
    pub reservoir_activation: f32,
    /// Signal generation latency
    pub generation_latency_us: Option<f64>,
    /// Custom metadata
    pub custom: HashMap<String, f64>,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(confidence_threshold: f32) -> Self {
        let config = DetectorConfig {
            confidence_threshold,
            ..Default::default()
        };
        
        let mut detector = Self {
            config,
            pattern_templates: HashMap::new(),
            detection_history: Vec::new(),
        };
        
        detector.initialize_templates();
        detector
    }
    
    /// Create pattern detector with custom configuration
    pub fn with_config(config: DetectorConfig) -> Self {
        let mut detector = Self {
            config,
            pattern_templates: HashMap::new(),
            detection_history: Vec::new(),
        };
        
        detector.initialize_templates();
        detector
    }
    
    /// Detect patterns from reservoir state and spike pattern
    pub fn detect(
        &mut self,
        reservoir_state: &ReservoirState,
        spike_pattern: &SpikePattern,
    ) -> Result<Vec<Pattern>, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        // Analyze reservoir activations
        let activation_patterns = self.analyze_activations(&reservoir_state.activations)?;
        
        // Analyze temporal dynamics
        let temporal_patterns = if self.config.enable_temporal {
            self.analyze_temporal_dynamics(reservoir_state, spike_pattern)?
        } else {
            Vec::new()
        };
        
        // Combine and filter patterns
        let mut detected_patterns = activation_patterns;
        detected_patterns.extend(temporal_patterns);
        
        // Apply confidence filtering
        detected_patterns.retain(|p| {
            self.get_pattern_confidence(p, reservoir_state) >= self.config.confidence_threshold
        });
        
        // Limit number of patterns
        detected_patterns.truncate(self.config.max_patterns);
        
        // Record detection events
        let latency = start_time.elapsed().as_micros() as f64;
        self.record_detections(&detected_patterns, reservoir_state, latency);
        
        Ok(detected_patterns)
    }
    
    /// Generate trading signal from detected patterns
    pub fn generate_signal(
        &self,
        patterns: &[Pattern],
        reservoir_state: &ReservoirState,
    ) -> TradingSignal {
        if patterns.is_empty() {
            return TradingSignal {
                direction: SignalDirection::Hold,
                strength: SignalStrength::Weak,
                confidence: 0.0,
                time_horizon_ms: 1000.0,
                patterns: Vec::new(),
                metadata: SignalMetadata::default(),
            };
        }
        
        // Determine signal direction
        let direction = self.determine_signal_direction(patterns);
        
        // Calculate signal strength
        let strength = self.calculate_signal_strength(patterns, reservoir_state);
        
        // Calculate overall confidence
        let confidence = self.calculate_signal_confidence(patterns, reservoir_state);
        
        // Create metadata
        let metadata = SignalMetadata {
            pattern_count: patterns.len(),
            avg_pattern_confidence: patterns.iter()
                .map(|p| self.get_pattern_confidence(p, reservoir_state))
                .sum::<f32>() / patterns.len() as f32,
            reservoir_activation: reservoir_state.average_activation,
            generation_latency_us: None,
            custom: HashMap::new(),
        };
        
        TradingSignal {
            direction,
            strength,
            confidence,
            time_horizon_ms: 5000.0, // 5 second horizon
            patterns: patterns.to_vec(),
            metadata,
        }
    }
    
    /// Initialize pattern templates
    fn initialize_templates(&mut self) {
        // Uptrend pattern: increasing activation in specific regions
        self.pattern_templates.insert(
            PatternType::UpTrend,
            PatternTemplate {
                activation_signature: vec![0.2, 0.4, 0.6, 0.8, 1.0],
                pattern_type: PatternType::UpTrend,
                threshold: 0.7,
                weight: 1.0,
            },
        );
        
        // Downtrend pattern: decreasing activation
        self.pattern_templates.insert(
            PatternType::DownTrend,
            PatternTemplate {
                activation_signature: vec![1.0, 0.8, 0.6, 0.4, 0.2],
                pattern_type: PatternType::DownTrend,
                threshold: 0.7,
                weight: 1.0,
            },
        );
        
        // Volatility spike: high activation variance
        self.pattern_templates.insert(
            PatternType::VolatilitySpike,
            PatternTemplate {
                activation_signature: vec![0.1, 0.9, 0.1, 0.9, 0.1],
                pattern_type: PatternType::VolatilitySpike,
                threshold: 0.6,
                weight: 0.8,
            },
        );
        
        // Sideways: stable activation
        self.pattern_templates.insert(
            PatternType::Sideways,
            PatternTemplate {
                activation_signature: vec![0.5, 0.5, 0.5, 0.5, 0.5],
                pattern_type: PatternType::Sideways,
                threshold: 0.8,
                weight: 0.6,
            },
        );
        
        // Add more templates as needed
        self.add_complex_templates();
    }
    
    /// Add complex pattern templates
    fn add_complex_templates(&mut self) {
        // Reversal patterns
        self.pattern_templates.insert(
            PatternType::BullishReversal,
            PatternTemplate {
                activation_signature: vec![0.8, 0.3, 0.2, 0.4, 0.7],
                pattern_type: PatternType::BullishReversal,
                threshold: 0.65,
                weight: 1.2,
            },
        );
        
        // Breakout patterns
        self.pattern_templates.insert(
            PatternType::UpwardBreakout,
            PatternTemplate {
                activation_signature: vec![0.5, 0.5, 0.6, 0.8, 0.9],
                pattern_type: PatternType::UpwardBreakout,
                threshold: 0.75,
                weight: 1.1,
            },
        );
    }
    
    /// Analyze activation patterns
    fn analyze_activations(
        &self,
        activations: &[f64],
    ) -> Result<Vec<Pattern>, Box<dyn std::error::Error>> {
        let mut patterns = Vec::new();
        
        if activations.is_empty() {
            return Ok(patterns);
        }
        
        // Divide activations into segments for analysis
        let segment_size = activations.len() / 5;
        if segment_size == 0 {
            return Ok(patterns);
        }
        
        let segments: Vec<f64> = (0..5)
            .map(|i| {
                let start = i * segment_size;
                let end = ((i + 1) * segment_size).min(activations.len());
                activations[start..end].iter().sum::<f64>() / (end - start) as f64
            })
            .collect();
        
        // Match against templates
        for (pattern_type, template) in &self.pattern_templates {
            let similarity = self.calculate_similarity(&segments, &template.activation_signature);
            
            if similarity >= template.threshold {
                let pattern = match pattern_type {
                    PatternType::UpTrend => Pattern::UpTrend,
                    PatternType::DownTrend => Pattern::DownTrend,
                    PatternType::Sideways => Pattern::Sideways,
                    PatternType::VolatilitySpike => Pattern::HighVolatility,
                    PatternType::BullishReversal => Pattern::Reversal,
                    PatternType::UpwardBreakout => Pattern::Breakout,
                    _ => Pattern::Custom(format!("{:?}", pattern_type)),
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Analyze temporal dynamics
    fn analyze_temporal_dynamics(
        &self,
        reservoir_state: &ReservoirState,
        _spike_pattern: &SpikePattern,
    ) -> Result<Vec<Pattern>, Box<dyn std::error::Error>> {
        let mut patterns = Vec::new();
        
        // Analyze memory capacity for temporal patterns
        if reservoir_state.dynamics.memory_capacity > 0.7 {
            patterns.push(Pattern::Custom("high_memory".to_string()));
        }
        
        // Analyze separation for distinct patterns
        if reservoir_state.dynamics.separation > 0.8 {
            patterns.push(Pattern::Custom("high_separation".to_string()));
        }
        
        // Analyze approximation for complex patterns
        if reservoir_state.dynamics.approximation > 0.75 {
            patterns.push(Pattern::Custom("complex_dynamics".to_string()));
        }
        
        Ok(patterns)
    }
    
    /// Calculate similarity between two activation patterns
    fn calculate_similarity(&self, pattern1: &[f64], pattern2: &[f64]) -> f32 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }
        
        // Calculate cosine similarity
        let dot_product: f64 = pattern1.iter()
            .zip(pattern2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1: f64 = pattern1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = pattern2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        (dot_product / (norm1 * norm2)) as f32
    }
    
    /// Get confidence for a specific pattern
    fn get_pattern_confidence(&self, pattern: &Pattern, reservoir_state: &ReservoirState) -> f32 {
        // Base confidence from reservoir activation
        let base_confidence = reservoir_state.average_activation;
        
        // Adjust based on pattern type
        let pattern_weight = match pattern {
            Pattern::UpTrend | Pattern::DownTrend => 1.0,
            Pattern::Reversal | Pattern::Breakout => 1.2,
            Pattern::HighVolatility => 0.8,
            Pattern::Sideways | Pattern::LowVolatility => 0.6,
            Pattern::Custom(_) => 0.7,
        };
        
        (base_confidence * pattern_weight).min(1.0)
    }
    
    /// Determine signal direction from patterns
    fn determine_signal_direction(&self, patterns: &[Pattern]) -> SignalDirection {
        let mut bullish_score = 0.0;
        let mut bearish_score = 0.0;
        
        for pattern in patterns {
            match pattern {
                Pattern::UpTrend | Pattern::Breakout => bullish_score += 1.0,
                Pattern::DownTrend => bearish_score += 1.0,
                Pattern::Reversal => {
                    // Context-dependent - could be either direction
                    bullish_score += 0.5;
                    bearish_score += 0.5;
                }
                Pattern::HighVolatility => {
                    // High volatility could go either way
                    bullish_score += 0.3;
                    bearish_score += 0.3;
                }
                _ => {} // Neutral patterns
            }
        }
        
        if bullish_score > bearish_score + 0.2 {
            SignalDirection::Buy
        } else if bearish_score > bullish_score + 0.2 {
            SignalDirection::Sell
        } else {
            SignalDirection::Hold
        }
    }
    
    /// Calculate signal strength
    fn calculate_signal_strength(&self, patterns: &[Pattern], reservoir_state: &ReservoirState) -> SignalStrength {
        let pattern_count = patterns.len() as f32;
        let avg_confidence = patterns.iter()
            .map(|p| self.get_pattern_confidence(p, reservoir_state))
            .sum::<f32>() / patterns.len().max(1) as f32;
        
        let strength_score = (pattern_count / 5.0) * avg_confidence;
        
        match strength_score {
            s if s >= 0.8 => SignalStrength::VeryStrong,
            s if s >= 0.6 => SignalStrength::Strong,
            s if s >= 0.4 => SignalStrength::Moderate,
            _ => SignalStrength::Weak,
        }
    }
    
    /// Calculate overall signal confidence
    fn calculate_signal_confidence(&self, patterns: &[Pattern], reservoir_state: &ReservoirState) -> f32 {
        if patterns.is_empty() {
            return 0.0;
        }
        
        let pattern_confidences: f32 = patterns.iter()
            .map(|p| self.get_pattern_confidence(p, reservoir_state))
            .sum();
        
        let avg_pattern_confidence = pattern_confidences / patterns.len() as f32;
        let reservoir_confidence = reservoir_state.average_activation;
        
        // Combine confidences with weights
        (0.7 * avg_pattern_confidence + 0.3 * reservoir_confidence).min(1.0)
    }
    
    /// Record detection events
    fn record_detections(
        &mut self,
        patterns: &[Pattern],
        reservoir_state: &ReservoirState,
        _latency: f64,
    ) {
        let state_hash = self.calculate_state_hash(&reservoir_state.activations);
        
        for pattern in patterns {
            let event = DetectionEvent {
                pattern: pattern.clone(),
                confidence: self.get_pattern_confidence(pattern, reservoir_state),
                timestamp: chrono::Utc::now(),
                reservoir_state_hash: state_hash,
            };
            
            self.detection_history.push(event);
        }
        
        // Keep only recent history
        if self.detection_history.len() > 1000 {
            self.detection_history.drain(0..100);
        }
    }
    
    /// Calculate hash of reservoir state
    fn calculate_state_hash(&self, activations: &[f64]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash a simplified version of the activation pattern
        let simplified: Vec<i32> = activations.iter()
            .map(|&x| (x * 1000.0) as i32)
            .collect();
        
        simplified.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Spike;

    #[test]
    fn test_pattern_detector_creation() {
        let detector = PatternDetector::new(0.5);
        assert!(!detector.pattern_templates.is_empty());
    }

    #[test]
    fn test_pattern_detection() {
        let mut detector = PatternDetector::new(0.3);
        
        // Create mock reservoir state
        let activations = vec![0.2, 0.4, 0.6, 0.8, 1.0]; // Uptrend pattern
        let reservoir_state = ReservoirState {
            activations,
            average_activation: 0.6,
            max_activation: 1.0,
            last_spike_count: 5,
            dynamics: crate::reservoir::DynamicsMetrics::default(),
        };
        
        let spikes = vec![Spike::new(0, 10.0)];
        let spike_pattern = SpikePattern::new(spikes, 100.0);
        
        let patterns = detector.detect(&reservoir_state, &spike_pattern).unwrap();
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_signal_generation() {
        let detector = PatternDetector::new(0.5);
        let patterns = vec![Pattern::UpTrend, Pattern::Breakout];
        
        let reservoir_state = ReservoirState {
            activations: vec![0.5; 10],
            average_activation: 0.7,
            max_activation: 0.9,
            last_spike_count: 10,
            dynamics: crate::reservoir::DynamicsMetrics::default(),
        };
        
        let signal = detector.generate_signal(&patterns, &reservoir_state);
        
        assert_eq!(signal.direction, SignalDirection::Buy);
        assert!(signal.confidence > 0.0);
        assert_eq!(signal.patterns.len(), 2);
    }

    #[test]
    fn test_similarity_calculation() {
        let detector = PatternDetector::new(0.5);
        
        let pattern1 = vec![1.0, 2.0, 3.0];
        let pattern2 = vec![1.0, 2.0, 3.0];
        let similarity = detector.calculate_similarity(&pattern1, &pattern2);
        assert!((similarity - 1.0).abs() < 0.001);
        
        let pattern3 = vec![3.0, 2.0, 1.0];
        let similarity2 = detector.calculate_similarity(&pattern1, &pattern3);
        assert!(similarity2 < similarity);
    }
}