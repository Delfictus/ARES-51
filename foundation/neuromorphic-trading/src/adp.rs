//! ADP Integration Module  
//! Imports and re-exports ADP functionality for quantum-enhanced trading decisions

// Re-export ADP core components
pub use csf_clogic::adp::{
    AdaptiveDecisionProcessor,
    AdpConfig,
    Decision,
    DecisionId, 
    Action,
    ReasoningStep,
    SilCore,
};

// Quantum decision components
pub use csf_clogic::adp::{
    QuantumDecisionInterface,
    QuantumConfig,
};

// Reinforcement learning
pub use csf_clogic::adp::{
    ReinforcementLearner,
    RlConfig,
};

/// Market-optimized ADP configuration
pub fn create_market_config() -> AdpConfig {
    AdpConfig {
        nn_layers: vec![64, 32, 16, 8], // Optimized for market features
        tree_depth: 8,
        learning_rate: 0.001,
        epsilon: 0.1,
        buffer_size: 10000,
        use_quantum: true,
        quantum_config: QuantumConfig::default(),
        rl_config: RlConfig::default(),
    }
}

/// Experience replay buffer for DRPP pattern learning
#[derive(Debug, Clone)]
pub struct DrppExperienceBuffer {
    buffer: Vec<DrppExperience>,
    capacity: usize,
    write_index: usize,
    is_full: bool,
}

#[derive(Debug, Clone)]
pub struct DrppExperience {
    pub pattern_features: Vec<f64>,
    pub action_taken: Action,
    pub reward: f64,
    pub next_pattern_features: Option<Vec<f64>>,
    pub timestamp_ns: u64,
    pub pattern_confidence: f64,
    pub market_volatility: f64,
}

impl DrppExperienceBuffer {
    /// Create new experience buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            write_index: 0,
            is_full: false,
        }
    }
    
    /// Add new DRPP pattern experience
    pub fn push(&mut self, experience: DrppExperience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.write_index] = experience;
            self.is_full = true;
        }
        
        self.write_index = (self.write_index + 1) % self.capacity;
    }
    
    /// Sample batch of experiences for learning
    pub fn sample_batch(&self, batch_size: usize) -> Vec<DrppExperience> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let available_size = if self.is_full { self.capacity } else { self.buffer.len() };
        if available_size == 0 {
            return Vec::new();
        }
        
        let effective_batch_size = batch_size.min(available_size);
        let mut rng = thread_rng();
        
        self.buffer.choose_multiple(&mut rng, effective_batch_size)
            .cloned()
            .collect()
    }
    
    /// Get prioritized experiences based on pattern confidence and reward magnitude
    pub fn sample_prioritized(&self, batch_size: usize) -> Vec<DrppExperience> {
        let available_size = if self.is_full { self.capacity } else { self.buffer.len() };
        if available_size == 0 {
            return Vec::new();
        }
        
        let mut indexed_experiences: Vec<_> = self.buffer.iter().enumerate()
            .map(|(i, exp)| {
                let priority = exp.pattern_confidence * exp.reward.abs() * (1.0 + exp.market_volatility);
                (priority, i, exp.clone())
            })
            .collect();
        
        // Sort by priority (highest first)
        indexed_experiences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        indexed_experiences.into_iter()
            .take(batch_size)
            .map(|(_, _, exp)| exp)
            .collect()
    }
    
    /// Get buffer statistics for monitoring
    pub fn stats(&self) -> ExperienceBufferStats {
        if self.buffer.is_empty() {
            return ExperienceBufferStats::default();
        }
        
        let rewards: Vec<f64> = self.buffer.iter().map(|e| e.reward).collect();
        let confidences: Vec<f64> = self.buffer.iter().map(|e| e.pattern_confidence).collect();
        
        ExperienceBufferStats {
            size: self.buffer.len(),
            capacity: self.capacity,
            avg_reward: rewards.iter().sum::<f64>() / rewards.len() as f64,
            max_reward: rewards.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            min_reward: rewards.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_confidence: confidences.iter().sum::<f64>() / confidences.len() as f64,
            is_full: self.is_full,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExperienceBufferStats {
    pub size: usize,
    pub capacity: usize,
    pub avg_reward: f64,
    pub max_reward: f64,
    pub min_reward: f64,
    pub avg_confidence: f64,
    pub is_full: bool,
}

/// Convert DRPP patterns to ADP decision features
pub fn patterns_to_features(patterns: &[crate::drpp::Pattern]) -> Vec<f64> {
    let mut features = Vec::new();
    
    // Pattern type distribution
    let mut type_counts = [0; 5];
    for pattern in patterns {
        let idx = match pattern.pattern_type {
            crate::drpp::PatternType::Synchronous => 0,
            crate::drpp::PatternType::Traveling => 1,
            crate::drpp::PatternType::Standing => 2,
            crate::drpp::PatternType::Chaotic => 3,
            crate::drpp::PatternType::Emergent => 4,
        };
        type_counts[idx] += 1;
    }
    
    // Normalize pattern type features
    let total = patterns.len() as f64;
    if total > 0.0 {
        for count in type_counts {
            features.push(count as f64 / total);
        }
    } else {
        features.extend_from_slice(&[0.0; 5]);
    }
    
    // Average pattern strength
    let avg_strength = patterns.iter()
        .map(|p| p.strength)
        .sum::<f64>() / total.max(1.0);
    features.push(avg_strength);
    
    // Pattern diversity (entropy)
    let entropy = calculate_pattern_entropy(patterns);
    features.push(entropy);
    
    features
}

/// Calculate pattern entropy for diversity measurement
fn calculate_pattern_entropy(patterns: &[crate::drpp::Pattern]) -> f64 {
    if patterns.is_empty() {
        return 0.0;
    }
    
    let mut type_probs = [0.0; 5];
    let total = patterns.len() as f64;
    
    for pattern in patterns {
        let idx = match pattern.pattern_type {
            crate::drpp::PatternType::Synchronous => 0,
            crate::drpp::PatternType::Traveling => 1,
            crate::drpp::PatternType::Standing => 2,
            crate::drpp::PatternType::Chaotic => 3,
            crate::drpp::PatternType::Emergent => 4,
        };
        type_probs[idx] += 1.0 / total;
    }
    
    // Calculate Shannon entropy
    type_probs.iter()
        .filter(|&p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}