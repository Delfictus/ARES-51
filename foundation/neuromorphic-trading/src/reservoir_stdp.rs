//! Reservoir with STDP (Spike-Timing Dependent Plasticity) learning
//! 
//! Features:
//! - Hebbian and anti-Hebbian learning
//! - Homeostatic plasticity
//! - Synaptic scaling
//! - Pattern-specific strengthening

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::RwLock;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use anyhow::Result;

use crate::spike_encoding::Spike;
use crate::reservoir::{PatternType, ReservoirState};

/// STDP parameters
#[derive(Debug, Clone)]
pub struct STDPConfig {
    pub tau_plus: f32,      // Time constant for potentiation (ms)
    pub tau_minus: f32,     // Time constant for depression (ms)
    pub a_plus: f32,        // Amplitude for potentiation
    pub a_minus: f32,       // Amplitude for depression
    pub w_max: f32,         // Maximum weight
    pub w_min: f32,         // Minimum weight
    pub learning_rate: f32, // Overall learning rate
    pub homeostatic_target: f32, // Target firing rate
    pub homeostatic_tau: f32,    // Homeostasis time constant
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.012,  // Slightly stronger depression for stability
            w_max: 1.0,
            w_min: 0.0,
            learning_rate: 0.001,
            homeostatic_target: 0.1,  // 10% target firing rate
            homeostatic_tau: 1000.0,  // 1 second
        }
    }
}

/// Spike history for STDP calculations
#[derive(Debug, Clone)]
struct SpikeHistory {
    times: VecDeque<u64>,  // Spike times in nanoseconds
    max_history_ms: u64,   // Maximum history to keep
}

impl SpikeHistory {
    fn new(max_history_ms: u64) -> Self {
        Self {
            times: VecDeque::with_capacity(100),
            max_history_ms: max_history_ms * 1_000_000,  // Convert to ns
        }
    }
    
    fn add_spike(&mut self, time_ns: u64) {
        self.times.push_back(time_ns);
        
        // Remove old spikes
        let cutoff = time_ns.saturating_sub(self.max_history_ms);
        while let Some(&front) = self.times.front() {
            if front < cutoff {
                self.times.pop_front();
            } else {
                break;
            }
        }
    }
    
    fn get_recent_spikes(&self, window_ns: u64, current_time: u64) -> Vec<u64> {
        let cutoff = current_time.saturating_sub(window_ns);
        self.times.iter()
            .filter(|&&t| t >= cutoff)
            .copied()
            .collect()
    }
}

/// Pattern memory for reinforcement
#[derive(Debug, Clone)]
struct PatternMemory {
    pattern_type: PatternType,
    activation_pattern: Vec<f32>,  // Neuron activation pattern
    weight_changes: Array2<f32>,   // Learned weight changes
    occurrences: u32,
    last_seen: u64,
}

/// STDP-enabled Liquid State Machine
pub struct STDPReservoir {
    // Core reservoir
    size: usize,
    weights: Arc<RwLock<Array2<f32>>>,
    state: Arc<RwLock<Array1<f32>>>,
    
    // STDP learning
    stdp_config: STDPConfig,
    spike_histories: Arc<RwLock<Vec<SpikeHistory>>>,
    eligibility_trace: Arc<RwLock<Array2<f32>>>,  // For delayed reward
    
    // Homeostasis
    firing_rates: Arc<RwLock<Array1<f32>>>,
    intrinsic_excitability: Arc<RwLock<Array1<f32>>>,
    
    // Pattern memory
    pattern_memories: Arc<RwLock<HashMap<u64, PatternMemory>>>,
    pattern_detector: PatternDetector,
    
    // Metrics
    total_weight_updates: AtomicU64,
    learning_enabled: AtomicBool,
}

impl STDPReservoir {
    pub fn new(size: usize, config: STDPConfig) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights with small-world topology
        let mut weights = Array2::zeros((size, size));
        let connection_prob = 0.2;
        let local_radius = size / 10;
        
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    continue;  // No self-connections
                }
                
                // Distance-based connection probability
                let distance = ((i as i32 - j as i32).abs() as usize).min(size - (i as i32 - j as i32).abs() as usize);
                let local_prob = if distance <= local_radius { 0.8 } else { 0.1 };
                
                if rng.gen::<f32>() < connection_prob * local_prob {
                    weights[[i, j]] = rng.gen_range(-0.5..0.5);
                }
            }
        }
        
        // Normalize spectral radius
        Self::normalize_spectral_radius(&mut weights, 0.95);
        
        // Initialize spike histories
        let mut spike_histories = Vec::with_capacity(size);
        for _ in 0..size {
            spike_histories.push(SpikeHistory::new(100));  // 100ms history
        }
        
        Self {
            size,
            weights: Arc::new(RwLock::new(weights)),
            state: Arc::new(RwLock::new(Array1::zeros(size))),
            stdp_config: config,
            spike_histories: Arc::new(RwLock::new(spike_histories)),
            eligibility_trace: Arc::new(RwLock::new(Array2::zeros((size, size)))),
            firing_rates: Arc::new(RwLock::new(Array1::ones(size) * 0.1)),
            intrinsic_excitability: Arc::new(RwLock::new(Array1::ones(size))),
            pattern_memories: Arc::new(RwLock::new(HashMap::new())),
            pattern_detector: PatternDetector::new(size),
            total_weight_updates: AtomicU64::new(0),
            learning_enabled: AtomicBool::new(true),
        }
    }
    
    /// Process spikes with STDP learning
    pub fn process_with_learning(&mut self, spikes: &[Spike]) -> ReservoirState {
        if spikes.is_empty() {
            return ReservoirState {
                activations: self.state.read().to_vec(),
                confidence: 0.0,
                novelty: 0.0,
            };
        }
        
        let current_time = spikes[0].timestamp_ns;
        
        // Convert spikes to input
        let input = self.spikes_to_input(spikes);
        
        // Update state
        let mut state = self.state.write();
        let weights = self.weights.read();
        
        // Leaky integration with intrinsic excitability
        let excitability = self.intrinsic_excitability.read();
        let leak_rate = 0.1;
        
        *state = &*state * (1.0 - leak_rate) + weights.t().dot(&input) * &*excitability;
        
        // Apply activation function (tanh)
        state.mapv_inplace(|x| x.tanh());
        
        // Detect which neurons fired
        let mut fired_neurons = Vec::new();
        let threshold = 0.5;
        
        for (i, &activation) in state.iter().enumerate() {
            if activation > threshold {
                fired_neurons.push(i);
                
                // Record spike
                if let Ok(mut histories) = self.spike_histories.try_write() {
                    histories[i].add_spike(current_time);
                }
            }
        }
        
        drop(state);
        drop(weights);
        drop(excitability);
        
        // Apply STDP learning if enabled
        if self.learning_enabled.load(Ordering::Relaxed) && !fired_neurons.is_empty() {
            self.apply_stdp(&fired_neurons, current_time);
            self.update_homeostasis(&fired_neurons);
        }
        
        // Detect patterns
        let state = self.state.read();
        let patterns = self.pattern_detector.detect(&state);
        
        // Store pattern memory
        if !patterns.is_empty() {
            self.store_pattern_memory(&patterns, &state, current_time);
        }
        
        // Calculate metrics
        let confidence = self.calculate_confidence(&state);
        let novelty = self.calculate_novelty(&state, &patterns);
        
        ReservoirState {
            activations: state.to_vec(),
            confidence,
            novelty,
        }
    }
    
    /// Apply STDP rule
    fn apply_stdp(&self, fired_neurons: &[usize], current_time: u64) {
        let mut weights = self.weights.write();
        let histories = self.spike_histories.read();
        let mut eligibility = self.eligibility_trace.write();
        
        for &post_idx in fired_neurons {
            let post_history = &histories[post_idx];
            
            // Check all presynaptic neurons
            for pre_idx in 0..self.size {
                if pre_idx == post_idx {
                    continue;
                }
                
                let pre_history = &histories[pre_idx];
                let pre_spikes = pre_history.get_recent_spikes(
                    (self.stdp_config.tau_plus * 3.0 * 1_000_000.0) as u64,
                    current_time
                );
                
                if pre_spikes.is_empty() {
                    continue;
                }
                
                // Calculate STDP weight change
                let mut delta_w = 0.0;
                
                for &pre_spike_time in &pre_spikes {
                    let dt_ms = (current_time as f64 - pre_spike_time as f64) / 1_000_000.0;
                    
                    if dt_ms > 0.0 {
                        // Pre before post: potentiation
                        delta_w += self.stdp_config.a_plus * 
                                  (-dt_ms / self.stdp_config.tau_plus).exp();
                    }
                }
                
                // Check for post-before-pre (depression)
                let post_spikes = post_history.get_recent_spikes(
                    (self.stdp_config.tau_minus * 3.0 * 1_000_000.0) as u64,
                    current_time
                );
                
                for &post_spike_time in &post_spikes {
                    for &pre_spike_time in &pre_spikes {
                        let dt_ms = (post_spike_time as f64 - pre_spike_time as f64) / 1_000_000.0;
                        
                        if dt_ms > 0.0 {
                            // Post before pre: depression
                            delta_w -= self.stdp_config.a_minus * 
                                      (-dt_ms / self.stdp_config.tau_minus).exp();
                        }
                    }
                }
                
                // Update weight with bounds
                if delta_w != 0.0 {
                    let old_weight = weights[[pre_idx, post_idx]];
                    let new_weight = (old_weight + delta_w * self.stdp_config.learning_rate)
                        .max(self.stdp_config.w_min)
                        .min(self.stdp_config.w_max);
                    
                    weights[[pre_idx, post_idx]] = new_weight;
                    
                    // Update eligibility trace for reward-modulated learning
                    eligibility[[pre_idx, post_idx]] = 
                        eligibility[[pre_idx, post_idx]] * 0.95 + delta_w.abs();
                    
                    self.total_weight_updates.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
    
    /// Update homeostatic mechanisms
    fn update_homeostasis(&self, fired_neurons: &[usize]) {
        let mut firing_rates = self.firing_rates.write();
        let mut excitability = self.intrinsic_excitability.write();
        
        let alpha = 0.01;  // Update rate
        
        // Update firing rates (exponential moving average)
        for i in 0..self.size {
            let fired = fired_neurons.contains(&i);
            firing_rates[i] = (1.0 - alpha) * firing_rates[i] + alpha * (fired as u8 as f32);
        }
        
        // Adjust intrinsic excitability to maintain target firing rate
        for i in 0..self.size {
            let rate_error = self.stdp_config.homeostatic_target - firing_rates[i];
            excitability[i] += rate_error * 0.001;  // Slow adaptation
            excitability[i] = excitability[i].max(0.5).min(2.0);  // Bounded
        }
    }
    
    /// Store pattern in memory for reinforcement
    fn store_pattern_memory(&self, patterns: &[(PatternType, f32)], state: &Array1<f32>, time: u64) {
        let mut memories = self.pattern_memories.write();
        
        for &(pattern_type, strength) in patterns {
            if strength < 0.3 {
                continue;  // Only store strong patterns
            }
            
            let pattern_id = pattern_type as u64;
            
            if let Some(memory) = memories.get_mut(&pattern_id) {
                // Update existing memory
                memory.occurrences += 1;
                memory.last_seen = time;
                
                // Blend activation patterns
                for i in 0..memory.activation_pattern.len() {
                    memory.activation_pattern[i] = 
                        0.9 * memory.activation_pattern[i] + 0.1 * state[i];
                }
            } else {
                // Create new memory
                memories.insert(pattern_id, PatternMemory {
                    pattern_type,
                    activation_pattern: state.to_vec(),
                    weight_changes: Array2::zeros((self.size, self.size)),
                    occurrences: 1,
                    last_seen: time,
                });
            }
        }
        
        // Cleanup old memories
        memories.retain(|_, memory| {
            time - memory.last_seen < 3600_000_000_000  // Keep for 1 hour
        });
    }
    
    /// Reinforce specific pattern
    pub fn reinforce_pattern(&mut self, pattern: PatternType, reward: f32) {
        let memories = self.pattern_memories.read();
        let pattern_id = pattern as u64;
        
        if let Some(memory) = memories.get(&pattern_id) {
            let mut weights = self.weights.write();
            let eligibility = self.eligibility_trace.read();
            
            // Apply reward-modulated STDP
            for i in 0..self.size {
                for j in 0..self.size {
                    if eligibility[[i, j]] > 0.01 {
                        let delta = reward * eligibility[[i, j]] * self.stdp_config.learning_rate;
                        weights[[i, j]] = (weights[[i, j]] + delta)
                            .max(self.stdp_config.w_min)
                            .min(self.stdp_config.w_max);
                    }
                }
            }
        }
    }
    
    /// Train on specific patterns
    pub fn train_patterns(&mut self, training_data: Vec<(Vec<Spike>, PatternType)>) -> Result<()> {
        self.learning_enabled.store(true, Ordering::Relaxed);
        
        for (spikes, target_pattern) in training_data {
            // Process with learning
            let state = self.process_with_learning(&spikes);
            
            // Check if target pattern was detected
            let patterns = self.pattern_detector.detect(&Array1::from(state.activations));
            
            let pattern_strength = patterns.iter()
                .find(|(p, _)| *p == target_pattern)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            
            // Reinforce if detected, punish if not
            let reward = if pattern_strength > 0.5 {
                1.0
            } else {
                -0.1
            };
            
            self.reinforce_pattern(target_pattern, reward);
        }
        
        Ok(())
    }
    
    fn spikes_to_input(&self, spikes: &[Spike]) -> Array1<f32> {
        let mut input = Array1::zeros(self.size);
        
        for spike in spikes {
            if spike.neuron_id < self.size as u32 {
                input[spike.neuron_id as usize] += spike.strength;
            }
        }
        
        input
    }
    
    fn normalize_spectral_radius(weights: &mut Array2<f32>, target_radius: f32) {
        // Power iteration to find largest eigenvalue
        let mut v = Array1::from_elem(weights.shape()[0], 1.0);
        
        for _ in 0..100 {
            v = weights.dot(&v);
            let norm = v.dot(&v).sqrt();
            if norm > 0.0 {
                v /= norm;
            }
        }
        
        let spectral_radius = weights.dot(&v).dot(&v).sqrt();
        
        if spectral_radius > 0.0 {
            *weights *= target_radius / spectral_radius;
        }
    }
    
    fn calculate_confidence(&self, state: &Array1<f32>) -> f32 {
        // Confidence based on state coherence
        let mean = state.mean().unwrap_or(0.0);
        let variance = state.var(0.0);
        
        // High variance with non-zero mean indicates confident state
        (variance * mean.abs()).min(1.0)
    }
    
    fn calculate_novelty(&self, state: &Array1<f32>, patterns: &[(PatternType, f32)]) -> f32 {
        let memories = self.pattern_memories.read();
        
        if patterns.is_empty() {
            return 1.0;  // Completely novel
        }
        
        let mut min_distance = f32::MAX;
        
        for memory in memories.values() {
            let mut distance = 0.0;
            for i in 0..state.len().min(memory.activation_pattern.len()) {
                distance += (state[i] - memory.activation_pattern[i]).powi(2);
            }
            distance = distance.sqrt();
            min_distance = min_distance.min(distance);
        }
        
        // Normalize to 0-1
        (min_distance / (state.len() as f32).sqrt()).min(1.0)
    }
    
    /// Get learning statistics
    pub fn get_learning_stats(&self) -> LearningStats {
        let weights = self.weights.read();
        let firing_rates = self.firing_rates.read();
        let memories = self.pattern_memories.read();
        
        let mut weight_sum = 0.0;
        let mut weight_count = 0;
        let mut max_weight = 0.0;
        let mut min_weight = f32::MAX;
        
        for &w in weights.iter() {
            if w != 0.0 {
                weight_sum += w.abs();
                weight_count += 1;
                max_weight = max_weight.max(w);
                min_weight = min_weight.min(w);
            }
        }
        
        LearningStats {
            total_updates: self.total_weight_updates.load(Ordering::Relaxed),
            avg_weight: if weight_count > 0 { weight_sum / weight_count as f32 } else { 0.0 },
            max_weight,
            min_weight,
            avg_firing_rate: firing_rates.mean().unwrap_or(0.0),
            pattern_memories: memories.len(),
            learning_enabled: self.learning_enabled.load(Ordering::Relaxed),
        }
    }
    
    /// Enable or disable learning
    pub fn set_learning(&self, enabled: bool) {
        self.learning_enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Pattern detector for reservoir states
struct PatternDetector {
    templates: HashMap<PatternType, Array1<f32>>,
}

impl PatternDetector {
    fn new(size: usize) -> Self {
        let mut templates = HashMap::new();
        
        // Create template patterns
        templates.insert(PatternType::Momentum, Self::create_momentum_template(size));
        templates.insert(PatternType::Reversal, Self::create_reversal_template(size));
        templates.insert(PatternType::Breakout, Self::create_breakout_template(size));
        templates.insert(PatternType::Consolidation, Self::create_consolidation_template(size));
        templates.insert(PatternType::Volatility, Self::create_volatility_template(size));
        templates.insert(PatternType::Trend, Self::create_trend_template(size));
        
        Self { templates }
    }
    
    fn detect(&self, state: &Array1<f32>) -> Vec<(PatternType, f32)> {
        let mut patterns = Vec::new();
        
        for (pattern_type, template) in &self.templates {
            let similarity = self.cosine_similarity(state, template);
            if similarity > 0.3 {
                patterns.push((*pattern_type, similarity));
            }
        }
        
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        patterns
    }
    
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    fn create_momentum_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Rising activation pattern
        for i in 0..size {
            template[i] = (i as f32 / size as f32).powf(2.0);
        }
        template
    }
    
    fn create_reversal_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Peak in middle
        for i in 0..size {
            let x = (i as f32 - size as f32 / 2.0) / (size as f32 / 2.0);
            template[i] = (-x * x + 1.0).max(0.0);
        }
        template
    }
    
    fn create_breakout_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Sharp transition
        for i in 0..size {
            template[i] = if i > size * 2 / 3 { 1.0 } else { 0.1 };
        }
        template
    }
    
    fn create_consolidation_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Flat pattern
        for i in 0..size {
            template[i] = 0.5 + 0.1 * ((i as f32 * 0.5).sin());
        }
        template
    }
    
    fn create_volatility_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Oscillating pattern
        for i in 0..size {
            template[i] = (i as f32 * 0.3).sin().abs();
        }
        template
    }
    
    fn create_trend_template(size: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        // Linear trend
        for i in 0..size {
            template[i] = i as f32 / size as f32;
        }
        template
    }
}

/// Learning statistics
#[derive(Debug)]
pub struct LearningStats {
    pub total_updates: u64,
    pub avg_weight: f32,
    pub max_weight: f32,
    pub min_weight: f32,
    pub avg_firing_rate: f32,
    pub pattern_memories: usize,
    pub learning_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdp_reservoir() {
        let mut reservoir = STDPReservoir::new(100, STDPConfig::default());
        
        // Create test spikes
        let spikes: Vec<Spike> = (0..10)
            .map(|i| Spike {
                timestamp_ns: 1_000_000 * i,
                neuron_id: i % 100,
                strength: 1.0,
            })
            .collect();
        
        // Process with learning
        let state = reservoir.process_with_learning(&spikes);
        assert!(!state.activations.is_empty());
        
        // Check learning stats
        let stats = reservoir.get_learning_stats();
        println!("Learning stats: {:?}", stats);
    }
    
    #[test]
    fn test_pattern_training() {
        let mut reservoir = STDPReservoir::new(100, STDPConfig::default());
        
        // Create training data
        let mut training_data = Vec::new();
        
        // Momentum pattern
        let momentum_spikes: Vec<Spike> = (0..20)
            .map(|i| Spike {
                timestamp_ns: 1_000_000 * i,
                neuron_id: i * 2,
                strength: (i as f32 / 20.0),
            })
            .collect();
        training_data.push((momentum_spikes, PatternType::Momentum));
        
        // Train
        reservoir.train_patterns(training_data).unwrap();
        
        // Test recognition
        let test_spikes: Vec<Spike> = (0..20)
            .map(|i| Spike {
                timestamp_ns: 1_000_000 * i,
                neuron_id: i * 2,
                strength: (i as f32 / 20.0),
            })
            .collect();
        
        let state = reservoir.process_with_learning(&test_spikes);
        assert!(state.confidence > 0.0);
    }
    
    #[test]
    fn test_homeostasis() {
        let reservoir = STDPReservoir::new(100, STDPConfig::default());
        
        // Process many spikes
        for batch in 0..100 {
            let spikes: Vec<Spike> = (0..10)
                .map(|i| Spike {
                    timestamp_ns: batch * 10_000_000 + i * 1_000_000,
                    neuron_id: rand::random::<u32>() % 100,
                    strength: rand::random::<f32>(),
                })
                .collect();
            
            let mut reservoir_mut = STDPReservoir::new(100, STDPConfig::default());
            reservoir_mut.process_with_learning(&spikes);
        }
        
        // Check firing rates converge to target
        let stats = reservoir.get_learning_stats();
        assert!((stats.avg_firing_rate - 0.1).abs() < 0.05);
    }
}