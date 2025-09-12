//! Liquid State Machine for Reservoir Computing
//! 
//! 5000-neuron reservoir with small-world topology for temporal pattern detection

use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::VecDeque;
use anyhow::Result;

use crate::spike_encoding::Spike;

/// Reservoir configuration
pub struct ReservoirConfig {
    pub size: usize,
    pub spectral_radius: f32,
    pub connection_probability: f32,
    pub leak_rate: f32,
}

/// Pattern types detected by the reservoir
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatternType {
    FlashCrashPrecursor,
    LiquidityWithdrawal,
    MomentumIgnition,
    OrderFlowImbalance,
    RegimeTransition,
    InstitutionalFootprint,
    Momentum,
    Reversal,
    Breakout,
    Consolidation,
    Volatility,
}

/// Reservoir state output
pub struct ReservoirState {
    pub activation: DVector<f32>,
    pub confidence: f32,
    pub novelty: f32,
}

/// Readout layer for pattern detection
struct ReadoutLayer {
    weights: DMatrix<f32>,
    target_pattern: PatternType,
    performance: f32,
}

/// Liquid State Machine
pub struct LiquidStateMachine {
    // Reservoir
    weights: DMatrix<f32>,
    state: DVector<f32>,
    bias: DVector<f32>,
    
    // Parameters
    spectral_radius: f32,
    leak_rate: f32,
    input_scaling: f32,
    
    // Readouts
    readout_weights: Vec<DMatrix<f32>>,
    readout_targets: Vec<PatternType>,
    
    // History for training
    state_history: VecDeque<DVector<f32>>,
    
    // Performance metrics
    separation_quality: f32,
    approximation_quality: f32,
}

impl LiquidStateMachine {
    pub fn new(config: ReservoirConfig) -> Self {
        let weights = Self::create_reservoir_weights(&config);
        
        Self {
            weights,
            state: DVector::zeros(config.size),
            bias: DVector::from_fn(config.size, |_, _| {
                rand::thread_rng().gen_range(-0.1..0.1)
            }),
            spectral_radius: config.spectral_radius,
            leak_rate: config.leak_rate,
            input_scaling: 1.0,
            readout_weights: Vec::new(),
            readout_targets: Vec::new(),
            state_history: VecDeque::with_capacity(1000),
            separation_quality: 0.0,
            approximation_quality: 0.0,
        }
    }
    
    fn create_reservoir_weights(config: &ReservoirConfig) -> DMatrix<f32> {
        let mut weights = DMatrix::zeros(config.size, config.size);
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        
        // Initial random connections
        for i in 0..config.size {
            for j in 0..config.size {
                if rng.gen::<f32>() < config.connection_probability {
                    weights[(i, j)] = uniform.sample(&mut rng);
                }
            }
        }
        
        // Rewire for small-world property
        let rewiring_prob = 0.1;
        for i in 0..config.size {
            for j in (i+1)..config.size {
                if weights[(i, j)] != 0.0 && rng.gen::<f32>() < rewiring_prob {
                    let new_target = rng.gen_range(0..config.size);
                    weights[(i, new_target)] = weights[(i, j)];
                    weights[(i, j)] = 0.0;
                }
            }
        }
        
        // Normalize spectral radius
        // Note: Full eigenvalue computation would be done here
        // For now, simple scaling
        weights *= config.spectral_radius;
        
        weights
    }
    
    /// Process input through reservoir
    pub fn process(&mut self, spikes: &[Spike]) -> ReservoirState {
        // Convert spikes to input vector
        let input = self.spikes_to_input(spikes);
        
        // Update reservoir state
        // x(t+1) = (1-α)x(t) + α*tanh(Wx(t) + W_in*u(t) + b)
        let new_state = (1.0 - self.leak_rate) * &self.state
            + self.leak_rate * ((&self.weights * &self.state + input + &self.bias)
                .map(|x| x.tanh()));
        
        self.state = new_state.clone();
        self.state_history.push_back(new_state.clone());
        if self.state_history.len() > 1000 {
            self.state_history.pop_front();
        }
        
        // Compute confidence and novelty
        let confidence = self.compute_confidence();
        let novelty = self.compute_novelty();
        
        ReservoirState {
            activation: self.state.clone(),
            confidence,
            novelty,
        }
    }
    
    fn spikes_to_input(&self, spikes: &[Spike]) -> DVector<f32> {
        let mut input = DVector::zeros(self.state.len());
        
        for spike in spikes {
            let idx = (spike.neuron_id as usize) % self.state.len();
            input[idx] += spike.strength * self.input_scaling;
        }
        
        input
    }
    
    fn compute_confidence(&self) -> f32 {
        // Simple confidence based on state magnitude
        let magnitude = self.state.norm();
        (magnitude / (self.state.len() as f32).sqrt()).min(1.0)
    }
    
    fn compute_novelty(&self) -> f32 {
        if self.state_history.len() < 2 {
            return 1.0;
        }
        
        // Compare with recent states
        let mut min_distance = f32::MAX;
        let current = &self.state;
        
        for past_state in self.state_history.iter().take(self.state_history.len() - 1) {
            let distance = (current - past_state).norm();
            if distance < min_distance {
                min_distance = distance;
            }
        }
        
        // Normalize novelty score
        (min_distance / (self.state.len() as f32).sqrt()).min(1.0)
    }
    
    /// Train readout layer for pattern detection
    pub fn train_readout(&mut self, target: PatternType, training_data: &[(DVector<f32>, f32)]) {
        // Simplified RLS implementation
        let n = self.state.len();
        let mut w = DVector::zeros(n);
        
        for (state, target_value) in training_data {
            let error = target_value - w.dot(state);
            let learning_rate = 0.01;
            w += learning_rate * error * state;
        }
        
        self.readout_weights.push(DMatrix::from_column_slice(1, n, w.as_slice()));
        self.readout_targets.push(target);
    }
    
    /// Detect patterns in current state
    pub fn detect_patterns(&self) -> Vec<(PatternType, f32)> {
        let mut detections = Vec::new();
        
        for (weights, pattern) in self.readout_weights.iter().zip(&self.readout_targets) {
            if weights.ncols() > 0 {
                let activation = (weights * &self.state)[(0, 0)];
                let confidence = activation.tanh().abs();
                
                if confidence > 0.5 {
                    detections.push((pattern.clone(), confidence));
                }
            }
        }
        
        detections
    }
    
    /// Compute separation property
    pub fn compute_separation(&mut self, input1: &[Spike], input2: &[Spike]) -> f32 {
        let state1 = {
            let original_state = self.state.clone();
            self.process(input1);
            let s1 = self.state.clone();
            self.state = original_state;
            s1
        };
        
        let state2 = {
            self.process(input2);
            self.state.clone()
        };
        
        (state1 - state2).norm() / (self.state.len() as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reservoir_creation() {
        let config = ReservoirConfig {
            size: 100,
            spectral_radius: 0.95,
            connection_probability: 0.2,
            leak_rate: 0.1,
        };
        
        let reservoir = LiquidStateMachine::new(config);
        assert_eq!(reservoir.state.len(), 100);
    }
    
    #[test]
    fn test_reservoir_processing() {
        let config = ReservoirConfig {
            size: 100,
            spectral_radius: 0.95,
            connection_probability: 0.2,
            leak_rate: 0.1,
        };
        
        let mut reservoir = LiquidStateMachine::new(config);
        
        let spikes = vec![
            Spike {
                timestamp_ns: 1000,
                neuron_id: 0,
                strength: 1.0,
            },
            Spike {
                timestamp_ns: 1000,
                neuron_id: 1,
                strength: 0.5,
            },
        ];
        
        let state = reservoir.process(&spikes);
        assert!(state.confidence >= 0.0 && state.confidence <= 1.0);
        assert!(state.novelty >= 0.0 && state.novelty <= 1.0);
    }
}