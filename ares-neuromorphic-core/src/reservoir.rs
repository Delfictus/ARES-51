//! Reservoir computing implementation for temporal pattern processing

use crate::types::SpikePattern;
use nalgebra::{DMatrix, DVector};
use anyhow::Result;

/// Reservoir computer for processing spike patterns
#[derive(Debug)]
pub struct ReservoirComputer {
    config: ReservoirConfig,
    weights_input: DMatrix<f64>,
    weights_reservoir: DMatrix<f64>,
    weights_output: DMatrix<f64>,
    state: DVector<f64>,
    previous_state: DVector<f64>,
    statistics: ReservoirStatistics,
}

/// Configuration for the reservoir computer
#[derive(Debug, Clone)]
pub struct ReservoirConfig {
    /// Number of neurons in the reservoir
    pub size: usize,
    /// Number of input neurons
    pub input_size: usize,
    /// Spectral radius of the reservoir matrix
    pub spectral_radius: f64,
    /// Connection probability between neurons
    pub connection_prob: f64,
    /// Leak rate for neuron dynamics
    pub leak_rate: f64,
    /// Input scaling factor
    pub input_scaling: f64,
    /// Noise level for reservoir dynamics
    pub noise_level: f64,
    /// Enable plasticity (STDP-like)
    pub enable_plasticity: bool,
}

impl Default for ReservoirConfig {
    fn default() -> Self {
        Self {
            size: 1000,
            input_size: 100,
            spectral_radius: 0.95,
            connection_prob: 0.1,
            leak_rate: 0.3,
            input_scaling: 1.0,
            noise_level: 0.01,
            enable_plasticity: false,
        }
    }
}

/// Current state of the reservoir
#[derive(Debug, Clone)]
pub struct ReservoirState {
    /// Current activation levels
    pub activations: Vec<f64>,
    /// Average activation level
    pub average_activation: f32,
    /// Maximum activation level
    pub max_activation: f32,
    /// Number of spikes processed
    pub last_spike_count: usize,
    /// Temporal dynamics measures
    pub dynamics: DynamicsMetrics,
}

/// Temporal dynamics measurements
#[derive(Debug, Clone)]
pub struct DynamicsMetrics {
    /// Memory capacity (how long information persists)
    pub memory_capacity: f64,
    /// Separation property (how well different inputs are distinguished)
    pub separation: f64,
    /// Approximation property (how well functions can be approximated)
    pub approximation: f64,
}

impl Default for DynamicsMetrics {
    fn default() -> Self {
        Self {
            memory_capacity: 0.0,
            separation: 0.0,
            approximation: 0.0,
        }
    }
}

/// Reservoir statistics for monitoring
#[derive(Debug, Default)]
struct ReservoirStatistics {
    patterns_processed: u64,
    total_spikes_processed: u64,
    average_activation: f64,
    max_activation_seen: f64,
}

impl ReservoirComputer {
    /// Create a new reservoir computer
    pub fn new(
        reservoir_size: usize,
        input_size: usize,
        spectral_radius: f64,
        connection_prob: f64,
        leak_rate: f64,
    ) -> Result<Self> {
        let config = ReservoirConfig {
            size: reservoir_size,
            input_size,
            spectral_radius,
            connection_prob,
            leak_rate,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Create reservoir with custom configuration
    pub fn with_config(config: ReservoirConfig) -> Result<Self> {
        if config.size == 0 {
            return Err(anyhow::anyhow!("Reservoir size must be greater than 0"));
        }
        if config.input_size == 0 {
            return Err(anyhow::anyhow!("Input size must be greater than 0"));
        }
        if config.spectral_radius <= 0.0 || config.spectral_radius >= 1.0 {
            return Err(anyhow::anyhow!("Spectral radius must be between 0 and 1"));
        }
        
        // Initialize weight matrices
        let weights_input = Self::generate_input_weights(&config);
        let weights_reservoir = Self::generate_reservoir_weights(&config)?;
        let weights_output = DMatrix::zeros(10, config.size); // 10 output classes
        
        // Initialize state vectors
        let state = DVector::zeros(config.size);
        let previous_state = DVector::zeros(config.size);
        
        Ok(Self {
            config,
            weights_input,
            weights_reservoir,
            weights_output,
            state,
            previous_state,
            statistics: ReservoirStatistics::default(),
        })
    }
    
    /// Process a spike pattern through the reservoir
    pub fn process(&mut self, pattern: &SpikePattern) -> Result<ReservoirState> {
        // Convert spike pattern to input vector
        let input_vector = self.pattern_to_input(pattern);
        
        // Update reservoir state
        self.update_state(&input_vector)?;
        
        // Calculate metrics
        let dynamics = self.calculate_dynamics();
        
        // Create state snapshot
        let reservoir_state = ReservoirState {
            activations: self.state.iter().cloned().collect(),
            average_activation: (self.state.mean()) as f32,
            max_activation: self.state.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as f32,
            last_spike_count: pattern.spike_count(),
            dynamics,
        };
        
        // Update statistics
        self.update_statistics(&reservoir_state);
        
        Ok(reservoir_state)
    }
    
    /// Reset reservoir state
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.previous_state.fill(0.0);
        self.statistics = ReservoirStatistics::default();
    }
    
    /// Get reservoir statistics
    pub fn get_statistics(&self) -> &ReservoirStatistics {
        &self.statistics
    }
    
    /// Generate input weight matrix
    fn generate_input_weights(config: &ReservoirConfig) -> DMatrix<f64> {
        let mut rng = rand::thread_rng();
        let mut weights = DMatrix::zeros(config.size, config.input_size);
        
        // Sparse random input weights
        for i in 0..config.size {
            for j in 0..config.input_size {
                if rand::Rng::gen::<f64>(&mut rng) < config.connection_prob {
                    weights[(i, j)] = (rand::Rng::gen::<f64>(&mut rng) * 2.0 - 1.0) * config.input_scaling;
                }
            }
        }
        
        weights
    }
    
    /// Generate reservoir weight matrix with specified spectral radius
    fn generate_reservoir_weights(config: &ReservoirConfig) -> Result<DMatrix<f64>> {
        let mut rng = rand::thread_rng();
        let mut weights = DMatrix::zeros(config.size, config.size);
        
        // Generate sparse random matrix
        for i in 0..config.size {
            for j in 0..config.size {
                if i != j && rand::Rng::gen::<f64>(&mut rng) < config.connection_prob {
                    weights[(i, j)] = rand::Rng::gen::<f64>(&mut rng) * 2.0 - 1.0;
                }
            }
        }
        
        // Scale to desired spectral radius
        if let Some(eigenvalues) = weights.eigenvalues() {
            let max_eigenvalue = eigenvalues.iter()
                .map(|c| c.abs())
                .fold(0.0, f64::max);
            
            if max_eigenvalue > 0.0 {
                weights *= config.spectral_radius / max_eigenvalue;
            }
        }
        
        Ok(weights)
    }
    
    /// Convert spike pattern to input vector
    fn pattern_to_input(&self, pattern: &SpikePattern) -> DVector<f64> {
        let mut input = DVector::zeros(self.config.input_size);
        
        // Create temporal bins
        let bin_duration = pattern.duration_ms / self.config.input_size as f64;
        
        // Bin spikes by time
        for spike in &pattern.spikes {
            let bin_index = ((spike.time_ms / bin_duration) as usize).min(self.config.input_size - 1);
            input[bin_index] += 1.0;
        }
        
        // Normalize by total spikes
        if pattern.spike_count() > 0 {
            input /= pattern.spike_count() as f64;
        }
        
        input
    }
    
    /// Update reservoir state based on input
    fn update_state(&mut self, input: &DVector<f64>) -> Result<()> {
        // Store previous state
        self.previous_state.copy_from(&self.state);
        
        // Compute input contribution
        let input_contribution = &self.weights_input * input;
        
        // Compute recurrent contribution
        let recurrent_contribution = &self.weights_reservoir * &self.previous_state;
        
        // Add noise if enabled
        let noise = if self.config.noise_level > 0.0 {
            let mut rng = rand::thread_rng();
            DVector::from_fn(self.config.size, |_, _| {
                (rand::Rng::gen::<f64>(&mut rng) * 2.0 - 1.0) * self.config.noise_level
            })
        } else {
            DVector::zeros(self.config.size)
        };
        
        // Leaky integrator dynamics
        for i in 0..self.config.size {
            let new_activation = (1.0 - self.config.leak_rate) * self.previous_state[i]
                + self.config.leak_rate * (
                    input_contribution[i] + recurrent_contribution[i] + noise[i]
                ).tanh();
            
            self.state[i] = new_activation;
        }
        
        // Apply plasticity if enabled
        if self.config.enable_plasticity {
            self.apply_plasticity(input);
        }
        
        Ok(())
    }
    
    /// Apply spike-timing dependent plasticity (STDP)
    fn apply_plasticity(&mut self, _input: &DVector<f64>) {
        // Simplified STDP implementation
        let learning_rate = 0.001;
        
        // Update weights based on correlation between pre- and post-synaptic activity
        for i in 0..self.config.size {
            for j in 0..self.config.size {
                if i != j {
                    let correlation = self.state[i] * self.previous_state[j];
                    self.weights_reservoir[(i, j)] += learning_rate * correlation;
                }
            }
        }
        
        // Maintain spectral radius constraint
        if let Some(eigenvalues) = self.weights_reservoir.eigenvalues() {
            let max_eigenvalue = eigenvalues.iter()
                .map(|c| c.abs())
                .fold(0.0, f64::max);
            
            if max_eigenvalue > self.config.spectral_radius {
                self.weights_reservoir *= self.config.spectral_radius / max_eigenvalue;
            }
        }
    }
    
    /// Calculate reservoir dynamics metrics
    fn calculate_dynamics(&self) -> DynamicsMetrics {
        // Memory capacity: measure how long information persists
        let memory_capacity = self.calculate_memory_capacity();
        
        // Separation: measure how well different inputs are distinguished
        let separation = self.calculate_separation();
        
        // Approximation: measure computational capability
        let approximation = self.calculate_approximation();
        
        DynamicsMetrics {
            memory_capacity,
            separation,
            approximation,
        }
    }
    
    /// Calculate memory capacity
    fn calculate_memory_capacity(&self) -> f64 {
        // Simplified memory capacity calculation
        // In practice, this would require multiple time steps
        let correlation = self.state.dot(&self.previous_state) 
            / (self.state.norm() * self.previous_state.norm());
        correlation.abs()
    }
    
    /// Calculate separation property
    fn calculate_separation(&self) -> f64 {
        // Measure of state space dimensionality
        let variance = self.state.variance();
        variance.min(1.0)
    }
    
    /// Calculate approximation property
    fn calculate_approximation(&self) -> f64 {
        // Rank approximation of reservoir matrix
        let rank_estimate = self.state.iter()
            .filter(|&&x| x.abs() > 0.01)
            .count() as f64 / self.config.size as f64;
        rank_estimate
    }
    
    /// Update internal statistics
    fn update_statistics(&mut self, state: &ReservoirState) {
        self.statistics.patterns_processed += 1;
        self.statistics.total_spikes_processed += state.last_spike_count as u64;
        
        // Update moving averages
        let alpha = 0.1; // Exponential moving average factor
        self.statistics.average_activation = alpha * state.average_activation as f64
            + (1.0 - alpha) * self.statistics.average_activation;
        
        if state.max_activation as f64 > self.statistics.max_activation_seen {
            self.statistics.max_activation_seen = state.max_activation as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Spike, PatternMetadata};

    #[test]
    fn test_reservoir_creation() {
        let reservoir = ReservoirComputer::new(100, 10, 0.9, 0.1, 0.3);
        assert!(reservoir.is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let result = ReservoirComputer::new(0, 10, 0.9, 0.1, 0.3);
        assert!(result.is_err());
        
        let result = ReservoirComputer::new(100, 0, 0.9, 0.1, 0.3);
        assert!(result.is_err());
        
        let result = ReservoirComputer::new(100, 10, 1.5, 0.1, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_spike_pattern_processing() {
        let mut reservoir = ReservoirComputer::new(50, 10, 0.9, 0.1, 0.3).unwrap();
        
        let spikes = vec![
            Spike::new(0, 10.0),
            Spike::new(1, 20.0),
            Spike::new(2, 30.0),
        ];
        let pattern = SpikePattern::new(spikes, 100.0);
        
        let result = reservoir.process(&pattern);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        assert_eq!(state.last_spike_count, 3);
        assert!(!state.activations.is_empty());
    }

    #[test]
    fn test_reservoir_reset() {
        let mut reservoir = ReservoirComputer::new(50, 10, 0.9, 0.1, 0.3).unwrap();
        
        // Process some data
        let spikes = vec![Spike::new(0, 10.0)];
        let pattern = SpikePattern::new(spikes, 100.0);
        reservoir.process(&pattern).unwrap();
        
        // Reset and check state is cleared
        reservoir.reset();
        assert!(reservoir.state.iter().all(|&x| x == 0.0));
        assert_eq!(reservoir.statistics.patterns_processed, 0);
    }

    #[test]
    fn test_dynamics_calculation() {
        let mut reservoir = ReservoirComputer::new(30, 5, 0.9, 0.1, 0.3).unwrap();
        
        let spikes = vec![
            Spike::new(0, 10.0),
            Spike::new(1, 20.0),
        ];
        let pattern = SpikePattern::new(spikes, 50.0);
        
        let state = reservoir.process(&pattern).unwrap();
        
        // Check that dynamics metrics are calculated
        assert!(state.dynamics.memory_capacity >= 0.0 && state.dynamics.memory_capacity <= 1.0);
        assert!(state.dynamics.separation >= 0.0 && state.dynamics.separation <= 1.0);
        assert!(state.dynamics.approximation >= 0.0 && state.dynamics.approximation <= 1.0);
    }
}