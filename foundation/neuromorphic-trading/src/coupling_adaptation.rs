//! PHASE 2C.1: Neural Oscillator Coupling Strength Adaptation
//! Revolutionary adaptive coupling for optimal market synchronization
//! Implements STDP-inspired plasticity rules for oscillator networks

use crate::drpp::{DynamicResonancePatternProcessor, DrppState, Pattern, PatternType};
use crate::phase_coherence::{PhaseCoherenceAnalyzer, CoherencePattern, MarketRegime};
use crate::multi_timeframe::{MultiTimeframeResult, TimeHorizon, CrossTimeInfo};
use crate::spike_encoding::Spike;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use anyhow::{Result, anyhow};
use tokio::time::{Duration, Instant};
use rand::prelude::*;

/// Adaptive coupling strength manager for neural oscillator networks
pub struct CouplingAdaptationEngine {
    /// Current coupling matrix (symmetric)
    coupling_matrix: Arc<RwLock<Vec<Vec<f64>>>>,
    /// Number of oscillators
    n_oscillators: usize,
    /// Adaptation learning rate
    learning_rate: f64,
    /// Coupling strength bounds
    min_coupling: f64,
    max_coupling: f64,
    /// Historical performance tracking
    performance_history: VecDeque<CouplingPerformance>,
    /// Adaptation strategy
    strategy: AdaptationStrategy,
    /// Market regime-specific coupling presets
    regime_presets: HashMap<MarketRegime, CouplingPreset>,
    /// Plasticity rules configuration
    plasticity_config: PlasticityConfig,
    /// Spike timing dependent plasticity (STDP) window
    stdp_window_ms: f64,
    /// Last adaptation timestamp
    last_adaptation: Instant,
    /// Minimum adaptation interval
    adaptation_interval: Duration,
}

/// Coupling performance metrics for adaptation feedback
#[derive(Debug, Clone)]
pub struct CouplingPerformance {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Pattern detection accuracy
    pub pattern_accuracy: f64,
    /// Network coherence score
    pub coherence_score: f64,
    /// Synchronization efficiency
    pub sync_efficiency: f64,
    /// Information flow rate
    pub info_flow_rate: f64,
    /// Coupling energy (total coupling strength)
    pub coupling_energy: f64,
    /// Market regime at time of measurement
    pub market_regime: MarketRegime,
}

/// Adaptation strategies for different market conditions
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Hebbian learning: strengthen connections between co-active oscillators
    Hebbian,
    /// Anti-Hebbian: weaken strong connections to prevent over-synchronization
    AntiHebbian,
    /// STDP-based: spike timing dependent plasticity
    SpikeTiming,
    /// Homeostatic: maintain target activity level
    Homeostatic,
    /// Market-aware: adapt based on market regime
    MarketAware,
    /// Hybrid: combination of multiple strategies
    Hybrid(Vec<AdaptationStrategy>),
}

/// Market regime-specific coupling configuration
#[derive(Debug, Clone)]
pub struct CouplingPreset {
    /// Base coupling strength
    pub base_strength: f64,
    /// Coupling variability (standard deviation)
    pub variability: f64,
    /// Long-range connection probability
    pub long_range_prob: f64,
    /// Small-world rewiring probability
    pub rewiring_prob: f64,
    /// Target synchronization level
    pub target_sync: f64,
}

/// Plasticity rule configuration
#[derive(Debug, Clone)]
pub struct PlasticityConfig {
    /// Long-term potentiation rate
    pub ltp_rate: f64,
    /// Long-term depression rate
    pub ltd_rate: f64,
    /// STDP time constant (milliseconds)
    pub stdp_tau_ms: f64,
    /// Homeostatic target firing rate
    pub target_rate: f64,
    /// Metaplasticity threshold
    pub meta_threshold: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            ltp_rate: 0.01,
            ltd_rate: 0.005,
            stdp_tau_ms: 20.0,
            target_rate: 10.0,
            meta_threshold: 0.8,
        }
    }
}

impl CouplingAdaptationEngine {
    /// Create new coupling adaptation engine
    pub fn new(n_oscillators: usize, learning_rate: f64, strategy: AdaptationStrategy) -> Self {
        // Initialize coupling matrix with small-world topology
        let coupling_matrix = Arc::new(RwLock::new(
            Self::initialize_small_world_coupling(n_oscillators, 0.2, 0.1)
        ));

        // Market regime presets
        let mut regime_presets = HashMap::new();
        
        regime_presets.insert(MarketRegime::Trending, CouplingPreset {
            base_strength: 0.4,
            variability: 0.1,
            long_range_prob: 0.3,
            rewiring_prob: 0.15,
            target_sync: 0.7,
        });
        
        regime_presets.insert(MarketRegime::Ranging, CouplingPreset {
            base_strength: 0.3,
            variability: 0.2,
            long_range_prob: 0.2,
            rewiring_prob: 0.25,
            target_sync: 0.5,
        });
        
        regime_presets.insert(MarketRegime::Chaotic, CouplingPreset {
            base_strength: 0.1,
            variability: 0.3,
            long_range_prob: 0.1,
            rewiring_prob: 0.4,
            target_sync: 0.2,
        });
        
        regime_presets.insert(MarketRegime::Transitional, CouplingPreset {
            base_strength: 0.35,
            variability: 0.25,
            long_range_prob: 0.25,
            rewiring_prob: 0.3,
            target_sync: 0.4,
        });
        
        regime_presets.insert(MarketRegime::RegimeShift, CouplingPreset {
            base_strength: 0.5,
            variability: 0.15,
            long_range_prob: 0.4,
            rewiring_prob: 0.1,
            target_sync: 0.8,
        });

        Self {
            coupling_matrix,
            n_oscillators,
            learning_rate,
            min_coupling: 0.01,
            max_coupling: 1.0,
            performance_history: VecDeque::with_capacity(1000),
            strategy,
            regime_presets,
            plasticity_config: PlasticityConfig::default(),
            stdp_window_ms: 50.0,
            last_adaptation: Instant::now(),
            adaptation_interval: Duration::from_millis(100), // Adapt every 100ms
        }
    }

    /// Initialize small-world coupling topology
    fn initialize_small_world_coupling(n: usize, base_strength: f64, rewiring_prob: f64) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; n]; n];
        let mut rng = rand::thread_rng();

        // Start with ring lattice (each oscillator connected to k=4 nearest neighbors)
        let k = 4;
        for i in 0..n {
            for j in 1..=k/2 {
                let neighbor1 = (i + j) % n;
                let neighbor2 = (i + n - j) % n;
                
                matrix[i][neighbor1] = base_strength * (0.5 + rng.gen::<f64>() * 0.5);
                matrix[neighbor1][i] = matrix[i][neighbor1]; // Symmetric
                
                matrix[i][neighbor2] = base_strength * (0.5 + rng.gen::<f64>() * 0.5);
                matrix[neighbor2][i] = matrix[i][neighbor2]; // Symmetric
            }
        }

        // Rewire connections with probability p to create small-world topology
        for i in 0..n {
            for j in i+1..n {
                if matrix[i][j] > 0.0 && rng.gen::<f64>() < rewiring_prob {
                    // Disconnect current connection
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 0.0;
                    
                    // Create random long-range connection
                    let mut new_target = rng.gen_range(0..n);
                    while new_target == i || matrix[i][new_target] > 0.0 {
                        new_target = rng.gen_range(0..n);
                    }
                    
                    matrix[i][new_target] = base_strength * (0.5 + rng.gen::<f64>() * 0.5);
                    matrix[new_target][i] = matrix[i][new_target];
                }
            }
        }

        matrix
    }

    /// Adapt coupling strengths based on network performance and market conditions
    pub async fn adapt_coupling(
        &mut self,
        drpp_state: &DrppState,
        coherence_patterns: &[CoherencePattern],
        multi_timeframe_result: &MultiTimeframeResult,
        spikes: &[Spike],
    ) -> Result<()> {
        // Rate limit adaptations
        if self.last_adaptation.elapsed() < self.adaptation_interval {
            return Ok(());
        }

        // Calculate current performance metrics
        let performance = self.calculate_performance_metrics(
            drpp_state,
            coherence_patterns,
            multi_timeframe_result,
        ).await?;

        // Add to history
        self.performance_history.push_back(performance.clone());
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // Apply adaptation strategy
        match &self.strategy {
            AdaptationStrategy::Hebbian => {
                self.apply_hebbian_adaptation(drpp_state, &performance).await?;
            },
            AdaptationStrategy::AntiHebbian => {
                self.apply_anti_hebbian_adaptation(drpp_state, &performance).await?;
            },
            AdaptationStrategy::SpikeTiming => {
                self.apply_stdp_adaptation(spikes, &performance).await?;
            },
            AdaptationStrategy::Homeostatic => {
                self.apply_homeostatic_adaptation(drpp_state, &performance).await?;
            },
            AdaptationStrategy::MarketAware => {
                self.apply_market_aware_adaptation(&performance).await?;
            },
            AdaptationStrategy::Hybrid(strategies) => {
                self.apply_hybrid_adaptation(strategies, drpp_state, spikes, &performance).await?;
            },
        }

        self.last_adaptation = Instant::now();
        Ok(())
    }

    /// Calculate comprehensive performance metrics
    async fn calculate_performance_metrics(
        &self,
        drpp_state: &DrppState,
        coherence_patterns: &[CoherencePattern],
        multi_timeframe_result: &MultiTimeframeResult,
    ) -> Result<CouplingPerformance> {
        // Pattern detection accuracy (based on coherence thresholds)
        let pattern_accuracy = if drpp_state.patterns.is_empty() {
            0.0
        } else {
            drpp_state.patterns.iter()
                .map(|p| if p.strength > 0.6 { 1.0 } else { 0.0 })
                .sum::<f64>() / drpp_state.patterns.len() as f64
        };

        // Network coherence score
        let coherence_score = if coherence_patterns.is_empty() {
            0.0
        } else {
            coherence_patterns.iter()
                .map(|p| p.coherence_score)
                .sum::<f64>() / coherence_patterns.len() as f64
        };

        // Synchronization efficiency (Kuramoto order parameter)
        let sync_efficiency = drpp_state.coherence;

        // Information flow rate (from multi-timeframe analysis)
        let info_flow_rate = multi_timeframe_result.cross_time_flows.values()
            .map(|flow| flow.mutual_information)
            .fold(0.0, f64::max);

        // Coupling energy (total coupling strength)
        let coupling_energy = {
            let matrix = self.coupling_matrix.read();
            let mut total = 0.0;
            for i in 0..self.n_oscillators {
                for j in i+1..self.n_oscillators {
                    total += matrix[i][j];
                }
            }
            total / (self.n_oscillators * (self.n_oscillators - 1) / 2) as f64
        };

        // Market regime (from global sync state)
        let market_regime = multi_timeframe_result.global_sync_state.dominant_regime;

        Ok(CouplingPerformance {
            timestamp: Instant::now(),
            pattern_accuracy,
            coherence_score,
            sync_efficiency,
            info_flow_rate,
            coupling_energy,
            market_regime,
        })
    }

    /// Apply Hebbian learning: strengthen connections between co-active oscillators
    async fn apply_hebbian_adaptation(&mut self, drpp_state: &DrppState, performance: &CouplingPerformance) -> Result<()> {
        let mut matrix = self.coupling_matrix.write();
        let phases = &drpp_state.oscillator_phases;

        if phases.len() < self.n_oscillators {
            return Ok(());
        }

        // Hebbian rule: Δwᵢⱼ = η * activity_i * activity_j
        for i in 0..self.n_oscillators {
            for j in i+1..self.n_oscillators {
                let activity_i = (phases[i].sin() + 1.0) / 2.0; // Normalize to [0,1]
                let activity_j = (phases[j].sin() + 1.0) / 2.0;
                
                let correlation = activity_i * activity_j;
                let delta = self.learning_rate * correlation * performance.pattern_accuracy;
                
                matrix[i][j] = (matrix[i][j] + delta).clamp(self.min_coupling, self.max_coupling);
                matrix[j][i] = matrix[i][j]; // Keep symmetric
            }
        }

        Ok(())
    }

    /// Apply anti-Hebbian learning: weaken strong connections to prevent over-synchronization
    async fn apply_anti_hebbian_adaptation(&mut self, drpp_state: &DrppState, performance: &CouplingPerformance) -> Result<()> {
        let mut matrix = self.coupling_matrix.write();
        let phases = &drpp_state.oscillator_phases;

        if phases.len() < self.n_oscillators {
            return Ok(());
        }

        // Anti-Hebbian rule: weaken strong connections when over-synchronized
        let over_sync_threshold = 0.8;
        if performance.sync_efficiency > over_sync_threshold {
            for i in 0..self.n_oscillators {
                for j in i+1..self.n_oscillators {
                    let phase_diff = (phases[i] - phases[j]).abs();
                    let synchrony = 1.0 - (phase_diff / std::f64::consts::PI).min(1.0);
                    
                    if synchrony > over_sync_threshold {
                        let delta = -self.learning_rate * synchrony * 0.5;
                        matrix[i][j] = (matrix[i][j] + delta).clamp(self.min_coupling, self.max_coupling);
                        matrix[j][i] = matrix[i][j];
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply spike timing dependent plasticity (STDP)
    async fn apply_stdp_adaptation(&mut self, spikes: &[Spike], performance: &CouplingPerformance) -> Result<()> {
        if spikes.len() < 2 {
            return Ok(());
        }

        let mut matrix = self.coupling_matrix.write();
        let window_ns = (self.stdp_window_ms * 1_000_000.0) as u64;

        // Group spikes by neuron for temporal analysis
        let mut neuron_spikes: HashMap<u32, Vec<&Spike>> = HashMap::new();
        for spike in spikes {
            neuron_spikes.entry(spike.neuron_id).or_default().push(spike);
        }

        // Apply STDP rule between oscillator pairs
        for (&pre_neuron, pre_spikes) in &neuron_spikes {
            for (&post_neuron, post_spikes) in &neuron_spikes {
                if pre_neuron >= post_neuron || pre_neuron as usize >= self.n_oscillators || post_neuron as usize >= self.n_oscillators {
                    continue;
                }

                let pre_idx = pre_neuron as usize;
                let post_idx = post_neuron as usize;

                // Find spike pairs within STDP window
                for pre_spike in pre_spikes {
                    for post_spike in post_spikes {
                        let dt = if post_spike.timestamp_ns > pre_spike.timestamp_ns {
                            post_spike.timestamp_ns - pre_spike.timestamp_ns
                        } else {
                            continue; // Only consider pre->post timing
                        };

                        if dt <= window_ns {
                            let dt_ms = dt as f64 / 1_000_000.0;
                            
                            // STDP kernel: exponential decay
                            let stdp_weight = (-dt_ms / self.plasticity_config.stdp_tau_ms).exp();
                            
                            // LTP: strengthen connection
                            let delta = self.plasticity_config.ltp_rate * stdp_weight * 
                                       pre_spike.strength as f64 * post_spike.strength as f64 *
                                       performance.pattern_accuracy;
                            
                            matrix[pre_idx][post_idx] = (matrix[pre_idx][post_idx] + delta)
                                .clamp(self.min_coupling, self.max_coupling);
                            matrix[post_idx][pre_idx] = matrix[pre_idx][post_idx];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply homeostatic adaptation to maintain target activity level
    async fn apply_homeostatic_adaptation(&mut self, drpp_state: &DrppState, performance: &CouplingPerformance) -> Result<()> {
        let mut matrix = self.coupling_matrix.write();
        let phases = &drpp_state.oscillator_phases;

        if phases.len() < self.n_oscillators {
            return Ok(());
        }

        // Calculate current activity levels
        let mut activities = vec![0.0; self.n_oscillators];
        for (i, &phase) in phases.iter().enumerate() {
            if i >= self.n_oscillators { break; }
            activities[i] = (phase.sin() + 1.0) / 2.0;
        }

        let target_activity = self.plasticity_config.target_rate / 20.0; // Normalized

        // Homeostatic scaling: adjust input coupling to maintain target activity
        for i in 0..self.n_oscillators {
            let activity_error = activities[i] - target_activity;
            let scaling_factor = 1.0 - self.learning_rate * activity_error * performance.coherence_score;

            for j in 0..self.n_oscillators {
                if i != j {
                    matrix[i][j] *= scaling_factor;
                    matrix[i][j] = matrix[i][j].clamp(self.min_coupling, self.max_coupling);
                }
            }
        }

        Ok(())
    }

    /// Apply market regime-aware adaptation
    async fn apply_market_aware_adaptation(&mut self, performance: &CouplingPerformance) -> Result<()> {
        let preset = self.regime_presets.get(&performance.market_regime)
            .cloned()
            .unwrap_or_else(|| CouplingPreset {
                base_strength: 0.3,
                variability: 0.2,
                long_range_prob: 0.2,
                rewiring_prob: 0.2,
                target_sync: 0.5,
            });

        let mut matrix = self.coupling_matrix.write();
        let mut rng = rand::thread_rng();

        // Gradually adapt towards regime-specific coupling pattern
        let adaptation_rate = 0.1; // Slow adaptation to avoid instability
        
        for i in 0..self.n_oscillators {
            for j in i+1..self.n_oscillators {
                let distance = ((i as f64 - j as f64).abs() / self.n_oscillators as f64).min(1.0);
                
                // Target coupling based on distance and regime
                let target_coupling = if distance < 0.1 {
                    // Local connections
                    preset.base_strength * (1.0 + rng.gen::<f64>() * preset.variability - preset.variability/2.0)
                } else if rng.gen::<f64>() < preset.long_range_prob {
                    // Long-range connections
                    preset.base_strength * 0.5 * (1.0 + rng.gen::<f64>() * preset.variability - preset.variability/2.0)
                } else {
                    0.0
                };

                // Gradual adaptation
                let current = matrix[i][j];
                let new_coupling = current + adaptation_rate * (target_coupling - current);
                
                matrix[i][j] = new_coupling.clamp(self.min_coupling, self.max_coupling);
                matrix[j][i] = matrix[i][j];
            }
        }

        Ok(())
    }

    /// Apply hybrid adaptation strategy
    async fn apply_hybrid_adaptation(
        &mut self,
        strategies: &[AdaptationStrategy],
        drpp_state: &DrppState,
        spikes: &[Spike],
        performance: &CouplingPerformance,
    ) -> Result<()> {
        // Apply each strategy with reduced learning rate
        let original_rate = self.learning_rate;
        self.learning_rate /= strategies.len() as f64;

        for strategy in strategies {
            match strategy {
                AdaptationStrategy::Hebbian => {
                    self.apply_hebbian_adaptation(drpp_state, performance).await?;
                },
                AdaptationStrategy::AntiHebbian => {
                    self.apply_anti_hebbian_adaptation(drpp_state, performance).await?;
                },
                AdaptationStrategy::SpikeTiming => {
                    self.apply_stdp_adaptation(spikes, performance).await?;
                },
                AdaptationStrategy::Homeostatic => {
                    self.apply_homeostatic_adaptation(drpp_state, performance).await?;
                },
                AdaptationStrategy::MarketAware => {
                    self.apply_market_aware_adaptation(performance).await?;
                },
                AdaptationStrategy::Hybrid(_) => {
                    // Prevent infinite recursion
                    continue;
                }
            }
        }

        self.learning_rate = original_rate;
        Ok(())
    }

    /// Get current coupling matrix
    pub fn get_coupling_matrix(&self) -> Vec<Vec<f64>> {
        self.coupling_matrix.read().clone()
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> Vec<CouplingPerformance> {
        self.performance_history.iter().cloned().collect()
    }

    /// Calculate network topology metrics
    pub fn calculate_network_metrics(&self) -> NetworkMetrics {
        let matrix = self.coupling_matrix.read();
        
        // Calculate clustering coefficient
        let clustering_coeff = self.calculate_clustering_coefficient(&matrix);
        
        // Calculate path length (simplified)
        let avg_path_length = self.calculate_average_path_length(&matrix);
        
        // Small-worldness index
        let small_worldness = if avg_path_length > 0.0 {
            clustering_coeff / avg_path_length
        } else {
            0.0
        };

        // Connection density
        let mut total_connections = 0;
        for i in 0..self.n_oscillators {
            for j in i+1..self.n_oscillators {
                if matrix[i][j] > 0.0 {
                    total_connections += 1;
                }
            }
        }
        let max_connections = self.n_oscillators * (self.n_oscillators - 1) / 2;
        let connection_density = total_connections as f64 / max_connections as f64;

        NetworkMetrics {
            clustering_coefficient: clustering_coeff,
            average_path_length: avg_path_length,
            small_worldness: small_worldness,
            connection_density,
        }
    }

    /// Calculate clustering coefficient
    fn calculate_clustering_coefficient(&self, matrix: &[Vec<f64>]) -> f64 {
        let mut total_clustering = 0.0;
        
        for i in 0..self.n_oscillators {
            let mut neighbors = Vec::new();
            
            // Find neighbors of node i
            for j in 0..self.n_oscillators {
                if i != j && matrix[i][j] > 0.0 {
                    neighbors.push(j);
                }
            }
            
            if neighbors.len() < 2 {
                continue; // Need at least 2 neighbors for clustering
            }
            
            // Count triangles
            let mut triangles = 0;
            for k1 in 0..neighbors.len() {
                for k2 in (k1+1)..neighbors.len() {
                    let n1 = neighbors[k1];
                    let n2 = neighbors[k2];
                    if matrix[n1][n2] > 0.0 {
                        triangles += 1;
                    }
                }
            }
            
            let max_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
            if max_triangles > 0 {
                total_clustering += triangles as f64 / max_triangles as f64;
            }
        }
        
        total_clustering / self.n_oscillators as f64
    }

    /// Calculate average path length (simplified BFS)
    fn calculate_average_path_length(&self, matrix: &[Vec<f64>]) -> f64 {
        let mut total_path_length = 0.0;
        let mut path_count = 0;

        // Simplified: only calculate for a sample of node pairs
        let sample_size = (self.n_oscillators / 4).max(10);
        
        for i in 0..sample_size.min(self.n_oscillators) {
            for j in (i+1)..sample_size.min(self.n_oscillators) {
                if let Some(path_length) = self.shortest_path(matrix, i, j) {
                    total_path_length += path_length as f64;
                    path_count += 1;
                }
            }
        }

        if path_count > 0 {
            total_path_length / path_count as f64
        } else {
            0.0
        }
    }

    /// Find shortest path between two nodes (BFS)
    fn shortest_path(&self, matrix: &[Vec<f64>], start: usize, end: usize) -> Option<usize> {
        if start == end {
            return Some(0);
        }

        let mut visited = vec![false; self.n_oscillators];
        let mut queue = VecDeque::new();
        let mut distances = vec![0; self.n_oscillators];

        queue.push_back(start);
        visited[start] = true;
        distances[start] = 0;

        while let Some(current) = queue.pop_front() {
            for neighbor in 0..self.n_oscillators {
                if matrix[current][neighbor] > 0.0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    distances[neighbor] = distances[current] + 1;
                    queue.push_back(neighbor);

                    if neighbor == end {
                        return Some(distances[neighbor]);
                    }
                }
            }
        }

        None // No path found
    }
}

/// Network topology metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub clustering_coefficient: f64,
    pub average_path_length: f64,
    pub small_worldness: f64,
    pub connection_density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::ResonancePatternType;

    #[tokio::test]
    async fn test_coupling_adaptation_engine_creation() {
        let engine = CouplingAdaptationEngine::new(
            64,
            0.01,
            AdaptationStrategy::Hebbian,
        );
        
        assert_eq!(engine.n_oscillators, 64);
        assert_eq!(engine.learning_rate, 0.01);
        
        let matrix = engine.get_coupling_matrix();
        assert_eq!(matrix.len(), 64);
        assert_eq!(matrix[0].len(), 64);
    }

    #[test]
    fn test_small_world_initialization() {
        let matrix = CouplingAdaptationEngine::initialize_small_world_coupling(20, 0.3, 0.1);
        
        // Check symmetry
        for i in 0..20 {
            for j in 0..20 {
                assert_eq!(matrix[i][j], matrix[j][i]);
            }
        }

        // Check diagonal is zero
        for i in 0..20 {
            assert_eq!(matrix[i][i], 0.0);
        }
    }

    #[tokio::test]
    async fn test_performance_metrics_calculation() {
        let engine = CouplingAdaptationEngine::new(16, 0.01, AdaptationStrategy::Hebbian);
        
        let drpp_state = DrppState {
            patterns: vec![],
            oscillator_phases: vec![0.0; 16],
            coherence: 0.5,
            novelty: 0.3,
        };
        
        // This would require more setup for full test
        // Just testing the structure compiles
        assert_eq!(engine.n_oscillators, 16);
    }

    #[test]
    fn test_network_metrics() {
        let engine = CouplingAdaptationEngine::new(10, 0.01, AdaptationStrategy::Homeostatic);
        let metrics = engine.calculate_network_metrics();
        
        assert!(metrics.clustering_coefficient >= 0.0);
        assert!(metrics.clustering_coefficient <= 1.0);
        assert!(metrics.connection_density >= 0.0);
        assert!(metrics.connection_density <= 1.0);
    }
}