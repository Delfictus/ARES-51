//! PHASE 2B.2: Transfer Entropy Engine for Market Causality Detection
//! Detects directional information flow between market variables using Shannon information theory

use std::collections::HashMap;
use anyhow::Result;
use crate::spike_encoding::Spike;

/// Transfer entropy engine for detecting causal relationships in market data
pub struct TransferEntropyEngine {
    /// History length for conditional probability estimation
    history_length: usize,
    /// Future prediction horizon  
    future_length: usize,
    /// Number of bins for signal discretization
    n_bins: usize,
    /// Minimum sample size for reliable entropy estimation
    min_samples: usize,
    /// Cache for probability distributions
    prob_cache: HashMap<String, f64>,
}

impl TransferEntropyEngine {
    /// Create new transfer entropy engine with market-optimized parameters
    pub fn new() -> Self {
        Self {
            history_length: 10,  // 10-step history for market memory
            future_length: 1,    // 1-step prediction horizon
            n_bins: 8,          // 8 bins for signal discretization (optimal for market data)
            min_samples: 50,    // Minimum samples for statistical reliability
            prob_cache: HashMap::new(),
        }
    }

    /// Configure engine parameters for different market timescales
    pub fn with_config(history_length: usize, future_length: usize, n_bins: usize) -> Self {
        Self {
            history_length,
            future_length,
            n_bins,
            min_samples: 3 * history_length, // 3x history for statistical significance
            prob_cache: HashMap::new(),
        }
    }

    /// Calculate transfer entropy from source to target series
    /// TE(S→T) = Σ p(t_{n+1}, t_n^k, s_n^l) * log(p(t_{n+1}|t_n^k, s_n^l) / p(t_{n+1}|t_n^k))
    pub fn calculate_transfer_entropy(
        &mut self, 
        source: &[f64], 
        target: &[f64]
    ) -> Result<f64> {
        if source.len() != target.len() || source.len() < self.min_samples {
            anyhow::bail!("Insufficient data: need at least {} samples, got {}", 
                self.min_samples, source.len().min(target.len()));
        }

        // Clear cache for new calculation
        self.prob_cache.clear();

        // Discretize continuous signals into bins
        let source_discrete = self.discretize_signal(source);
        let target_discrete = self.discretize_signal(target);

        // Build conditional probability distributions
        let p_future_given_both = self.estimate_conditional_probability_3way(
            &target_discrete,
            &source_discrete,
        )?;

        let p_future_given_target = self.estimate_conditional_probability_2way(
            &target_discrete,
        )?;

        // Calculate transfer entropy
        let mut te = 0.0;
        
        for ((target_future, target_past, source_past), &prob_both) in &p_future_given_both {
            if prob_both > 0.0 {
                let key = format!("{}_{}", target_future, target_past);
                if let Some(&prob_target_only) = p_future_given_target.get(&key) {
                    if prob_target_only > 0.0 {
                        te += prob_both * (prob_both / prob_target_only).ln();
                    }
                }
            }
        }

        // Convert from natural log to bits (information theory standard)
        Ok(te / std::f64::consts::LN_2)
    }

    /// Detect causal relationships in market spike data
    pub fn analyze_spike_causality(
        &mut self,
        spikes_a: &[Spike],
        spikes_b: &[Spike],
        time_window_ms: u64,
    ) -> Result<CausalityAnalysis> {
        // Convert spikes to time series
        let series_a = self.spikes_to_time_series(spikes_a, time_window_ms);
        let series_b = self.spikes_to_time_series(spikes_b, time_window_ms);

        // Calculate bidirectional transfer entropy
        let te_a_to_b = self.calculate_transfer_entropy(&series_a, &series_b)?;
        let te_b_to_a = self.calculate_transfer_entropy(&series_b, &series_a)?;

        // Determine causal direction and strength
        let net_causality = te_a_to_b - te_b_to_a;
        let total_causality = te_a_to_b + te_b_to_a;
        
        let causal_direction = if net_causality.abs() < 0.01 {
            CausalDirection::Bidirectional
        } else if net_causality > 0.0 {
            CausalDirection::AToB
        } else {
            CausalDirection::BToA
        };

        let causal_strength = if total_causality > 0.5 {
            CausalStrength::Strong
        } else if total_causality > 0.2 {
            CausalStrength::Moderate
        } else if total_causality > 0.05 {
            CausalStrength::Weak
        } else {
            CausalStrength::None
        };

        Ok(CausalityAnalysis {
            te_a_to_b,
            te_b_to_a,
            net_causality,
            total_causality,
            causal_direction,
            causal_strength,
            significance_level: self.calculate_significance(total_causality, series_a.len()),
        })
    }

    /// Analyze multi-asset causality network
    pub fn analyze_market_causality_network(
        &mut self,
        asset_data: &HashMap<String, Vec<f64>>,
    ) -> Result<CausalityNetwork> {
        let assets: Vec<_> = asset_data.keys().cloned().collect();
        let mut network = CausalityNetwork::new(assets.clone());

        // Calculate pairwise transfer entropy for all asset pairs
        for (i, asset_a) in assets.iter().enumerate() {
            for (j, asset_b) in assets.iter().enumerate() {
                if i != j {
                    if let (Some(data_a), Some(data_b)) = 
                        (asset_data.get(asset_a), asset_data.get(asset_b)) {
                        
                        match self.calculate_transfer_entropy(data_a, data_b) {
                            Ok(te) => {
                                network.set_causality(asset_a, asset_b, te);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to calculate TE for {}→{}: {}", 
                                    asset_a, asset_b, e);
                            }
                        }
                    }
                }
            }
        }

        // Identify dominant causal relationships
        network.identify_hubs();
        network.detect_causal_clusters();

        Ok(network)
    }

    /// Discretize continuous signal into bins for entropy calculation
    fn discretize_signal(&self, signal: &[f64]) -> Vec<usize> {
        // Calculate signal statistics
        let min_val = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        if range == 0.0 {
            // Constant signal
            return vec![0; signal.len()];
        }

        let bin_width = range / self.n_bins as f64;
        
        signal.iter().map(|&value| {
            let bin = ((value - min_val) / bin_width).floor() as usize;
            bin.min(self.n_bins - 1) // Ensure we don't exceed bin count
        }).collect()
    }

    /// Estimate conditional probability P(target_future | target_past, source_past)
    fn estimate_conditional_probability_3way(
        &self,
        target_discrete: &[usize],
        source_discrete: &[usize],
    ) -> Result<HashMap<(usize, usize, usize), f64>> {
        let mut joint_counts = HashMap::new();
        let mut condition_counts = HashMap::new();
        
        let data_len = target_discrete.len();
        
        // Count occurrences
        for i in self.history_length..data_len - self.future_length {
            // Future target value
            let target_future = target_discrete[i + self.future_length];
            
            // Past target state (simplified to single step for computational efficiency)
            let target_past = target_discrete[i];
            
            // Past source state
            let source_past = source_discrete[i];
            
            let joint_key = (target_future, target_past, source_past);
            let condition_key = (target_past, source_past);
            
            *joint_counts.entry(joint_key).or_insert(0) += 1;
            *condition_counts.entry(condition_key).or_insert(0) += 1;
        }

        // Convert counts to probabilities
        let mut probabilities = HashMap::new();
        
        for (joint_key, joint_count) in joint_counts {
            let condition_key = (joint_key.1, joint_key.2);
            if let Some(&condition_count) = condition_counts.get(&condition_key) {
                let probability = joint_count as f64 / condition_count as f64;
                probabilities.insert(joint_key, probability);
            }
        }

        Ok(probabilities)
    }

    /// Estimate conditional probability P(target_future | target_past)
    fn estimate_conditional_probability_2way(
        &self,
        target_discrete: &[usize],
    ) -> Result<HashMap<String, f64>> {
        let mut joint_counts = HashMap::new();
        let mut condition_counts = HashMap::new();
        
        let data_len = target_discrete.len();
        
        // Count occurrences
        for i in self.history_length..data_len - self.future_length {
            let target_future = target_discrete[i + self.future_length];
            let target_past = target_discrete[i];
            
            let joint_key = format!("{}_{}", target_future, target_past);
            
            *joint_counts.entry(joint_key.clone()).or_insert(0) += 1;
            *condition_counts.entry(target_past).or_insert(0) += 1;
        }

        // Convert counts to probabilities
        let mut probabilities = HashMap::new();
        
        for (joint_key, joint_count) in joint_counts {
            let parts: Vec<&str> = joint_key.split('_').collect();
            if parts.len() == 2 {
                if let Ok(target_past) = parts[1].parse::<usize>() {
                    if let Some(&condition_count) = condition_counts.get(&target_past) {
                        let probability = joint_count as f64 / condition_count as f64;
                        probabilities.insert(joint_key, probability);
                    }
                }
            }
        }

        Ok(probabilities)
    }

    /// Convert spike trains to time series for entropy analysis
    fn spikes_to_time_series(&self, spikes: &[Spike], window_ms: u64) -> Vec<f64> {
        if spikes.is_empty() {
            return vec![0.0; 100]; // Default empty series
        }

        // Find time range
        let min_time = spikes.iter().map(|s| s.timestamp_ns).min().unwrap_or(0);
        let max_time = spikes.iter().map(|s| s.timestamp_ns).max().unwrap_or(0);
        
        let window_ns = window_ms * 1_000_000;
        let num_windows = ((max_time - min_time) / window_ns + 1) as usize;
        
        let mut series = vec![0.0; num_windows];
        
        // Bin spikes into time windows
        for spike in spikes {
            let window_idx = ((spike.timestamp_ns - min_time) / window_ns) as usize;
            if window_idx < series.len() {
                series[window_idx] += spike.strength as f64;
            }
        }

        series
    }

    /// Calculate statistical significance of transfer entropy result
    fn calculate_significance(&self, te_value: f64, sample_size: usize) -> f64 {
        // Simplified significance test based on sample size and TE magnitude
        // In production, this would use bootstrap or surrogate data methods
        
        let degrees_of_freedom = (self.n_bins - 1).pow(3) as f64;
        let expected_random_te = degrees_of_freedom / (2.0 * sample_size as f64);
        
        if te_value > 2.0 * expected_random_te {
            0.95 // High significance
        } else if te_value > expected_random_te {
            0.75 // Moderate significance
        } else {
            0.25 // Low significance
        }
    }
}

impl Default for TransferEntropyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of causality analysis between two variables
#[derive(Debug, Clone)]
pub struct CausalityAnalysis {
    pub te_a_to_b: f64,           // Transfer entropy A → B
    pub te_b_to_a: f64,           // Transfer entropy B → A  
    pub net_causality: f64,       // Net causal flow (A→B minus B→A)
    pub total_causality: f64,     // Total information transfer
    pub causal_direction: CausalDirection,
    pub causal_strength: CausalStrength,
    pub significance_level: f64,  // Statistical significance [0,1]
}

/// Direction of causal relationship
#[derive(Debug, Clone, PartialEq)]
pub enum CausalDirection {
    AToB,           // A causes B
    BToA,           // B causes A
    Bidirectional,  // Mutual causation
    None,           // No significant causation
}

/// Strength of causal relationship
#[derive(Debug, Clone, PartialEq)]
pub enum CausalStrength {
    Strong,    // TE > 0.5 bits
    Moderate,  // TE > 0.2 bits
    Weak,      // TE > 0.05 bits
    None,      // TE ≤ 0.05 bits
}

/// Network of causal relationships between multiple assets
#[derive(Debug, Clone)]
pub struct CausalityNetwork {
    pub assets: Vec<String>,
    pub causality_matrix: Vec<Vec<f64>>,  // [from][to] transfer entropy
    pub causal_hubs: Vec<String>,         // Assets with high outgoing causality
    pub causal_sinks: Vec<String>,        // Assets with high incoming causality
    pub clusters: Vec<Vec<String>>,       // Groups of mutually causal assets
}

impl CausalityNetwork {
    pub fn new(assets: Vec<String>) -> Self {
        let n = assets.len();
        Self {
            assets,
            causality_matrix: vec![vec![0.0; n]; n],
            causal_hubs: Vec::new(),
            causal_sinks: Vec::new(),
            clusters: Vec::new(),
        }
    }

    pub fn set_causality(&mut self, from: &str, to: &str, te_value: f64) {
        if let (Some(from_idx), Some(to_idx)) = (
            self.assets.iter().position(|a| a == from),
            self.assets.iter().position(|a| a == to),
        ) {
            self.causality_matrix[from_idx][to_idx] = te_value;
        }
    }

    pub fn get_causality(&self, from: &str, to: &str) -> Option<f64> {
        let from_idx = self.assets.iter().position(|a| a == from)?;
        let to_idx = self.assets.iter().position(|a| a == to)?;
        Some(self.causality_matrix[from_idx][to_idx])
    }

    /// Identify assets that are causal hubs (high outgoing causality)
    pub fn identify_hubs(&mut self) {
        let mut outgoing_causality: Vec<(String, f64)> = self.assets.iter()
            .enumerate()
            .map(|(i, asset)| {
                let total_outgoing = self.causality_matrix[i].iter().sum::<f64>();
                (asset.clone(), total_outgoing)
            })
            .collect();

        outgoing_causality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Top 20% are considered hubs
        let hub_count = (self.assets.len() / 5).max(1);
        self.causal_hubs = outgoing_causality.into_iter()
            .take(hub_count)
            .filter(|(_, causality)| *causality > 0.1) // Minimum threshold
            .map(|(asset, _)| asset)
            .collect();

        // Identify sinks (high incoming causality)
        let mut incoming_causality: Vec<(String, f64)> = self.assets.iter()
            .enumerate()
            .map(|(j, asset)| {
                let total_incoming = (0..self.assets.len())
                    .map(|i| self.causality_matrix[i][j])
                    .sum::<f64>();
                (asset.clone(), total_incoming)
            })
            .collect();

        incoming_causality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let sink_count = (self.assets.len() / 5).max(1);
        self.causal_sinks = incoming_causality.into_iter()
            .take(sink_count)
            .filter(|(_, causality)| *causality > 0.1)
            .map(|(asset, _)| asset)
            .collect();
    }

    /// Detect clusters of mutually causal assets
    pub fn detect_causal_clusters(&mut self) {
        // Simple clustering based on bidirectional high causality
        let threshold = 0.2; // Minimum TE for cluster membership
        let mut visited = vec![false; self.assets.len()];
        
        for i in 0..self.assets.len() {
            if visited[i] {
                continue;
            }

            let mut cluster = vec![self.assets[i].clone()];
            visited[i] = true;

            // Find strongly connected assets
            for j in 0..self.assets.len() {
                if i != j && !visited[j] {
                    let causality_ij = self.causality_matrix[i][j];
                    let causality_ji = self.causality_matrix[j][i];
                    
                    // Mutual high causality indicates cluster membership
                    if causality_ij > threshold && causality_ji > threshold {
                        cluster.push(self.assets[j].clone());
                        visited[j] = true;
                    }
                }
            }

            if cluster.len() > 1 {
                self.clusters.push(cluster);
            }
        }
    }

    /// Get summary statistics of the causality network
    pub fn get_network_statistics(&self) -> NetworkStatistics {
        let total_connections = self.causality_matrix.iter()
            .flatten()
            .filter(|&&te| te > 0.05)
            .count();

        let strong_connections = self.causality_matrix.iter()
            .flatten()
            .filter(|&&te| te > 0.5)
            .count();

        let average_causality = self.causality_matrix.iter()
            .flatten()
            .filter(|&&te| te > 0.0)
            .sum::<f64>() / self.causality_matrix.iter().flatten().filter(|&&te| te > 0.0).count().max(1) as f64;

        NetworkStatistics {
            total_assets: self.assets.len(),
            total_connections,
            strong_connections,
            average_causality,
            num_hubs: self.causal_hubs.len(),
            num_sinks: self.causal_sinks.len(),
            num_clusters: self.clusters.len(),
            network_density: total_connections as f64 / (self.assets.len() * (self.assets.len() - 1)) as f64,
        }
    }
}

/// Network-level statistics for causality analysis
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub total_assets: usize,
    pub total_connections: usize,      // Connections with TE > 0.05
    pub strong_connections: usize,     // Connections with TE > 0.5
    pub average_causality: f64,        // Average non-zero TE
    pub num_hubs: usize,              // Number of causal hubs
    pub num_sinks: usize,             // Number of causal sinks
    pub num_clusters: usize,          // Number of causal clusters
    pub network_density: f64,         // Fraction of possible connections present
}