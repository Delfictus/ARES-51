//! PHASE 2B.3: Phase Coherence Analysis Across Oscillator Networks
//! Multi-scale coherence analysis for market regime detection and cross-frequency coupling

use std::collections::VecDeque;
use anyhow::Result;
use crate::spike_encoding::Spike;

/// Multi-scale phase coherence analyzer for oscillator networks
pub struct PhaseCoherenceAnalyzer {
    /// Number of oscillators in the network
    n_oscillators: usize,
    /// History buffer for phase evolution (circular buffer for efficiency)
    phase_history: VecDeque<Vec<f64>>,
    /// Maximum history length (1 second at 1kHz = 1000 samples)
    max_history: usize,
    /// Frequency bands for multi-scale analysis
    frequency_bands: Vec<FrequencyBand>,
    /// Coherence threshold for significant coupling detection
    coherence_threshold: f64,
    /// Cross-frequency coupling analyzer
    cfc_analyzer: CrossFrequencyCouplingAnalyzer,
}

/// Frequency band definition for multi-scale analysis
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub name: String,
    pub freq_min: f64,  // Hz
    pub freq_max: f64,  // Hz
    pub market_significance: String, // Market interpretation
}

/// Phase coherence analysis result
#[derive(Debug, Clone)]
pub struct PhaseCoherenceResult {
    /// Global coherence across all oscillators [0,1]
    pub global_coherence: f64,
    /// Local coherence within frequency bands
    pub band_coherences: Vec<BandCoherence>,
    /// Cross-frequency coupling strength
    pub cross_frequency_coupling: Vec<CrossFrequencyCoupling>,
    /// Network topology metrics
    pub network_metrics: NetworkTopologyMetrics,
    /// Market regime classification based on coherence patterns
    pub market_regime: MarketRegime,
    /// Timestamp of analysis
    pub timestamp_ns: u64,
}

/// Coherence analysis for specific frequency band
#[derive(Debug, Clone)]
pub struct BandCoherence {
    pub band: FrequencyBand,
    pub coherence_strength: f64,      // Phase locking strength [0,1]
    pub participating_oscillators: Vec<usize>, // Oscillator indices
    pub dominant_frequency: f64,      // Peak frequency in band
    pub phase_variance: f64,          // Phase distribution spread
    pub temporal_stability: f64,      // Coherence persistence over time
}

/// Cross-frequency coupling between bands
#[derive(Debug, Clone)]
pub struct CrossFrequencyCoupling {
    pub low_freq_band: String,
    pub high_freq_band: String,
    pub coupling_strength: f64,       // Modulation strength [0,1]
    pub coupling_type: CouplingType,
    pub phase_lag: f64,              // Phase relationship (radians)
}

/// Types of cross-frequency coupling
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingType {
    PhaseAmplitude,     // Low-freq phase modulates high-freq amplitude
    PhasePhase,         // Phase-phase coupling
    AmplitudeAmplitude, // Amplitude correlation
    PhaseFrequency,     // Low-freq phase modulates high-freq frequency
}

/// Network topology metrics
#[derive(Debug, Clone)]
pub struct NetworkTopologyMetrics {
    pub clustering_coefficient: f64,  // Local clustering strength
    pub path_length: f64,            // Average shortest path length
    pub small_worldness: f64,        // Small-world network index
    pub modularity: f64,             // Community structure strength
    pub synchronization_landscape: Vec<f64>, // Sync strength at each freq
}

/// Market regime classification based on coherence patterns
#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    /// High coherence across frequencies - trending market
    Trending {
        direction: TrendDirection,
        strength: f64,
    },
    /// Moderate coherence with oscillations - ranging market
    Ranging {
        volatility: f64,
        support_resistance_strength: f64,
    },
    /// Low coherence, high cross-freq coupling - transition period
    Transitional {
        uncertainty: f64,
        expected_direction: Option<TrendDirection>,
    },
    /// Very low coherence - chaotic/crisis market
    Chaotic {
        stress_level: f64,
    },
    /// High low-freq coherence - regime change
    RegimeShift {
        shift_magnitude: f64,
        time_to_stabilization_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Cross-frequency coupling analyzer
pub struct CrossFrequencyCouplingAnalyzer {
    /// Window size for CFC analysis
    window_size: usize,
    /// Overlap between analysis windows
    overlap: f64,
    /// Phase extraction method
    phase_method: PhaseExtractionMethod,
}

#[derive(Debug, Clone)]
pub enum PhaseExtractionMethod {
    Hilbert,      // Hilbert transform
    Wavelet,      // Wavelet transform
    Empirical,    // Empirical mode decomposition
}

impl PhaseCoherenceAnalyzer {
    /// Create new phase coherence analyzer with market-optimized frequency bands
    pub fn new(n_oscillators: usize) -> Self {
        let frequency_bands = vec![
            FrequencyBand {
                name: "Ultra-High".to_string(),
                freq_min: 50.0,
                freq_max: 100.0,
                market_significance: "Tick-level noise and HFT activity".to_string(),
            },
            FrequencyBand {
                name: "High".to_string(),
                freq_min: 10.0,
                freq_max: 50.0,
                market_significance: "Minute-level price movements".to_string(),
            },
            FrequencyBand {
                name: "Medium".to_string(),
                freq_min: 1.0,
                freq_max: 10.0,
                market_significance: "Hourly trends and intraday patterns".to_string(),
            },
            FrequencyBand {
                name: "Low".to_string(),
                freq_min: 0.1,
                freq_max: 1.0,
                market_significance: "Daily trends and swing trading".to_string(),
            },
            FrequencyBand {
                name: "Ultra-Low".to_string(),
                freq_min: 0.01,
                freq_max: 0.1,
                market_significance: "Weekly/monthly regime changes".to_string(),
            },
        ];

        Self {
            n_oscillators,
            phase_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            frequency_bands,
            coherence_threshold: 0.3,
            cfc_analyzer: CrossFrequencyCouplingAnalyzer::new(),
        }
    }

    /// Analyze phase coherence across oscillator network
    pub fn analyze_coherence(&mut self, phases: &[f64]) -> Result<PhaseCoherenceResult> {
        if phases.len() != self.n_oscillators {
            anyhow::bail!("Phase array size {} doesn't match oscillator count {}", 
                phases.len(), self.n_oscillators);
        }

        // Store current phases in history
        self.phase_history.push_back(phases.to_vec());
        if self.phase_history.len() > self.max_history {
            self.phase_history.pop_front();
        }

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate global coherence (Kuramoto order parameter)
        let global_coherence = self.calculate_global_coherence(phases);

        // Analyze coherence within each frequency band
        let band_coherences = self.analyze_band_coherences(phases)?;

        // Detect cross-frequency coupling
        let cross_frequency_coupling = self.detect_cross_frequency_coupling()?;

        // Calculate network topology metrics
        let network_metrics = self.calculate_network_metrics(phases)?;

        // Classify market regime
        let market_regime = self.classify_market_regime(
            global_coherence,
            &band_coherences,
            &cross_frequency_coupling,
        );

        Ok(PhaseCoherenceResult {
            global_coherence,
            band_coherences,
            cross_frequency_coupling,
            network_metrics,
            market_regime,
            timestamp_ns: current_time,
        })
    }

    /// Calculate global coherence using Kuramoto order parameter
    fn calculate_global_coherence(&self, phases: &[f64]) -> f64 {
        let n = phases.len() as f64;
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for &phase in phases {
            real_sum += phase.cos();
            imag_sum += phase.sin();
        }

        // Kuramoto order parameter R = |<e^(iθ)>|
        ((real_sum / n).powi(2) + (imag_sum / n).powi(2)).sqrt()
    }

    /// Analyze coherence within specific frequency bands
    fn analyze_band_coherences(&self, phases: &[f64]) -> Result<Vec<BandCoherence>> {
        let mut band_coherences = Vec::new();

        for band in &self.frequency_bands {
            // Filter oscillators by natural frequency (simplified - would use actual freq analysis)
            let oscillators_in_band: Vec<usize> = (0..self.n_oscillators)
                .filter(|&i| {
                    // Assume oscillator frequency correlates with index for demonstration
                    let osc_freq = 0.01 + (i as f64 / self.n_oscillators as f64) * 99.99;
                    osc_freq >= band.freq_min && osc_freq < band.freq_max
                })
                .collect();

            if oscillators_in_band.len() < 3 {
                continue; // Need minimum oscillators for meaningful coherence
            }

            // Calculate coherence for oscillators in this band
            let band_phases: Vec<f64> = oscillators_in_band.iter()
                .map(|&i| phases[i])
                .collect();

            let coherence_strength = self.calculate_global_coherence(&band_phases);

            // Calculate band statistics
            let phase_variance = self.calculate_phase_variance(&band_phases);
            let dominant_frequency = (band.freq_min + band.freq_max) / 2.0; // Simplified
            let temporal_stability = self.calculate_temporal_stability(&oscillators_in_band);

            band_coherences.push(BandCoherence {
                band: band.clone(),
                coherence_strength,
                participating_oscillators: oscillators_in_band,
                dominant_frequency,
                phase_variance,
                temporal_stability,
            });
        }

        Ok(band_coherences)
    }

    /// Detect cross-frequency coupling between bands
    fn detect_cross_frequency_coupling(&self) -> Result<Vec<CrossFrequencyCoupling>> {
        let mut couplings = Vec::new();

        if self.phase_history.len() < 50 {
            return Ok(couplings); // Need sufficient history for CFC analysis
        }

        // Analyze coupling between each pair of frequency bands
        for (i, low_band) in self.frequency_bands.iter().enumerate() {
            for high_band in self.frequency_bands.iter().skip(i + 1) {
                if let Some(coupling) = self.analyze_phase_amplitude_coupling(low_band, high_band)? {
                    couplings.push(coupling);
                }
            }
        }

        Ok(couplings)
    }

    /// Analyze phase-amplitude coupling between two frequency bands
    fn analyze_phase_amplitude_coupling(
        &self,
        low_freq_band: &FrequencyBand,
        high_freq_band: &FrequencyBand,
    ) -> Result<Option<CrossFrequencyCoupling>> {
        // Extract phases for low frequency band oscillators
        let low_freq_oscillators: Vec<usize> = (0..self.n_oscillators)
            .filter(|&i| {
                let osc_freq = 0.01 + (i as f64 / self.n_oscillators as f64) * 99.99;
                osc_freq >= low_freq_band.freq_min && osc_freq < low_freq_band.freq_max
            })
            .collect();

        let high_freq_oscillators: Vec<usize> = (0..self.n_oscillators)
            .filter(|&i| {
                let osc_freq = 0.01 + (i as f64 / self.n_oscillators as f64) * 99.99;
                osc_freq >= high_freq_band.freq_min && osc_freq < high_freq_band.freq_max
            })
            .collect();

        if low_freq_oscillators.len() < 3 || high_freq_oscillators.len() < 3 {
            return Ok(None);
        }

        // Calculate phase-amplitude coupling using mean vector length method
        let mut coupling_values = Vec::new();
        
        for history_step in self.phase_history.iter().rev().take(50) {
            // Extract low-freq phase (averaged across oscillators in band)
            let low_phase = low_freq_oscillators.iter()
                .map(|&i| history_step[i])
                .sum::<f64>() / low_freq_oscillators.len() as f64;

            // Extract high-freq amplitude (phase derivative approximation)
            let high_amplitude = high_freq_oscillators.iter()
                .map(|&i| history_step[i].cos().abs()) // Simplified amplitude estimate
                .sum::<f64>() / high_freq_oscillators.len() as f64;

            coupling_values.push((low_phase, high_amplitude));
        }

        // Calculate modulation index (simplified version)
        let coupling_strength = self.calculate_modulation_index(&coupling_values);
        
        // Calculate phase lag
        let phase_lag = self.calculate_phase_lag(&coupling_values);

        if coupling_strength > 0.1 { // Significant coupling threshold
            Ok(Some(CrossFrequencyCoupling {
                low_freq_band: low_freq_band.name.clone(),
                high_freq_band: high_freq_band.name.clone(),
                coupling_strength,
                coupling_type: CouplingType::PhaseAmplitude,
                phase_lag,
            }))
        } else {
            Ok(None)
        }
    }

    /// Calculate modulation index for phase-amplitude coupling
    fn calculate_modulation_index(&self, coupling_data: &[(f64, f64)]) -> f64 {
        if coupling_data.len() < 10 {
            return 0.0;
        }

        // Bin phases and calculate amplitude distribution
        let n_bins = 18; // 20-degree bins
        let mut bin_amplitudes = vec![Vec::new(); n_bins];
        
        for &(phase, amplitude) in coupling_data {
            let normalized_phase = (phase + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
            let bin = (normalized_phase / (2.0 * std::f64::consts::PI) * n_bins as f64) as usize;
            if bin < n_bins {
                bin_amplitudes[bin].push(amplitude);
            }
        }

        // Calculate mean amplitude per bin
        let bin_means: Vec<f64> = bin_amplitudes.iter()
            .map(|bin| {
                if bin.is_empty() {
                    0.0
                } else {
                    bin.iter().sum::<f64>() / bin.len() as f64
                }
            })
            .collect();

        // Calculate modulation index as variance of bin means
        let overall_mean = bin_means.iter().sum::<f64>() / bin_means.len() as f64;
        let variance = bin_means.iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>() / bin_means.len() as f64;

        (variance / overall_mean.max(0.001)).clamp(0.0, 1.0)
    }

    /// Calculate phase lag between coupled frequencies
    fn calculate_phase_lag(&self, coupling_data: &[(f64, f64)]) -> f64 {
        // Simplified cross-correlation approach
        let phases: Vec<f64> = coupling_data.iter().map(|&(p, _)| p).collect();
        let amplitudes: Vec<f64> = coupling_data.iter().map(|&(_, a)| a).collect();

        if phases.len() < 5 {
            return 0.0;
        }

        // Calculate circular-linear correlation (simplified)
        let mut best_lag = 0.0;
        let mut max_correlation = 0.0;

        for lag_steps in 0..5 {
            let lag = lag_steps as f64 * 0.1; // Test different phase lags
            
            let mut correlation_sum = 0.0;
            for i in 0..phases.len() {
                let lagged_phase = phases[i] + lag;
                correlation_sum += lagged_phase.cos() * amplitudes[i];
            }
            
            let correlation = correlation_sum / phases.len() as f64;
            if correlation > max_correlation {
                max_correlation = correlation;
                best_lag = lag;
            }
        }

        best_lag
    }

    /// Calculate network topology metrics
    fn calculate_network_metrics(&self, phases: &[f64]) -> Result<NetworkTopologyMetrics> {
        // Calculate pairwise phase differences for connectivity
        let mut connectivity_matrix = vec![vec![0.0; self.n_oscillators]; self.n_oscillators];
        
        for i in 0..self.n_oscillators {
            for j in i+1..self.n_oscillators {
                let phase_diff = (phases[i] - phases[j]).abs();
                let normalized_diff = (phase_diff % (2.0 * std::f64::consts::PI)) / std::f64::consts::PI;
                
                // Strong coupling if phase difference is small
                let coupling_strength = 1.0 - normalized_diff;
                connectivity_matrix[i][j] = coupling_strength;
                connectivity_matrix[j][i] = coupling_strength;
            }
        }

        // Calculate clustering coefficient
        let clustering_coefficient = self.calculate_clustering_coefficient(&connectivity_matrix);
        
        // Calculate average path length (simplified)
        let path_length = self.calculate_average_path_length(&connectivity_matrix);
        
        // Small-worldness index
        let small_worldness = clustering_coefficient / path_length.max(0.001);
        
        // Modularity (simplified community detection)
        let modularity = self.calculate_modularity(&connectivity_matrix);
        
        // Synchronization landscape
        let synchronization_landscape = self.calculate_sync_landscape(phases);

        Ok(NetworkTopologyMetrics {
            clustering_coefficient,
            path_length,
            small_worldness,
            modularity,
            synchronization_landscape,
        })
    }

    /// Calculate clustering coefficient of the network
    fn calculate_clustering_coefficient(&self, connectivity: &[Vec<f64>]) -> f64 {
        let threshold = 0.5; // Connection threshold
        let mut total_clustering = 0.0;
        let mut valid_nodes = 0;

        for i in 0..self.n_oscillators {
            let neighbors: Vec<usize> = (0..self.n_oscillators)
                .filter(|&j| j != i && connectivity[i][j] > threshold)
                .collect();

            if neighbors.len() < 2 {
                continue; // Need at least 2 neighbors for clustering
            }

            let mut triangles = 0;
            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;

            for k in 0..neighbors.len() {
                for l in k+1..neighbors.len() {
                    if connectivity[neighbors[k]][neighbors[l]] > threshold {
                        triangles += 1;
                    }
                }
            }

            total_clustering += triangles as f64 / possible_triangles as f64;
            valid_nodes += 1;
        }

        if valid_nodes > 0 {
            total_clustering / valid_nodes as f64
        } else {
            0.0
        }
    }

    /// Calculate average path length (simplified)
    fn calculate_average_path_length(&self, connectivity: &[Vec<f64>]) -> f64 {
        // Simplified: use inverse of average connectivity as proxy for path length
        let total_connectivity: f64 = connectivity.iter()
            .flatten()
            .filter(|&&c| c > 0.5)
            .sum();
        
        let avg_connectivity = total_connectivity / (self.n_oscillators * (self.n_oscillators - 1)) as f64;
        
        if avg_connectivity > 0.0 {
            1.0 / avg_connectivity
        } else {
            f64::INFINITY
        }
    }

    /// Calculate network modularity
    fn calculate_modularity(&self, connectivity: &[Vec<f64>]) -> f64 {
        // Simplified modularity calculation
        // In practice, would use sophisticated community detection algorithms
        let total_edges: f64 = connectivity.iter()
            .flatten()
            .filter(|&&c| c > 0.5)
            .sum::<f64>() / 2.0; // Divide by 2 for undirected graph

        if total_edges < 1.0 {
            return 0.0;
        }

        // Detect communities using simple degree-based clustering
        let mut modularity = 0.0;
        let num_communities = 4; // Assume 4 communities for simplification
        let community_size = self.n_oscillators / num_communities;

        for c in 0..num_communities {
            let start = c * community_size;
            let end = if c == num_communities - 1 { self.n_oscillators } else { start + community_size };
            
            let mut internal_edges = 0.0;
            let mut external_edges = 0.0;

            for i in start..end {
                for j in i+1..self.n_oscillators {
                    if connectivity[i][j] > 0.5 {
                        if j < end {
                            internal_edges += 1.0;
                        } else {
                            external_edges += 1.0;
                        }
                    }
                }
            }

            let expected_internal = (end - start) as f64 * (end - start - 1) as f64 / (2.0 * total_edges);
            modularity += (internal_edges - expected_internal) / total_edges;
        }

        modularity.clamp(-1.0, 1.0)
    }

    /// Calculate synchronization landscape across frequencies
    fn calculate_sync_landscape(&self, phases: &[f64]) -> Vec<f64> {
        // Calculate synchronization strength at different frequency scales
        let mut landscape = Vec::new();
        
        for band in &self.frequency_bands {
            let oscillators_in_band: Vec<usize> = (0..self.n_oscillators)
                .filter(|&i| {
                    let osc_freq = 0.01 + (i as f64 / self.n_oscillators as f64) * 99.99;
                    osc_freq >= band.freq_min && osc_freq < band.freq_max
                })
                .collect();

            if oscillators_in_band.len() > 2 {
                let band_phases: Vec<f64> = oscillators_in_band.iter()
                    .map(|&i| phases[i])
                    .collect();
                let sync_strength = self.calculate_global_coherence(&band_phases);
                landscape.push(sync_strength);
            } else {
                landscape.push(0.0);
            }
        }

        landscape
    }

    /// Calculate phase variance within a group of oscillators
    fn calculate_phase_variance(&self, phases: &[f64]) -> f64 {
        if phases.len() < 2 {
            return 0.0;
        }

        // Calculate circular variance
        let mean_cos = phases.iter().map(|&p| p.cos()).sum::<f64>() / phases.len() as f64;
        let mean_sin = phases.iter().map(|&p| p.sin()).sum::<f64>() / phases.len() as f64;
        let r = (mean_cos.powi(2) + mean_sin.powi(2)).sqrt();
        
        1.0 - r // Circular variance = 1 - mean resultant length
    }

    /// Calculate temporal stability of coherence
    fn calculate_temporal_stability(&self, oscillator_indices: &[usize]) -> f64 {
        if self.phase_history.len() < 10 || oscillator_indices.len() < 2 {
            return 0.0;
        }

        // Calculate coherence over recent history
        let mut coherence_history = Vec::new();
        
        for history_step in self.phase_history.iter().rev().take(20) {
            let phases: Vec<f64> = oscillator_indices.iter()
                .map(|&i| history_step[i])
                .collect();
            let coherence = self.calculate_global_coherence(&phases);
            coherence_history.push(coherence);
        }

        // Calculate stability as inverse of variance
        let mean_coherence = coherence_history.iter().sum::<f64>() / coherence_history.len() as f64;
        let variance = coherence_history.iter()
            .map(|&c| (c - mean_coherence).powi(2))
            .sum::<f64>() / coherence_history.len() as f64;

        1.0 - variance.clamp(0.0, 1.0) // Stability = 1 - variance
    }

    /// Classify market regime based on coherence patterns
    fn classify_market_regime(
        &self,
        global_coherence: f64,
        band_coherences: &[BandCoherence],
        cross_freq_coupling: &[CrossFrequencyCoupling],
    ) -> MarketRegime {
        // Market regime classification logic based on coherence patterns
        
        // High global coherence usually indicates trending
        if global_coherence > 0.8 {
            // Check for directional bias in frequency bands
            let low_freq_coherence = band_coherences.iter()
                .find(|bc| bc.band.name == "Low")
                .map(|bc| bc.coherence_strength)
                .unwrap_or(0.0);

            let direction = if low_freq_coherence > 0.7 {
                TrendDirection::Bullish // Assume bullish for high low-freq coherence
            } else {
                TrendDirection::Bearish
            };

            return MarketRegime::Trending {
                direction,
                strength: global_coherence,
            };
        }

        // Strong cross-frequency coupling often indicates transitions
        let strong_cfc = cross_freq_coupling.iter()
            .any(|cfc| cfc.coupling_strength > 0.5);

        if strong_cfc && global_coherence < 0.5 {
            return MarketRegime::Transitional {
                uncertainty: 1.0 - global_coherence,
                expected_direction: None, // Would require more sophisticated analysis
            };
        }

        // Medium coherence with oscillatory patterns = ranging
        if global_coherence > 0.4 && global_coherence < 0.7 {
            let medium_band_coherence = band_coherences.iter()
                .find(|bc| bc.band.name == "Medium")
                .map(|bc| bc.coherence_strength)
                .unwrap_or(0.0);

            return MarketRegime::Ranging {
                volatility: 1.0 - global_coherence,
                support_resistance_strength: medium_band_coherence,
            };
        }

        // Very low coherence = chaos
        if global_coherence < 0.2 {
            return MarketRegime::Chaotic {
                stress_level: 1.0 - global_coherence,
            };
        }

        // Check for regime shift pattern
        let ultra_low_coherence = band_coherences.iter()
            .find(|bc| bc.band.name == "Ultra-Low")
            .map(|bc| bc.coherence_strength)
            .unwrap_or(0.0);

        if ultra_low_coherence > 0.6 {
            return MarketRegime::RegimeShift {
                shift_magnitude: ultra_low_coherence,
                time_to_stabilization_ms: 3600000, // 1 hour estimate
            };
        }

        // Default fallback
        MarketRegime::Ranging {
            volatility: 0.5,
            support_resistance_strength: global_coherence,
        }
    }
}

impl CrossFrequencyCouplingAnalyzer {
    pub fn new() -> Self {
        Self {
            window_size: 256,
            overlap: 0.5,
            phase_method: PhaseExtractionMethod::Hilbert,
        }
    }
}

impl Default for CrossFrequencyCouplingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}