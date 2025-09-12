//! PHASE 2B.4: Multi-Timeframe Oscillator Networks
//! Revolutionary hierarchical temporal processing for market analysis
//! Implements cross-timescale pattern coherence and information flow

use crate::drpp::{DynamicResonancePatternProcessor, DrppConfig, Pattern, PatternType};
use crate::phase_coherence::{PhaseCoherenceAnalyzer, FrequencyBand, MarketRegime, CoherencePattern};
use crate::spike_encoding::Spike;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use anyhow::{Result, anyhow};
use tokio::time::{Duration, Instant};

/// Time horizons for multi-scale analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeHorizon {
    /// Ultra-high frequency: 100ms windows, for order flow analysis
    UltraHigh,
    /// High frequency: 1-second windows, for tick-level patterns  
    High,
    /// Medium frequency: 1-minute windows, for price action
    Medium,
    /// Low frequency: 15-minute windows, for trend analysis
    Low,
    /// Strategic: 1-hour+ windows, for regime detection
    Strategic,
}

impl TimeHorizon {
    pub fn window_duration(&self) -> Duration {
        match self {
            Self::UltraHigh => Duration::from_millis(100),
            Self::High => Duration::from_secs(1),
            Self::Medium => Duration::from_secs(60),
            Self::Low => Duration::from_secs(900),     // 15 minutes
            Self::Strategic => Duration::from_secs(3600), // 1 hour
        }
    }

    pub fn frequency_bands(&self) -> Vec<FrequencyBand> {
        match self {
            Self::UltraHigh => vec![
                FrequencyBand::new("Gamma", 30.0, 100.0),     // Gamma-like frequencies
                FrequencyBand::new("Beta", 15.0, 30.0),       // Beta-like frequencies
                FrequencyBand::new("Alpha", 8.0, 15.0),       // Alpha-like frequencies
            ],
            Self::High => vec![
                FrequencyBand::new("Fast", 1.0, 10.0),        // Fast oscillations
                FrequencyBand::new("Medium", 0.1, 1.0),       // Medium oscillations
                FrequencyBand::new("Slow", 0.01, 0.1),        // Slow oscillations
            ],
            Self::Medium => vec![
                FrequencyBand::new("HF", 0.1, 1.0),           // High frequency (minutes)
                FrequencyBand::new("MF", 0.01, 0.1),          // Medium frequency 
                FrequencyBand::new("LF", 0.001, 0.01),        // Low frequency
            ],
            Self::Low => vec![
                FrequencyBand::new("Trend", 0.001, 0.01),     // Trend components
                FrequencyBand::new("Cycle", 0.0001, 0.001),   // Cyclic components
            ],
            Self::Strategic => vec![
                FrequencyBand::new("Regime", 0.0001, 0.001),  // Regime shifts
                FrequencyBand::new("Macro", 0.00001, 0.0001), // Macro trends
            ],
        }
    }

    pub fn oscillator_count(&self) -> usize {
        match self {
            Self::UltraHigh => 512, // High resolution for microsecond patterns
            Self::High => 256,      // High resolution for tick patterns
            Self::Medium => 128,    // Medium resolution for price action
            Self::Low => 64,        // Lower resolution for trends
            Self::Strategic => 32,  // Minimal resolution for regimes
        }
    }
}

/// Cross-timescale information flow metrics
#[derive(Debug, Clone)]
pub struct CrossTimeInfo {
    /// Information flow from fast to slow timescale
    pub upward_flow: f64,
    /// Information flow from slow to fast timescale  
    pub downward_flow: f64,
    /// Net information flow (upward - downward)
    pub net_flow: f64,
    /// Mutual information between timescales
    pub mutual_information: f64,
    /// Phase synchronization between timescales
    pub phase_sync: f64,
    /// Cross-frequency coupling strength
    pub coupling_strength: f64,
}

/// Pattern propagation across timescales
#[derive(Debug, Clone)]
pub struct PatternPropagation {
    /// Source timescale
    pub source: TimeHorizon,
    /// Target timescale
    pub target: TimeHorizon,
    /// Pattern type being propagated
    pub pattern_type: PatternType,
    /// Propagation strength (0.0 to 1.0)
    pub strength: f64,
    /// Latency of propagation
    pub latency_ms: u64,
    /// Coherence maintained during propagation
    pub coherence_preservation: f64,
}

/// Multi-timeframe oscillator network
pub struct MultiTimeframeNetwork {
    /// DRPP processors for each timescale
    processors: HashMap<TimeHorizon, Arc<RwLock<DynamicResonancePatternProcessor>>>,
    /// Phase coherence analyzers for each timescale
    coherence_analyzers: HashMap<TimeHorizon, Arc<RwLock<PhaseCoherenceAnalyzer>>>,
    /// Cross-timescale information flows
    cross_time_flows: HashMap<(TimeHorizon, TimeHorizon), CrossTimeInfo>,
    /// Pattern propagation history
    propagation_history: VecDeque<PatternPropagation>,
    /// Maximum history length
    max_history: usize,
    /// Network synchronization state
    global_sync_state: Arc<RwLock<NetworkSyncState>>,
    /// Last update timestamps per timescale
    last_updates: HashMap<TimeHorizon, Instant>,
}

/// Global network synchronization state
#[derive(Debug, Clone)]
pub struct NetworkSyncState {
    /// Global order parameter across all timescales
    pub global_order: f64,
    /// Dominant regime across all timescales
    pub dominant_regime: MarketRegime,
    /// Network coherence score (0.0 to 1.0)
    pub network_coherence: f64,
    /// Cross-scale coupling matrix
    pub coupling_matrix: Vec<Vec<f64>>,
    /// Information cascade direction (upward/downward)
    pub cascade_direction: f64,
}

impl MultiTimeframeNetwork {
    /// Create new multi-timeframe oscillator network
    pub async fn new(max_history: usize) -> Result<Self> {
        let mut processors = HashMap::new();
        let mut coherence_analyzers = HashMap::new();
        let mut last_updates = HashMap::new();
        
        let timescales = [
            TimeHorizon::UltraHigh,
            TimeHorizon::High, 
            TimeHorizon::Medium,
            TimeHorizon::Low,
            TimeHorizon::Strategic,
        ];

        // Initialize processors and analyzers for each timescale
        for &timescale in &timescales {
            let drpp_config = DrppConfig {
                num_oscillators: timescale.oscillator_count(),
                coupling_strength: match timescale {
                    TimeHorizon::UltraHigh => 0.1,  // Weak coupling for noise rejection
                    TimeHorizon::High => 0.2,       // Moderate coupling  
                    TimeHorizon::Medium => 0.3,     // Strong coupling for patterns
                    TimeHorizon::Low => 0.4,        // Very strong for trend detection
                    TimeHorizon::Strategic => 0.5,  // Maximum for regime detection
                },
                pattern_threshold: match timescale {
                    TimeHorizon::UltraHigh => 0.85, // High threshold for noise rejection
                    TimeHorizon::High => 0.75,      // High threshold for quality
                    TimeHorizon::Medium => 0.65,    // Medium threshold for sensitivity
                    TimeHorizon::Low => 0.55,       // Lower for trend detection
                    TimeHorizon::Strategic => 0.45, // Lowest for regime shifts
                },
                frequency_range: match timescale {
                    TimeHorizon::UltraHigh => (10.0, 100.0),  // High frequencies
                    TimeHorizon::High => (1.0, 50.0),         // Medium-high frequencies
                    TimeHorizon::Medium => (0.1, 10.0),       // Medium frequencies  
                    TimeHorizon::Low => (0.01, 1.0),          // Low frequencies
                    TimeHorizon::Strategic => (0.001, 0.1),   // Ultra-low frequencies
                },
                time_window_ms: timescale.window_duration().as_millis() as u64,
                adaptive_tuning: true,
                channel_config: Default::default(),
            };

            let processor = Arc::new(RwLock::new(
                DynamicResonancePatternProcessor::new(drpp_config).await?
            ));
            
            let coherence_analyzer = Arc::new(RwLock::new(
                PhaseCoherenceAnalyzer::new(
                    timescale.oscillator_count(),
                    timescale.frequency_bands(),
                )?
            ));

            processors.insert(timescale, processor);
            coherence_analyzers.insert(timescale, coherence_analyzer);
            last_updates.insert(timescale, Instant::now());
        }

        // Initialize cross-timescale flows
        let mut cross_time_flows = HashMap::new();
        for &source in &timescales {
            for &target in &timescales {
                if source != target {
                    cross_time_flows.insert((source, target), CrossTimeInfo {
                        upward_flow: 0.0,
                        downward_flow: 0.0,
                        net_flow: 0.0,
                        mutual_information: 0.0,
                        phase_sync: 0.0,
                        coupling_strength: 0.0,
                    });
                }
            }
        }

        // Initialize global sync state
        let num_timescales = timescales.len();
        let coupling_matrix = vec![vec![0.0; num_timescales]; num_timescales];
        let global_sync_state = Arc::new(RwLock::new(NetworkSyncState {
            global_order: 0.0,
            dominant_regime: MarketRegime::Ranging,
            network_coherence: 0.0,
            coupling_matrix,
            cascade_direction: 0.0,
        }));

        Ok(Self {
            processors,
            coherence_analyzers,
            cross_time_flows,
            propagation_history: VecDeque::with_capacity(max_history),
            max_history,
            global_sync_state,
            last_updates,
        })
    }

    /// Process market data across all timescales
    pub async fn process_multi_timeframe(&mut self, spikes: &[Spike]) -> Result<MultiTimeframeResult> {
        let start_time = Instant::now();
        let mut timescale_results = HashMap::new();
        
        // Process each timescale in parallel
        let mut tasks = Vec::new();
        
        for (&timescale, processor) in &self.processors {
            let processor_clone = Arc::clone(processor);
            let coherence_analyzer = Arc::clone(&self.coherence_analyzers[&timescale]);
            let spikes_clone = spikes.to_vec();
            
            let task = tokio::spawn(async move {
                let result = Self::process_single_timescale(
                    timescale,
                    processor_clone,
                    coherence_analyzer,
                    &spikes_clone,
                ).await;
                (timescale, result)
            });
            
            tasks.push(task);
        }

        // Collect results from all timescales
        for task in tasks {
            let (timescale, result) = task.await??;
            timescale_results.insert(timescale, result);
        }

        // Analyze cross-timescale information flows
        self.analyze_cross_timescale_flows(&timescale_results).await?;

        // Detect pattern propagations
        let propagations = self.detect_pattern_propagations(&timescale_results).await?;

        // Update global synchronization state
        self.update_global_sync_state(&timescale_results).await?;

        // Build comprehensive result
        let global_sync = self.global_sync_state.read().clone();
        
        Ok(MultiTimeframeResult {
            timescale_results,
            cross_time_flows: self.cross_time_flows.clone(),
            pattern_propagations: propagations,
            global_sync_state: global_sync,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Process single timescale
    async fn process_single_timescale(
        timescale: TimeHorizon,
        processor: Arc<RwLock<DynamicResonancePatternProcessor>>,
        coherence_analyzer: Arc<RwLock<PhaseCoherenceAnalyzer>>,
        spikes: &[Spike],
    ) -> Result<TimescaleResult> {
        // Filter spikes relevant to this timescale
        let filtered_spikes = Self::filter_spikes_for_timescale(timescale, spikes);

        // Process through DRPP
        let patterns = {
            let mut proc = processor.write();
            proc.process_spikes(&filtered_spikes).await?
        };

        // Get oscillator phases for coherence analysis
        let phases = {
            let proc = processor.read();
            proc.get_oscillator_phases()
        };

        // Analyze phase coherence
        let coherence_patterns = {
            let mut analyzer = coherence_analyzer.write();
            analyzer.analyze_coherence(&phases)?
        };

        // Classify market regime for this timescale
        let regime = Self::classify_regime_for_timescale(timescale, &coherence_patterns);

        Ok(TimescaleResult {
            timescale,
            patterns,
            coherence_patterns,
            regime,
            oscillator_phases: phases,
            spike_count: filtered_spikes.len(),
        })
    }

    /// Filter spikes relevant to specific timescale
    fn filter_spikes_for_timescale(timescale: TimeHorizon, spikes: &[Spike]) -> Vec<Spike> {
        let window_ns = timescale.window_duration().as_nanos() as u64;
        
        if spikes.is_empty() {
            return vec![];
        }

        let latest_time = spikes.iter().map(|s| s.timestamp_ns).max().unwrap_or(0);
        let cutoff_time = latest_time.saturating_sub(window_ns);

        spikes.iter()
            .filter(|spike| spike.timestamp_ns >= cutoff_time)
            .cloned()
            .collect()
    }

    /// Analyze cross-timescale information flows
    async fn analyze_cross_timescale_flows(
        &mut self,
        results: &HashMap<TimeHorizon, TimescaleResult>,
    ) -> Result<()> {
        for (&source, &target) in self.cross_time_flows.keys() {
            if let (Some(source_result), Some(target_result)) = 
                (results.get(&source), results.get(&target)) {
                
                let flow = self.compute_information_flow(source_result, target_result).await?;
                self.cross_time_flows.insert((source, target), flow);
            }
        }
        Ok(())
    }

    /// Compute information flow between two timescales
    async fn compute_information_flow(
        &self,
        source: &TimescaleResult,
        target: &TimescaleResult,
    ) -> Result<CrossTimeInfo> {
        // Mutual information between phase patterns
        let mutual_info = self.compute_mutual_information(
            &source.oscillator_phases,
            &target.oscillator_phases,
        );

        // Phase synchronization using Kuramoto order parameter
        let phase_sync = self.compute_phase_synchronization(
            &source.oscillator_phases,
            &target.oscillator_phases,
        );

        // Cross-frequency coupling strength
        let coupling_strength = self.compute_cross_frequency_coupling(source, target);

        // Information flow direction based on complexity
        let source_complexity = self.compute_pattern_complexity(&source.patterns);
        let target_complexity = self.compute_pattern_complexity(&target.patterns);
        
        let upward_flow = if source_complexity > target_complexity {
            phase_sync * coupling_strength
        } else {
            0.0
        };

        let downward_flow = if target_complexity > source_complexity {
            phase_sync * coupling_strength
        } else {
            0.0
        };

        Ok(CrossTimeInfo {
            upward_flow,
            downward_flow,
            net_flow: upward_flow - downward_flow,
            mutual_information: mutual_info,
            phase_sync,
            coupling_strength,
        })
    }

    /// Compute mutual information between phase arrays
    fn compute_mutual_information(&self, phases1: &[f64], phases2: &[f64]) -> f64 {
        if phases1.is_empty() || phases2.is_empty() {
            return 0.0;
        }

        let n_bins = 32;
        let min_len = phases1.len().min(phases2.len());
        
        // Discretize phases into bins
        let mut joint_hist = vec![vec![0; n_bins]; n_bins];
        let mut marg1_hist = vec![0; n_bins];
        let mut marg2_hist = vec![0; n_bins];
        
        for i in 0..min_len {
            let bin1 = ((phases1[i] + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * n_bins as f64) as usize;
            let bin2 = ((phases2[i] + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * n_bins as f64) as usize;
            
            let bin1 = bin1.min(n_bins - 1);
            let bin2 = bin2.min(n_bins - 1);
            
            joint_hist[bin1][bin2] += 1;
            marg1_hist[bin1] += 1;
            marg2_hist[bin2] += 1;
        }

        // Compute mutual information
        let mut mi = 0.0;
        let total = min_len as f64;
        
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint_hist[i][j] > 0 && marg1_hist[i] > 0 && marg2_hist[j] > 0 {
                    let p_joint = joint_hist[i][j] as f64 / total;
                    let p_marg1 = marg1_hist[i] as f64 / total;
                    let p_marg2 = marg2_hist[j] as f64 / total;
                    
                    mi += p_joint * (p_joint / (p_marg1 * p_marg2)).ln();
                }
            }
        }

        mi.max(0.0)
    }

    /// Compute phase synchronization using Kuramoto order parameter
    fn compute_phase_synchronization(&self, phases1: &[f64], phases2: &[f64]) -> f64 {
        if phases1.is_empty() || phases2.is_empty() {
            return 0.0;
        }

        let min_len = phases1.len().min(phases2.len());
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for i in 0..min_len {
            let phase_diff = phases1[i] - phases2[i];
            sum_cos += phase_diff.cos();
            sum_sin += phase_diff.sin();
        }

        let order_param = ((sum_cos.powi(2) + sum_sin.powi(2)) / (min_len as f64).powi(2)).sqrt();
        order_param.min(1.0)
    }

    /// Compute cross-frequency coupling strength
    fn compute_cross_frequency_coupling(
        &self,
        source: &TimescaleResult,
        target: &TimescaleResult,
    ) -> f64 {
        // Use coherence patterns to estimate coupling
        let source_coherence = source.coherence_patterns.iter()
            .map(|p| p.coherence_score)
            .fold(0.0, f64::max);
            
        let target_coherence = target.coherence_patterns.iter()
            .map(|p| p.coherence_score)
            .fold(0.0, f64::max);

        // Coupling strength based on coherence overlap
        (source_coherence * target_coherence).sqrt()
    }

    /// Compute pattern complexity for information flow direction
    fn compute_pattern_complexity(&self, patterns: &[Pattern]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        // Complex patterns have higher variability and more types
        let mut pattern_types = std::collections::HashSet::new();
        let mut strength_sum = 0.0;
        let mut strength_sq_sum = 0.0;

        for pattern in patterns {
            pattern_types.insert(pattern.pattern_type);
            strength_sum += pattern.strength;
            strength_sq_sum += pattern.strength.powi(2);
        }

        let mean_strength = strength_sum / patterns.len() as f64;
        let variance = (strength_sq_sum / patterns.len() as f64) - mean_strength.powi(2);
        let diversity = pattern_types.len() as f64;

        // Complexity as combination of strength variance and type diversity
        variance.sqrt() * diversity
    }

    /// Detect pattern propagations across timescales
    async fn detect_pattern_propagations(
        &mut self,
        results: &HashMap<TimeHorizon, TimescaleResult>,
    ) -> Result<Vec<PatternPropagation>> {
        let mut propagations = Vec::new();
        let current_time = Instant::now();

        // Check for pattern similarities across timescales
        for (&source_scale, source_result) in results {
            for (&target_scale, target_result) in results {
                if source_scale == target_scale {
                    continue;
                }

                // Find matching patterns between timescales
                for source_pattern in &source_result.patterns {
                    for target_pattern in &target_result.patterns {
                        if source_pattern.pattern_type == target_pattern.pattern_type {
                            let strength_corr = (source_pattern.strength * target_pattern.strength).sqrt();
                            
                            if strength_corr > 0.3 { // Minimum propagation threshold
                                let propagation = PatternPropagation {
                                    source: source_scale,
                                    target: target_scale,
                                    pattern_type: source_pattern.pattern_type,
                                    strength: strength_corr,
                                    latency_ms: current_time.elapsed().as_millis() as u64,
                                    coherence_preservation: strength_corr,
                                };
                                
                                propagations.push(propagation);
                            }
                        }
                    }
                }
            }
        }

        // Update propagation history
        for prop in &propagations {
            self.propagation_history.push_back(prop.clone());
            if self.propagation_history.len() > self.max_history {
                self.propagation_history.pop_front();
            }
        }

        Ok(propagations)
    }

    /// Update global network synchronization state
    async fn update_global_sync_state(
        &mut self,
        results: &HashMap<TimeHorizon, TimescaleResult>,
    ) -> Result<()> {
        let mut global_sync = self.global_sync_state.write();

        // Compute global order parameter across all timescales
        let mut total_coherence = 0.0;
        let mut regime_votes = HashMap::new();
        let mut timescale_count = 0;

        for result in results.values() {
            // Average coherence across patterns
            let timescale_coherence = if result.coherence_patterns.is_empty() {
                0.0
            } else {
                result.coherence_patterns.iter()
                    .map(|p| p.coherence_score)
                    .sum::<f64>() / result.coherence_patterns.len() as f64
            };

            total_coherence += timescale_coherence;
            timescale_count += 1;

            // Vote for dominant regime
            *regime_votes.entry(result.regime).or_insert(0) += 1;
        }

        global_sync.global_order = if timescale_count > 0 {
            total_coherence / timescale_count as f64
        } else {
            0.0
        };

        // Determine dominant regime by majority vote
        global_sync.dominant_regime = regime_votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(regime, _)| regime)
            .unwrap_or(MarketRegime::Ranging);

        // Network coherence based on cross-timescale flows
        let flow_coherence = self.cross_time_flows.values()
            .map(|flow| flow.phase_sync)
            .fold(0.0, f64::max);

        global_sync.network_coherence = (global_sync.global_order + flow_coherence) / 2.0;

        // Cascade direction: positive = upward (fast->slow), negative = downward (slow->fast)  
        let net_flow = self.cross_time_flows.values()
            .map(|flow| flow.net_flow)
            .sum::<f64>();

        global_sync.cascade_direction = net_flow;

        Ok(())
    }

    /// Classify market regime for specific timescale
    fn classify_regime_for_timescale(
        timescale: TimeHorizon,
        coherence_patterns: &[CoherencePattern],
    ) -> MarketRegime {
        if coherence_patterns.is_empty() {
            return MarketRegime::Ranging;
        }

        // Weight regimes based on timescale characteristics
        let mut regime_scores = HashMap::new();

        for pattern in coherence_patterns {
            let weight = match timescale {
                TimeHorizon::UltraHigh => {
                    // Ultra-high frequency: sensitive to chaotic patterns
                    match pattern.regime {
                        MarketRegime::Chaotic => 2.0,
                        MarketRegime::Transitional => 1.5,
                        _ => 1.0,
                    }
                },
                TimeHorizon::High => {
                    // High frequency: balanced sensitivity
                    1.0
                },
                TimeHorizon::Medium => {
                    // Medium frequency: trend-sensitive
                    match pattern.regime {
                        MarketRegime::Trending => 2.0,
                        MarketRegime::RegimeShift => 1.5,
                        _ => 1.0,
                    }
                },
                TimeHorizon::Low => {
                    // Low frequency: regime-sensitive
                    match pattern.regime {
                        MarketRegime::RegimeShift => 3.0,
                        MarketRegime::Trending => 2.0,
                        _ => 1.0,
                    }
                },
                TimeHorizon::Strategic => {
                    // Strategic: regime shift detection
                    match pattern.regime {
                        MarketRegime::RegimeShift => 5.0,
                        _ => 1.0,
                    }
                },
            };

            let score = pattern.coherence_score * weight;
            *regime_scores.entry(pattern.regime).or_insert(0.0) += score;
        }

        regime_scores.into_iter()
            .max_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
            .map(|(regime, _)| regime)
            .unwrap_or(MarketRegime::Ranging)
    }

    /// Get current global synchronization state
    pub fn get_global_sync_state(&self) -> NetworkSyncState {
        self.global_sync_state.read().clone()
    }

    /// Get cross-timescale information flows
    pub fn get_cross_time_flows(&self) -> &HashMap<(TimeHorizon, TimeHorizon), CrossTimeInfo> {
        &self.cross_time_flows
    }

    /// Get pattern propagation history
    pub fn get_propagation_history(&self) -> Vec<PatternPropagation> {
        self.propagation_history.iter().cloned().collect()
    }
}

/// Result from single timescale processing
#[derive(Debug, Clone)]
pub struct TimescaleResult {
    pub timescale: TimeHorizon,
    pub patterns: Vec<Pattern>,
    pub coherence_patterns: Vec<CoherencePattern>,
    pub regime: MarketRegime,
    pub oscillator_phases: Vec<f64>,
    pub spike_count: usize,
}

/// Comprehensive multi-timeframe processing result
#[derive(Debug, Clone)]
pub struct MultiTimeframeResult {
    /// Results per timescale
    pub timescale_results: HashMap<TimeHorizon, TimescaleResult>,
    /// Cross-timescale information flows
    pub cross_time_flows: HashMap<(TimeHorizon, TimeHorizon), CrossTimeInfo>,
    /// Detected pattern propagations
    pub pattern_propagations: Vec<PatternPropagation>,
    /// Global network synchronization state
    pub global_sync_state: NetworkSyncState,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike_encoding::{Spike, NeuronType};

    #[tokio::test]
    async fn test_multi_timeframe_network_creation() {
        let network = MultiTimeframeNetwork::new(1000).await.unwrap();
        
        // Should have processors for all timescales
        assert_eq!(network.processors.len(), 5);
        assert_eq!(network.coherence_analyzers.len(), 5);
        
        // Should have cross-flows for all pairs (5*4 = 20 flows)
        assert_eq!(network.cross_time_flows.len(), 20);
    }

    #[tokio::test]  
    async fn test_timescale_properties() {
        assert_eq!(TimeHorizon::UltraHigh.window_duration(), Duration::from_millis(100));
        assert_eq!(TimeHorizon::High.window_duration(), Duration::from_secs(1));
        assert_eq!(TimeHorizon::Medium.window_duration(), Duration::from_secs(60));
        
        assert_eq!(TimeHorizon::UltraHigh.oscillator_count(), 512);
        assert_eq!(TimeHorizon::Strategic.oscillator_count(), 32);
    }

    #[tokio::test]
    async fn test_spike_filtering() {
        let spikes = vec![
            Spike { neuron_id: 0, timestamp_ns: 1000000000, strength: 0.5 }, // 1 sec
            Spike { neuron_id: 1, timestamp_ns: 1500000000, strength: 0.6 }, // 1.5 sec
            Spike { neuron_id: 2, timestamp_ns: 2000000000, strength: 0.7 }, // 2 sec
        ];

        let filtered = MultiTimeframeNetwork::filter_spikes_for_timescale(
            TimeHorizon::High, // 1-second window
            &spikes
        );

        // Should include spikes from last second (1.5s and 2s)
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].timestamp_ns, 1500000000);
        assert_eq!(filtered[1].timestamp_ns, 2000000000);
    }

    #[tokio::test]
    async fn test_phase_synchronization() {
        let network = MultiTimeframeNetwork::new(1000).await.unwrap();
        
        // Perfect synchronization
        let phases1 = vec![0.0, 1.0, 2.0, 3.0];
        let phases2 = vec![0.1, 1.1, 2.1, 3.1]; // Small phase difference
        
        let sync = network.compute_phase_synchronization(&phases1, &phases2);
        assert!(sync > 0.9); // High synchronization

        // Random phases (low synchronization)
        let phases3 = vec![0.0, 2.5, 1.8, 4.2];
        let sync_low = network.compute_phase_synchronization(&phases1, &phases3);
        assert!(sync_low < sync);
    }
}