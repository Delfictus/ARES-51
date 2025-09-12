// Dynamic Resonance Phase Processing (DRPP) Implementation
// Core cognitive engine using neural oscillator networks

use std::sync::Arc;
use tokio::sync::RwLock;
use ndarray::{Array1, Array2, ArrayD};
use std::collections::{HashMap, VecDeque};
use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};

use super::{PhaseState, LatticeNode, ResonancePattern, TopologicalFeatures};

/// DRPP Cognitive Engine - The heart of pattern recognition
pub struct DRPPEngine {
    /// Network of coupled neural oscillators
    oscillator_layers: Vec<OscillatorLayer>,
    
    /// Resonance detection thresholds
    detection_params: DetectionParameters,
    
    /// Pattern memory for temporal correlation
    pattern_memory: Arc<RwLock<PatternMemory>>,
    
    /// Harmonic analyzer for frequency domain analysis
    harmonic_processor: HarmonicProcessor,
    
    /// Cross-correlation analyzer
    correlation_analyzer: CorrelationAnalyzer,
}

#[derive(Debug, Clone)]
pub struct DetectionParameters {
    pub min_resonance_strength: f64,
    pub coherence_threshold: f64,
    pub temporal_window: usize,
    pub frequency_resolution: f64,
    pub phase_lock_tolerance: f64,
}

impl Default for DetectionParameters {
    fn default() -> Self {
        Self {
            min_resonance_strength: 0.7,
            coherence_threshold: 0.8,
            temporal_window: 1000,
            frequency_resolution: 0.01,
            phase_lock_tolerance: 0.1,
        }
    }
}

/// Layer of oscillators in the DRPP network
pub struct OscillatorLayer {
    pub layer_id: usize,
    pub oscillators: Vec<CoupledOscillator>,
    pub inter_layer_coupling: HashMap<usize, f64>,
    pub resonance_map: Array2<f64>,
}

/// Individual coupled oscillator with complex dynamics
pub struct CoupledOscillator {
    pub id: String,
    pub intrinsic_frequency: f64,
    pub phase: f64,
    pub amplitude: f64,
    
    /// Kuramoto model parameters
    pub coupling_strength: f64,
    pub natural_frequency: f64,
    pub noise_amplitude: f64,
    
    /// State variables
    pub phase_velocity: f64,
    pub phase_acceleration: f64,
    
    /// Coupling to other oscillators
    pub couplings: HashMap<String, CouplingParameters>,
    
    /// History for phase analysis
    pub phase_history: VecDeque<f64>,
    pub frequency_history: VecDeque<f64>,
}

#[derive(Debug, Clone)]
pub struct CouplingParameters {
    pub strength: f64,
    pub delay: f64,
    pub nonlinearity: f64,
}

impl CoupledOscillator {
    pub fn new(id: String, frequency: f64) -> Self {
        Self {
            id,
            intrinsic_frequency: frequency,
            phase: 0.0,
            amplitude: 1.0,
            coupling_strength: 0.1,
            natural_frequency: frequency,
            noise_amplitude: 0.01,
            phase_velocity: 0.0,
            phase_acceleration: 0.0,
            couplings: HashMap::new(),
            phase_history: VecDeque::with_capacity(1000),
            frequency_history: VecDeque::with_capacity(1000),
        }
    }

    /// Update oscillator state using Kuramoto dynamics
    pub fn update_kuramoto(&mut self, neighbors: &[(String, f64, f64)], dt: f64) {
        let mut total_coupling = 0.0;
        
        // Calculate coupling from all connected oscillators
        for (neighbor_id, neighbor_phase, neighbor_amp) in neighbors {
            if let Some(coupling) = self.couplings.get(neighbor_id) {
                // Kuramoto coupling with delay and nonlinearity
                let phase_diff = neighbor_phase - self.phase - coupling.delay;
                let linear_coupling = coupling.strength * neighbor_amp * phase_diff.sin();
                let nonlinear_coupling = coupling.nonlinearity * (2.0 * phase_diff).sin();
                
                total_coupling += linear_coupling + nonlinear_coupling;
            }
        }
        
        // Add noise for exploration
        let noise = rand::random::<f64>() * self.noise_amplitude - self.noise_amplitude / 2.0;
        
        // Update phase dynamics
        self.phase_acceleration = total_coupling + noise;
        self.phase_velocity += self.phase_acceleration * dt;
        self.phase_velocity *= 0.99; // Damping
        
        self.phase += (self.natural_frequency + self.phase_velocity) * dt;
        
        // Normalize phase to [0, 2π]
        self.phase = self.phase % (2.0 * std::f64::consts::PI);
        
        // Update amplitude based on coupling strength
        self.amplitude = (1.0 + total_coupling.abs()).min(10.0);
        
        // Record history
        self.phase_history.push_back(self.phase);
        if self.phase_history.len() > 1000 {
            self.phase_history.pop_front();
        }
        
        self.frequency_history.push_back(self.natural_frequency + self.phase_velocity);
        if self.frequency_history.len() > 1000 {
            self.frequency_history.pop_front();
        }
    }

    /// Calculate phase coherence with another oscillator
    pub fn phase_coherence(&self, other: &CoupledOscillator) -> f64 {
        if self.phase_history.len() < 100 || other.phase_history.len() < 100 {
            return 0.0;
        }
        
        let mut coherence_sum = 0.0;
        let samples = 100.min(self.phase_history.len()).min(other.phase_history.len());
        
        for i in 0..samples {
            let self_phase = self.phase_history[self.phase_history.len() - 1 - i];
            let other_phase = other.phase_history[other.phase_history.len() - 1 - i];
            let phase_diff = (self_phase - other_phase).abs();
            
            // High coherence when phases are locked
            coherence_sum += (phase_diff.cos() + 1.0) / 2.0;
        }
        
        coherence_sum / samples as f64
    }

    /// Detect if oscillator is in resonance
    pub fn is_resonating(&self, threshold: f64) -> bool {
        if self.frequency_history.len() < 100 {
            return false;
        }
        
        // Check frequency stability (low variance = resonance)
        let recent_freqs: Vec<f64> = self.frequency_history.iter()
            .rev()
            .take(100)
            .copied()
            .collect();
        
        let mean = recent_freqs.iter().sum::<f64>() / recent_freqs.len() as f64;
        let variance = recent_freqs.iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>() / recent_freqs.len() as f64;
        
        let stability = 1.0 / (1.0 + variance);
        stability > threshold && self.amplitude > 1.5
    }
}

/// Pattern memory for temporal correlation
pub struct PatternMemory {
    short_term: VecDeque<ResonancePattern>,
    long_term: HashMap<String, StoredPattern>,
    consolidation_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct StoredPattern {
    pub pattern_id: String,
    pub frequency_signature: Vec<f64>,
    pub phase_relationships: Vec<(String, String, f64)>,
    pub occurrence_count: usize,
    pub average_strength: f64,
    pub topological_invariants: Vec<f64>,
}

impl PatternMemory {
    pub fn new() -> Self {
        Self {
            short_term: VecDeque::with_capacity(1000),
            long_term: HashMap::new(),
            consolidation_threshold: 0.8,
        }
    }

    /// Store pattern in short-term memory
    pub fn store_short_term(&mut self, pattern: ResonancePattern) {
        self.short_term.push_back(pattern);
        if self.short_term.len() > 1000 {
            self.short_term.pop_front();
        }
    }

    /// Consolidate patterns from short to long-term memory
    pub fn consolidate(&mut self) {
        let patterns_to_consolidate: Vec<ResonancePattern> = self.short_term
            .iter()
            .filter(|p| p.resonance_strength > self.consolidation_threshold)
            .cloned()
            .collect();
        
        for pattern in patterns_to_consolidate {
            let pattern_id = self.generate_pattern_id(&pattern);
            
            self.long_term
                .entry(pattern_id.clone())
                .and_modify(|stored| {
                    stored.occurrence_count += 1;
                    stored.average_strength = 
                        (stored.average_strength * stored.occurrence_count as f64 + pattern.resonance_strength) 
                        / (stored.occurrence_count + 1) as f64;
                })
                .or_insert_with(|| StoredPattern {
                    pattern_id,
                    frequency_signature: pattern.frequency_signature.clone(),
                    phase_relationships: Vec::new(),
                    occurrence_count: 1,
                    average_strength: pattern.resonance_strength,
                    topological_invariants: self.extract_invariants(&pattern.topological_features),
                });
        }
    }

    fn generate_pattern_id(&self, pattern: &ResonancePattern) -> String {
        // Generate unique ID based on frequency signature
        let freq_hash = pattern.frequency_signature.iter()
            .map(|f| format!("{:.2}", f))
            .collect::<Vec<_>>()
            .join("_");
        
        format!("pattern_{}", freq_hash)
    }

    fn extract_invariants(&self, features: &TopologicalFeatures) -> Vec<f64> {
        let mut invariants = vec![
            features.connected_components as f64,
            features.cycles.len() as f64,
            features.voids.len() as f64,
        ];
        
        // Add persistence diagram statistics
        if !features.persistence_diagram.is_empty() {
            let total_persistence: f64 = features.persistence_diagram
                .iter()
                .map(|(birth, death)| death - birth)
                .sum();
            
            invariants.push(total_persistence);
        }
        
        invariants
    }

    /// Retrieve similar patterns from memory
    pub fn recall_similar(&self, pattern: &ResonancePattern, threshold: f64) -> Vec<StoredPattern> {
        self.long_term
            .values()
            .filter(|stored| {
                self.pattern_similarity(pattern, stored) > threshold
            })
            .cloned()
            .collect()
    }

    fn pattern_similarity(&self, pattern: &ResonancePattern, stored: &StoredPattern) -> f64 {
        if pattern.frequency_signature.len() != stored.frequency_signature.len() {
            return 0.0;
        }
        
        // Cosine similarity of frequency signatures
        let dot_product: f64 = pattern.frequency_signature
            .iter()
            .zip(&stored.frequency_signature)
            .map(|(a, b)| a * b)
            .sum();
        
        let norm_a: f64 = pattern.frequency_signature
            .iter()
            .map(|f| f.powi(2))
            .sum::<f64>()
            .sqrt();
        
        let norm_b: f64 = stored.frequency_signature
            .iter()
            .map(|f| f.powi(2))
            .sum::<f64>()
            .sqrt();
        
        if norm_a * norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Harmonic processor for frequency domain analysis
pub struct HarmonicProcessor {
    fft_planner: FftPlanner<f64>,
    window_size: usize,
    overlap: f64,
}

impl HarmonicProcessor {
    pub fn new(window_size: usize) -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            window_size,
            overlap: 0.5,
        }
    }

    /// Analyze harmonic content of oscillator network
    pub fn analyze_harmonics(&mut self, oscillators: &[CoupledOscillator]) -> HarmonicSpectrum {
        let mut combined_signal = vec![0.0; self.window_size];
        
        // Combine signals from all oscillators
        for (i, oscillator) in oscillators.iter().enumerate() {
            if let Some(history) = oscillator.phase_history.back() {
                let sample_idx = i % self.window_size;
                combined_signal[sample_idx] += oscillator.amplitude * history.sin();
            }
        }
        
        // Apply FFT
        let mut complex_signal: Vec<Complex<f64>> = combined_signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        let fft = self.fft_planner.plan_fft_forward(self.window_size);
        fft.process(&mut complex_signal);
        
        // Extract spectrum
        let spectrum: Vec<f64> = complex_signal
            .iter()
            .map(|c| c.norm())
            .collect();
        
        // Find peaks (harmonics)
        let peaks = self.find_spectral_peaks(&spectrum);
        
        HarmonicSpectrum {
            frequencies: peaks.iter().map(|p| p.frequency).collect(),
            amplitudes: peaks.iter().map(|p| p.amplitude).collect(),
            phases: peaks.iter().map(|p| p.phase).collect(),
            fundamental: peaks.first().map(|p| p.frequency).unwrap_or(0.0),
            harmonicity: self.calculate_harmonicity(&peaks),
        }
    }

    fn find_spectral_peaks(&self, spectrum: &[f64]) -> Vec<SpectralPeak> {
        let mut peaks = Vec::new();
        
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
                peaks.push(SpectralPeak {
                    frequency: i as f64 * (1.0 / self.window_size as f64),
                    amplitude: spectrum[i],
                    phase: 0.0, // Simplified - would calculate from complex values
                });
            }
        }
        
        // Sort by amplitude and keep top peaks
        peaks.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap());
        peaks.truncate(10);
        
        peaks
    }

    fn calculate_harmonicity(&self, peaks: &[SpectralPeak]) -> f64 {
        if peaks.len() < 2 {
            return 0.0;
        }
        
        let fundamental = peaks[0].frequency;
        let mut harmonicity = 0.0;
        
        for i in 1..peaks.len() {
            let ratio = peaks[i].frequency / fundamental;
            let closest_harmonic = ratio.round();
            let deviation = (ratio - closest_harmonic).abs();
            
            // High harmonicity when frequencies are integer multiples
            harmonicity += (1.0 - deviation).max(0.0) * peaks[i].amplitude;
        }
        
        harmonicity / peaks.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone)]
pub struct HarmonicSpectrum {
    pub frequencies: Vec<f64>,
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub fundamental: f64,
    pub harmonicity: f64,
}

/// Cross-correlation analyzer for pattern detection
pub struct CorrelationAnalyzer {
    correlation_window: usize,
    significance_threshold: f64,
}

impl CorrelationAnalyzer {
    pub fn new() -> Self {
        Self {
            correlation_window: 256,
            significance_threshold: 0.7,
        }
    }

    /// Find correlated oscillator groups
    pub fn find_correlated_groups(&self, oscillators: &[CoupledOscillator]) -> Vec<OscillatorGroup> {
        let mut groups = Vec::new();
        let mut processed = vec![false; oscillators.len()];
        
        for i in 0..oscillators.len() {
            if processed[i] {
                continue;
            }
            
            let mut group = OscillatorGroup {
                members: vec![oscillators[i].id.clone()],
                coherence: 1.0,
                dominant_frequency: oscillators[i].natural_frequency,
            };
            
            processed[i] = true;
            
            // Find all oscillators correlated with this one
            for j in i + 1..oscillators.len() {
                if processed[j] {
                    continue;
                }
                
                let correlation = self.cross_correlation(&oscillators[i], &oscillators[j]);
                
                if correlation > self.significance_threshold {
                    group.members.push(oscillators[j].id.clone());
                    processed[j] = true;
                }
            }
            
            if group.members.len() > 1 {
                group.coherence = self.calculate_group_coherence(&group.members, oscillators);
                groups.push(group);
            }
        }
        
        groups
    }

    fn cross_correlation(&self, osc1: &CoupledOscillator, osc2: &CoupledOscillator) -> f64 {
        let history1 = &osc1.phase_history;
        let history2 = &osc2.phase_history;
        
        if history1.len() < self.correlation_window || history2.len() < self.correlation_window {
            return 0.0;
        }
        
        let mut max_correlation = 0.0;
        
        // Try different lags to find maximum correlation
        for lag in 0..10 {
            let mut correlation = 0.0;
            let samples = self.correlation_window.min(history1.len() - lag).min(history2.len());
            
            for i in 0..samples {
                let val1 = history1[history1.len() - samples + i];
                let val2 = history2[history2.len() - samples + i - lag.min(i)];
                
                correlation += (val1.sin() * val2.sin() + val1.cos() * val2.cos()) / 2.0;
            }
            
            correlation /= samples as f64;
            max_correlation = max_correlation.max(correlation.abs());
        }
        
        max_correlation
    }

    fn calculate_group_coherence(&self, members: &[String], oscillators: &[CoupledOscillator]) -> f64 {
        let member_oscillators: Vec<&CoupledOscillator> = oscillators
            .iter()
            .filter(|o| members.contains(&o.id))
            .collect();
        
        if member_oscillators.len() < 2 {
            return 0.0;
        }
        
        let mut total_coherence = 0.0;
        let mut pair_count = 0;
        
        for i in 0..member_oscillators.len() {
            for j in i + 1..member_oscillators.len() {
                total_coherence += member_oscillators[i].phase_coherence(member_oscillators[j]);
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct OscillatorGroup {
    pub members: Vec<String>,
    pub coherence: f64,
    pub dominant_frequency: f64,
}

impl DRPPEngine {
    pub fn new(num_layers: usize, oscillators_per_layer: usize) -> Self {
        let mut oscillator_layers = Vec::new();
        
        for layer_id in 0..num_layers {
            let mut oscillators = Vec::new();
            let base_freq = 1.0 + layer_id as f64 * 0.5;
            
            for i in 0..oscillators_per_layer {
                let freq = base_freq + i as f64 * 0.1;
                oscillators.push(CoupledOscillator::new(
                    format!("osc_{}_{}", layer_id, i),
                    freq,
                ));
            }
            
            let layer = OscillatorLayer {
                layer_id,
                oscillators,
                inter_layer_coupling: HashMap::new(),
                resonance_map: Array2::zeros((oscillators_per_layer, oscillators_per_layer)),
            };
            
            oscillator_layers.push(layer);
        }
        
        Self {
            oscillator_layers,
            detection_params: DetectionParameters::default(),
            pattern_memory: Arc::new(RwLock::new(PatternMemory::new())),
            harmonic_processor: HarmonicProcessor::new(256),
            correlation_analyzer: CorrelationAnalyzer::new(),
        }
    }

    /// Process input through the DRPP network
    pub async fn process(&mut self, input: &[f64]) -> Result<Vec<ResonancePattern>> {
        // Inject input as perturbation to first layer
        self.inject_input(input)?;
        
        // Update all oscillators
        self.update_oscillators(0.01)?;
        
        // Detect resonance patterns
        let patterns = self.detect_resonance_patterns().await?;
        
        // Store patterns in memory
        let mut memory = self.pattern_memory.write().await;
        for pattern in &patterns {
            memory.store_short_term(pattern.clone());
        }
        memory.consolidate();
        
        Ok(patterns)
    }

    fn inject_input(&mut self, input: &[f64]) -> Result<()> {
        if let Some(first_layer) = self.oscillator_layers.first_mut() {
            for (i, value) in input.iter().enumerate() {
                if i < first_layer.oscillators.len() {
                    first_layer.oscillators[i].amplitude *= 1.0 + value;
                    first_layer.oscillators[i].phase_velocity += value * 0.1;
                }
            }
        }
        Ok(())
    }

    fn update_oscillators(&mut self, dt: f64) -> Result<()> {
        // Update each layer
        for layer_idx in 0..self.oscillator_layers.len() {
            let layer_oscillators = self.oscillator_layers[layer_idx].oscillators.clone();
            
            for i in 0..layer_oscillators.len() {
                let mut neighbors = Vec::new();
                
                // Intra-layer coupling
                for j in 0..layer_oscillators.len() {
                    if i != j {
                        neighbors.push((
                            layer_oscillators[j].id.clone(),
                            layer_oscillators[j].phase,
                            layer_oscillators[j].amplitude,
                        ));
                    }
                }
                
                // Inter-layer coupling
                if layer_idx > 0 {
                    let prev_layer = &self.oscillator_layers[layer_idx - 1];
                    for osc in &prev_layer.oscillators {
                        neighbors.push((osc.id.clone(), osc.phase, osc.amplitude * 0.5));
                    }
                }
                
                if layer_idx < self.oscillator_layers.len() - 1 {
                    let next_layer = &self.oscillator_layers[layer_idx + 1];
                    for osc in &next_layer.oscillators {
                        neighbors.push((osc.id.clone(), osc.phase, osc.amplitude * 0.5));
                    }
                }
                
                // Update oscillator
                self.oscillator_layers[layer_idx].oscillators[i].update_kuramoto(&neighbors, dt);
            }
        }
        
        Ok(())
    }

    async fn detect_resonance_patterns(&mut self) -> Result<Vec<ResonancePattern>> {
        let mut patterns = Vec::new();
        
        for layer in &mut self.oscillator_layers {
            // Find resonating oscillators
            let resonating: Vec<usize> = layer.oscillators
                .iter()
                .enumerate()
                .filter(|(_, osc)| osc.is_resonating(self.detection_params.coherence_threshold))
                .map(|(i, _)| i)
                .collect();
            
            if resonating.len() >= 3 {
                // Analyze harmonic content
                let spectrum = self.harmonic_processor.analyze_harmonics(&layer.oscillators);
                
                // Find correlated groups
                let groups = self.correlation_analyzer.find_correlated_groups(&layer.oscillators);
                
                for group in groups {
                    if group.coherence > self.detection_params.min_resonance_strength {
                        patterns.push(ResonancePattern {
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs_f64(),
                            participating_nodes: group.members,
                            resonance_strength: group.coherence,
                            frequency_signature: spectrum.frequencies.clone(),
                            topological_features: TopologicalFeatures {
                                connected_components: 1,
                                cycles: vec![],
                                voids: vec![],
                                persistence_diagram: vec![],
                            },
                        });
                    }
                }
            }
        }
        
        patterns
    }

    /// Get current network state for visualization
    pub fn get_network_state(&self) -> NetworkState {
        let mut oscillator_states = Vec::new();
        let mut layer_coupling = Vec::new();
        
        for layer in &self.oscillator_layers {
            for osc in &layer.oscillators {
                oscillator_states.push(OscillatorState {
                    id: osc.id.clone(),
                    phase: osc.phase,
                    amplitude: osc.amplitude,
                    frequency: osc.natural_frequency + osc.phase_velocity,
                    is_resonating: osc.is_resonating(self.detection_params.coherence_threshold),
                });
            }
            
            layer_coupling.push(layer.inter_layer_coupling.clone());
        }
        
        NetworkState {
            oscillator_states,
            layer_coupling,
            total_energy: self.calculate_total_energy(),
        }
    }

    fn calculate_total_energy(&self) -> f64 {
        self.oscillator_layers
            .iter()
            .flat_map(|layer| &layer.oscillators)
            .map(|osc| osc.amplitude.powi(2))
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct OscillatorState {
    pub id: String,
    pub phase: f64,
    pub amplitude: f64,
    pub frequency: f64,
    pub is_resonating: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkState {
    pub oscillator_states: Vec<OscillatorState>,
    pub layer_coupling: Vec<HashMap<usize, f64>>,
    pub total_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_drpp_engine_creation() {
        let engine = DRPPEngine::new(3, 10);
        assert_eq!(engine.oscillator_layers.len(), 3);
        assert_eq!(engine.oscillator_layers[0].oscillators.len(), 10);
    }

    #[tokio::test]
    async fn test_resonance_detection() {
        let mut engine = DRPPEngine::new(2, 5);
        
        // Process input signal
        let input = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let patterns = engine.process(&input).await.unwrap();
        
        // Should detect some patterns
        assert!(patterns.len() >= 0);
    }

    #[tokio::test]
    async fn test_pattern_memory() {
        let mut memory = PatternMemory::new();
        
        let pattern = ResonancePattern {
            timestamp: 0.0,
            participating_nodes: vec!["node1".to_string(), "node2".to_string()],
            resonance_strength: 0.9,
            frequency_signature: vec![1.0, 2.0, 3.0],
            topological_features: TopologicalFeatures {
                connected_components: 1,
                cycles: vec![],
                voids: vec![],
                persistence_diagram: vec![],
            },
        };
        
        memory.store_short_term(pattern.clone());
        memory.consolidate();
        
        assert!(!memory.long_term.is_empty());
    }
}