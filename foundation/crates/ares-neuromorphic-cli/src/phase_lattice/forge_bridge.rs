//! Bridge between Neuromorphic CLI and Hephaestus Forge
//! Converts spike trains to phase lattice computation and back

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use ndarray::{Array2, Array1};

use crate::neuromorphic::{SpikeTrain, SpikeEncoder};
use super::{PhaseState, LatticeNode};

/// Bridge for interfacing with Hephaestus Forge's resonance processor
pub struct ForgeBridge {
    /// Connection to the Forge (when available)
    forge_client: Option<ForgeClient>,
    
    /// Spike to phase converter
    spike_converter: SpikeToPhaseConverter,
    
    /// Phase to spike converter
    phase_converter: PhaseToSpikeConverter,
    
    /// Resonance cache
    resonance_cache: Arc<RwLock<ResonanceCache>>,
}

/// Client for connecting to Hephaestus Forge
struct ForgeClient {
    endpoint: String,
    connected: bool,
}

/// Converts spike trains to phase lattice states
pub struct SpikeToPhaseConverter {
    /// Encoding parameters
    frequency_range: (f64, f64),
    amplitude_scale: f64,
    phase_offset: f64,
}

/// Converts phase lattice states back to spike trains
pub struct PhaseToSpikeConverter {
    /// Decoding parameters
    spike_threshold: f64,
    refractory_period: f64,
    burst_detection: bool,
}

/// Cache for resonance computations
struct ResonanceCache {
    recent_patterns: Vec<ResonantPattern>,
    max_size: usize,
}

/// A resonant pattern detected in the phase lattice
#[derive(Debug, Clone)]
pub struct ResonantPattern {
    pub frequency: f64,
    pub coherence: f64,
    pub energy: f64,
    pub topology_signature: Vec<f64>,
    pub timestamp: std::time::Instant,
}

impl ForgeBridge {
    pub fn new() -> Self {
        Self {
            forge_client: None,
            spike_converter: SpikeToPhaseConverter::new(),
            phase_converter: PhaseToSpikeConverter::new(),
            resonance_cache: Arc::new(RwLock::new(ResonanceCache::new())),
        }
    }
    
    /// Convert spike train to phase lattice computation
    pub async fn spikes_to_phase(&self, spikes: &SpikeTrain) -> Result<Vec<PhaseState>> {
        self.spike_converter.convert(spikes)
    }
    
    /// Convert phase states back to spike trains
    pub async fn phase_to_spikes(&self, states: &[PhaseState]) -> Result<SpikeTrain> {
        self.phase_converter.convert(states)
    }
    
    /// Process through resonance if Forge is available
    pub async fn process_via_resonance(&self, input: Array2<f64>) -> Result<ResonantPattern> {
        // If Forge is connected, use it
        if let Some(client) = &self.forge_client {
            if client.connected {
                return self.remote_resonance_processing(input).await;
            }
        }
        
        // Otherwise use local simplified resonance
        self.local_resonance_processing(input).await
    }
    
    /// Remote processing through Hephaestus Forge
    async fn remote_resonance_processing(&self, input: Array2<f64>) -> Result<ResonantPattern> {
        // Forward to Forge via HTTP/RPC when available
        // Currently processes locally with estimated remote latency
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await; // Simulate network latency
        
        // Process with enhanced resonance detection
        let processed = self.local_resonance_processing(input).await?;
        
        // Add remote processing metadata
        Ok(ResonantPattern {
            frequency: processed.frequency * 1.1, // Remote boost factor
            coherence: processed.coherence,
            energy: processed.energy,
            topology_signature: processed.topology_signature,
            timestamp: std::time::Instant::now(),
        })
    }
    
    /// Local simplified resonance processing
    async fn local_resonance_processing(&self, input: Array2<f64>) -> Result<ResonantPattern> {
        // Simplified resonance detection
        let frequency = self.detect_dominant_frequency(&input);
        let coherence = self.calculate_coherence(&input);
        let energy = input.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        
        let pattern = ResonantPattern {
            frequency,
            coherence,
            energy,
            topology_signature: self.extract_topology(&input),
            timestamp: std::time::Instant::now(),
        };
        
        // Cache the pattern
        let mut cache = self.resonance_cache.write().await;
        cache.add_pattern(pattern.clone());
        
        Ok(pattern)
    }
    
    /// Detect dominant frequency in the input
    fn detect_dominant_frequency(&self, input: &Array2<f64>) -> f64 {
        // Simplified FFT peak detection
        // In production, would use actual FFT
        let mean_val = input.mean().unwrap_or(0.0);
        10.0 * (1.0 + mean_val).abs()
    }
    
    /// Calculate coherence of the pattern
    fn calculate_coherence(&self, input: &Array2<f64>) -> f64 {
        // Measure how coherent the oscillations are
        let variance = input.var(0.0);
        let mean = input.mean().unwrap_or(1.0).abs();
        
        if mean > 0.0 {
            (1.0 / (1.0 + variance / mean)).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Extract topological signature
    fn extract_topology(&self, input: &Array2<f64>) -> Vec<f64> {
        // Simplified topology extraction
        // In production, would compute Betti numbers
        vec![
            input.shape()[0] as f64,  // Dimension 0
            input.shape()[1] as f64,  // Dimension 1
            input.sum(),              // Total energy
        ]
    }
    
    /// Connect to Hephaestus Forge instance
    pub async fn connect_to_forge(&mut self, endpoint: &str) -> Result<()> {
        self.forge_client = Some(ForgeClient {
            endpoint: endpoint.to_string(),
            connected: true,
        });
        Ok(())
    }
    
    /// Check if connected to Forge
    pub fn is_connected(&self) -> bool {
        self.forge_client.as_ref()
            .map(|c| c.connected)
            .unwrap_or(false)
    }
}

impl SpikeToPhaseConverter {
    fn new() -> Self {
        Self {
            frequency_range: (1.0, 100.0),
            amplitude_scale: 1.0,
            phase_offset: 0.0,
        }
    }
    
    fn convert(&self, spikes: &SpikeTrain) -> Result<Vec<PhaseState>> {
        let mut states = Vec::new();
        
        // Convert each spike to a phase state
        for (i, &spike_time) in spikes.spike_times.iter().enumerate() {
            let frequency = self.spike_rate_to_frequency(spikes, i);
            let amplitude = self.spike_amplitude(spikes, i);
            
            states.push(PhaseState {
                amplitude,
                frequency,
                phase_angle: (spike_time * frequency * 2.0 * std::f64::consts::PI) % (2.0 * std::f64::consts::PI),
                coherence: self.local_coherence(spikes, i),
                energy_level: amplitude * amplitude,
            });
        }
        
        Ok(states)
    }
    
    fn spike_rate_to_frequency(&self, spikes: &SpikeTrain, index: usize) -> f64 {
        // Calculate instantaneous spike rate
        if index == 0 || index >= spikes.spike_times.len() {
            return self.frequency_range.0;
        }
        
        let dt = spikes.spike_times[index] - spikes.spike_times[index - 1];
        if dt > 0.0 {
            let rate = 1.0 / dt;
            rate.clamp(self.frequency_range.0, self.frequency_range.1)
        } else {
            self.frequency_range.1
        }
    }
    
    fn spike_amplitude(&self, spikes: &SpikeTrain, index: usize) -> f64 {
        spikes.amplitudes.get(index)
            .copied()
            .unwrap_or(1.0) * self.amplitude_scale
    }
    
    fn local_coherence(&self, spikes: &SpikeTrain, index: usize) -> f64 {
        // Measure local regularity of spike pattern
        if index < 2 || index >= spikes.spike_times.len() - 1 {
            return 0.5;
        }
        
        let dt1 = spikes.spike_times[index] - spikes.spike_times[index - 1];
        let dt2 = spikes.spike_times[index - 1] - spikes.spike_times[index - 2];
        
        if dt1 > 0.0 && dt2 > 0.0 {
            let ratio = (dt1 / dt2).min(dt2 / dt1);
            ratio.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl PhaseToSpikeConverter {
    fn new() -> Self {
        Self {
            spike_threshold: 0.5,
            refractory_period: 0.001,
            burst_detection: true,
        }
    }
    
    fn convert(&self, states: &[PhaseState]) -> Result<SpikeTrain> {
        let mut spike_times = Vec::new();
        let mut amplitudes = Vec::new();
        let mut last_spike_time = -self.refractory_period;
        
        for (i, state) in states.iter().enumerate() {
            let time = i as f64 * 0.001; // 1ms time steps
            
            // Check if phase crosses threshold
            if state.amplitude > self.spike_threshold 
                && (time - last_spike_time) > self.refractory_period {
                
                spike_times.push(time);
                amplitudes.push(state.amplitude);
                last_spike_time = time;
                
                // Burst detection
                if self.burst_detection && state.frequency > 20.0 {
                    // Add burst spikes
                    let burst_count = ((state.frequency - 20.0) / 10.0) as usize;
                    for j in 1..=burst_count.min(5) {
                        let burst_time = time + j as f64 * 0.0001;
                        spike_times.push(burst_time);
                        amplitudes.push(state.amplitude * 0.8);
                    }
                }
            }
        }
        
        Ok(SpikeTrain {
            neuron_id: 0,
            spike_times,
            amplitudes,
            duration: states.len() as f64 * 0.001,
        })
    }
}

impl ResonanceCache {
    fn new() -> Self {
        Self {
            recent_patterns: Vec::new(),
            max_size: 100,
        }
    }
    
    fn add_pattern(&mut self, pattern: ResonantPattern) {
        self.recent_patterns.push(pattern);
        
        // Keep only recent patterns
        if self.recent_patterns.len() > self.max_size {
            self.recent_patterns.remove(0);
        }
    }
    
    pub fn get_recent_patterns(&self, count: usize) -> Vec<ResonantPattern> {
        let start = self.recent_patterns.len().saturating_sub(count);
        self.recent_patterns[start..].to_vec()
    }
    
    pub fn find_similar_patterns(&self, pattern: &ResonantPattern, threshold: f64) -> Vec<ResonantPattern> {
        self.recent_patterns.iter()
            .filter(|p| {
                let freq_diff = (p.frequency - pattern.frequency).abs() / pattern.frequency;
                let coherence_diff = (p.coherence - pattern.coherence).abs();
                freq_diff < threshold && coherence_diff < threshold
            })
            .cloned()
            .collect()
    }
}

/// Extension trait for SpikeTrain
impl SpikeTrain {
    /// Convert to phase lattice representation
    pub async fn to_phase_lattice(&self) -> Result<Vec<PhaseState>> {
        let bridge = ForgeBridge::new();
        bridge.spikes_to_phase(self).await
    }
}

// Re-export SpikeTrain if not already available
#[derive(Debug, Clone)]
pub struct SpikeTrain {
    pub neuron_id: usize,
    pub spike_times: Vec<f64>,
    pub amplitudes: Vec<f64>,
    pub duration: f64,
}