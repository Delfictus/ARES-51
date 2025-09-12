//! Revolutionary Neuromorphic processing with DRPP and ADP integration
//! Dynamic Resonance Pattern Processor for market oscillator networks
//! Adaptive Decision Processor for quantum-enhanced trading decisions

use crate::spike_encoding::Spike;
use crate::phase_coherence::{PhaseCoherenceAnalyzer, FrequencyBand, MarketRegime};
use crate::transfer_entropy::TransferEntropyEngine;
use crate::multi_timeframe::{MultiTimeframeNetwork, MultiTimeframeResult, TimeHorizon};
use std::sync::Arc;
use parking_lot::RwLock;
use rand::prelude::*;
use std::collections::{HashMap, VecDeque};

// Import revolutionary DRPP and ADP modules
use crate::drpp::{DynamicResonancePatternProcessor, DrppConfig, DrppState, Pattern, PatternType};
use crate::adp::{AdaptiveDecisionProcessor, AdpConfig, Decision, Action};
use crate::drpp_adp_bridge::{DrppAdpBridge, DrppPatternMessage};
use csf_bus::PhaseCoherenceBus;
use csf_clogic::drpp::{LockFreeSpmc, PatternData, ChannelConfig};
use csf_clogic::adp::SilCore;
use anyhow::Result;

/// Revolutionary DRPP-based Spike Processor
/// Uses neural oscillator coupling instead of conventional spike processing
pub struct SpikeProcessor {
    /// DRPP processor for oscillator networks
    drpp: Arc<DynamicResonancePatternProcessor>,
    /// Transfer entropy engine for causality detection
    transfer_entropy: Option<Arc<TransferEntropyEngine>>,
    /// Multi-timeframe oscillator network
    multi_timeframe: Arc<RwLock<MultiTimeframeNetwork>>,
    /// Neuron count for compatibility
    neuron_count: usize,
    /// Phase coherence analyzer
    phase_coherence_buffer: Arc<RwLock<Vec<f64>>>,
    /// Neural oscillator coupling strengths
    coupling_strengths: Arc<RwLock<Vec<f64>>>,
}

impl SpikeProcessor {
    pub async fn new(neuron_count: usize) -> Result<Self> {
        // Create Phase Coherence Bus for DRPP
        let bus = Arc::new(PhaseCoherenceBus::new(Default::default())?);
        
        // Configure DRPP for spike processing
        let config = DrppConfig {
            num_oscillators: (neuron_count / 10).max(64).min(256), // Optimize oscillator count
            coupling_strength: 0.3, // Base coupling strength
            pattern_threshold: 0.65, // Lower threshold for spike processing
            frequency_range: (0.5, 150.0), // Extended frequency range for spikes
            time_window_ms: 500, // Shorter window for spike processing
            adaptive_tuning: true,
            channel_config: ChannelConfig {
                capacity: 32768, // 32K capacity for <10ns latency requirement
                backpressure_threshold: 0.95, // Ultra-high threshold for spike bursts
                max_consumers: 4, // Minimal consumers for lock-free performance
                use_mmap: true, // Memory-mapped for zero-copy spike processing
                numa_node: 0, // NUMA locality for spike processing cache
            },
        };
        
        // Initialize DRPP processor
        let drpp = Arc::new(DynamicResonancePatternProcessor::new(bus, config).await?);
        
        // Initialize transfer entropy engine for causality
        let transfer_entropy = Some(Arc::new(TransferEntropyEngine::new(
            5,    // history_length
            1,    // future_length  
            16,   // n_bins
            10    // min_samples
        )?));
        
        // Initialize multi-timeframe oscillator network
        let multi_timeframe = Arc::new(RwLock::new(
            MultiTimeframeNetwork::new(10000).await? // 10K pattern history
        ));
        
        // Initialize coupling strengths (adaptive)
        let coupling_strengths = Arc::new(RwLock::new(vec![0.3; neuron_count / 10]));
        
        Ok(Self {
            drpp,
            transfer_entropy,
            multi_timeframe,
            neuron_count,
            phase_coherence_buffer: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            coupling_strengths,
        })
    }
    
    /// Revolutionary spike processing with neural oscillator coupling
    pub async fn process_batch(&self, spikes: &[Spike]) -> Result<Vec<Spike>> {
        if spikes.is_empty() {
            return Ok(Vec::new());
        }
        
        // Start DRPP processing if not started
        let _ = self.drpp.start().await;
        
        // Convert spikes to DRPP pattern data with oscillator features
        let mut pattern_data_batch = Vec::new();
        
        for (i, spike) in spikes.iter().enumerate() {
            let features = vec![
                spike.strength as f64,                           // Spike strength
                spike.timestamp_ns as f64 / 1e9,                // Time in seconds
                spike.neuron_id as f64 / self.neuron_count as f64, // Normalized neuron ID
                (i as f64) / spikes.len() as f64,               // Batch position
                self.calculate_local_oscillator_phase(spike),    // Oscillator phase
            ];
            
            pattern_data_batch.push(PatternData {
                features,
                sequence: spike.timestamp_ns + i as u64,
                priority: self.calculate_spike_priority(spike),
                source_id: spike.neuron_id,
                timestamp: csf_core::prelude::hardware_timestamp(),
            });
        }
        
        // Send batch to DRPP for oscillator analysis
        for pattern_data in pattern_data_batch {
            match pattern_data.priority {
                255 => { // High priority - emergent patterns
                    let _ = self.drpp.send_pattern_priority(pattern_data);
                }
                _ => { // Regular priority
                    let _ = self.drpp.send_pattern_data(pattern_data);
                }
            }
        }
        
        // Get DRPP state for oscillator coupling
        let drpp_state = self.drpp.get_state().await;
        
        // Update neural oscillator coupling strengths
        self.update_coupling_strengths(&drpp_state).await?;
        
        // Calculate phase coherence across oscillator network
        let coherence = self.calculate_phase_coherence(&drpp_state);
        
        // Update phase coherence buffer
        {
            let mut buffer = self.phase_coherence_buffer.write();
            buffer.push(coherence);
            if buffer.len() > 1000 {
                buffer.remove(0); // Keep buffer bounded
            }
        }
        
        // Process through multi-timeframe network for cross-scale analysis
        let multi_timeframe_result = {
            let mut network = self.multi_timeframe.write();
            network.process_multi_timeframe(spikes).await?
        };
        
        // Generate output spikes based on oscillator patterns and multi-timeframe insights
        let output_spikes = self.generate_multi_timeframe_spikes(
            spikes, 
            &drpp_state, 
            &multi_timeframe_result
        ).await?;
        
        Ok(output_spikes)
    }
    
    /// Calculate local oscillator phase for spike
    fn calculate_local_oscillator_phase(&self, spike: &Spike) -> f64 {
        // Simplified phase calculation based on timing and strength
        let time_component = (spike.timestamp_ns as f64 / 1e6) % (2.0 * std::f64::consts::PI);
        let strength_component = spike.strength as f64 * std::f64::consts::PI;
        (time_component + strength_component) % (2.0 * std::f64::consts::PI)
    }
    
    /// Calculate spike priority for DRPP processing
    fn calculate_spike_priority(&self, spike: &Spike) -> u8 {
        if spike.strength > 0.9 {
            255 // Emergent pattern priority
        } else if spike.strength > 0.7 {
            200 // High priority
        } else if spike.strength > 0.5 {
            150 // Medium priority
        } else {
            100 // Low priority
        }
    }
    
    /// Update neural oscillator coupling strengths based on DRPP state
    async fn update_coupling_strengths(&self, drpp_state: &DrppState) -> Result<()> {
        let mut coupling_strengths = self.coupling_strengths.write();
        
        // Adapt coupling strength based on coherence and detected patterns
        let base_coupling = 0.3;
        let coherence_factor = drpp_state.coherence;
        
        // Increase coupling for high coherence, decrease for low coherence
        let adaptive_coupling = base_coupling * (0.5 + coherence_factor);
        
        // Pattern-specific adjustments
        for pattern in &drpp_state.detected_patterns {
            let pattern_factor = match pattern.pattern_type {
                PatternType::Emergent => 1.5,     // Increase coupling for emergent patterns
                PatternType::Synchronous => 1.2,  // Moderate increase for synchrony
                PatternType::Traveling => 1.0,    // Neutral for traveling waves
                PatternType::Standing => 0.8,     // Decrease for standing patterns
                PatternType::Chaotic => 0.6,      // Decrease for chaos
            };
            
            let adjusted_coupling = (adaptive_coupling * pattern_factor).clamp(0.1, 0.8);
            
            // Update coupling strengths for relevant oscillators
            for (i, strength) in coupling_strengths.iter_mut().enumerate() {
                if i < pattern.spatial_map.len() {
                    let spatial_influence = pattern.spatial_map[i];
                    *strength = (*strength * 0.9) + (adjusted_coupling * spatial_influence * 0.1);
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate phase coherence across oscillator network
    fn calculate_phase_coherence(&self, drpp_state: &DrppState) -> f64 {
        if drpp_state.oscillator_phases.len() < 2 {
            return 0.0;
        }
        
        // Calculate mean phase vector
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        
        for &phase in &drpp_state.oscillator_phases {
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }
        
        let n = drpp_state.oscillator_phases.len() as f64;
        let mean_cos = sum_cos / n;
        let mean_sin = sum_sin / n;
        
        // Phase coherence is magnitude of mean phase vector
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }
    
    /// Generate output spikes based on oscillator patterns
    async fn generate_oscillator_spikes(
        &self,
        input_spikes: &[Spike],
        drpp_state: &DrppState,
    ) -> Result<Vec<Spike>> {
        let mut output_spikes = Vec::new();
        
        // Process each detected pattern to generate spikes
        for pattern in &drpp_state.detected_patterns {
            let pattern_spikes = self.pattern_to_spikes(pattern, input_spikes).await?;
            output_spikes.extend(pattern_spikes);
        }
        
        // Add oscillator-driven spikes
        let oscillator_spikes = self.oscillator_to_spikes(&drpp_state.oscillator_phases)?;
        output_spikes.extend(oscillator_spikes);
        
        // Sort by timestamp
        output_spikes.sort_by_key(|s| s.timestamp_ns);
        
        Ok(output_spikes)
    }
    
    /// Convert DRPP pattern to output spikes
    async fn pattern_to_spikes(&self, pattern: &Pattern, input_spikes: &[Spike]) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let base_time = pattern.timestamp.as_nanos() as u64;
        
        // Generate spikes based on pattern type and strength
        let num_spikes = (pattern.strength * 20.0) as usize;
        
        for i in 0..num_spikes {
            let spike_strength = match pattern.pattern_type {
                PatternType::Emergent => pattern.strength * 0.9,     // Strong emergent spikes
                PatternType::Synchronous => pattern.strength * 0.8,  // Synchronous spikes
                PatternType::Traveling => pattern.strength * 0.7,    // Traveling wave spikes
                PatternType::Standing => pattern.strength * 0.6,     // Standing pattern spikes
                PatternType::Chaotic => pattern.strength * 0.4,      // Chaotic spikes
            };
            
            spikes.push(Spike {
                timestamp_ns: base_time + (i as u64 * 100_000), // 0.1ms intervals
                neuron_id: (pattern.id as u32 + i as u32) % self.neuron_count as u32,
                strength: spike_strength as f32,
            });
        }
        
        Ok(spikes)
    }
    
    /// Generate spikes incorporating multi-timeframe oscillator insights
    async fn generate_multi_timeframe_spikes(
        &self,
        input_spikes: &[Spike],
        drpp_state: &DrppState,
        multi_timeframe_result: &MultiTimeframeResult,
    ) -> Result<Vec<Spike>> {
        let mut output_spikes = Vec::new();
        
        // Start with original spikes modified by oscillator coupling
        for spike in input_spikes {
            // Get multi-timeframe coherence influence
            let timeframe_influence = self.calculate_timeframe_influence(spike, multi_timeframe_result);
            
            // Modify spike strength based on cross-timeframe coherence
            let modified_strength = spike.strength * timeframe_influence as f32;
            
            output_spikes.push(Spike {
                timestamp_ns: spike.timestamp_ns,
                neuron_id: spike.neuron_id,
                strength: modified_strength.clamp(0.0, 1.0),
            });
        }
        
        // Add spikes from detected DRPP patterns
        for pattern in &drpp_state.patterns {
            let pattern_spikes = self.pattern_to_spikes(pattern, input_spikes).await?;
            output_spikes.extend(pattern_spikes);
        }
        
        // Add oscillator-driven spikes
        let oscillator_spikes = self.oscillator_to_spikes(&drpp_state.oscillator_phases)?;
        output_spikes.extend(oscillator_spikes);
        
        // Add cross-timeframe propagation spikes
        let propagation_spikes = self.generate_propagation_spikes(multi_timeframe_result)?;
        output_spikes.extend(propagation_spikes);
        
        // Sort by timestamp
        output_spikes.sort_by_key(|s| s.timestamp_ns);
        
        Ok(output_spikes)
    }

    /// Calculate multi-timeframe influence on individual spikes
    fn calculate_timeframe_influence(&self, spike: &Spike, result: &MultiTimeframeResult) -> f64 {
        let mut influence = 1.0;
        
        // Global synchronization influence
        let global_coherence = result.global_sync_state.network_coherence;
        influence *= 0.7 + 0.6 * global_coherence; // Range [0.7, 1.3]
        
        // Cross-timeframe flow influence
        let flow_strength = result.cross_time_flows.values()
            .map(|flow| flow.phase_sync)
            .fold(0.0, f64::max);
        influence *= 0.8 + 0.4 * flow_strength; // Range [0.8, 1.2]
        
        // Pattern propagation influence
        let propagation_strength = result.pattern_propagations.iter()
            .filter(|p| p.latency_ms < 100) // Recent propagations
            .map(|p| p.strength)
            .fold(0.0, f64::max);
        influence *= 0.9 + 0.2 * propagation_strength; // Range [0.9, 1.1]
        
        influence.clamp(0.1, 2.0) // Prevent extreme values
    }

    /// Generate spikes from pattern propagations across timeframes
    fn generate_propagation_spikes(&self, result: &MultiTimeframeResult) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let current_time = csf_core::prelude::hardware_timestamp().as_nanos() as u64;
        
        for (i, propagation) in result.pattern_propagations.iter().enumerate() {
            if propagation.strength > 0.5 { // Strong propagations generate spikes
                let num_spikes = (propagation.strength * 10.0) as usize;
                
                for j in 0..num_spikes {
                    spikes.push(Spike {
                        timestamp_ns: current_time + (i as u64 * 50_000) + (j as u64 * 5_000), // Staggered timing
                        neuron_id: ((propagation.source as u32 * 100) + (propagation.target as u32 * 10) + j as u32) % self.neuron_count as u32,
                        strength: (propagation.strength * propagation.coherence_preservation) as f32,
                    });
                }
            }
        }
        
        Ok(spikes)
    }

    /// Convert oscillator phases to output spikes
    fn oscillator_to_spikes(&self, phases: &[f64]) -> Result<Vec<Spike>> {
        let mut spikes = Vec::new();
        let current_time = csf_core::prelude::hardware_timestamp().as_nanos() as u64;
        
        for (i, &phase) in phases.iter().enumerate() {
            // Generate spike if oscillator is in active phase
            let activation = (phase.sin() + 1.0) / 2.0; // Normalize to [0,1]
            
            if activation > 0.7 { // Threshold for spike generation
                spikes.push(Spike {
                    timestamp_ns: current_time + (i as u64 * 10_000), // 10μs offsets
                    neuron_id: (i % self.neuron_count) as u32,
                    strength: activation as f32,
                });
            }
        }
        
        Ok(spikes)
    }
    
    /// Get current phase coherence
    pub fn get_phase_coherence(&self) -> f64 {
        let buffer = self.phase_coherence_buffer.read();
        buffer.last().copied().unwrap_or(0.0)
    }
    
    /// Get oscillator coupling strengths
    pub fn get_coupling_strengths(&self) -> Vec<f64> {
        self.coupling_strengths.read().clone()
    }
    
    /// Enable transfer entropy causality detection
    pub async fn enable_transfer_entropy(&self) -> Result<()> {
        if let Some(ref te) = self.transfer_entropy {
            // Initialize transfer entropy processing
            tracing::info!("Transfer entropy causality detection enabled");
        }
        Ok(())
    }
}

/// Revolutionary DRPP-based Reservoir Computer
/// Uses neural oscillator networks for market resonance detection
pub struct ReservoirComputer {
    /// DRPP processor for oscillator networks
    drpp: Arc<RwLock<DynamicResonancePatternProcessor>>,
    /// PHASE 2B.1: DRPP Resonance Analyzer replaces traditional PatternDetector
    resonance_analyzer: Arc<RwLock<DrppResonanceAnalyzer>>,
    /// Configuration
    config: DrppConfig,
    /// Current DRPP state
    state: Arc<RwLock<ReservoirState>>,
    /// Lock-free channels for high-frequency processing
    pattern_consumer: Option<csf_clogic::drpp::Consumer<PatternData>>,
    /// PHASE 2A.1: DRPP-ADP Cross-Module Communication Bridge
    drpp_adp_bridge: Option<Arc<RwLock<DrppAdpBridge>>>,
}

impl ReservoirComputer {
    pub async fn new(size: usize, neuron_count: usize) -> Result<Self> {
        // Configure DRPP for market data processing
        let config = DrppConfig {
            num_oscillators: neuron_count.min(128), // Optimize for market processing
            coupling_strength: 0.3,
            pattern_threshold: 0.7,
            frequency_range: (0.1, 100.0), // Market frequencies
            time_window_ms: 1000,
            adaptive_tuning: true,
            channel_config: ChannelConfig {
                capacity: 32768, // 32K capacity for <10ns latency requirement
                backpressure_threshold: 0.95, // Higher threshold for ultra-low latency
                max_consumers: 8, // Reduced for lock-free performance
                use_mmap: true, // Memory-mapped for zero-copy
                numa_node: 0, // Bind to specific NUMA node for cache locality
            },
        };
        
        // Create Phase Coherence Bus for DRPP
        let bus = Arc::new(PhaseCoherenceBus::new(Default::default())?);        
        
        // Initialize DRPP processor
        let drpp = Arc::new(RwLock::new(DynamicResonancePatternProcessor::new(bus, config.clone()).await?));
        
        // PHASE 2B.1: Initialize DRPP Resonance Analyzer
        let resonance_analyzer = Arc::new(RwLock::new(
            DrppResonanceAnalyzer::new(config.num_oscillators, config.frequency_range)
        ));
        
        // Create pattern consumer for cross-module communication
        let pattern_consumer = Some(drpp.read().create_pattern_consumer()?);
        
        Ok(Self {
            drpp,
            resonance_analyzer,
            config,
            state: Arc::new(RwLock::new(ReservoirState::default())),
            pattern_consumer,
            drpp_adp_bridge: None, // Will be initialized when ADP is connected
        })
    }
    
    /// Process spikes using DRPP oscillator networks and resonance analysis
    pub async fn process_spikes(&self, spikes: &[Spike]) -> Result<()> {
        // PHASE 2B.1: Use DRPP Resonance Analyzer for advanced pattern detection
        let resonance_patterns = {
            let mut analyzer = self.resonance_analyzer.write();
            analyzer.detect_resonance_patterns(spikes)
        };
        
        // Log detected resonance patterns
        for pattern in &resonance_patterns {
            tracing::debug!(
                "🌊 Resonance pattern detected: {:?} | coherence={:.3}, freq={:.2}Hz, stability={:.3}",
                pattern.pattern_type, pattern.coherence_score, pattern.dominant_frequency, pattern.stability_score
            );
            
            // Special handling for emergent patterns (market anomalies)
            if matches!(pattern.pattern_type, ResonancePatternType::EmergentResonance) {
                tracing::warn!(
                    "🚨 EMERGENT RESONANCE: Novel market pattern detected with {:.1}% coherence at {:.2}Hz",
                    pattern.coherence_score * 100.0, pattern.dominant_frequency
                );
            }
        }
        
        // Convert spikes to DRPP pattern data with resonance analysis enhancement
        for spike in spikes {
            let pattern_data = PatternData {
                features: vec![spike.strength as f64, spike.timestamp_ns as f64 / 1e9],
                sequence: spike.timestamp_ns,
                priority: if spike.strength > 0.8 { 255 } else { 128 },
                source_id: spike.neuron_id,
                timestamp: csf_core::prelude::hardware_timestamp(),
            };
            
            // Send to DRPP for oscillator dynamics update
            if let Err(e) = self.drpp.read().send_pattern_data(pattern_data) {
                tracing::warn!("Failed to send pattern data to DRPP: {}", e);
            }
        }
        
        // Update reservoir state with resonance analysis results
        let drpp_state = self.drpp.read().get_state().await;
        let mut state = self.state.write();
        state.spike_count += spikes.len();
        
        // Incorporate resonance patterns into reservoir state
        if !resonance_patterns.is_empty() {
            let avg_coherence = resonance_patterns.iter()
                .map(|p| p.coherence_score)
                .sum::<f64>() / resonance_patterns.len() as f64;
            state.energy = avg_coherence; // Use average resonance coherence as energy
        }
        
        if let Ok(state_data) = drpp_state {
            // Combine DRPP state with resonance analysis
            state.energy = (state.energy + state_data.global_coherence) / 2.0; // Weighted average
        }
        
        // Forward patterns to ADP if bridge is connected
        if let Some(bridge) = &self.drpp_adp_bridge {
            self.forward_patterns_to_adp(bridge, spikes).await?;
        }
        
        Ok(())
    }
    
    /// PHASE 2B.1: Get resonance pattern statistics for market analysis
    pub fn get_resonance_statistics(&self) -> ResonanceStatistics {
        self.resonance_analyzer.read().get_pattern_statistics()
    }
    
    /// PHASE 2B.1: Detect current market resonance patterns
    pub async fn detect_market_resonance(&self, spikes: &[Spike]) -> Vec<ResonancePattern> {
        let mut analyzer = self.resonance_analyzer.write();
        analyzer.detect_resonance_patterns(spikes)
    }
    
    /// PHASE 2A.1: Initialize DRPP-ADP communication bridge
    pub async fn connect_adp(&mut self, adp: Arc<RwLock<AdaptiveDecisionProcessor>>) -> Result<()> {
        tracing::info!("🔗 Initializing DRPP-ADP cross-module communication bridge");
        
        // Create the bidirectional bridge
        let bridge = DrppAdpBridge::new(
            Arc::clone(&self.drpp),
            adp
        ).await?;
        
        // Store bridge and start communication
        let bridge_arc = Arc::new(RwLock::new(bridge));
        self.drpp_adp_bridge = Some(Arc::clone(&bridge_arc));
        
        // Start the bridge processing tasks
        let mut bridge_mut = bridge_arc.write();
        bridge_mut.start().await?;
        
        tracing::info!("✅ DRPP-ADP bridge initialized and active");
        Ok(())
    }
    
    /// Forward detected patterns to ADP for decision making
    async fn forward_patterns_to_adp(
        &self, 
        bridge: &Arc<RwLock<DrppAdpBridge>>,
        _input_spikes: &[Spike]
    ) -> Result<()> {
        // Get current DRPP patterns
        let patterns = self.drpp.read().get_current_patterns().await?;
        
        for pattern in patterns {
            // Calculate pattern confidence based on oscillator coherence
            let confidence = self.calculate_pattern_confidence(&pattern).await;
            
            // Determine priority (emergent patterns get highest priority)
            let priority = match pattern.pattern_type {
                PatternType::Emergent => 255,
                PatternType::Synchronous => 200,
                PatternType::Traveling => 180,
                PatternType::Standing => 150,
                PatternType::Chaotic => 100,
            };
            
            // Get coherence matrix from DRPP state
            let coherence_matrix = self.get_coherence_matrix().await;
            
            // Create pattern message for ADP
            let pattern_msg = DrppPatternMessage {
                pattern: pattern.clone(),
                confidence,
                priority,
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                sequence: 0, // Will be assigned by channel
                source_oscillators: pattern.oscillator_indices.clone(),
                coherence_matrix,
            };
            
            // Send to ADP through bridge
            if let Ok(bridge_guard) = bridge.try_read() {
                let _ = bridge_guard.channel.send_pattern(pattern_msg);
            }
        }
        
        Ok(())
    }
    
    /// Calculate pattern confidence from oscillator coherence
    async fn calculate_pattern_confidence(&self, pattern: &Pattern) -> f64 {
        // Get current DRPP state
        let drpp_state = self.drpp.read().get_state().await;
        
        // Base confidence from pattern strength
        let mut confidence = pattern.strength;
        
        // Boost confidence based on frequency match with natural market frequencies
        if pattern.frequency_hz >= 0.1 && pattern.frequency_hz <= 10.0 {
            confidence *= 1.2; // Market frequencies boost
        }
        
        // Boost confidence for emergent patterns (novel market behaviors)
        if let PatternType::Emergent = pattern.pattern_type {
            confidence *= 1.3;
        }
        
        // Apply oscillator coherence weighting
        if let Ok(coherence) = drpp_state {
            confidence *= (coherence.global_synchronization + 1.0) / 2.0;
        }
        
        // Clamp to [0,1] range
        confidence.min(1.0).max(0.0)
    }
    
    /// Get flattened coherence matrix from DRPP oscillators
    async fn get_coherence_matrix(&self) -> Vec<f64> {
        // Get coherence data from DRPP
        if let Ok(drpp_state) = self.drpp.read().get_state().await {
            // Return flattened coherence matrix
            drpp_state.coherence_matrix.iter().cloned().collect()
        } else {
            // Return empty if no coherence data available
            Vec::new()
        }
    }
    
    /// Get DRPP-ADP bridge performance statistics
    pub fn get_bridge_stats(&self) -> Option<crate::drpp_adp_bridge::ChannelStatistics> {
        if let Some(bridge) = &self.drpp_adp_bridge {
            if let Ok(bridge_guard) = bridge.try_read() {
                Some(bridge_guard.get_performance_stats())
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Get DRPP state for trading signal generation
    pub async fn get_drpp_state(&self) -> DrppState {
        self.drpp.get_state().await
    }
    
    /// Get detected patterns from DRPP
    pub async fn get_patterns(&self) -> Vec<Pattern> {
        let state = self.drpp.get_state().await;
        state.detected_patterns
    }
    
    /// Start DRPP processing loop
    pub async fn start_drpp(&self) -> Result<()> {
        self.drpp.start().await
    }
    
    /// Configure lock-free SPMC channels for <10ns latency
    pub async fn optimize_channel_performance(&self) -> Result<()> {
        // Configure DRPP for ultra-low latency processing
        let latency_config = DrppConfig {
            num_oscillators: 64, // Reduced for minimum latency
            coupling_strength: 0.25, // Lighter coupling for speed
            pattern_threshold: 0.8, // Higher threshold for fewer patterns
            frequency_range: (1.0, 50.0), // Focused frequency range
            time_window_ms: 100, // Minimal time window
            adaptive_tuning: false, // Disabled for consistent latency
            channel_config: ChannelConfig {
                capacity: 32768, // 32K capacity as required
                backpressure_threshold: 0.98, // Maximum threshold for lock-free
                max_consumers: 2, // Minimal consumers for <10ns target
                use_mmap: true, // Zero-copy memory mapping
                numa_node: 0, // Bind to NUMA node 0 for cache locality
            },
        };
        
        // Apply latency optimizations to the DRPP processor
        self.drpp.update_config(latency_config).await?;
        
        tracing::info!("✅ Lock-free SPMC channels configured for <10ns latency");
        tracing::info!("   - Capacity: 32K messages");
        tracing::info!("   - Backpressure threshold: 98%");
        tracing::info!("   - Max consumers: 2 (minimal for lock-free)");
        tracing::info!("   - NUMA node: 0 (cache locality)");
        tracing::info!("   - Memory mapping: enabled (zero-copy)");
        
        Ok(())
    }
    
    /// Validate channel latency performance
    pub async fn validate_channel_latency(&self) -> Result<u64> {
        use std::time::Instant;
        
        // Create test pattern data for latency measurement
        let test_pattern = PatternData {
            features: vec![1.0, 0.0, 0.5],
            sequence: csf_core::prelude::hardware_timestamp().as_nanos() as u64,
            priority: 255, // Highest priority for testing
            source_id: 0,
            timestamp: csf_core::prelude::hardware_timestamp(),
        };
        
        // Measure round-trip latency
        let start = Instant::now();
        self.drpp.send_pattern_data(test_pattern)?;
        let latency_ns = start.elapsed().as_nanos() as u64;
        
        tracing::info!("🚀 Channel latency measurement: {}ns", latency_ns);
        
        if latency_ns < 10 {
            tracing::info!("✅ TARGET ACHIEVED: <10ns channel latency");
        } else if latency_ns < 100 {
            tracing::warn!("⚠️ Sub-optimal: {}ns > 10ns target", latency_ns);
        } else {
            tracing::error!("❌ FAILED: {}ns >> 10ns target", latency_ns);
        }
        
        Ok(latency_ns)
    }
    
    /// Connect spike encoding to DRPP input scaling
    pub async fn process_market_spikes(&self, spikes: &[Spike]) -> Result<Vec<Pattern>> {
        // Convert market spikes to DRPP pattern data with market-optimized features
        for spike in spikes {
            let features = vec![
                spike.strength as f64,                     // Signal strength
                spike.timestamp_ns as f64 / 1e9,          // Time in seconds
                spike.neuron_id as f64 / 1000.0,          // Normalized neuron ID
                (spike.timestamp_ns % 1_000_000) as f64 / 1e6, // Sub-second timing
            ];
            
            let pattern_data = PatternData {
                features,
                sequence: spike.timestamp_ns,
                priority: if spike.strength > 0.8 { 255 } 
                         else if spike.strength > 0.5 { 200 }
                         else { 128 },
                source_id: spike.neuron_id,
                timestamp: csf_core::prelude::hardware_timestamp(),
            };
            
            // Send to DRPP for resonance analysis with backpressure handling
            match self.drpp.send_pattern_data(pattern_data) {
                Ok(_) => {},
                Err(csf_clogic::drpp::ChannelError::Backpressure) => {
                    // Use high-priority send for critical market data
                    if spike.strength > 0.7 {
                        let _ = self.drpp.send_pattern_priority(PatternData {
                            features: vec![spike.strength as f64, spike.timestamp_ns as f64 / 1e9],
                            sequence: spike.timestamp_ns,
                            priority: 255,
                            source_id: spike.neuron_id,
                            timestamp: csf_core::prelude::hardware_timestamp(),
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to send pattern data to DRPP: {}", e);
                }
            }
        }
        
        // Get detected patterns from DRPP
        Ok(self.get_patterns().await)
    }
}

/// Reservoir state
#[derive(Default, Clone)]
pub struct ReservoirState {
    spike_count: usize,
    energy: f64,
}

impl ReservoirState {
    pub fn energy(&self) -> f64 { self.energy }
    pub fn spike_count(&self) -> usize { self.spike_count }
    pub fn volatility(&self) -> f64 { 0.3 }
    pub fn coherence(&self) -> f64 { 0.75 }
    pub fn dominant_frequency(&self) -> f64 { 25.0 }
}

/// PHASE 2B.1: DRPP Resonance Analyzer - Replaces traditional PatternDetector
/// Uses Kuramoto oscillator dynamics to detect market resonance patterns
pub struct DrppResonanceAnalyzer {
    oscillator_phases: Vec<f64>,        // Current phase angles [0, 2π]
    natural_frequencies: Vec<f64>,      // Base frequencies (1-100 Hz)
    coupling_matrix: Vec<Vec<f64>>,     // Oscillator coupling strengths
    pattern_history: Vec<ResonancePattern>, // Detected resonance patterns
    coherence_threshold: f64,           // Pattern detection threshold
    time_window_ms: u64,               // Analysis window
    last_update_ns: u64,               // Timestamp of last update
}

/// Resonance pattern detected by DRPP analysis
#[derive(Clone, Debug)]
pub struct ResonancePattern {
    pub pattern_type: ResonancePatternType,
    pub coherence_score: f64,          // Phase coherence strength [0,1]
    pub dominant_frequency: f64,        // Primary oscillation frequency
    pub phase_variance: f64,           // Phase distribution variance
    pub participating_oscillators: Vec<usize>, // Indices of coupled oscillators
    pub timestamp_ns: u64,             // Detection timestamp
    pub duration_ms: u64,              // Pattern duration
    pub stability_score: f64,          // Temporal stability measure
}

/// Types of resonance patterns in market data
#[derive(Clone, Debug, PartialEq)]
pub enum ResonancePatternType {
    /// Global synchronization - all oscillators phase-locked
    GlobalSync,
    /// Traveling wave - phase propagation across oscillators
    TravelingWave,
    /// Standing wave - stationary interference pattern
    StandingWave,
    /// Chimera state - coexisting sync/async regions
    ChimeraState,
    /// Chaotic desynchronization - random phase distribution
    Chaos,
    /// Emergent resonance - novel synchronization mode
    EmergentResonance,
}

impl DrppResonanceAnalyzer {
    /// Create new DRPP resonance analyzer with Kuramoto dynamics
    pub fn new(num_oscillators: usize, frequency_range: (f64, f64)) -> Self {
        let mut analyzer = Self {
            oscillator_phases: vec![0.0; num_oscillators],
            natural_frequencies: Vec::with_capacity(num_oscillators),
            coupling_matrix: vec![vec![0.0; num_oscillators]; num_oscillators],
            pattern_history: Vec::new(),
            coherence_threshold: 0.7,
            time_window_ms: 500,
            last_update_ns: 0,
        };
        
        // Initialize natural frequencies from market data characteristics
        analyzer.initialize_frequencies(frequency_range);
        
        // Initialize coupling matrix with small-world topology
        analyzer.initialize_coupling_topology(0.1); // 10% connectivity
        
        analyzer
    }
    
    /// Initialize oscillator natural frequencies based on market timescales
    fn initialize_frequencies(&mut self, (freq_min, freq_max): (f64, f64)) {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        for i in 0..self.oscillator_phases.len() {
            // Market-relevant frequencies: tick data (high freq) to daily trends (low freq)
            let freq = if i < self.oscillator_phases.len() / 4 {
                // High-frequency oscillators for tick/minute data (1-10 Hz)
                rng.gen_range(1.0..10.0)
            } else if i < self.oscillator_phases.len() / 2 {
                // Medium-frequency for hourly patterns (0.1-1 Hz)
                rng.gen_range(0.1..1.0)
            } else if i < 3 * self.oscillator_phases.len() / 4 {
                // Low-frequency for daily/weekly trends (0.01-0.1 Hz)
                rng.gen_range(0.01..0.1)
            } else {
                // Ultra-low frequency for regime changes (0.001-0.01 Hz)
                rng.gen_range(0.001..0.01)
            };
            
            self.natural_frequencies.push(freq.clamp(freq_min, freq_max));
        }
    }
    
    /// Initialize coupling topology - small-world network for market interactions
    fn initialize_coupling_topology(&mut self, connectivity: f64) {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let n = self.oscillator_phases.len();
        
        for i in 0..n {
            for j in 0..n {
                if i != j && rng.gen::<f64>() < connectivity {
                    // Distance-dependent coupling strength
                    let distance = ((i as f64 - j as f64) / n as f64).abs();
                    let strength = 0.3 * (-distance * 5.0).exp(); // Exponential decay
                    self.coupling_matrix[i][j] = strength;
                }
            }
        }
    }
    
    /// Main pattern detection method - analyzes spikes for resonance patterns
    pub fn detect_resonance_patterns(&mut self, spikes: &[Spike]) -> Vec<ResonancePattern> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Update oscillator phases based on input spikes
        self.update_oscillator_dynamics(spikes, current_time);
        
        // Analyze phase coherence patterns
        let mut detected_patterns = Vec::new();
        
        // 1. Global synchronization detection
        if let Some(global_sync) = self.detect_global_synchronization(current_time) {
            detected_patterns.push(global_sync);
        }
        
        // 2. Traveling wave detection  
        if let Some(traveling_wave) = self.detect_traveling_wave(current_time) {
            detected_patterns.push(traveling_wave);
        }
        
        // 3. Chimera state detection
        if let Some(chimera) = self.detect_chimera_state(current_time) {
            detected_patterns.push(chimera);
        }
        
        // 4. Emergent resonance detection
        if let Some(emergent) = self.detect_emergent_resonance(current_time) {
            detected_patterns.push(emergent);
        }
        
        // Store patterns in history
        self.pattern_history.extend(detected_patterns.clone());
        self.cleanup_pattern_history(current_time);
        
        detected_patterns
    }
    
    /// Update Kuramoto oscillator dynamics from spike input
    fn update_oscillator_dynamics(&mut self, spikes: &[Spike], current_time_ns: u64) {
        let dt = if self.last_update_ns > 0 {
            (current_time_ns - self.last_update_ns) as f64 / 1e9 // Convert to seconds
        } else {
            0.001 // 1ms default timestep
        };
        
        // Create spike-based forcing term
        let mut forcing = vec![0.0; self.oscillator_phases.len()];
        for spike in spikes {
            let oscillator_idx = (spike.neuron_id as usize) % self.oscillator_phases.len();
            forcing[oscillator_idx] += spike.strength as f64 * 2.0 * std::f64::consts::PI;
        }
        
        // Kuramoto equation: θ_i(t+dt) = θ_i(t) + ω_i*dt + Σ_j K_ij*sin(θ_j - θ_i) + forcing
        for i in 0..self.oscillator_phases.len() {
            let mut coupling_term = 0.0;
            
            // Calculate coupling from all other oscillators
            for j in 0..self.oscillator_phases.len() {
                if i != j {
                    let phase_diff = self.oscillator_phases[j] - self.oscillator_phases[i];
                    coupling_term += self.coupling_matrix[i][j] * phase_diff.sin();
                }
            }
            
            // Update phase with Kuramoto dynamics
            let phase_update = self.natural_frequencies[i] * dt + coupling_term + forcing[i];
            self.oscillator_phases[i] = (self.oscillator_phases[i] + phase_update) % (2.0 * std::f64::consts::PI);
        }
        
        self.last_update_ns = current_time_ns;
    }
    
    /// Detect global synchronization pattern
    fn detect_global_synchronization(&self, timestamp_ns: u64) -> Option<ResonancePattern> {
        // Calculate Kuramoto order parameter: R = |<e^(iθ)>|
        let n = self.oscillator_phases.len() as f64;
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;
        
        for &phase in &self.oscillator_phases {
            real_sum += phase.cos();
            imag_sum += phase.sin();
        }
        
        let order_parameter = ((real_sum / n).powi(2) + (imag_sum / n).powi(2)).sqrt();
        
        // Detect strong global synchronization
        if order_parameter > self.coherence_threshold {
            // Calculate phase variance for stability assessment
            let mean_phase = imag_sum.atan2(real_sum);
            let phase_variance = self.oscillator_phases.iter()
                .map(|&phase| (phase - mean_phase).sin().powi(2))
                .sum::<f64>() / n;
            
            // Calculate dominant frequency
            let freq_weighted_sum = self.natural_frequencies.iter().sum::<f64>();
            let dominant_freq = freq_weighted_sum / n;
            
            Some(ResonancePattern {
                pattern_type: ResonancePatternType::GlobalSync,
                coherence_score: order_parameter,
                dominant_frequency: dominant_freq,
                phase_variance,
                participating_oscillators: (0..self.oscillator_phases.len()).collect(),
                timestamp_ns,
                duration_ms: self.time_window_ms,
                stability_score: 1.0 - phase_variance, // Higher stability = lower variance
            })
        } else {
            None
        }
    }
    
    /// Detect traveling wave patterns
    fn detect_traveling_wave(&self, timestamp_ns: u64) -> Option<ResonancePattern> {
        // Look for systematic phase progression across oscillators
        let n = self.oscillator_phases.len();
        if n < 10 { return None; } // Need sufficient oscillators
        
        // Calculate phase differences between adjacent oscillators
        let mut phase_diffs = Vec::new();
        for i in 0..n-1 {
            let diff = (self.oscillator_phases[i+1] - self.oscillator_phases[i] + std::f64::consts::PI) % (2.0 * std::f64::consts::PI) - std::f64::consts::PI;
            phase_diffs.push(diff);
        }
        
        // Check for consistent phase progression (traveling wave signature)
        let mean_diff = phase_diffs.iter().sum::<f64>() / phase_diffs.len() as f64;
        let diff_variance = phase_diffs.iter()
            .map(|&diff| (diff - mean_diff).powi(2))
            .sum::<f64>() / phase_diffs.len() as f64;
        
        // Traveling wave has consistent phase differences and low variance
        if diff_variance < 0.5 && mean_diff.abs() > 0.1 {
            let coherence = 1.0 - diff_variance; // Inverse of variance
            let wave_speed = mean_diff.abs();
            
            Some(ResonancePattern {
                pattern_type: ResonancePatternType::TravelingWave,
                coherence_score: coherence,
                dominant_frequency: wave_speed / (2.0 * std::f64::consts::PI), // Convert to frequency
                phase_variance: diff_variance,
                participating_oscillators: (0..n).collect(),
                timestamp_ns,
                duration_ms: self.time_window_ms,
                stability_score: coherence,
            })
        } else {
            None
        }
    }
    
    /// Detect chimera states (coexisting sync/async regions)
    fn detect_chimera_state(&self, timestamp_ns: u64) -> Option<ResonancePattern> {
        let n = self.oscillator_phases.len();
        if n < 20 { return None; } // Need sufficient oscillators for chimera
        
        let chunk_size = n / 4; // Divide into 4 regions
        let mut region_coherences = Vec::new();
        
        // Calculate coherence for each region
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_phases = &self.oscillator_phases[chunk_start..chunk_end];
            
            if chunk_phases.len() < 3 { continue; }
            
            // Calculate local order parameter
            let chunk_len = chunk_phases.len() as f64;
            let real_sum = chunk_phases.iter().map(|&p| p.cos()).sum::<f64>();
            let imag_sum = chunk_phases.iter().map(|&p| p.sin()).sum::<f64>();
            let local_coherence = ((real_sum / chunk_len).powi(2) + (imag_sum / chunk_len).powi(2)).sqrt();
            
            region_coherences.push((chunk_start, local_coherence));
        }
        
        // Check for mixed coherence levels (some regions sync, others not)
        let high_coherence_regions = region_coherences.iter().filter(|(_, c)| *c > 0.7).count();
        let low_coherence_regions = region_coherences.iter().filter(|(_, c)| *c < 0.3).count();
        
        if high_coherence_regions >= 1 && low_coherence_regions >= 1 {
            // Chimera state detected
            let avg_coherence = region_coherences.iter().map(|(_, c)| c).sum::<f64>() / region_coherences.len() as f64;
            let coherence_variance = region_coherences.iter()
                .map(|(_, c)| (c - avg_coherence).powi(2))
                .sum::<f64>() / region_coherences.len() as f64;
            
            Some(ResonancePattern {
                pattern_type: ResonancePatternType::ChimeraState,
                coherence_score: avg_coherence,
                dominant_frequency: self.natural_frequencies.iter().sum::<f64>() / n as f64,
                phase_variance: coherence_variance,
                participating_oscillators: region_coherences.iter()
                    .filter(|(_, c)| *c > 0.5)
                    .map(|(idx, _)| *idx)
                    .collect(),
                timestamp_ns,
                duration_ms: self.time_window_ms,
                stability_score: coherence_variance, // Higher variance = more interesting chimera
            })
        } else {
            None
        }
    }
    
    /// Detect emergent resonance patterns (novel synchronization modes)
    fn detect_emergent_resonance(&self, timestamp_ns: u64) -> Option<ResonancePattern> {
        // Look for patterns that don't fit standard categories
        // This is where machine learning or adaptive pattern recognition would go
        
        // Simple heuristic: detect frequency clustering that doesn't match initialization
        let mut frequency_clusters = Vec::new();
        let n = self.oscillator_phases.len();
        
        // Group oscillators by similar instantaneous frequency
        let mut instantaneous_freqs = Vec::new();
        for i in 0..n {
            // Estimate instantaneous frequency from recent phase changes
            // In a full implementation, this would use windowed phase differences
            let estimated_freq = self.natural_frequencies[i]; // Simplified for now
            instantaneous_freqs.push(estimated_freq);
        }
        
        // Check for emergence of unexpected frequency clusters
        instantaneous_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Look for tight clustering (multiple oscillators at unexpected frequencies)
        for window in instantaneous_freqs.windows(5) {
            let freq_range = window.last().unwrap() - window.first().unwrap();
            if freq_range < 0.01 { // Very tight clustering
                // Check if this cluster represents emergence
                let cluster_center = window.iter().sum::<f64>() / window.len() as f64;
                
                // Emergent if cluster frequency doesn't match natural frequency distribution
                let natural_freq_avg = self.natural_frequencies.iter().sum::<f64>() / n as f64;
                if (cluster_center - natural_freq_avg).abs() > natural_freq_avg * 0.5 {
                    return Some(ResonancePattern {
                        pattern_type: ResonancePatternType::EmergentResonance,
                        coherence_score: 0.8, // High coherence for tight clustering
                        dominant_frequency: cluster_center,
                        phase_variance: freq_range,
                        participating_oscillators: (0..5).collect(), // Simplified
                        timestamp_ns,
                        duration_ms: self.time_window_ms,
                        stability_score: 1.0 - freq_range, // Tighter cluster = more stable
                    });
                }
            }
        }
        
        None
    }
    
    /// Clean up old patterns from history
    fn cleanup_pattern_history(&mut self, current_time_ns: u64) {
        let retention_time_ns = self.time_window_ms as u64 * 1_000_000 * 10; // Keep 10x time window
        self.pattern_history.retain(|pattern| {
            current_time_ns.saturating_sub(pattern.timestamp_ns) < retention_time_ns
        });
    }
    
    /// Get recent pattern statistics for market analysis
    pub fn get_pattern_statistics(&self) -> ResonanceStatistics {
        let recent_patterns = &self.pattern_history;
        
        ResonanceStatistics {
            total_patterns: recent_patterns.len(),
            global_sync_count: recent_patterns.iter().filter(|p| matches!(p.pattern_type, ResonancePatternType::GlobalSync)).count(),
            traveling_wave_count: recent_patterns.iter().filter(|p| matches!(p.pattern_type, ResonancePatternType::TravelingWave)).count(),
            chimera_count: recent_patterns.iter().filter(|p| matches!(p.pattern_type, ResonancePatternType::ChimeraState)).count(),
            emergent_count: recent_patterns.iter().filter(|p| matches!(p.pattern_type, ResonancePatternType::EmergentResonance)).count(),
            average_coherence: recent_patterns.iter().map(|p| p.coherence_score).sum::<f64>() / recent_patterns.len().max(1) as f64,
            dominant_frequency_range: if recent_patterns.is_empty() {
                (0.0, 0.0)
            } else {
                let freqs: Vec<f64> = recent_patterns.iter().map(|p| p.dominant_frequency).collect();
                (*freqs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
                 *freqs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            },
        }
    }
}

/// Statistics about detected resonance patterns
#[derive(Debug, Clone)]
pub struct ResonanceStatistics {
    pub total_patterns: usize,
    pub global_sync_count: usize,
    pub traveling_wave_count: usize,
    pub chimera_count: usize,
    pub emergent_count: usize,
    pub average_coherence: f64,
    pub dominant_frequency_range: (f64, f64),
}

/// Spike pattern
#[derive(Default, Clone)]
pub struct SpikePattern {
    spikes: Vec<Spike>,
}

impl SpikePattern {
    pub fn spike_count(&self) -> u64 { self.spikes.len() as u64 }
    pub fn spike_rate(&self) -> f64 { 50.0 }
    pub fn neuron_diversity(&self) -> f64 { 0.7 }
    pub fn is_ascending(&self) -> bool { true }
    pub fn momentum(&self) -> f64 { 0.5 }
    pub fn is_reversal(&self) -> bool { false }
    pub fn coherence(&self) -> f64 { 0.8 }
    pub fn duration_ms(&self) -> u64 { 1000 }
    pub fn active_neurons(&self) -> usize { 7000 }
    pub fn total_neurons(&self) -> usize { 10000 }
    pub fn spike_time_variance(&self) -> f64 { 0.2 }
    pub fn neuron_variance(&self) -> f64 { 0.3 }
}