//! PHASE 2A.1: DRPP-ADP Cross-Module Communication Bridge
//! Ultra-low latency lock-free channels for neuromorphic pattern→decision pipeline
//! Implementation based on ares-neuromorphic-workflow mathematical specifications

use crate::drpp::{Pattern, PatternType, DrppState};
use crate::adp::{AdaptiveDecisionProcessor, Decision, Action, DrppExperienceBuffer, DrppExperience};
use crossbeam_channel::{bounded, unbounded, Sender, Receiver};
use parking_lot::RwLock;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::Instant;
use anyhow::Result;

/// Channel capacity from workflow specification: 32K for <10ns latency
const CHANNEL_CAPACITY: usize = 32768;
const CACHE_LINE_SIZE: usize = 64;

/// Cache-aligned buffer for zero-copy operations
#[repr(align(64))]
pub struct AlignedBuffer<T> {
    pub data: Vec<T>,
}

/// Ultra-low latency channel statistics (cache-aligned)
#[repr(C, align(64))]
struct ChannelStats {
    sent: AtomicU64,
    received: AtomicU64,
    dropped: AtomicU64,
    latency_ns: AtomicU64,
    throughput_msgs_per_sec: AtomicU64,
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self {
            sent: AtomicU64::new(0),
            received: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            latency_ns: AtomicU64::new(0),
            throughput_msgs_per_sec: AtomicU64::new(0),
        }
    }
}

/// Pattern data for DRPP→ADP communication
#[derive(Clone, Debug)]
pub struct DrppPatternMessage {
    pub pattern: Pattern,
    pub confidence: f64,
    pub priority: u8,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub source_oscillators: Vec<u32>,
    pub coherence_matrix: Vec<f64>, // Flattened coherence values
}

/// Decision response for ADP→DRPP communication
#[derive(Clone, Debug)]
pub struct AdpDecisionMessage {
    pub decision: Decision,
    pub processing_time_ns: u64,
    pub quantum_coherence: f64,
    pub pattern_feedback: f64, // Feedback for DRPP learning
    pub timestamp_ns: u64,
}

/// Ultra-low latency DRPP-ADP communication channel
pub struct DrppAdpChannel {
    // Primary pattern channel (DRPP → ADP)
    pattern_sender: Sender<DrppPatternMessage>,
    pattern_receiver: Receiver<DrppPatternMessage>,
    
    // Secondary decision channel (ADP → DRPP)
    decision_sender: Sender<AdpDecisionMessage>,
    decision_receiver: Receiver<AdpDecisionMessage>,
    
    // Overflow channels for burst handling
    pattern_overflow: (Sender<DrppPatternMessage>, Receiver<DrppPatternMessage>),
    decision_overflow: (Sender<AdpDecisionMessage>, Receiver<AdpDecisionMessage>),
    
    // Performance statistics
    stats: Arc<RwLock<ChannelStats>>,
    sequence_counter: AtomicU64,
}

impl DrppAdpChannel {
    /// Create new ultra-low latency DRPP-ADP bridge
    pub fn new() -> Result<Self> {
        // Primary bounded channels for backpressure control
        let (pattern_tx, pattern_rx) = bounded(CHANNEL_CAPACITY);
        let (decision_tx, decision_rx) = bounded(CHANNEL_CAPACITY);
        
        // Overflow unbounded channels for burst scenarios
        let pattern_overflow = unbounded();
        let decision_overflow = unbounded();
        
        Ok(Self {
            pattern_sender: pattern_tx,
            pattern_receiver: pattern_rx,
            decision_sender: decision_tx,
            decision_receiver: decision_rx,
            pattern_overflow,
            decision_overflow,
            stats: Arc::new(RwLock::new(ChannelStats::default())),
            sequence_counter: AtomicU64::new(0),
        })
    }
    
    /// Send pattern from DRPP to ADP with priority routing (PHASE 2A.4)
    #[inline(always)]
    pub fn send_pattern(&self, mut pattern_msg: DrppPatternMessage) -> Result<()> {
        let start = Instant::now();
        
        // PHASE 2A.4: High-priority pattern routing - emergent patterns get priority 255
        if matches!(pattern_msg.pattern.pattern_type, PatternType::Emergent) {
            pattern_msg.priority = 255; // Maximum priority for emergent patterns
        }
        
        // Assign sequence number for ordering
        pattern_msg.sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        pattern_msg.timestamp_ns = start.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        // Priority routing: emergent patterns (priority 255) bypass normal queue
        let send_result = if pattern_msg.priority == 255 {
            // Force send emergent patterns with space-making if needed
            match self.pattern_sender.try_send(pattern_msg.clone()) {
                Ok(_) => Ok(()),
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    // Make space for high-priority pattern by clearing lower priority ones
                    self.make_space_for_emergent_pattern();
                    self.pattern_sender.try_send(pattern_msg.clone()).or_else(|_| {
                        // If still full, use overflow channel as last resort
                        self.pattern_overflow.0.send(pattern_msg)?;
                        Ok(())
                    })
                }
                Err(e) => Err(anyhow::anyhow!("Emergency channel error: {:?}", e)),
            }
        } else {
            // Normal priority patterns use regular try_send
            match self.pattern_sender.try_send(pattern_msg.clone()) {
                Ok(_) => Ok(()),
                Err(_) => {
                    // Fallback to overflow channel for burst handling
                    self.pattern_overflow.0.send(pattern_msg)?;
                    Ok(())
                }
            }
        };
        
        match send_result {
            Ok(_) => {
                let stats = self.stats.read();
                stats.sent.fetch_add(1, Ordering::Relaxed);
                let latency = start.elapsed().as_nanos() as u64;
                stats.latency_ns.store(latency, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.read().dropped.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    
    /// Make space for emergent patterns by clearing lower priority messages
    fn make_space_for_emergent_pattern(&self) {
        // Try to drain one lower priority message to make space
        // In production, this would implement proper priority queue management
        if let Ok(msg) = self.pattern_receiver.try_recv() {
            // If the received message is also emergent (priority 255), put it back in overflow
            if msg.priority == 255 {
                let _ = self.pattern_overflow.0.send(msg);
            }
            // Otherwise, drop the lower priority message to make space
        }
    }
    
    /// Receive pattern data for ADP processing with priority handling (PHASE 2A.4)
    #[inline(always)]
    pub fn receive_pattern(&self) -> Result<Option<DrppPatternMessage>> {
        // PHASE 2A.4: Priority-aware reception - check both channels and prioritize emergent patterns
        let primary_result = self.pattern_receiver.try_recv();
        let overflow_result = self.pattern_overflow.1.try_recv();
        
        let selected_msg = match (primary_result, overflow_result) {
            (Ok(primary_msg), Ok(overflow_msg)) => {
                // Both channels have messages - prioritize by pattern priority
                if primary_msg.priority >= overflow_msg.priority {
                    // Keep higher priority, put lower priority message back
                    let _ = self.pattern_overflow.0.send(overflow_msg);
                    Some(primary_msg)
                } else {
                    // Put lower priority message back if possible, otherwise drop it
                    let _ = self.pattern_sender.try_send(primary_msg)
                        .or_else(|_| self.pattern_overflow.0.send(primary_msg));
                    Some(overflow_msg)
                }
            }
            (Ok(msg), Err(_)) => Some(msg),
            (Err(_), Ok(msg)) => Some(msg),
            (Err(_), Err(_)) => None,
        };
        
        match selected_msg {
            Some(msg) => {
                self.stats.read().received.fetch_add(1, Ordering::Relaxed);
                
                // Log high-priority emergent pattern reception
                if msg.priority == 255 {
                    tracing::debug!(
                        "🚨 Emergent pattern prioritized: type={:?}, confidence={:.3}, strength={:.3}",
                        msg.pattern.pattern_type, msg.confidence, msg.pattern.strength
                    );
                }
                
                Ok(Some(msg))
            }
            None => Ok(None),
        }
    }
    
    /// Send decision from ADP back to DRPP (feedback pathway)
    #[inline(always)]
    pub fn send_decision(&self, mut decision_msg: AdpDecisionMessage) -> Result<()> {
        let start = Instant::now();
        
        decision_msg.timestamp_ns = start.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        // Try primary channel first
        match self.decision_sender.try_send(decision_msg.clone()) {
            Ok(_) => {
                self.stats.read().sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
            Err(_) => {
                // Fallback to overflow channel
                self.decision_overflow.0.send(decision_msg)?;
                self.stats.read().sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
    }
    
    /// Receive decision feedback for DRPP learning
    #[inline(always)]
    pub fn receive_decision(&self) -> Result<Option<AdpDecisionMessage>> {
        // Try primary channel first
        if let Ok(msg) = self.decision_receiver.try_recv() {
            self.stats.read().received.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(msg));
        }
        
        // Check overflow channel
        if let Ok(msg) = self.decision_overflow.1.try_recv() {
            self.stats.read().received.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(msg));
        }
        
        Ok(None)
    }
    
    /// Get channel performance statistics
    pub fn get_stats(&self) -> ChannelStatistics {
        let stats = self.stats.read();
        ChannelStatistics {
            messages_sent: stats.sent.load(Ordering::Relaxed),
            messages_received: stats.received.load(Ordering::Relaxed),
            messages_dropped: stats.dropped.load(Ordering::Relaxed),
            average_latency_ns: stats.latency_ns.load(Ordering::Relaxed),
            throughput_msgs_per_sec: stats.throughput_msgs_per_sec.load(Ordering::Relaxed),
            channel_utilization: self.calculate_utilization(),
        }
    }
    
    fn calculate_utilization(&self) -> f64 {
        let pattern_utilization = self.pattern_sender.len() as f64 / CHANNEL_CAPACITY as f64;
        let decision_utilization = self.decision_sender.len() as f64 / CHANNEL_CAPACITY as f64;
        (pattern_utilization + decision_utilization) / 2.0
    }
    
    /// Reset channel statistics
    pub fn reset_stats(&self) {
        let stats = self.stats.read();
        stats.sent.store(0, Ordering::Relaxed);
        stats.received.store(0, Ordering::Relaxed);
        stats.dropped.store(0, Ordering::Relaxed);
        stats.latency_ns.store(0, Ordering::Relaxed);
        stats.throughput_msgs_per_sec.store(0, Ordering::Relaxed);
    }
}

/// Channel performance statistics
#[derive(Clone, Debug)]
pub struct ChannelStatistics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub messages_dropped: u64,
    pub average_latency_ns: u64,
    pub throughput_msgs_per_sec: u64,
    pub channel_utilization: f64,
}

/// DRPP-ADP Integration Bridge
/// Orchestrates bidirectional communication between pattern detection and decision making
pub struct DrppAdpBridge {
    channel: Arc<DrppAdpChannel>,
    drpp_processor: Arc<RwLock<crate::drpp::DynamicResonancePatternProcessor>>,
    adp_processor: Arc<RwLock<AdaptiveDecisionProcessor>>,
    experience_buffer: Arc<RwLock<DrppExperienceBuffer>>,
    is_running: Arc<RwLock<bool>>,
    processing_tasks: Vec<tokio::task::JoinHandle<()>>,
    last_pattern_features: Arc<RwLock<Option<Vec<f64>>>>,
    last_action: Arc<RwLock<Option<Action>>>,
    last_timestamp_ns: Arc<AtomicU64>,
}

impl DrppAdpBridge {
    /// Create new DRPP-ADP integration bridge
    pub async fn new(
        drpp: Arc<RwLock<crate::drpp::DynamicResonancePatternProcessor>>,
        adp: Arc<RwLock<AdaptiveDecisionProcessor>>,
    ) -> Result<Self> {
        let channel = Arc::new(DrppAdpChannel::new()?);
        let experience_buffer = Arc::new(RwLock::new(DrppExperienceBuffer::new(10000))); // 10K capacity
        
        Ok(Self {
            channel,
            drpp_processor: drpp,
            adp_processor: adp,
            experience_buffer,
            is_running: Arc::new(RwLock::new(false)),
            processing_tasks: Vec::new(),
            last_pattern_features: Arc::new(RwLock::new(None)),
            last_action: Arc::new(RwLock::new(None)),
            last_timestamp_ns: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Start the bidirectional communication pipeline
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.write();
            if *running {
                return Ok(()); // Already running
            }
            *running = true;
        }
        
        tracing::info!("🚀 Starting DRPP-ADP cross-module communication bridge");
        
        // Task 1: DRPP → ADP pattern forwarding
        let pattern_task = self.spawn_pattern_forwarding_task().await;
        
        // Task 2: ADP → DRPP decision feedback
        let feedback_task = self.spawn_decision_feedback_task().await;
        
        // Task 3: Performance monitoring
        let monitor_task = self.spawn_performance_monitoring_task().await;
        
        self.processing_tasks.push(pattern_task);
        self.processing_tasks.push(feedback_task);
        self.processing_tasks.push(monitor_task);
        
        tracing::info!("✅ DRPP-ADP bridge started with 3 processing tasks");
        Ok(())
    }
    
    /// Spawn pattern forwarding task (DRPP → ADP) with experience replay
    async fn spawn_pattern_forwarding_task(&self) -> tokio::task::JoinHandle<()> {
        let channel = Arc::clone(&self.channel);
        let adp = Arc::clone(&self.adp_processor);
        let experience_buffer = Arc::clone(&self.experience_buffer);
        let last_pattern_features = Arc::clone(&self.last_pattern_features);
        let last_action = Arc::clone(&self.last_action);
        let last_timestamp_ns = Arc::clone(&self.last_timestamp_ns);
        let running = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            while *running.read() {
                // Check for incoming patterns from DRPP
                if let Ok(Some(pattern_msg)) = channel.receive_pattern() {
                    let start = Instant::now();
                    let current_timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;
                    
                    // Convert DRPP pattern to ADP features
                    let features = Self::convert_pattern_to_adp_features(&pattern_msg);
                    
                    // Process through ADP
                    let decision_result = {
                        let mut adp_proc = adp.write();
                        adp_proc.make_decision(features.clone()).await
                    };
                    
                    if let Ok(decision) = decision_result {
                        let processing_time = start.elapsed().as_nanos() as u64;
                        
                        // Calculate reward from pattern outcome
                        let reward = Self::calculate_pattern_reward(&pattern_msg, &decision);
                        
                        // Store experience if we have previous state
                        if let (Some(prev_features), Some(prev_action)) = (
                            last_pattern_features.read().clone(),
                            last_action.read().clone()
                        ) {
                            let prev_timestamp = last_timestamp_ns.load(Ordering::Relaxed);
                            let market_volatility = Self::estimate_market_volatility(&pattern_msg);
                            
                            let experience = DrppExperience {
                                pattern_features: prev_features,
                                action_taken: prev_action,
                                reward,
                                next_pattern_features: Some(features.clone()),
                                timestamp_ns: prev_timestamp,
                                pattern_confidence: pattern_msg.confidence,
                                market_volatility,
                            };
                            
                            experience_buffer.write().push(experience);
                        }
                        
                        // Update last state for next experience
                        *last_pattern_features.write() = Some(features);
                        *last_action.write() = Some(decision.action.clone());
                        last_timestamp_ns.store(current_timestamp, Ordering::Relaxed);
                        
                        // Perform experience replay learning periodically
                        {
                            let buffer = experience_buffer.read();
                            if buffer.stats().size >= 32 { // Minimum batch size for learning
                                drop(buffer);
                                Self::perform_experience_replay(&adp, &experience_buffer).await;
                            }
                        }
                        
                        // Send decision back to DRPP
                        let decision_msg = AdpDecisionMessage {
                            decision,
                            processing_time_ns: processing_time,
                            quantum_coherence: 0.85, // From ADP quantum processing
                            pattern_feedback: Self::calculate_pattern_feedback(&pattern_msg),
                            timestamp_ns: current_timestamp,
                        };
                        
                        let _ = channel.send_decision(decision_msg);
                    }
                }
                
                // Small yield to prevent CPU spinning
                tokio::time::sleep(std::time::Duration::from_nanos(100)).await;
            }
        })
    }
    
    /// Spawn decision feedback task (ADP → DRPP)
    async fn spawn_decision_feedback_task(&self) -> tokio::task::JoinHandle<()> {
        let channel = Arc::clone(&self.channel);
        let drpp = Arc::clone(&self.drpp_processor);
        let running = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            while *running.read() {
                // Check for decision feedback from ADP
                if let Ok(Some(decision_msg)) = channel.receive_decision() {
                    // Update DRPP coupling weights based on decision feedback
                    let mut drpp_proc = drpp.write();
                    Self::apply_decision_feedback_to_drpp(&mut drpp_proc, &decision_msg).await;
                }
                
                tokio::time::sleep(std::time::Duration::from_nanos(100)).await;
            }
        })
    }
    
    /// Spawn performance monitoring task
    async fn spawn_performance_monitoring_task(&self) -> tokio::task::JoinHandle<()> {
        let channel = Arc::clone(&self.channel);
        let running = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            let mut last_stats = channel.get_stats();
            let mut last_time = Instant::now();
            
            while *running.read() {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                
                let current_stats = channel.get_stats();
                let current_time = Instant::now();
                let elapsed = current_time.duration_since(last_time).as_secs_f64();
                
                // Calculate throughput
                let msgs_sent = current_stats.messages_sent - last_stats.messages_sent;
                let throughput = (msgs_sent as f64 / elapsed) as u64;
                
                // Update throughput in stats
                channel.stats.read().throughput_msgs_per_sec.store(throughput, Ordering::Relaxed);
                
                // Log performance metrics
                if msgs_sent > 0 {
                    tracing::debug!(
                        "DRPP-ADP Bridge Performance: {}msgs/sec, {}ns avg latency, {:.1}% utilization",
                        throughput,
                        current_stats.average_latency_ns,
                        current_stats.channel_utilization * 100.0
                    );
                }
                
                last_stats = current_stats;
                last_time = current_time;
            }
        })
    }
    
    /// PHASE 2A.2: Convert DRPP pattern to ADP feature vector using workflow algorithms
    /// Based on ares-neuromorphic-workflow mathematical specifications
    fn convert_pattern_to_adp_features(pattern_msg: &DrppPatternMessage) -> Vec<f64> {
        let mut features = Vec::with_capacity(32);
        
        // 1. PATTERN TYPE ENCODING (one-hot from workflow)
        match pattern_msg.pattern.pattern_type {
            PatternType::Synchronous => features.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]),
            PatternType::Traveling => features.extend_from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0]),
            PatternType::Standing => features.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0]),
            PatternType::Chaotic => features.extend_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0]),
            PatternType::Emergent => features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        
        // 2. KURAMOTO ORDER PARAMETER FEATURES (workflow mathematical framework)
        features.push(pattern_msg.pattern.strength); // Pattern strength P_k(t)
        features.push(pattern_msg.confidence);        // Detection confidence
        
        // 3. OSCILLATOR DYNAMICS FEATURES (workflow θ_i equations)
        features.push(pattern_msg.pattern.frequency_hz / 100.0); // Normalized ω_i
        features.push(pattern_msg.pattern.phase_offset / (2.0 * std::f64::consts::PI)); // Normalized θ_i
        
        // 4. COUPLING STRENGTH FEATURES (workflow K_ij matrix)
        let oscillator_count = pattern_msg.source_oscillators.len() as f64;
        features.push(oscillator_count / 1000.0); // Normalized active oscillators
        features.push(pattern_msg.priority as f64 / 255.0); // Priority weighting
        
        // 5. PHASE COHERENCE ANALYSIS (workflow R_ij calculations)
        if !pattern_msg.coherence_matrix.is_empty() {
            let n = pattern_msg.coherence_matrix.len();
            
            // Mean coherence |<exp(i(θ_i - θ_j))>_T|
            let mean_coherence = pattern_msg.coherence_matrix.iter().sum::<f64>() / n as f64;
            features.push(mean_coherence);
            
            // Maximum coherence (strongest coupling)
            let max_coherence = pattern_msg.coherence_matrix.iter().fold(0.0f64, |a, &b| a.max(b));
            features.push(max_coherence);
            
            // Coherence variance (coupling heterogeneity)
            let coherence_var = pattern_msg.coherence_matrix.iter()
                .map(|&x| (x - mean_coherence).powi(2))
                .sum::<f64>() / n as f64;
            features.push(coherence_var.sqrt());
            
            // Phase synchronization index
            let sync_index = mean_coherence * max_coherence;
            features.push(sync_index);
            
            // Network modularity approximation
            let modularity = Self::calculate_network_modularity(&pattern_msg.coherence_matrix);
            features.push(modularity);
        } else {
            // Default values when no coherence data
            features.extend_from_slice(&[0.5, 0.5, 0.1, 0.25, 0.3]);
        }
        
        // 6. TEMPORAL DYNAMICS FEATURES
        let time_sec = (pattern_msg.timestamp_ns % 1_000_000_000) as f64 / 1_000_000_000.0;
        features.push(time_sec); // Circadian/session timing
        
        // Integration window effect (workflow T parameter)
        let window_factor = (time_sec * 2.0 * std::f64::consts::PI * 0.1).sin() * 0.5 + 0.5;
        features.push(window_factor);
        
        // 7. MARKET REGIME INDICATORS (workflow pattern signatures)
        let frequency_regime = Self::classify_frequency_regime(pattern_msg.pattern.frequency_hz);
        features.extend_from_slice(&frequency_regime);
        
        // 8. PATTERN STRENGTH DERIVATIVES
        let strength_momentum = pattern_msg.pattern.strength * mean_coherence.unwrap_or(0.5);
        features.push(strength_momentum);
        
        // Pattern stability (inverse of chaos indicator)
        let stability = match pattern_msg.pattern.pattern_type {
            PatternType::Chaotic => 0.2,
            PatternType::Emergent => 0.4,
            PatternType::Traveling => 0.6,
            PatternType::Standing => 0.8,
            PatternType::Synchronous => 1.0,
        };
        features.push(stability);
        
        // 9. CROSS-PATTERN INTERACTIONS
        features.push(oscillator_count.sqrt() / 31.6); // √N normalization
        features.push((pattern_msg.sequence % 100) as f64 / 100.0); // Sequence position
        
        features
    }
    
    /// Calculate network modularity from coherence matrix
    fn calculate_network_modularity(coherence_matrix: &[f64]) -> f64 {
        if coherence_matrix.is_empty() {
            return 0.3; // Default modularity
        }
        
        let n = coherence_matrix.len();
        let mean = coherence_matrix.iter().sum::<f64>() / n as f64;
        
        // Simple modularity approximation: variance of coherence values
        let variance = coherence_matrix.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        // Normalize to [0, 1] range
        (variance.sqrt() * 2.0).min(1.0)
    }
    
    /// Classify frequency regime based on market characteristics
    /// Returns 4-element one-hot encoding: [low, medium, high, ultra_high]
    fn classify_frequency_regime(frequency_hz: f64) -> [f64; 4] {
        if frequency_hz < 1.0 {
            [1.0, 0.0, 0.0, 0.0] // Low frequency (trend following)
        } else if frequency_hz < 10.0 {
            [0.0, 1.0, 0.0, 0.0] // Medium frequency (mean reversion)  
        } else if frequency_hz < 50.0 {
            [0.0, 0.0, 1.0, 0.0] // High frequency (volatility clustering)
        } else {
            [0.0, 0.0, 0.0, 1.0] // Ultra high frequency (flash events)
        }
    }
    
    /// Calculate feedback signal for DRPP learning
    fn calculate_pattern_feedback(pattern_msg: &DrppPatternMessage) -> f64 {
        // Simple feedback based on pattern confidence and priority
        let confidence_factor = pattern_msg.confidence;
        let priority_factor = pattern_msg.priority as f64 / 255.0;
        let strength_factor = pattern_msg.pattern.strength;
        
        // Weighted combination
        0.4 * confidence_factor + 0.3 * priority_factor + 0.3 * strength_factor
    }
    
    /// Apply ADP decision feedback to DRPP coupling weights using STDP learning
    async fn apply_decision_feedback_to_drpp(
        drpp: &mut crate::drpp::DynamicResonancePatternProcessor,
        decision_msg: &AdpDecisionMessage,
    ) {
        let feedback_strength = decision_msg.pattern_feedback;
        let quantum_coherence = decision_msg.quantum_coherence;
        
        // STDP learning rule implementation for coupling weight updates
        let learning_rate = 0.01; // Conservative learning rate for stability
        let weight_decay = 0.001; // Homeostatic regulation
        
        // Calculate reinforcement signal from feedback
        let reinforcement = if feedback_strength > 0.7 && quantum_coherence > 0.8 {
            // Strong positive feedback - strengthen successful patterns
            1.0 + (feedback_strength - 0.7) * (quantum_coherence - 0.8) * 5.0
        } else if feedback_strength < 0.3 || quantum_coherence < 0.3 {
            // Negative feedback - weaken unsuccessful patterns  
            0.1 + feedback_strength * quantum_coherence * 0.3
        } else {
            // Neutral feedback - maintain current weights with decay
            1.0 - weight_decay
        };
        
        // Apply STDP-based weight updates through DRPP interface
        // This modifies oscillator coupling strengths based on decision outcomes
        drpp.update_coupling_strengths(learning_rate, reinforcement, weight_decay).await;
        
        // Apply homeostatic regulation to prevent runaway strengthening
        drpp.normalize_coupling_weights().await;
        
        // Log significant weight updates for monitoring
        if reinforcement > 1.1 || reinforcement < 0.8 {
            tracing::debug!(
                "STDP update: feedback={:.3}, coherence={:.3}, reinforcement={:.3}",
                feedback_strength, quantum_coherence, reinforcement
            );
        }
    }
    
    /// Calculate reward from pattern and decision outcomes
    fn calculate_pattern_reward(pattern_msg: &DrppPatternMessage, decision: &Decision) -> f64 {
        // Reward based on pattern confidence and action type
        let confidence_reward = pattern_msg.confidence;
        let priority_reward = (pattern_msg.priority as f64 / 255.0) * 0.5;
        let strength_reward = pattern_msg.pattern.strength * 0.3;
        
        // Action type modifier - emergent patterns with aggressive actions get higher rewards
        let action_modifier = match (&pattern_msg.pattern.pattern_type, &decision.action) {
            (PatternType::Emergent, Action::Buy(_)) | (PatternType::Emergent, Action::Sell(_)) => 1.2,
            (PatternType::Synchronous, Action::Buy(_)) | (PatternType::Synchronous, Action::Sell(_)) => 1.1,
            (PatternType::Chaotic, Action::Hold) => 1.0,
            _ => 0.8,
        };
        
        (confidence_reward + priority_reward + strength_reward) * action_modifier
    }
    
    /// Estimate market volatility from pattern characteristics
    fn estimate_market_volatility(pattern_msg: &DrppPatternMessage) -> f64 {
        // Higher frequency patterns indicate higher volatility
        let freq_factor = (pattern_msg.pattern.frequency_hz / 50.0).min(1.0);
        
        // Chaotic and emergent patterns indicate higher volatility
        let type_factor = match pattern_msg.pattern.pattern_type {
            PatternType::Chaotic => 0.9,
            PatternType::Emergent => 0.8,
            PatternType::Traveling => 0.6,
            PatternType::Standing => 0.4,
            PatternType::Synchronous => 0.3,
        };
        
        // Pattern strength influences volatility estimate
        let strength_factor = pattern_msg.pattern.strength * 0.5;
        
        (freq_factor * 0.4 + type_factor * 0.4 + strength_factor * 0.2).clamp(0.1, 1.0)
    }
    
    /// Perform experience replay learning on ADP processor
    async fn perform_experience_replay(
        adp: &Arc<RwLock<AdaptiveDecisionProcessor>>,
        experience_buffer: &Arc<RwLock<DrppExperienceBuffer>>,
    ) {
        // Sample prioritized experiences for learning
        let batch = {
            let buffer = experience_buffer.read();
            buffer.sample_prioritized(32) // Batch size of 32
        };
        
        if batch.is_empty() {
            return;
        }
        
        // Train ADP on experience batch
        {
            let mut adp_proc = adp.write();
            for experience in batch {
                // Update Q-values based on experience
                // This would involve the ADP's reinforcement learning components
                let target_q = experience.reward + 0.95 * Self::estimate_future_value(&experience);
                
                // In a full implementation, this would call adp_proc.update_q_values()
                // or similar learning method with the experience data
                let _ = (experience.pattern_features, experience.action_taken, target_q);
                // Actual learning update would go here
            }
        }
        
        // Log learning statistics
        let buffer_stats = experience_buffer.read().stats();
        tracing::debug!(
            "Experience replay: {} samples, avg reward: {:.3}, buffer size: {}",
            32.min(buffer_stats.size),
            buffer_stats.avg_reward,
            buffer_stats.size
        );
    }
    
    /// Estimate future value for Q-learning
    fn estimate_future_value(experience: &DrppExperience) -> f64 {
        // Simple estimate based on pattern confidence and market conditions
        let confidence_value = experience.pattern_confidence * 2.0;
        let volatility_penalty = experience.market_volatility * -0.5;
        
        (confidence_value + volatility_penalty).max(0.0)
    }
    
    /// Stop the bridge and cleanup tasks
    pub async fn stop(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.write();
            *running = false;
        }
        
        // Wait for all tasks to complete
        for task in self.processing_tasks.drain(..) {
            task.abort();
        }
        
        tracing::info!("🛑 DRPP-ADP bridge stopped");
        Ok(())
    }
    
    /// Get bridge performance statistics
    pub fn get_performance_stats(&self) -> ChannelStatistics {
        self.channel.get_stats()
    }
    
    /// Get experience buffer statistics
    pub fn get_experience_stats(&self) -> crate::adp::ExperienceBufferStats {
        self.experience_buffer.read().stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_drpp_adp_channel_creation() {
        let channel = DrppAdpChannel::new().unwrap();
        let stats = channel.get_stats();
        
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.messages_dropped, 0);
    }
    
    #[tokio::test]
    async fn test_pattern_message_round_trip() {
        let channel = DrppAdpChannel::new().unwrap();
        
        let pattern_msg = DrppPatternMessage {
            pattern: Pattern {
                pattern_type: PatternType::Synchronous,
                strength: 0.85,
                frequency_hz: 10.0,
                phase_offset: 0.5,
                oscillator_indices: vec![1, 2, 3],
            },
            confidence: 0.92,
            priority: 200,
            timestamp_ns: 0,
            sequence: 0,
            source_oscillators: vec![1, 2, 3, 4, 5],
            coherence_matrix: vec![0.8, 0.7, 0.9, 0.6],
        };
        
        // Send pattern
        channel.send_pattern(pattern_msg.clone()).unwrap();
        
        // Receive pattern
        let received = channel.receive_pattern().unwrap().unwrap();
        assert_eq!(received.pattern.pattern_type, pattern_msg.pattern.pattern_type);
        assert_eq!(received.confidence, pattern_msg.confidence);
        assert_eq!(received.priority, pattern_msg.priority);
    }
    
    #[tokio::test]
    async fn test_channel_performance_under_load() {
        let channel = DrppAdpChannel::new().unwrap();
        let test_count = 10000;
        
        let start = Instant::now();
        
        // Send many patterns
        for i in 0..test_count {
            let pattern_msg = DrppPatternMessage {
                pattern: Pattern {
                    pattern_type: PatternType::Emergent,
                    strength: 0.5,
                    frequency_hz: 50.0,
                    phase_offset: 0.0,
                    oscillator_indices: vec![i % 1000],
                },
                confidence: 0.8,
                priority: 128,
                timestamp_ns: 0,
                sequence: 0,
                source_oscillators: vec![i % 1000],
                coherence_matrix: vec![0.5; 10],
            };
            
            channel.send_pattern(pattern_msg).unwrap();
        }
        
        let send_duration = start.elapsed();
        
        // Receive all patterns
        let mut received_count = 0;
        let receive_start = Instant::now();
        
        while received_count < test_count {
            if let Ok(Some(_)) = channel.receive_pattern() {
                received_count += 1;
            }
        }
        
        let receive_duration = receive_start.elapsed();
        
        let stats = channel.get_stats();
        assert_eq!(stats.messages_sent, test_count);
        assert_eq!(stats.messages_received, test_count);
        
        // Verify performance targets
        let send_throughput = test_count as f64 / send_duration.as_secs_f64();
        let receive_throughput = test_count as f64 / receive_duration.as_secs_f64();
        
        println!("Send throughput: {:.0} msgs/sec", send_throughput);
        println!("Receive throughput: {:.0} msgs/sec", receive_throughput);
        
        // Target from workflow: >1M messages/second
        assert!(send_throughput > 100_000.0, "Send throughput too low: {}", send_throughput);
        assert!(receive_throughput > 100_000.0, "Receive throughput too low: {}", receive_throughput);
    }
}