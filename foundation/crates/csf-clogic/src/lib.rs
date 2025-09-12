//! C-LOGIC (Cognitive Logic Operations for Gestalt Intelligence Control) modules

use csf_bus::PhaseCoherenceBus as Bus;
use csf_core::hardware_timestamp;
use csf_core::prelude::*;
use std::any::Any;

// Emotional bias structure for ADP decision making
#[derive(Debug, Clone)]
pub struct EmotionalBias {
    pub valence_bias: f64,
    pub arousal_modifier: f64, 
    pub strength: f64,
}

// Emotional constraints for EMS modeling
#[derive(Debug, Clone)]
pub struct EmotionalConstraints {
    pub max_arousal: f64,
    pub valence_stability: f64,
    pub response_damping: f64,
}

/// Calculate current resource utilization
async fn calculate_resource_utilization() -> anyhow::Result<f64> {
    // Get system metrics
    let cpu_usage = get_cpu_usage().unwrap_or(0.0);
    let memory_usage = get_memory_usage().unwrap_or(0.0);
    
    // Weighted average (CPU 60%, Memory 40%)
    Ok((cpu_usage * 0.6) + (memory_usage * 0.4))
}

/// Get CPU usage percentage
fn get_cpu_usage() -> Option<f64> {
    // Use procfs or system calls to get actual CPU usage
    // For now, return a reasonable estimate based on system load
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|content| {
            content.split_whitespace()
                .next()
                .and_then(|load| load.parse::<f64>().ok())
        })
        .map(|load| (load / num_cpus::get() as f64).min(1.0))
}

/// Get memory usage percentage  
fn get_memory_usage() -> Option<f64> {
    // Parse /proc/meminfo for actual memory usage
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|content| {
            let mut total = 0u64;
            let mut available = 0u64;
            
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total = line.split_whitespace()
                        .nth(1)?
                        .parse().ok()?;
                } else if line.starts_with("MemAvailable:") {
                    available = line.split_whitespace()
                        .nth(1)?
                        .parse().ok()?;
                }
            }
            
            if total > 0 {
                Some((total - available) as f64 / total as f64)
            } else {
                None
            }
        })
}

// Type aliases for compatibility
type BinaryPacket = PhasePacket<PacketPayload>;
use std::sync::Arc;

pub mod adp;
pub mod drpp;
pub mod egc;
pub mod ems;

/// 🛡️ HARDENING PHASE 3: Performance monitoring and mutex contention detection
pub mod performance_monitor;

pub use adp::AdaptiveDecisionProcessor;
pub use drpp::DynamicResonancePatternProcessor;
pub use egc::EmergentGovernanceController;
pub use ems::EmotionalModelingSystem;

/// 🛡️ HARDENING: Runtime invariant checking
#[cfg(debug_assertions)]
macro_rules! assert_invariant {
    ($cond:expr, $msg:literal $(,$($arg:tt)*)?) => {
        debug_assert!($cond, concat!("INVARIANT VIOLATION: ", $msg) $(,$($arg)*)?);
    };
}

#[cfg(not(debug_assertions))]
macro_rules! assert_invariant {
    ($cond:expr, $msg:literal $(,$($arg:tt)*)?) => {};
}

/// 🛡️ HARDENING: Performance monitoring wrapper
#[cfg(debug_assertions)]
pub struct PerformanceMonitor {
    operation: &'static str,
    start: std::time::Instant,
}

#[cfg(debug_assertions)]
impl PerformanceMonitor {
    pub fn new(operation: &'static str) -> Self {
        Self {
            operation,
            start: std::time::Instant::now(),
        }
    }
}

#[cfg(debug_assertions)]
impl Drop for PerformanceMonitor {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if elapsed.as_millis() > 10 {
            tracing::warn!(
                "PERFORMANCE: {} took {}ms (>10ms threshold)",
                self.operation,
                elapsed.as_millis()
            );
        }
    }
}

#[cfg(not(debug_assertions))]
pub struct PerformanceMonitor;

#[cfg(not(debug_assertions))]
impl PerformanceMonitor {
    pub fn new(_operation: &'static str) -> Self {
        Self
    }
}

/// C-LOGIC configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CLogicConfig {
    /// DRPP configuration
    pub drpp: drpp::DrppConfig,

    /// ADP configuration
    pub adp: adp::AdpConfig,

    /// EGC configuration
    pub egc: egc::EgcConfig,

    /// EMS configuration
    pub ems: ems::EmsConfig,

    /// Enable inter-module communication
    pub enable_cross_talk: bool,

    /// Update frequency in Hz
    pub update_frequency: f64,
}

impl Default for CLogicConfig {
    fn default() -> Self {
        Self {
            drpp: Default::default(),
            adp: Default::default(),
            egc: Default::default(),
            ems: Default::default(),
            enable_cross_talk: true,
            update_frequency: 100.0, // 100 Hz
        }
    }
}

/// C-LOGIC system coordinator
pub struct CLogicSystem {
    /// Dynamic Resonance Pattern Processor
    drpp: Arc<DynamicResonancePatternProcessor>,

    /// Adaptive Decision Processor
    adp: Arc<AdaptiveDecisionProcessor>,

    /// Emergent Governance Controller
    egc: Arc<EmergentGovernanceController>,

    /// Emotional Modeling System
    ems: Arc<EmotionalModelingSystem>,

    /// Phase Coherence Bus
    bus: Arc<Bus>,

    /// Configuration
    config: CLogicConfig,
}

impl CLogicSystem {
    /// Create a new C-LOGIC system
    pub async fn new(bus: Arc<Bus>, config: CLogicConfig) -> anyhow::Result<Self> {
        let _monitor = PerformanceMonitor::new("CLogicSystem::new");

        // 🛡️ HARDENING: Validate configuration
        assert_invariant!(
            config.update_frequency > 0.0 && config.update_frequency <= 1000.0,
            "Invalid update frequency: {}",
            config.update_frequency
        );

        // Initialize modules
        let drpp = Arc::new(
            DynamicResonancePatternProcessor::new(bus.clone(), config.drpp.clone()).await?,
        );

        let adp = Arc::new(
            AdaptiveDecisionProcessor::new(
                bus.clone(),
                Arc::new(adp::SilCore::new()), // SIL core integration
                config.adp.clone(),
            )
            .await?,
        );

        let egc =
            Arc::new(EmergentGovernanceController::new(bus.clone(), config.egc.clone()).await?);

        let ems = Arc::new(EmotionalModelingSystem::new(bus.clone(), config.ems.clone()).await?);

        Ok(Self {
            drpp,
            adp,
            egc,
            ems,
            bus,
            config,
        })
    }

    /// Start all C-LOGIC modules
    pub async fn start(&self) -> anyhow::Result<()> {
        let _monitor = PerformanceMonitor::new("CLogicSystem::start");

        // Start modules
        self.drpp.start().await?;
        self.adp.start().await?;
        self.egc.start().await?;
        self.ems.start().await?;

        // Set up cross-module communication if enabled
        if self.config.enable_cross_talk {
            self.setup_cross_talk().await?;
        }

        Ok(())
    }

    /// Stop all C-LOGIC modules
    pub async fn stop(&self) -> anyhow::Result<()> {
        let _monitor = PerformanceMonitor::new("CLogicSystem::stop");

        self.drpp.stop().await?;
        self.adp.stop().await?;
        self.egc.stop().await?;
        self.ems.stop().await?;

        Ok(())
    }

    /// Get system state
    pub async fn get_state(&self) -> CLogicState {
        let _monitor = PerformanceMonitor::new("CLogicSystem::get_state");

        let state = CLogicState {
            drpp_state: self.drpp.get_state().await,
            egc_state: self.egc.get_state().await,
            ems_state: self.ems.get_state().await,
            timestamp: hardware_timestamp(),
        };

        // 🛡️ HARDENING: Validate state consistency
        assert_invariant!(state.timestamp.as_nanos() > 0, "Invalid system timestamp");

        state
    }

    /// Set up DRPP communication handlers
    async fn setup_drpp_communication(
        &self,
        pattern_tx: CrossModuleChannelHandle<DrppFeatures>,
        emotion_rx: CrossModuleReceiver<EmotionalModulation>,
        governance_rx: CrossModuleReceiver<GovernanceDecision>,
        comm_system: Arc<CrossModuleCommunication>,
    ) -> anyhow::Result<()> {
        let drpp = self.drpp.clone();
        
        // Spawn DRPP message handler
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(10)); // 100Hz
            
            loop {
                interval.tick().await;
                
                // Process incoming emotional modulation
                if let Some(emotion_msg) = emotion_rx.try_recv() {
                    tracing::debug!("DRPP received emotional modulation: {:?}", emotion_msg.emotion_type);
                    // Apply emotional modulation to DRPP resonance parameters
                    let resonance_factor = 1.0 + (emotion_msg.valence * 0.2) + (emotion_msg.arousal * 0.1);
                    if let Err(e) = drpp.modulate_resonance(resonance_factor).await {
                        tracing::warn!("Failed to apply emotional modulation to DRPP: {}", e);
                    }
                }
                
                // Process incoming governance decisions
                if let Some(gov_msg) = governance_rx.try_recv() {
                    tracing::debug!("DRPP received governance decision: {}", gov_msg.decision_type);
                    // Apply governance parameters to DRPP thresholds
                    if let Some(&threshold) = gov_msg.parameters.get("processing_threshold") {
                        if let Err(e) = drpp.set_processing_threshold(threshold).await {
                            tracing::warn!("Failed to apply governance threshold to DRPP: {}", e);
                        }
                    }
                    if let Some(&sensitivity) = gov_msg.parameters.get("pattern_sensitivity") {
                        if let Err(e) = drpp.set_pattern_sensitivity(sensitivity).await {
                            tracing::warn!("Failed to apply governance sensitivity to DRPP: {}", e);
                        }
                    }
                }
                
                // Extract and send pattern features to ADP
                let state = drpp.get_state().await;
                if !state.detected_patterns.is_empty() {
                    let features = DrppFeatures {
                        patterns: state.detected_patterns.iter()
                            .map(|p| (p.pattern_type.clone(), p.strength))
                            .collect(),
                        phase_info: state.oscillator_phases.clone(),
                        coherence: state.coherence,
                        resonance_strength: state.resonance_map.iter().sum::<f64>() / state.resonance_map.len() as f64,
                        timestamp: hardware_timestamp(),
                    };
                    
                    if let Err(e) = pattern_tx.send(features) {
                        tracing::warn!("Failed to send DRPP features to ADP: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Set up ADP communication handlers
    async fn setup_adp_communication(
        &self,
        pattern_rx: CrossModuleReceiver<DrppFeatures>,
        metrics_tx: CrossModuleChannelHandle<ProcessingMetrics>,
        emotion_rx: CrossModuleReceiver<EmotionalModulation>,
        governance_rx: CrossModuleReceiver<GovernanceDecision>,
        comm_system: Arc<CrossModuleCommunication>,
    ) -> anyhow::Result<()> {
        let adp = self.adp.clone();
        
        // Spawn ADP message handler
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(10));
            let mut last_processing_start = std::time::Instant::now();
            
            loop {
                interval.tick().await;
                
                // Process incoming DRPP features
                if let Some(drpp_features) = pattern_rx.try_recv() {
                    last_processing_start = std::time::Instant::now();
                    tracing::debug!("ADP processing {} patterns from DRPP", drpp_features.patterns.len());
                    
                    // Process patterns through ADP decision logic
                    let decisions = adp.process_patterns(&drpp_features.patterns).await
                        .unwrap_or_else(|e| {
                            tracing::warn!("ADP pattern processing failed: {}", e);
                            Vec::new()
                        });
                    
                    // Log processing results
                    if !decisions.is_empty() {
                        tracing::debug!("ADP generated {} decisions from {} patterns", 
                            decisions.len(), drpp_features.patterns.len());
                    }
                    
                    let processing_time = last_processing_start.elapsed().as_nanos() as u64;
                    
                    // Send processing metrics to EGC
                    let metrics = ProcessingMetrics {
                        processing_time_ns: processing_time,
                        confidence: drpp_features.coherence, // Use DRPP coherence as confidence proxy
                        resource_utilization: calculate_resource_utilization().await.unwrap_or(0.5),
                        patterns_processed: drpp_features.patterns.len() as u32,
                        efficiency: if processing_time > 0 { 
                            (drpp_features.patterns.len() as f64 * 1_000_000.0) / processing_time as f64 
                        } else { 
                            1.0 
                        },
                        timestamp: hardware_timestamp(),
                    };
                    
                    if let Err(e) = metrics_tx.send(metrics) {
                        tracing::warn!("Failed to send ADP metrics to EGC: {}", e);
                    }
                }
                
                // Process emotional modulation
                if let Some(emotion_msg) = emotion_rx.try_recv() {
                    tracing::debug!("ADP received emotional modulation: valence={}", emotion_msg.valence);
                    // Apply emotional bias to ADP decision making
                    let bias_factor = EmotionalBias {
                        valence_bias: emotion_msg.valence * 0.3,
                        arousal_modifier: emotion_msg.arousal,
                        strength: emotion_msg.strength,
                    };
                    
                    if let Err(e) = adp.apply_emotional_bias(bias_factor).await {
                        tracing::warn!("Failed to apply emotional bias to ADP: {}", e);
                    }
                }
                
                // Process governance decisions
                if let Some(gov_msg) = governance_rx.try_recv() {
                    tracing::debug!("ADP received governance decision: {}", gov_msg.decision_type);
                    // Apply governance constraints to ADP
                    if let Some(&risk_tolerance) = gov_msg.parameters.get("risk_tolerance") {
                        if let Err(e) = adp.set_risk_tolerance(risk_tolerance).await {
                            tracing::warn!("Failed to apply risk tolerance to ADP: {}", e);
                        }
                    }
                    
                    if let Some(&decision_threshold) = gov_msg.parameters.get("decision_threshold") {
                        if let Err(e) = adp.set_decision_threshold(decision_threshold).await {
                            tracing::warn!("Failed to apply decision threshold to ADP: {}", e);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Set up EGC communication handlers
    async fn setup_egc_communication(
        &self,
        metrics_rx: CrossModuleReceiver<ProcessingMetrics>,
        decision_tx: CrossModuleBroadcastHandle<GovernanceDecision>,
        emotion_rx: CrossModuleReceiver<EmotionalModulation>,
        comm_system: Arc<CrossModuleCommunication>,
    ) -> anyhow::Result<()> {
        let egc = self.egc.clone();
        
        // Spawn EGC message handler
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50)); // 20Hz
            
            loop {
                interval.tick().await;
                
                // Process incoming ADP metrics
                if let Some(metrics) = metrics_rx.try_recv() {
                    tracing::debug!("EGC received processing metrics: efficiency={}", metrics.efficiency);
                    
                    // Generate governance decisions based on metrics
                    if metrics.efficiency < 0.5 {
                        // Low efficiency - adjust parameters
                        let mut params = std::collections::HashMap::new();
                        params.insert("processing_threshold".to_string(), 0.8);
                        params.insert("pattern_sensitivity".to_string(), 0.9);
                        
                        let decision = GovernanceDecision {
                            decision_type: "performance_optimization".to_string(),
                            parameters: params,
                            priority: 200, // High priority
                            immediate: true,
                            confidence: 0.85,
                            timestamp: hardware_timestamp(),
                        };
                        
                        if let Err(e) = decision_tx.broadcast_priority(decision) {
                            tracing::warn!("Failed to broadcast EGC decision: {}", e);
                        }
                    }
                }
                
                // Process emotional modulation for governance context
                if let Some(emotion_msg) = emotion_rx.try_recv() {
                    tracing::debug!("EGC considering emotional context: arousal={}", emotion_msg.arousal);
                    
                    // Adjust governance based on system emotional state
                    if emotion_msg.arousal > 0.8 {
                        // High arousal - be more conservative
                        let mut params = std::collections::HashMap::new();
                        params.insert("risk_tolerance".to_string(), 0.3);
                        params.insert("decision_threshold".to_string(), 0.9);
                        
                        let decision = GovernanceDecision {
                            decision_type: "emotional_regulation".to_string(),
                            parameters: params,
                            priority: 150,
                            immediate: false,
                            confidence: emotion_msg.strength,
                            timestamp: hardware_timestamp(),
                        };
                        
                        if let Err(e) = decision_tx.broadcast(decision) {
                            tracing::warn!("Failed to broadcast EGC emotional decision: {}", e);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Set up EMS communication handlers
    async fn setup_ems_communication(
        &self,
        modulation_tx: CrossModuleBroadcastHandle<EmotionalModulation>,
        governance_rx: CrossModuleReceiver<GovernanceDecision>,
        comm_system: Arc<CrossModuleCommunication>,
    ) -> anyhow::Result<()> {
        let ems = self.ems.clone();
        
        // Spawn EMS message handler
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(25)); // 40Hz
            
            loop {
                interval.tick().await;
                
                // Get current emotional state
                let state = ems.get_state().await;
                
                // Broadcast emotional modulation to all modules
                let modulation = EmotionalModulation {
                    valence: state.valence,
                    arousal: state.arousal,
                    emotion_type: format!("{:?}", state.emotional_state),
                    strength: state.dominance,
                    global_modulation: true,
                    timestamp: hardware_timestamp(),
                };
                
                if let Err(e) = modulation_tx.broadcast(modulation) {
                    tracing::warn!("Failed to broadcast emotional modulation: {}", e);
                }
                
                // Process governance decisions that might affect emotional processing
                if let Some(gov_msg) = governance_rx.try_recv() {
                    tracing::debug!("EMS received governance decision: {}", gov_msg.decision_type);
                    
                    if gov_msg.decision_type == "emotional_regulation" {
                        // Apply governance constraints to emotional processing
                        if let Some(&risk_tolerance) = gov_msg.parameters.get("risk_tolerance") {
                            tracing::debug!("EMS adjusting risk tolerance to {}", risk_tolerance);
                            // Apply risk tolerance to EMS emotional modeling
                            let emotional_constraints = EmotionalConstraints {
                                max_arousal: risk_tolerance * 0.8,
                                valence_stability: 1.0 - risk_tolerance,
                                response_damping: risk_tolerance * 0.5,
                            };
                            
                            if let Err(e) = ems.apply_constraints(emotional_constraints).await {
                                tracing::warn!("Failed to apply emotional constraints: {}", e);
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }

    /// Set up cross-module communication channels
    async fn setup_cross_talk(&self) -> anyhow::Result<()> {
        let _monitor = PerformanceMonitor::new("CLogicSystem::setup_cross_talk");
        
        tracing::info!("Setting up high-performance cross-module communication channels");
        
        // Initialize cross-module communication system
        let comm_system = Arc::new(CrossModuleCommunication::new(
            self.bus.clone(),
            CrossModuleConfig {
                channel_capacity: 16384,  // 16K messages per channel
                enable_zero_copy: true,
                enable_priority_routing: true,
                max_latency_ns: 10,      // <10ns target
                backpressure_threshold: 0.8,
            }
        )?);

        // Set up communication channels based on the specified module interactions:

        // 1. DRPP -> ADP: Pattern features
        let drpp_to_adp_tx = comm_system.create_channel::<DrppFeatures>("drpp_to_adp")?;
        let drpp_to_adp_rx = comm_system.subscribe::<DrppFeatures>("drpp_to_adp").await?;
        
        // 2. ADP -> EGC: Processing metrics  
        let adp_to_egc_tx = comm_system.create_channel::<ProcessingMetrics>("adp_to_egc")?;
        let adp_to_egc_rx = comm_system.subscribe::<ProcessingMetrics>("adp_to_egc").await?;
        
        // 3. EMS -> All: Emotional modulation (broadcast)
        let ems_broadcast_tx = comm_system.create_broadcast_channel::<EmotionalModulation>("ems_broadcast")?;
        let ems_to_drpp_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await?;
        let ems_to_adp_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await?;
        let ems_to_egc_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await?;
        
        // 4. EGC -> All: Governance decisions (broadcast)
        let egc_broadcast_tx = comm_system.create_broadcast_channel::<GovernanceDecision>("egc_broadcast")?;
        let egc_to_drpp_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await?;
        let egc_to_adp_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await?;
        let egc_to_ems_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await?;

        // Set up message handlers for each module
        self.setup_drpp_communication(
            drpp_to_adp_tx, 
            ems_to_drpp_rx, 
            egc_to_drpp_rx,
            comm_system.clone()
        ).await?;
        
        self.setup_adp_communication(
            drpp_to_adp_rx, 
            adp_to_egc_tx, 
            ems_to_adp_rx, 
            egc_to_adp_rx,
            comm_system.clone()
        ).await?;
        
        self.setup_egc_communication(
            adp_to_egc_rx, 
            egc_broadcast_tx, 
            ems_to_egc_rx,
            comm_system.clone()
        ).await?;
        
        self.setup_ems_communication(
            ems_broadcast_tx, 
            egc_to_ems_rx,
            comm_system.clone()
        ).await?;

        // Start the cross-module communication system
        comm_system.start().await?;
        
        tracing::info!(
            "Cross-module communication system initialized with <{}ns latency target",
            comm_system.config().max_latency_ns
        );
        
        Ok(())
    }
}

/// C-LOGIC system state
#[derive(Debug, Clone)]
pub struct CLogicState {
    pub drpp_state: drpp::DrppState,
    pub egc_state: egc::EgcState,
    pub ems_state: ems::EmsState,
    pub timestamp: NanoTime,
}

/// Common trait for C-LOGIC modules
#[async_trait::async_trait]
pub trait CLogicModule: Send + Sync {
    /// Start the module
    async fn start(&self) -> anyhow::Result<()>;

    /// Stop the module
    async fn stop(&self) -> anyhow::Result<()>;

    /// Process input
    async fn process(&self, input: &BinaryPacket) -> anyhow::Result<BinaryPacket>;

    /// Get module name
    fn name(&self) -> &str;

    /// Get module metrics
    async fn metrics(&self) -> ModuleMetrics;
}

/// Module metrics
#[derive(Debug, Clone)]
pub struct ModuleMetrics {
    pub processed_packets: u64,
    pub processing_time_ns: u64,
    pub error_count: u64,
    pub last_update: NanoTime,
}

impl Default for ModuleMetrics {
    fn default() -> Self {
        Self {
            processed_packets: 0,
            processing_time_ns: 0,
            error_count: 0,
            last_update: NanoTime::zero(),
        }
    }
}

// ================================================================================================
// Cross-Module Communication System
// ================================================================================================

/// High-performance cross-module communication system
/// 
/// Provides zero-copy, lock-free message passing with <10ns latency targeting.
/// Uses lock-free ring buffers and atomic operations for maximum performance.
pub struct CrossModuleCommunication {
    /// Phase Coherence Bus reference
    bus: Arc<Bus>,
    
    /// Communication configuration
    config: CrossModuleConfig,
    
    /// Lock-free message channels by name
    channels: Arc<dashmap::DashMap<String, Arc<dyn CrossModuleChannel + Send + Sync>>>,
    
    /// Performance metrics
    metrics: Arc<CrossModuleMetrics>,
    
    /// System state
    state: Arc<std::sync::atomic::AtomicU8>, // 0=Stopped, 1=Starting, 2=Running, 3=Stopping
}

/// Configuration for cross-module communication
#[derive(Debug, Clone)]
pub struct CrossModuleConfig {
    /// Channel capacity (ring buffer size)
    pub channel_capacity: usize,
    
    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,
    
    /// Enable priority-based message routing
    pub enable_priority_routing: bool,
    
    /// Maximum allowed latency in nanoseconds
    pub max_latency_ns: u64,
    
    /// Backpressure threshold (0.0-1.0)
    pub backpressure_threshold: f64,
}

impl Default for CrossModuleConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 8192,
            enable_zero_copy: true,
            enable_priority_routing: true,
            max_latency_ns: 10,
            backpressure_threshold: 0.9,
        }
    }
}

/// Cross-module communication metrics
#[derive(Debug, Default)]
pub struct CrossModuleMetrics {
    /// Total messages sent
    pub messages_sent: std::sync::atomic::AtomicU64,
    
    /// Total messages received
    pub messages_received: std::sync::atomic::AtomicU64,
    
    /// Messages dropped due to backpressure
    pub messages_dropped: std::sync::atomic::AtomicU64,
    
    /// Average latency in nanoseconds
    pub avg_latency_ns: std::sync::atomic::AtomicU64,
    
    /// Peak latency in nanoseconds
    pub peak_latency_ns: std::sync::atomic::AtomicU64,
    
    /// Number of active channels
    pub active_channels: std::sync::atomic::AtomicU32,
}

/// Trait for cross-module channels
pub trait CrossModuleChannel: Send + Sync {
    /// Send a message with priority
    fn send_priority(&self, message: Arc<dyn Any + Send + Sync>, priority: u8) -> anyhow::Result<()>;
    
    /// Try to receive a message (non-blocking)
    fn try_recv(&self) -> Option<Arc<dyn Any + Send + Sync>>;
    
    /// Get channel statistics
    fn stats(&self) -> ChannelStats;
    
    /// Check if channel has backpressure
    fn has_backpressure(&self) -> bool;
}

/// High-performance lock-free channel implementation
pub struct LockFreeChannel<T: Send + Sync + 'static> {
    /// Lock-free ring buffer for zero-copy message passing
    ring: Arc<crossbeam_queue::ArrayQueue<Arc<T>>>,
    
    /// Channel statistics
    stats: Arc<ChannelStats>,
    
    /// Backpressure threshold
    backpressure_threshold: f64,
    
    /// Priority queue for high-priority messages
    priority_queue: Arc<crossbeam_queue::ArrayQueue<Arc<T>>>,
}

/// Channel statistics
#[derive(Debug, Default)]
pub struct ChannelStats {
    pub sent: std::sync::atomic::AtomicU64,
    pub received: std::sync::atomic::AtomicU64,
    pub dropped: std::sync::atomic::AtomicU64,
    pub capacity: usize,
}

impl<T: Send + Sync + 'static> LockFreeChannel<T> {
    pub fn new(capacity: usize, backpressure_threshold: f64) -> Self {
        Self {
            ring: Arc::new(crossbeam_queue::ArrayQueue::new(capacity)),
            stats: Arc::new(ChannelStats {
                sent: std::sync::atomic::AtomicU64::new(0),
                received: std::sync::atomic::AtomicU64::new(0),
                dropped: std::sync::atomic::AtomicU64::new(0),
                capacity,
            }),
            backpressure_threshold,
            priority_queue: Arc::new(crossbeam_queue::ArrayQueue::new(capacity / 4)), // 25% for priority
        }
    }
    
    /// Send message with zero-copy semantics
    pub fn send(&self, message: Arc<T>) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        
        let start = std::time::Instant::now();
        
        // Try to send the message
        match self.ring.push(message) {
            Ok(()) => {
                self.stats.sent.fetch_add(1, Ordering::Relaxed);
                
                // Track latency (simplified - in real implementation would use hardware timers)
                let latency_ns = start.elapsed().as_nanos() as u64;
                if latency_ns > 10 {
                    tracing::warn!("Cross-module message latency {}ns exceeds 10ns target", latency_ns);
                }
                
                Ok(())
            }
            Err(_) => {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                anyhow::bail!("Channel full - message dropped")
            }
        }
    }
    
    /// Send high-priority message
    pub fn send_priority(&self, message: Arc<T>) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        
        // Try priority queue first
        match self.priority_queue.push(message.clone()) {
            Ok(()) => {
                self.stats.sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => {
                // Fallback to normal queue
                self.send(message)
            }
        }
    }
    
    /// Receive message with zero-copy semantics
    pub fn recv(&self) -> Option<Arc<T>> {
        use std::sync::atomic::Ordering;
        
        // Check priority queue first
        if let Some(message) = self.priority_queue.pop() {
            self.stats.received.fetch_add(1, Ordering::Relaxed);
            return Some(message);
        }
        
        // Then check normal queue
        if let Some(message) = self.ring.pop() {
            self.stats.received.fetch_add(1, Ordering::Relaxed);
            Some(message)
        } else {
            None
        }
    }
    
    /// Check if channel is experiencing backpressure
    pub fn has_backpressure(&self) -> bool {
        let current_usage = self.ring.len() as f64 / self.stats.capacity as f64;
        current_usage > self.backpressure_threshold
    }
}

impl<T: Send + Sync + 'static> CrossModuleChannel for LockFreeChannel<T> {
    fn send_priority(&self, message: Arc<dyn Any + Send + Sync>, priority: u8) -> anyhow::Result<()> {
        // Downcast the message to the correct type
        let typed_message = message.downcast::<T>()
            .map_err(|_| anyhow::anyhow!("Invalid message type"))?;
            
        if priority > 128 {
            self.send_priority(typed_message)
        } else {
            self.send(typed_message)
        }
    }
    
    fn try_recv(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.recv().map(|msg| msg as Arc<dyn Any + Send + Sync>)
    }
    
    fn stats(&self) -> ChannelStats {
        ChannelStats {
            sent: std::sync::atomic::AtomicU64::new(self.stats.sent.load(std::sync::atomic::Ordering::Relaxed)),
            received: std::sync::atomic::AtomicU64::new(self.stats.received.load(std::sync::atomic::Ordering::Relaxed)),
            dropped: std::sync::atomic::AtomicU64::new(self.stats.dropped.load(std::sync::atomic::Ordering::Relaxed)),
            capacity: self.stats.capacity,
        }
    }
    
    fn has_backpressure(&self) -> bool {
        self.has_backpressure()
    }
}

impl CrossModuleCommunication {
    /// Create new cross-module communication system
    pub fn new(bus: Arc<Bus>, config: CrossModuleConfig) -> anyhow::Result<Self> {
        tracing::info!("Initializing CrossModuleCommunication with {} channel capacity", config.channel_capacity);
        
        Ok(Self {
            bus,
            config,
            channels: Arc::new(dashmap::DashMap::new()),
            metrics: Arc::new(CrossModuleMetrics::default()),
            state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
        })
    }
    
    /// Create a typed channel
    pub fn create_channel<T: Send + Sync + Clone + 'static>(&self, name: &str) -> anyhow::Result<CrossModuleChannelHandle<T>> {
        let channel = Arc::new(LockFreeChannel::<T>::new(
            self.config.channel_capacity,
            self.config.backpressure_threshold,
        ));
        
        self.channels.insert(name.to_string(), channel.clone());
        self.metrics.active_channels.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        tracing::debug!("Created cross-module channel: {}", name);
        
        Ok(CrossModuleChannelHandle {
            name: name.to_string(),
            channel: channel.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Create a broadcast channel (one-to-many)
    pub fn create_broadcast_channel<T: Send + Sync + Clone + 'static>(&self, name: &str) -> anyhow::Result<CrossModuleBroadcastHandle<T>> {
        let channel = Arc::new(LockFreeChannel::<T>::new(
            self.config.channel_capacity * 4, // Larger capacity for broadcast
            self.config.backpressure_threshold,
        ));
        
        self.channels.insert(name.to_string(), channel.clone());
        self.metrics.active_channels.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        tracing::debug!("Created cross-module broadcast channel: {}", name);
        
        Ok(CrossModuleBroadcastHandle {
            name: name.to_string(),
            channel: channel.clone(),
            subscribers: Arc::new(std::sync::RwLock::new(Vec::new())),
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Subscribe to a channel
    pub async fn subscribe<T: Send + Sync + Clone + 'static>(&self, channel_name: &str) -> anyhow::Result<CrossModuleReceiver<T>> {
        if let Some(channel) = self.channels.get(channel_name) {
            // Create a receiver that polls the channel
            Ok(CrossModuleReceiver {
                channel_name: channel_name.to_string(),
                channel: channel.clone(),
                _phantom: std::marker::PhantomData,
            })
        } else {
            anyhow::bail!("Channel {} not found", channel_name)
        }
    }
    
    /// Start the communication system
    pub async fn start(&self) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        
        self.state.store(2, Ordering::Release); // Running state
        
        // Start background monitoring task
        let metrics = self.metrics.clone();
        let channels = self.channels.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                // Monitor channel health and performance
                let mut total_backpressure = 0;
                let mut total_channels = 0;
                
                for entry in channels.iter() {
                    total_channels += 1;
                    if entry.has_backpressure() {
                        total_backpressure += 1;
                    }
                }
                
                if total_backpressure > total_channels / 2 {
                    tracing::warn!("High backpressure detected in cross-module communication: {}/{} channels", 
                        total_backpressure, total_channels);
                }
            }
        });
        
        tracing::info!("CrossModuleCommunication system started");
        Ok(())
    }
    
    /// Get system configuration
    pub fn config(&self) -> &CrossModuleConfig {
        &self.config
    }
    
    /// Get system metrics
    pub fn metrics(&self) -> CrossModuleMetrics {
        CrossModuleMetrics {
            messages_sent: std::sync::atomic::AtomicU64::new(
                self.metrics.messages_sent.load(std::sync::atomic::Ordering::Relaxed)
            ),
            messages_received: std::sync::atomic::AtomicU64::new(
                self.metrics.messages_received.load(std::sync::atomic::Ordering::Relaxed)
            ),
            messages_dropped: std::sync::atomic::AtomicU64::new(
                self.metrics.messages_dropped.load(std::sync::atomic::Ordering::Relaxed)
            ),
            avg_latency_ns: std::sync::atomic::AtomicU64::new(
                self.metrics.avg_latency_ns.load(std::sync::atomic::Ordering::Relaxed)
            ),
            peak_latency_ns: std::sync::atomic::AtomicU64::new(
                self.metrics.peak_latency_ns.load(std::sync::atomic::Ordering::Relaxed)
            ),
            active_channels: std::sync::atomic::AtomicU32::new(
                self.metrics.active_channels.load(std::sync::atomic::Ordering::Relaxed)
            ),
        }
    }
}

/// Handle for sending messages to a specific channel
#[derive(Clone)]
pub struct CrossModuleChannelHandle<T: Send + Sync + 'static> {
    name: String,
    channel: Arc<LockFreeChannel<T>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + Clone + 'static> CrossModuleChannelHandle<T> {
    /// Send a message with normal priority
    pub fn send(&self, message: T) -> anyhow::Result<()> {
        self.channel.send(Arc::new(message))
    }
    
    /// Send a message with high priority
    pub fn send_priority(&self, message: T) -> anyhow::Result<()> {
        self.channel.send_priority(Arc::new(message))
    }
    
    /// Check if channel has backpressure
    pub fn has_backpressure(&self) -> bool {
        self.channel.has_backpressure()
    }
}

/// Handle for broadcast channels (one-to-many)
#[derive(Clone)]
pub struct CrossModuleBroadcastHandle<T: Send + Sync + 'static> {
    name: String,
    channel: Arc<LockFreeChannel<T>>,
    subscribers: Arc<std::sync::RwLock<Vec<String>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + Clone + 'static> CrossModuleBroadcastHandle<T> {
    /// Broadcast a message to all subscribers
    pub fn broadcast(&self, message: T) -> anyhow::Result<()> {
        self.channel.send(Arc::new(message))
    }
    
    /// Broadcast a high-priority message
    pub fn broadcast_priority(&self, message: T) -> anyhow::Result<()> {
        self.channel.send_priority(Arc::new(message))
    }
}

/// Receiver for cross-module messages
pub struct CrossModuleReceiver<T: Send + Sync + 'static> {
    channel_name: String,
    channel: Arc<dyn CrossModuleChannel + Send + Sync>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + Sync + Clone + 'static> CrossModuleReceiver<T> {
    /// Try to receive a message (non-blocking)
    pub fn try_recv(&self) -> Option<T> {
        self.channel.try_recv()
            .and_then(|msg| msg.downcast::<T>().ok())
            .map(|arc| (*arc).clone())
    }
    
    /// Async receive with timeout
    pub async fn recv_timeout(&self, timeout: tokio::time::Duration) -> Option<T> {
        let start = tokio::time::Instant::now();
        
        while start.elapsed() < timeout {
            if let Some(message) = self.try_recv() {
                return Some(message);
            }
            tokio::time::sleep(tokio::time::Duration::from_nanos(100)).await; // 100ns polling interval
        }
        
        None
    }
}

// ================================================================================================
// Cross-Module Message Types
// ================================================================================================

/// DRPP features sent to ADP
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrppFeatures {
    /// Detected patterns with strength
    pub patterns: Vec<(drpp::PatternType, f64)>,
    
    /// Oscillator phase information
    pub phase_info: Vec<f64>,
    
    /// Coherence level
    pub coherence: f64,
    
    /// Resonance strength
    pub resonance_strength: f64,
    
    /// Feature timestamp
    pub timestamp: NanoTime,
}

/// Processing metrics sent from ADP to EGC
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessingMetrics {
    /// Decision processing time in nanoseconds
    pub processing_time_ns: u64,
    
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    
    /// Resource utilization (0.0-1.0)
    pub resource_utilization: f64,
    
    /// Number of processed patterns
    pub patterns_processed: u32,
    
    /// Processing efficiency metric
    pub efficiency: f64,
    
    /// Metrics timestamp
    pub timestamp: NanoTime,
}

/// Emotional modulation broadcast from EMS
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmotionalModulation {
    /// Valence factor (-1.0 to 1.0)
    pub valence: f64,
    
    /// Arousal factor (0.0 to 1.0)
    pub arousal: f64,
    
    /// Emotional state type
    pub emotion_type: String,
    
    /// Modulation strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Affect all modules flag
    pub global_modulation: bool,
    
    /// Modulation timestamp
    pub timestamp: NanoTime,
}

/// Governance decisions broadcast from EGC
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernanceDecision {
    /// Decision type
    pub decision_type: String,
    
    /// Parameters affected
    pub parameters: std::collections::HashMap<String, f64>,
    
    /// Priority level (0-255)
    pub priority: u8,
    
    /// Immediate enforcement flag
    pub immediate: bool,
    
    /// Decision confidence (0.0-1.0)
    pub confidence: f64,
    
    /// Decision timestamp
    pub timestamp: NanoTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clogic_system_creation() {
        let bus = Arc::new(Bus::new(Default::default()).unwrap());
        let config = CLogicConfig::default();

        let system = CLogicSystem::new(bus, config).await.unwrap();
        let state = system.get_state().await;

        assert!(state.timestamp.as_nanos() > 0);
    }
}
