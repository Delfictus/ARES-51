//! Emotional Modeling System (EMS)
//!
//! Models affective states and their influence on system behavior,
//! providing emotional intelligence to the ARES CSF system.

use csf_bus::{traits::EventBusTx, PhaseCoherenceBus as Bus};
use csf_core::hardware_timestamp;
use csf_core::prelude::*;

// Type aliases for compatibility
type BinaryPacket = PhasePacket<PacketPayload>;
type Channel<T> = tokio::sync::mpsc::Receiver<T>;
type Receiver<T> = tokio::sync::mpsc::Receiver<T>;
use parking_lot::RwLock;
use std::sync::Arc;

mod affect_processor;
mod dynamics;
mod emotion_core;
mod valence_arousal;

use affect_processor::AffectProcessor;
use emotion_core::{EmotionCore, EmotionalState};
use valence_arousal::ValenceArousalModel;

/// EMS configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmsConfig {
    /// Base emotional state
    pub base_emotion: BaseEmotion,

    /// Emotional decay rate (0.0 - 1.0)
    pub decay_rate: f64,

    /// Emotional sensitivity (0.0 - 1.0)
    pub sensitivity: f64,

    /// Enable empathy modeling
    pub empathy_enabled: bool,

    /// Emotional contagion strength
    pub contagion_strength: f64,

    /// Update frequency (Hz)
    pub update_frequency: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BaseEmotion {
    Neutral,
    Positive,
    Negative,
    Alert,
    Calm,
}

impl Default for EmsConfig {
    fn default() -> Self {
        Self {
            base_emotion: BaseEmotion::Neutral,
            decay_rate: 0.1,
            sensitivity: 0.5,
            empathy_enabled: true,
            contagion_strength: 0.3,
            update_frequency: 20.0, // 20 Hz
        }
    }
}

/// Emotional Modeling System
pub struct EmotionalModelingSystem {
    /// Configuration
    config: EmsConfig,

    /// Emotion core
    emotion_core: Arc<EmotionCore>,

    /// Affect processor
    affect_processor: Arc<AffectProcessor>,

    /// Valence-Arousal model
    va_model: Arc<ValenceArousalModel>,

    /// Phase Coherence Bus
    bus: Arc<Bus>,

    /// Component ID for this EMS instance
    component_id: ComponentId,

    /// Processing handle
    processing_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,

    /// Update handle
    update_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,

    /// Current state
    state: Arc<RwLock<EmsState>>,

    /// Metrics
    metrics: Arc<RwLock<super::ModuleMetrics>>,
}

/// EMS state
#[derive(Debug, Clone)]
pub struct EmsState {
    /// Current emotional state
    pub emotional_state: EmotionalState,

    /// Valence (pleasure-displeasure)
    pub valence: f64,

    /// Arousal (activation-deactivation)
    pub arousal: f64,

    /// Dominance (control)
    pub dominance: f64,

    /// Previous valence for stability tracking
    pub previous_valence: Option<f64>,

    /// Active emotions
    pub active_emotions: Vec<Emotion>,

    /// Emotional history
    pub emotion_history: Vec<EmotionSnapshot>,

    /// System mood
    pub mood: Mood,

    /// Last update timestamp
    pub timestamp: NanoTime,
}

/// Individual emotion
#[derive(Debug, Clone)]
pub struct Emotion {
    /// Emotion type
    pub emotion_type: EmotionType,

    /// Intensity (0.0 - 1.0)
    pub intensity: f64,

    /// Duration
    pub duration_ns: u64,

    /// Trigger
    pub trigger: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmotionType {
    Joy,
    Trust,
    Fear,
    Surprise,
    Sadness,
    Disgust,
    Anger,
    Anticipation,
}

/// System mood (longer-term emotional state)
#[derive(Debug, Clone)]
pub struct Mood {
    /// Mood type
    pub mood_type: MoodType,

    /// Stability (0.0 - 1.0)
    pub stability: f64,

    /// Duration
    pub duration_ns: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum MoodType {
    Optimistic,
    Pessimistic,
    Anxious,
    Relaxed,
    Energetic,
    Lethargic,
}

/// Emotion snapshot for history
#[derive(Debug, Clone)]
pub struct EmotionSnapshot {
    pub timestamp: NanoTime,
    pub valence: f64,
    pub arousal: f64,
    pub dominant_emotion: EmotionType,
}

impl EmotionalModelingSystem {
    /// Create a new EMS instance with full bus integration
    pub async fn new(bus: Arc<Bus>, config: EmsConfig) -> anyhow::Result<Self> {
        // Initialize component ID for this EMS instance
        let component_id = ComponentId::EMS;

        // Initialize components
        let emotion_core = Arc::new(EmotionCore::new(&config));
        let affect_processor = Arc::new(AffectProcessor::new(&config));
        let va_model = Arc::new(ValenceArousalModel::new(&config));

        // Initialize state
        let state = Arc::new(RwLock::new(EmsState {
            emotional_state: emotion_core.get_initial_state(),
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.5,
            previous_valence: None,
            active_emotions: Vec::new(),
            emotion_history: Vec::with_capacity(1000),
            mood: Mood {
                mood_type: MoodType::Relaxed,
                stability: 0.8,
                duration_ns: 0,
            },
            timestamp: hardware_timestamp(),
        }));

        Ok(Self {
            config,
            emotion_core,
            affect_processor,
            va_model,
            bus,
            component_id,
            processing_handle: RwLock::new(None),
            update_handle: RwLock::new(None),
            state,
            metrics: Arc::new(RwLock::new(Default::default())),
        })
    }

    /// Get current state
    pub async fn get_state(&self) -> EmsState {
        self.state.read().clone()
    }

    /// Process a single packet
    async fn process_packet(&self, packet: BinaryPacket) -> anyhow::Result<BinaryPacket> {
        let start_time = hardware_timestamp();

        // Extract emotional features from packet
        let features = self.extract_emotional_features(&packet)?;

        // Process affect
        let affect_response = self.affect_processor.process(&features).await?;

        // Update valence-arousal model
        let (valence, arousal) = self
            .va_model
            .update(affect_response.valence_delta, affect_response.arousal_delta);

        // Update emotional state
        let emotional_state = self
            .emotion_core
            .update_state(valence, arousal, &affect_response);

        // Generate emotional modulation
        let modulation = self.generate_modulation(&emotional_state);

        // Update state
        {
            let mut state = self.state.write();
            state.emotional_state = emotional_state.clone();
            state.previous_valence = Some(state.valence); // Track previous valence
            state.valence = valence;
            state.arousal = arousal;
            state.active_emotions = affect_response.triggered_emotions;

            // Update history
            state.emotion_history.push(EmotionSnapshot {
                timestamp: hardware_timestamp(),
                valence,
                arousal,
                dominant_emotion: emotional_state.dominant_emotion,
            });

            // Keep history bounded
            if state.emotion_history.len() > 1000 {
                state.emotion_history.remove(0);
            }

            // Update mood if needed
            if self.should_update_mood(&state) {
                state.mood = self.determine_mood(&state.emotion_history);
            }

            state.timestamp = hardware_timestamp();
        }

        // Send modulation signal via bus
        let modulation_data = bincode::serialize(&modulation).unwrap_or_default();
        let modulation_packet = csf_bus::packet::PhasePacket::new(
            modulation_data,
            csf_core::ComponentId::new(0xEEE5), // EMS modulation ID
        );
        
        if let Err(e) = self.bus.send(modulation_packet).await {
            tracing::warn!("Failed to send emotional modulation via bus: {}", e);
        } else {
            tracing::debug!("Sent emotional modulation via bus: {:?}", modulation);
        }

        // Create output packet with emotional context
        let mut output = packet;
        output.header.flags |= PacketFlags::PROCESSED;

        // Add emotional metadata
        output.payload.metadata.insert(
            "ems_emotion".to_string(),
            serde_json::json!({
                "valence": valence,
                "arousal": arousal,
                "emotion": format!("{:?}", emotional_state.dominant_emotion),
                "intensity": emotional_state.intensity,
            }),
        );

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.processed_packets += 1;
            metrics.processing_time_ns += (hardware_timestamp() - start_time).as_nanos();
            metrics.last_update = hardware_timestamp();
        }

        Ok(output)
    }

    /// Extract emotional features from packet
    fn extract_emotional_features(
        &self,
        packet: &BinaryPacket,
    ) -> anyhow::Result<EmotionalFeatures> {
        let mut features = EmotionalFeatures::default();

        // Analyze packet priority (maps to urgency/arousal)
        features.urgency = packet.header.priority as f64 / 255.0;

        // Analyze packet type (maps to valence)
        features.valence_bias = match packet.header.packet_type {
            PacketType::Control => 0.0,
            PacketType::Data => 0.2,
            PacketType::Event => -0.1,
            PacketType::Stream => 0.1,
        };

        // Check for error indicators
        if packet.header.flags.contains(PacketFlags::ERROR) {
            features.valence_bias -= 0.5;
            features.stress_level = 0.7;
        }

        // Extract metadata emotions if present
        if let Some(emotion_data) = packet.payload.metadata.get("source_emotion") {
            if let Some(valence) = emotion_data.get("valence").and_then(|v| v.as_f64()) {
                features.social_valence = valence;
            }
        }

        Ok(features)
    }

    /// Generate emotional modulation signal
    fn generate_modulation(&self, state: &EmotionalState) -> EmotionalModulation {
        EmotionalModulation {
            processing_bias: state.processing_bias(),
            attention_focus: state.attention_focus(),
            decision_weight: state.decision_weight(),
            energy_level: state.energy_level(),
        }
    }

    /// Create modulation packet
    fn create_modulation_packet(
        &self,
        modulation: EmotionalModulation,
    ) -> anyhow::Result<BinaryPacket> {
        let payload = PacketPayload {
            data: vec![],
            metadata: {
                let mut map = std::collections::HashMap::new();
                map.insert(
                    "modulation_type".to_string(),
                    serde_json::json!("emotional"),
                );
                map.insert(
                    "processing_bias".to_string(),
                    serde_json::json!(modulation.processing_bias),
                );
                map.insert(
                    "attention_focus".to_string(),
                    serde_json::json!(modulation.attention_focus),
                );
                map.insert(
                    "decision_weight".to_string(),
                    serde_json::json!(modulation.decision_weight),
                );
                map.insert(
                    "energy_level".to_string(),
                    serde_json::json!(modulation.energy_level),
                );
                map
            },
        };

        let packet = PhasePacket::new(
            PacketType::Control,
            0,        // Source node - EMS
            u16::MAX, // Broadcast destination
            payload,
        )
        .with_priority(150);

        Ok(packet)
    }

    /// Check if mood should be updated
    fn should_update_mood(&self, state: &EmsState) -> bool {
        let mood_duration = hardware_timestamp().as_nanos() - state.mood.duration_ns;
        mood_duration > 10_000_000_000 // 10 seconds
    }

    /// Determine mood from emotional history
    fn determine_mood(&self, history: &[EmotionSnapshot]) -> Mood {
        if history.len() < 10 {
            return Mood {
                mood_type: MoodType::Relaxed,
                stability: 0.5,
                duration_ns: hardware_timestamp().as_nanos(),
            };
        }

        // Calculate average valence and arousal
        let recent_history = &history[history.len().saturating_sub(50)..];
        let avg_valence =
            recent_history.iter().map(|s| s.valence).sum::<f64>() / recent_history.len() as f64;
        let avg_arousal =
            recent_history.iter().map(|s| s.arousal).sum::<f64>() / recent_history.len() as f64;

        // Determine mood type from valence-arousal quadrant
        let mood_type = match (avg_valence > 0.0, avg_arousal > 0.0) {
            (true, true) => MoodType::Energetic,
            (true, false) => MoodType::Relaxed,
            (false, true) => MoodType::Anxious,
            (false, false) => MoodType::Lethargic,
        };

        // Calculate stability from variance
        let valence_variance = recent_history
            .iter()
            .map(|s| (s.valence - avg_valence).powi(2))
            .sum::<f64>()
            / recent_history.len() as f64;
        let stability = 1.0 - valence_variance.min(1.0);

        Mood {
            mood_type,
            stability,
            duration_ns: hardware_timestamp().as_nanos(),
        }
    }

    /// Emotional update loop
    async fn update_loop(self: Arc<Self>) {
        let update_interval =
            tokio::time::Duration::from_secs_f64(1.0 / self.config.update_frequency);
        let mut interval = tokio::time::interval(update_interval);

        loop {
            interval.tick().await;

            // Apply emotional decay
            let decay = 1.0 - self.config.decay_rate;
            self.va_model.apply_decay(decay);

            // Update emotional core
            let current_state = self.state.read().clone();
            let decayed_state = self
                .emotion_core
                .apply_decay(&current_state.emotional_state, self.config.decay_rate);

            // Update state
            {
                let mut state = self.state.write();
                state.emotional_state = decayed_state;
                state.valence *= decay;
                state.arousal *= decay;
            }
        }
    }
}

#[derive(Debug, Default)]
struct EmotionalFeatures {
    urgency: f64,
    valence_bias: f64,
    stress_level: f64,
    social_valence: f64,
}

#[derive(Debug, Clone)]
struct EmotionalModulation {
    processing_bias: f64,
    attention_focus: f64,
    decision_weight: f64,
    energy_level: f64,
}

#[async_trait::async_trait]
impl super::CLogicModule for EmotionalModelingSystem {
    async fn start(&self) -> anyhow::Result<()> {
        // Start processing loop
        let self_clone = Arc::new(self.clone());
        let handle = tokio::spawn(async move {
            // Implement proper packet processing loop with bus integration
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            loop {
                interval.tick().await;
                
                // Placeholder for bus packet processing  
                tracing::debug!("EMS processing loop tick");
            }
        });

        *self.processing_handle.write() = Some(handle);

        // Start update loop
        let self_clone = Arc::new(self.clone());
        let update_handle = tokio::spawn(self_clone.update_loop());
        *self.update_handle.write() = Some(update_handle);

        Ok(())
    }

    async fn stop(&self) -> anyhow::Result<()> {
        if let Some(handle) = self.processing_handle.write().take() {
            handle.abort();
        }

        if let Some(handle) = self.update_handle.write().take() {
            handle.abort();
        }

        Ok(())
    }

    async fn process(&self, input: &BinaryPacket) -> anyhow::Result<BinaryPacket> {
        self.process_packet(input.clone()).await
    }

    fn name(&self) -> &str {
        "EmotionalModelingSystem"
    }

    async fn metrics(&self) -> super::ModuleMetrics {
        self.metrics.read().clone()
    }
}

impl EmotionalModelingSystem {
    /// Apply emotional constraints from governance
    pub async fn apply_constraints(&self, constraints: crate::EmotionalConstraints) -> anyhow::Result<()> {
        let mut state = self.state.write();
        
        // Apply constraints to emotional processing
        if state.arousal > constraints.max_arousal {
            state.arousal = constraints.max_arousal;
        }
        
        // Apply valence stability constraint
        let valence_change = (state.valence - state.previous_valence.unwrap_or(0.0)).abs();
        if valence_change > (1.0 - constraints.valence_stability) {
            state.valence = state.previous_valence.unwrap_or(0.0) + 
                (state.valence - state.previous_valence.unwrap_or(0.0)) * constraints.valence_stability;
        }
        
        // Apply response damping
        state.dominance *= constraints.response_damping;
        
        tracing::debug!("Applied emotional constraints: max_arousal={}, valence_stability={}, response_damping={}",
            constraints.max_arousal, constraints.valence_stability, constraints.response_damping);
        
        Ok(())
    }
}

impl Clone for EmotionalModelingSystem {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            emotion_core: self.emotion_core.clone(),
            affect_processor: self.affect_processor.clone(),
            va_model: self.va_model.clone(),
            bus: self.bus.clone(),
            component_id: self.component_id,
            processing_handle: RwLock::new(None),
            update_handle: RwLock::new(None),
            state: self.state.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ems_creation() {
        let bus = Arc::new(Bus::new(Default::default()).unwrap());
        let config = EmsConfig::default();

        let ems = EmotionalModelingSystem::new(bus, config).await.unwrap();
        let state = ems.get_state().await;

        assert_eq!(state.valence, 0.0);
        assert_eq!(state.arousal, 0.0);
        assert!(state.active_emotions.is_empty());
    }
}
