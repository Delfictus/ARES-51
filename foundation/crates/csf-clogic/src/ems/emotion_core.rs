//! Core emotion processing for EMS

use super::{BaseEmotion, EmotionType};

/// Core emotion processing engine
pub struct EmotionCore {
    /// Base emotional tendency
    base_emotion: BaseEmotion,

    /// Emotion wheel model
    emotion_wheel: PlutchikWheel,

    /// Sensitivity factor
    sensitivity: f64,
}

/// Emotional state representation
#[derive(Debug, Clone)]
pub struct EmotionalState {
    /// Dominant emotion
    pub dominant_emotion: EmotionType,

    /// Emotion intensities
    pub emotion_intensities: [f64; 8],

    /// Overall intensity
    pub intensity: f64,

    /// Emotional complexity
    pub complexity: f64,
}

/// Plutchik's wheel of emotions
struct PlutchikWheel {
    /// Primary emotions arranged in opposing pairs
    emotions: [(EmotionType, EmotionType); 4],
}

impl EmotionCore {
    /// Create a new emotion core
    pub fn new(config: &super::EmsConfig) -> Self {
        Self {
            base_emotion: config.base_emotion,
            emotion_wheel: PlutchikWheel {
                emotions: [
                    (EmotionType::Joy, EmotionType::Sadness),
                    (EmotionType::Trust, EmotionType::Disgust),
                    (EmotionType::Fear, EmotionType::Anger),
                    (EmotionType::Surprise, EmotionType::Anticipation),
                ],
            },
            sensitivity: config.sensitivity,
        }
    }

    /// Get initial emotional state
    pub fn get_initial_state(&self) -> EmotionalState {
        let mut emotion_intensities = [0.0; 8];

        // Set base emotion
        match self.base_emotion {
            BaseEmotion::Neutral => {
                // All emotions at low baseline
                emotion_intensities.fill(0.1);
            }
            BaseEmotion::Positive => {
                emotion_intensities[EmotionType::Joy as usize] = 0.3;
                emotion_intensities[EmotionType::Trust as usize] = 0.2;
            }
            BaseEmotion::Negative => {
                emotion_intensities[EmotionType::Sadness as usize] = 0.3;
                emotion_intensities[EmotionType::Fear as usize] = 0.2;
            }
            BaseEmotion::Alert => {
                emotion_intensities[EmotionType::Fear as usize] = 0.2;
                emotion_intensities[EmotionType::Surprise as usize] = 0.3;
                emotion_intensities[EmotionType::Anticipation as usize] = 0.3;
            }
            BaseEmotion::Calm => {
                emotion_intensities[EmotionType::Trust as usize] = 0.3;
                emotion_intensities[EmotionType::Joy as usize] = 0.2;
            }
        }

        EmotionalState {
            dominant_emotion: self.find_dominant_emotion(&emotion_intensities),
            emotion_intensities,
            intensity: 0.2,
            complexity: 0.1,
        }
    }

    /// Update emotional state based on valence and arousal
    pub fn update_state(
        &self,
        valence: f64,
        arousal: f64,
        affect_response: &super::affect_processor::AffectResponse,
    ) -> EmotionalState {
        let mut emotion_intensities = [0.0; 8];

        // Map valence-arousal to emotions using Russell's circumplex model
        if valence > 0.0 && arousal > 0.0 {
            // High valence, high arousal: excited, elated
            emotion_intensities[EmotionType::Joy as usize] = valence * arousal;
            emotion_intensities[EmotionType::Anticipation as usize] = arousal * 0.5;
        } else if valence > 0.0 && arousal <= 0.0 {
            // High valence, low arousal: content, serene
            emotion_intensities[EmotionType::Trust as usize] = valence * (1.0 - arousal.abs());
            emotion_intensities[EmotionType::Joy as usize] = valence * 0.5;
        } else if valence <= 0.0 && arousal > 0.0 {
            // Low valence, high arousal: tense, nervous
            emotion_intensities[EmotionType::Fear as usize] = arousal * valence.abs();
            emotion_intensities[EmotionType::Anger as usize] = arousal * 0.5;
        } else {
            // Low valence, low arousal: sad, depressed
            emotion_intensities[EmotionType::Sadness as usize] =
                valence.abs() * (1.0 - arousal.abs());
            emotion_intensities[EmotionType::Disgust as usize] = valence.abs() * 0.3;
        }

        // Add surprise based on novelty
        emotion_intensities[EmotionType::Surprise as usize] = affect_response.novelty * 0.5;

        // Apply sensitivity
        for intensity in &mut emotion_intensities {
            *intensity *= self.sensitivity;
        }

        // Add triggered emotions from affect response
        for emotion in &affect_response.triggered_emotions {
            if let Ok(idx) = TryInto::<usize>::try_into(emotion.emotion_type as u8) {
                if idx < emotion_intensities.len() {
                    emotion_intensities[idx] += emotion.intensity * 0.3;
                }
            }
        }

        // Normalize intensities
        let total_intensity: f64 = emotion_intensities.iter().sum();
        if total_intensity > 1.0 {
            for intensity in &mut emotion_intensities {
                *intensity /= total_intensity;
            }
        }

        // Calculate complexity (entropy of emotion distribution)
        let complexity = self.calculate_emotional_complexity(&emotion_intensities);

        EmotionalState {
            dominant_emotion: self.find_dominant_emotion(&emotion_intensities),
            emotion_intensities,
            intensity: total_intensity.min(1.0),
            complexity,
        }
    }

    /// Apply decay to emotional state
    pub fn apply_decay(&self, state: &EmotionalState, decay_rate: f64) -> EmotionalState {
        let mut new_intensities = state.emotion_intensities;

        for intensity in &mut new_intensities {
            *intensity *= 1.0 - decay_rate;
            if *intensity < 0.01 {
                *intensity = 0.0;
            }
        }

        EmotionalState {
            dominant_emotion: self.find_dominant_emotion(&new_intensities),
            emotion_intensities: new_intensities,
            intensity: state.intensity * (1.0 - decay_rate),
            complexity: self.calculate_emotional_complexity(&new_intensities),
        }
    }

    /// Find dominant emotion from intensities
    fn find_dominant_emotion(&self, intensities: &[f64; 8]) -> EmotionType {
        let (idx, _) = intensities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        match idx {
            0 => EmotionType::Joy,
            1 => EmotionType::Trust,
            2 => EmotionType::Fear,
            3 => EmotionType::Surprise,
            4 => EmotionType::Sadness,
            5 => EmotionType::Disgust,
            6 => EmotionType::Anger,
            7 => EmotionType::Anticipation,
            _ => EmotionType::Joy,
        }
    }

    /// Calculate emotional complexity (entropy)
    fn calculate_emotional_complexity(&self, intensities: &[f64; 8]) -> f64 {
        let total: f64 = intensities.iter().sum();
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &intensity in intensities {
            if intensity > 0.0 {
                let p = intensity / total;
                entropy -= p * p.log2();
            }
        }

        // Normalize to 0-1 range (max entropy for 8 emotions is log2(8) = 3)
        entropy / 3.0
    }
}

impl EmotionalState {
    /// Get processing bias based on emotional state
    pub fn processing_bias(&self) -> f64 {
        // Positive emotions increase processing speed
        let positive_bias = self.emotion_intensities[EmotionType::Joy as usize]
            + self.emotion_intensities[EmotionType::Trust as usize] * 0.5
            + self.emotion_intensities[EmotionType::Anticipation as usize] * 0.3;

        // Negative emotions decrease processing speed
        let negative_bias = self.emotion_intensities[EmotionType::Fear as usize] * 0.5
            + self.emotion_intensities[EmotionType::Sadness as usize] * 0.7
            + self.emotion_intensities[EmotionType::Disgust as usize] * 0.3;

        positive_bias - negative_bias
    }

    /// Get attention focus based on emotional state
    pub fn attention_focus(&self) -> f64 {
        // Fear and surprise increase attention
        self.emotion_intensities[EmotionType::Fear as usize] * 0.8
            + self.emotion_intensities[EmotionType::Surprise as usize] * 0.6
            + self.emotion_intensities[EmotionType::Anger as usize] * 0.4
    }

    /// Get decision weight modifier
    pub fn decision_weight(&self) -> f64 {
        // Trust and joy increase confidence in decisions
        let confidence = self.emotion_intensities[EmotionType::Trust as usize] * 0.7
            + self.emotion_intensities[EmotionType::Joy as usize] * 0.3;

        // Fear and sadness decrease confidence
        let doubt = self.emotion_intensities[EmotionType::Fear as usize] * 0.5
            + self.emotion_intensities[EmotionType::Sadness as usize] * 0.3;

        (confidence - doubt + 1.0) / 2.0 // Normalize to 0-1
    }

    /// Get energy level based on emotional state
    pub fn energy_level(&self) -> f64 {
        // High arousal emotions increase energy
        self.emotion_intensities[EmotionType::Joy as usize] * 0.6
            + self.emotion_intensities[EmotionType::Anger as usize] * 0.7
            + self.emotion_intensities[EmotionType::Anticipation as usize] * 0.5
            + self.emotion_intensities[EmotionType::Surprise as usize] * 0.4
    }
}
