//! Affect processing for emotional responses

use super::{Emotion, EmotionType};

/// Affect processor for generating emotional responses
pub struct AffectProcessor {
    /// Empathy enabled
    empathy_enabled: bool,

    /// Emotional contagion strength
    contagion_strength: f64,

    /// Appraisal patterns
    appraisal_patterns: Vec<AppraisalPattern>,
}

/// Affect response from processing
#[derive(Debug, Clone)]
pub struct AffectResponse {
    /// Valence change
    pub valence_delta: f64,

    /// Arousal change
    pub arousal_delta: f64,

    /// Triggered emotions
    pub triggered_emotions: Vec<Emotion>,

    /// Novelty factor
    pub novelty: f64,

    /// Social influence
    pub social_influence: f64,
}

/// Appraisal pattern for emotion generation
#[derive(Debug, Clone)]
struct AppraisalPattern {
    /// Pattern name
    name: String,

    /// Trigger conditions
    triggers: Vec<TriggerCondition>,

    /// Resulting emotion
    emotion: EmotionType,

    /// Base intensity
    intensity: f64,
}

#[derive(Debug, Clone)]
struct TriggerCondition {
    feature: String,
    threshold: f64,
    comparison: Comparison,
}

#[derive(Debug, Clone, Copy)]
enum Comparison {
    GreaterThan,
    LessThan,
    Equal,
}

impl AffectProcessor {
    /// Create a new affect processor
    pub fn new(config: &super::EmsConfig) -> Self {
        Self {
            empathy_enabled: config.empathy_enabled,
            contagion_strength: config.contagion_strength,
            appraisal_patterns: Self::create_default_patterns(),
        }
    }

    /// Process emotional features to generate affect response
    pub async fn process(
        &self,
        features: &super::EmotionalFeatures,
    ) -> anyhow::Result<AffectResponse> {
        let mut response = AffectResponse {
            valence_delta: 0.0,
            arousal_delta: 0.0,
            triggered_emotions: Vec::new(),
            novelty: 0.0,
            social_influence: 0.0,
        };

        // Basic valence/arousal mapping
        response.valence_delta = features.valence_bias - features.stress_level * 0.5;
        response.arousal_delta = features.urgency + features.stress_level * 0.3;

        // Apply appraisal patterns
        for pattern in &self.appraisal_patterns {
            if self.evaluate_pattern(pattern, features) {
                response.triggered_emotions.push(Emotion {
                    emotion_type: pattern.emotion,
                    intensity: pattern.intensity * (1.0 + features.urgency * 0.5),
                    duration_ns: 1_000_000_000, // 1 second base duration
                    trigger: pattern.name.clone(),
                });
            }
        }

        // Apply emotional contagion if enabled
        if self.empathy_enabled && features.social_valence.abs() > 0.1 {
            response.social_influence = features.social_valence * self.contagion_strength;
            response.valence_delta += response.social_influence;

            // Mirror emotions from social context
            if features.social_valence > 0.5 {
                response.triggered_emotions.push(Emotion {
                    emotion_type: EmotionType::Joy,
                    intensity: features.social_valence * self.contagion_strength,
                    duration_ns: 500_000_000, // 0.5 second
                    trigger: "emotional_contagion".to_string(),
                });
            } else if features.social_valence < -0.5 {
                response.triggered_emotions.push(Emotion {
                    emotion_type: EmotionType::Sadness,
                    intensity: features.social_valence.abs() * self.contagion_strength,
                    duration_ns: 500_000_000,
                    trigger: "emotional_contagion".to_string(),
                });
            }
        }

        // Calculate novelty (simplified - in real system would track history)
        response.novelty = (features.urgency - 0.5).abs() * 0.5;

        // Clamp values
        response.valence_delta = response.valence_delta.clamp(-1.0, 1.0);
        response.arousal_delta = response.arousal_delta.clamp(-1.0, 1.0);

        Ok(response)
    }

    /// Create default appraisal patterns
    fn create_default_patterns() -> Vec<AppraisalPattern> {
        vec![
            // High stress triggers fear
            AppraisalPattern {
                name: "high_stress".to_string(),
                triggers: vec![TriggerCondition {
                    feature: "stress_level".to_string(),
                    threshold: 0.7,
                    comparison: Comparison::GreaterThan,
                }],
                emotion: EmotionType::Fear,
                intensity: 0.6,
            },
            // Very high urgency triggers anticipation
            AppraisalPattern {
                name: "high_urgency".to_string(),
                triggers: vec![TriggerCondition {
                    feature: "urgency".to_string(),
                    threshold: 0.8,
                    comparison: Comparison::GreaterThan,
                }],
                emotion: EmotionType::Anticipation,
                intensity: 0.7,
            },
            // Positive valence with low stress triggers trust
            AppraisalPattern {
                name: "positive_calm".to_string(),
                triggers: vec![
                    TriggerCondition {
                        feature: "valence_bias".to_string(),
                        threshold: 0.3,
                        comparison: Comparison::GreaterThan,
                    },
                    TriggerCondition {
                        feature: "stress_level".to_string(),
                        threshold: 0.3,
                        comparison: Comparison::LessThan,
                    },
                ],
                emotion: EmotionType::Trust,
                intensity: 0.5,
            },
            // Negative valence with high stress triggers anger
            AppraisalPattern {
                name: "frustrated".to_string(),
                triggers: vec![
                    TriggerCondition {
                        feature: "valence_bias".to_string(),
                        threshold: -0.3,
                        comparison: Comparison::LessThan,
                    },
                    TriggerCondition {
                        feature: "stress_level".to_string(),
                        threshold: 0.5,
                        comparison: Comparison::GreaterThan,
                    },
                ],
                emotion: EmotionType::Anger,
                intensity: 0.6,
            },
        ]
    }

    /// Evaluate if a pattern matches current features
    fn evaluate_pattern(
        &self,
        pattern: &AppraisalPattern,
        features: &super::EmotionalFeatures,
    ) -> bool {
        pattern.triggers.iter().all(|trigger| {
            let value = match trigger.feature.as_str() {
                "urgency" => features.urgency,
                "valence_bias" => features.valence_bias,
                "stress_level" => features.stress_level,
                "social_valence" => features.social_valence,
                _ => 0.0,
            };

            match trigger.comparison {
                Comparison::GreaterThan => value > trigger.threshold,
                Comparison::LessThan => value < trigger.threshold,
                Comparison::Equal => (value - trigger.threshold).abs() < 0.01,
            }
        })
    }
}
