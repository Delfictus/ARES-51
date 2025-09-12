//! Valence-Arousal model for emotional state representation

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Valence-Arousal model based on Russell's circumplex model of affect
pub struct ValenceArousalModel {
    /// Current valence (-1.0 to 1.0)
    valence: RwLock<f64>,

    /// Current arousal (-1.0 to 1.0)
    arousal: RwLock<f64>,

    /// Momentum for smooth transitions
    valence_momentum: AtomicU64,
    arousal_momentum: AtomicU64,

    /// Model parameters
    inertia: f64,
    damping: f64,
}

impl ValenceArousalModel {
    /// Create a new V-A model
    pub fn new(config: &super::EmsConfig) -> Self {
        Self {
            valence: RwLock::new(0.0),
            arousal: RwLock::new(0.0),
            valence_momentum: AtomicU64::new(0.0f64.to_bits()),
            arousal_momentum: AtomicU64::new(0.0f64.to_bits()),
            inertia: 0.7, // How much past state influences current
            damping: 0.1, // How quickly momentum decays
        }
    }

    /// Update valence and arousal with deltas
    pub fn update(&self, valence_delta: f64, arousal_delta: f64) -> (f64, f64) {
        // Update valence with momentum
        let new_valence = {
            let mut valence = self.valence.write();
            let momentum = f64::from_bits(self.valence_momentum.load(Ordering::Relaxed));

            // Apply inertia to delta
            let effective_delta = valence_delta * (1.0 - self.inertia) + momentum * self.inertia;

            // Update value
            *valence += effective_delta;
            *valence = valence.clamp(-1.0, 1.0);

            // Update momentum
            let new_momentum = effective_delta * (1.0 - self.damping);
            self.valence_momentum
                .store(new_momentum.to_bits(), Ordering::Relaxed);

            *valence
        };

        // Update arousal with momentum
        let new_arousal = {
            let mut arousal = self.arousal.write();
            let momentum = f64::from_bits(self.arousal_momentum.load(Ordering::Relaxed));

            // Apply inertia to delta
            let effective_delta = arousal_delta * (1.0 - self.inertia) + momentum * self.inertia;

            // Update value
            *arousal += effective_delta;
            *arousal = arousal.clamp(-1.0, 1.0);

            // Update momentum
            let new_momentum = effective_delta * (1.0 - self.damping);
            self.arousal_momentum
                .store(new_momentum.to_bits(), Ordering::Relaxed);

            *arousal
        };

        (new_valence, new_arousal)
    }

    /// Apply decay to both dimensions
    pub fn apply_decay(&self, decay_factor: f64) {
        {
            let mut valence = self.valence.write();
            *valence *= decay_factor;
            if valence.abs() < 0.01 {
                *valence = 0.0;
            }
        }

        {
            let mut arousal = self.arousal.write();
            *arousal *= decay_factor;
            if arousal.abs() < 0.01 {
                *arousal = 0.0;
            }
        }

        // Decay momentum as well
        let valence_momentum = f64::from_bits(self.valence_momentum.load(Ordering::Relaxed));
        self.valence_momentum.store(
            (valence_momentum * decay_factor).to_bits(),
            Ordering::Relaxed,
        );

        let arousal_momentum = f64::from_bits(self.arousal_momentum.load(Ordering::Relaxed));
        self.arousal_momentum.store(
            (arousal_momentum * decay_factor).to_bits(),
            Ordering::Relaxed,
        );
    }

    /// Get current valence and arousal
    pub fn get_state(&self) -> (f64, f64) {
        (*self.valence.read(), *self.arousal.read())
    }

    /// Get the quadrant in the V-A space
    pub fn get_quadrant(&self) -> VAQuadrant {
        let (valence, arousal) = self.get_state();

        match (valence >= 0.0, arousal >= 0.0) {
            (true, true) => VAQuadrant::HighValenceHighArousal,
            (true, false) => VAQuadrant::HighValenceLowArousal,
            (false, true) => VAQuadrant::LowValenceHighArousal,
            (false, false) => VAQuadrant::LowValenceLowArousal,
        }
    }

    /// Get the distance from neutral (origin)
    pub fn get_intensity(&self) -> f64 {
        let (valence, arousal) = self.get_state();
        (valence.powi(2) + arousal.powi(2)).sqrt()
    }

    /// Get the angle in V-A space (in radians)
    pub fn get_angle(&self) -> f64 {
        let (valence, arousal) = self.get_state();
        arousal.atan2(valence)
    }

    /// Map V-A coordinates to a specific emotion region
    pub fn map_to_emotion(&self) -> CoreAffect {
        let (valence, arousal) = self.get_state();
        let angle = self.get_angle();
        let intensity = self.get_intensity();

        // Map angle to emotion categories (simplified)
        let emotion = match angle {
            a if a >= -0.39 && a < 0.39 => CoreAffect::Happy,
            a if a >= 0.39 && a < 1.18 => CoreAffect::Excited,
            a if a >= 1.18 && a < 1.96 => CoreAffect::Alert,
            a if a >= 1.96 && a < 2.75 => CoreAffect::Tense,
            a if a >= 2.75 || a < -2.75 => CoreAffect::Upset,
            a if a >= -2.75 && a < -1.96 => CoreAffect::Sad,
            a if a >= -1.96 && a < -1.18 => CoreAffect::Depressed,
            a if a >= -1.18 && a < -0.39 => CoreAffect::Calm,
            _ => CoreAffect::Neutral,
        };

        // Low intensity maps to neutral
        if intensity < 0.2 {
            CoreAffect::Neutral
        } else {
            emotion
        }
    }
}

/// Quadrants in the V-A space
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VAQuadrant {
    HighValenceHighArousal, // Excited, elated
    HighValenceLowArousal,  // Content, serene
    LowValenceHighArousal,  // Tense, nervous
    LowValenceLowArousal,   // Sad, depressed
}

/// Core affect states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoreAffect {
    Neutral,
    Happy,
    Excited,
    Alert,
    Tense,
    Upset,
    Sad,
    Depressed,
    Calm,
}
