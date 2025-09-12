//! Neural oscillator implementation for DRPP

use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

/// Neural oscillator with adaptive frequency
pub struct NeuralOscillator {
    /// Oscillator ID
    id: usize,

    /// Current phase (radians)
    phase: AtomicU64,

    /// Natural frequency (Hz)
    frequency: AtomicU64,

    /// Amplitude
    amplitude: AtomicU64,

    /// Phase coupling strength
    coupling: AtomicU64,
}

impl NeuralOscillator {
    /// Create a new neural oscillator
    pub fn new(id: usize, config: &super::DrppConfig) -> Self {
        // Initialize with random phase and frequency within range
        let phase = rand::random::<f64>() * 2.0 * PI;
        let freq_range = config.frequency_range.1 - config.frequency_range.0;
        let frequency = config.frequency_range.0 + rand::random::<f64>() * freq_range;

        Self {
            id,
            phase: AtomicU64::new(phase.to_bits()),
            frequency: AtomicU64::new(frequency.to_bits()),
            amplitude: AtomicU64::new(1.0f64.to_bits()),
            coupling: AtomicU64::new(config.coupling_strength.to_bits()),
        }
    }

    /// Update oscillator state
    pub fn update(&self, input: f64, neighbors: &[Self], coupling_strength: f64) {
        let current_phase = f64::from_bits(self.phase.load(Ordering::Relaxed));
        let frequency = f64::from_bits(self.frequency.load(Ordering::Relaxed));
        let amplitude = f64::from_bits(self.amplitude.load(Ordering::Relaxed));

        // Calculate phase update from natural frequency
        let dt = 0.001; // 1ms timestep
        let mut phase_delta = 2.0 * PI * frequency * dt;

        // Add coupling from neighbors
        let mut coupling_sum = 0.0;
        let mut neighbor_count = 0;

        for neighbor in neighbors {
            if neighbor.id != self.id {
                let neighbor_phase = neighbor.phase();
                let phase_diff = neighbor_phase - current_phase;
                coupling_sum += phase_diff.sin();
                neighbor_count += 1;
            }
        }

        if neighbor_count > 0 {
            phase_delta += coupling_strength * coupling_sum / neighbor_count as f64;
        }

        // Add input modulation
        phase_delta += input * amplitude * 0.1;

        // Update phase (wrap around 2π)
        let new_phase = (current_phase + phase_delta) % (2.0 * PI);
        self.phase.store(new_phase.to_bits(), Ordering::Relaxed);

        // Adaptive frequency adjustment
        if input.abs() > 0.5 {
            let freq_adjustment = input * 0.01;
            let new_frequency = (frequency + freq_adjustment).max(0.1).min(100.0);
            self.frequency
                .store(new_frequency.to_bits(), Ordering::Relaxed);
        }
    }

    /// Get current phase
    pub fn phase(&self) -> f64 {
        f64::from_bits(self.phase.load(Ordering::Relaxed))
    }

    /// Get current frequency
    pub fn frequency(&self) -> f64 {
        f64::from_bits(self.frequency.load(Ordering::Relaxed))
    }

    /// Get current amplitude
    pub fn amplitude(&self) -> f64 {
        f64::from_bits(self.amplitude.load(Ordering::Relaxed))
    }

    /// Calculate instantaneous output
    pub fn output(&self) -> f64 {
        let phase = self.phase();
        let amplitude = self.amplitude();
        amplitude * phase.sin()
    }
}

impl Clone for NeuralOscillator {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            phase: AtomicU64::new(self.phase.load(Ordering::Relaxed)),
            frequency: AtomicU64::new(self.frequency.load(Ordering::Relaxed)),
            amplitude: AtomicU64::new(self.amplitude.load(Ordering::Relaxed)),
            coupling: AtomicU64::new(self.coupling.load(Ordering::Relaxed)),
        }
    }
}
