//! Pattern detection algorithms for DRPP

use super::{NeuralOscillator, Pattern, PatternType};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// 🛡️ HARDENING: Circuit breaker constants
const MAX_CONSECUTIVE_FAILURES: usize = 10;
const CIRCUIT_RECOVERY_TIME_NS: u64 = 1_000_000_000; // 1 second
const MAX_HISTORY_SIZE: usize = 10_000;
const SUCCESS_THRESHOLD_TO_CLOSE: usize = 5;

/// Pattern detector for oscillator networks with circuit breaker protection
pub struct PatternDetector {
    /// Detection threshold
    threshold: f64,

    /// Time window for pattern analysis
    time_window: usize,

    /// Historical states
    history: Arc<Mutex<VecDeque<Vec<f64>>>>,

    /// Pattern ID counter
    pattern_id_counter: Arc<AtomicU64>,

    /// 🛡️ HARDENING: Circuit breaker state
    failure_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    last_failure_time: Arc<AtomicU64>, // NanoTime as u64
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(config: &super::DrppConfig) -> Self {
        Self {
            threshold: config.pattern_threshold,
            time_window: (config.time_window_ms / 10) as usize, // 10ms samples
            history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            pattern_id_counter: Arc::new(AtomicU64::new(0)),
            // 🛡️ HARDENING: Initialize circuit breaker
            failure_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            last_failure_time: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Detect patterns in oscillator network with circuit breaker protection
    pub fn detect(&self, oscillators: &[NeuralOscillator]) -> Vec<Pattern> {
        // 🛡️ HARDENING: Check circuit breaker state
        if self.is_circuit_open() {
            return Vec::new(); // Circuit open - fail fast
        }

        // 🛡️ HARDENING: Input validation
        if oscillators.is_empty() {
            self.record_failure("Empty oscillator array");
            return Vec::new();
        }

        // 🛡️ HARDENING: Prevent ID overflow (extremely unlikely but possible)
        let current_id = self.pattern_id_counter.load(Ordering::Relaxed);
        if current_id > u64::MAX - 1000 {
            self.record_failure("Pattern ID counter near overflow");
            return Vec::new();
        }

        // Capture current state
        let current_state: Vec<f64> = oscillators.iter().map(|o| o.output()).collect();

        // Update history with resource limits
        {
            let mut history = self.history.lock().unwrap();

            // 🛡️ HARDENING: Enforce maximum history size
            if history.len() >= MAX_HISTORY_SIZE {
                tracing::warn!(
                    "Pattern history at maximum size {}, dropping oldest entries",
                    MAX_HISTORY_SIZE
                );
                while history.len() >= MAX_HISTORY_SIZE / 2 {
                    history.pop_front();
                }
            }

            history.push_back(current_state.clone());
            if history.len() > self.time_window {
                history.pop_front();
            }
        }

        let mut patterns = Vec::new();

        // Check for synchronous patterns
        if let Some(sync_pattern) = self.detect_synchrony(oscillators) {
            patterns.push(sync_pattern);
        }

        // Check for traveling waves
        if let Some(wave_pattern) = self.detect_traveling_wave(&current_state) {
            patterns.push(wave_pattern);
        }

        // Check for standing waves
        if let Some(standing_pattern) = self.detect_standing_wave() {
            patterns.push(standing_pattern);
        }

        // Check for emergent patterns
        if let Some(emergent_pattern) = self.detect_emergent_pattern(&current_state) {
            patterns.push(emergent_pattern);
        }

        // 🛡️ HARDENING: Record successful detection
        self.record_success();

        patterns
    }

    /// 🛡️ HARDENING: Check if circuit breaker is open
    fn is_circuit_open(&self) -> bool {
        let failure_count = self.failure_count.load(Ordering::Relaxed);

        // Check if we've exceeded failure threshold
        if failure_count < MAX_CONSECUTIVE_FAILURES {
            return false;
        }

        // Check if recovery time has elapsed
        let last_failure = self.last_failure_time.load(Ordering::Relaxed);
        let now = csf_core::types::hardware_timestamp().as_nanos() as u64;

        if now.saturating_sub(last_failure) > CIRCUIT_RECOVERY_TIME_NS {
            // Recovery time elapsed - allow half-open state
            return false;
        }

        tracing::warn!(
            "Pattern detector circuit breaker OPEN: {} failures, last failure {}ns ago",
            failure_count,
            now.saturating_sub(last_failure)
        );
        true
    }

    /// 🛡️ HARDENING: Record a failure and update circuit breaker state
    fn record_failure(&self, reason: &str) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        let now = csf_core::types::hardware_timestamp().as_nanos() as u64;
        self.last_failure_time.store(now, Ordering::Relaxed);

        tracing::warn!("Pattern detector failure #{}: {}", failure_count, reason);

        if failure_count >= MAX_CONSECUTIVE_FAILURES {
            tracing::error!(
                "Pattern detector circuit breaker OPENED after {} failures",
                failure_count
            );
        }
    }

    /// 🛡️ HARDENING: Record a success and potentially close the circuit breaker
    fn record_success(&self) {
        let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Reset failure count on success
        let previous_failures = self.failure_count.swap(0, Ordering::Relaxed);

        if previous_failures > 0 && success_count % SUCCESS_THRESHOLD_TO_CLOSE == 0 {
            tracing::info!(
                "Pattern detector circuit breaker CLOSED after {} successes (recovered from {} failures)",
                success_count,
                previous_failures
            );
        }
    }

    /// Detect synchronous activity
    fn detect_synchrony(&self, oscillators: &[NeuralOscillator]) -> Option<Pattern> {
        let phases: Vec<f64> = oscillators.iter().map(|o| o.phase()).collect();

        // Calculate phase coherence
        let mean_phase = phases.iter().sum::<f64>() / phases.len() as f64;
        let coherence =
            phases.iter().map(|&p| (p - mean_phase).cos()).sum::<f64>() / phases.len() as f64;

        if coherence > self.threshold {
            let id = self.pattern_id_counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(Pattern {
                id,
                pattern_type: PatternType::Synchronous,
                strength: coherence,
                frequencies: oscillators.iter().map(|o| o.frequency()).collect(),
                spatial_map: phases,
                timestamp: csf_core::types::hardware_timestamp(),
            })
        } else {
            None
        }
    }

    /// Detect traveling wave patterns
    fn detect_traveling_wave(&self, state: &[f64]) -> Option<Pattern> {
        if self.history.lock().unwrap().len() < 3 {
            return None;
        }

        // Calculate spatial gradient
        let gradient: Vec<f64> = state.windows(2).map(|w| w[1] - w[0]).collect();

        // Check for consistent gradient direction
        let mean_gradient = gradient.iter().sum::<f64>() / gradient.len() as f64;
        let gradient_consistency = gradient
            .iter()
            .map(|&g| if g * mean_gradient > 0.0 { 1.0 } else { 0.0 })
            .sum::<f64>()
            / gradient.len() as f64;

        if gradient_consistency > self.threshold {
            let id = self.pattern_id_counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(Pattern {
                id,
                pattern_type: PatternType::Traveling,
                strength: gradient_consistency,
                frequencies: vec![mean_gradient.abs()], // Wave speed proxy
                spatial_map: gradient,
                timestamp: csf_core::types::hardware_timestamp(),
            })
        } else {
            None
        }
    }

    /// Detect standing wave patterns
    fn detect_standing_wave(&self) -> Option<Pattern> {
        let history = self.history.lock().unwrap();
        if history.len() < self.time_window {
            return None;
        }

        // Calculate temporal variance at each position
        let n_oscillators = history[0].len();
        let mut variances = vec![0.0; n_oscillators];

        for i in 0..n_oscillators {
            let values: Vec<f64> = history.iter().map(|state| state[i]).collect();

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

            variances[i] = variance;
        }

        // Look for nodes (low variance) and antinodes (high variance)
        let mean_variance = variances.iter().sum::<f64>() / variances.len() as f64;
        let variance_range = variances
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - variances
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

        if variance_range / mean_variance > 2.0 {
            let id = self.pattern_id_counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(Pattern {
                id,
                pattern_type: PatternType::Standing,
                strength: variance_range / mean_variance,
                frequencies: vec![], // Would calculate from FFT
                spatial_map: variances,
                timestamp: csf_core::types::hardware_timestamp(),
            })
        } else {
            None
        }
    }

    /// Detect emergent patterns using complexity measures
    fn detect_emergent_pattern(&self, state: &[f64]) -> Option<Pattern> {
        // Simple emergence detection using local vs global variance
        let global_mean = state.iter().sum::<f64>() / state.len() as f64;
        let global_variance = state
            .iter()
            .map(|&v| (v - global_mean).powi(2))
            .sum::<f64>()
            / state.len() as f64;

        // Calculate local variances
        let window_size = 8;
        let mut local_variances = Vec::new();

        for chunk in state.chunks(window_size) {
            let local_mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let local_var =
                chunk.iter().map(|&v| (v - local_mean).powi(2)).sum::<f64>() / chunk.len() as f64;
            local_variances.push(local_var);
        }

        // High local variance with low global variance indicates emergence
        let mean_local_var = local_variances.iter().sum::<f64>() / local_variances.len() as f64;
        let emergence_score = mean_local_var / (global_variance + 1e-6);

        if emergence_score > 2.0 {
            let id = self.pattern_id_counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(Pattern {
                id,
                pattern_type: PatternType::Emergent,
                strength: emergence_score.min(1.0),
                frequencies: vec![],
                spatial_map: state.to_vec(),
                timestamp: csf_core::types::hardware_timestamp(),
            })
        } else {
            None
        }
    }
}
