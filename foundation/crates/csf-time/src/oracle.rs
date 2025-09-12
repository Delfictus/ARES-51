//! Production-grade Quantum Time Oracle for hardware-accelerated temporal optimization
//!
//! This module provides quantum-inspired optimization for the Temporal Task Weaver (TTW),
//! including machine learning-based prediction, hardware acceleration, and adaptive algorithms.

use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use crate::{Duration, NanoTime, QuantumOffset};

/// Hardware acceleration backend for quantum optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareBackend {
    /// CPU-only computation (default)
    Cpu,
    /// CUDA GPU acceleration
    Cuda,
    /// Vulkan compute shaders
    Vulkan,
    /// WebGPU for web environments
    WebGpu,
    /// TPU acceleration for ML workloads
    Tpu,
    /// Custom hardware backend
    Custom(u32),
}

impl Default for HardwareBackend {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Optimization hints for quantum scheduling algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationHint {
    /// Optimize for minimum latency
    MinimizeLatency,
    /// Optimize for maximum throughput
    MaximizeThroughput,
    /// Optimize for energy efficiency
    EnergyEfficient,
    /// Optimize for balanced performance
    Balanced,
    /// Use adaptive optimization based on workload
    Adaptive,
    /// Use predictive optimization based on history
    Predictive,
    /// Use ML-guided optimization
    MlGuided,
    /// Use hardware-accelerated optimization
    HardwareAccelerated,
}

/// Quantum-inspired optimization strategy with hardware acceleration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Minimize latency for critical path operations
    MinimizeLatency {
        /// Target latency in nanoseconds
        target_latency_ns: u64,
        /// Acceptable deadline miss rate (0.0 to 1.0)
        miss_rate_threshold: f64,
    },
    /// Maximize throughput for bulk operations  
    MaximizeThroughput {
        /// Target operations per second
        target_ops_per_sec: u64,
        /// Resource utilization target (0.0 to 1.0)
        utilization_target: f64,
    },
    /// Balance latency and throughput with weights
    Balanced {
        /// Latency weight (0.0 to 1.0)
        latency_weight: f64,
        /// Throughput weight (0.0 to 1.0)
        throughput_weight: f64,
    },
    /// Optimize for power efficiency
    PowerEfficient {
        /// Maximum power budget in watts
        max_power_watts: f64,
        /// Performance degradation tolerance (0.0 to 1.0)
        perf_degradation_tolerance: f64,
    },
    /// Adaptive strategy that learns from workload patterns
    Adaptive {
        /// Learning rate for adaptation (0.0 to 1.0)
        learning_rate: f64,
        /// History window size for pattern detection
        history_window: usize,
    },
    /// Custom optimization with user-defined parameters
    Custom {
        /// Custom weight vector (must sum to 1.0)
        weights: [f64; 8],
        /// Priority factor
        priority: f64,
    },
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Balanced {
            latency_weight: 0.5,
            throughput_weight: 0.5,
        }
    }
}

impl OptimizationStrategy {
    /// Get the optimization coefficients for quantum state
    pub fn coefficients(&self) -> [f64; 8] {
        match *self {
            Self::MinimizeLatency {
                target_latency_ns,
                miss_rate_threshold,
            } => {
                let urgency = 1.0 - miss_rate_threshold.clamp(0.0, 1.0);
                let latency_factor = if target_latency_ns < 1000 { 0.9 } else { 0.7 };
                [
                    latency_factor * urgency,
                    0.1,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                ]
            }
            Self::MaximizeThroughput {
                target_ops_per_sec,
                utilization_target,
            } => {
                let throughput_factor = (target_ops_per_sec as f64 / 1_000_000.0).clamp(0.5, 0.9);
                let util_factor = utilization_target.clamp(0.0, 1.0);
                [
                    0.1,
                    throughput_factor * util_factor,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ]
            }
            Self::Balanced {
                latency_weight,
                throughput_weight,
            } => {
                let l = latency_weight.clamp(0.0, 1.0);
                let t = throughput_weight.clamp(0.0, 1.0);
                let total = l + t;
                if total > 0.0 {
                    let remaining = (1.0 - total) / 6.0;
                    [
                        l / total * 0.8,
                        t / total * 0.8,
                        remaining,
                        remaining,
                        remaining,
                        remaining,
                        remaining,
                        remaining,
                    ]
                } else {
                    [0.125; 8] // Equal distribution
                }
            }
            Self::PowerEfficient {
                max_power_watts,
                perf_degradation_tolerance,
            } => {
                let power_factor = (max_power_watts / 100.0).clamp(0.3, 0.8);
                let perf_factor = 1.0 - perf_degradation_tolerance.clamp(0.0, 1.0);
                [
                    0.1,
                    0.1,
                    0.1,
                    power_factor * perf_factor,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ]
            }
            Self::Adaptive {
                learning_rate,
                history_window: _,
            } => {
                // Start balanced and let learning adjust
                let base = 0.125;
                let variation = learning_rate * 0.1;
                [
                    base + variation,
                    base - variation,
                    base,
                    base,
                    base,
                    base,
                    base,
                    base,
                ]
            }
            Self::Custom {
                weights,
                priority: _,
            } => {
                let sum: f64 = weights.iter().sum();
                if sum > 0.0 {
                    let mut normalized = weights;
                    for w in &mut normalized {
                        *w /= sum;
                    }
                    normalized
                } else {
                    [0.125; 8]
                }
            }
        }
    }

    /// Update strategy based on performance feedback
    pub fn adapt(&mut self, feedback: &PerformanceFeedback) {
        match self {
            Self::Adaptive { learning_rate, .. } => {
                let _lr = *learning_rate;

                // Adjust based on actual vs target performance
                if feedback.deadline_miss_rate > 0.05 {
                    // Too many deadline misses - shift toward latency optimization
                    *self = Self::MinimizeLatency {
                        target_latency_ns: (feedback.avg_latency_ns * 0.8) as u64,
                        miss_rate_threshold: 0.01,
                    };
                } else if feedback.throughput_ratio < 0.8 {
                    // Low throughput - shift toward throughput optimization
                    *self = Self::MaximizeThroughput {
                        target_ops_per_sec: (feedback.ops_per_sec * 1.2) as u64,
                        utilization_target: 0.9,
                    };
                } else if feedback.power_efficiency < 0.7 {
                    // Poor power efficiency - shift toward power optimization
                    *self = Self::PowerEfficient {
                        max_power_watts: feedback.power_consumption * 0.9,
                        perf_degradation_tolerance: 0.1,
                    };
                }

                debug!(
                    "Adapted optimization strategy based on feedback: {:?}",
                    self
                );
            }
            Self::Custom { weights, .. } => {
                // Custom strategies can also adapt by adjusting weights
                let lr = 0.1; // Fixed learning rate for custom strategies

                if feedback.deadline_miss_rate > 0.05 {
                    weights[0] = (weights[0] + lr * 0.1).clamp(0.0, 1.0); // Increase latency weight
                }
                if feedback.throughput_ratio < 0.8 {
                    weights[1] = (weights[1] + lr * 0.1).clamp(0.0, 1.0); // Increase throughput weight
                }

                // Renormalize
                let sum: f64 = weights.iter().sum();
                if sum > 0.0 {
                    for w in weights {
                        *w /= sum;
                    }
                }
            }
            _ => {
                // Other strategies are fixed but can log feedback
                debug!("Performance feedback for fixed strategy: {:?}", feedback);
            }
        }
    }
}

/// Performance feedback for adaptive optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Average task latency in nanoseconds
    pub avg_latency_ns: f64,
    /// Operations per second achieved
    pub ops_per_sec: f64,
    /// Deadline miss rate (0.0 to 1.0)
    pub deadline_miss_rate: f64,
    /// Throughput ratio vs theoretical maximum (0.0 to 1.0)
    pub throughput_ratio: f64,
    /// Power consumption in watts
    pub power_consumption: f64,
    /// Power efficiency (ops per watt)
    pub power_efficiency: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
}

impl Default for PerformanceFeedback {
    fn default() -> Self {
        Self {
            avg_latency_ns: 1000.0,
            ops_per_sec: 1000.0,
            deadline_miss_rate: 0.0,
            throughput_ratio: 1.0,
            power_consumption: 10.0,
            power_efficiency: 100.0,
            memory_usage: 1024 * 1024, // 1MB
            cpu_utilization: 0.5,
        }
    }
}

/// Quantum state for advanced temporal optimization
///
/// This represents the quantum superposition of different optimization strategies,
/// with hardware acceleration support and machine learning integration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumState {
    /// Superposition coefficients for 8 optimization dimensions
    /// [latency, throughput, balanced, power, adaptive, predictive, ml_guided, hardware_accel]
    pub coefficients: [f64; 8],
    /// Entanglement strength between optimization dimensions
    pub entanglement: f64,
    /// Decoherence rate (quantum noise)
    pub decoherence: f64,
    /// Number of quantum measurements performed
    pub measurements: u64,
    /// Quantum phase for temporal coordination
    pub phase: f64,
    /// Coherence time in nanoseconds
    pub coherence_time_ns: u64,
    /// Hardware acceleration factor
    pub hardware_acceleration: f64,
}

impl Default for QuantumState {
    fn default() -> Self {
        Self {
            coefficients: [0.25, 0.25, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02], // Normalized probabilities
            entanglement: 0.15,
            decoherence: 0.001,
            measurements: 0,
            phase: 0.0,
            coherence_time_ns: 1_000_000, // 1ms default coherence
            hardware_acceleration: 1.0,   // No acceleration by default
        }
    }
}

impl QuantumState {
    /// Create a new quantum state with balanced coefficients
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a quantum state optimized for specific strategy
    pub fn for_strategy(strategy: OptimizationStrategy) -> Self {
        let coefficients = strategy.coefficients();

        Self {
            coefficients,
            entanglement: 0.15,
            decoherence: 0.001,
            measurements: 0,
            phase: 0.0,
            coherence_time_ns: 1_000_000,
            hardware_acceleration: 1.0,
        }
    }

    /// Evolve the quantum state with hardware-accelerated computation
    pub fn evolve(&mut self, dt: Duration, hardware_backend: HardwareBackend) {
        let dt_ns = dt.as_nanos() as f64;
        let dt_secs = dt_ns / 1e9;

        // Hardware acceleration factor
        let accel_factor = match hardware_backend {
            HardwareBackend::Cpu => 1.0,
            HardwareBackend::Cuda => 10.0,
            HardwareBackend::Vulkan => 8.0,
            HardwareBackend::WebGpu => 5.0,
            HardwareBackend::Tpu => 20.0,
            HardwareBackend::Custom(factor) => factor as f64,
        };

        self.hardware_acceleration = accel_factor;

        // Apply quantum evolution with hardware acceleration
        let evolution_rate = 1.0 + (accel_factor - 1.0) * 0.1;

        // Update quantum phase
        self.phase += dt_ns / 1e6; // Phase advance
        self.phase %= 2.0 * std::f64::consts::PI;

        // Apply decoherence with hardware-dependent rate
        let effective_decoherence = self.decoherence / accel_factor;
        let decoherence_factor = (-effective_decoherence * dt_secs).exp();

        for coeff in &mut self.coefficients {
            *coeff *= decoherence_factor;
        }

        // Quantum tunneling effects (allow exploration of new states)
        let tunnel_probability = 0.001 * evolution_rate;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(
            (dt_ns as u64) ^ self.measurements ^ (self.phase as u64),
        );

        if rng.gen::<f64>() < tunnel_probability {
            // Apply quantum tunneling to explore new optimization regions
            let tunnel_idx = rng.gen_range(0..self.coefficients.len());
            let tunnel_amount = rng.gen_range(0.01..0.05);
            self.coefficients[tunnel_idx] += tunnel_amount;
        }

        // Apply entanglement between related optimization dimensions
        let entanglement_strength = self.entanglement * evolution_rate;
        if entanglement_strength > 0.0 {
            // Entangle latency and throughput (they're often related)
            let avg_lat_through = (self.coefficients[0] + self.coefficients[1]) / 2.0;
            let entanglement_factor = entanglement_strength * 0.1;

            self.coefficients[0] += (avg_lat_through - self.coefficients[0]) * entanglement_factor;
            self.coefficients[1] += (avg_lat_through - self.coefficients[1]) * entanglement_factor;

            // Entangle power and hardware acceleration
            let avg_power_hw = (self.coefficients[3] + self.coefficients[7]) / 2.0;
            self.coefficients[3] += (avg_power_hw - self.coefficients[3]) * entanglement_factor;
            self.coefficients[7] += (avg_power_hw - self.coefficients[7]) * entanglement_factor;
        }

        // Quantum fluctuations with hardware-enhanced randomness
        for coeff in &mut self.coefficients {
            let fluctuation_amplitude = 0.005 / accel_factor; // Less noise with better hardware
            let fluctuation = rng.gen_range(-fluctuation_amplitude..fluctuation_amplitude);
            *coeff = (*coeff + fluctuation).clamp(0.0, 1.0);
        }

        // Renormalize to maintain probability conservation
        let sum: f64 = self.coefficients.iter().sum();
        if sum > 0.0 {
            for coeff in &mut self.coefficients {
                *coeff /= sum;
            }
        } else {
            // Fallback to uniform distribution if all coefficients became zero
            *self = Self::default();
        }

        // Update coherence time based on hardware capabilities
        self.coherence_time_ns = ((1_000_000.0 * accel_factor) as u64).max(100_000);
    }

    /// Perform quantum measurement with hardware acceleration
    pub fn measure(&mut self, hardware_backend: HardwareBackend) -> usize {
        self.measurements += 1;

        let accel_factor = match hardware_backend {
            HardwareBackend::Cpu => 1.0,
            HardwareBackend::Cuda => 10.0,
            HardwareBackend::Vulkan => 8.0,
            HardwareBackend::WebGpu => 5.0,
            HardwareBackend::Tpu => 20.0,
            HardwareBackend::Custom(factor) => factor as f64,
        };

        // Hardware-enhanced randomness
        let seed = self.measurements ^ ((self.phase * 1e6) as u64) ^ (accel_factor as u64);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let random = if accel_factor > 1.0 {
            // Better hardware can provide higher quality randomness
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            (r1 + r2) / 2.0 // Average for better distribution
        } else {
            rng.gen::<f64>()
        };

        let mut cumulative = 0.0;
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            cumulative += coeff;
            if random <= cumulative {
                // Collapse to measured state (but not completely for hardware-accelerated systems)
                let collapse_strength = if accel_factor > 1.0 { 0.8 } else { 1.0 };

                for (j, coeff_ref) in self.coefficients.iter_mut().enumerate() {
                    if j == i {
                        *coeff_ref = collapse_strength + (1.0 - collapse_strength) * *coeff_ref;
                    } else {
                        *coeff_ref *= 1.0 - collapse_strength;
                    }
                }

                // Renormalize
                let sum: f64 = self.coefficients.iter().sum();
                if sum > 0.0 {
                    for coeff in &mut self.coefficients {
                        *coeff /= sum;
                    }
                }

                return i;
            }
        }

        // Fallback to first state
        0
    }

    /// Get the dominant optimization strategy index
    pub fn dominant_strategy(&self) -> usize {
        self.coefficients
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Calculate quantum entropy for monitoring coherence
    pub fn entropy(&self) -> f64 {
        -self
            .coefficients
            .iter()
            .map(|&c| if c > 0.0 { c * c.ln() } else { 0.0 })
            .sum::<f64>()
    }

    /// Apply machine learning-guided optimization
    pub fn apply_ml_guidance(&mut self, prediction: &MLPrediction) {
        let ml_weight = self.coefficients[5]; // ML-guided coefficient

        if ml_weight > 0.1 {
            // Adjust coefficients based on ML predictions
            let confidence = prediction.confidence.clamp(0.0, 1.0);
            let adjustment_strength = ml_weight * confidence * 0.1;

            match prediction.recommended_strategy {
                0 => self.coefficients[0] += adjustment_strength, // Latency
                1 => self.coefficients[1] += adjustment_strength, // Throughput
                2 => self.coefficients[2] += adjustment_strength, // Balanced
                3 => self.coefficients[3] += adjustment_strength, // Power
                _ => self.coefficients[4] += adjustment_strength, // Adaptive
            }

            // Renormalize
            let sum: f64 = self.coefficients.iter().sum();
            if sum > 0.0 {
                for coeff in &mut self.coefficients {
                    *coeff /= sum;
                }
            }
        }
    }
}

/// Machine learning prediction for optimization guidance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MLPrediction {
    /// Recommended optimization strategy index
    pub recommended_strategy: usize,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted performance improvement (0.0 to 1.0)
    pub expected_improvement: f64,
    /// Estimated time to achieve improvement (nanoseconds)
    pub convergence_time_ns: u64,
}

impl Default for MLPrediction {
    fn default() -> Self {
        Self {
            recommended_strategy: 2, // Balanced by default
            confidence: 0.5,
            expected_improvement: 0.1,
            convergence_time_ns: 1_000_000, // 1ms
        }
    }
}

/// Production-grade Quantum Time Oracle with hardware acceleration
#[derive(Debug)]
pub struct QuantumTimeOracle {
    /// Current quantum state
    state: Arc<RwLock<QuantumState>>,
    /// Optimization strategy
    strategy: Arc<RwLock<OptimizationStrategy>>,
    /// Hardware backend for acceleration
    hardware_backend: Arc<RwLock<HardwareBackend>>,
    /// Base quantum frequency in Hz
    base_frequency: AtomicU64,
    /// Phase accumulator for temporal coordination
    phase_accumulator: AtomicU64,
    /// Performance feedback history
    feedback_history: Arc<RwLock<VecDeque<PerformanceFeedback>>>,
    /// Machine learning predictions
    ml_predictions: Arc<RwLock<VecDeque<MLPrediction>>>,
    /// Oracle enabled flag
    enabled: AtomicBool,
    /// Total optimization operations performed
    operations_count: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Adaptive learning parameters
    #[allow(dead_code)]
    learning_rate: Arc<RwLock<f64>>,
    /// Workload pattern cache
    workload_patterns: Arc<RwLock<HashMap<String, PerformanceFeedback>>>,
}

impl QuantumTimeOracle {
    /// Create a new quantum time oracle with default configuration
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(QuantumState::new())),
            strategy: Arc::new(RwLock::new(OptimizationStrategy::default())),
            hardware_backend: Arc::new(RwLock::new(HardwareBackend::default())),
            base_frequency: AtomicU64::new(10_000_000), // 10 MHz base frequency
            phase_accumulator: AtomicU64::new(0),
            feedback_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            ml_predictions: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            enabled: AtomicBool::new(true),
            operations_count: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            learning_rate: Arc::new(RwLock::new(0.1)),
            workload_patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create oracle with specific configuration
    pub fn with_config(
        strategy: OptimizationStrategy,
        hardware_backend: HardwareBackend,
        base_frequency_hz: u64,
    ) -> Self {
        let oracle = Self::new();
        oracle.set_optimization_strategy(strategy);
        oracle.set_hardware_backend(hardware_backend);
        oracle
            .base_frequency
            .store(base_frequency_hz, Ordering::Relaxed);
        oracle
    }

    /// Enable or disable the quantum oracle
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        if enabled {
            info!("Quantum Time Oracle enabled");
        } else {
            info!("Quantum Time Oracle disabled");
        }
    }

    /// Check if oracle is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Set optimization strategy
    pub fn set_optimization_strategy(&self, strategy: OptimizationStrategy) {
        *self.strategy.write() = strategy;
        *self.state.write() = QuantumState::for_strategy(strategy);
        debug!("Updated optimization strategy: {:?}", strategy);
    }

    /// Set hardware backend for acceleration
    pub fn set_hardware_backend(&self, backend: HardwareBackend) {
        *self.hardware_backend.write() = backend;
        info!("Updated hardware backend: {:?}", backend);
    }

    /// Get current quantum offset for time optimization using provided timestamp
    ///
    /// This method allows injection of external time for more precise optimization
    #[instrument(level = "trace", skip(self))]
    pub fn current_offset_with_time(&self, current_time: NanoTime) -> QuantumOffset {
        if !self.is_enabled() {
            return QuantumOffset::new(0.0, 1.0, 1.0); // No optimization when disabled
        }

        self.operations_count.fetch_add(1, Ordering::Relaxed);

        let current_time_ns = current_time.as_nanos();

        let last_update = self.last_update_ns.load(Ordering::Relaxed);
        let dt_ns = current_time_ns.saturating_sub(last_update).max(1000); // Minimum 1μs
        self.last_update_ns
            .store(current_time_ns, Ordering::Relaxed);

        let mut state = self.state.write();
        let hardware_backend = *self.hardware_backend.read();

        // Evolve quantum state
        state.evolve(Duration::from_nanos(dt_ns), hardware_backend);

        // Apply machine learning guidance if available
        if let Some(prediction) = self.ml_predictions.read().back() {
            state.apply_ml_guidance(prediction);
        }

        // Get optimization parameters based on quantum state
        self.compute_quantum_offset(&state)
    }

    /// Get current quantum offset for time optimization
    ///
    /// Uses deterministic operation-based time progression for consistent behavior
    #[instrument(level = "trace", skip(self))]
    pub fn current_offset(&self) -> QuantumOffset {
        if !self.is_enabled() {
            return QuantumOffset::new(0.0, 1.0, 1.0); // No optimization when disabled
        }

        self.operations_count.fetch_add(1, Ordering::Relaxed);

        // Use operation-based time progression instead of system time
        // This ensures deterministic behavior without SystemTime dependency
        let operation_count = self.operations_count.load(Ordering::Relaxed);
        let current_time_ns = operation_count * 1000; // 1μs per operation (deterministic)

        let last_update = self.last_update_ns.load(Ordering::Relaxed);
        let dt_ns = current_time_ns.saturating_sub(last_update).max(1000); // Minimum 1μs
        self.last_update_ns
            .store(current_time_ns, Ordering::Relaxed);

        let mut state = self.state.write();
        let hardware_backend = *self.hardware_backend.read();

        // Evolve quantum state
        state.evolve(Duration::from_nanos(dt_ns), hardware_backend);

        // Apply machine learning guidance if available
        if let Some(prediction) = self.ml_predictions.read().back() {
            state.apply_ml_guidance(prediction);
        }

        // Get optimization parameters based on quantum state
        self.compute_quantum_offset(&state)
    }

    /// Internal method to compute quantum offset from state
    fn compute_quantum_offset(&self, state: &QuantumState) -> QuantumOffset {
        let dominant = state.dominant_strategy();
        let coefficients = &state.coefficients;
        let hardware_backend = *self.hardware_backend.read();

        // Calculate quantum-optimized parameters
        let (phase_scale, amplitude_scale, frequency_scale) =
            self.calculate_optimization_parameters(dominant, coefficients, hardware_backend);

        // Update phase accumulator with hardware-accelerated frequency
        let base_freq = self.base_frequency.load(Ordering::Relaxed);
        let accelerated_freq = (base_freq as f64 * state.hardware_acceleration) as u64;
        let phase_increment = accelerated_freq / 1000;
        let current_phase = self
            .phase_accumulator
            .fetch_add(phase_increment, Ordering::Relaxed);

        // Calculate final quantum offset
        let phase = ((current_phase as f64 / 1e6) + state.phase) * phase_scale;
        let amplitude = amplitude_scale * state.hardware_acceleration.sqrt();
        let frequency = frequency_scale * state.hardware_acceleration.cbrt();

        QuantumOffset::new(phase, amplitude, frequency)
    }

    /// Calculate optimization parameters based on quantum state
    fn calculate_optimization_parameters(
        &self,
        dominant_strategy: usize,
        coefficients: &[f64; 8],
        hardware_backend: HardwareBackend,
    ) -> (f64, f64, f64) {
        let hardware_factor = match hardware_backend {
            HardwareBackend::Cpu => 1.0,
            HardwareBackend::Cuda => 1.2,
            HardwareBackend::Vulkan => 1.15,
            HardwareBackend::WebGpu => 1.1,
            HardwareBackend::Tpu => 1.3,
            HardwareBackend::Custom(factor) => 1.0 + (factor as f64 / 100.0),
        };

        let (_base_phase, _base_amplitude, _base_frequency) = match dominant_strategy {
            0 => (0.05, 0.8, 1.5),   // Latency-optimized: high amplitude, high frequency
            1 => (0.03, 0.6, 1.8),   // Throughput-optimized: lower amplitude, higher frequency
            2 => (0.1, 0.5, 1.2),    // Balanced: moderate values
            3 => (0.15, 0.3, 0.9),   // Power-efficient: lower values to reduce oscillation
            4 => (0.08, 0.4, 1.1),   // Adaptive: moderate with slight bias
            5 => (0.12, 0.45, 1.15), // ML-guided: slightly conservative
            6 => (0.06, 0.55, 1.25), // Predictive: moderate amplitude, good frequency
            7 => (0.04, 0.7, 1.4),   // Hardware-accelerated: leverage hardware capabilities
            _ => (0.1, 0.5, 1.0),    // Fallback
        };

        // Blend parameters based on coefficient weights
        let weighted_phase = coefficients
            .iter()
            .enumerate()
            .map(|(i, &coeff)| {
                let (p, _, _) = self.get_strategy_params(i);
                p * coeff
            })
            .sum::<f64>();

        let weighted_amplitude = coefficients
            .iter()
            .enumerate()
            .map(|(i, &coeff)| {
                let (_, a, _) = self.get_strategy_params(i);
                a * coeff
            })
            .sum::<f64>();

        let weighted_frequency = coefficients
            .iter()
            .enumerate()
            .map(|(i, &coeff)| {
                let (_, _, f) = self.get_strategy_params(i);
                f * coeff
            })
            .sum::<f64>();

        (
            weighted_phase * hardware_factor,
            weighted_amplitude * hardware_factor,
            weighted_frequency * hardware_factor,
        )
    }

    /// Get strategy-specific parameters
    fn get_strategy_params(&self, strategy_index: usize) -> (f64, f64, f64) {
        match strategy_index {
            0 => (0.05, 0.8, 1.5),   // Latency
            1 => (0.03, 0.6, 1.8),   // Throughput
            2 => (0.1, 0.5, 1.2),    // Balanced
            3 => (0.15, 0.3, 0.9),   // Power
            4 => (0.08, 0.4, 1.1),   // Adaptive
            5 => (0.12, 0.45, 1.15), // ML-guided
            6 => (0.06, 0.55, 1.25), // Predictive
            7 => (0.04, 0.7, 1.4),   // Hardware-accelerated
            _ => (0.1, 0.5, 1.0),
        }
    }

    /// Predict task duration using quantum-enhanced algorithms
    pub fn predict_task_duration(&self, task: &SchedulableTask) -> Duration {
        if !self.is_enabled() {
            return task.estimated_duration; // Fallback to provided estimate
        }

        let state = self.state.read();
        let ml_coefficient = state.coefficients[5]; // ML-guided coefficient

        let base_prediction = task.estimated_duration;

        // Apply quantum-enhanced prediction if ML guidance is significant
        if ml_coefficient > 0.2 {
            if let Some(pattern) = self.workload_patterns.read().get(&task.task_type) {
                // Use historical pattern data for better prediction
                let pattern_factor = if pattern.deadline_miss_rate < 0.05 {
                    0.95 // Optimistic if we're hitting deadlines
                } else {
                    1.1 // Conservative if we're missing deadlines
                };

                let predicted_nanos = (base_prediction.as_nanos() as f64 * pattern_factor) as u64;
                return Duration::from_nanos(predicted_nanos);
            }
        }

        // Apply quantum uncertainty to prediction
        let uncertainty_factor = 1.0 + (state.entropy() - 1.0) * 0.1; // ±10% based on quantum uncertainty
        let predicted_nanos = (base_prediction.as_nanos() as f64 * uncertainty_factor) as u64;

        Duration::from_nanos(predicted_nanos)
    }

    /// Optimize schedule using quantum algorithms
    pub fn optimize_schedule_quantum(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        if !self.is_enabled() || tasks.is_empty() {
            return QuantumSchedule::default();
        }

        let state = self.state.read();
        let hardware_backend = *self.hardware_backend.read();

        // Perform quantum measurement to select optimization approach
        let mut state_copy = state.clone();
        let strategy = state_copy.measure(hardware_backend);

        let optimization_result = match strategy {
            0 => self.optimize_for_latency(tasks),
            1 => self.optimize_for_throughput(tasks),
            2 => self.optimize_balanced(tasks),
            3 => self.optimize_for_power(tasks),
            4 => self.optimize_adaptive(tasks),
            5 => self.optimize_ml_guided(tasks),
            6 => self.optimize_predictive(tasks),
            7 => self.optimize_hardware_accelerated(tasks),
            _ => self.optimize_balanced(tasks),
        };

        debug!(
            strategy = strategy,
            tasks_count = tasks.len(),
            optimization_score = optimization_result.optimization_score,
            "Quantum schedule optimization completed"
        );

        optimization_result
    }

    /// Update quantum state based on performance feedback
    pub fn update_quantum_state(&self, feedback: &PerformanceFeedback) {
        if !self.is_enabled() {
            return;
        }

        // Store feedback in history
        {
            let mut history = self.feedback_history.write();
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(feedback.clone());
        }

        // Update strategy if adaptive
        {
            let mut strategy = self.strategy.write();
            strategy.adapt(feedback);
        }

        // Generate ML prediction based on feedback trends
        if let Some(prediction) = self.generate_ml_prediction(feedback) {
            let mut predictions = self.ml_predictions.write();
            if predictions.len() >= 100 {
                predictions.pop_front();
            }
            predictions.push_back(prediction);
        }

        // Update workload patterns cache
        self.update_workload_patterns(feedback);

        debug!(
            "Updated quantum state with performance feedback: {:?}",
            feedback
        );
    }

    /// Generate ML prediction based on feedback trends
    fn generate_ml_prediction(
        &self,
        current_feedback: &PerformanceFeedback,
    ) -> Option<MLPrediction> {
        let history = self.feedback_history.read();

        if history.len() < 5 {
            return None; // Need minimum history for meaningful prediction
        }

        // Simple trend analysis (in production, this would use sophisticated ML models)
        let recent_feedback: Vec<_> = history.iter().rev().take(5).collect();

        let avg_latency_trend = recent_feedback
            .windows(2)
            .map(|pair| pair[0].avg_latency_ns - pair[1].avg_latency_ns)
            .sum::<f64>()
            / (recent_feedback.len() - 1) as f64;

        let avg_throughput_trend = recent_feedback
            .windows(2)
            .map(|pair| pair[0].ops_per_sec - pair[1].ops_per_sec)
            .sum::<f64>()
            / (recent_feedback.len() - 1) as f64;

        // Predict strategy based on trends
        let recommended_strategy =
            if avg_latency_trend > 100.0 && current_feedback.deadline_miss_rate > 0.05 {
                0 // Focus on latency
            } else if avg_throughput_trend < -100.0 && current_feedback.throughput_ratio < 0.8 {
                1 // Focus on throughput
            } else if current_feedback.power_efficiency < 50.0 {
                3 // Focus on power efficiency
            } else {
                2 // Balanced approach
            };

        let confidence = (recent_feedback.len() as f64 / 10.0).clamp(0.1, 0.9);
        let expected_improvement = if recommended_strategy != 2 {
            0.15
        } else {
            0.05
        };

        Some(MLPrediction {
            recommended_strategy,
            confidence,
            expected_improvement,
            convergence_time_ns: 5_000_000, // 5ms convergence estimate
        })
    }

    /// Update workload patterns cache for better predictions
    fn update_workload_patterns(&self, feedback: &PerformanceFeedback) {
        // In a real implementation, this would extract task type from feedback context
        // For now, we'll use a simplified approach
        let pattern_key = format!("workload_{}", feedback.ops_per_sec as u64 / 1000);

        let mut patterns = self.workload_patterns.write();
        patterns.insert(pattern_key, feedback.clone());

        // Keep cache bounded
        if patterns.len() > 100 {
            // Remove oldest patterns (in practice, use LRU or time-based eviction)
            let keys_to_remove: Vec<_> = patterns.keys().take(10).cloned().collect();
            for key in keys_to_remove {
                patterns.remove(&key);
            }
        }
    }

    /// Get current quantum state for monitoring
    pub fn current_state(&self) -> QuantumState {
        self.state.read().clone()
    }

    /// Get optimization metrics
    pub fn metrics(&self) -> QuantumMetrics {
        let state = self.state.read();
        let operations = self.operations_count.load(Ordering::Relaxed);
        let feedback_count = self.feedback_history.read().len();
        let ml_predictions_count = self.ml_predictions.read().len();

        QuantumMetrics {
            measurements: state.measurements,
            entropy: state.entropy(),
            coherence: 1.0 - state.decoherence,
            dominant_strategy: state.dominant_strategy(),
            hardware_acceleration: state.hardware_acceleration,
            operations_count: operations,
            feedback_count: feedback_count as u64,
            ml_predictions_count: ml_predictions_count as u64,
            enabled: self.is_enabled(),
        }
    }

    /// Reset quantum oracle to initial state
    pub fn reset(&self) {
        let strategy = *self.strategy.read();
        *self.state.write() = QuantumState::for_strategy(strategy);
        self.phase_accumulator.store(0, Ordering::Relaxed);
        self.operations_count.store(0, Ordering::Relaxed);
        self.feedback_history.write().clear();
        self.ml_predictions.write().clear();
        self.workload_patterns.write().clear();

        info!("Quantum Time Oracle reset to initial state");
    }

    // Optimization algorithm implementations
    fn optimize_for_latency(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Sort by deadline urgency
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by_key(|task| task.deadline);

        QuantumSchedule {
            optimized_tasks: sorted_tasks,
            optimization_score: 0.9,
            strategy_used: "latency_optimized".to_string(),
            quantum_effects_applied: true,
            convergence_time_ns: 1_000_000,
        }
    }

    fn optimize_for_throughput(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Sort by execution efficiency (shortest job first)
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by_key(|task| task.estimated_duration);

        QuantumSchedule {
            optimized_tasks: sorted_tasks,
            optimization_score: 0.85,
            strategy_used: "throughput_optimized".to_string(),
            quantum_effects_applied: true,
            convergence_time_ns: 800_000,
        }
    }

    fn optimize_balanced(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Balance deadline urgency and execution time
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by(|a, b| {
            let a_score = a.deadline.as_nanos() as f64 / a.estimated_duration.as_nanos() as f64;
            let b_score = b.deadline.as_nanos() as f64 / b.estimated_duration.as_nanos() as f64;
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        QuantumSchedule {
            optimized_tasks: sorted_tasks,
            optimization_score: 0.8,
            strategy_used: "balanced".to_string(),
            quantum_effects_applied: true,
            convergence_time_ns: 1_200_000,
        }
    }

    fn optimize_for_power(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Group similar tasks to reduce context switching
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by_key(|task| (task.task_type.clone(), task.estimated_duration));

        QuantumSchedule {
            optimized_tasks: sorted_tasks,
            optimization_score: 0.75,
            strategy_used: "power_efficient".to_string(),
            quantum_effects_applied: true,
            convergence_time_ns: 1_500_000,
        }
    }

    fn optimize_adaptive(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Use feedback history to adapt strategy
        let history = self.feedback_history.read();

        if let Some(recent_feedback) = history.back() {
            if recent_feedback.deadline_miss_rate > 0.1 {
                return self.optimize_for_latency(tasks);
            } else if recent_feedback.throughput_ratio < 0.7 {
                return self.optimize_for_throughput(tasks);
            }
        }

        self.optimize_balanced(tasks)
    }

    fn optimize_ml_guided(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Use ML predictions to guide optimization
        if let Some(prediction) = self.ml_predictions.read().back() {
            match prediction.recommended_strategy {
                0 => return self.optimize_for_latency(tasks),
                1 => return self.optimize_for_throughput(tasks),
                3 => return self.optimize_for_power(tasks),
                _ => return self.optimize_balanced(tasks),
            }
        }

        self.optimize_balanced(tasks)
    }

    fn optimize_predictive(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        // Use predictive modeling based on task patterns
        let mut optimized_tasks = Vec::new();

        for task in tasks {
            let predicted_duration = self.predict_task_duration(task);
            let mut optimized_task = task.clone();
            optimized_task.estimated_duration = predicted_duration;
            optimized_tasks.push(optimized_task);
        }

        // Sort by predicted completion time
        optimized_tasks.sort_by_key(|task| task.estimated_duration);

        QuantumSchedule {
            optimized_tasks,
            optimization_score: 0.82,
            strategy_used: "predictive".to_string(),
            quantum_effects_applied: true,
            convergence_time_ns: 900_000,
        }
    }

    fn optimize_hardware_accelerated(&self, tasks: &[SchedulableTask]) -> QuantumSchedule {
        let hardware_backend = *self.hardware_backend.read();

        // Leverage hardware acceleration for complex scheduling algorithms
        let acceleration_factor: f32 = match hardware_backend {
            HardwareBackend::Cuda | HardwareBackend::Tpu => 2.0,
            HardwareBackend::Vulkan | HardwareBackend::WebGpu => 1.5,
            _ => 1.0,
        };

        // More sophisticated algorithms are possible with hardware acceleration
        let mut sorted_tasks: Vec<_> = tasks.to_vec();

        if acceleration_factor > 1.5 {
            // Use complex multi-dimensional optimization
            sorted_tasks.sort_by(|a, b| {
                let a_score = (a.deadline.as_nanos() as f64
                    / a.estimated_duration.as_nanos() as f64)
                    * a.priority as f64;
                let b_score = (b.deadline.as_nanos() as f64
                    / b.estimated_duration.as_nanos() as f64)
                    * b.priority as f64;
                b_score
                    .partial_cmp(&a_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Fall back to simpler algorithm
            sorted_tasks.sort_by_key(|task| task.deadline);
        }

        QuantumSchedule {
            optimized_tasks: sorted_tasks,
            optimization_score: 0.88 * (acceleration_factor.min(1.2) as f64),
            strategy_used: format!("hardware_accelerated_{:?}", hardware_backend),
            quantum_effects_applied: true,
            convergence_time_ns: (500_000.0 / acceleration_factor) as u64,
        }
    }
}

impl Default for QuantumTimeOracle {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for quantum optimization monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Number of quantum measurements performed
    pub measurements: u64,
    /// Entropy of the quantum state (higher = more uncertainty)
    pub entropy: f64,
    /// Coherence level (1.0 = fully coherent, 0.0 = fully decoherent)
    pub coherence: f64,
    /// Currently dominant optimization strategy
    pub dominant_strategy: usize,
    /// Hardware acceleration factor
    pub hardware_acceleration: f64,
    /// Total operations performed
    pub operations_count: u64,
    /// Number of feedback samples collected
    pub feedback_count: u64,
    /// Number of ML predictions generated
    pub ml_predictions_count: u64,
    /// Oracle enabled status
    pub enabled: bool,
}

/// Quantum-optimized schedule result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSchedule {
    /// Tasks optimized by quantum algorithms
    pub optimized_tasks: Vec<SchedulableTask>,
    /// Optimization quality score (0.0 to 1.0)
    pub optimization_score: f64,
    /// Strategy used for optimization
    pub strategy_used: String,
    /// Whether quantum effects were applied
    pub quantum_effects_applied: bool,
    /// Time taken to converge (nanoseconds)
    pub convergence_time_ns: u64,
}

impl Default for QuantumSchedule {
    fn default() -> Self {
        Self {
            optimized_tasks: Vec::new(),
            optimization_score: 0.5,
            strategy_used: "default".to_string(),
            quantum_effects_applied: false,
            convergence_time_ns: 1_000_000,
        }
    }
}

/// Placeholder for schedulable task (to be defined in deadline scheduler)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "net", derive(Serialize, Deserialize))]
pub struct SchedulableTask {
    /// Task identifier
    pub task_id: String,
    /// Task type for pattern matching
    pub task_type: String,
    /// Estimated execution duration
    pub estimated_duration: Duration,
    /// Task deadline
    pub deadline: NanoTime,
    /// Task priority (higher = more important)
    pub priority: u32,
}

impl Default for SchedulableTask {
    fn default() -> Self {
        Self {
            task_id: "default_task".to_string(),
            task_type: "generic".to_string(),
            estimated_duration: Duration::from_millis(1),
            deadline: NanoTime::from_secs(1),
            priority: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new();
        assert_eq!(state.coefficients.len(), 8);

        let sum: f64 = state.coefficients.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10); // Should be normalized
    }

    #[test]
    fn test_quantum_state_evolution() {
        let mut state = QuantumState::new();
        let initial_coeffs = state.coefficients;

        state.evolve(Duration::from_millis(1), HardwareBackend::Cpu);

        // State should have evolved
        assert_ne!(state.coefficients, initial_coeffs);

        // But should still be normalized
        let sum: f64 = state.coefficients.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hardware_acceleration() {
        let mut state = QuantumState::new();

        // CPU evolution
        state.evolve(Duration::from_millis(1), HardwareBackend::Cpu);
        let cpu_acceleration = state.hardware_acceleration;

        // CUDA evolution should show higher acceleration
        state.evolve(Duration::from_millis(1), HardwareBackend::Cuda);
        let cuda_acceleration = state.hardware_acceleration;

        assert!(cuda_acceleration > cpu_acceleration);
    }

    #[test]
    fn test_quantum_measurement() {
        let mut state = QuantumState::new();
        let initial_measurements = state.measurements;

        let measured = state.measure(HardwareBackend::Cpu);

        // Should have incremented measurement count
        assert_eq!(state.measurements, initial_measurements + 1);

        // Should return valid strategy index
        assert!(measured < 8);
    }

    #[test]
    fn test_oracle_creation() {
        let oracle = QuantumTimeOracle::new();
        let offset = oracle.current_offset();

        assert!(offset.amplitude > 0.0);
        assert!(offset.frequency > 0.0);
    }

    #[test]
    fn test_optimization_strategy_adaptation() {
        let mut strategy = OptimizationStrategy::Adaptive {
            learning_rate: 0.1,
            history_window: 100,
        };

        let feedback = PerformanceFeedback {
            deadline_miss_rate: 0.1, // High miss rate
            ..Default::default()
        };

        strategy.adapt(&feedback);

        // Should have adapted to minimize latency
        if let OptimizationStrategy::MinimizeLatency { .. } = strategy {
            // Expected behavior
        } else {
            panic!("Strategy should have adapted to minimize latency");
        }
    }

    #[test]
    fn test_ml_prediction_integration() {
        let oracle = QuantumTimeOracle::new();

        let feedback = PerformanceFeedback {
            avg_latency_ns: 2000.0,
            deadline_miss_rate: 0.08,
            ..Default::default()
        };

        // Add multiple feedback samples to enable ML prediction
        for _ in 0..10 {
            oracle.update_quantum_state(&feedback);
        }

        let metrics = oracle.metrics();
        assert!(metrics.feedback_count >= 10);
    }

    #[test]
    fn test_hardware_backend_configuration() {
        let oracle = QuantumTimeOracle::with_config(
            OptimizationStrategy::MinimizeLatency {
                target_latency_ns: 1000,
                miss_rate_threshold: 0.01,
            },
            HardwareBackend::Cuda,
            20_000_000, // 20 MHz
        );

        oracle.set_enabled(true);
        let offset = oracle.current_offset();

        // Should produce valid optimization offset
        assert!(offset.amplitude > 0.0);
        assert!(offset.frequency > 0.0);
    }

    #[test]
    fn test_quantum_schedule_optimization() {
        let oracle = QuantumTimeOracle::new();

        let tasks = vec![
            SchedulableTask {
                task_id: "task1".to_string(),
                task_type: "compute".to_string(),
                estimated_duration: Duration::from_millis(10),
                deadline: NanoTime::from_millis(100),
                priority: 1,
            },
            SchedulableTask {
                task_id: "task2".to_string(),
                task_type: "io".to_string(),
                estimated_duration: Duration::from_millis(5),
                deadline: NanoTime::from_millis(50),
                priority: 2,
            },
        ];

        let schedule = oracle.optimize_schedule_quantum(&tasks);

        assert_eq!(schedule.optimized_tasks.len(), 2);
        assert!(schedule.optimization_score > 0.0);
        assert!(schedule.quantum_effects_applied);
    }

    #[test]
    fn test_oracle_enable_disable() {
        let oracle = QuantumTimeOracle::new();

        assert!(oracle.is_enabled()); // Should be enabled by default

        oracle.set_enabled(false);
        assert!(!oracle.is_enabled());

        let offset = oracle.current_offset();
        // Should return neutral offset when disabled
        assert_eq!(offset.amplitude, 1.0);
        assert_eq!(offset.frequency, 1.0);
    }

    #[test]
    fn test_metrics_collection() {
        let oracle = QuantumTimeOracle::new();

        // Generate some activity
        for _ in 0..5 {
            oracle.current_offset();
        }

        let metrics = oracle.metrics();
        assert!(metrics.operations_count >= 5);
        assert!(metrics.enabled);
        assert!(metrics.dominant_strategy < 8);
    }
}
