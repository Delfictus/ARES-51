//! Stub Python bridge used when building with `native-only` feature.
//! Provides minimal structs and methods to satisfy the backend without
//! requiring PyO3 or Python at compile/runtime.

use anyhow::Result;
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct StdpConfig {
    pub tau_pre_ms: f64,
    pub tau_post_ms: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub w_min: f64,
    pub w_max: f64,
    pub w_init_mean: f64,
    pub w_init_std: f64,
    pub heterosynaptic_scaling: bool,
    pub homeostatic_plasticity: bool,
}

#[derive(Debug, Clone)]
pub struct Brian2NetworkConfig {
    pub input_neurons: usize,
    pub hidden_neurons: usize,
    pub output_neurons: usize,
    pub dt_ms: f64,
    pub simulation_duration_ms: f64,
    pub threshold_mv: f64,
    pub reset_mv: f64,
    pub refractory_ms: f64,
    pub membrane_resistance: f64,
    pub membrane_capacitance: f64,
    pub stdp_config: StdpConfig,
    pub device: String,
    pub code_generation: bool,
    pub optimization_level: String,
}

#[derive(Debug)]
pub struct Brian2PythonBridge {
    _cfg: Brian2NetworkConfig,
}

impl Brian2PythonBridge {
    pub fn new(cfg: Brian2NetworkConfig) -> Result<Self> { Ok(Self { _cfg: cfg }) }
    pub fn initialize_network(&mut self) -> Result<()> { Ok(()) }
    pub fn process_spikes(&mut self, spikes: Array1<f64>) -> Result<Array1<f64>> { Ok(spikes) }
    pub fn train_pattern(&mut self, _input: &Array1<f64>, _target: &Array1<f64>) -> Result<()> { Ok(()) }
}

#[derive(Debug, Clone)]
pub struct LavaProcessConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub neuron_model: String,
    pub connection_type: String,
    pub target_hardware: bool,
    pub weight_precision: String,
}

#[derive(Debug)]
pub struct LavaPythonBridge {
    _cfg: LavaProcessConfig,
}

impl LavaPythonBridge {
    pub fn new(cfg: LavaProcessConfig) -> Result<Self> { Ok(Self { _cfg: cfg }) }
    pub fn initialize_process(&mut self) -> Result<()> { Ok(()) }
    pub fn process_spikes(&mut self, spikes: Array1<f64>) -> Result<Array1<f64>> { Ok(spikes) }
}

