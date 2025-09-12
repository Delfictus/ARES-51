//! Brian2 Python-Rust Bridge
//! 
//! Integrates Brian2 spiking neural network simulator via PyO3

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, IntoPyDict};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use std::collections::HashMap;
use anyhow::Result;
use parking_lot::RwLock;

use crate::spike_encoding::Spike;

/// Network configuration
#[derive(Clone)]
pub struct NetworkConfig {
    pub num_excitatory: usize,
    pub num_inhibitory: usize,
    pub spectral_radius: f32,
    pub tau_exc: f32,  // Excitatory time constant (ms)
    pub tau_inh: f32,  // Inhibitory time constant (ms)
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            num_excitatory: 8000,
            num_inhibitory: 2000,
            spectral_radius: 0.95,
            tau_exc: 5.0,
            tau_inh: 10.0,
        }
    }
}

/// Runtime mode for Brian2
#[derive(Clone, Debug)]
pub enum RuntimeMode {
    CPU,
    CUDA,
    OpenCL,
}

/// Output spike from Brian2
#[derive(Clone, Debug)]
pub struct OutputSpike {
    pub timestamp_ns: u64,
    pub neuron_id: u32,
    pub neuron_type: NeuronType,
}

#[derive(Clone, Debug)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
}

/// Brian2 network bridge
pub struct Brian2Bridge {
    network: Option<PyObject>,
    neurons_exc: Option<PyObject>,
    neurons_inh: Option<PyObject>,
    synapses: Option<PyObject>,
    spike_monitor: Option<PyObject>,
    state_monitor: Option<PyObject>,
    runtime_mode: RuntimeMode,
    config: NetworkConfig,
}

impl Brian2Bridge {
    /// Initialize Brian2 bridge
    pub fn initialize() -> Result<Self> {
        Python::with_gil(|py| {
            // Import Brian2
            let brian2 = py.import_bound("brian2").map_err(|e| {
                anyhow::anyhow!("Failed to import Brian2: {}. Install with: pip install brian2", e)
            })?;
            
            // Set Brian2 preferences
            py.run_bound(r#"
import brian2 as b2
import numpy as np

# Set preferences for performance
b2.prefs.codegen.target = 'numpy'  # Use numpy for CPU, 'cython' for more speed
b2.prefs.core.default_float_dtype = np.float32
b2.prefs.core.network.default_schedule = ['start', 'groups', 'synapses', 'thresholds', 'resets', 'end']

# Suppress Brian2 warnings during initialization
import warnings
warnings.filterwarnings('ignore', module='brian2')
            "#, None, None)?;
            
            Ok(Self {
                network: None,
                neurons_exc: None,
                neurons_inh: None,
                synapses: None,
                spike_monitor: None,
                state_monitor: None,
                runtime_mode: RuntimeMode::CPU,
                config: NetworkConfig::default(),
            })
        })
    }
    
    /// Create trading-optimized neural network
    pub fn create_trading_network(&mut self, config: NetworkConfig) -> Result<()> {
        self.config = config.clone();
        
        Python::with_gil(|py| {
            let code = format!(r#"
import brian2 as b2
import numpy as np

# Hodgkin-Huxley equations for biological realism
eqs_hh = '''
dv/dt = (gL*(EL-v) + I_syn + I_ext - gNa*m**3*h*(v-ENa) - gK*n**4*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m) - beta_m*m : 1
dh/dt = alpha_h*(1-h) - beta_h*h : 1
dn/dt = alpha_n*(1-n) - beta_n*n : 1

I_syn = ge*(Ee-v) + gi*(Ei-v) : amp
dge/dt = -ge/tau_e : siemens
dgi/dt = -gi/tau_i : siemens

# HH rate functions
alpha_m = 0.1*(mV**-1)*(-v+25*mV)/(exp((-v+25*mV)/(10*mV))-1)/ms : Hz
beta_m = 4*exp(-v/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
beta_h = 1/(exp((-v+30*mV)/(10*mV))+1)/ms : Hz
alpha_n = 0.01*(mV**-1)*(-v+10*mV)/(exp((-v+10*mV)/(10*mV))-1)/ms : Hz
beta_n = 0.125*exp(-v/(80*mV))/ms : Hz

# Parameters
gNa : siemens (constant)
gK : siemens (constant)
gL : siemens (constant)
ENa : volt (constant)
EK : volt (constant)
EL : volt (constant)
Cm : farad (constant)
Ee : volt (constant)
Ei : volt (constant)
tau_e : second (constant)
tau_i : second (constant)
I_ext : amp
'''

# Create neuron groups
neurons_exc = b2.NeuronGroup({num_exc}, eqs_hh, 
    threshold='v > -20*mV',
    refractory='v > -40*mV',
    method='exponential_euler',
    name='exc_neurons')

neurons_inh = b2.NeuronGroup({num_inh}, eqs_hh,
    threshold='v > -20*mV',
    refractory='v > -40*mV',
    method='exponential_euler',
    name='inh_neurons')

# Set parameters
neurons_exc.gNa = 120*msiemens
neurons_exc.gK = 36*msiemens
neurons_exc.gL = 0.3*msiemens
neurons_exc.ENa = 50*mV
neurons_exc.EK = -77*mV
neurons_exc.EL = -54.4*mV
neurons_exc.Cm = 1*ufarad
neurons_exc.Ee = 0*mV
neurons_exc.Ei = -80*mV
neurons_exc.tau_e = {tau_e}*ms
neurons_exc.tau_i = {tau_i}*ms
neurons_exc.v = -65*mV
neurons_exc.I_ext = 0*amp

neurons_inh.gNa = 120*msiemens
neurons_inh.gK = 36*msiemens
neurons_inh.gL = 0.3*msiemens
neurons_inh.ENa = 50*mV
neurons_inh.EK = -77*mV
neurons_inh.EL = -54.4*mV
neurons_inh.Cm = 1*ufarad
neurons_inh.Ee = 0*mV
neurons_inh.Ei = -80*mV
neurons_inh.tau_e = {tau_e}*ms
neurons_inh.tau_i = {tau_i}*ms
neurons_inh.v = -65*mV
neurons_inh.I_ext = 0*amp

# STDP synapses
stdp_model = '''
w : volt (shared)
dApre/dt = -Apre/tau_pre : 1 (event-driven)
dApost/dt = -Apost/tau_post : 1 (event-driven)
tau_pre : second (shared, constant)
tau_post : second (shared, constant)
dApre_param : 1 (shared, constant)
dApost_param : 1 (shared, constant)
wmax : volt (shared, constant)
'''

stdp_on_pre = '''
ge += w
Apre += dApre_param
w = clip(w + Apost*wmax*0.01, 0*mV, wmax)
'''

stdp_on_post = '''
Apost += dApost_param
w = clip(w + Apre*wmax*0.01, 0*mV, wmax)
'''

# Create synapses
syn_ee = b2.Synapses(neurons_exc, neurons_exc,
    model=stdp_model,
    on_pre=stdp_on_pre,
    on_post=stdp_on_post,
    name='syn_ee')

# Connect with probability
syn_ee.connect(p=0.1)

# Set synapse parameters
syn_ee.w = 0.5*mV
syn_ee.tau_pre = 20*ms
syn_ee.tau_post = 20*ms
syn_ee.dApre_param = 0.01
syn_ee.dApost_param = -0.012
syn_ee.wmax = 2*mV

# Create monitors
spike_mon = b2.SpikeMonitor(neurons_exc, name='spike_mon')
state_mon = b2.StateMonitor(neurons_exc, 'v', record=[0, 1, 2], name='state_mon')

# Create network
network = b2.Network(neurons_exc, neurons_inh, syn_ee, spike_mon, state_mon)

# Store network components globally for access
globals()['network'] = network
globals()['neurons_exc'] = neurons_exc
globals()['neurons_inh'] = neurons_inh
globals()['syn_ee'] = syn_ee
globals()['spike_mon'] = spike_mon
globals()['state_mon'] = state_mon
"#, 
                num_exc = config.num_excitatory,
                num_inh = config.num_inhibitory,
                tau_e = config.tau_exc,
                tau_i = config.tau_inh
            );
            
            let globals = PyDict::new_bound(py);
            py.run_bound(&code, Some(&globals), Some(&globals))?;
            
            // Store references to network components
            self.network = Some(globals.get_item("network")?.unwrap().to_object(py));
            self.neurons_exc = Some(globals.get_item("neurons_exc")?.unwrap().to_object(py));
            self.neurons_inh = Some(globals.get_item("neurons_inh")?.unwrap().to_object(py));
            self.synapses = Some(globals.get_item("syn_ee")?.unwrap().to_object(py));
            self.spike_monitor = Some(globals.get_item("spike_mon")?.unwrap().to_object(py));
            self.state_monitor = Some(globals.get_item("state_mon")?.unwrap().to_object(py));
            
            Ok(())
        })
    }
    
    /// Inject spikes into the network
    pub fn inject_spikes(&mut self, spikes: Vec<Spike>) -> Result<()> {
        if spikes.is_empty() {
            return Ok(());
        }
        
        Python::with_gil(|py| {
            // Convert spikes to numpy arrays
            let times: Vec<f64> = spikes.iter()
                .map(|s| s.timestamp_ns as f64 / 1e9)
                .collect();
            let indices: Vec<i32> = spikes.iter()
                .map(|s| s.neuron_id as i32)
                .collect();
            
            let py_times = times.into_pyarray_bound(py);
            let py_indices = indices.into_pyarray_bound(py);
            
            // Create SpikeGeneratorGroup and connect
            let code = r#"
import brian2 as b2

# Create spike generator
spike_gen = b2.SpikeGeneratorGroup(N, indices, times*b2.second, name='spike_gen')

# Connect to excitatory neurons
spike_syn = b2.Synapses(spike_gen, neurons_exc, on_pre='ge += 1*mV', name='spike_syn')
spike_syn.connect(j='i')

# Add to network
network.add(spike_gen)
network.add(spike_syn)
"#;
            
            let locals = [
                ("indices", py_indices.to_object(py)),
                ("times", py_times.to_object(py)),
                ("N", (self.config.num_excitatory as i32).to_object(py)),
                ("network", self.network.as_ref().unwrap().clone_ref(py)),
                ("neurons_exc", self.neurons_exc.as_ref().unwrap().clone_ref(py)),
            ].into_py_dict_bound(py);
            
            py.run_bound(code, None, Some(&locals))?;
            
            Ok(())
        })
    }
    
    /// Run simulation for specified duration
    pub fn run(&mut self, duration_ms: f64) -> Result<Vec<OutputSpike>> {
        Python::with_gil(|py| {
            // Run the simulation
            let code = format!("network.run({}*b2.ms)", duration_ms);
            
            let locals = [
                ("network", self.network.as_ref().unwrap().clone_ref(py)),
            ].into_py_dict_bound(py);
            
            py.run_bound(&code, None, Some(&locals))?;
            
            // Extract spikes from monitor
            let output_spikes = self.extract_spikes(py)?;
            
            Ok(output_spikes)
        })
    }
    
    /// Process spikes through the network
    pub async fn process(&mut self, spikes: Vec<Spike>) -> Result<Vec<Spike>> {
        // Inject spikes
        self.inject_spikes(spikes)?;
        
        // Run simulation
        let output_spikes = self.run(10.0)?; // 10ms simulation
        
        // Convert back to Spike format
        let result_spikes: Vec<Spike> = output_spikes.iter()
            .map(|s| Spike {
                timestamp_ns: s.timestamp_ns,
                neuron_id: s.neuron_id,
                strength: 1.0, // Default strength
            })
            .collect();
        
        Ok(result_spikes)
    }
    
    /// Extract spikes from the spike monitor
    fn extract_spikes(&self, py: Python) -> Result<Vec<OutputSpike>> {
        let spike_mon = self.spike_monitor.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Spike monitor not initialized"))?;
        
        // Get spike times and indices from Brian2
        let times_obj = spike_mon.getattr(py, "t")?;
        let indices_obj = spike_mon.getattr(py, "i")?;
        
        // Convert Brian2 quantities to numpy arrays
        // Brian2 returns Quantity objects, we need to extract the values
        let code = r#"
import numpy as np

# Extract values from Brian2 Quantity objects
if hasattr(times, 'flatten'):
    times_array = np.array(times.flatten())
else:
    times_array = np.array(times)
    
if hasattr(indices, 'flatten'):
    indices_array = np.array(indices.flatten(), dtype=np.int32)
else:
    indices_array = np.array(indices, dtype=np.int32)

# Convert times to nanoseconds (Brian2 uses seconds)
times_ns = (times_array * 1e9).astype(np.uint64)
"#;
        
        let locals = [
            ("times", times_obj),
            ("indices", indices_obj),
        ].into_py_dict_bound(py);
        
        py.run_bound(code, None, Some(&locals))?;
        
        // Get the processed arrays
        let times_ns_array: &Bound<PyArray1<u64>> = locals
            .get_item("times_ns")
            .map_err(|e| anyhow::anyhow!("Failed to get times_ns: {}", e))?
            .ok_or_else(|| anyhow::anyhow!("Failed to get times_ns array"))?
            .downcast()
            .map_err(|e| anyhow::anyhow!("Failed to downcast times_ns: {:?}", e))?;
        
        let indices_array: &Bound<PyArray1<i32>> = locals
            .get_item("indices_array")
            .map_err(|e| anyhow::anyhow!("Failed to get indices_array: {}", e))?
            .ok_or_else(|| anyhow::anyhow!("Failed to get indices array"))?
            .downcast()
            .map_err(|e| anyhow::anyhow!("Failed to downcast indices: {:?}", e))?;
        
        // Convert numpy arrays to Rust vectors
        // In PyO3 0.24, we use try_readonly() or extract()
        let times_vec: Vec<u64> = times_ns_array.extract()?;
        let indices_vec: Vec<i32> = indices_array.extract()?;
        
        // Create output spikes
        let mut output_spikes = Vec::with_capacity(times_vec.len());
        for (i, (&time_ns, &neuron_idx)) in times_vec.iter().zip(indices_vec.iter()).enumerate() {
            // Determine neuron type based on index
            let neuron_type = if (neuron_idx as usize) < self.config.num_excitatory {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };
            
            output_spikes.push(OutputSpike {
                timestamp_ns: time_ns,
                neuron_id: neuron_idx as u32,
                neuron_type,
            });
        }
        
        // Sort by timestamp for consistency
        output_spikes.sort_by_key(|s| s.timestamp_ns);
        
        Ok(output_spikes)
    }
    
    /// Save network state
    pub fn save_state(&self, path: &str) -> Result<()> {
        Python::with_gil(|py| {
            let code = format!(r#"
import pickle

# Save network state
with open('{}', 'wb') as f:
    state = {{
        'network': network,
        'neurons_exc': neurons_exc,
        'neurons_inh': neurons_inh,
        'synapses': syn_ee,
    }}
    pickle.dump(state, f)
"#, path);
            
            let locals = [
                ("network", self.network.as_ref().unwrap().clone_ref(py)),
                ("neurons_exc", self.neurons_exc.as_ref().unwrap().clone_ref(py)),
                ("neurons_inh", self.neurons_inh.as_ref().unwrap().clone_ref(py)),
                ("syn_ee", self.synapses.as_ref().unwrap().clone_ref(py)),
            ].into_py_dict_bound(py);
            
            py.run_bound(&code, None, Some(&locals))?;
            
            Ok(())
        })
    }
    
    /// Load network state
    pub fn load_state(&mut self, path: &str) -> Result<()> {
        Python::with_gil(|py| {
            let code = format!(r#"
import pickle

# Load network state
with open('{}', 'rb') as f:
    state = pickle.load(f)
    
globals()['network'] = state['network']
globals()['neurons_exc'] = state['neurons_exc']
globals()['neurons_inh'] = state['neurons_inh']
globals()['syn_ee'] = state['synapses']
"#, path);
            
            let globals = PyDict::new_bound(py);
            py.run_bound(&code, Some(&globals), Some(&globals))?;
            
            // Update references
            self.network = Some(globals.get_item("network")?.unwrap().to_object(py));
            self.neurons_exc = Some(globals.get_item("neurons_exc")?.unwrap().to_object(py));
            self.neurons_inh = Some(globals.get_item("neurons_inh")?.unwrap().to_object(py));
            self.synapses = Some(globals.get_item("syn_ee")?.unwrap().to_object(py));
            
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires Brian2 installation
    fn test_brian2_initialization() {
        let result = Brian2Bridge::initialize();
        assert!(result.is_ok(), "Brian2 should initialize if installed");
    }
    
    #[test]
    #[ignore] // Requires Brian2 installation
    fn test_network_creation() {
        let mut bridge = Brian2Bridge::initialize().unwrap();
        let config = NetworkConfig {
            num_excitatory: 100,
            num_inhibitory: 25,
            spectral_radius: 0.95,
            tau_exc: 5.0,
            tau_inh: 10.0,
        };
        
        let result = bridge.create_trading_network(config);
        assert!(result.is_ok());
    }
}