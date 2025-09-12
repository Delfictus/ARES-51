//! WebAssembly bindings for ARES CSF

use super::*;
use serde_wasm_bindgen::{from_value, to_value};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{console, Performance, Window};

// Global runtime instance for WASM
static mut WASM_RUNTIME: Option<WasmRuntime> = None;

struct WasmRuntime {
    bus: Arc<csf_bus::Bus>,
    performance: Performance,
}

/// Initialize CSF for WebAssembly
#[wasm_bindgen]
pub fn csf_wasm_init() -> Result<(), JsValue> {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize logger
    wasm_logger::init(wasm_logger::Config::default());

    // Get performance API
    let window = web_sys::window().ok_or("No window found")?;
    let performance = window.performance().ok_or("No performance API")?;

    // Create runtime
    let bus = Arc::new(csf_bus::Bus::new(Default::default()));

    unsafe {
        WASM_RUNTIME = Some(WasmRuntime { bus, performance });
    }

    console::log_1(&"CSF WASM initialized".into());
    Ok(())
}

/// Get current timestamp in nanoseconds
#[wasm_bindgen]
pub fn csf_wasm_timestamp() -> f64 {
    unsafe {
        match &WASM_RUNTIME {
            Some(runtime) => runtime.performance.now() * 1_000_000.0,
            None => 0.0,
        }
    }
}

/// Phase packet for WASM
#[wasm_bindgen]
pub struct WasmPhasePacket {
    packet_type: u8,
    priority: u8,
    data: Vec<u8>,
    metadata: JsValue,
}

#[wasm_bindgen]
impl WasmPhasePacket {
    #[wasm_bindgen(constructor)]
    pub fn new(packet_type: u8, priority: u8, data: Vec<u8>) -> Self {
        Self {
            packet_type,
            priority,
            data,
            metadata: JsValue::from(js_sys::Object::new()),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn packet_type(&self) -> u8 {
        self.packet_type
    }

    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> u8 {
        self.priority
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> JsValue {
        self.metadata.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_metadata(&mut self, metadata: JsValue) {
        self.metadata = metadata;
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();

        js_sys::Reflect::set(&obj, &"packet_type".into(), &self.packet_type.into())?;
        js_sys::Reflect::set(&obj, &"priority".into(), &self.priority.into())?;

        let data_array = js_sys::Uint8Array::from(&self.data[..]);
        js_sys::Reflect::set(&obj, &"data".into(), &data_array)?;
        js_sys::Reflect::set(&obj, &"metadata".into(), &self.metadata)?;

        Ok(obj.into())
    }
}

/// Tensor for WASM
#[wasm_bindgen]
pub struct WasmTensor {
    data: Vec<f32>,
    shape: Vec<i32>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<i32>) -> Self {
        Self { data, shape }
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<i32> {
        self.shape.clone()
    }

    /// Get number of elements
    pub fn numel(&self) -> i32 {
        self.shape.iter().product()
    }

    /// Reshape tensor
    pub fn reshape(&mut self, new_shape: Vec<i32>) -> Result<(), JsValue> {
        let new_numel: i32 = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err("Invalid shape".into());
        }
        self.shape = new_shape;
        Ok(())
    }
}

/// MLIR module for WASM
#[wasm_bindgen]
pub struct WasmMLIRModule {
    name: String,
    mlir_code: String,
    module_id: Option<u64>,
}

#[wasm_bindgen]
impl WasmMLIRModule {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, mlir_code: String) -> Self {
        Self {
            name,
            mlir_code,
            module_id: None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mlir_code(&self) -> String {
        self.mlir_code.clone()
    }

    /// Compile module (simulated for WASM)
    pub async fn compile(&mut self) -> Result<(), JsValue> {
        // In WASM, we can't actually compile MLIR
        // This would send to a server for compilation
        self.module_id = Some(rand::random::<u64>());
        console::log_1(&format!("Module {} compiled (simulated)", self.name).into());
        Ok(())
    }
}

/// Quantum circuit for WASM
#[wasm_bindgen]
pub struct WasmQuantumCircuit {
    num_qubits: u32,
    operations: Vec<String>,
}

#[wasm_bindgen]
impl WasmQuantumCircuit {
    #[wasm_bindgen(constructor)]
    pub fn new(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            operations: Vec::new(),
        }
    }

    /// Add Hadamard gate
    pub fn h(&mut self, qubit: u32) -> Result<(), JsValue> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }
        self.operations.push(format!("H {}", qubit));
        Ok(())
    }

    /// Add CNOT gate
    pub fn cnot(&mut self, control: u32, target: u32) -> Result<(), JsValue> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }
        self.operations.push(format!("CNOT {} {}", control, target));
        Ok(())
    }

    /// Add X rotation
    pub fn rx(&mut self, qubit: u32, angle: f64) -> Result<(), JsValue> {
        if qubit >= self.num_qubits {
            return Err("Qubit index out of range".into());
        }
        self.operations.push(format!("RX {} {}", qubit, angle));
        Ok(())
    }

    /// Get circuit as QASM string
    pub fn to_qasm(&self) -> String {
        let mut qasm = format!(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{}];\ncreg c[{}];\n",
            self.num_qubits, self.num_qubits
        );

        for op in &self.operations {
            qasm.push_str(&format!("{};\\n", op.to_lowercase()));
        }

        qasm
    }

    /// Simulate circuit (basic)
    pub fn simulate(&self) -> Result<JsValue, JsValue> {
        let result = js_sys::Object::new();

        // Simple simulation - return uniform distribution
        let counts = js_sys::Object::new();
        let num_states = 1 << self.num_qubits;

        for i in 0..num_states.min(16) {
            // Limit to 16 states for performance
            let state = format!("{:0width$b}", i, width = self.num_qubits as usize);
            js_sys::Reflect::set(&counts, &state.into(), &(1000 / num_states).into())?;
        }

        js_sys::Reflect::set(&result, &"counts".into(), &counts)?;
        js_sys::Reflect::set(&result, &"shots".into(), &1000.into())?;

        Ok(result.into())
    }
}

/// C-LOGIC state for WASM
#[wasm_bindgen]
pub struct WasmCLogicState {
    drpp_coherence: f64,
    adp_load: f64,
    egc_decisions: u32,
    ems_valence: f64,
    ems_arousal: f64,
}

#[wasm_bindgen]
impl WasmCLogicState {
    /// Get current C-LOGIC state (simulated)
    pub fn get_current() -> Self {
        Self {
            drpp_coherence: rand::random::<f64>(),
            adp_load: rand::random::<f64>() * 0.8,
            egc_decisions: rand::random::<u32>() % 10,
            ems_valence: (rand::random::<f64>() - 0.5) * 2.0,
            ems_arousal: (rand::random::<f64>() - 0.5) * 2.0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn drpp_coherence(&self) -> f64 {
        self.drpp_coherence
    }

    #[wasm_bindgen(getter)]
    pub fn adp_load(&self) -> f64 {
        self.adp_load
    }

    #[wasm_bindgen(getter)]
    pub fn egc_decisions(&self) -> u32 {
        self.egc_decisions
    }

    #[wasm_bindgen(getter)]
    pub fn ems_valence(&self) -> f64 {
        self.ems_valence
    }

    #[wasm_bindgen(getter)]
    pub fn ems_arousal(&self) -> f64 {
        self.ems_arousal
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();

        js_sys::Reflect::set(&obj, &"drpp_coherence".into(), &self.drpp_coherence.into())?;
        js_sys::Reflect::set(&obj, &"adp_load".into(), &self.adp_load.into())?;
        js_sys::Reflect::set(&obj, &"egc_decisions".into(), &self.egc_decisions.into())?;
        js_sys::Reflect::set(&obj, &"ems_valence".into(), &self.ems_valence.into())?;
        js_sys::Reflect::set(&obj, &"ems_arousal".into(), &self.ems_arousal.into())?;

        Ok(obj.into())
    }
}

/// Utility functions
#[wasm_bindgen]
pub fn csf_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn csf_wasm_create_buffer(size: usize) -> Vec<u8> {
    vec![0; size]
}

#[wasm_bindgen]
pub fn csf_wasm_hash_data(data: &[u8]) -> Vec<u8> {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize().as_bytes().to_vec()
}
