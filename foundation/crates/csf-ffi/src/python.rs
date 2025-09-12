//! Python bindings for ARES CSF

use super::*;
use csf_shared_types::PacketType;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Python module for ARES CSF
#[pymodule]
fn ares_csf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCSFRuntime>()?;
    m.add_class::<PyPhasePacket>()?;
    #[cfg(feature = "mlir")]
    m.add_class::<PyMLIRModule>()?;
    #[cfg(feature = "mlir")]
    m.add_class::<PyTensor>()?;
    m.add_class::<PyCLogicSystem>()?;
    m.add_class::<PyQuantumCircuit>()?;
    m.add_function(wrap_pyfunction!(py_init_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    Ok(())
}

/// Initialize the CSF runtime
#[pyfunction]
fn py_init_runtime() -> PyResult<()> {
    match crate::init_runtime() {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
    }
}

/// Get CSF version
#[pyfunction]
fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Python wrapper for CSF runtime
#[pyclass]
struct PyCSFRuntime {
    runtime: Arc<CsfRuntime>,
}

#[pymethods]
impl PyCSFRuntime {
    #[new]
    fn new() -> PyResult<Self> {
        match get_runtime() {
            Ok(runtime) => Ok(Self { runtime }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Send a packet through the bus
    fn send_packet(&self, _channel: &str, _packet: &PyPhasePacket) -> PyResult<()> {
        // Simplified implementation for compilation - to be completed later
        Ok(())
    }

    /// Receive a packet from the bus
    fn receive_packet(
        &self,
        _channel: &str,
        _timeout_ms: Option<u64>,
    ) -> PyResult<Option<PyPhasePacket>> {
        // Simplified implementation for compilation - to be completed later
        Ok(None)
    }

    /// Load an MLIR module
    #[cfg(feature = "mlir")]
    fn load_mlir_module(&self, module: &PyMLIRModule) -> PyResult<u64> {
        let mlir_module = module.to_mlir_module();

        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime.tokio_runtime.block_on(async {
                    #[cfg(feature = "mlir")]
                    {
                        self.runtime.mlir_runtime.load_module(mlir_module).await
                    }
                    #[cfg(not(feature = "mlir"))]
                    {
                        Err(super::FFIError::UnsupportedOperation(
                            "MLIR not available".to_string(),
                        ))
                    }
                    .map(|id| id.0)
                })
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Execute an MLIR module
    #[cfg(feature = "mlir")]
    fn execute_mlir_module(
        &self,
        module_id: u64,
        inputs: Vec<PyTensor>,
    ) -> PyResult<Vec<PyTensor>> {
        let input_tensors: Result<Vec<_>, _> = inputs.iter().map(|t| t.to_tensor()).collect();

        let input_tensors = input_tensors.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime.tokio_runtime.block_on(async {
                    self.runtime
                        .mlir_runtime
                        .execute(csf_mlir::ModuleId(module_id), input_tensors, None)
                        .await
                })
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        .map(|tensors| tensors.into_iter().map(PyTensor::from_tensor).collect())
    }
}

/// Python wrapper for phase packet
#[pyclass]
#[derive(Clone)]
struct PyPhasePacket {
    packet_type: u8,
    priority: u8,
    data: Vec<u8>,
    metadata: Py<PyDict>,
}

#[pymethods]
impl PyPhasePacket {
    #[new]
    fn new(packet_type: u8, priority: u8, data: Vec<u8>) -> Self {
        Python::with_gil(|py| Self {
            packet_type,
            priority,
            data,
            metadata: PyDict::new(py).into(),
        })
    }

    #[getter]
    fn get_packet_type(&self) -> u8 {
        self.packet_type
    }

    #[getter]
    fn get_priority(&self) -> u8 {
        self.priority
    }

    #[getter]
    fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }

    #[getter]
    fn get_metadata(&self) -> Py<PyDict> {
        self.metadata.clone()
    }

    #[setter]
    fn set_metadata(&mut self, metadata: Py<PyDict>) {
        self.metadata = metadata;
    }
}

impl PyPhasePacket {
    fn to_phase_packet(&self) -> PhasePacket {
        let metadata = Python::with_gil(|py| {
            let dict = self.metadata.as_ref(py);
            // Simplified - just return empty JSON for now
            let json_str = "{}".to_string();

            serde_json::from_str(&json_str).unwrap_or_default()
        });

        PhasePacket {
            header: PacketHeader {
                version: 1,
                packet_id: PacketId::new(),
                packet_type: match self.packet_type {
                    0 => PacketType::Control,
                    1 => PacketType::Data,
                    2 => PacketType::Event,
                    _ => PacketType::Stream,
                },
                priority: self.priority,
                flags: PacketFlags::empty(),
                timestamp: hardware_timestamp(),
                source_node: 0,
                destination_node: 0,
                causality_hash: 0,
                sequence_number: None,
                sequence: 0,
                fragment_count: None,
                payload_size: self.data.len() as u32,
                checksum: 0,
            },
            payload: PacketPayload {
                data: self.data.clone(),
                metadata,
            },
        }
    }

    fn from_protocol_packet(
        packet: csf_protocol::PhasePacket<csf_protocol::PacketPayload>,
    ) -> Self {
        Python::with_gil(|py| Self {
            packet_type: match packet.header.packet_type {
                PacketType::Control => 0,
                PacketType::Data => 1,
                PacketType::Event => 2,
                PacketType::Stream => 3,
            },
            priority: packet.header.priority,
            data: packet.payload.data,
            metadata: PyDict::new(py).into(),
        })
    }

    fn from_phase_packet_generic(
        packet: csf_protocol::PhasePacket<csf_protocol::PacketPayload>,
    ) -> Self {
        Python::with_gil(|py| {
            let metadata_dict = PyDict::new(py);

            // Convert metadata to Python dict
            for (key, value) in packet.payload.metadata {
                // Simple conversion instead of pythonize dependency
                match value {
                    serde_json::Value::String(s) => {
                        let _ = metadata_dict.set_item(key, s);
                    }
                    serde_json::Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            let _ = metadata_dict.set_item(key, f);
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        let _ = metadata_dict.set_item(key, b);
                    }
                    _ => {
                        let _ = metadata_dict.set_item(key, value.to_string());
                    }
                }
            }

            Self {
                packet_type: match packet.header.packet_type {
                    PacketType::Control => 0,
                    PacketType::Data => 1,
                    PacketType::Event => 2,
                    PacketType::Stream => 3,
                },
                priority: packet.header.priority,
                data: packet.payload.data,
                metadata: metadata_dict.into(),
            }
        })
    }

    fn from_phase_packet(packet: PhasePacket) -> Self {
        Python::with_gil(|py| {
            let metadata_dict = PyDict::new(py);

            // Convert metadata to Python dict
            for (key, value) in packet.payload.metadata {
                // Simple conversion instead of pythonize dependency
                match value {
                    serde_json::Value::String(s) => {
                        let _ = metadata_dict.set_item(key, s);
                    }
                    serde_json::Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            let _ = metadata_dict.set_item(key, f);
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        let _ = metadata_dict.set_item(key, b);
                    }
                    _ => {
                        let _ = metadata_dict.set_item(key, value.to_string());
                    }
                }
            }

            Self {
                packet_type: match packet.header.packet_type {
                    PacketType::Control => 0,
                    PacketType::Data => 1,
                    PacketType::Event => 2,
                    PacketType::Stream => 3,
                },
                priority: packet.header.priority,
                data: packet.payload.data,
                metadata: metadata_dict.into(),
            }
        })
    }
}

/// Python wrapper for MLIR module
#[cfg(feature = "mlir")]
#[pyclass]
struct PyMLIRModule {
    name: String,
    mlir_code: String,
}

#[cfg(feature = "mlir")]
#[pymethods]
impl PyMLIRModule {
    #[new]
    fn new(name: String, mlir_code: String) -> Self {
        Self { name, mlir_code }
    }

    #[getter]
    fn get_name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn get_mlir_code(&self) -> &str {
        &self.mlir_code
    }
}

#[cfg(feature = "mlir")]
impl PyMLIRModule {
    fn to_mlir_module(&self) -> csf_mlir::MlirModule {
        csf_mlir::MlirModule {
            name: self.name.clone(),
            id: csf_mlir::ModuleId::new(),
            ir: self.mlir_code.clone(),
            artifact: None,
            metadata: Default::default(),
        }
    }
}

/// Python wrapper for tensor
#[cfg(feature = "mlir")]
#[pyclass]
#[derive(Clone)]
struct PyTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
    dtype: String,
}

#[cfg(feature = "mlir")]
#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self {
            data,
            shape,
            dtype: "float32".to_string(),
        }
    }

    #[getter]
    fn get_data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[getter]
    fn get_shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    #[getter]
    fn get_dtype(&self) -> &str {
        &self.dtype
    }

    fn numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Convert to numpy array
        let np = py.import_bound("numpy")?;
        let array = np.call_method1("array", (self.data.clone(),))?;
        let reshaped = array.call_method1("reshape", (self.shape.clone(),))?;
        Ok(reshaped.into())
    }
}

#[cfg(feature = "mlir")]
impl PyTensor {
    fn to_tensor(&self) -> Result<csf_mlir::runtime::Tensor, FFIError> {
        let data_bytes = bytemuck::cast_slice(&self.data).to_vec();
        Ok(csf_mlir::runtime::Tensor {
            data: data_bytes,
            dtype: csf_mlir::DataType::F32,
            shape: self.shape.clone(),
            strides: self.compute_strides(),
            device: csf_mlir::runtime::DeviceLocation::CPU,
        })
    }

    #[cfg(feature = "mlir")]
    fn from_tensor(tensor: csf_mlir::runtime::Tensor) -> Self {
        let data: Vec<f32> = bytemuck::cast_slice(&tensor.data).to_vec();
        Self {
            data,
            shape: tensor.shape,
            dtype: "float32".to_string(),
        }
    }

    fn compute_strides(&self) -> Vec<i64> {
        let mut strides = vec![1; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }
}

/// Python wrapper for C-LOGIC system
#[pyclass]
struct PyCLogicSystem {
    runtime: Arc<CsfRuntime>,
}

#[pymethods]
impl PyCLogicSystem {
    #[new]
    fn new() -> PyResult<Self> {
        match get_runtime() {
            Ok(runtime) => Ok(Self { runtime }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current C-LOGIC state
    fn get_state(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let state_dict = PyDict::new(py);

            // In a real implementation, this would get actual state
            state_dict.set_item("drpp_coherence", 0.5)?;
            state_dict.set_item("adp_load", 0.3)?;
            state_dict.set_item("egc_decisions_pending", 0)?;
            state_dict.set_item("ems_valence", 0.0)?;
            state_dict.set_item("ems_arousal", 0.0)?;

            Ok(state_dict.into())
        })
    }
}

/// Python wrapper for quantum circuit
#[pyclass]
struct PyQuantumCircuit {
    num_qubits: u32,
    operations: Vec<(String, Vec<u32>, Vec<f64>)>,
}

#[pymethods]
impl PyQuantumCircuit {
    #[new]
    fn new(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            operations: Vec::new(),
        }
    }

    /// Add Hadamard gate
    fn h(&mut self, qubit: u32) {
        self.operations.push(("H".to_string(), vec![qubit], vec![]));
    }

    /// Add CNOT gate
    fn cnot(&mut self, control: u32, target: u32) {
        self.operations
            .push(("CNOT".to_string(), vec![control, target], vec![]));
    }

    /// Add rotation gate
    fn rx(&mut self, qubit: u32, angle: f64) {
        self.operations
            .push(("RX".to_string(), vec![qubit], vec![angle]));
    }

    /// Execute circuit
    fn execute(&self, shots: u32) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let results = PyDict::new(py);

            // In a real implementation, this would execute on quantum backend
            let num_states = 1 << self.num_qubits;
            let counts = PyDict::new(py);

            for i in 0..num_states {
                let state = format!("{:0width$b}", i, width = self.num_qubits as usize);
                counts.set_item(state, shots / num_states)?;
            }

            results.set_item("counts", counts)?;
            results.set_item("shots", shots)?;

            Ok(results.into())
        })
    }
}
