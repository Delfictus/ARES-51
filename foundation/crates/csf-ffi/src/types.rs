//! FFI type definitions

use csf_core::prelude::*;
use csf_shared_types::PacketType;
use std::collections::HashMap;
use std::os::raw::c_char;

// Conditional MLIR imports only when available
#[cfg(feature = "mlir")]
use csf_mlir::{DataType, MemoryLayout, Tensor};

/// FFI representation of a phase packet
#[repr(C)]
pub struct CSF_Packet {
    pub packet_id_high: u64,
    pub packet_id_low: u64,
    pub packet_type: u8,
    pub priority: u8,
    pub flags: u16,
    pub timestamp: u64,
    pub source_node: u16,
    pub destination_node: u16,
    pub causality_hash: u64,
    pub data: *mut u8,
    pub data_len: usize,
    pub metadata: *mut c_char, // JSON string
}

impl CSF_Packet {
    pub fn from(packet: PhasePacket) -> Self {
        let id_bytes = packet.header.packet_id.as_u128().to_be_bytes();
        let packet_id_high = u64::from_be_bytes(id_bytes[0..8].try_into().unwrap_or([0; 8]));
        let packet_id_low = u64::from_be_bytes(id_bytes[8..16].try_into().unwrap_or([0; 8]));

        let mut data = packet.payload.data;
        let data_ptr = data.as_mut_ptr();
        let data_len = data.len();
        std::mem::forget(data); // Prevent deallocation

        let metadata_str =
            serde_json::to_string(&packet.payload.metadata).unwrap_or_else(|_| "{}".to_string());
        let metadata = safe_cstring(metadata_str).into_raw();

        Self {
            packet_id_high,
            packet_id_low,
            packet_type: match packet.header.packet_type {
                PacketType::Control => 0,
                PacketType::Data => 1,
                PacketType::Event => 2,
                PacketType::Stream => 3,
            },
            priority: packet.header.priority,
            flags: packet.header.flags.bits() as u16,
            timestamp: packet.header.timestamp.as_nanos(),
            source_node: packet.header.source_node,
            destination_node: packet.header.destination_node,
            causality_hash: packet.header.causality_hash,
            data: data_ptr,
            data_len,
            metadata,
        }
    }

    pub fn to_phase_packet(&self) -> PhasePacket {
        let packet_id_bytes = [
            self.packet_id_high.to_be_bytes(),
            self.packet_id_low.to_be_bytes(),
        ]
        .concat();
        let packet_id = PacketId::new(); // Use a new UUID instead of reconstructing

        let data = if self.data.is_null() || self.data_len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.data_len).to_vec() }
        };

        let metadata = if self.metadata.is_null() {
            serde_json::Map::new()
        } else {
            unsafe {
                let c_str = std::ffi::CStr::from_ptr(self.metadata);
                let json_str = c_str.to_str().unwrap_or("{}");
                serde_json::from_str(json_str).unwrap_or_default()
            }
        };

        PhasePacket {
            header: PacketHeader {
                version: 1, // Default protocol version
                packet_id,
                packet_type: match self.packet_type {
                    0 => PacketType::Control,
                    1 => PacketType::Data,
                    2 => PacketType::Event,
                    _ => PacketType::Stream,
                },
                priority: self.priority,
                flags: PacketFlags::from_bits_truncate(self.flags.into()),
                timestamp: NanoTime::from_nanos(self.timestamp),
                source_node: self.source_node,
                destination_node: self.destination_node,
                causality_hash: self.causality_hash,
                sequence_number: None,
                sequence: 0,
                fragment_count: None,
                payload_size: self.data_len as u32,
                checksum: 0, // Will be calculated if needed
            },
            payload: PacketPayload {
                data,
                metadata: HashMap::from_iter(metadata.into_iter()),
            },
        }
    }
}

/// FFI representation of a tensor
#[cfg(feature = "mlir")]
#[repr(C)]
pub struct CSF_Tensor {
    pub data: *mut u8,
    pub data_len: usize,
    pub dtype: u8,
    pub shape: *mut i64,
    pub shape_len: usize,
    pub strides: *mut i64,
    pub strides_len: usize,
    pub layout: u8,
}

#[cfg(feature = "mlir")]
impl CSF_Tensor {
    pub fn from_tensor(tensor: Tensor) -> Self {
        let mut data = tensor.data;
        let data_ptr = data.as_mut_ptr();
        let data_len = data.len();
        std::mem::forget(data);

        let mut shape = tensor.shape;
        let shape_ptr = shape.as_mut_ptr();
        let shape_len = shape.len();
        std::mem::forget(shape);

        let mut strides = tensor.strides;
        let strides_ptr = strides.as_mut_ptr();
        let strides_len = strides.len();
        std::mem::forget(strides);

        Self {
            data: data_ptr,
            data_len,
            dtype: dtype_to_u8(tensor.dtype),
            shape: shape_ptr,
            shape_len,
            strides: strides_ptr,
            strides_len,
            layout: match tensor.device {
                csf_mlir::runtime::DeviceLocation::CPU => 0,
                csf_mlir::runtime::DeviceLocation::GPU(_) => 1,
                csf_mlir::runtime::DeviceLocation::TPU(_) => 2,
            },
        }
    }

    pub fn to_tensor(&self) -> Result<Tensor, super::FFIError> {
        if self.data.is_null() || self.shape.is_null() || self.strides.is_null() {
            return Err(super::FFIError::NullPointer);
        }

        let data = unsafe { std::slice::from_raw_parts(self.data, self.data_len).to_vec() };

        let shape = unsafe { std::slice::from_raw_parts(self.shape, self.shape_len).to_vec() };

        let strides =
            unsafe { std::slice::from_raw_parts(self.strides, self.strides_len).to_vec() };

        Ok(Tensor {
            data,
            dtype: u8_to_dtype(self.dtype)?,
            shape,
            strides,
            device: match self.layout {
                0 => csf_mlir::runtime::DeviceLocation::CPU,
                1 => csf_mlir::runtime::DeviceLocation::GPU(0),
                2 => csf_mlir::runtime::DeviceLocation::TPU(0),
                _ => {
                    return Err(super::FFIError::InvalidArgument(
                        "Invalid device location".to_string(),
                    ))
                }
            },
        })
    }
}

/// FFI representation of an MLIR module
#[cfg(feature = "mlir")]
#[repr(C)]
pub struct CSF_MLIRModule {
    pub module_id: u64,
    pub name: *mut c_char,
    pub mlir_code: *mut c_char,
    pub is_compiled: bool,
}

#[cfg(feature = "mlir")]
impl CSF_MLIRModule {
    pub fn from(module: csf_mlir::MlirModule) -> Self {
        let name = safe_cstring(module.name).into_raw();
        let mlir_code = safe_cstring(module.ir).into_raw();

        Self {
            module_id: module.id.0,
            name,
            mlir_code,
            is_compiled: module.artifact.is_some(),
        }
    }

    pub fn to_mlir_module(&self) -> csf_mlir::MlirModule {
        let name = unsafe {
            std::ffi::CStr::from_ptr(self.name)
                .to_str()
                .unwrap_or("")
                .to_string()
        };

        let ir = unsafe {
            std::ffi::CStr::from_ptr(self.mlir_code)
                .to_str()
                .unwrap_or("")
                .to_string()
        };

        csf_mlir::MlirModule {
            name,
            id: csf_mlir::ModuleId(self.module_id),
            ir,
            artifact: None,
            metadata: Default::default(),
        }
    }
}

/// FFI representation of C-LOGIC state
#[repr(C)]
pub struct CSF_CLogicState {
    pub drpp_coherence: f64,
    pub adp_load: f64,
    pub egc_decisions_pending: u32,
    pub ems_valence: f64,
    pub ems_arousal: f64,
    pub timestamp: u64,
}

/// FFI representation of a quantum circuit
#[repr(C)]
pub struct CSF_QuantumCircuit {
    pub num_qubits: u32,
    pub operations: *mut CSF_QuantumOp,
    pub num_operations: usize,
    pub measurements: *mut u32,
    pub num_measurements: usize,
}

/// FFI representation of a quantum operation
#[repr(C)]
pub struct CSF_QuantumOp {
    pub op_type: u8,
    pub qubit1: u32,
    pub qubit2: u32,
    pub param1: f64,
    pub param2: f64,
    pub param3: f64,
}

// Helper functions

/// Create a CString from arbitrary input by replacing interior NULs with a safe placeholder.
/// This ensures we never panic across the FFI boundary due to NUL bytes.
fn safe_cstring<S: Into<String>>(s: S) -> std::ffi::CString {
    let mut owned = s.into();
    if owned.as_bytes().contains(&0) {
        owned = owned.replace('\0', "");
    }
    match std::ffi::CString::new(owned) {
        Ok(s) => s,
        Err(_) => unsafe { std::ffi::CString::from_vec_unchecked(Vec::new()) },
    }
}

#[cfg(feature = "mlir")]
fn dtype_to_u8(dtype: DataType) -> u8 {
    match dtype {
        DataType::F16 => 0,
        DataType::F32 => 1,
        DataType::F64 => 2,
        DataType::I8 => 3,
        DataType::I16 => 4,
        DataType::I32 => 5,
        DataType::I64 => 6,
        DataType::U8 => 7,
        DataType::U16 => 8,
        DataType::U32 => 9,
        DataType::U64 => 10,
        DataType::Bool => 11,
        DataType::Complex64 => 12,
        DataType::Complex128 => 13,
    }
}

#[cfg(feature = "mlir")]
fn u8_to_dtype(dtype: u8) -> Result<DataType, super::FFIError> {
    Ok(match dtype {
        0 => DataType::F16,
        1 => DataType::F32,
        2 => DataType::F64,
        3 => DataType::I8,
        4 => DataType::I16,
        5 => DataType::I32,
        6 => DataType::I64,
        7 => DataType::U8,
        8 => DataType::U16,
        9 => DataType::U32,
        10 => DataType::U64,
        11 => DataType::Bool,
        12 => DataType::Complex64,
        13 => DataType::Complex128,
        _ => {
            return Err(super::FFIError::InvalidArgument(
                "Invalid data type".to_string(),
            ))
        }
    })
}
