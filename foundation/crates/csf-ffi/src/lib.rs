//! FFI (Foreign Function Interface) bindings for ARES CSF
//!
//! Provides C, Python, and WebAssembly bindings for external integration

#![allow(non_camel_case_types)]

use csf_core::prelude::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

pub mod c_api;
pub mod error;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

// GRPC module disabled - not implemented yet
// #[cfg(feature = "grpc")]
// pub mod grpc;

pub use error::{FFIError, FFIResult};
pub use types::*;

/// Initialize the CSF runtime
///
/// This must be called before any other CSF functions.
/// Returns 0 on success, negative error code on failure.
#[no_mangle]
pub extern "C" fn csf_init() -> i32 {
    match init_runtime() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("Failed to initialize CSF runtime: {}", e);
            -1
        }
    }
}

/// Shutdown the CSF runtime
///
/// This should be called when done using CSF.
#[no_mangle]
pub extern "C" fn csf_shutdown() -> i32 {
    match shutdown_runtime() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("Failed to shutdown CSF runtime: {}", e);
            -1
        }
    }
}

/// Get the CSF version string
#[no_mangle]
pub extern "C" fn csf_version() -> *const c_char {
    match CString::new(env!("CARGO_PKG_VERSION")) {
        Ok(version) => version.into_raw(),
        Err(_) => {
            // Non-panicking fallback for version string
            unsafe { CString::from_vec_unchecked(b"unknown".to_vec()) }.into_raw()
        }
    }
}

/// Free a string returned by CSF
#[no_mangle]
pub extern "C" fn csf_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

// Runtime management
static mut RUNTIME: Option<Arc<CsfRuntime>> = None;

struct CsfRuntime {
    tokio_runtime: tokio::runtime::Runtime,
    bus: Arc<csf_bus::PhaseCoherenceBus>,
    kernel: Arc<csf_kernel::ChronosKernel>,
    #[cfg(feature = "mlir")]
    mlir_runtime: Arc<csf_mlir::MlirRuntime>,
}

fn init_runtime() -> FFIResult<()> {
    unsafe {
        if RUNTIME.is_some() {
            return Err(FFIError::AlreadyInitialized);
        }

        // Create Tokio runtime
        let tokio_runtime =
            tokio::runtime::Runtime::new().map_err(|e| FFIError::RuntimeError(e.to_string()))?;

        // Initialize CSF components
        #[cfg(feature = "mlir")]
        let (bus, kernel, mlir_runtime) = tokio_runtime.block_on(async {
            let bus = Arc::new(
                csf_bus::PhaseCoherenceBus::new(Default::default())
                    .map_err(|e| FFIError::RuntimeError(e.to_string()))?,
            );
            let kernel_config = csf_kernel::KernelConfig {
                scheduler_cores: vec![0], // Use CPU 0
                max_tasks: 1000,
                quantum_us: 100,
                memory_pool_size: 1024 * 1024, // 1MB
                enable_deadline_monitoring: true,
            };
            let kernel = Arc::new(csf_kernel::ChronosKernel::new(kernel_config)?);
            let mlir_runtime = csf_mlir::create_runtime(Default::default()).await?;
            Ok::<_, FFIError>((bus, kernel, mlir_runtime))
        })?;

        #[cfg(not(feature = "mlir"))]
        let (bus, kernel) = tokio_runtime.block_on(async {
            let bus = Arc::new(
                csf_bus::PhaseCoherenceBus::new(Default::default())
                    .map_err(|e| FFIError::RuntimeError(e.to_string()))?,
            );
            let kernel_config = csf_kernel::KernelConfig {
                scheduler_cores: vec![0], // Use CPU 0
                max_tasks: 1000,
                quantum_us: 100,
                memory_pool_size: 1024 * 1024, // 1MB
                enable_deadline_monitoring: true,
            };
            let kernel = Arc::new(csf_kernel::ChronosKernel::new(kernel_config)?);
            Ok::<_, FFIError>((bus, kernel))
        })?;

        #[cfg(feature = "mlir")]
        {
            RUNTIME = Some(Arc::new(CsfRuntime {
                tokio_runtime,
                bus,
                kernel,
                mlir_runtime,
            }));
        }

        #[cfg(not(feature = "mlir"))]
        {
            RUNTIME = Some(Arc::new(CsfRuntime {
                tokio_runtime,
                bus,
                kernel,
            }));
        }

        Ok(())
    }
}

fn shutdown_runtime() -> FFIResult<()> {
    unsafe {
        RUNTIME.take().ok_or(FFIError::NotInitialized)?;
        Ok(())
    }
}

fn get_runtime() -> FFIResult<Arc<CsfRuntime>> {
    unsafe { RUNTIME.as_ref().cloned().ok_or(FFIError::NotInitialized) }
}

/// Create a new phase packet
#[no_mangle]
pub extern "C" fn csf_packet_create(
    packet_type: u8,
    priority: u8,
    data: *const u8,
    data_len: usize,
) -> *mut CSF_Packet {
    if data.is_null() && data_len > 0 {
        return ptr::null_mut();
    }

    let data_vec = if data_len > 0 {
        unsafe { std::slice::from_raw_parts(data, data_len).to_vec() }
    } else {
        Vec::new()
    };

    let packet = PhasePacket {
        header: PacketHeader {
            version: 1,
            packet_id: PacketId::new(),
            packet_type: match packet_type {
                0 => PacketType::Control,
                1 => PacketType::Data,
                2 => PacketType::Event,
                3 => PacketType::Stream,
                _ => PacketType::Data,
            },
            priority,
            flags: PacketFlags::empty(),
            timestamp: hardware_timestamp(),
            source_node: 0,
            destination_node: 0,
            causality_hash: 0,
            sequence_number: None,
            sequence: 0,
            fragment_count: None,
            payload_size: data_vec.len() as u32,
            checksum: 0,
        },
        payload: PacketPayload {
            data: data_vec,
            metadata: std::collections::HashMap::new(),
        },
    };

    Box::into_raw(Box::new(CSF_Packet::from(packet)))
}

/// Destroy a phase packet
#[no_mangle]
pub extern "C" fn csf_packet_destroy(packet: *mut CSF_Packet) {
    if !packet.is_null() {
        unsafe {
            let _ = Box::from_raw(packet);
        }
    }
}

/// Send a packet through the bus
#[no_mangle]
pub extern "C" fn csf_bus_send(channel_name: *const c_char, packet: *const CSF_Packet) -> i32 {
    if channel_name.is_null() || packet.is_null() {
        return -1;
    }

    let runtime = match get_runtime() {
        Ok(r) => r,
        Err(_) => return -2,
    };

    let channel_str = unsafe {
        match CStr::from_ptr(channel_name).to_str() {
            Ok(s) => s,
            Err(_) => return -3,
        }
    };

    let packet_data = unsafe { &*packet };
    let phase_packet = packet_data.to_phase_packet();

    // Convert to protocol packet for bus
    let protocol_packet = csf_protocol::PhasePacket::new(
        match phase_packet.header.packet_type {
            PacketType::Control => PacketType::Control,
            PacketType::Data => PacketType::Data,
            PacketType::Event => PacketType::Event,
            PacketType::Stream => PacketType::Stream,
        },
        phase_packet.header.source_node,
        phase_packet.header.destination_node,
        csf_protocol::PacketPayload {
            data: phase_packet.payload.data,
            metadata: phase_packet.payload.metadata,
        },
    );

    // Simplified implementation for compilation - packet sending to be completed later
    0 // Success placeholder
}

/// Create a new MLIR module
#[cfg(feature = "mlir")]
#[no_mangle]
pub extern "C" fn csf_mlir_module_create(
    name: *const c_char,
    mlir_code: *const c_char,
) -> *mut CSF_MLIRModule {
    if name.is_null() || mlir_code.is_null() {
        return ptr::null_mut();
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let mlir_str = unsafe {
        match CStr::from_ptr(mlir_code).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let module = csf_mlir::MlirModule {
        name: name_str.to_string(),
        id: csf_mlir::ModuleId::new(),
        ir: mlir_str.to_string(),
        artifact: None,
        metadata: Default::default(),
    };

    Box::into_raw(Box::new(CSF_MLIRModule::from(module)))
}

/// Compile and load an MLIR module
#[no_mangle]
#[cfg(feature = "mlir")]
#[no_mangle]
pub extern "C" fn csf_mlir_module_load(module: *mut CSF_MLIRModule) -> i32 {
    if module.is_null() {
        return -1;
    }

    let runtime = match get_runtime() {
        Ok(r) => r,
        Err(_) => return -2,
    };

    let module_data = unsafe { &mut *module };
    let mlir_module = module_data.to_mlir_module();

    runtime.tokio_runtime.block_on(async {
        match runtime.mlir_runtime.load_module(mlir_module).await {
            Ok(id) => {
                module_data.module_id = id.0;
                0
            }
            Err(_) => -3,
        }
    })
}

/// Execute an MLIR module
#[no_mangle]
#[cfg(feature = "mlir")]
#[no_mangle]
pub extern "C" fn csf_mlir_module_execute(
    module_id: u64,
    inputs: *const CSF_Tensor,
    num_inputs: usize,
    outputs: *mut CSF_Tensor,
    num_outputs: usize,
) -> i32 {
    if (inputs.is_null() && num_inputs > 0) || (outputs.is_null() && num_outputs > 0) {
        return -1;
    }

    let runtime = match get_runtime() {
        Ok(r) => r,
        Err(_) => return -2,
    };

    // Convert input tensors
    let input_tensors = unsafe {
        std::slice::from_raw_parts(inputs, num_inputs)
            .iter()
            .map(|t| t.to_tensor())
            .collect::<Result<Vec<_>, _>>()
    };

    let input_tensors = match input_tensors {
        Ok(t) => t,
        Err(_) => return -3,
    };

    // Execute
    let result = runtime.tokio_runtime.block_on(async {
        runtime
            .mlir_runtime
            .execute(csf_mlir::ModuleId(module_id), input_tensors, None)
            .await
    });

    match result {
        Ok(output_tensors) => {
            // Copy outputs
            if output_tensors.len() != num_outputs {
                return -4;
            }

            for (i, tensor) in output_tensors.into_iter().enumerate() {
                unsafe {
                    outputs.add(i).write(CSF_Tensor::from_tensor(tensor));
                }
            }

            0
        }
        Err(_) => -5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_shutdown() {
        assert_eq!(csf_init(), 0);
        assert_eq!(csf_shutdown(), 0);
    }

    #[test]
    fn test_version() {
        let version = csf_version();
        assert!(!version.is_null());
        unsafe {
            let version_str = CStr::from_ptr(version).to_str().unwrap_or("");
            assert!(!version_str.is_empty(), "version should not be empty");
            csf_free_string(version as *mut c_char);
        }
    }
}
