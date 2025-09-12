//! MLIR dialect implementations

pub mod chrono;
pub mod quantum;
pub mod tensor;

/// Initialize all custom dialects
pub fn initialize_dialects() -> crate::simple_error::MlirResult<()> {
    quantum::register_quantum_dialect()?;
    tensor::register_tensor_dialect()?;
    chrono::register_chrono_dialect()?;
    Ok(())
}
