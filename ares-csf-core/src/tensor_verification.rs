//! Tensor verification and validation utilities.

use crate::tensor::{RealTensor, ComplexTensor};
use crate::error::{Error, Result};

/// Verification utilities for tensors
pub struct TensorVerifier;

impl TensorVerifier {
    /// Verify tensor is finite (no NaN or infinity values)
    pub fn verify_finite(tensor: &RealTensor) -> Result<()> {
        for &value in tensor.data() {
            if !value.is_finite() {
                return Err(Error::tensor(format!("Non-finite value detected: {}", value)));
            }
        }
        Ok(())
    }

    /// Verify tensor dimensions are valid
    pub fn verify_dimensions(shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::tensor("Tensor must have at least one dimension".to_string()));
        }
        
        for &dim in shape {
            if dim == 0 {
                return Err(Error::tensor("Tensor dimensions must be positive".to_string()));
            }
        }
        
        Ok(())
    }

    /// Verify complex tensor is finite
    pub fn verify_complex_finite(tensor: &ComplexTensor) -> Result<()> {
        for value in tensor.data() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(Error::tensor(format!("Non-finite complex value detected: {}", value)));
            }
        }
        Ok(())
    }
}