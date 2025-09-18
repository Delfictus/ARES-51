//! Real-valued tensor operations and optimizations.

pub use crate::tensor::{RealTensor, TensorOps};

/// Additional operations specific to real tensors
impl RealTensor {
    /// Calculate tensor norm (Frobenius norm)
    pub fn norm(&self) -> f64 {
        self.data().iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Scale all elements by a factor
    pub fn scale(&mut self, factor: f64) {
        for element in self.data_mut() {
            *element *= factor;
        }
    }

    /// Add another tensor element-wise
    pub fn add(&mut self, other: &RealTensor) -> crate::Result<()> {
        if self.shape() != other.shape() {
            return Err(crate::Error::tensor("Tensor shapes must match for addition".to_string()));
        }
        
        for (a, b) in self.data_mut().iter_mut().zip(other.data().iter()) {
            *a += b;
        }
        
        Ok(())
    }
}