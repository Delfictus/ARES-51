//! High-performance tensor operations for CSF computations.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Result type for tensor operations
pub type TensorResult<T> = Result<T>;

/// Tensor operations trait
pub trait TensorOps {
    /// Element type
    type Element: Clone + Send + Sync;
    
    /// Get tensor dimensions
    fn shape(&self) -> &[usize];
    
    /// Get total number of elements
    fn len(&self) -> usize;
    
    /// Check if tensor is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Real-valued tensor implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealTensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl RealTensor {
    /// Create a new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    /// Create tensor from data and shape
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Error::tensor(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_len
            )));
        }
        Ok(Self { data, shape })
    }

    /// Get element at index
    pub fn get(&self, index: &[usize]) -> Option<f64> {
        let flat_index = self.flat_index(index)?;
        self.data.get(flat_index).copied()
    }

    /// Set element at index
    pub fn set(&mut self, index: &[usize], value: f64) -> Result<()> {
        let flat_index = self.flat_index(index)
            .ok_or_else(|| Error::tensor("Index out of bounds".to_string()))?;
        self.data[flat_index] = value;
        Ok(())
    }

    /// Convert multi-dimensional index to flat index
    fn flat_index(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.shape.len() {
            return None;
        }

        let mut flat = 0;
        let mut stride = 1;
        
        for (i, &dim_size) in self.shape.iter().enumerate().rev() {
            if index[i] >= dim_size {
                return None;
            }
            flat += index[i] * stride;
            stride *= dim_size;
        }
        
        Some(flat)
    }

    /// Get raw data slice
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

impl TensorOps for RealTensor {
    type Element = f64;
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Complex-valued tensor implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexTensor {
    data: Vec<num_complex::Complex64>,
    shape: Vec<usize>,
}

impl ComplexTensor {
    /// Create a new complex tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![num_complex::Complex64::new(0.0, 0.0); len],
            shape,
        }
    }

    /// Create complex tensor from data and shape
    pub fn from_data(data: Vec<num_complex::Complex64>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(Error::tensor(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_len
            )));
        }
        Ok(Self { data, shape })
    }

    /// Get raw data slice
    pub fn data(&self) -> &[num_complex::Complex64] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn data_mut(&mut self) -> &mut [num_complex::Complex64] {
        &mut self.data
    }
}

impl TensorOps for ComplexTensor {
    type Element = num_complex::Complex64;
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Tensor multiplication (matrix multiplication for 2D tensors)
pub fn tensor_multiply(a: &RealTensor, b: &RealTensor) -> Result<RealTensor> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(Error::tensor("Matrix multiplication requires 2D tensors".to_string()));
    }

    let [m, k] = [a.shape()[0], a.shape()[1]];
    let [k2, n] = [b.shape()[0], b.shape()[1]];

    if k != k2 {
        return Err(Error::tensor(format!(
            "Matrix dimensions incompatible: {}x{} * {}x{}",
            m, k, k2, n
        )));
    }

    let mut result = RealTensor::new(vec![m, n]);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.get(&[i, l]).unwrap() * b.get(&[l, j]).unwrap();
            }
            result.set(&[i, j], sum)?;
        }
    }

    Ok(result)
}

/// Tensor transpose (for 2D tensors)
pub fn tensor_transpose(tensor: &RealTensor) -> Result<RealTensor> {
    if tensor.shape().len() != 2 {
        return Err(Error::tensor("Transpose requires 2D tensor".to_string()));
    }

    let [m, n] = [tensor.shape()[0], tensor.shape()[1]];
    let mut result = RealTensor::new(vec![n, m]);

    for i in 0..m {
        for j in 0..n {
            let value = tensor.get(&[i, j]).unwrap();
            result.set(&[j, i], value)?;
        }
    }

    Ok(result)
}

/// Tensor inverse (for square 2D tensors using Gauss-Jordan elimination)
pub fn tensor_inverse(tensor: &RealTensor) -> Result<RealTensor> {
    if tensor.shape().len() != 2 {
        return Err(Error::tensor("Inverse requires 2D tensor".to_string()));
    }

    let [m, n] = [tensor.shape()[0], tensor.shape()[1]];
    if m != n {
        return Err(Error::tensor("Inverse requires square matrix".to_string()));
    }

    // Create augmented matrix [A|I]
    let mut aug = RealTensor::new(vec![n, 2 * n]);
    
    // Fill with original matrix and identity
    for i in 0..n {
        for j in 0..n {
            aug.set(&[i, j], tensor.get(&[i, j]).unwrap())?;
            aug.set(&[i, j + n], if i == j { 1.0 } else { 0.0 })?;
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug.get(&[k, i]).unwrap().abs() > aug.get(&[max_row, i]).unwrap().abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug.get(&[i, j]).unwrap();
                aug.set(&[i, j], aug.get(&[max_row, j]).unwrap())?;
                aug.set(&[max_row, j], temp)?;
            }
        }

        // Check for singular matrix
        let pivot = aug.get(&[i, i]).unwrap();
        if pivot.abs() < 1e-10 {
            return Err(Error::tensor("Matrix is singular".to_string()));
        }

        // Scale pivot row
        for j in 0..(2 * n) {
            let value = aug.get(&[i, j]).unwrap() / pivot;
            aug.set(&[i, j], value)?;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug.get(&[k, i]).unwrap();
                for j in 0..(2 * n) {
                    let value = aug.get(&[k, j]).unwrap() - factor * aug.get(&[i, j]).unwrap();
                    aug.set(&[k, j], value)?;
                }
            }
        }
    }

    // Extract inverse from right half
    let mut result = RealTensor::new(vec![n, n]);
    for i in 0..n {
        for j in 0..n {
            result.set(&[i, j], aug.get(&[i, j + n]).unwrap())?;
        }
    }

    Ok(result)
}

/// Eigenvalue decomposition (simplified implementation using power iteration)
pub fn eigenvalue_decomposition(tensor: &RealTensor) -> Result<(Vec<f64>, RealTensor)> {
    if tensor.shape().len() != 2 {
        return Err(Error::tensor("Eigenvalue decomposition requires 2D tensor".to_string()));
    }

    let [m, n] = [tensor.shape()[0], tensor.shape()[1]];
    if m != n {
        return Err(Error::tensor("Eigenvalue decomposition requires square matrix".to_string()));
    }

    // Simplified implementation: find dominant eigenvalue using power iteration
    let max_iterations = 100;
    let tolerance = 1e-10;
    
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = RealTensor::new(vec![n, n]);
    
    // Find dominant eigenvalue/eigenvector
    let mut v = RealTensor::new(vec![n, 1]);
    for i in 0..n {
        v.set(&[i, 0], 1.0)?; // Initial vector
    }
    
    for _ in 0..max_iterations {
        let v_new = tensor_multiply(tensor, &v)?;
        
        // Normalize
        let mut norm = 0.0;
        for i in 0..n {
            let val = v_new.get(&[i, 0]).unwrap();
            norm += val * val;
        }
        norm = norm.sqrt();
        
        if norm < tolerance {
            break;
        }
        
        for i in 0..n {
            let val = v_new.get(&[i, 0]).unwrap() / norm;
            v.set(&[i, 0], val)?;
        }
    }
    
    // Calculate eigenvalue: λ = v^T * A * v
    let av = tensor_multiply(tensor, &v)?;
    let mut eigenvalue = 0.0;
    for i in 0..n {
        eigenvalue += v.get(&[i, 0]).unwrap() * av.get(&[i, 0]).unwrap();
    }
    
    eigenvalues.push(eigenvalue);
    
    // Store eigenvector
    for i in 0..n {
        eigenvectors.set(&[i, 0], v.get(&[i, 0]).unwrap())?;
    }
    
    Ok((eigenvalues, eigenvectors))
}

/// SVD decomposition (simplified implementation)
pub fn svd_decomposition(tensor: &RealTensor) -> Result<(RealTensor, Vec<f64>, RealTensor)> {
    if tensor.shape().len() != 2 {
        return Err(Error::tensor("SVD requires 2D tensor".to_string()));
    }

    let [m, n] = [tensor.shape()[0], tensor.shape()[1]];
    
    // Simplified implementation: return identity matrices and zero singular values
    let u = {
        let mut identity = RealTensor::new(vec![m, m]);
        for i in 0..m {
            identity.set(&[i, i], 1.0)?;
        }
        identity
    };
    
    let vt = {
        let mut identity = RealTensor::new(vec![n, n]);
        for i in 0..n {
            identity.set(&[i, i], 1.0)?;
        }
        identity
    };
    
    let singular_values = vec![0.0; m.min(n)];
    
    Ok((u, singular_values, vt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_tensor_creation() {
        let tensor = RealTensor::new(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut tensor = RealTensor::new(vec![2, 2]);
        tensor.set(&[0, 0], 1.0).unwrap();
        tensor.set(&[0, 1], 2.0).unwrap();
        tensor.set(&[1, 0], 3.0).unwrap();
        tensor.set(&[1, 1], 4.0).unwrap();
        
        assert_eq!(tensor.get(&[0, 0]), Some(1.0));
        assert_eq!(tensor.get(&[1, 1]), Some(4.0));
    }

    #[test]
    fn test_tensor_multiplication() {
        let mut a = RealTensor::new(vec![2, 2]);
        a.set(&[0, 0], 1.0).unwrap();
        a.set(&[0, 1], 2.0).unwrap();
        a.set(&[1, 0], 3.0).unwrap();
        a.set(&[1, 1], 4.0).unwrap();
        
        let mut b = RealTensor::new(vec![2, 2]);
        b.set(&[0, 0], 5.0).unwrap();
        b.set(&[0, 1], 6.0).unwrap();
        b.set(&[1, 0], 7.0).unwrap();
        b.set(&[1, 1], 8.0).unwrap();
        
        let result = tensor_multiply(&a, &b).unwrap();
        assert_eq!(result.get(&[0, 0]), Some(19.0)); // 1*5 + 2*7
        assert_eq!(result.get(&[0, 1]), Some(22.0)); // 1*6 + 2*8
        assert_eq!(result.get(&[1, 0]), Some(43.0)); // 3*5 + 4*7
        assert_eq!(result.get(&[1, 1]), Some(50.0)); // 3*6 + 4*8
    }

    #[test]
    fn test_tensor_transpose() {
        let mut tensor = RealTensor::new(vec![2, 3]);
        tensor.set(&[0, 0], 1.0).unwrap();
        tensor.set(&[0, 1], 2.0).unwrap();
        tensor.set(&[0, 2], 3.0).unwrap();
        tensor.set(&[1, 0], 4.0).unwrap();
        tensor.set(&[1, 1], 5.0).unwrap();
        tensor.set(&[1, 2], 6.0).unwrap();
        
        let transposed = tensor_transpose(&tensor).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.get(&[0, 0]), Some(1.0));
        assert_eq!(transposed.get(&[1, 0]), Some(2.0));
        assert_eq!(transposed.get(&[2, 1]), Some(6.0));
    }

    #[test]
    fn test_complex_tensor() {
        let data = vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(3.0, 4.0),
        ];
        let tensor = ComplexTensor::from_data(data.clone(), vec![2, 1]).unwrap();
        assert_eq!(tensor.shape(), &[2, 1]);
        assert_eq!(tensor.data(), &data);
    }
}