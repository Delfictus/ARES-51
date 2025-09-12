//! Comprehensive production-grade validation for RelationalTensor mathematical operations
//!
//! This test suite provides exhaustive validation of tensor operations with quantum-aware
//! semantics, mathematical accuracy verification, and performance benchmarks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use csf_core::tensor::{RelationalMetadata, RelationalTensor};
use csf_core::{ComponentId, NanoTime};
use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array, Array2, Array3, ArrayD, Axis, IxDyn};
use num_complex::Complex64;
use num_traits::{One, Zero};

/// Configuration for comprehensive tensor validation
#[derive(Clone)]
struct TensorValidationConfig {
    small_tensor_size: usize,
    medium_tensor_size: usize,
    large_tensor_size: usize,
    stress_tensor_size: usize,
    thread_count: usize,
    performance_iterations: usize,
    accuracy_epsilon: f64,
    coherence_threshold: f64,
}

impl Default for TensorValidationConfig {
    fn default() -> Self {
        Self {
            small_tensor_size: 10,
            medium_tensor_size: 100,
            large_tensor_size: 1000,
            stress_tensor_size: 5000,
            thread_count: 8,
            performance_iterations: 1000,
            accuracy_epsilon: 1e-15,
            coherence_threshold: 0.95,
        }
    }
}

/// Generate test data with various patterns
fn generate_test_tensor<T>(rows: usize, cols: usize, pattern: TestPattern) -> RelationalTensor<T>
where
    T: Clone + num_traits::Num + num_traits::NumCast + PartialEq + num_traits::FromPrimitive,
{
    let mut data = Vec::new();

    match pattern {
        TestPattern::Identity => {
            for i in 0..rows {
                for j in 0..cols {
                    let val = if i == j { T::one() } else { T::zero() };
                    data.push(val);
                }
            }
        }
        TestPattern::Ones => {
            data.resize(rows * cols, T::one());
        }
        TestPattern::Sequential => {
            for i in 0..(rows * cols) {
                data.push(T::from_usize(i).unwrap_or(T::zero()));
            }
        }
        TestPattern::Random => {
            for _ in 0..(rows * cols) {
                let val = T::from_f64(rand::random::<f64>()).unwrap_or(T::zero());
                data.push(val);
            }
        }
        TestPattern::Harmonic => {
            for i in 0..rows {
                for j in 0..cols {
                    let harmonic = 1.0 / ((i + j + 1) as f64);
                    data.push(T::from_f64(harmonic).unwrap_or(T::zero()));
                }
            }
        }
    }

    let shape = vec![rows, cols];
    let ndarray = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
    let metadata = RelationalMetadata::default();

    RelationalTensor::from_ndarray(ndarray, metadata).unwrap()
}

#[derive(Clone)]
enum TestPattern {
    Identity,
    Ones,
    Sequential,
    Random,
    Harmonic,
}

#[cfg(test)]
mod mathematical_accuracy_tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication_accuracy() {
        let config = TensorValidationConfig::default();

        // Test against nalgebra reference implementation
        let size = config.medium_tensor_size;
        let tensor_a = generate_test_tensor::<f64>(size, size, TestPattern::Sequential);
        let tensor_b = generate_test_tensor::<f64>(size, size, TestPattern::Harmonic);

        // RelationalTensor multiplication
        let tensor_result = tensor_a.matrix_multiply(&tensor_b).unwrap();

        // nalgebra reference
        let nalgebra_a = DMatrix::from_fn(size, size, |i, j| (i * size + j) as f64);
        let nalgebra_b = DMatrix::from_fn(size, size, |i, j| 1.0 / ((i + j + 1) as f64));
        let nalgebra_result = &nalgebra_a * &nalgebra_b;

        // Compare results
        for i in 0..size {
            for j in 0..size {
                let tensor_val = tensor_result.data.get([i, j]).unwrap();
                let nalgebra_val = nalgebra_result[(i, j)];
                let diff = (tensor_val - nalgebra_val).abs();

                assert!(
                    diff < config.accuracy_epsilon,
                    "Matrix multiplication accuracy failure at ({}, {}): {} vs {}",
                    i,
                    j,
                    tensor_val,
                    nalgebra_val
                );
            }
        }
    }

    #[test]
    fn test_eigenvalue_decomposition_accuracy() {
        let size = 50;

        // Create symmetric matrix for stable eigenvalue computation
        let mut symmetric_data = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let val = if i <= j {
                    1.0 / ((i + j + 1) as f64)
                } else {
                    1.0 / ((j + i + 1) as f64)
                };
                symmetric_data.push(val);
            }
        }

        let shape = vec![size, size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), symmetric_data).unwrap();
        let tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Get eigenvalues using RelationalTensor
        let eigenvalues = tensor.eigenvalues().unwrap();

        // Verify eigenvalues are finite (for symmetric matrix)
        for eigenval in &eigenvalues {
            assert!(
                eigenval.is_finite(),
                "Eigenvalue should be finite: {}",
                eigenval
            );
        }

        // Verify eigenvalues are in descending order
        for i in 0..eigenvalues.len() - 1 {
            assert!(
                eigenvalues[i] >= eigenvalues[i + 1],
                "Eigenvalues should be ordered"
            );
        }
    }

    #[test]
    fn test_svd_accuracy() {
        let config = TensorValidationConfig::default();
        let rows = config.medium_tensor_size;
        let cols = config.medium_tensor_size / 2;

        let tensor = generate_test_tensor::<f64>(rows, cols, TestPattern::Sequential);
        let (u, s, vt) = tensor.svd().unwrap();

        // Convert singular values to diagonal matrix tensor for reconstruction
        let mut s_diagonal_data = vec![0.0; u.shape[1] * vt.shape[0]];
        for i in 0..s.len().min(u.shape[1]).min(vt.shape[0]) {
            s_diagonal_data[i * vt.shape[0] + i] = s[i];
        }
        let s_tensor = RelationalTensor::new(
            s_diagonal_data,
            vec![u.shape[1], vt.shape[0]],
            RelationalMetadata::default(),
        )
        .unwrap();

        // Reconstruct matrix from SVD: U * S * V^T
        let reconstructed = u
            .matrix_multiply(&s_tensor.matrix_multiply(&vt).unwrap())
            .unwrap();

        // Compare with original
        for i in 0..rows {
            for j in 0..cols {
                let original_val = tensor.data.get([i, j]).unwrap();
                let reconstructed_val = reconstructed.data.get([i, j]).unwrap();
                let diff = (original_val - reconstructed_val).abs();

                assert!(
                    diff < config.accuracy_epsilon * 100.0, // Allow for numerical accumulation
                    "SVD reconstruction error at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Verify singular values are non-negative and ordered
        for i in 0..s.len() - 1 {
            assert!(s[i] >= 0.0, "Singular value should be non-negative");
            assert!(s[i] >= s[i + 1], "Singular values should be ordered");
        }
    }

    #[test]
    fn test_lu_decomposition_accuracy() {
        let size = 80;
        let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Random);

        let (l, u, p) = tensor.lu_decomposition().unwrap();

        // Verify PA = LU
        let pa = p.matrix_multiply(&tensor).unwrap();
        let lu = l.matrix_multiply(&u).unwrap();

        for i in 0..size {
            for j in 0..size {
                let pa_val = pa.data.get([i, j]).unwrap();
                let lu_val = lu.data.get([i, j]).unwrap();
                let diff = (pa_val - lu_val).abs();

                assert!(
                    diff < 1e-12,
                    "LU decomposition error at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Verify L is lower triangular
        for i in 0..size {
            for j in (i + 1)..size {
                let l_val = l.data.get([i, j]).unwrap();
                assert!(l_val.abs() < 1e-15, "L should be lower triangular");
            }
        }

        // Verify U is upper triangular
        for i in 1..size {
            for j in 0..i {
                let u_val = u.data.get([i, j]).unwrap();
                assert!(u_val.abs() < 1e-15, "U should be upper triangular");
            }
        }
    }

    #[test]
    fn test_qr_decomposition_accuracy() {
        let config = TensorValidationConfig::default();
        let rows = config.medium_tensor_size;
        let cols = config.medium_tensor_size / 2;

        let tensor = generate_test_tensor::<f64>(rows, cols, TestPattern::Harmonic);
        let (q, r) = tensor.qr_decomposition().unwrap();

        // Verify A = QR
        let qr = q.matrix_multiply(&r).unwrap();

        for i in 0..rows {
            for j in 0..cols {
                let orig_val = tensor.data.get([i, j]).unwrap();
                let qr_val = qr.data.get([i, j]).unwrap();
                let diff = (orig_val - qr_val).abs();

                assert!(
                    diff < 1e-12,
                    "QR decomposition error at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Verify Q is orthogonal (Q^T * Q = I)
        let qt = q.transpose();
        let qtq = qt.matrix_multiply(&q).unwrap();

        for i in 0..cols {
            for j in 0..cols {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = qtq.data.get([i, j]).unwrap();
                let diff = (actual - expected).abs();

                assert!(
                    diff < 1e-12,
                    "Q is not orthogonal at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Verify R is upper triangular
        for i in 1..cols.min(rows) {
            for j in 0..i {
                let r_val = r.data.get([i, j]).unwrap();
                assert!(r_val.abs() < 1e-15, "R should be upper triangular");
            }
        }
    }

    #[test]
    fn test_matrix_inverse_accuracy() {
        let size = 60;

        // Create well-conditioned matrix
        let mut data = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let val = if i == j {
                    10.0 + i as f64
                } else {
                    1.0 / ((i + j + 1) as f64).max(1.0)
                };
                data.push(val);
            }
        }

        let shape = vec![size, size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
        let tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        let inverse = tensor.inverse().unwrap();

        // Verify A * A^(-1) = I
        let product = tensor.matrix_multiply(&inverse).unwrap();

        for i in 0..size {
            for j in 0..size {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = product.data.get([i, j]).unwrap();
                let diff = (actual - expected).abs();

                assert!(
                    diff < 1e-10,
                    "Matrix inverse error at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_determinant_accuracy() {
        // Test with known determinants
        let identity = generate_test_tensor::<f64>(50, 50, TestPattern::Identity);
        let det = identity.determinant().unwrap();
        assert!(
            (det - 1.0).abs() < 1e-15,
            "Identity determinant should be 1.0"
        );

        // Test triangular matrix
        let size = 40;
        let mut triangular_data = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let val = if j >= i { (i + 1) as f64 } else { 0.0 };
                triangular_data.push(val);
            }
        }

        let shape = vec![size, size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), triangular_data).unwrap();
        let triangular =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        let det = triangular.determinant().unwrap();
        let expected_det: f64 = (1..=size).map(|i| i as f64).product();

        assert!(
            (det - expected_det).abs() / expected_det < 1e-12,
            "Triangular determinant error: {} vs {}",
            det,
            expected_det
        );
    }

    #[test]
    fn test_norm_calculations() {
        let config = TensorValidationConfig::default();
        let tensor =
            generate_test_tensor::<f64>(config.medium_tensor_size, 1, TestPattern::Sequential);

        // Verify different norms
        let l2_norm = tensor.norm(); // Frobenius norm (L2 norm for matrices)
        let frobenius_norm = tensor.frobenius_norm();
        assert!(
            (l2_norm - frobenius_norm).abs() < config.accuracy_epsilon,
            "norm() and frobenius_norm() should match"
        );

        // L2 norm should be sqrt(sum of squares)
        let expected_l2: f64 = tensor.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(
            (l2_norm - expected_l2).abs() < config.accuracy_epsilon,
            "L2 norm calculation should match"
        );

        // Verify norm is positive for non-zero tensor
        assert!(l2_norm > 0.0, "Norm should be positive for non-zero tensor");
    }
}

#[cfg(test)]
mod quantum_coherence_tests {
    use super::*;

    #[test]
    fn test_quantum_state_normalization() {
        let size = 100;
        let mut quantum_data = Vec::new();

        // Create quantum state vector with complex amplitudes
        for i in 0..size {
            let amplitude = Complex64::new(
                (i as f64).sin() / (size as f64).sqrt(),
                (i as f64).cos() / (size as f64).sqrt(),
            );
            quantum_data.push(amplitude);
        }

        let shape = vec![size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), quantum_data).unwrap();
        let mut quantum_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Normalize the quantum state
        quantum_tensor.normalize_quantum_state();

        // Verify normalization (|ψ|² = 1)
        let norm_squared: f64 = quantum_tensor.data.iter().map(|&amp| amp.norm_sqr()).sum();

        assert!(
            (norm_squared - 1.0).abs() < 1e-12,
            "Quantum state normalization failed: {} != 1.0",
            norm_squared
        );
    }

    #[test]
    fn test_quantum_entanglement_preservation() {
        let config = TensorValidationConfig::default();

        // Create Bell state: (|00⟩ + |11⟩)/√2
        let bell_state = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |00⟩ amplitude
            Complex64::new(0.0, 0.0),                  // |01⟩ amplitude
            Complex64::new(0.0, 0.0),                  // |10⟩ amplitude
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), // |11⟩ amplitude
        ];

        let shape = vec![4];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), bell_state).unwrap();
        let entangled_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Calculate entanglement entropy
        let entanglement = entangled_tensor.calculate_entanglement_entropy(2).unwrap();

        // Bell state should have maximum entanglement (ln(2) ≈ 0.693)
        let expected_entropy = 2.0_f64.ln();
        assert!(
            (entanglement - expected_entropy).abs() < config.accuracy_epsilon,
            "Bell state entanglement entropy incorrect: {} vs {}",
            entanglement,
            expected_entropy
        );
    }

    #[test]
    fn test_quantum_phase_operations() {
        let size = 50;

        // Create superposition state
        let mut superposition = Vec::new();
        for i in 0..size {
            superposition.push(Complex64::new(1.0 / (size as f64).sqrt(), 0.0));
        }

        let shape = vec![size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), superposition).unwrap();
        let mut tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Apply phase shift
        let phase_angle = std::f64::consts::PI / 4.0;
        tensor.apply_quantum_phase_shift(phase_angle);

        // Verify phase was applied correctly
        let expected_phase = Complex64::from_polar(1.0, phase_angle);
        for amp in tensor.data.iter() {
            let expected_amp = expected_phase / (size as f64).sqrt();
            let diff = (amp - expected_amp).norm();
            assert!(diff < 1e-14, "Phase shift error: {}", diff);
        }

        // Verify normalization preserved
        let norm_squared: f64 = tensor.data.iter().map(|&amp| amp.norm_sqr()).sum();
        assert!((norm_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantum_fidelity_calculation() {
        let size = 64;

        // Create two similar quantum states
        let mut state1_data = Vec::new();
        let mut state2_data = Vec::new();

        for i in 0..size {
            let angle1 = (i as f64) * std::f64::consts::PI / (size as f64);
            let angle2 = angle1 + 0.1; // Slight phase difference

            state1_data.push(Complex64::from_polar(1.0 / (size as f64).sqrt(), angle1));
            state2_data.push(Complex64::from_polar(1.0 / (size as f64).sqrt(), angle2));
        }

        let shape = vec![size];
        let ndarray1 = Array::from_shape_vec(IxDyn(&shape), state1_data).unwrap();
        let ndarray2 = Array::from_shape_vec(IxDyn(&shape), state2_data).unwrap();

        let tensor1 =
            RelationalTensor::from_ndarray(ndarray1, RelationalMetadata::default()).unwrap();
        let tensor2 =
            RelationalTensor::from_ndarray(ndarray2, RelationalMetadata::default()).unwrap();

        let fidelity = tensor1.quantum_fidelity(&tensor2).unwrap();

        // Fidelity should be high but less than 1 due to phase difference
        assert!(fidelity > 0.9, "Fidelity too low: {}", fidelity);
        assert!(fidelity <= 1.0, "Fidelity cannot exceed 1.0: {}", fidelity);

        // Self-fidelity should be 1.0
        let self_fidelity = tensor1.quantum_fidelity(&tensor1).unwrap();
        assert!(
            (self_fidelity - 1.0).abs() < 1e-15,
            "Self-fidelity should be 1.0: {}",
            self_fidelity
        );
    }

    #[test]
    fn test_quantum_coherence_metrics() {
        let size = 32;

        // Create coherent superposition
        let mut coherent_data = Vec::new();
        for i in 0..size {
            coherent_data.push(Complex64::new(
                1.0 / (size as f64).sqrt(),
                1.0 / (size as f64).sqrt(),
            ));
        }

        let shape = vec![size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), coherent_data).unwrap();
        let coherent_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        let coherence = coherent_tensor.calculate_quantum_coherence().unwrap();

        // Coherent superposition should have high coherence
        assert!(coherence > 0.8, "Coherence too low: {}", coherence);

        // Create incoherent state (diagonal in computational basis)
        let mut incoherent_data = Vec::new();
        for i in 0..size {
            if i == 0 {
                incoherent_data.push(Complex64::new(1.0, 0.0));
            } else {
                incoherent_data.push(Complex64::new(0.0, 0.0));
            }
        }

        let incoherent_ndarray = Array::from_shape_vec(IxDyn(&shape), incoherent_data).unwrap();
        let incoherent_tensor =
            RelationalTensor::from_ndarray(incoherent_ndarray, RelationalMetadata::default())
                .unwrap();

        let incoherent_coherence = incoherent_tensor.calculate_quantum_coherence().unwrap();

        // Incoherent state should have low coherence
        assert!(
            incoherent_coherence < 0.1,
            "Incoherent coherence too high: {}",
            incoherent_coherence
        );
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication_performance() {
        let config = TensorValidationConfig::default();
        let sizes = [50, 100, 200, 400];

        for &size in &sizes {
            let tensor_a = generate_test_tensor::<f64>(size, size, TestPattern::Random);
            let tensor_b = generate_test_tensor::<f64>(size, size, TestPattern::Random);

            let start = Instant::now();
            for _ in 0..10 {
                let _result = tensor_a.matrix_multiply(&tensor_b).unwrap();
            }
            let duration = start.elapsed();

            let avg_ms = duration.as_millis() / 10;
            println!("Matrix multiplication {}x{}: {}ms", size, size, avg_ms);

            // Performance scaling check (should be roughly O(n³))
            let expected_max_ms = (size as u128).pow(3) / 1_000_000; // Rough estimate
            assert!(
                avg_ms < expected_max_ms * 10,
                "Matrix multiplication too slow for size {}",
                size
            );
        }
    }

    #[test]
    fn test_eigenvalue_performance() {
        let sizes = [50, 100, 200];

        for &size in &sizes {
            let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Harmonic);

            let start = Instant::now();
            let _eigenvalues = tensor.eigenvalues().unwrap();
            let duration = start.elapsed();

            println!(
                "Eigenvalue decomposition {}x{}: {}ms",
                size,
                size,
                duration.as_millis()
            );

            // Should complete in reasonable time
            let max_seconds = (size as u64).pow(3) / 10_000; // Rough scaling
            assert!(
                duration.as_secs() < max_seconds,
                "Eigenvalue computation too slow for size {}",
                size
            );
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let config = TensorValidationConfig::default();
        let sizes = [10, 50, 100, 500];

        for &size in &sizes {
            let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Sequential);

            let tensor_size = std::mem::size_of_val(&tensor);
            let data_size = size * size * std::mem::size_of::<f64>();
            let overhead_ratio = tensor_size as f64 / data_size as f64;

            println!(
                "Tensor {}x{}: {} bytes, overhead: {:.2}x",
                size, size, tensor_size, overhead_ratio
            );

            // Overhead should be reasonable
            assert!(
                overhead_ratio < 2.0,
                "Memory overhead too high: {:.2}x",
                overhead_ratio
            );
        }
    }

    #[test]
    fn test_simd_acceleration() {
        let config = TensorValidationConfig::default();
        let size = config.large_tensor_size;

        let tensor_a = generate_test_tensor::<f64>(size, 1, TestPattern::Sequential);
        let tensor_b = generate_test_tensor::<f64>(size, 1, TestPattern::Random);

        // Element-wise operations should benefit from SIMD
        let start = Instant::now();
        for _ in 0..config.performance_iterations {
            let _result = tensor_a.add(&tensor_b).unwrap();
        }
        let simd_duration = start.elapsed();

        // Manual scalar implementation for comparison
        let start = Instant::now();
        for _ in 0..config.performance_iterations {
            let mut result_data = Vec::with_capacity(size);
            for i in 0..size {
                let a_val = tensor_a.data.get([i, 0]).unwrap();
                let b_val = tensor_b.data.get([i, 0]).unwrap();
                result_data.push(a_val + b_val);
            }
        }
        let scalar_duration = start.elapsed();

        let speedup = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
        println!("SIMD speedup for element-wise addition: {:.2}x", speedup);

        // Should see some acceleration for large vectors
        if size >= 1000 {
            assert!(
                speedup >= 1.2,
                "Insufficient SIMD acceleration: {:.2}x",
                speedup
            );
        }
    }

    #[test]
    fn test_cache_efficiency() {
        let config = TensorValidationConfig::default();

        // Test cache-friendly vs cache-unfriendly access patterns
        let size = 512; // Power of 2 for clean cache behavior
        let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Random);

        // Row-major access (cache-friendly)
        let start = Instant::now();
        let mut sum_row = 0.0;
        for i in 0..size {
            for j in 0..size {
                sum_row += tensor.data.get([i, j]).unwrap();
            }
        }
        let row_duration = start.elapsed();

        // Column-major access (cache-unfriendly)
        let start = Instant::now();
        let mut sum_col = 0.0;
        for j in 0..size {
            for i in 0..size {
                sum_col += tensor.data.get([i, j]).unwrap();
            }
        }
        let col_duration = start.elapsed();

        let cache_ratio = col_duration.as_nanos() as f64 / row_duration.as_nanos() as f64;
        println!("Cache efficiency ratio (col/row): {:.2}x", cache_ratio);

        // Sums should be equal
        assert!(
            (sum_row - sum_col).abs() < 1e-10,
            "Access pattern sums should be equal"
        );

        // Column access should be slower due to cache misses
        assert!(
            cache_ratio > 1.5,
            "Cache efficiency not demonstrated: {:.2}x",
            cache_ratio
        );
    }

    #[test]
    fn test_batch_operation_efficiency() {
        let config = TensorValidationConfig::default();
        let batch_size = 100;
        let tensor_size = 50;

        // Create batch of tensors
        let tensors: Vec<_> = (0..batch_size)
            .map(|_| generate_test_tensor::<f64>(tensor_size, tensor_size, TestPattern::Random))
            .collect();

        // Individual operations
        let start = Instant::now();
        let mut individual_results = Vec::new();
        for tensor in &tensors {
            individual_results.push(tensor.trace().unwrap());
        }
        let individual_duration = start.elapsed();

        // Batch operation
        let start = Instant::now();
        let batch_results = RelationalTensor::batch_trace(&tensors).unwrap();
        let batch_duration = start.elapsed();

        // Results should be identical
        assert_eq!(individual_results.len(), batch_results.len());
        for i in 0..individual_results.len() {
            assert!((individual_results[i] - batch_results[i]).abs() < 1e-15);
        }

        let speedup = individual_duration.as_nanos() as f64 / batch_duration.as_nanos() as f64;
        println!("Batch operation speedup: {:.2}x", speedup);

        // Batch should be at least as fast as individual operations
        assert!(
            speedup >= 0.9,
            "Batch operations slower than individual: {:.2}x",
            speedup
        );
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_concurrent_tensor_operations() {
        let config = TensorValidationConfig::default();
        let shared_tensor = Arc::new(generate_test_tensor::<f64>(
            config.medium_tensor_size,
            config.medium_tensor_size,
            TestPattern::Sequential,
        ));

        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let tensor_clone = shared_tensor.clone();
            let results_clone = results.clone();

            let handle = thread::spawn(move || {
                let mut local_results = Vec::new();

                for i in 0..config.performance_iterations / config.thread_count {
                    // Perform various operations
                    let trace = tensor_clone.trace().unwrap();
                    let det = tensor_clone.determinant().unwrap();
                    let norm = tensor_clone.frobenius_norm();

                    local_results.push((trace, det, norm, thread_id, i));
                }

                results_clone.lock().unwrap().extend(local_results);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), config.performance_iterations);

        // All results should be identical (deterministic operations)
        if let Some(first) = final_results.first() {
            for result in final_results.iter() {
                assert!((result.0 - first.0).abs() < 1e-15, "Trace mismatch");
                assert!((result.1 - first.1).abs() < 1e-12, "Determinant mismatch");
                assert!((result.2 - first.2).abs() < 1e-15, "Norm mismatch");
            }
        }
    }

    #[test]
    fn test_send_sync_implementation() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<RelationalTensor<f64>>();
        assert_sync::<RelationalTensor<f64>>();
        assert_send::<RelationalTensor<Complex64>>();
        assert_sync::<RelationalTensor<Complex64>>();
        assert_send::<RelationalMetadata>();
        assert_sync::<RelationalMetadata>();
        // QuantumMetadata type removed - no longer available
    }

    #[test]
    fn test_concurrent_modifications() {
        let config = TensorValidationConfig::default();
        let tensor_count = 100;

        // Create multiple tensors to modify concurrently
        let tensors: Vec<_> = (0..tensor_count)
            .map(|i| {
                Arc::new(Mutex::new(generate_test_tensor::<f64>(
                    50,
                    50,
                    TestPattern::Sequential,
                )))
            })
            .collect();

        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let tensors_clone = tensors.clone();

            let handle = thread::spawn(move || {
                for i in 0..config.performance_iterations / config.thread_count {
                    let tensor_idx = (thread_id + i) % tensor_count;
                    let mut tensor = tensors_clone[tensor_idx].lock().unwrap();

                    // Modify the tensor
                    *tensor = tensor.multiply_scalar(1.001);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all tensors are still valid
        for tensor_mutex in &tensors {
            let tensor = tensor_mutex.lock().unwrap();
            assert!(tensor.data.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_memory_safety_under_pressure() {
        let config = TensorValidationConfig::default();
        let large_tensor_count = 50;

        let mut handles = Vec::new();

        for thread_id in 0..config.thread_count {
            let handle = thread::spawn(move || {
                let mut local_tensors = Vec::new();

                // Create and manipulate large tensors
                for i in 0..large_tensor_count / config.thread_count {
                    let size = 200 + (thread_id * 10) + i;
                    let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Random);

                    // Perform operations that allocate memory
                    let _transposed = tensor.transpose();
                    let _trace = tensor.trace().unwrap();

                    local_tensors.push(tensor);
                }

                // Verify all tensors are still accessible
                for tensor in &local_tensors {
                    let _norm = tensor.frobenius_norm();
                }

                local_tensors.len()
            });

            handles.push(handle);
        }

        let mut total_tensors = 0;
        for handle in handles {
            total_tensors += handle.join().unwrap();
        }

        assert_eq!(total_tensors, large_tensor_count);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_tensor_handling() {
        let empty_shape = vec![0, 0];
        let empty_data = Vec::<f64>::new();
        let ndarray = Array::from_shape_vec(IxDyn(&empty_shape), empty_data).unwrap();
        let empty_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Operations on empty tensors should handle gracefully
        assert_eq!(empty_tensor.shape(), &[0, 0]);
        assert_eq!(empty_tensor.data.len(), 0);
    }

    #[test]
    fn test_single_element_tensor() {
        let single_data = vec![42.0];
        let shape = vec![1, 1];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), single_data).unwrap();
        let single_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        assert_eq!(single_tensor.trace().unwrap(), 42.0);
        assert_eq!(single_tensor.determinant().unwrap(), 42.0);
        assert_eq!(single_tensor.frobenius_norm(), 42.0);
    }

    #[test]
    fn test_very_large_dimensions() {
        let large_dim = 10000;

        // Create large tensor with minimal memory usage
        let sparse_data = vec![1.0; large_dim]; // Single row
        let shape = vec![1, large_dim];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), sparse_data).unwrap();
        let large_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        let norm = large_tensor.frobenius_norm();
        let expected_norm = (large_dim as f64).sqrt();

        assert!((norm - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_edge_cases() {
        let edge_values = vec![
            0.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            1e-100,
            1e100,
            std::f64::consts::PI,
            std::f64::consts::E,
        ];

        for &value in &edge_values {
            let data = vec![value; 9];
            let shape = vec![3, 3];
            let ndarray = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let tensor =
                RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

            // Basic operations should handle edge values
            let trace = tensor.trace().unwrap();
            let norm = tensor.frobenius_norm();

            assert!(trace.is_finite() || value.is_infinite());
            assert!(norm.is_finite() || value.is_infinite());
        }
    }

    #[test]
    fn test_infinity_nan_handling() {
        let problematic_values = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        for &value in &problematic_values {
            let data = vec![value, 1.0, 2.0, 3.0];
            let shape = vec![2, 2];
            let ndarray = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let tensor =
                RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

            // Operations should propagate NaN/infinity appropriately
            let trace = tensor.trace().unwrap();
            let norm = tensor.frobenius_norm();

            if value.is_nan() {
                assert!(trace.is_nan());
                assert!(norm.is_nan());
            } else if value.is_infinite() {
                assert!(!trace.is_finite() || trace.is_infinite());
                assert!(!norm.is_finite());
            }
        }
    }

    #[test]
    fn test_precision_boundaries() {
        let config = TensorValidationConfig::default();

        // Test operations near machine epsilon
        let epsilon_data = vec![1.0, 1.0 + f64::EPSILON, 1.0 - f64::EPSILON, f64::EPSILON];

        let shape = vec![2, 2];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), epsilon_data).unwrap();
        let epsilon_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        let trace = epsilon_tensor.trace().unwrap();
        let expected_trace = 2.0 + f64::EPSILON;

        assert!((trace - expected_trace).abs() < config.accuracy_epsilon);
    }

    #[test]
    fn test_dimension_mismatch_handling() {
        let tensor_a = generate_test_tensor::<f64>(3, 4, TestPattern::Sequential);
        let tensor_b = generate_test_tensor::<f64>(5, 3, TestPattern::Sequential);

        // Matrix multiplication with incompatible dimensions should fail gracefully
        let result = tensor_a.matrix_multiply(&tensor_b);
        assert!(result.is_err(), "Should fail with dimension mismatch");

        // But compatible multiplication should succeed
        let tensor_c = generate_test_tensor::<f64>(4, 5, TestPattern::Sequential);
        let result = tensor_a.matrix_multiply(&tensor_c);
        assert!(result.is_ok(), "Compatible dimensions should succeed");
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_memory_pressure_stress() {
        let config = TensorValidationConfig::default();
        let stress_tensor_count = 100;

        // Create many large tensors simultaneously
        let tensors: Vec<_> = (0..stress_tensor_count)
            .map(|_| {
                generate_test_tensor::<f64>(
                    config.stress_tensor_size / 10,
                    config.stress_tensor_size / 10,
                    TestPattern::Random,
                )
            })
            .collect();

        // Perform operations on all tensors
        let start = Instant::now();
        let results: Vec<_> = tensors
            .iter()
            .map(|tensor| {
                let trace = tensor.trace().unwrap();
                let norm = tensor.frobenius_norm();
                let det = tensor.determinant().unwrap();
                (trace, norm, det)
            })
            .collect();

        let duration = start.elapsed();
        println!(
            "Stress test: {} tensors in {}ms",
            stress_tensor_count,
            duration.as_millis()
        );

        // All operations should complete
        assert_eq!(results.len(), stress_tensor_count);

        // Results should be finite
        for (trace, norm, det) in results {
            assert!(trace.is_finite());
            assert!(norm.is_finite());
            assert!(det.is_finite() || det.is_infinite()); // Determinant might overflow
        }
    }

    #[test]
    fn test_computational_intensity_stress() {
        let size = 800; // Large enough to be computationally intensive

        let tensor_a = generate_test_tensor::<f64>(size, size, TestPattern::Random);
        let tensor_b = generate_test_tensor::<f64>(size, size, TestPattern::Harmonic);

        let start = Instant::now();

        // Chain of intensive operations
        let multiplied = tensor_a.matrix_multiply(&tensor_b).unwrap();
        let _eigenvalues = multiplied.eigenvalues().unwrap();
        let inverted = multiplied.inverse().unwrap();
        let _final_product = multiplied.matrix_multiply(&inverted).unwrap();

        let duration = start.elapsed();
        println!("Intensive computation chain: {}s", duration.as_secs());

        // Should complete in reasonable time
        assert!(
            duration.as_secs() < 60,
            "Computational chain too slow: {}s",
            duration.as_secs()
        );
    }

    #[test]
    fn test_concurrent_stress() {
        let config = TensorValidationConfig::default();
        let thread_count = 16; // High thread count
        let operations_per_thread = 500;

        let mut handles = Vec::new();
        let success_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for thread_id in 0..thread_count {
            let counter_clone = success_counter.clone();

            let handle = thread::spawn(move || {
                let mut local_successes = 0;

                for i in 0..operations_per_thread {
                    let size = 50 + (i % 50); // Variable sizes
                    let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Random);

                    // Perform various operations
                    if let Ok(_trace) = tensor.trace() {
                        if let Ok(_det) = tensor.determinant() {
                            let _norm = tensor.frobenius_norm();
                            local_successes += 1;
                        }
                    }
                }

                counter_clone.fetch_add(local_successes, std::sync::atomic::Ordering::Relaxed);
                local_successes
            });

            handles.push(handle);
        }

        let mut total_operations = 0;
        for handle in handles {
            total_operations += handle.join().unwrap();
        }

        let expected_operations = thread_count * operations_per_thread;
        let success_rate = total_operations as f64 / expected_operations as f64;

        println!(
            "Concurrent stress test: {}/{} operations succeeded ({:.1}%)",
            total_operations,
            expected_operations,
            success_rate * 100.0
        );

        // Should have high success rate
        assert!(
            success_rate > 0.95,
            "Success rate too low: {:.1}%",
            success_rate * 100.0
        );
    }
}

#[cfg(test)]
mod integration_compatibility_tests {
    use super::*;

    #[test]
    fn test_nalgebra_interoperability() {
        let size = 100;
        let tensor = generate_test_tensor::<f64>(size, size, TestPattern::Sequential);

        // Convert to nalgebra matrix
        let nalgebra_matrix = tensor.to_dmatrix().unwrap();

        // Perform operations in nalgebra
        let transposed = nalgebra_matrix.transpose();
        let multiplied = &nalgebra_matrix * &transposed;

        // Convert back to RelationalTensor
        let result_tensor = RelationalTensor::from_dmatrix(multiplied).unwrap();

        // Verify dimensions and basic properties
        assert_eq!(result_tensor.shape(), &[size, size]);
        assert!(result_tensor.trace().unwrap().is_finite());
        assert!(result_tensor.frobenius_norm().is_finite());
    }

    #[test]
    fn test_ndarray_compatibility() {
        let config = TensorValidationConfig::default();
        let tensor = generate_test_tensor::<f64>(
            config.medium_tensor_size,
            config.medium_tensor_size,
            TestPattern::Harmonic,
        );

        // Extract ndarray for direct manipulation
        let ndarray_ref = tensor.as_ndarray();

        // Perform ndarray operations
        let sum = ndarray_ref.sum();
        let mean = ndarray_ref.mean().unwrap();
        let std_dev = ndarray_ref.std(0.0);

        // Compare with tensor methods
        let tensor_sum: f64 = tensor.data.iter().sum();
        let tensor_mean = tensor_sum / (tensor.data.len() as f64);

        assert!((sum - tensor_sum).abs() < config.accuracy_epsilon);
        assert!((mean - tensor_mean).abs() < config.accuracy_epsilon);
        assert!(std_dev >= 0.0);
    }

    #[test]
    fn test_serde_compatibility() {
        let tensor = generate_test_tensor::<f64>(50, 50, TestPattern::Random);

        // JSON serialization
        let json = serde_json::to_string(&tensor).unwrap();
        let deserialized_tensor: RelationalTensor<f64> = serde_json::from_str(&json).unwrap();

        // Verify data integrity
        assert_eq!(tensor.shape(), deserialized_tensor.shape());

        for i in 0..50 {
            for j in 0..50 {
                let original = tensor.data.get([i, j]).unwrap();
                let deserialized = deserialized_tensor.data.get([i, j]).unwrap();
                assert!((original - deserialized).abs() < 1e-15);
            }
        }

        // Binary serialization with bincode
        let binary = bincode::serialize(&tensor).unwrap();
        let bin_deserialized: RelationalTensor<f64> = bincode::deserialize(&binary).unwrap();

        // Binary should be exact
        assert_eq!(tensor.shape(), bin_deserialized.shape());
        for i in 0..50 {
            for j in 0..50 {
                let original = tensor.data.get([i, j]).unwrap();
                let deserialized = bin_deserialized.data.get([i, j]).unwrap();
                assert_eq!(*original, *deserialized);
            }
        }
    }

    #[test]
    fn test_complex_number_support() {
        let size = 64;
        let mut complex_data = Vec::new();

        for i in 0..size {
            for j in 0..size {
                complex_data.push(Complex64::new((i + j) as f64, (i * j) as f64));
            }
        }

        let shape = vec![size, size];
        let ndarray = Array::from_shape_vec(IxDyn(&shape), complex_data).unwrap();
        let complex_tensor =
            RelationalTensor::from_ndarray(ndarray, RelationalMetadata::default()).unwrap();

        // Complex operations
        let conjugate_transpose = complex_tensor.conjugate_transpose();
        let trace = complex_tensor.complex_trace().unwrap();

        // Verify complex properties
        assert!(trace.im != 0.0); // Should have imaginary part
        assert_eq!(conjugate_transpose.shape(), complex_tensor.shape());

        // Hermitian property: (A†)† = A
        let double_conjugate = conjugate_transpose.conjugate_transpose();

        for i in 0..size {
            for j in 0..size {
                let original = complex_tensor.data.get([i, j]).unwrap();
                let double_conj = double_conjugate.data.get([i, j]).unwrap();
                let diff = (original - double_conj).norm();
                assert!(
                    diff < 1e-15,
                    "Hermitian property violated at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
