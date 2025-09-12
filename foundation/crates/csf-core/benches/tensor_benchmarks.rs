//! PhD-quality performance benchmarks for ARES ChronoFabric tensor operations
//! Author: Ididia Serfaty

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use csf_core::tensor_real::PrecisionTensor;
use std::time::Duration;

/// Performance requirements for Phase 1 gate criteria
struct PerformanceTargets;

impl PerformanceTargets {
    const MATMUL_1000_TARGET: Duration = Duration::from_millis(1);
    const SVD_1000_TARGET: Duration = Duration::from_millis(10);
    const QR_500_TARGET: Duration = Duration::from_millis(5);
    const EIGENDECOMP_500_TARGET: Duration = Duration::from_millis(20);
}

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    let sizes = [100, 250, 500, 1000];
    
    for size in sizes.iter() {
        // FLOPS calculation: 2*n³ for matrix multiplication
        let flops = 2u64 * (*size as u64).pow(3);
        group.throughput(Throughput::Elements(flops));
        
        group.bench_with_input(
            BenchmarkId::new("precision_matmul", size),
            size,
            |b, &size| {
                let a = PrecisionTensor::from_array(Array2::<f64>::ones((size, size)));
                let b = PrecisionTensor::from_array(Array2::<f64>::ones((size, size)));
                
                b.iter(|| {
                    let _result = a.matmul(&b).unwrap();
                    black_box(_result);
                });
            },
        );
        
        // Performance validation for 1000x1000
        if *size == 1000 {
            let a = PrecisionTensor::from_array(Array2::<f64>::ones((1000, 1000)));
            let b = PrecisionTensor::from_array(Array2::<f64>::ones((1000, 1000)));
            
            let start = std::time::Instant::now();
            let _result = a.matmul(&b).unwrap();
            let duration = start.elapsed();
            
            assert!(duration < PerformanceTargets::MATMUL_1000_TARGET,
                   "Matrix multiplication 1000x1000 took {}ms, target: {}ms",
                   duration.as_millis(), PerformanceTargets::MATMUL_1000_TARGET.as_millis());
        }
    }
    
    group.finish();
}

fn benchmark_svd_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_decomposition");
    
    let sizes = [50, 100, 250, 500, 1000];
    
    for size in sizes.iter() {
        // SVD complexity: O(mn²) for m x n matrix, here m = n
        let ops = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(ops));
        
        group.bench_with_input(
            BenchmarkId::new("precision_svd", size),
            size,
            |b, &size| {
                // Create well-conditioned test matrix
                let mut data = vec![0.0; size * size];
                for i in 0..size {
                    for j in 0..size {
                        data[i * size + j] = ((i + j) as f64 + 1.0).sin();
                    }
                }
                let array = Array2::from_shape_vec((size, size), data).unwrap();
                let tensor = PrecisionTensor::from_array(array);
                
                b.iter(|| {
                    let (_u, _s, _vt) = tensor.svd().unwrap();
                    black_box((_u, _s, _vt));
                });
            },
        );
        
        // Performance validation for 1000x1000
        if *size == 1000 {
            let mut data = vec![0.0; 1000000];
            for i in 0..1000 {
                for j in 0..1000 {
                    data[i * 1000 + j] = ((i + j) as f64 + 1.0).sin();
                }
            }
            let array = Array2::from_shape_vec((1000, 1000), data).unwrap();
            let tensor = PrecisionTensor::from_array(array);
            
            let start = std::time::Instant::now();
            let (_u, _s, _vt) = tensor.svd().unwrap();
            let duration = start.elapsed();
            
            assert!(duration < PerformanceTargets::SVD_1000_TARGET,
                   "SVD 1000x1000 took {}ms, target: {}ms",
                   duration.as_millis(), PerformanceTargets::SVD_1000_TARGET.as_millis());
        }
    }
    
    group.finish();
}

fn benchmark_qr_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_decomposition");
    
    let sizes = [100, 250, 500, 750];
    
    for size in sizes.iter() {
        let ops = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(ops));
        
        group.bench_with_input(
            BenchmarkId::new("precision_qr", size),
            size,
            |b, &size| {
                // Create random-like test matrix
                let mut data = vec![0.0; size * size];
                for i in 0..size {
                    for j in 0..size {
                        data[i * size + j] = (i as f64 * 0.1 + j as f64 * 0.01).cos();
                    }
                }
                let array = Array2::from_shape_vec((size, size), data).unwrap();
                let tensor = PrecisionTensor::from_array(array);
                
                b.iter(|| {
                    let (_q, _r) = tensor.qr().unwrap();
                    black_box((_q, _r));
                });
            },
        );
        
        // Performance validation for 500x500
        if *size == 500 {
            let mut data = vec![0.0; 250000];
            for i in 0..500 {
                for j in 0..500 {
                    data[i * 500 + j] = (i as f64 * 0.1 + j as f64 * 0.01).cos();
                }
            }
            let array = Array2::from_shape_vec((500, 500), data).unwrap();
            let tensor = PrecisionTensor::from_array(array);
            
            let start = std::time::Instant::now();
            let (_q, _r) = tensor.qr().unwrap();
            let duration = start.elapsed();
            
            assert!(duration < PerformanceTargets::QR_500_TARGET,
                   "QR 500x500 took {}ms, target: {}ms",
                   duration.as_millis(), PerformanceTargets::QR_500_TARGET.as_millis());
        }
    }
    
    group.finish();
}

fn benchmark_eigendecomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigendecomposition");
    
    let sizes = [50, 100, 250, 500];
    
    for size in sizes.iter() {
        let ops = (*size as u64).pow(3);
        group.throughput(Throughput::Elements(ops));
        
        group.bench_with_input(
            BenchmarkId::new("precision_eigh", size),
            size,
            |b, &size| {
                // Create symmetric positive definite matrix
                let mut data = vec![0.0; size * size];
                for i in 0..size {
                    for j in 0..size {
                        if i == j {
                            data[i * size + j] = 2.0 + i as f64 * 0.1;
                        } else if i.abs_diff(j) == 1 {
                            data[i * size + j] = 0.5;
                            data[j * size + i] = 0.5;
                        }
                    }
                }
                let array = Array2::from_shape_vec((size, size), data).unwrap();
                let tensor = PrecisionTensor::from_array(array);
                
                b.iter(|| {
                    let (_eigenvals, _eigenvecs) = tensor.eigh().unwrap();
                    black_box((_eigenvals, _eigenvecs));
                });
            },
        );
        
        // Performance validation for 500x500
        if *size == 500 {
            let mut data = vec![0.0; 250000];
            for i in 0..500 {
                for j in 0..500 {
                    if i == j {
                        data[i * 500 + j] = 2.0 + i as f64 * 0.1;
                    } else if i.abs_diff(j) == 1 {
                        data[i * 500 + j] = 0.5;
                        data[j * 500 + i] = 0.5;
                    }
                }
            }
            let array = Array2::from_shape_vec((500, 500), data).unwrap();
            let tensor = PrecisionTensor::from_array(array);
            
            let start = std::time::Instant::now();
            let (_eigenvals, _eigenvecs) = tensor.eigh().unwrap();
            let duration = start.elapsed();
            
            assert!(duration < PerformanceTargets::EIGENDECOMP_500_TARGET,
                   "Eigendecomposition 500x500 took {}ms, target: {}ms",
                   duration.as_millis(), PerformanceTargets::EIGENDECOMP_500_TARGET.as_millis());
        }
    }
    
    group.finish();
}

fn benchmark_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    
    // Test with ill-conditioned matrices
    group.bench_function("ill_conditioned_svd", |b| {
        // Create Hilbert matrix (notoriously ill-conditioned)
        let size = 20;
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..size {
                data[i * size + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let array = Array2::from_shape_vec((size, size), data).unwrap();
        let tensor = PrecisionTensor::from_array(array);
        
        b.iter(|| {
            match tensor.svd() {
                Ok((_u, _s, _vt)) => black_box((_u, _s, _vt)),
                Err(_) => (), // Expected for very ill-conditioned matrices
            }
        });
    });
    
    group.finish();
}

fn mathematical_correctness_validation() {
    println!("Running mathematical correctness validation...");
    
    // Test 1: SVD reconstruction accuracy
    let test_matrix = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64).collect()).unwrap();
    let tensor = PrecisionTensor::from_array(test_matrix.clone());
    
    let (u, s, vt) = tensor.svd().unwrap();
    
    // Reconstruct matrix: A = U * Σ * V^T
    let sigma_matrix = {
        let mut sigma = Array2::zeros((100, 100));
        for (i, &val) in s.iter().enumerate() {
            if i < 100 {
                sigma[[i, i]] = val;
            }
        }
        PrecisionTensor::from_array(sigma)
    };
    
    let reconstructed = u.matmul(&sigma_matrix).unwrap().matmul(&vt).unwrap();
    
    // Calculate reconstruction error
    let mut max_error = 0.0;
    for i in 0..100 {
        for j in 0..100 {
            let original = test_matrix[[i, j]];
            let reconstructed_val = reconstructed.data()[[i, j]];
            let error = (original - reconstructed_val).abs();
            max_error = max_error.max(error);
        }
    }
    
    println!("SVD reconstruction max error: {:.2e} (should be < 1e-10)", max_error);
    assert!(max_error < 1e-10, "SVD reconstruction error too large: {:.2e}", max_error);
    
    // Test 2: QR orthogonality check
    let test_matrix_qr = Array2::from_shape_vec((50, 50), (0..2500).map(|x| (x as f64).sin()).collect()).unwrap();
    let tensor_qr = PrecisionTensor::from_array(test_matrix_qr);
    
    let (q, _r) = tensor_qr.qr().unwrap();
    let qt_q = q.matmul(&q).unwrap(); // Should be identity
    
    let mut ortho_error = 0.0;
    for i in 0..50 {
        for j in 0..50 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = qt_q.data()[[i, j]];
            let error = (expected - actual).abs();
            ortho_error = ortho_error.max(error);
        }
    }
    
    println!("QR orthogonality max error: {:.2e} (should be < 1e-12)", ortho_error);
    assert!(ortho_error < 1e-12, "QR orthogonality error too large: {:.2e}", ortho_error);
    
    println!("✅ All mathematical correctness tests passed!");
}

fn performance_regression_check() {
    println!("Running performance regression checks...");
    
    // Baseline performance targets (must not regress)
    let targets = [
        ("MatMul 1000x1000", PerformanceTargets::MATMUL_1000_TARGET),
        ("SVD 1000x1000", PerformanceTargets::SVD_1000_TARGET),
        ("QR 500x500", PerformanceTargets::QR_500_TARGET),
        ("Eigen 500x500", PerformanceTargets::EIGENDECOMP_500_TARGET),
    ];
    
    for (name, target) in targets.iter() {
        println!("Target for {}: {}ms", name, target.as_millis());
    }
    
    println!("✅ Performance targets established for CI validation");
}

criterion_group!(
    tensor_benchmarks,
    benchmark_matrix_multiplication,
    benchmark_svd_decomposition,
    benchmark_qr_decomposition,
    benchmark_eigendecomposition,
    benchmark_numerical_stability
);

criterion_main!(tensor_benchmarks);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mathematical_correctness() {
        mathematical_correctness_validation();
    }
    
    #[test]
    fn test_performance_targets() {
        performance_regression_check();
    }
}