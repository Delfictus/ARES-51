//! Integration tests for MLIR runtime

use csf_mlir::*;
use csf_mlir::runtime::Tensor;

#[tokio::test]
async fn test_mlir_runtime_full_pipeline() {
    let config = RuntimeConfig {
        enable_jit: true,
        optimization_level: 2,
        backends: vec![Backend::CPU],
        memory_pool_size: 1024 * 1024, // 1MB
        thread_pool_size: 4,
        enable_profiling: true,
    };

    let runtime = create_runtime(config).await.unwrap();

    // Test simple MLIR module
    let mlir_code = r#"
        func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
        }
    "#;

    let module_id = runtime.compile_mlir("simple_add", mlir_code).await.unwrap();

    // Create test tensors
    let tensor1 = runtime.create_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let tensor2 = runtime.create_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![4]).unwrap();

    // Execute
    let outputs = runtime.execute(module_id, vec![tensor1, tensor2], None).await.unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![4]);

    // Verify statistics
    let stats = runtime.get_stats();
    assert_eq!(stats.modules_loaded, 1);
    assert_eq!(stats.executions, 1);
    assert!(stats.avg_compilation_time_ms > 0.0);
}

#[tokio::test]
async fn test_quantum_classical_interface() {
    let runtime = create_runtime(RuntimeConfig::default()).await.unwrap();

    // Test quantum circuit compilation
    let quantum_mlir = r#"
        quantum.circuit @bell_state() -> (!quantum.qreg<2>) {
            %q0 = quantum.alloc() : !quantum.qubit
            %q1 = quantum.alloc() : !quantum.qubit
            %qreg = quantum.pack %q0, %q1 : !quantum.qreg<2>
            
            quantum.h %q0 : !quantum.qubit
            quantum.cnot %q0, %q1 : !quantum.qubit, !quantum.qubit
            
            return %qreg : !quantum.qreg<2>
        }
    "#;

    let module_id = runtime.compile_mlir("bell_state", quantum_mlir).await.unwrap();

    // Test empty execution (placeholder for quantum simulation)
    let outputs = runtime.execute(module_id, vec![], None).await.unwrap();
    assert_eq!(outputs.len(), 0); // Placeholder behavior
}

#[tokio::test]
async fn test_memory_management() {
    let config = RuntimeConfig {
        memory_pool_size: 1024 * 1024, // 1MB
        ..Default::default()
    };

    let runtime = create_runtime(config).await.unwrap();

    // Create multiple tensors to test memory allocation
    let mut tensors = Vec::new();
    for i in 0..100 {
        let data: Vec<f32> = (0..256).map(|x| x as f32 + i as f32).collect();
        let tensor = runtime.create_tensor(data, vec![16, 16]).unwrap();
        tensors.push(tensor);
    }

    assert_eq!(tensors.len(), 100);
    
    // Verify each tensor has correct properties
    for tensor in &tensors {
        assert_eq!(tensor.shape, vec![16, 16]);
        assert_eq!(tensor.numel(), 256);
    }
}

#[tokio::test]
async fn test_backend_selection() {
    let config = RuntimeConfig {
        backends: vec![Backend::CPU, Backend::CUDA, Backend::Vulkan],
        ..Default::default()
    };

    let runtime = create_runtime(config).await.unwrap();

    // Simple computation that should work on any backend
    let mlir_code = r#"
        func @matrix_multiply(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>) -> tensor<32x32xf32> {
            %0 = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>) 
                              outs(%lhs : tensor<32x32xf32>) -> tensor<32x32xf32>
            return %0 : tensor<32x32xf32>
        }
    "#;

    let module_id = runtime.compile_mlir("matrix_multiply", mlir_code).await.unwrap();

    // Create test matrices
    let data: Vec<f32> = (0..1024).map(|x| x as f32).collect();
    let matrix1 = runtime.create_tensor(data.clone(), vec![32, 32]).unwrap();
    let matrix2 = runtime.create_tensor(data, vec![32, 32]).unwrap();

    // Execute multiple times to test backend switching
    for _ in 0..5 {
        let outputs = runtime.execute(module_id, vec![matrix1.clone(), matrix2.clone()], None).await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![32, 32]);
    }

    let stats = runtime.get_stats();
    assert_eq!(stats.executions, 5);
}

#[test]
fn test_quantum_operations() {
    use csf_mlir::dialects::quantum::ops::*;
    use csf_mlir::dialects::quantum::transforms::*;

    // Test quantum gate creation
    let _hadamard = QuantumGateOp {
        gate_type: GateType::H,
        qubits: vec![0],
        parameters: vec![],
    };

    let _cnot = QuantumGateOp {
        gate_type: GateType::CNOT,
        qubits: vec![0, 1],
        parameters: vec![],
    };

    // Test circuit optimization
    let mut circuit = CircuitOp {
        num_qubits: 2,
        operations: vec![],
    };

    let fusion_pass = GateFusionPass;
    fusion_pass.run(&mut circuit).unwrap();
}

#[test]
fn test_tensor_operations() {
    // Test tensor creation and manipulation
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor = Tensor::new(data, vec![2, 4], DataType::F32).unwrap();

    assert_eq!(tensor.numel(), 8);
    assert_eq!(tensor.shape, vec![2, 4]);
    assert_eq!(tensor.strides, vec![4, 1]);
    assert_eq!(tensor.nbytes(), 32); // 8 f32 values * 4 bytes each

    // Test different data types
    let complex_tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2], DataType::Complex64).unwrap();
    assert_eq!(complex_tensor.dtype as u8, DataType::Complex64 as u8);
}

#[test]
fn test_memory_pool_stress() {
    use csf_mlir::memory::MemoryManager;
    use csf_mlir::runtime::DeviceLocation;

    let manager = MemoryManager::new(1024 * 1024).unwrap(); // 1MB

    // Allocate many small blocks
    let mut allocations = Vec::new();
    for i in 0..100 {
        let size = 1024 + (i % 10) * 100; // Variable sizes
        let alloc = manager.allocate(size, 64, DeviceLocation::CPU).unwrap();
        allocations.push(alloc);
    }

    // Check stats
    let stats = manager.get_stats();
    assert_eq!(stats.allocation_count, 100);
    assert!(stats.total_allocated > 100 * 1024);

    // Deallocate every other allocation
    for (i, alloc) in allocations.into_iter().enumerate() {
        if i % 2 == 0 {
            manager.deallocate(alloc).unwrap();
        }
    }

    // Check stats after partial deallocation
    let stats = manager.get_stats();
    assert_eq!(stats.deallocation_count, 50);
}

#[tokio::test]
async fn test_performance_validation() {
    let config = RuntimeConfig {
        optimization_level: 3,
        enable_profiling: true,
        ..Default::default()
    };

    let runtime = create_runtime(config).await.unwrap();

    // Large tensor operation for performance testing
    let mlir_code = r#"
        func @large_computation(%input: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
            %c1 = arith.constant 1.0 : f32
            %splat = tensor.splat %c1 : tensor<1024x1024xf32>
            %result = arith.addf %input, %splat : tensor<1024x1024xf32>
            return %result : tensor<1024x1024xf32>
        }
    "#;

    let module_id = runtime.compile_mlir("large_computation", mlir_code).await.unwrap();

    // Create large tensor (1M elements)
    let data: Vec<f32> = (0..1024*1024).map(|x| x as f32).collect();
    let large_tensor = runtime.create_tensor(data, vec![1024, 1024]).unwrap();

    let start = std::time::Instant::now();
    let outputs = runtime.execute(module_id, vec![large_tensor], None).await.unwrap();
    let duration = start.elapsed();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].numel(), 1024 * 1024);
    
    // Performance validation - should complete in reasonable time
    assert!(duration.as_millis() < 1000); // Under 1 second for placeholder implementation

    let stats = runtime.get_stats();
    assert!(stats.avg_execution_time_ms < 1000.0);
}