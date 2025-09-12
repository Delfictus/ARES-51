use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use csf_mlir::*;
use csf_mlir::runtime::Tensor;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    for size in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &size| {
            let data: Vec<f32> = (0..size*size).map(|x| x as f32).collect();
            let shape = vec![size as i64, size as i64];
            
            b.iter(|| {
                black_box(Tensor::new(
                    black_box(data.clone()),
                    black_box(shape.clone()),
                    black_box(DataType::F32)
                ).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_runtime_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("runtime_creation", |b| {
        b.to_async(&rt).iter(|| async {
            let config = RuntimeConfig {
                memory_pool_size: 1024 * 1024,
                thread_pool_size: 4,
                ..Default::default()
            };
            
            black_box(create_runtime(black_box(config)).await.unwrap())
        });
    });
}

fn benchmark_mlir_compilation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("mlir_compilation", |b| {
        b.to_async(&rt).iter(|| async {
            let config = RuntimeConfig::default();
            let runtime = create_runtime(config).await.unwrap();
            
            let mlir_code = r#"
                func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                }
            "#;
            
            black_box(runtime.compile_mlir("bench_add", mlir_code).await.unwrap())
        });
    });
}

fn benchmark_execution_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("execution_throughput");
    
    for tensor_size in [16, 64, 256].iter() {
        group.bench_with_input(
            BenchmarkId::new("cpu_execution", tensor_size),
            tensor_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let config = RuntimeConfig::default();
                    let runtime = create_runtime(config).await.unwrap();
                    
                    let mlir_code = r#"
                        func @matrix_op(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
                            %c1 = arith.constant 1.0 : f32
                            %splat = tensor.splat %c1 : tensor<?x?xf32>
                            %result = arith.addf %input, %splat : tensor<?x?xf32>
                            return %result : tensor<?x?xf32>
                        }
                    "#;
                    
                    let module_id = runtime.compile_mlir("matrix_op", mlir_code).await.unwrap();
                    
                    let data: Vec<f32> = (0..size*size).map(|x| x as f32).collect();
                    let tensor = runtime.create_tensor(data, vec![size as i64, size as i64]).unwrap();
                    
                    black_box(
                        runtime.execute(module_id, vec![tensor], None).await.unwrap()
                    )
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_management(c: &mut Criterion) {
    use csf_mlir::memory::MemoryManager;
    use csf_mlir::runtime::DeviceLocation;
    
    let mut group = c.benchmark_group("memory_management");
    
    group.bench_function("allocation_1kb", |b| {
        let manager = MemoryManager::new(1024 * 1024).unwrap();
        
        b.iter(|| {
            let alloc = manager.allocate(
                black_box(1024),
                black_box(64),
                black_box(DeviceLocation::CPU)
            ).unwrap();
            
            black_box(manager.deallocate(alloc).unwrap())
        });
    });
    
    group.bench_function("allocation_64kb", |b| {
        let manager = MemoryManager::new(1024 * 1024).unwrap();
        
        b.iter(|| {
            let alloc = manager.allocate(
                black_box(64 * 1024),
                black_box(64),
                black_box(DeviceLocation::CPU)
            ).unwrap();
            
            black_box(manager.deallocate(alloc).unwrap())
        });
    });
    
    group.finish();
}

fn benchmark_backend_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("backend_selection", |b| {
        b.to_async(&rt).iter(|| async {
            use csf_mlir::backend::BackendSelector;
            
            let backends = vec![Backend::CPU, Backend::CUDA, Backend::Vulkan];
            let selector = BackendSelector::new(&backends);
            
            let module = MlirModule {
                name: "test_module".to_string(),
                id: ModuleId::new(),
                ir: "func @test() { return }".to_string(),
                artifact: None,
                metadata: ModuleMetadata {
                    inputs: vec![],
                    outputs: vec![],
                    flops: 1000000,
                    memory_bytes: 1024 * 1024,
                    parallelism: ParallelismInfo {
                        thread_count: 4,
                        simd_width: 8,
                        pipeline_depth: 2,
                    },
                },
            };
            
            black_box(selector.select(&module).await.unwrap())
        });
    });
}

fn benchmark_quantum_operations(c: &mut Criterion) {
    use csf_mlir::dialects::quantum::ops::*;
    use csf_mlir::dialects::quantum::transforms::*;
    
    let mut group = c.benchmark_group("quantum_operations");
    
    group.bench_function("gate_creation", |b| {
        b.iter(|| {
            black_box(QuantumGateOp {
                gate_type: black_box(GateType::H),
                qubits: black_box(vec![0]),
                parameters: black_box(vec![]),
            })
        });
    });
    
    group.bench_function("circuit_optimization", |b| {
        b.iter(|| {
            let mut circuit = CircuitOp {
                num_qubits: black_box(10),
                operations: black_box(vec![]),
            };
            
            let fusion_pass = GateFusionPass;
            black_box(fusion_pass.run(&mut circuit).unwrap())
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_tensor_creation,
    benchmark_runtime_creation,
    benchmark_mlir_compilation,
    benchmark_execution_throughput,
    benchmark_memory_management,
    benchmark_backend_selection,
    benchmark_quantum_operations
);

criterion_main!(benches);