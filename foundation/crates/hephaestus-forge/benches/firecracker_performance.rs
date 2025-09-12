use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hephaestus_forge::sandbox::*;
use hephaestus_forge::types::*;
use tokio::runtime::Runtime;
use std::time::Duration;

/// Benchmark Firecracker VM startup time - MUST be <50ms
fn bench_vm_startup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("firecracker_startup");
    group.sample_size(10); // Smaller sample size for expensive operations
    group.measurement_time(Duration::from_secs(30));
    
    // Test different VM configurations
    let configs = vec![
        ("minimal_128mb", SandboxConfig {
            isolation_type: IsolationType::FirecrackerVM,
            resource_limits: ResourceLimits {
                cpu_cores: 1.0,
                memory_mb: 128,
                disk_mb: 64,
                network_mbps: 10,
            },
            network_isolation: true,
        }),
        ("minimal_256mb", SandboxConfig {
            isolation_type: IsolationType::FirecrackerVM,
            resource_limits: ResourceLimits {
                cpu_cores: 1.0,
                memory_mb: 256,
                disk_mb: 64,
                network_mbps: 10,
            },
            network_isolation: true,
        }),
    ];
    
    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::new("vm_init", name), &config, |b, config| {
            b.to_async(&rt).iter(|| async {
                let sandbox = HardenedSandbox::new(config.clone()).await;
                black_box(sandbox)
            });
        });
    }
    
    group.finish();
}

/// Benchmark VM startup to execution ready time
fn bench_vm_full_startup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("firecracker_full_startup");
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(60));
    
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    group.bench_function("full_vm_ready", |b| {
        b.to_async(&rt).iter(|| async {
            let sandbox = HardenedSandbox::new(config.clone()).await.unwrap();
            
            // Create a minimal test module
            let test_module = VersionedModule {
                id: ModuleId("test".to_string()),
                version: 1,
                code: b"echo 'test'".to_vec(),
                proof: None,
                metadata: ModuleMetadata {
                    created_at: chrono::Utc::now(),
                    risk_score: 0.1,
                    complexity_score: 0.1,
                    performance_profile: PerformanceProfile {
                        cpu_usage_percent: 1.0,
                        memory_mb: 1,
                        latency_p99_ms: 10.0,
                        throughput_ops_per_sec: 100,
                    },
                },
            };
            
            let test_input = TestInput {
                data: b"test input".to_vec(),
                timeout_ms: 5000,
                environment: std::collections::HashMap::new(),
            };
            
            // Measure full execution pipeline
            let start = std::time::Instant::now();
            let result = sandbox.execute_module(&test_module, test_input).await;
            let duration = start.elapsed();
            
            // Verify performance requirement
            if duration > Duration::from_millis(50) {
                eprintln!("WARNING: VM startup took {}ms, exceeding 50ms target", duration.as_millis());
            }
            
            black_box((result, duration))
        });
    });
    
    group.finish();
}

/// Benchmark VM memory isolation overhead
fn bench_memory_isolation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_isolation");
    group.sample_size(20);
    
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    group.bench_function("memory_bounds_check", |b| {
        b.to_async(&rt).iter(|| async {
            let sandbox = HardenedSandbox::new(config.clone()).await.unwrap();
            
            // Test module that attempts to allocate memory beyond limits
            let test_module = VersionedModule {
                id: ModuleId("memory_test".to_string()),
                version: 1,
                code: b"dd if=/dev/zero of=/tmp/test bs=1M count=200".to_vec(), // Attempt to allocate 200MB
                proof: None,
                metadata: ModuleMetadata {
                    created_at: chrono::Utc::now(),
                    risk_score: 0.8,
                    complexity_score: 0.5,
                    performance_profile: PerformanceProfile {
                        cpu_usage_percent: 10.0,
                        memory_mb: 200,
                        latency_p99_ms: 1000.0,
                        throughput_ops_per_sec: 1,
                    },
                },
            };
            
            let test_input = TestInput {
                data: vec![],
                timeout_ms: 10000,
                environment: std::collections::HashMap::new(),
            };
            
            let result = sandbox.execute_module(&test_module, test_input).await;
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark network isolation effectiveness
fn bench_network_isolation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("network_isolation");
    group.sample_size(10);
    
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 1, // Very limited
        },
        network_isolation: true,
    };
    
    group.bench_function("network_exfiltration_test", |b| {
        b.to_async(&rt).iter(|| async {
            let sandbox = HardenedSandbox::new(config.clone()).await.unwrap();
            
            // Test module that attempts network access
            let test_module = VersionedModule {
                id: ModuleId("network_test".to_string()),
                version: 1,
                code: b"curl -m 2 http://google.com || echo 'BLOCKED'".to_vec(),
                proof: None,
                metadata: ModuleMetadata {
                    created_at: chrono::Utc::now(),
                    risk_score: 0.9,
                    complexity_score: 0.3,
                    performance_profile: PerformanceProfile {
                        cpu_usage_percent: 5.0,
                        memory_mb: 10,
                        latency_p99_ms: 2000.0,
                        throughput_ops_per_sec: 1,
                    },
                },
            };
            
            let test_input = TestInput {
                data: vec![],
                timeout_ms: 5000,
                environment: std::collections::HashMap::new(),
            };
            
            let result = sandbox.execute_module(&test_module, test_input).await;
            
            // Verify network was blocked
            if let Ok(output) = &result {
                if !String::from_utf8_lossy(&output.output).contains("BLOCKED") {
                    eprintln!("WARNING: Network isolation may have failed");
                }
            }
            
            black_box(result)
        });
    });
    
    group.finish();
}

/// Test container escape prevention
fn bench_escape_prevention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("escape_prevention");
    group.sample_size(5);
    
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    let escape_attempts = vec![
        ("proc_escape", b"cat /proc/1/environ".to_vec()),
        ("mount_escape", b"mount --bind /proc /tmp/proc".to_vec()),
        ("device_escape", b"ls -la /dev/".to_vec()),
        ("capability_escape", b"capsh --print".to_vec()),
    ];
    
    for (name, code) in escape_attempts {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let sandbox = HardenedSandbox::new(config.clone()).await.unwrap();
                
                let test_module = VersionedModule {
                    id: ModuleId(format!("escape_test_{}", name)),
                    version: 1,
                    code,
                    proof: None,
                    metadata: ModuleMetadata {
                        created_at: chrono::Utc::now(),
                        risk_score: 1.0, // Maximum risk
                        complexity_score: 0.5,
                        performance_profile: PerformanceProfile {
                            cpu_usage_percent: 5.0,
                            memory_mb: 10,
                            latency_p99_ms: 100.0,
                            throughput_ops_per_sec: 10,
                        },
                    },
                };
                
                let test_input = TestInput {
                    data: vec![],
                    timeout_ms: 2000,
                    environment: std::collections::HashMap::new(),
                };
                
                let result = sandbox.execute_module(&test_module, test_input).await;
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Performance regression test - ensure execution overhead is <1ms
fn bench_execution_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("execution_overhead");
    group.sample_size(50);
    
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 64,
            disk_mb: 32,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    group.bench_function("minimal_execution", |b| {
        b.to_async(&rt).iter(|| async {
            let sandbox = HardenedSandbox::new(config.clone()).await.unwrap();
            
            // Minimal test that should execute very fast
            let test_module = VersionedModule {
                id: ModuleId("minimal".to_string()),
                version: 1,
                code: b"echo 'ok'".to_vec(),
                proof: None,
                metadata: ModuleMetadata {
                    created_at: chrono::Utc::now(),
                    risk_score: 0.0,
                    complexity_score: 0.0,
                    performance_profile: PerformanceProfile {
                        cpu_usage_percent: 0.1,
                        memory_mb: 1,
                        latency_p99_ms: 1.0,
                        throughput_ops_per_sec: 1000,
                    },
                },
            };
            
            let test_input = TestInput {
                data: b"".to_vec(),
                timeout_ms: 100,
                environment: std::collections::HashMap::new(),
            };
            
            let start = std::time::Instant::now();
            let result = sandbox.execute_module(&test_module, test_input).await;
            let duration = start.elapsed();
            
            // Check execution overhead requirement
            if duration > Duration::from_millis(1) {
                eprintln!("WARNING: Execution overhead {}μs exceeds 1ms target", duration.as_micros());
            }
            
            black_box((result, duration))
        });
    });
    
    group.finish();
}

criterion_group!(
    firecracker_benches,
    bench_vm_startup,
    bench_vm_full_startup,
    bench_memory_isolation,
    bench_network_isolation,
    bench_escape_prevention,
    bench_execution_overhead
);

criterion_main!(firecracker_benches);