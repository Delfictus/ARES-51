//! Comprehensive tests for enterprise hardening features

use hephaestus_forge::{
    HephaestusForge, ForgeConfigBuilder, OperationalMode,
    OptimizationIntent, Objective, Constraint, Priority, ForgeError,
};
use std::time::Duration;

#[tokio::test]
async fn test_rate_limiting() {
    // Create forge with a very strict rate limit for testing
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Supervised)
        .rate_limit(1.0, 1) // 1 rps, 1 burst
        .build()
        .expect("Failed to build config");
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    forge.start().await.expect("Failed to start forge");
    
    // Submit a burst of requests to trigger rate limiting
    for i in 0..3 { // Should allow 1, reject 2
        let intent = OptimizationIntent::builder()
            .target_module(&format!("burst_test_{}", i))
            .add_objective(Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .build()
            .expect("Failed to build intent");
        
        let _ = forge.submit_intent(intent).await;
    }

    // Wait for the processing to happen
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let status = forge.status().await;
    println!("Total optimizations after burst: {}", status.total_optimizations);

    // With a rate limit of 1 rps and a burst of 1, only 1 should have been processed.
    assert_eq!(status.total_optimizations, 1, "Rate limiter should have allowed only one intent to be processed");

    // Submit another request after the window, it should be accepted
    tokio::time::sleep(Duration::from_millis(1000)).await;
    let intent = OptimizationIntent::builder()
        .target_module("second_wave")
        .add_objective(Objective::MinimizeLatency {
            percentile: 99.0,
            target_ms: 10.0,
        })
        .build()
        .expect("Failed to build intent");
    let _ = forge.submit_intent(intent).await;

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let status = forge.status().await;
    assert_eq!(status.total_optimizations, 2, "A second intent should have been processed after the rate limit window");


    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_circuit_breaker() {
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Supervised)
        .build()
        .expect("Failed to build config");
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    forge.start().await.expect("Failed to start forge");
    
    // Submit intents that will fail
    for i in 0..6 {
        let intent = OptimizationIntent::builder()
            .target_module(&format!("nonexistent_module_{}", i))
            .add_objective(Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 0.001, // Impossible target
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(Priority::Low)
            .build()
            .expect("Failed to build intent");
        
        let result = forge.submit_intent(intent).await;
        
        if i >= 5 {
            // Circuit breaker should open after 5 failures
            match result {
                Err(e) => println!("Circuit breaker opened: {}", e),
                Ok(_) => println!("Intent {} submitted", i),
            }
        }
        
        // Wait a bit between submissions
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_chaos_engineering() {
    // Create forge with chaos engineering enabled
    let mut config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .build()
        .expect("Failed to build config");
    config.testing_config.chaos_engineering = true;
    
    let forge = HephaestusForge::new_async_public(config.clone()).await
        .expect("Failed to create forge");
    
    // Enable chaos mode with a high probability of failure
    forge.enable_chaos_engineering().await
        .expect("Failed to enable chaos engineering");
    
    if let Some(chaos_engine) = forge.chaos_engine.as_ref() {
        chaos_engine.set_failure_probability(1.0).await;
    }

    forge.start().await.expect("Failed to start forge");
    
    // Submit intents and see how system handles injected failures
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for i in 0..20 {
        let intent = OptimizationIntent::builder()
            .target_module(&format!("test_module_{}", i))
            .add_objective(Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(Priority::Medium)
            .build()
            .expect("Failed to build intent");
        
        let result = forge.submit_intent(intent).await;

        // The spawned task in submit_intent will return an error, but we can't easily await it here.
        // Instead, we can check the logs or the status of the system after a delay.
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let status = forge.status().await;
    println!("Total optimizations after chaos: {}", status.total_optimizations);

    // We expect that some intents failed to process due to chaos.
    // This is an indirect check. A better test would be to get the failure count from the forge.
    assert!(status.total_optimizations < 20, "Chaos should cause some intents to fail");

    forge.disable_chaos_engineering().await
        .expect("Failed to disable chaos");
    
    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_resource_monitoring() {
    let mut builder = ForgeConfigBuilder::new();
    builder = builder.mode(OperationalMode::Supervised);
    let mut config = builder.build().expect("Failed to build config");
    config.resource_limits.testing_memory_gb = 1; // Limit memory
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    forge.start().await.expect("Failed to start forge");
    
    // Try to submit memory-intensive intents
    for i in 0..5 {
        let intent = OptimizationIntent::builder()
            .target_module(&format!("large_module_{}", i))
            .add_objective(Objective::ReduceMemory {
                target_mb: 512,
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(Priority::High)
            .build()
            .expect("Failed to build intent");
        
        match forge.submit_intent(intent).await {
            Ok(id) => println!("Intent {} submitted: {:?}", i, id),
            Err(e) => println!("Intent {} rejected (resources?): {}", i, e),
        }
        
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    
    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_input_validation() {
    use hephaestus_forge::resonance::{DynamicResonanceProcessor, ComputationTensor};
    use nalgebra::DMatrix;
    
    let processor = DynamicResonanceProcessor::new((8, 8, 4)).await;
    
    // Test with invalid input (NaN values)
    let mut data = DMatrix::zeros(32, 32);
    data[(5, 5)] = f64::NAN;
    
    let tensor = ComputationTensor::from_matrix(data);
    let result = processor.process_via_resonance(tensor).await;
    
    // Should reject invalid input
    assert!(result.is_err(), "Should reject NaN input");
    
    // Test with extreme values
    let mut data = DMatrix::zeros(32, 32);
    data[(10, 10)] = 1e10; // Very large value
    
    let tensor = ComputationTensor::from_matrix(data);
    let result = processor.process_via_resonance(tensor).await;
    
    // Should handle or reject extreme values gracefully
    match result {
        Ok(solution) => {
            println!("Handled extreme value, coherence: {}", solution.coherence);
            assert!(solution.coherence <= 1.0, "Coherence should be bounded");
        },
        Err(e) => {
            println!("Rejected extreme value: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_audit_logging() {
    let mut builder = ForgeConfigBuilder::new();
    builder = builder.mode(OperationalMode::Supervised);
    let config = builder.build().expect("Failed to build config");
    // Audit logging is enabled by default
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    forge.start().await.expect("Failed to start forge");
    
    // Submit a few intents
    for i in 0..3 {
        let intent = OptimizationIntent::builder()
            .target_module(&format!("audit_test_{}", i))
            .add_objective(Objective::MinimizeLatency {
                percentile: 95.0,
                target_ms: 20.0,
            })
            .add_constraint(Constraint::MaintainCorrectness)
            .priority(Priority::Medium)
            .build()
            .expect("Failed to build intent");
        
        let result = forge.submit_intent(intent).await;
        println!("Intent {} result: {:?}", i, result);
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Audit logs should be written to file
    // In production, we'd verify the audit log contents
    
    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_graceful_degradation() {
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .enable_resonance_processing(true)
        .build()
        .expect("Failed to build config");
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    forge.start().await.expect("Failed to start forge");
    
    // Simulate various failure conditions
    let test_scenarios = vec![
        ("high_load", 50),     // Many requests
        ("complex", 5),        // Complex optimization
        ("invalid", 10),       // Invalid targets
    ];
    
    for (scenario, count) in test_scenarios {
        println!("Testing scenario: {}", scenario);
        
        for i in 0..count {
            let intent = match scenario {
                "high_load" => OptimizationIntent::builder()
                    .target_module(&format!("load_test_{}", i))
                    .add_objective(Objective::MaximizeThroughput {
                        target_ops_per_sec: 100000.0,
                    })
                    .priority(Priority::Low)
                    .build(),
                "complex" => OptimizationIntent::builder()
                    .target_module(&format!("complex_{}", i))
                    .add_objective(Objective::MinimizeLatency {
                        percentile: 99.99,
                        target_ms: 0.1,
                    })
                    .add_objective(Objective::ReduceMemory {
                        target_mb: 1,
                    })
                    .add_objective(Objective::MaximizeThroughput {
                        target_ops_per_sec: 1000000.0,
                    })
                    .add_constraint(Constraint::MaintainCorrectness)
                    .add_constraint(Constraint::RequireProof)
                    .priority(Priority::Critical)
                    .build(),
                _ => OptimizationIntent::builder()
                    .target_module(&format!("invalid_{}", i))
                    .add_objective(Objective::MinimizeLatency {
                        percentile: -1.0, // Invalid
                        target_ms: -10.0,  // Invalid
                    })
                    .priority(Priority::Low)
                    .build(),
            };
            
            if let Ok(intent) = intent {
                let _ = forge.submit_intent(intent).await;
            }
        }
        
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    
    // System should still be running despite various failures
    let status = forge.status().await;
    assert!(status.is_running, "System should degrade gracefully");
    
    forge.stop().await.expect("Failed to stop forge");
}
