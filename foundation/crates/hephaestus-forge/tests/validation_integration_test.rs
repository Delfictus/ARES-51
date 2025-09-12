use hephaestus_forge::validation::MetamorphicTestSuite;
use hephaestus_forge::types::{ValidationConfig, VersionedModule, ModuleMetadata, ModuleId, PerformanceProfile};
use std::time::{Duration, Instant};
use tokio;

#[tokio::test]
async fn test_property_based_testing_performance() {
    let config = ValidationConfig {
        property_testing: true,
        differential_testing: false,
        chaos_engineering: false,
        parallel_threads: 4,
        max_test_cases_per_property: 10_000,
        memory_limit_mb: 512,
        regression_threshold: 0.1,
        shrinking_enabled: true,
        max_shrinking_attempts: 100,
        test_timeout_ms: 5000,
    };
    
    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
    
    let test_module = create_test_module();
    
    let start_time = Instant::now();
    let result = test_suite.validate_candidates(vec![test_module]).await;
    let elapsed = start_time.elapsed();
    
    assert!(result.is_ok(), "Validation should succeed");
    println!("Validation completed in {:?}", elapsed);
    
    // The test should complete in reasonable time
    assert!(elapsed < Duration::from_secs(10), "Validation should complete within 10 seconds");
}

#[tokio::test]
async fn test_differential_testing() {
    let config = ValidationConfig {
        property_testing: false,
        differential_testing: true,
        chaos_engineering: false,
        parallel_threads: 2,
        max_test_cases_per_property: 1_000,
        memory_limit_mb: 256,
        regression_threshold: 0.1,
        shrinking_enabled: false,
        max_shrinking_attempts: 50,
        test_timeout_ms: 3000,
    };
    
    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
    
    let test_module = create_test_module();
    
    let result = test_suite.validate_candidates(vec![test_module]).await;
    
    assert!(result.is_ok(), "Differential testing should succeed");
}

#[tokio::test]
async fn test_chaos_engineering() {
    let config = ValidationConfig {
        property_testing: false,
        differential_testing: false,
        chaos_engineering: true,
        parallel_threads: 1,
        max_test_cases_per_property: 100,
        memory_limit_mb: 256,
        regression_threshold: 0.2,
        shrinking_enabled: false,
        max_shrinking_attempts: 10,
        test_timeout_ms: 15000, // Chaos tests need more time
    };
    
    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
    
    let test_module = create_test_module();
    
    let result = test_suite.validate_candidates(vec![test_module]).await;
    
    assert!(result.is_ok(), "Chaos engineering tests should succeed");
}

#[tokio::test]
async fn test_comprehensive_validation() {
    let config = ValidationConfig {
        property_testing: true,
        differential_testing: true,
        chaos_engineering: true,
        parallel_threads: 4,
        max_test_cases_per_property: 1_000,
        memory_limit_mb: 512,
        regression_threshold: 0.1,
        shrinking_enabled: true,
        max_shrinking_attempts: 50,
        test_timeout_ms: 30000, // Comprehensive test needs more time
    };
    
    let test_suite = MetamorphicTestSuite::new(config).await.unwrap();
    
    let test_modules: Vec<VersionedModule> = (0..3).map(|i| {
        let mut module = create_test_module();
        module.metadata.id = ModuleId(format!("test_module_{}", i));
        module.metadata.name = format!("Test Module {}", i);
        module
    }).collect();
    
    let start_time = Instant::now();
    let result = test_suite.validate_candidates(test_modules).await;
    let elapsed = start_time.elapsed();
    
    assert!(result.is_ok(), "Comprehensive validation should succeed");
    println!("Comprehensive validation completed in {:?}", elapsed);
}

fn create_test_module() -> VersionedModule {
    VersionedModule {
        metadata: ModuleMetadata {
            id: ModuleId("test_module".to_string()),
            version: "1.0.0".to_string(),
            name: "Test Module".to_string(),
            description: "Module for testing validation system".to_string(),
            tags: vec!["test".to_string(), "validation".to_string()],
            created_at: chrono::Utc::now(),
            risk_score: 0.3,
            complexity_score: 2.5,
        },
        bytecode: vec![0u8; 2048], // 2KB test bytecode
        dependencies: vec![],
        performance_profile: PerformanceProfile {
            cpu_usage_percent: 15.0,
            memory_mb: 128,
            latency_p99_ms: 50.0,
            throughput_ops_per_sec: 1000,
        },
        safety_invariants: vec![],
        proof_certificate: None,
    }
}