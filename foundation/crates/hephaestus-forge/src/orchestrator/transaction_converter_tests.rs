//! Test suite for transaction-to-module conversion system
//! Validates correctness, atomicity, and error handling

#[cfg(test)]
mod tests {
    use super::super::transaction_converter::*;
    use super::super::*;
    use crate::types::*;
    use crate::temporal::TemporalSwapCoordinator;
    use std::sync::Arc;
    use tokio;

    /// Create test transaction for performance enhancement
    fn create_test_performance_transaction() -> MetamorphicTransaction {
        MetamorphicTransaction {
            id: uuid::Uuid::new_v4(),
            module_id: ModuleId("test_module".to_string()),
            change_type: ChangeType::PerformanceEnhancement,
            risk_score: 0.3,
            proof: None,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Create test transaction with proof certificate
    fn create_test_security_transaction() -> MetamorphicTransaction {
        let proof = ProofCertificate {
            smt_proof: b"(assert (> throughput 1000))".to_vec(),
            invariants: vec![
                SafetyInvariant {
                    id: "security_invariant_1".to_string(),
                    description: "No buffer overflows".to_string(),
                    smt_formula: "(assert (< buffer_access buffer_size))".to_string(),
                    criticality: InvariantCriticality::Critical,
                }
            ],
            solver_used: "Z3".to_string(),
            verification_time_ms: 150,
        };
        
        MetamorphicTransaction {
            id: uuid::Uuid::new_v4(),
            module_id: ModuleId("secure_module".to_string()),
            change_type: ChangeType::SecurityPatch,
            risk_score: 0.8,
            proof: Some(proof),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Create test runtime config
    fn create_test_runtime_config() -> RuntimeConfig {
        RuntimeConfig {
            max_concurrent_swaps: 2,
            shadow_traffic_percent: 10.0,
            rollback_window_ms: 30000,
        }
    }
    
    #[tokio::test]
    async fn test_transaction_converter_initialization() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        
        let converter = TransactionModuleConverter::new(
            temporal_coordinator,
            config
        ).await;
        
        assert!(converter.is_ok(), "Transaction converter should initialize successfully");
    }
    
    #[tokio::test]
    async fn test_transaction_parsing() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_performance_transaction();
        let spec = converter.parse_transaction(&transaction).await;
        
        assert!(spec.is_ok(), "Transaction parsing should succeed");
        let spec = spec.unwrap();
        assert!(!spec.id.is_empty(), "Spec should have non-empty ID");
        assert!(spec.performance_targets.is_some(), "Performance transaction should have performance targets");
    }
    
    #[tokio::test]
    async fn test_transaction_to_intent_conversion() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_performance_transaction();
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&transaction, &spec).await;
        
        assert!(intent.is_ok(), "Transaction to intent conversion should succeed");
        let intent = intent.unwrap();
        
        // Verify intent properties
        assert_eq!(intent.target, OptimizationTarget::Module(transaction.module_id));
        assert!(!intent.objectives.is_empty(), "Intent should have objectives");
        assert!(intent.synthesis_strategy.is_some(), "Intent should have synthesis strategy");
    }
    
    #[tokio::test]
    async fn test_security_transaction_with_proof() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_security_transaction();
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&transaction, &spec).await.unwrap();
        
        // Verify security transaction specific properties
        assert_eq!(intent.priority, Priority::Critical, "Security patches should be critical priority");
        assert!(intent.deadline.is_some(), "Security patches should have tight deadlines");
        
        let deadline = intent.deadline.unwrap();
        assert!(deadline.as_secs() <= 300, "Security patch deadline should be <= 5 minutes");
    }
    
    #[tokio::test]
    async fn test_synthesis_from_intent() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_performance_transaction();
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&transaction, &spec).await.unwrap();
        
        let result = converter.synthesize_from_intent(&intent).await;
        assert!(result.is_ok(), "Module synthesis should succeed");
        
        let module = result.unwrap();
        assert_eq!(module.id.0.starts_with("synthesized_"), true, "Synthesized module should have correct ID prefix");
        assert!(!module.code.is_empty(), "Synthesized module should have non-empty code");
        assert!(module.metadata.risk_score <= 1.0, "Risk score should be valid");
        assert!(module.metadata.complexity_score <= 1.0, "Complexity score should be valid");
    }
    
    #[tokio::test]
    async fn test_deployment_strategy_selection() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator.clone(), config.clone()).await.unwrap();
        
        let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
        
        // Test different change types
        let test_cases = vec![
            (ChangeType::SecurityPatch, "Should use immediate deployment"),
            (ChangeType::PerformanceEnhancement, "Should use canary deployment"),
            (ChangeType::ArchitecturalRefactor, "Should use blue-green deployment"),
            (ChangeType::ModuleOptimization, "Should use shadow deployment"),
        ];
        
        for (change_type, description) in test_cases {
            let mut transaction = create_test_performance_transaction();
            transaction.change_type = change_type;
            
            let strategy = orchestrator.determine_deployment_strategy(&transaction);
            assert!(strategy.is_ok(), "{}: strategy selection should succeed", description);
            
            let strategy = strategy.unwrap();
            match change_type {
                ChangeType::SecurityPatch => {
                    assert!(matches!(strategy, DeploymentStrategy::Immediate), 
                        "Security patches should use immediate deployment");
                }
                ChangeType::PerformanceEnhancement => {
                    assert!(matches!(strategy, DeploymentStrategy::Canary { .. }), 
                        "Performance enhancements should use canary deployment");
                }
                ChangeType::ArchitecturalRefactor => {
                    assert!(matches!(strategy, DeploymentStrategy::BlueGreen), 
                        "Architectural refactors should use blue-green deployment");
                }
                ChangeType::ModuleOptimization => {
                    assert!(matches!(strategy, DeploymentStrategy::Shadow { .. }), 
                        "Module optimizations should use shadow deployment");
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_risk_based_constraint_generation() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        // Test high-risk transaction
        let mut high_risk_transaction = create_test_performance_transaction();
        high_risk_transaction.risk_score = 0.9;
        
        let spec = converter.parse_transaction(&high_risk_transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&high_risk_transaction, &spec).await.unwrap();
        
        // High-risk transactions should have formal verification constraints
        assert!(
            intent.constraints.len() >= 2,
            "High-risk transactions should have multiple constraints"
        );
        
        // Test low-risk transaction
        let mut low_risk_transaction = create_test_performance_transaction();
        low_risk_transaction.risk_score = 0.1;
        
        let spec = converter.parse_transaction(&low_risk_transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&low_risk_transaction, &spec).await.unwrap();
        
        // Low-risk transactions should have fewer constraints
        assert!(
            intent.constraints.len() >= 1,
            "All transactions should have at least basic correctness constraint"
        );
    }
    
    #[tokio::test]
    async fn test_candidate_selection_scoring() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_performance_transaction();
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&transaction, &spec).await.unwrap();
        
        // Create test candidates with different performance profiles
        let candidate1 = VersionedModule {
            id: ModuleId("candidate1".to_string()),
            version: 1,
            code: vec![1, 2, 3, 4],
            proof: None,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                risk_score: 0.2,
                complexity_score: 0.3,
                performance_profile: PerformanceProfile {
                    cpu_usage_percent: 20.0,
                    memory_mb: 128,
                    latency_p99_ms: 5.0,
                    throughput_ops_per_sec: 15000,
                },
            },
        };
        
        let candidate2 = VersionedModule {
            id: ModuleId("candidate2".to_string()),
            version: 1,
            code: vec![5, 6, 7, 8],
            proof: None,
            metadata: ModuleMetadata {
                created_at: chrono::Utc::now(),
                risk_score: 0.4,
                complexity_score: 0.5,
                performance_profile: PerformanceProfile {
                    cpu_usage_percent: 30.0,
                    memory_mb: 256,
                    latency_p99_ms: 10.0,
                    throughput_ops_per_sec: 8000,
                },
            },
        };
        
        let candidates = vec![candidate1.clone(), candidate2.clone()];
        let selected = converter.select_optimal_candidate(&candidates, &intent).await.unwrap();
        
        // Candidate1 should be selected (better performance profile)
        assert_eq!(selected.id, candidate1.id, "Should select candidate with better performance");
    }
    
    #[tokio::test]
    async fn test_deployment_validation() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let transaction = create_test_performance_transaction();
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        
        // Test successful deployment validation
        let successful_deployment = crate::orchestrator::SwapReport {
            module_id: transaction.module_id.clone(),
            old_version: Some(1),
            new_version: 2,
            strategy_used: "canary".to_string(),
            duration_ms: 5000,
            success: true,
            metrics: crate::orchestrator::SwapMetrics {
                error_rate: 0.01,
                latency_p99_ms: 8.0,
                throughput_change_percent: 15.0,
                memory_delta_mb: 32,
            },
        };
        
        let validation_result = converter.validate_deployment(&successful_deployment, &spec).await;
        assert!(validation_result.is_ok(), "Successful deployment should pass validation");
        
        // Test failed deployment validation
        let mut failed_deployment = successful_deployment.clone();
        failed_deployment.success = false;
        
        let validation_result = converter.validate_deployment(&failed_deployment, &spec).await;
        assert!(validation_result.is_err(), "Failed deployment should fail validation");
    }
    
    #[tokio::test]
    async fn test_atomic_checkpoint_operations() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        
        let module_ids = vec![
            ModuleId("module1".to_string()),
            ModuleId("module2".to_string()),
        ];
        
        // Test atomic checkpoint creation
        let checkpoint = temporal_coordinator.create_atomic_checkpoint(&module_ids).await;
        assert!(checkpoint.is_ok(), "Atomic checkpoint creation should succeed");
        
        let checkpoint = checkpoint.unwrap();
        assert_eq!(checkpoint.module_checkpoints.len(), 2, "Should create checkpoint for each module");
        assert!(!checkpoint.committed, "New checkpoint should not be committed");
        
        // Test checkpoint commit
        let commit_result = temporal_coordinator.commit_atomic_checkpoint(&checkpoint).await;
        assert!(commit_result.is_ok(), "Checkpoint commit should succeed");
        
        // Test checkpoint rollback (with a new checkpoint)
        let rollback_checkpoint = temporal_coordinator.create_atomic_checkpoint(&module_ids).await.unwrap();
        let rollback_result = temporal_coordinator.rollback_atomic_checkpoint(&rollback_checkpoint).await;
        assert!(rollback_result.is_ok(), "Checkpoint rollback should succeed");
    }
    
    #[tokio::test]
    async fn test_error_handling_and_recovery() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        // Test invalid transaction handling
        let mut invalid_transaction = create_test_performance_transaction();
        invalid_transaction.risk_score = 1.5; // Invalid risk score > 1.0
        
        // The system should handle this gracefully rather than panic
        let spec_result = converter.parse_transaction(&invalid_transaction).await;
        assert!(spec_result.is_ok(), "Parser should handle invalid risk scores gracefully");
    }
    
    #[tokio::test]
    async fn test_performance_regression_detection() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
        
        // Test performance regression detection
        let regression_metrics = crate::orchestrator::SwapMetrics {
            error_rate: 0.05,
            latency_p99_ms: 50.0,
            throughput_change_percent: -10.0, // Regression!
            memory_delta_mb: 128,
        };
        
        let module_id = ModuleId("test_module".to_string());
        let validation_result = orchestrator.validate_performance_improvement(
            &regression_metrics,
            &module_id
        ).await;
        
        assert!(validation_result.is_err(), "Performance regression should be detected and cause validation failure");
        
        // Test performance improvement validation
        let improvement_metrics = crate::orchestrator::SwapMetrics {
            error_rate: 0.01,
            latency_p99_ms: 5.0,
            throughput_change_percent: 20.0, // Improvement!
            memory_delta_mb: -32,
        };
        
        let validation_result = orchestrator.validate_performance_improvement(
            &improvement_metrics,
            &module_id
        ).await;
        
        assert!(validation_result.is_ok(), "Performance improvement should pass validation");
    }
    
    #[tokio::test]
    async fn test_comprehensive_integration_flow() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
        
        // Test complete transaction integration flow
        let transactions = vec![
            create_test_performance_transaction(),
            create_test_security_transaction(),
        ];
        
        let integration_result = orchestrator.integrate_changes(transactions).await;
        assert!(integration_result.is_ok(), "Transaction integration should succeed");
        
        let integrated_modules = integration_result.unwrap();
        assert_eq!(integrated_modules.len(), 2, "Should integrate all valid transactions");
    }
    
    #[tokio::test]
    async fn test_concurrent_transaction_processing() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let orchestrator = Arc::new(
            MetamorphicRuntimeOrchestrator::new(config).await.unwrap()
        );
        
        // Test concurrent transaction processing
        let mut handles = Vec::new();
        for i in 0..3 {
            let orchestrator_clone = orchestrator.clone();
            let mut transaction = create_test_performance_transaction();
            transaction.module_id = ModuleId(format!("concurrent_module_{}", i));
            
            let handle = tokio::spawn(async move {
                orchestrator_clone.integrate_changes(vec![transaction]).await
            });
            handles.push(handle);
        }
        
        // Wait for all concurrent transactions to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Concurrent transaction processing should succeed");
        }
    }
    
    #[tokio::test]
    async fn test_transaction_conversion_metrics() {
        let temporal_coordinator = Arc::new(
            TemporalSwapCoordinator::new().await.unwrap()
        );
        let config = create_test_runtime_config();
        let converter = TransactionModuleConverter::new(temporal_coordinator, config).await.unwrap();
        
        let start_time = std::time::Instant::now();
        let transaction = create_test_performance_transaction();
        
        // Measure conversion performance
        let spec = converter.parse_transaction(&transaction).await.unwrap();
        let intent = converter.transaction_to_intent(&transaction, &spec).await.unwrap();
        let _module = converter.synthesize_from_intent(&intent).await.unwrap();
        
        let conversion_time = start_time.elapsed();
        
        // Verify conversion completes within reasonable time
        assert!(
            conversion_time.as_millis() < 10000,
            "Transaction conversion should complete within 10 seconds, took {}ms",
            conversion_time.as_millis()
        );
        
        println!("Transaction conversion completed in {}ms", conversion_time.as_millis());
    }
}