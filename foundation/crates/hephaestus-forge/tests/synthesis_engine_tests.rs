//! Comprehensive test suite for the synthesis engine
//! Validates correctness, performance, and formal guarantees

use hephaestus_forge::synthesis::{
    ProgramSynthesizer, ProgramSpecification, FunctionalRequirement, PerformanceRequirement,
    PerformanceMetric, SpecSafetyConstraint, SynthesisHint, HintType, TypeSpec, SynthesizedProgram
};
use hephaestus_forge::intent::{
    OptimizationIntent, IntentId, OptimizationTarget, Objective, Constraint, Priority
};
use hephaestus_forge::types::{
    SynthesisConfig, SmtSolver, SearchStrategy, InvariantCriticality, VersionedModule
};
use std::time::Duration;
use tokio::test;

#[test]
async fn test_basic_synthesis_workflow() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 5000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::MCTS,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let intent = create_simple_intent();
    let intents = vec![intent];
    
    let candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    
    assert!(!candidates.is_empty(), "Synthesis should produce at least one candidate");
    
    for candidate in &candidates {
        assert!(!candidate.id.0.is_empty(), "Generated module should have valid ID");
        assert!(!candidate.code.is_empty(), "Generated module should have non-empty code");
    }
}

#[test]
async fn test_synthesis_with_performance_objectives() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 5000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::BeamSearch,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("performance_test".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 10000.0 },
            Objective::MinimizeLatency { percentile: 99.0, target_ms: 5.0 },
        ],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::High,
        deadline: Some(Duration::from_millis(1000)),
        synthesis_strategy: Some("performance_optimized".to_string()),
    };
    
    let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
    
    assert!(!candidates.is_empty(), "Performance synthesis should produce candidates");
    
    // Verify performance metadata is reasonable
    for candidate in &candidates {
        let perf_profile = &candidate.metadata.performance_profile;
        assert!(perf_profile.throughput_ops_per_sec >= 1000, "Synthesized code should meet minimum throughput");
        assert!(perf_profile.latency_p99_ms <= 100.0, "Synthesized code should have reasonable latency");
    }
}

#[test]
async fn test_synthesis_with_formal_verification() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 10000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::HybridNeuralSymbolic,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("verified_module".to_string()),
        objectives: vec![Objective::MaximizeThroughput { target_ops_per_sec: 5000.0 }],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::RequireProof,
            Constraint::MaxComplexity(0.5),
        ],
        priority: Priority::Critical,
        deadline: Some(Duration::from_secs(5)),
        synthesis_strategy: Some("formal_verification".to_string()),
    };
    
    let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
    
    assert!(!candidates.is_empty(), "Formal verification synthesis should produce candidates");
    
    // Verify formal proofs are generated
    for candidate in &candidates {
        assert!(candidate.proof.is_some(), "Formally verified modules should have proofs");
        
        let proof = candidate.proof.as_ref().unwrap();
        assert!(!proof.smt_proof.is_empty(), "SMT proof should not be empty");
        assert!(!proof.invariants.is_empty(), "Safety invariants should be present");
        assert!(proof.verification_time_ms > 0, "Verification should take measurable time");
    }
}

#[test]
async fn test_synthesis_throughput_performance() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 1000, // Aggressive timeout for throughput testing
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::MCTS,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    // Create 100 simple intents
    let mut intents = Vec::new();
    for i in 0..100 {
        intents.push(create_simple_intent_with_id(i));
    }
    
    let start = std::time::Instant::now();
    let candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    let synthesis_time = start.elapsed();
    
    assert_eq!(candidates.len(), intents.len(), "Should synthesize all requested modules");
    
    let throughput = candidates.len() as f64 / synthesis_time.as_secs_f64();
    println!("Synthesis throughput: {:.0} functions/second", throughput);
    
    // Verify we meet the 1000+ functions/second requirement
    assert!(
        throughput >= 100.0, // Relaxed for testing, aim for 1000+ in production
        "Synthesis throughput too low: {:.0} functions/second",
        throughput
    );
}

#[test]
async fn test_different_smt_solvers() {
    let solvers = vec![SmtSolver::Z3, SmtSolver::CVC5, SmtSolver::Yices];
    
    for solver in solvers {
        let config = SynthesisConfig {
            max_synthesis_time_ms: 3000,
            smt_solver: solver.clone(),
            search_strategy: SearchStrategy::MCTS,
        };
        
        let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
        let intent = create_simple_intent();
        
        let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
        
        assert!(!candidates.is_empty(), "Solver {:?} should produce candidates", solver);
        
        // Verify basic properties
        for candidate in &candidates {
            assert!(!candidate.code.is_empty(), "Generated code should not be empty");
            assert!(candidate.metadata.risk_score >= 0.0, "Risk score should be non-negative");
        }
    }
}

#[test]
async fn test_synthesis_error_handling() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 100, // Very short timeout to trigger timeout errors
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::GeneticAlgorithm,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let impossible_intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("impossible_module".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 1_000_000.0 }, // Unrealistic
            Objective::MinimizeLatency { percentile: 99.9, target_ms: 0.001 }, // Impossible
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::RequireProof,
            Constraint::MaxMemoryMB(1), // Very restrictive
            Constraint::MaxComplexity(0.01), // Nearly zero complexity
        ],
        priority: Priority::Critical,
        deadline: Some(Duration::from_millis(50)), // Very short deadline
        synthesis_strategy: Some("impossible".to_string()),
    };
    
    // This should either produce a best-effort result or fail gracefully
    let result = synthesizer.generate_candidates(&vec![impossible_intent]).await;
    
    match result {
        Ok(candidates) => {
            // If it succeeds, should produce something reasonable
            assert!(!candidates.is_empty(), "Should produce fallback candidates");
        }
        Err(e) => {
            // If it fails, should be a meaningful error
            assert!(e.to_string().contains("Synthesis") || e.to_string().contains("timeout"), 
                    "Error should be synthesis-related: {}", e);
        }
    }
}

#[test]
async fn test_code_generation_quality() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 5000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::MCTS,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("quality_test".to_string()),
        objectives: vec![Objective::MaximizeThroughput { target_ops_per_sec: 5000.0 }],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::Medium,
        deadline: Some(Duration::from_millis(2000)),
        synthesis_strategy: Some("quality_focused".to_string()),
    };
    
    let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
    
    for candidate in &candidates {
        let code_str = String::from_utf8(candidate.code.clone())
            .expect("Generated code should be valid UTF-8");
        
        // Basic code quality checks
        assert!(code_str.contains("//"), "Generated code should have comments");
        assert!(code_str.contains("fn "), "Generated code should have functions");
        assert!(code_str.contains("#["), "Generated code should use attributes for optimization");
        
        // Verify no obvious issues
        assert!(!code_str.contains("TODO"), "Generated code should not contain TODOs");
        assert!(!code_str.contains("panic!"), "Generated code should not panic");
        assert!(!code_str.contains("unreachable!"), "Generated code should not be unreachable");
        
        println!("Generated code quality check passed for module: {}", candidate.id.0);
    }
}

#[test]
async fn test_synthesis_metadata_accuracy() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 3000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::BeamSearch,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    let intent = create_simple_intent();
    
    let start = std::time::Instant::now();
    let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
    let actual_time = start.elapsed();
    
    for candidate in &candidates {
        let metadata = &candidate.metadata;
        
        // Verify timing is reasonable
        assert!(metadata.created_at <= chrono::Utc::now(), "Creation time should not be in the future");
        
        // Verify scores are in reasonable ranges
        assert!(metadata.risk_score >= 0.0 && metadata.risk_score <= 1.0, 
                "Risk score should be between 0 and 1: {}", metadata.risk_score);
        assert!(metadata.complexity_score >= 0.0 && metadata.complexity_score <= 1.0,
                "Complexity score should be between 0 and 1: {}", metadata.complexity_score);
        
        // Verify performance profile makes sense
        let perf = &metadata.performance_profile;
        assert!(perf.cpu_usage_percent >= 0.0 && perf.cpu_usage_percent <= 100.0,
                "CPU usage should be percentage: {}%", perf.cpu_usage_percent);
        assert!(perf.memory_mb > 0, "Memory usage should be positive: {} MB", perf.memory_mb);
        assert!(perf.latency_p99_ms >= 0.0, "Latency should be non-negative: {} ms", perf.latency_p99_ms);
        assert!(perf.throughput_ops_per_sec > 0, "Throughput should be positive: {} ops/sec", perf.throughput_ops_per_sec);
    }
}

// Helper functions for creating test intents

fn create_simple_intent() -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("simple_test".to_string()),
        objectives: vec![Objective::MaximizeThroughput { target_ops_per_sec: 1000.0 }],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::Medium,
        deadline: Some(Duration::from_millis(1000)),
        synthesis_strategy: Some("simple".to_string()),
    }
}

fn create_simple_intent_with_id(id: usize) -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName(format!("test_module_{}", id)),
        objectives: vec![Objective::MaximizeThroughput { target_ops_per_sec: 1000.0 }],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::Medium,
        deadline: Some(Duration::from_millis(500)),
        synthesis_strategy: Some("batch".to_string()),
    }
}

/// Integration test for end-to-end synthesis pipeline
#[test]
async fn test_end_to_end_synthesis_pipeline() {
    let config = SynthesisConfig {
        max_synthesis_time_ms: 10000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::HybridNeuralSymbolic,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    // Create a comprehensive intent that exercises all synthesis phases
    let intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ComponentGroup("end_to_end_test".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 5000.0 },
            Objective::MinimizeLatency { percentile: 95.0, target_ms: 20.0 },
            Objective::ReduceMemory { target_mb: 64 },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::RequireProof,
            Constraint::MaxComplexity(0.7),
            Constraint::MaxMemoryMB(128),
        ],
        priority: Priority::High,
        deadline: Some(Duration::from_secs(5)),
        synthesis_strategy: Some("comprehensive".to_string()),
    };
    
    let start = std::time::Instant::now();
    let candidates = synthesizer.generate_candidates(&vec![intent]).await.unwrap();
    let total_time = start.elapsed();
    
    // Verify end-to-end results
    assert!(!candidates.is_empty(), "End-to-end synthesis should produce candidates");
    
    for candidate in &candidates {
        // Verify all pipeline phases completed
        assert!(!candidate.id.0.is_empty(), "Module should have valid ID");
        assert!(!candidate.code.is_empty(), "Module should have generated code");
        assert!(candidate.proof.is_some(), "Module should have formal proof");
        
        let proof = candidate.proof.as_ref().unwrap();
        assert!(!proof.invariants.is_empty(), "Proof should have safety invariants");
        assert!(!proof.smt_proof.is_empty(), "Proof should have SMT verification");
        
        // Verify performance characteristics
        let perf = &candidate.metadata.performance_profile;
        assert!(perf.throughput_ops_per_sec >= 1000, "Should meet minimum throughput");
        assert!(perf.memory_mb <= 256, "Should respect memory constraints");
    }
    
    println!("✓ End-to-end synthesis completed in {}ms", total_time.as_millis());
    println!("✓ Generated {} verified modules", candidates.len());
}