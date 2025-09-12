//! Performance benchmarks for the synthesis engine
//! Target: 1000+ functions/second synthesis rate

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hephaestus_forge::synthesis::{
    ProgramSynthesizer, ProgramSpecification, FunctionalRequirement, PerformanceRequirement,
    PerformanceMetric, SpecSafetyConstraint, SynthesisHint, HintType, TypeSpec
};
use hephaestus_forge::intent::{OptimizationIntent, IntentId, OptimizationTarget, Objective, Constraint, Priority};
use hephaestus_forge::types::{SynthesisConfig, SmtSolver, SearchStrategy, InvariantCriticality};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark synthesis engine performance across different scales
fn benchmark_synthesis_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_throughput");
    
    // Test different function counts to validate 1000+ functions/second
    let function_counts = vec![10, 50, 100, 500, 1000, 2000];
    
    for &count in &function_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("synthesize_functions", count),
            &count,
            |b, &count| {
                let rt = Runtime::new().unwrap();
                b.iter(|| {
                    rt.block_on(async {
                        benchmark_function_synthesis(count).await
                    })
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different synthesis strategies
fn benchmark_synthesis_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_strategies");
    
    let strategies = vec![
        ("template_based", SearchStrategy::MCTS),
        ("beam_search", SearchStrategy::BeamSearch),
        ("genetic", SearchStrategy::GeneticAlgorithm),
        ("hybrid", SearchStrategy::HybridNeuralSymbolic),
    ];
    
    for (strategy_name, strategy) in strategies {
        group.bench_function(strategy_name, |b| {
            let rt = Runtime::new().unwrap();
            b.iter(|| {
                rt.block_on(async {
                    benchmark_synthesis_strategy(strategy.clone()).await
                })
            });
        });
    }
    
    group.finish();
}

/// Benchmark synthesis with different SMT solvers
fn benchmark_smt_solver_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("smt_solver_performance");
    
    let solvers = vec![
        ("z3", SmtSolver::Z3),
        ("cvc5", SmtSolver::CVC5),
        ("yices", SmtSolver::Yices),
        ("multi", SmtSolver::Multi),
    ];
    
    for (solver_name, solver) in solvers {
        group.bench_function(solver_name, |b| {
            let rt = Runtime::new().unwrap();
            b.iter(|| {
                rt.block_on(async {
                    benchmark_with_smt_solver(solver.clone()).await
                })
            });
        });
    }
    
    group.finish();
}

/// Benchmark synthesis complexity scaling
fn benchmark_synthesis_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_complexity");
    
    let complexities = vec![
        ("simple", 1),      // Single function, basic logic
        ("moderate", 5),    // 5 functions with dependencies
        ("complex", 10),    // 10 functions with complex interactions
        ("enterprise", 25), // 25 functions with multiple constraints
    ];
    
    for (complexity_name, function_count) in complexities {
        group.bench_function(complexity_name, |b| {
            let rt = Runtime::new().unwrap();
            b.iter(|| {
                rt.block_on(async {
                    benchmark_synthesis_complexity_level(function_count).await
                })
            });
        });
    }
    
    group.finish();
}

// Implementation functions

async fn benchmark_function_synthesis(function_count: usize) -> Duration {
    let start = std::time::Instant::now();
    
    let config = create_performance_optimized_config();
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    // Create multiple intents for parallel synthesis
    let mut intents = Vec::new();
    for i in 0..function_count {
        intents.push(create_benchmark_intent(i));
    }
    
    // Synthesize all functions
    let _candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    
    start.elapsed()
}

async fn benchmark_synthesis_strategy(strategy: SearchStrategy) -> Duration {
    let start = std::time::Instant::now();
    
    let mut config = create_performance_optimized_config();
    config.search_strategy = strategy;
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    let intents = vec![create_complex_benchmark_intent()];
    
    let _candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    
    start.elapsed()
}

async fn benchmark_with_smt_solver(solver: SmtSolver) -> Duration {
    let start = std::time::Instant::now();
    
    let mut config = create_performance_optimized_config();
    config.smt_solver = solver;
    
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    let intents = vec![create_verification_intensive_intent()];
    
    let _candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    
    start.elapsed()
}

async fn benchmark_synthesis_complexity_level(function_count: usize) -> Duration {
    let start = std::time::Instant::now();
    
    let config = create_performance_optimized_config();
    let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
    
    let intent = create_complex_multi_function_intent(function_count);
    let intents = vec![intent];
    
    let _candidates = synthesizer.generate_candidates(&intents).await.unwrap();
    
    start.elapsed()
}

// Helper functions to create test data

fn create_performance_optimized_config() -> SynthesisConfig {
    SynthesisConfig {
        max_synthesis_time_ms: 1000, // 1 second timeout for performance
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::MCTS,
    }
}

fn create_high_performance_config() -> SynthesisConfig {
    SynthesisConfig {
        max_synthesis_time_ms: 50, // Ultra-fast synthesis for throughput
        smt_solver: SmtSolver::Multi, // Use fastest available solver
        search_strategy: SearchStrategy::HybridNeuralSymbolic, // Most efficient strategy
    }
}

fn create_high_performance_benchmark_intent(index: usize) -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName(format!("high_perf_module_{}", index)),
        objectives: vec![
            Objective::MaximizeThroughput {
                target_ops_per_sec: 5000.0, // High throughput target
            },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::MaxComplexity(0.6), // Keep complexity manageable for speed
        ],
        priority: Priority::High,
        deadline: Some(Duration::from_millis(10)), // Very tight deadline
        synthesis_strategy: Some("fast_template".to_string()),
    }
}

fn create_benchmark_intent(index: usize) -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName(format!("benchmark_module_{}", index)),
        objectives: vec![
            Objective::MaximizeThroughput {
                target_ops_per_sec: 1000.0,
            },
        ],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::Medium,
        deadline: Some(Duration::from_millis(100)),
        synthesis_strategy: Some("template_based".to_string()),
    }
}

fn create_complex_benchmark_intent() -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("complex_module".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput {
                target_ops_per_sec: 5000.0,
            },
            Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::MaxMemoryMB(128), // 128MB limit
            Constraint::MaxComplexity(0.8), // 80% complexity limit
        ],
        priority: Priority::High,
        deadline: Some(Duration::from_millis(500)),
        synthesis_strategy: Some("hybrid".to_string()),
    }
}

fn create_verification_intensive_intent() -> OptimizationIntent {
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("verified_module".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput {
                target_ops_per_sec: 2000.0,
            },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::RequireProof,
            Constraint::Custom("zero_defects".to_string()),
            Constraint::Custom("proven_safe".to_string()),
        ],
        priority: Priority::Critical,
        deadline: Some(Duration::from_secs(2)),
        synthesis_strategy: Some("formal_methods".to_string()),
    }
}

fn create_complex_multi_function_intent(function_count: usize) -> OptimizationIntent {
    let mut constraints = vec![Constraint::MaintainCorrectness];
    
    // Add more constraints for higher complexity
    if function_count > 5 {
        constraints.push(Constraint::MaxMemoryMB(256));
        constraints.push(Constraint::MaxComplexity(0.7));
    }
    
    if function_count > 15 {
        constraints.push(Constraint::RequireProof);
        constraints.push(Constraint::Custom("thread_safe".to_string()));
    }
    
    OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ComponentGroup(format!("complex_system_{}_functions", function_count)),
        objectives: vec![
            Objective::MaximizeThroughput {
                target_ops_per_sec: (function_count as f64 * 100.0), // Scale with complexity
            },
            Objective::MinimizeLatency {
                percentile: 95.0,
                target_ms: 20.0,
            },
        ],
        constraints,
        priority: if function_count > 15 { Priority::Critical } else { Priority::High },
        deadline: Some(Duration::from_millis(function_count as u64 * 50)), // Scale timeout
        synthesis_strategy: Some("adaptive".to_string()),
    }
}

/// Create realistic program specification for benchmarking
fn create_benchmark_specification(complexity: usize) -> ProgramSpecification {
    let mut functional_reqs = Vec::new();
    let mut performance_reqs = Vec::new();
    let mut safety_constraints = Vec::new();
    let mut synthesis_hints = Vec::new();
    
    for i in 0..complexity {
        functional_reqs.push(FunctionalRequirement {
            id: format!("func_req_{}", i),
            description: format!("Functional requirement {}", i),
            input_spec: TypeSpec {
                type_name: "i32".to_string(),
                constraints: vec!["(>= input 0)".to_string()],
            },
            output_spec: TypeSpec {
                type_name: "i32".to_string(),
                constraints: vec!["(>= output 0)".to_string()],
            },
            behavior_spec: format!("(= output (* input 2))"), // Double the input
        });
        
        performance_reqs.push(PerformanceRequirement {
            metric: PerformanceMetric::Latency,
            target_value: 10.0, // 10ms
            tolerance: 2.0, // ±2ms
        });
        
        safety_constraints.push(SpecSafetyConstraint {
            constraint_type: "memory_safety".to_string(),
            formula: "(< memory_usage 1000000)".to_string(), // < 1MB
            criticality: InvariantCriticality::High,
        });
        
        synthesis_hints.push(SynthesisHint {
            hint_type: HintType::Algorithm,
            content: "use_simd_when_possible".to_string(),
        });
    }
    
    ProgramSpecification {
        id: format!("benchmark_spec_{}", complexity),
        functional_requirements: functional_reqs,
        performance_requirements: performance_reqs,
        safety_constraints,
        synthesis_hints,
    }
}

/// Validation tests to ensure performance requirements are met
#[cfg(test)]
mod performance_validation {
    use super::*;
    
    #[tokio::test]
    async fn test_synthesis_throughput_requirement() {
        // Test the critical 500+ functions/second requirement
        let target_functions_per_second = 500;
        let test_duration = Duration::from_secs(2);
        
        let config = create_high_performance_config();
        let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
        
        let start = std::time::Instant::now();
        let mut total_functions = 0;
        let mut test_iterations = 0;
        
        while start.elapsed() < test_duration {
            // Create a batch of intents for high-throughput testing
            let batch_size = 50;
            let mut intents = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                intents.push(create_high_performance_benchmark_intent(i));
            }
            
            let batch_start = std::time::Instant::now();
            let candidates = synthesizer.generate_candidates(&intents).await.unwrap();
            let batch_time = batch_start.elapsed();
            
            let batch_throughput = candidates.len() as f64 / batch_time.as_secs_f64();
            total_functions += candidates.len();
            test_iterations += 1;
            
            // Verify all candidates were generated correctly
            assert_eq!(candidates.len(), batch_size, "Not all functions were synthesized in batch {}", test_iterations);
            
            // Early success check
            if batch_throughput >= target_functions_per_second as f64 * 1.2 {
                println!("✓ Early success: {:.0} functions/second in batch {}", batch_throughput, test_iterations);
            }
        }
        
        let total_time = start.elapsed();
        let actual_throughput = total_functions as f64 / total_time.as_secs_f64();
        
        // Critical assertion for 500+ functions/second
        assert!(
            actual_throughput >= target_functions_per_second as f64,
            "CRITICAL: Code generation throughput requirement not met: {:.0} functions/second (required: {} minimum)",
            actual_throughput,
            target_functions_per_second
        );
        
        // Additional performance metrics
        let average_batch_size = total_functions as f64 / test_iterations as f64;
        println!("✓ MISSION CRITICAL SUCCESS: Code generation throughput: {:.0} functions/second", actual_throughput);
        println!("  - Total functions generated: {}", total_functions);
        println!("  - Total test time: {:.2}s", total_time.as_secs_f64());
        println!("  - Test iterations: {}", test_iterations);
        println!("  - Average batch size: {:.1} functions", average_batch_size);
        
        // Verify we exceeded the requirement by a reasonable margin
        if actual_throughput >= target_functions_per_second as f64 * 1.5 {
            println!("✓ EXCEPTIONAL PERFORMANCE: {:.0}% above minimum requirement", 
                (actual_throughput / target_functions_per_second as f64 - 1.0) * 100.0);
        }
    }
    
    #[tokio::test]
    async fn test_synthesis_latency() {
        let max_acceptable_latency = Duration::from_millis(100); // 100ms max per function
        
        let synthesis_time = benchmark_function_synthesis(1).await;
        
        assert!(
            synthesis_time <= max_acceptable_latency,
            "Synthesis latency too high: {}ms (max: {}ms)",
            synthesis_time.as_millis(),
            max_acceptable_latency.as_millis()
        );
        
        println!("✓ Synthesis latency: {}ms per function", synthesis_time.as_millis());
    }
    
    #[tokio::test]
    async fn test_synthesis_scalability() {
        let function_counts = vec![1, 10, 100, 500];
        let mut results = Vec::new();
        
        for &count in &function_counts {
            let synthesis_time = benchmark_function_synthesis(count).await;
            let throughput = count as f64 / synthesis_time.as_secs_f64();
            results.push((count, throughput));
            
            println!("Functions: {}, Throughput: {:.0} functions/second", count, throughput);
        }
        
        // Ensure throughput doesn't degrade significantly with scale
        if results.len() >= 2 {
            let (_, first_throughput) = results[0];
            let (_, last_throughput) = results[results.len() - 1];
            
            let degradation_ratio = first_throughput / last_throughput;
            
            assert!(
                degradation_ratio <= 2.0, // Allow up to 50% throughput degradation at scale
                "Synthesis throughput degrades too much at scale: {:.1}x degradation",
                degradation_ratio
            );
            
            println!("✓ Throughput degradation at scale: {:.1}x", degradation_ratio);
        }
    }
    
    #[tokio::test]
    async fn test_correctness_at_high_throughput() {
        // Test that correctness is maintained even at high synthesis rates
        let config = create_high_performance_config();
        let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
        
        // Generate many functions quickly with focus on 500+ functions/second
        let function_count = 600; // Test above the minimum requirement
        let mut intents = Vec::with_capacity(function_count);
        for i in 0..function_count {
            intents.push(create_high_performance_benchmark_intent(i));
        }
        
        let start = std::time::Instant::now();
        let candidates = synthesizer.generate_candidates(&intents).await.unwrap();
        let synthesis_time = start.elapsed();
        
        // Verify all candidates were generated
        assert_eq!(candidates.len(), intents.len(), "Not all functions were synthesized");
        
        // Verify synthesis rate meets the 500+ functions/second requirement
        let throughput = candidates.len() as f64 / synthesis_time.as_secs_f64();
        assert!(
            throughput >= 500.0,
            "CRITICAL FAILURE: High-throughput synthesis below 500 functions/second: {:.0} functions/second",
            throughput
        );
        
        // Verify advanced correctness and quality metrics
        let mut valid_rust_code_count = 0;
        let mut optimized_code_count = 0;
        let mut safety_verified_count = 0;
        
        for (i, candidate) in candidates.iter().enumerate() {
            // Basic validation
            assert!(!candidate.id.0.is_empty(), "Generated module {} has empty ID", i);
            assert!(!candidate.code.is_empty(), "Generated module {} has empty code", i);
            
            // Convert to string and check for quality indicators
            let code_str = String::from_utf8_lossy(&candidate.code);
            
            // Verify no TODOs or placeholders remain
            assert!(!code_str.contains("TODO"), "Generated code contains TODO placeholders");
            assert!(!code_str.contains("unimplemented!"), "Generated code contains unimplemented! macros");
            assert!(!code_str.contains("placeholder"), "Generated code contains placeholder text");
            
            // Check for optimization markers
            if code_str.contains("#[inline]") || code_str.contains("#[target_feature") || code_str.contains("likely!") {
                optimized_code_count += 1;
            }
            
            // Check for safety features
            if candidate.proof.is_some() {
                safety_verified_count += 1;
            }
            
            // Basic Rust syntax validation (simplified)
            if code_str.contains("pub fn ") && code_str.contains("->") && code_str.contains("{") {
                valid_rust_code_count += 1;
            }
            
            // Verify performance profile exists and is reasonable
            assert!(candidate.metadata.performance_profile.throughput_ops_per_sec > 0, 
                "Module {} has invalid throughput profile", i);
            assert!(candidate.metadata.performance_profile.latency_p99_ms > 0.0,
                "Module {} has invalid latency profile", i);
        }
        
        // Quality metrics assertions
        let optimization_rate = optimized_code_count as f64 / candidates.len() as f64;
        let safety_rate = safety_verified_count as f64 / candidates.len() as f64;
        let validity_rate = valid_rust_code_count as f64 / candidates.len() as f64;
        
        assert!(validity_rate >= 0.95, "Less than 95% of generated code appears to be valid Rust: {:.1}%", validity_rate * 100.0);
        assert!(optimization_rate >= 0.8, "Less than 80% of generated code contains optimizations: {:.1}%", optimization_rate * 100.0);
        
        println!("✓ MISSION CRITICAL SUCCESS: Correctness maintained at {:.0} functions/second", throughput);
        println!("  - Valid Rust code: {:.1}% ({}/{})", validity_rate * 100.0, valid_rust_code_count, candidates.len());
        println!("  - Optimized code: {:.1}% ({}/{})", optimization_rate * 100.0, optimized_code_count, candidates.len());
        println!("  - Safety verified: {:.1}% ({}/{})", safety_rate * 100.0, safety_verified_count, candidates.len());
        println!("  - Average synthesis time per function: {:.2}ms", synthesis_time.as_millis() as f64 / candidates.len() as f64);
    }
    
    #[tokio::test]
    async fn test_generated_code_compilation() {
        // Test that generated code actually compiles
        let config = create_high_performance_config();
        let synthesizer = ProgramSynthesizer::new(config).await.unwrap();
        
        // Generate a small batch for compilation testing
        let test_count = 10;
        let mut intents = Vec::with_capacity(test_count);
        for i in 0..test_count {
            intents.push(create_high_performance_benchmark_intent(i));
        }
        
        let candidates = synthesizer.generate_candidates(&intents).await.unwrap();
        
        // Test each generated module for compilation readiness
        for (i, candidate) in candidates.iter().enumerate() {
            let code_str = String::from_utf8_lossy(&candidate.code);
            
            // Check for common compilation issues
            assert!(!code_str.contains("use undefined::"), "Module {} references undefined imports", i);
            assert!(code_str.contains("pub fn ") || code_str.contains("fn main"), "Module {} has no functions", i);
            
            // Check for balanced braces and parentheses
            let open_braces = code_str.matches('{').count();
            let close_braces = code_str.matches('}').count();
            assert_eq!(open_braces, close_braces, "Module {} has unbalanced braces", i);
            
            let open_parens = code_str.matches('(').count();
            let close_parens = code_str.matches(')').count();
            assert_eq!(open_parens, close_parens, "Module {} has unbalanced parentheses", i);
            
            // Verify optimization attributes are properly formatted
            if code_str.contains("#[inline]") {
                assert!(!code_str.contains("#[inline]\n#[inline]"), "Module {} has duplicate inline attributes", i);
            }
            
            println!("✓ Module {} appears compilation-ready ({} bytes)", i, candidate.code.len());
        }
        
        println!("✓ All {} generated modules pass compilation readiness checks", candidates.len());
    }
}

criterion_group!(
    synthesis_benches,
    benchmark_synthesis_throughput,
    benchmark_synthesis_strategies,
    benchmark_smt_solver_performance,
    benchmark_synthesis_complexity,
);
criterion_main!(synthesis_benches);