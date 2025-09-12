//! Demonstration of the Hephaestus Forge Synthesis Engine
//! Shows core synthesis capabilities and performance

use hephaestus_forge::synthesis::ProgramSynthesizer;
use hephaestus_forge::intent::{OptimizationIntent, IntentId, OptimizationTarget, Objective, Constraint, Priority};
use hephaestus_forge::types::{SynthesisConfig, SmtSolver, SearchStrategy};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Hephaestus Forge - Core Synthesis Engine Demo");
    println!("================================================\n");
    
    // Initialize synthesis engine
    let config = SynthesisConfig {
        max_synthesis_time_ms: 5000,
        smt_solver: SmtSolver::Z3,
        search_strategy: SearchStrategy::HybridNeuralSymbolic,
    };
    
    let synthesizer = ProgramSynthesizer::new(config).await?;
    println!("✓ Synthesis engine initialized with Z3 SMT solver\n");
    
    // Demo 1: Basic Function Synthesis
    println!("🧪 Demo 1: Basic Function Synthesis");
    println!("-----------------------------------");
    
    let basic_intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("math_utilities".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 10000.0 },
        ],
        constraints: vec![Constraint::MaintainCorrectness],
        priority: Priority::Medium,
        deadline: Some(Duration::from_secs(2)),
        synthesis_strategy: Some("template_based".to_string()),
    };
    
    let start = Instant::now();
    let candidates = synthesizer.generate_candidates(&vec![basic_intent]).await?;
    let synthesis_time = start.elapsed();
    
    println!("Synthesized {} candidates in {}ms", candidates.len(), synthesis_time.as_millis());
    
    for (i, candidate) in candidates.iter().enumerate() {
        println!("  Candidate {}: {} (risk: {:.2}, complexity: {:.2})", 
                 i + 1, 
                 candidate.id.0,
                 candidate.metadata.risk_score,
                 candidate.metadata.complexity_score);
        
        let code_str = String::from_utf8_lossy(&candidate.code);
        if !code_str.trim().is_empty() {
            println!("    Generated {} lines of code", code_str.lines().count());
        }
    }
    println!();
    
    // Demo 2: High-Performance Synthesis
    println!("⚡ Demo 2: High-Performance Synthesis");
    println!("------------------------------------");
    
    let performance_intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("high_performance_compute".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 50000.0 },
            Objective::MinimizeLatency { percentile: 99.0, target_ms: 5.0 },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::MaxMemoryMB(256),
        ],
        priority: Priority::High,
        deadline: Some(Duration::from_millis(1500)),
        synthesis_strategy: Some("performance_optimized".to_string()),
    };
    
    let start = Instant::now();
    let perf_candidates = synthesizer.generate_candidates(&vec![performance_intent]).await?;
    let perf_time = start.elapsed();
    
    println!("High-performance synthesis completed in {}ms", perf_time.as_millis());
    
    for candidate in &perf_candidates {
        let perf = &candidate.metadata.performance_profile;
        println!("  📊 Performance Profile:");
        println!("    Throughput: {} ops/sec", perf.throughput_ops_per_sec);
        println!("    Latency P99: {:.2}ms", perf.latency_p99_ms);
        println!("    Memory Usage: {}MB", perf.memory_mb);
        println!("    CPU Usage: {:.1}%", perf.cpu_usage_percent);
    }
    println!();
    
    // Demo 3: Formally Verified Synthesis
    println!("🔒 Demo 3: Formally Verified Synthesis");
    println!("-------------------------------------");
    
    let verified_intent = OptimizationIntent {
        id: IntentId::new(),
        target: OptimizationTarget::ModuleName("safety_critical_module".to_string()),
        objectives: vec![
            Objective::MaximizeThroughput { target_ops_per_sec: 5000.0 },
        ],
        constraints: vec![
            Constraint::MaintainCorrectness,
            Constraint::RequireProof,
            Constraint::MaxComplexity(0.6),
        ],
        priority: Priority::Critical,
        deadline: Some(Duration::from_secs(3)),
        synthesis_strategy: Some("formal_verification".to_string()),
    };
    
    let start = Instant::now();
    let verified_candidates = synthesizer.generate_candidates(&vec![verified_intent]).await?;
    let verified_time = start.elapsed();
    
    println!("Formal verification synthesis completed in {}ms", verified_time.as_millis());
    
    for candidate in &verified_candidates {
        if let Some(proof) = &candidate.proof {
            println!("  🛡️  Formal Proof:");
            println!("    Solver: {}", proof.solver_used);
            println!("    Verification Time: {}ms", proof.verification_time_ms);
            println!("    Safety Invariants: {}", proof.invariants.len());
            println!("    Proof Size: {} bytes", proof.smt_proof.len());
            
            for invariant in proof.invariants.iter().take(3) {
                println!("      - {} ({:?})", invariant.description, invariant.criticality);
            }
        }
    }
    println!();
    
    // Demo 4: Batch Synthesis Performance Test
    println!("🚀 Demo 4: Batch Synthesis Performance Test");
    println!("-------------------------------------------");
    
    let mut batch_intents = Vec::new();
    for i in 0..50 {
        batch_intents.push(OptimizationIntent {
            id: IntentId::new(),
            target: OptimizationTarget::ModuleName(format!("batch_module_{}", i)),
            objectives: vec![Objective::MaximizeThroughput { target_ops_per_sec: 2000.0 }],
            constraints: vec![Constraint::MaintainCorrectness],
            priority: Priority::Medium,
            deadline: Some(Duration::from_millis(500)),
            synthesis_strategy: Some("batch_optimized".to_string()),
        });
    }
    
    let start = Instant::now();
    let batch_candidates = synthesizer.generate_candidates(&batch_intents).await?;
    let batch_time = start.elapsed();
    
    let throughput = batch_candidates.len() as f64 / batch_time.as_secs_f64();
    println!("Batch synthesis results:");
    println!("  📦 Modules synthesized: {}", batch_candidates.len());
    println!("  ⏱️  Total time: {}ms", batch_time.as_millis());
    println!("  🏎️  Throughput: {:.0} functions/second", throughput);
    println!("  ✅ Success rate: {:.1}%", 
             (batch_candidates.len() as f64 / batch_intents.len() as f64) * 100.0);
    
    // Performance analysis
    if throughput >= 1000.0 {
        println!("  🎯 PERFORMANCE TARGET MET: >1000 functions/second");
    } else {
        println!("  ⚠️  Performance below target (1000 functions/second)");
    }
    println!();
    
    // Demo 5: Code Quality Analysis
    println!("📊 Demo 5: Code Quality Analysis");
    println!("--------------------------------");
    
    let mut total_code_lines = 0;
    let mut modules_with_proofs = 0;
    let mut avg_complexity = 0.0;
    let mut avg_risk_score = 0.0;
    
    let all_candidates = vec![&candidates, &perf_candidates, &verified_candidates, &batch_candidates]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    
    for candidate in &all_candidates {
        let code_str = String::from_utf8_lossy(&candidate.code);
        total_code_lines += code_str.lines().count();
        
        if candidate.proof.is_some() {
            modules_with_proofs += 1;
        }
        
        avg_complexity += candidate.metadata.complexity_score;
        avg_risk_score += candidate.metadata.risk_score;
    }
    
    let num_candidates = all_candidates.len();
    avg_complexity /= num_candidates as f64;
    avg_risk_score /= num_candidates as f64;
    
    println!("Quality metrics across all synthesized modules:");
    println!("  📝 Total lines of code generated: {}", total_code_lines);
    println!("  🔒 Modules with formal proofs: {}/{} ({:.1}%)",
             modules_with_proofs,
             num_candidates,
             (modules_with_proofs as f64 / num_candidates as f64) * 100.0);
    println!("  📈 Average complexity score: {:.2}", avg_complexity);
    println!("  ⚠️  Average risk score: {:.2}", avg_risk_score);
    
    if avg_risk_score <= 0.3 {
        println!("  ✅ LOW RISK: Synthesized code meets safety standards");
    } else if avg_risk_score <= 0.7 {
        println!("  🟡 MEDIUM RISK: Additional validation recommended");
    } else {
        println!("  🔴 HIGH RISK: Extensive testing required");
    }
    
    println!("\n🏁 Synthesis Engine Demo Complete!");
    println!("=====================================");
    
    // Summary
    let total_modules = num_candidates;
    let total_time = synthesis_time + perf_time + verified_time + batch_time;
    let overall_throughput = total_modules as f64 / total_time.as_secs_f64();
    
    println!("📋 Summary:");
    println!("  Total modules synthesized: {}", total_modules);
    println!("  Total synthesis time: {}ms", total_time.as_millis());
    println!("  Overall throughput: {:.0} functions/second", overall_throughput);
    println!("  SMT solver used: Z3");
    println!("  Search strategy: Hybrid Neural-Symbolic");
    
    Ok(())
}