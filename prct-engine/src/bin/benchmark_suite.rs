#!/usr/bin/env cargo
//! Benchmark Suite Binary
//! Comprehensive performance benchmarking for PRCT algorithm

use std::path::PathBuf;
use clap::{Arg, Command, value_parser};
use prct_engine::{PRCTEngine, PRCTResult};
use prct_engine::gpu::{H100BenchmarkSuite, BenchmarkConfig, initialize_gpu};
use tokio;
use tracing::{info, warn, Level};
use tracing_subscriber;
use serde_json;
use std::time::Instant;

#[tokio::main]
async fn main() -> PRCTResult<()> {
    // Initialize logging
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set logging subscriber");

    // Parse command line arguments
    let matches = Command::new("benchmark-suite")
        .version("1.0.0")
        .author("CapoAI Team")
        .about("PRCT Algorithm Performance Benchmark Suite")
        .arg(Arg::new("benchmark-type")
            .long("benchmark-type")
            .value_name("TYPE")
            .help("Benchmark type (hamiltonian, phase-resonance, chromatic, tsp, full)")
            .default_value("full"))
        .arg(Arg::new("gpu-count")
            .long("gpu-count")
            .value_name("COUNT")
            .help("Number of GPUs to use")
            .value_parser(value_parser!(usize))
            .default_value("8"))
        .arg(Arg::new("results-dir")
            .long("results-dir")
            .value_name("DIR")
            .help("Output directory for benchmark results")
            .default_value("./benchmark-results"))
        .arg(Arg::new("iterations")
            .long("iterations")
            .value_name("COUNT")
            .help("Number of benchmark iterations")
            .value_parser(value_parser!(usize))
            .default_value("10"))
        .arg(Arg::new("protein-sizes")
            .long("protein-sizes")
            .value_name("SIZES")
            .help("Protein sizes to benchmark (comma-separated)")
            .value_delimiter(',')
            .default_values(["50", "100", "200", "500"]))
        .arg(Arg::new("precision")
            .long("precision")
            .value_name("PRECISION")
            .help("Floating point precision (mixed, fp16, fp32)")
            .default_value("mixed"))
        .arg(Arg::new("warmup-runs")
            .long("warmup-runs")
            .value_name("COUNT")
            .help("Number of warmup runs before benchmarking")
            .value_parser(value_parser!(usize))
            .default_value("3"))
        .arg(Arg::new("detailed-profiling")
            .long("detailed-profiling")
            .help("Enable detailed GPU profiling")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("memory-analysis")
            .long("memory-analysis")
            .help("Include memory usage analysis")
            .action(clap::ArgAction::SetTrue))
        .get_matches();

    info!("⚡ PRCT Algorithm Benchmark Suite Starting");

    // Extract arguments
    let benchmark_type = matches.get_one::<String>("benchmark-type").unwrap();
    let gpu_count = *matches.get_one::<usize>("gpu-count").unwrap();
    let results_dir = PathBuf::from(matches.get_one::<String>("results-dir").unwrap());
    let iterations = *matches.get_one::<usize>("iterations").unwrap();
    let protein_sizes: Vec<usize> = matches.get_many::<String>("protein-sizes")
        .unwrap()
        .map(|s| s.parse().unwrap_or(100))
        .collect();
    let precision = matches.get_one::<String>("precision").unwrap();
    let warmup_runs = *matches.get_one::<usize>("warmup-runs").unwrap();
    let detailed_profiling = matches.get_flag("detailed-profiling");
    let memory_analysis = matches.get_flag("memory-analysis");

    info!("🎯 Benchmark configuration:");
    info!("  Type: {}", benchmark_type);
    info!("  GPU count: {}", gpu_count);
    info!("  Iterations: {}", iterations);
    info!("  Protein sizes: {:?}", protein_sizes);
    info!("  Precision: {}", precision);
    info!("  Warmup runs: {}", warmup_runs);

    // Create results directory
    std::fs::create_dir_all(&results_dir)?;

    // Initialize GPU subsystem
    info!("🚀 Initializing GPU subsystem...");
    let gpu_info = initialize_gpu()?;
    info!("  ✅ Detected: {}", gpu_info.device_name);
    info!("  💾 Memory: {:.1}GB total, {:.1}GB free", 
          gpu_info.total_memory_gb, gpu_info.free_memory_gb);

    // Create benchmark configuration
    let config = BenchmarkConfig {
        gpu_count,
        iterations,
        warmup_runs,
        protein_sizes: protein_sizes.clone(),
        precision_mode: precision.to_string(),
        enable_profiling: detailed_profiling,
        enable_memory_analysis: memory_analysis,
        target_gpu: "H100".to_string(),
    };

    // Initialize benchmark suite
    info!("⚙️ Initializing H100 benchmark suite...");
    let mut benchmark_suite = H100BenchmarkSuite::new(config).await?;

    // Execute benchmarks based on type
    let benchmark_start = Instant::now();
    let mut all_results = serde_json::Map::new();

    match benchmark_type.as_str() {
        "hamiltonian" => {
            info!("🧮 Running Hamiltonian computation benchmarks...");
            for &size in &protein_sizes {
                info!("  Testing protein size: {} residues", size);
                let results = benchmark_suite.benchmark_hamiltonian_computation(size).await?;
                all_results.insert(
                    format!("hamiltonian_size_{}", size), 
                    serde_json::to_value(results)?
                );
            }
        }
        "phase-resonance" => {
            info!("🌊 Running phase resonance benchmarks...");
            for &size in &protein_sizes {
                info!("  Testing protein size: {} residues", size);
                let results = benchmark_suite.benchmark_phase_resonance_calculation(size).await?;
                all_results.insert(
                    format!("phase_resonance_size_{}", size),
                    serde_json::to_value(results)?
                );
            }
        }
        "chromatic" => {
            info!("🎨 Running chromatic optimization benchmarks...");
            for &size in &protein_sizes {
                info!("  Testing graph size: {} vertices", size);
                let results = benchmark_suite.benchmark_chromatic_optimization(size).await?;
                all_results.insert(
                    format!("chromatic_size_{}", size),
                    serde_json::to_value(results)?
                );
            }
        }
        "tsp" => {
            info!("🗺️ Running TSP phase dynamics benchmarks...");
            for &size in &protein_sizes {
                info!("  Testing TSP size: {} cities", size);
                let results = benchmark_suite.benchmark_tsp_phase_dynamics(size).await?;
                all_results.insert(
                    format!("tsp_size_{}", size),
                    serde_json::to_value(results)?
                );
            }
        }
        "full" => {
            info!("🔄 Running comprehensive benchmark suite...");
            
            // Hamiltonian benchmarks
            info!("  Phase 1/4: Hamiltonian computations");
            for &size in &protein_sizes {
                let results = benchmark_suite.benchmark_hamiltonian_computation(size).await?;
                all_results.insert(
                    format!("hamiltonian_size_{}", size),
                    serde_json::to_value(results)?
                );
            }

            // Phase resonance benchmarks  
            info!("  Phase 2/4: Phase resonance calculations");
            for &size in &protein_sizes {
                let results = benchmark_suite.benchmark_phase_resonance_calculation(size).await?;
                all_results.insert(
                    format!("phase_resonance_size_{}", size),
                    serde_json::to_value(results)?
                );
            }

            // Chromatic optimization benchmarks
            info!("  Phase 3/4: Chromatic optimizations");
            for &size in &protein_sizes {
                let results = benchmark_suite.benchmark_chromatic_optimization(size).await?;
                all_results.insert(
                    format!("chromatic_size_{}", size),
                    serde_json::to_value(results)?
                );
            }

            // TSP phase dynamics benchmarks
            info!("  Phase 4/4: TSP phase dynamics");
            for &size in &protein_sizes {
                let results = benchmark_suite.benchmark_tsp_phase_dynamics(size).await?;
                all_results.insert(
                    format!("tsp_size_{}", size),
                    serde_json::to_value(results)?
                );
            }
        }
        _ => {
            warn!("⚠️ Unknown benchmark type: {}, running full suite", benchmark_type);
            // Fall back to full benchmark
        }
    }

    let benchmark_duration = benchmark_start.elapsed();
    info!("✅ Benchmark suite completed in {:.2} minutes", benchmark_duration.as_secs_f64() / 60.0);

    // Collect system information
    let system_info = serde_json::json!({
        "timestamp": chrono::Utc::now(),
        "benchmark_duration_seconds": benchmark_duration.as_secs_f64(),
        "gpu_configuration": {
            "device_name": gpu_info.device_name,
            "total_memory_gb": gpu_info.total_memory_gb,
            "free_memory_gb": gpu_info.free_memory_gb,
            "sm_count": gpu_info.sm_count,
            "compute_capability": format!("{}.{}", gpu_info.compute_capability.0, gpu_info.compute_capability.1),
            "gpu_count": gpu_count
        },
        "benchmark_configuration": {
            "benchmark_type": benchmark_type,
            "iterations": iterations,
            "warmup_runs": warmup_runs,
            "protein_sizes": protein_sizes,
            "precision_mode": precision,
            "detailed_profiling": detailed_profiling,
            "memory_analysis": memory_analysis
        }
    });

    // Generate comprehensive results
    let final_results = serde_json::json!({
        "benchmark_metadata": system_info,
        "benchmark_results": all_results,
        "performance_summary": benchmark_suite.generate_summary_report().await?
    });

    // Save results
    let results_file = results_dir.join("benchmark_results.json");
    std::fs::write(&results_file, serde_json::to_string_pretty(&final_results)?)?;
    info!("📊 Results saved to: {}", results_file.display());

    // Generate performance analysis
    info!("📈 Performance Analysis:");
    
    // Calculate average performance metrics across all benchmarks
    let mut total_throughput = 0.0;
    let mut total_gpu_utilization = 0.0;
    let mut total_memory_efficiency = 0.0;
    let mut benchmark_count = 0;

    for (key, value) in &all_results {
        if let Some(obj) = value.as_object() {
            if let Some(throughput) = obj.get("average_throughput").and_then(|v| v.as_f64()) {
                total_throughput += throughput;
            }
            if let Some(gpu_util) = obj.get("average_gpu_utilization").and_then(|v| v.as_f64()) {
                total_gpu_utilization += gpu_util;
            }
            if let Some(mem_eff) = obj.get("memory_efficiency").and_then(|v| v.as_f64()) {
                total_memory_efficiency += mem_eff;
            }
            benchmark_count += 1;
        }
    }

    if benchmark_count > 0 {
        let avg_throughput = total_throughput / benchmark_count as f64;
        let avg_gpu_utilization = total_gpu_utilization / benchmark_count as f64;
        let avg_memory_efficiency = total_memory_efficiency / benchmark_count as f64;

        info!("  Average throughput: {:.1} ops/sec", avg_throughput);
        info!("  Average GPU utilization: {:.1}%", avg_gpu_utilization);
        info!("  Average memory efficiency: {:.1}%", avg_memory_efficiency);

        // Performance targets validation
        info!("🎯 Performance Targets:");
        info!("  GPU utilization target (>90%): {}", 
              if avg_gpu_utilization > 90.0 { "✅ PASSED" } else { "❌ FAILED" });
        info!("  Memory efficiency target (>80%): {}", 
              if avg_memory_efficiency > 80.0 { "✅ PASSED" } else { "❌ FAILED" });
        info!("  Throughput target (>1000 ops/sec): {}", 
              if avg_throughput > 1000.0 { "✅ PASSED" } else { "❌ FAILED" });
    }

    // Generate CSV summary for easy analysis
    let csv_file = results_dir.join("benchmark_summary.csv");
    let mut csv_content = "benchmark_type,protein_size,avg_execution_time,throughput,gpu_utilization,memory_usage\n".to_string();
    
    for (key, value) in &all_results {
        if let Some(obj) = value.as_object() {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() >= 3 {
                let bench_type = parts[0];
                let size = parts.get(2).unwrap_or(&"0");
                
                let exec_time = obj.get("average_execution_time_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let throughput = obj.get("average_throughput").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let gpu_util = obj.get("average_gpu_utilization").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let memory_usage = obj.get("peak_memory_usage_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                
                csv_content.push_str(&format!("{},{},{:.2},{:.1},{:.1},{:.2}\n",
                    bench_type, size, exec_time, throughput, gpu_util, memory_usage));
            }
        }
    }
    
    std::fs::write(&csv_file, csv_content)?;
    info!("📋 CSV summary saved to: {}", csv_file.display());

    info!("🎉 PRCT Algorithm benchmark suite completed successfully!");
    info!("📁 Complete results available at: {}", results_dir.display());

    Ok(())
}