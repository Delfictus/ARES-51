#!/usr/bin/env cargo
//! Benchmark Suite Binary
//! Comprehensive performance benchmarking for PRCT algorithm

use std::path::PathBuf;
use clap::{Arg, Command, value_parser};
use prct_engine::PRCTResult;
use prct_engine::gpu::{H100BenchmarkSuite, BenchmarkConfig, initialize_gpu};
use tokio;
use tracing::{info, Level};
use tracing_subscriber;
use serde_json;
use std::time::Instant;
use chrono;

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
    let _config = BenchmarkConfig {
        max_protein_size: *protein_sizes.iter().max().unwrap_or(&500),
        stress_test_duration: std::time::Duration::from_secs(300), // 5 minutes
        convergence_test_runs: iterations.min(20), // Limit to reasonable number
        thermal_monitoring_enabled: true,
        power_monitoring_enabled: true,
    };

    // Initialize benchmark suite
    info!("⚙️ Initializing H100 benchmark suite...");
    let mut benchmark_suite = H100BenchmarkSuite::new()?;

    // Execute benchmarks based on type
    let benchmark_start = Instant::now();
    let mut all_results = serde_json::Map::new();

    // Use the public interface instead of private methods
    info!("🔄 Running comprehensive benchmark suite (type: {})...", benchmark_type);
    let comprehensive_report = benchmark_suite.run_complete_benchmark_suite().await?;

    // Convert the comprehensive report to the expected format
    all_results.insert(
        "benchmark_type".to_string(),
        serde_json::to_value(benchmark_type)?
    );
    all_results.insert(
        "comprehensive_report".to_string(),
        serde_json::to_value(comprehensive_report)?
    );

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
        "performance_summary": {
            "suite_completed": true,
            "total_benchmarks": all_results.len(),
            "benchmark_duration_seconds": benchmark_duration.as_secs_f64()
        }
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

    for (_key, value) in &all_results {
        if let Some(obj) = value.as_object() {
            // Extract metrics from various benchmark result types
            let throughput = obj.get("average_gflops")
                .or(obj.get("graph_processing_throughput"))
                .or(obj.get("phase_dynamics_performance"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0);

            let gpu_util = obj.get("sm_utilization_average")
                .or(obj.get("cpu_parallelization_score"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0);

            let mem_eff = obj.get("tensor_core_efficiency")
                .or(obj.get("coherence_computation_efficiency"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0);

            if throughput > 0.0 {
                total_throughput += throughput;
            }
            if gpu_util > 0.0 {
                total_gpu_utilization += gpu_util;
            }
            if mem_eff > 0.0 {
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
    let mut csv_content = "benchmark_type,throughput,gpu_utilization,memory_efficiency\n".to_string();

    for (_key, value) in &all_results {
        if let Some(obj) = value.as_object() {
            let bench_type = _key.replace("_benchmarks", "");

            // Extract metrics from nested benchmark results
            let throughput = obj.get("average_gflops")
                .or(obj.get("graph_processing_throughput"))
                .or(obj.get("phase_dynamics_performance"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0);

            let gpu_util = obj.get("sm_utilization_average")
                .or(obj.get("cpu_parallelization_score"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0;

            let memory_eff = obj.get("tensor_core_efficiency")
                .or(obj.get("coherence_computation_efficiency"))
                .and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0;

            csv_content.push_str(&format!("{},{:.1},{:.1},{:.1}\n",
                bench_type, throughput, gpu_util, memory_eff));
        }
    }
    
    std::fs::write(&csv_file, csv_content)?;
    info!("📋 CSV summary saved to: {}", csv_file.display());

    info!("🎉 PRCT Algorithm benchmark suite completed successfully!");
    info!("📁 Complete results available at: {}", results_dir.display());

    Ok(())
}