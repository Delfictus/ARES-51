#!/usr/bin/env cargo
//! PRCT Algorithm Validator Binary
//! Executes complete PRCT algorithm validation on CASP16 targets

use std::path::PathBuf;
use clap::{Arg, Command, value_parser};
use prct_engine::{PRCTEngine, PRCTResult};
use prct_engine::data::{CASP16Loader, BlindTestProtocol};
use prct_engine::gpu::{H100PerformanceProfiler, initialize_gpu};
use tokio;
use tracing::{info, warn, error, Level};
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
    let matches = Command::new("prct-validator")
        .version("1.0.0")
        .author("CapoAI Team")
        .about("PRCT Algorithm Validation on CASP16 Dataset")
        .arg(Arg::new("casp16-data")
            .long("casp16-data")
            .value_name("DIR")
            .help("CASP16 data directory")
            .required(true))
        .arg(Arg::new("results-dir")
            .long("results-dir")
            .value_name("DIR")
            .help("Output directory for validation results")
            .default_value("./prct-results"))
        .arg(Arg::new("gpu-count")
            .long("gpu-count")
            .value_name("COUNT")
            .help("Number of GPUs to use")
            .value_parser(value_parser!(usize))
            .default_value("8"))
        .arg(Arg::new("precision")
            .long("precision")
            .value_name("PRECISION")
            .help("Floating point precision (mixed, fp16, fp32)")
            .default_value("mixed"))
        .arg(Arg::new("batch-size")
            .long("batch-size")
            .value_name("SIZE")
            .help("Batch size for processing (auto for automatic)")
            .default_value("auto"))
        .arg(Arg::new("log-performance")
            .long("log-performance")
            .help("Enable detailed performance logging")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("targets")
            .long("targets")
            .value_name("TARGETS")
            .help("Specific targets to validate (comma-separated)")
            .value_delimiter(','))
        .arg(Arg::new("blind-test")
            .long("blind-test")
            .help("Enable strict blind test protocol")
            .action(clap::ArgAction::SetTrue))
        .get_matches();

    info!("🧬 PRCT Algorithm Validation Starting");
    
    // Extract arguments
    let casp16_dir = PathBuf::from(matches.get_one::<String>("casp16-data").unwrap());
    let results_dir = PathBuf::from(matches.get_one::<String>("results-dir").unwrap());
    let gpu_count = *matches.get_one::<usize>("gpu-count").unwrap();
    let precision = matches.get_one::<String>("precision").unwrap();
    let batch_size = matches.get_one::<String>("batch-size").unwrap();
    let log_performance = matches.get_flag("log-performance");
    let enable_blind_test = matches.get_flag("blind-test");
    let specific_targets: Option<Vec<String>> = matches.get_many::<String>("targets")
        .map(|vals| vals.map(|s| s.to_string()).collect());

    info!("📁 CASP16 data: {}", casp16_dir.display());
    info!("📁 Results directory: {}", results_dir.display());
    info!("🖥️ GPU count: {}", gpu_count);
    info!("⚡ Precision: {}", precision);
    info!("📊 Batch size: {}", batch_size);

    // Create results directory
    std::fs::create_dir_all(&results_dir)?;
    
    // Initialize GPU subsystem
    info!("🚀 Initializing GPU subsystem...");
    let gpu_info = initialize_gpu()?;
    info!("  ✅ Detected: {}", gpu_info.device_name);
    info!("  💾 Memory: {:.1}GB total, {:.1}GB free", 
          gpu_info.total_memory_gb, gpu_info.free_memory_gb);

    // Initialize performance profiler if requested
    let mut profiler = if log_performance {
        info!("📈 Enabling performance profiling...");
        Some(H100PerformanceProfiler::new().await?)
    } else {
        None
    };

    // Initialize PRCT engine
    info!("⚙️ Initializing PRCT Algorithm Engine...");
    let prct_engine = PRCTEngine::new();
    info!("  Algorithm version: {}", prct_engine.algorithm_version);
    info!("  Energy tolerance: {:.2e}", prct_engine.energy_conservation_tolerance);
    info!("  Phase threshold: {:.3}", prct_engine.phase_coherence_threshold);

    // Load CASP16 data
    info!("📊 Loading CASP16 dataset...");
    let casp16_loader = CASP16Loader::new(casp16_dir.clone()).await?;
    let targets = match specific_targets {
        Some(target_list) => {
            info!("  Loading {} specific targets", target_list.len());
            casp16_loader.get_targets_by_ids(&target_list).await?
        }
        None => {
            info!("  Loading all available targets");
            casp16_loader.get_all_targets().await?
        }
    };
    info!("  ✅ Loaded {} targets for validation", targets.len());

    // Initialize blind test protocol if enabled
    let blind_test = if enable_blind_test {
        info!("🔐 Initializing blind test protocol...");
        let protocol = BlindTestProtocol::new(casp16_dir.clone()).await?;
        protocol.enforce_blind_conditions(&targets).await?;
        info!("  ✅ Blind test conditions enforced");
        Some(protocol)
    } else {
        None
    };

    // Execute validation
    info!("🎯 Starting PRCT algorithm validation...");
    let validation_start = Instant::now();
    
    let mut validation_results = Vec::new();
    let mut total_predictions = 0;
    let mut successful_predictions = 0;

    for target in &targets {
        info!("  Processing target: {}", target.target_id);
        
        // Start profiling for this target
        if let Some(ref mut prof) = profiler {
            prof.start_profiling(&target.target_id).await?;
        }

        // Execute PRCT folding
        match prct_engine.fold_protein(&target.sequence).await {
            Ok(structure) => {
                successful_predictions += 1;
                
                // Calculate performance metrics
                let gdt_ts_score = structure.calculate_gdt_ts_score();
                let execution_time = structure.computation_time_seconds;
                
                validation_results.push(serde_json::json!({
                    "target_id": target.target_id,
                    "sequence_length": target.sequence.len(),
                    "gdt_ts_score": gdt_ts_score,
                    "execution_time_seconds": execution_time,
                    "energy_conservation_error": structure.energy_conservation_error,
                    "phase_coherence": structure.phase_coherence,
                    "convergence_achieved": structure.converged,
                    "rmsd": structure.rmsd_to_native,
                    "timestamp": chrono::Utc::now()
                }));
                
                info!("    ✅ Success - GDT-TS: {:.2}, Time: {:.1}s", 
                      gdt_ts_score, execution_time);
            }
            Err(e) => {
                warn!("    ⚠️ Failed: {}", e);
                validation_results.push(serde_json::json!({
                    "target_id": target.target_id,
                    "error": e.to_string(),
                    "timestamp": chrono::Utc::now()
                }));
            }
        }

        // Stop profiling
        if let Some(ref mut prof) = profiler {
            prof.stop_profiling().await?;
        }
        
        total_predictions += 1;
    }

    let validation_duration = validation_start.elapsed();
    info!("✅ Validation completed in {:.2} minutes", validation_duration.as_secs_f64() / 60.0);

    // Generate comprehensive results
    let final_results = serde_json::json!({
        "validation_summary": {
            "total_targets": total_predictions,
            "successful_predictions": successful_predictions,
            "success_rate": (successful_predictions as f64 / total_predictions as f64) * 100.0,
            "total_duration_seconds": validation_duration.as_secs_f64(),
            "average_time_per_target": validation_duration.as_secs_f64() / total_predictions as f64,
            "validation_timestamp": chrono::Utc::now(),
            "prct_engine_version": prct_engine.algorithm_version,
            "gpu_configuration": {
                "device_name": gpu_info.device_name,
                "total_memory_gb": gpu_info.total_memory_gb,
                "gpu_count": gpu_count,
                "precision_mode": precision
            }
        },
        "predictions": validation_results,
        "performance_summary": if let Some(ref prof) = profiler {
            Some(prof.generate_summary_report().await?)
        } else {
            None
        }
    });

    // Save results
    let results_file = results_dir.join("casp16_validation_report.json");
    std::fs::write(&results_file, serde_json::to_string_pretty(&final_results)?)?;
    info!("📊 Results saved to: {}", results_file.display());

    // Calculate key metrics
    let successful_results: Vec<_> = validation_results.iter()
        .filter(|r| r.get("gdt_ts_score").is_some())
        .collect();
    
    if !successful_results.is_empty() {
        let mean_gdt_ts: f64 = successful_results.iter()
            .map(|r| r["gdt_ts_score"].as_f64().unwrap_or(0.0))
            .sum::<f64>() / successful_results.len() as f64;
        
        let mean_time: f64 = successful_results.iter()
            .map(|r| r["execution_time_seconds"].as_f64().unwrap_or(0.0))
            .sum::<f64>() / successful_results.len() as f64;

        info!("📈 Validation Summary:");
        info!("  Success rate: {:.1}%", (successful_predictions as f64 / total_predictions as f64) * 100.0);
        info!("  Mean GDT-TS score: {:.2}", mean_gdt_ts);
        info!("  Mean execution time: {:.1}s", mean_time);
        info!("  Total processing time: {:.1} minutes", validation_duration.as_secs_f64() / 60.0);
    }

    info!("🎯 PRCT Algorithm validation completed successfully!");
    info!("📁 Complete results available at: {}", results_dir.display());

    Ok(())
}