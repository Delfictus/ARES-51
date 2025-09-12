//! Enterprise-grade integration tests for ARES neuromorphic CLI
//! 
//! Comprehensive test suite validating all enterprise requirements:
//! - Brian2/Lava backend integration
//! - Natural language processing accuracy
//! - Dynamic resource allocation
//! - C-LOGIC system integration
//! - Performance benchmarks
//! 
//! Author: Ididia Serfaty

use anyhow::Result;
use std::time::Duration;
use tokio::time::timeout;
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_cli_help_command() -> Result<()> {
    let mut cmd = Command::cargo_bin("ares")?;
    cmd.arg("--help");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ARES Neuromorphic"))
        .stdout(predicate::str::contains("enhanced"))
        .stdout(predicate::str::contains("interactive"));
    
    Ok(())
}

#[tokio::test]
async fn test_enhanced_mode_startup() -> Result<()> {
    // Test that enhanced mode starts without immediate crashes
    let mut cmd = Command::cargo_bin("ares")?;
    cmd.arg("enhanced")
        .timeout(Duration::from_secs(30))
        .kill_on_drop(true);
    
    // This should start the interactive mode
    // We can't easily test interactive input, but we can verify it doesn't crash immediately
    let output = cmd.output()?;
    
    // Should not have compilation errors
    assert!(!String::from_utf8_lossy(&output.stderr).contains("error:"));
    
    Ok(())
}

#[tokio::test]
async fn test_natural_language_query() -> Result<()> {
    let mut cmd = Command::cargo_bin("ares")?;
    cmd.arg("query")
        .arg("show system status");
    
    cmd.assert()
        .success();
    
    Ok(())
}

#[tokio::test]
async fn test_status_command() -> Result<()> {
    let mut cmd = Command::cargo_bin("ares")?;
    cmd.arg("status");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ARES"));
    
    Ok(())
}

#[tokio::test]
async fn test_neuromorphic_system_initialization() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    // Test system initialization with default config
    let _system = timeout(
        Duration::from_secs(30),
        UnifiedNeuromorphicSystem::initialize(None)
    ).await??;
    
    Ok(())
}

#[tokio::test]
async fn test_enhanced_system_initialization() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::EnhancedUnifiedNeuromorphicSystem;
    
    // Test enhanced system initialization
    let _enhanced_system = timeout(
        Duration::from_secs(30),
        EnhancedUnifiedNeuromorphicSystem::initialize(None)
    ).await??;
    
    Ok(())
}

#[tokio::test]
async fn test_backend_selection() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::{backend, hardware};
    use ares_neuromorphic_cli::neuromorphic::NeuromorphicSystemConfig;
    
    let hardware_caps = hardware::HardwareDetector::detect().await?;
    let config = NeuromorphicSystemConfig::default();
    
    let _backend = backend::auto_select_backend(&hardware_caps, &config).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_nlp_processing() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    // Test natural language command processing
    let test_inputs = vec![
        "show status",
        "check quantum metrics", 
        "optimize performance",
        "enable learning mode"
    ];
    
    for input in test_inputs {
        let intent = timeout(
            Duration::from_secs(10),
            system.process_natural_language(input)
        ).await??;
        
        // Verify intent is processed
        assert!(!intent.command.is_empty());
        assert!(intent.confidence >= 0.0 && intent.confidence <= 1.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_learning_mode_toggle() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    // Test learning mode toggle
    let initial_state = system.get_state().await;
    let initial_learning = initial_state.learning_active;
    
    let new_state = system.toggle_learning().await?;
    assert_ne!(initial_learning, new_state);
    
    // Toggle back
    let final_state = system.toggle_learning().await?;
    assert_eq!(initial_learning, final_state);
    
    Ok(())
}

#[tokio::test] 
async fn test_clogic_integration() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    // Test C-LOGIC system state access
    let clogic_state = system.get_clogic_state().await?;
    
    // Verify C-LOGIC modules are active
    assert!(clogic_state.drpp_state.coherence >= 0.0);
    assert!(clogic_state.drpp_state.coherence <= 1.0);
    
    Ok(())
}

#[tokio::test]
async fn test_performance_metrics() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    // Process several commands to generate metrics
    for i in 0..5 {
        let input = format!("test command {}", i);
        let _intent = system.process_natural_language(&input).await?;
    }
    
    let state = system.get_state().await;
    
    // Verify metrics are being tracked
    assert!(state.commands_processed >= 5);
    assert!(state.avg_processing_time_ms >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_resource_allocation() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    let state = system.get_state().await;
    
    // Verify resource allocation sums to reasonable total
    let total_allocation = state.resource_allocation.nlp + 
                          state.resource_allocation.drpp +
                          state.resource_allocation.ems +
                          state.resource_allocation.adp +
                          state.resource_allocation.egc;
    
    assert!(total_allocation >= 0.9);
    assert!(total_allocation <= 1.1); // Allow small rounding errors
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_loading() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    // Create temporary config file
    let mut config_file = NamedTempFile::new()?;
    writeln!(config_file, r#"
[brian2]
device = "cpu"
use_gpu = false

[lava]  
prefer_hardware = false
precision = "fp32"

[learning]
learning_rate = 0.005
confidence_threshold = 0.9
"#)?;
    
    let system = UnifiedNeuromorphicSystem::initialize(Some(config_file.path())).await?;
    let _state = system.get_state().await;
    
    Ok(())
}

/// Enterprise-grade stress test for neuromorphic processing
#[tokio::test]
async fn test_concurrent_processing() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    use std::sync::Arc;
    
    let system = Arc::new(UnifiedNeuromorphicSystem::initialize(None).await?);
    
    // Test concurrent command processing
    let mut handles = vec![];
    
    for i in 0..10 {
        let system_clone = Arc::clone(&system);
        let handle = tokio::spawn(async move {
            let input = format!("concurrent test command {}", i);
            system_clone.process_natural_language(&input).await
        });
        handles.push(handle);
    }
    
    // Wait for all concurrent operations
    for handle in handles {
        let _intent = handle.await??;
    }
    
    let final_state = system.get_state().await;
    assert!(final_state.commands_processed >= 10);
    
    Ok(())
}

/// Performance benchmark test
#[tokio::test]
async fn test_processing_latency_benchmark() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    use std::time::Instant;
    
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    // Benchmark processing latency
    let test_commands = vec![
        "show quantum status",
        "optimize system performance",
        "check temporal coherence", 
        "analyze neural patterns",
        "deploy configuration"
    ];
    
    let mut total_time = Duration::ZERO;
    
    for cmd in test_commands {
        let start = Instant::now();
        let _intent = system.process_natural_language(cmd).await?;
        let duration = start.elapsed();
        total_time += duration;
        
        // Enterprise requirement: <50ms per command
        assert!(duration.as_millis() < 50, 
               "Command '{}' took {}ms, exceeds 50ms limit", cmd, duration.as_millis());
    }
    
    let avg_time = total_time / 5;
    println!("Average processing time: {}ms", avg_time.as_millis());
    
    // Enterprise requirement: Average <20ms
    assert!(avg_time.as_millis() < 20, 
           "Average processing time {}ms exceeds 20ms enterprise requirement", 
           avg_time.as_millis());
    
    Ok(())
}

/// Memory usage validation test
#[tokio::test]
async fn test_memory_efficiency() -> Result<()> {
    use ares_neuromorphic_cli::neuromorphic::UnifiedNeuromorphicSystem;
    use sysinfo::{System, SystemExt, PidExt, ProcessExt};
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let initial_memory = if let Some(process) = sys.process(sysinfo::get_current_pid().unwrap()) {
        process.memory()
    } else {
        0
    };
    
    // Initialize system and process commands
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    for i in 0..50 {
        let cmd = format!("memory test command {}", i);
        let _intent = system.process_natural_language(&cmd).await?;
    }
    
    sys.refresh_all();
    let final_memory = if let Some(process) = sys.process(sysinfo::get_current_pid().unwrap()) {
        process.memory()
    } else {
        0
    };
    
    let memory_growth = final_memory.saturating_sub(initial_memory);
    
    // Enterprise requirement: <100MB growth for 50 commands
    assert!(memory_growth < 100 * 1024 * 1024, 
           "Memory growth {}MB exceeds 100MB enterprise limit", 
           memory_growth / (1024 * 1024));
    
    Ok(())
}