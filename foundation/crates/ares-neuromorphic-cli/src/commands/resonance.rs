//! Enterprise-grade resonance processing commands for the neuromorphic CLI
//! 
//! Provides interactive access to phase lattice computation

use anyhow::{Result, Context};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};
use tokio::time::sleep;

use crate::neuromorphic::{NeuromorphicCore, SpikeTrain};
use crate::phase_lattice::forge_bridge::{ForgeBridge, ResonantPattern};
use crate::commands::CommandResult;

/// Process input through resonance and display results
pub async fn process_resonance(
    core: &mut NeuromorphicCore,
    input: &str,
) -> Result<CommandResult> {
    println!("{}", "🌊 Resonance Processing".bright_cyan().bold());
    println!("{}", "━".repeat(50).bright_black());
    
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Invalid template")
            .progress_chars("#>-")
    );
    
    // Step 1: Encode input to spikes
    pb.set_message("Encoding to spike trains...");
    pb.set_position(20);
    
    let spike_train = core.encode_to_spikes(input).await?;
    let spike_count = spike_train.spike_times.len();
    
    println!("  {} Encoded {} spikes from input", 
             "✓".green(), spike_count);
    
    // Step 2: Convert spikes to phase states
    pb.set_message("Converting to phase lattice...");
    pb.set_position(40);
    
    let bridge = ForgeBridge::new();
    let phase_states = bridge.spikes_to_phase(&spike_train).await?;
    
    println!("  {} Generated {} phase states", 
             "✓".green(), phase_states.len());
    
    // Step 3: Process through resonance
    pb.set_message("Evolving resonance patterns...");
    pb.set_position(60);
    
    // Create input matrix from phase states
    use nalgebra::DMatrix;
    let size = (phase_states.len() as f64).sqrt().ceil() as usize;
    let mut matrix = DMatrix::zeros(size, size);
    
    for (i, state) in phase_states.iter().enumerate() {
        let row = i / size;
        let col = i % size;
        if row < size && col < size {
            matrix[(row, col)] = state.amplitude * state.frequency;
        }
    }
    
    let start = Instant::now();
    let pattern = bridge.process_via_resonance(matrix).await?;
    let resonance_time = start.elapsed();
    
    pb.set_message("Analyzing results...");
    pb.set_position(80);
    
    // Step 4: Convert back to spikes
    let output_spikes = bridge.phase_to_spikes(&phase_states).await?;
    
    pb.set_message("Complete!");
    pb.set_position(100);
    pb.finish_and_clear();
    
    // Display results
    display_resonance_results(&pattern, resonance_time, &output_spikes);
    
    // Store pattern for learning
    if pattern.coherence > 0.7 {
        println!("\n{} High coherence pattern detected! Storing for future use.",
                 "💡".yellow());
        // Store in learning system
        core.store_resonant_pattern(pattern.clone()).await?;
    }
    
    Ok(CommandResult {
        success: true,
        message: format!("Resonance processing complete ({}ms)", 
                        resonance_time.as_millis()),
        data: None,
    })
}

/// Analyze patterns using resonance
pub async fn analyze_resonance(
    core: &mut NeuromorphicCore,
    pattern_type: Option<&str>,
) -> Result<CommandResult> {
    println!("{}", "📊 Resonance Analysis".bright_cyan().bold());
    println!("{}", "━".repeat(50).bright_black());
    
    let bridge = ForgeBridge::new();
    
    // Get recent spike activity
    let recent_spikes = core.get_recent_activity(Duration::from_secs(5)).await?;
    
    if recent_spikes.is_empty() {
        return Ok(CommandResult {
            success: false,
            message: "No recent activity to analyze".to_string(),
            data: None,
        });
    }
    
    println!("Analyzing {} spike trains...", recent_spikes.len());
    
    let mut patterns = Vec::new();
    
    for (i, spike_train) in recent_spikes.iter().enumerate() {
        let spinner = ProgressBar::new_spinner();
        spinner.set_message(format!("Processing train {}...", i + 1));
        spinner.enable_steady_tick(Duration::from_millis(100));
        
        // Convert to phase and process
        let phase_states = bridge.spikes_to_phase(spike_train).await?;
        
        // Create matrix from states
        let size = (phase_states.len() as f64).sqrt().ceil() as usize;
        let mut matrix = DMatrix::zeros(size, size);
        
        for (j, state) in phase_states.iter().enumerate() {
            let row = j / size;
            let col = j % size;
            if row < size && col < size {
                matrix[(row, col)] = state.energy_level;
            }
        }
        
        let pattern = bridge.process_via_resonance(matrix).await?;
        patterns.push(pattern);
        
        spinner.finish_and_clear();
    }
    
    // Analyze patterns
    display_pattern_analysis(&patterns, pattern_type);
    
    Ok(CommandResult {
        success: true,
        message: format!("Analyzed {} patterns", patterns.len()),
        data: None,
    })
}

/// Visualize resonance in real-time
pub async fn visualize_resonance(
    core: &mut NeuromorphicCore,
    duration_secs: u64,
) -> Result<CommandResult> {
    println!("{}", "🎭 Real-time Resonance Visualization".bright_cyan().bold());
    println!("{}", "━".repeat(50).bright_black());
    
    let bridge = ForgeBridge::new();
    let end_time = Instant::now() + Duration::from_secs(duration_secs);
    
    println!("Monitoring for {} seconds...\n", duration_secs);
    
    while Instant::now() < end_time {
        // Get current activity
        let activity = core.get_instantaneous_activity().await?;
        
        if let Some(spike_train) = activity {
            // Process through resonance
            let phase_states = bridge.spikes_to_phase(&spike_train).await?;
            
            // Create simple visualization
            visualize_phase_lattice(&phase_states);
            
            // Calculate resonance metrics
            let total_energy: f64 = phase_states.iter()
                .map(|s| s.energy_level)
                .sum();
            
            let avg_coherence: f64 = phase_states.iter()
                .map(|s| s.coherence)
                .sum::<f64>() / phase_states.len() as f64;
            
            println!("\n{} Energy: {:.2} | Coherence: {:.2}%", 
                     "📊", total_energy, avg_coherence * 100.0);
        }
        
        sleep(Duration::from_millis(100)).await;
        
        // Clear for next frame (in production, use proper terminal control)
        print!("\x1B[2J\x1B[1;1H");
    }
    
    Ok(CommandResult {
        success: true,
        message: "Visualization complete".to_string(),
        data: None,
    })
}

/// Connect to Hephaestus Forge for enhanced processing
pub async fn connect_to_forge(
    core: &mut NeuromorphicCore,
    endpoint: &str,
) -> Result<CommandResult> {
    println!("{}", "🔗 Connecting to Hephaestus Forge".bright_cyan().bold());
    println!("{}", "━".repeat(50).bright_black());
    
    let mut bridge = ForgeBridge::new();
    
    // Attempt connection
    let pb = ProgressBar::new_spinner();
    pb.set_message(format!("Connecting to {}...", endpoint));
    pb.enable_steady_tick(Duration::from_millis(100));
    
    match bridge.connect_to_forge(endpoint).await {
        Ok(_) => {
            pb.finish_and_clear();
            println!("{} Successfully connected to Forge!", "✓".green().bold());
            println!("  Endpoint: {}", endpoint.bright_white());
            println!("  Status: {}", "Active".green());
            println!("  Mode: {}", "Enhanced Resonance Processing".yellow());
            
            // Store bridge in core
            core.set_forge_bridge(bridge).await?;
            
            Ok(CommandResult {
                success: true,
                message: format!("Connected to Forge at {}", endpoint),
                data: None,
            })
        },
        Err(e) => {
            pb.finish_and_clear();
            println!("{} Connection failed: {}", "✗".red().bold(), e);
            
            Ok(CommandResult {
                success: false,
                message: format!("Failed to connect: {}", e),
                data: None,
            })
        }
    }
}

/// Display resonance processing results
fn display_resonance_results(
    pattern: &ResonantPattern,
    processing_time: Duration,
    output_spikes: &SpikeTrain,
) {
    println!("\n{}", "📈 Resonance Results".bright_green().bold());
    println!("{}", "─".repeat(40).bright_black());
    
    // Display pattern characteristics
    println!("  {} {:.2} Hz", 
             "Frequency:".bright_white(), pattern.frequency);
    
    // Coherence with visual bar
    let coherence_bar = create_bar(pattern.coherence, 20);
    println!("  {} {} {:.1}%", 
             "Coherence:".bright_white(), 
             coherence_bar,
             pattern.coherence * 100.0);
    
    // Energy with visual indicator
    let energy_indicator = if pattern.energy > 5.0 {
        "⚡ High".yellow()
    } else if pattern.energy > 2.0 {
        "⚡ Medium".bright_yellow()
    } else {
        "⚡ Low".white()
    };
    println!("  {} {:.2} {}", 
             "Energy:".bright_white(), 
             pattern.energy,
             energy_indicator);
    
    // Topology signature
    println!("  {} {:?}", 
             "Topology:".bright_white(), 
             pattern.topology_signature);
    
    // Processing metrics
    println!("\n{}", "⚙️  Processing Metrics".bright_white());
    println!("  {} {}ms", 
             "Time:".bright_white(), 
             processing_time.as_millis());
    println!("  {} {} spikes", 
             "Output:".bright_white(), 
             output_spikes.spike_times.len());
    
    // Efficiency calculation
    let efficiency = pattern.coherence * (1.0 / (1.0 + processing_time.as_millis() as f64 / 1000.0));
    println!("  {} {:.2}%", 
             "Efficiency:".bright_white(), 
             efficiency * 100.0);
}

/// Display pattern analysis results
fn display_pattern_analysis(patterns: &[ResonantPattern], filter: Option<&str>) {
    println!("\n{}", "📊 Pattern Analysis".bright_green().bold());
    println!("{}", "─".repeat(40).bright_black());
    
    // Filter patterns if requested
    let filtered_patterns: Vec<_> = if let Some(pattern_type) = filter {
        patterns.iter().filter(|p| {
            match pattern_type {
                "high_freq" => p.frequency > 20.0,
                "low_freq" => p.frequency < 5.0,
                "coherent" => p.coherence > 0.8,
                "energetic" => p.energy > 3.0,
                _ => true,
            }
        }).collect()
    } else {
        patterns.iter().collect()
    };
    
    if filtered_patterns.is_empty() {
        println!("  No patterns match the filter criteria");
        return;
    }
    
    // Statistics
    let avg_frequency: f64 = filtered_patterns.iter()
        .map(|p| p.frequency)
        .sum::<f64>() / filtered_patterns.len() as f64;
    
    let avg_coherence: f64 = filtered_patterns.iter()
        .map(|p| p.coherence)
        .sum::<f64>() / filtered_patterns.len() as f64;
    
    let max_energy = filtered_patterns.iter()
        .map(|p| p.energy)
        .fold(0.0, f64::max);
    
    println!("  {} {}", 
             "Patterns found:".bright_white(), 
             filtered_patterns.len());
    println!("  {} {:.2} Hz", 
             "Avg frequency:".bright_white(), 
             avg_frequency);
    println!("  {} {:.1}%", 
             "Avg coherence:".bright_white(), 
             avg_coherence * 100.0);
    println!("  {} {:.2}", 
             "Max energy:".bright_white(), 
             max_energy);
    
    // Pattern distribution
    println!("\n  {}", "Frequency Distribution:".bright_white());
    let freq_histogram = create_frequency_histogram(&filtered_patterns);
    for (range, count) in freq_histogram {
        let bar = create_bar(count as f64 / filtered_patterns.len() as f64, 20);
        println!("    {:>6} Hz: {} {}", range, bar, count);
    }
}

/// Visualize phase lattice state
fn visualize_phase_lattice(phase_states: &[crate::phase_lattice::PhaseState]) {
    println!("{}", "🌐 Phase Lattice State".bright_cyan());
    
    let size = (phase_states.len() as f64).sqrt().ceil() as usize;
    let size = size.min(16); // Limit display size
    
    for row in 0..size {
        print!("  ");
        for col in 0..size {
            let idx = row * size + col;
            if idx < phase_states.len() {
                let state = &phase_states[idx];
                let intensity = (state.amplitude * 9.0).min(9.0) as u8;
                
                // Color based on coherence
                let symbol = format!("{}", intensity);
                let colored = if state.coherence > 0.8 {
                    symbol.bright_green()
                } else if state.coherence > 0.5 {
                    symbol.yellow()
                } else {
                    symbol.bright_black()
                };
                
                print!("{} ", colored);
            }
        }
        println!();
    }
}

/// Create a visual bar
fn create_bar(value: f64, width: usize) -> String {
    let filled = (value * width as f64) as usize;
    let empty = width - filled;
    
    format!("{}{}{}{}",
        "[".bright_black(),
        "█".repeat(filled).bright_cyan(),
        "░".repeat(empty).bright_black(),
        "]".bright_black())
}

/// Create frequency histogram
fn create_frequency_histogram(patterns: &[&ResonantPattern]) -> Vec<(String, usize)> {
    let mut histogram = vec![
        ("0-5".to_string(), 0),
        ("5-10".to_string(), 0),
        ("10-20".to_string(), 0),
        ("20-50".to_string(), 0),
        ("50+".to_string(), 0),
    ];
    
    for pattern in patterns {
        let idx = if pattern.frequency < 5.0 {
            0
        } else if pattern.frequency < 10.0 {
            1
        } else if pattern.frequency < 20.0 {
            2
        } else if pattern.frequency < 50.0 {
            3
        } else {
            4
        };
        histogram[idx].1 += 1;
    }
    
    histogram
}

/// Export resonance patterns for analysis
pub async fn export_patterns(
    core: &mut NeuromorphicCore,
    output_path: &str,
) -> Result<CommandResult> {
    println!("{}", "💾 Exporting Resonance Patterns".bright_cyan().bold());
    
    // Get stored patterns
    let patterns = core.get_stored_patterns().await?;
    
    if patterns.is_empty() {
        return Ok(CommandResult {
            success: false,
            message: "No patterns to export".to_string(),
            data: None,
        });
    }
    
    // Create JSON export
    use serde_json;
    let json = serde_json::to_string_pretty(&patterns)?;
    
    // Write to file
    std::fs::write(output_path, json)?;
    
    println!("{} Exported {} patterns to {}", 
             "✓".green(), 
             patterns.len(),
             output_path);
    
    Ok(CommandResult {
        success: true,
        message: format!("Exported {} patterns", patterns.len()),
        data: None,
    })
}