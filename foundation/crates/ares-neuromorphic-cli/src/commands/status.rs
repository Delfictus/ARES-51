//! Status command implementation

use anyhow::Result;
use clap::Args;
use colored::*;
use std::collections::HashMap;

#[cfg(not(feature = "status-only"))]
use crate::neuromorphic::UnifiedNeuromorphicSystem;
#[cfg(feature = "status-only")]
use crate::neuromorphic_status_only::UnifiedNeuromorphicSystem;
use crate::commands::{CommandResult, utils::*};

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Show detailed neuromorphic metrics
    #[arg(long)]
    pub detailed: bool,
    
    /// Show only specific subsystem
    #[arg(long, value_name = "SYSTEM")]
    pub subsystem: Option<String>,
    
    /// Output metrics in JSON format
    #[arg(long)]
    pub json: bool,
}

pub async fn execute(args: StatusArgs, system: UnifiedNeuromorphicSystem) -> CommandResult {
    if args.json {
        show_json_status(&system).await
    } else if args.detailed {
        show_detailed_status(&system, args.subsystem.as_deref()).await
    } else {
        show_basic_status(&system).await
    }
}

async fn show_basic_status(system: &UnifiedNeuromorphicSystem) -> CommandResult {
    println!("{}", "ARES Neuromorphic System Status".cyan().bold());
    println!("═══════════════════════════════════════");
    
    let state = system.get_state().await;
    
    // System overview
    quantum_message(&format!("Backend: {}", state.backend_info));
    
    // Quick metrics
    println!("📊 Quick Metrics:");
    println!("   Commands processed: {}", state.commands_processed);
    println!("   Avg processing time: {:.1}ms", state.avg_processing_time_ms);
    
    // Learning status
    if state.learning_active {
        neural_message("🧠 Learning mode: ACTIVE");
    } else {
        info_message("Learning mode: Inactive");
    }
    
    // Resource utilization bar
    println!("💾 Resource Allocation:");
    print_resource_bar("NLP", state.resource_allocation.nlp);
    print_resource_bar("DRPP", state.resource_allocation.drpp);
    print_resource_bar("EMS", state.resource_allocation.ems);
    
    Ok(())
}

async fn show_detailed_status(system: &UnifiedNeuromorphicSystem, subsystem: Option<&str>) -> CommandResult {
    println!("{}", "ARES Neuromorphic System - Detailed Status".cyan().bold());
    println!("═════════════════════════════════════════════════════");
    
    let state = system.get_state().await;
    
    match subsystem {
        Some("neuromorphic") => show_neuromorphic_details(system).await?,
        Some("clogic") => show_clogic_details(system).await?,
        Some("learning") => show_learning_details(system).await?,
        None => {
            show_neuromorphic_details(system).await?;
            println!();
            show_clogic_details(system).await?;
            println!();
            show_learning_details(system).await?;
        },
        Some(unknown) => {
            error_message(&format!("Unknown subsystem: {}", unknown));
            info_message("Available subsystems: neuromorphic, clogic, learning");
        }
    }
    
    Ok(())
}

async fn show_neuromorphic_details(system: &UnifiedNeuromorphicSystem) -> CommandResult {
    println!("{}", "🧠 Neuromorphic Backend Details".green().bold());
    println!("────────────────────────────────────────");
    
    let state = system.get_state().await;
    
    // Backend information
    println!("Backend Type: {}", state.backend_info.bold());
    
    // Performance metrics would come from the backend
    println!("Performance:");
    println!("  • Processing latency: {:.1}ms", state.avg_processing_time_ms);
    println!("  • Commands processed: {}", state.commands_processed);
    println!("  • Success rate: N/A"); // Would be calculated
    
    // Hardware utilization
    println!("Hardware:");
    println!("  • CPU utilization: N/A");
    println!("  • GPU utilization: N/A");
    println!("  • Memory usage: N/A");
    
    Ok(())
}

#[cfg(not(feature = "status-only"))]
async fn show_clogic_details(system: &UnifiedNeuromorphicSystem) -> CommandResult {
    println!("{}", "⚛️  C-LOGIC Module Status".magenta().bold());
    println!("─────────────────────────────────────");
    
    // Get C-LOGIC system state
    let clogic_state = system.get_clogic_state().await?;
    
    // DRPP status
    println!("{}", "DRPP (Dynamic Resonance Pattern Processor):".blue());
    println!("  • Oscillators: {}", clogic_state.drpp_state.oscillator_phases.len());
    println!("  • Coherence: {:.3}", clogic_state.drpp_state.coherence);
    println!("  • Detected patterns: {}", clogic_state.drpp_state.detected_patterns.len());
    
    // EMS status  
    println!("{}", "EMS (Emotional Modeling System):".blue());
    println!("  • Valence: {:.3}", clogic_state.ems_state.valence);
    println!("  • Arousal: {:.3}", clogic_state.ems_state.arousal);
    println!("  • Active emotions: {}", clogic_state.ems_state.active_emotions.len());
    println!("  • Mood: {:?}", clogic_state.ems_state.mood.mood_type);
    
    Ok(())
}

#[cfg(feature = "status-only")]
async fn show_clogic_details(_system: &UnifiedNeuromorphicSystem) -> CommandResult {
    println!("{}", "C-LOGIC Module Status".magenta().bold());
    println!("{}", "-".repeat(40));
    println!("DRPP: N/A (status-only build)");
    println!("EMS:  N/A (status-only build)");
    Ok(())
}

async fn show_learning_details(system: &UnifiedNeuromorphicSystem) -> CommandResult {
    println!("{}", "📚 Learning System Status".yellow().bold());
    println!("───────────────────────────────────────");
    
    let state = system.get_state().await;
    
    if state.learning_active {
        neural_message("Learning Mode: ACTIVE");
        
        // Learning metrics would come from the learning system
        println!("Session Statistics:");
        println!("  • Patterns learned: N/A");
        println!("  • Corrections made: N/A");
        println!("  • Average confidence: N/A");
        
        println!("Recent Activity:");
        println!("  • Last learning event: N/A");
        println!("  • Learning rate: N/A");
        println!("  • Pattern accuracy: N/A");
    } else {
        info_message("Learning Mode: Inactive");
        
        println!("Historical Data:");
        println!("  • Total patterns learned: N/A");
        println!("  • Overall accuracy: N/A");
        println!("  • Most confident patterns: N/A");
    }
    
    Ok(())
}

async fn show_json_status(system: &UnifiedNeuromorphicSystem) -> CommandResult {
    let state = system.get_state().await;
    
    let status = serde_json::json!({
        "neuromorphic_system": {
            "backend": state.backend_info,
            "learning_active": state.learning_active,
            "commands_processed": state.commands_processed,
            "avg_processing_time_ms": state.avg_processing_time_ms,
            "resource_allocation": {
                "nlp": state.resource_allocation.nlp,
                "drpp": state.resource_allocation.drpp,
                "ems": state.resource_allocation.ems,
                "adp": state.resource_allocation.adp,
                "egc": state.resource_allocation.egc
            }
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    println!("{}", serde_json::to_string_pretty(&status)?);
    
    Ok(())
}

fn print_resource_bar(name: &str, allocation: f64) {
    let percentage = (allocation * 100.0) as u8;
    let filled = (allocation * 20.0) as usize; // 20 character bar
    let empty = 20 - filled;
    
    let bar = format!("{}{}",
        "█".repeat(filled).green(),
        "░".repeat(empty).white()
    );
    
    println!("   {:<6} [{}] {:>3}%", name, bar, percentage);
}
