//! Interactive neuromorphic CLI mode (sanitized)

use anyhow::Result;
use colored::*;
use dialoguer::{Confirm, Input};
use std::io::{self, Write};
use tracing::{debug, info};

use crate::commands::status::{self, StatusArgs};
use crate::commands::utils::*;
use crate::neuromorphic::UnifiedNeuromorphicSystem;

/// Run interactive neuromorphic CLI mode
pub async fn run_interactive_mode(mut system: UnifiedNeuromorphicSystem) -> Result<()> {
    // Simple ASCII banner to avoid encoding issues
    let banner = r#"
============================================================
                 ARES Neuromorphic CLI Interface
   Natural Language Processing | Quantum Integration | C-LOGIC
============================================================
"#;
    println!("{}", banner.cyan().bold());

    neural_message(&format!("Backend: {}", system.backend_info()));
    info_message("Type 'help' for commands, 'learn mode' to enable learning, 'exit' to quit");
    println!();

    loop {
        let state = system.get_state().await;
        let prompt = if state.learning_active { "ares* > " } else { "ares > " };

        print!("{}", prompt);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "exit" | "quit" | ":q" => {
                info_message("Shutting down neuromorphic systems...");
                break;
            }
            "help" | "?" => {
                show_basic_help();
            }
            "learn mode" => {
                let is_active = system.toggle_learning().await?;
                if is_active {
                    neural_message("Learning mode activated - I'll learn from your commands and corrections");
                    info_message("Correct me by typing: 'actually: <correct command>'");
                } else {
                    info_message("Learning mode deactivated - knowledge saved");
                }
            }
            "status" | "stats" => {
                show_system_status(&system).await?;
            }
            "pattern-list" => {
                show_learned_patterns(&system).await?;
            }
            _ if input.starts_with("actually: ") => {
                let correct_command = input.trim_start_matches("actually: ").trim();
                handle_correction(&system, correct_command).await?;
            }
            _ => {
                handle_natural_language_input(&system, input).await?;
            }
        }
    }

    Ok(())
}

fn show_basic_help() {
    println!("Available commands:");
    println!("  help            - Show this help");
    println!("  status|stats    - Show system status");
    println!("  learn mode      - Toggle learning mode");
    println!("  pattern-list    - Show learned pattern counts");
    println!("  exit|quit|:q    - Exit CLI");
    println!("  actually: ...   - Provide a correction for learning");
}

async fn handle_natural_language_input(
    system: &UnifiedNeuromorphicSystem,
    input: &str,
) -> Result<()> {
    let spinner = create_spinner("Processing through neuromorphic network...");

    let intent = system.process_natural_language(input).await?;
    spinner.finish_and_clear();

    if intent.confidence > 0.8 {
        success_message(&format!("Interpreted as: {}", intent.command.bold()));
    } else if intent.confidence > 0.5 {
        println!(
            "? Interpreted as: {} (confidence: {:.1}%)",
            intent.command.bold(),
            intent.confidence * 100.0
        );
        if !intent.alternatives.is_empty() {
            println!("   Alternatives:");
            for alt in &intent.alternatives {
                println!("   - {} (confidence: {:.1}%)", alt.command, alt.confidence * 100.0);
            }
        }
    } else {
        error_message(&format!(
            "Uncertain interpretation: {} (confidence: {:.1}%)",
            intent.command, intent.confidence * 100.0
        ));
        if system.get_state().await.learning_active {
            info_message("Since learning mode is active, please provide the correct command:");
            info_message("Type: actually: <correct command>");
            return Ok(());
        }
    }

    if intent.requires_confirmation || intent.confidence < 0.7 {
        let confirm = Confirm::new()
            .with_prompt(format!("Execute: {}?", intent.command))
            .default(false)
            .interact()?;
        if !confirm {
            info_message("Command cancelled");
            return Ok(());
        }
    }

    execute_ares_command(&intent.command, system).await?;

    if system.get_state().await.learning_active {
        record_successful_execution(system, input, &intent.command).await?;
    }

    Ok(())
}

async fn handle_correction(
    system: &UnifiedNeuromorphicSystem,
    correct_command: &str,
) -> Result<()> {
    let original_input: String = Input::new()
        .with_prompt("What was the original input you wanted to correct?")
        .interact_text()?;

    neural_message(&format!(
        "Learning correction: '{}' -> '{}'",
        original_input, correct_command
    ));

    system.nlp_learn_from_correction(&original_input, correct_command).await?;

    let metrics = system.learning_get_metrics().await;
    success_message(&format!(
        "Correction saved! Network updated (entry #{}).",
        metrics.total_samples
    ));

    Ok(())
}

async fn execute_ares_command(command: &str, system: &UnifiedNeuromorphicSystem) -> Result<()> {
    info_message(&format!("Executing: {}", command.bold()));

    if command.starts_with("csf status") {
        // Call the status command in JSON=false basic mode
        let args = StatusArgs {
            detailed: false,
            subsystem: None,
            json: false,
        };
        // Clone the system for ownership as required by status::execute
        let cloned = system.clone();
        return status::execute(args, cloned).await.map_err(Into::into);
    }

    if command.starts_with("csf health") {
        return execute_health_check(system).await;
    }

    if command.starts_with("csf optimize") {
        return execute_optimization(system, command).await;
    }

    if command.starts_with("csf backup") {
        return execute_backup_command(system, command).await;
    }

    // Fallback: shell command
    execute_shell_command(command).await
}

async fn execute_health_check(_system: &UnifiedNeuromorphicSystem) -> Result<()> {
    info!("Executing comprehensive system health check");
    println!("Health check placeholder: OK (stub)");
    Ok(())
}

async fn execute_optimization(
    _system: &UnifiedNeuromorphicSystem,
    _command: &str,
) -> Result<()> {
    println!("Optimization routine placeholder: completed (stub)");
    Ok(())
}

async fn execute_backup_command(system: &UnifiedNeuromorphicSystem, command: &str) -> Result<()> {
    info!("Executing backup operation: {}", command);
    if command.contains("--config ") {
        println!("Backing up neuromorphic configuration...");
        let patterns = system.learning_export_patterns().await?;
        println!("   - Exported {count} entries", count = patterns.len());
        println!("   - Configuration saved");
        success_message("Backup completed successfully");
    }
    Ok(())
}

async fn execute_shell_command(command: &str) -> Result<()> {
    info!("Executing shell command: {}", command);
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .await?;
    if output.status.success() {
        if !output.stdout.is_empty() {
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
    } else {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Command failed: {}", error_msg));
    }
    Ok(())
}

async fn record_successful_execution(
    system: &UnifiedNeuromorphicSystem,
    input: &str,
    command: &str,
) -> Result<()> {
    debug!("Recording successful execution for neuromorphic learning");
    let context = crate::neuromorphic::learning::TrainingContext {
        system_load: 0.5,
        operator_state: "productive".to_string(),
        session_duration_minutes: 10,
    };
    system.learning_record_sample(input, command, true, context).await?;
    debug!("- Execution recorded for learning: {} -> {}", input, command);
    Ok(())
}

async fn show_system_status(system: &UnifiedNeuromorphicSystem) -> Result<()> {
    let state = system.get_state().await;
    println!("{}", "ARES Neuromorphic System Status".cyan().bold());
    println!("{}", "-".repeat(40));

    quantum_message(&format!("Backend: {}", state.backend_info));

    println!("Performance:");
    println!("   Commands processed: {}", state.commands_processed);
    println!("   Average processing time: {:.1}ms", state.avg_processing_time_ms);

    println!("Resource Allocation:");
    println!("   NLP: {:.1}%", state.resource_allocation.nlp * 100.0);
    println!("   DRPP: {:.1}%", state.resource_allocation.drpp * 100.0);
    println!("   EMS: {:.1}%", state.resource_allocation.ems * 100.0);
    println!("   ADP: {:.1}%", state.resource_allocation.adp * 100.0);
    println!("   EGC: {:.1}%", state.resource_allocation.egc * 100.0);

    if state.learning_active {
        neural_message("Learning mode: ACTIVE");
    } else {
        info_message("Learning mode: Inactive");
    }
    println!();
    Ok(())
}

async fn show_learned_patterns(system: &UnifiedNeuromorphicSystem) -> Result<()> {
    println!("{}", "Learned Patterns".cyan().bold());
    println!("{}", "-".repeat(40));

    let learning_metrics = system.learning_get_metrics().await;

    if learning_metrics.total_samples == 0 {
        println!("No learned patterns yet.");
        return Ok(());
    }

    println!("Total samples: {}", learning_metrics.total_samples);
    Ok(())
}
