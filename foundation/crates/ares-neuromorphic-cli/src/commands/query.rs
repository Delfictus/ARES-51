//! Natural language query processing

use anyhow::Result;
use colored::*;

use crate::neuromorphic::UnifiedNeuromorphicSystem;
use crate::commands::{CommandResult, utils::*};

/// Execute a natural language query
pub async fn execute_natural_language_query(
    input: String,
    system: UnifiedNeuromorphicSystem,
) -> CommandResult {
    println!("{} {}", "🧠".cyan(), format!("Processing: \"{}\"", input).bold());
    
    let spinner = create_spinner("Analyzing through neuromorphic network...");
    
    // Process through neuromorphic system
    let intent = system.process_natural_language(&input).await?;
    
    spinner.finish_and_clear();
    
    // Display results based on confidence
    if intent.confidence > 0.8 {
        success_message(&format!("High confidence interpretation"));
        println!("   Command: {}", intent.command.green().bold());
        println!("   Confidence: {:.1}%", intent.confidence * 100.0);
        
        // Show context if interesting
        if intent.context.urgency > 0.5 {
            println!("   Urgency: {:.1}% {}", 
                intent.context.urgency * 100.0, 
                "⚠️".yellow());
        }
    } else if intent.confidence > 0.4 {
        println!("{} {}", "?".yellow().bold(), "Medium confidence interpretation".yellow());
        println!("   Command: {}", intent.command.yellow());
        println!("   Confidence: {:.1}%", intent.confidence * 100.0);
        
        if !intent.alternatives.is_empty() {
            println!("   Alternatives:");
            for alt in &intent.alternatives {
                println!("     • {} ({:.1}%)", alt.command, alt.confidence * 100.0);
            }
        }
    } else {
        error_message("Low confidence - interpretation uncertain");
        println!("   Best guess: {}", intent.command.red());
        println!("   Confidence: {:.1}%", intent.confidence * 100.0);
        
        if system.get_state().await.learning_active {
            info_message("Learning mode active - provide feedback to improve accuracy");
        } else {
            info_message("Enable learning mode to improve interpretation accuracy");
        }
    }
    
    // Show command domain and sentiment analysis
    println!("\n{}", "📊 Analysis Details:".blue().bold());
    println!("   Domain: {:?}", intent.context.domain);
    println!("   Operator state: {:?}", intent.context.sentiment.operator_state);
    
    if intent.context.sentiment.valence != 0.0 || intent.context.sentiment.arousal != 0.0 {
        println!("   Emotional tone: valence={:.2}, arousal={:.2}", 
                intent.context.sentiment.valence,
                intent.context.sentiment.arousal);
    }
    
    // Show any detected parameters
    if !intent.context.parameters.is_empty() {
        println!("   Parameters:");
        for (key, value) in &intent.context.parameters {
            println!("     {}: {}", key, value);
        }
    }
    
    Ok(())
}