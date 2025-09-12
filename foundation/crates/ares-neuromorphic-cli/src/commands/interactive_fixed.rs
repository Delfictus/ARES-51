//! Fixed version of the show_learned_patterns function

use anyhow::Result;
use colored::*;

async fn show_learned_patterns(system: &UnifiedNeuromorphicSystem) -> Result<()> {
    let title = "Learned Patterns";
    println!("{}", title.cyan().bold());
    println!("=========================================");
    
    // Show actual learned patterns from the neuromorphic system
    let learning_metrics = system.learning_system.get_metrics().await;
    let _nlp_metrics = system.nlp_processor.get_metrics().await;
    
    if learning_metrics.total_samples == 0 {
        println!("No patterns learned yet");
        return Ok(());
    }
    
    let sample_msg = format!("Total samples: {}", learning_metrics.total_samples);
    println!("{}", sample_msg);
    
    Ok(())
}