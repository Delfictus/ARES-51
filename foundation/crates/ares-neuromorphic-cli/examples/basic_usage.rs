//! Basic usage example of the ARES Neuromorphic CLI

use anyhow::Result;
use ares_neuromorphic_cli::{UnifiedNeuromorphicSystem, CliConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the library
    ares_neuromorphic_cli::init().await?;
    
    // Load configuration
    let config = CliConfig::load_or_default(None).await?;
    
    // Initialize neuromorphic system
    let system = UnifiedNeuromorphicSystem::initialize(None).await?;
    
    println!("🧠 ARES Neuromorphic System Initialized");
    println!("Backend: {}", system.backend_info());
    
    // Example natural language queries
    let queries = vec![
        "show me system status",
        "what's the quantum coherence level?",
        "optimize performance",
        "scan for anomalies",
        "enable learning mode",
    ];
    
    for query in queries {
        println!("\n📝 Processing: \"{}\"", query);
        
        let intent = system.process_natural_language(query).await?;
        
        println!("   → Command: {}", intent.command);
        println!("   → Confidence: {:.1}%", intent.confidence * 100.0);
        println!("   → Domain: {:?}", intent.context.domain);
        
        if intent.confidence < 0.7 {
            println!("   ⚠️  Low confidence - would request clarification");
        }
    }
    
    // Toggle learning mode
    println!("\n🧠 Activating learning mode...");
    let learning_active = system.toggle_learning().await?;
    println!("Learning mode: {}", if learning_active { "ACTIVE" } else { "INACTIVE" });
    
    // Example learning interaction
    if learning_active {
        println!("\n📚 Example learning interaction:");
        println!("User: 'show quantum stuff'");
        println!("System: Interpreted as 'csf status'");
        println!("User: 'actually: csf quantum metrics --detailed'");
        println!("System: 🧠 Pattern learned!");
        
        // This would actually train the system
        // system.learn_from_correction("show quantum stuff", "csf quantum metrics --detailed").await?;
    }
    
    Ok(())
}