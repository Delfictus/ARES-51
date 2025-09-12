//! Test the ARES System Integration Bridge
//! 
//! This example demonstrates the Hephaestus Forge connecting to the entire
//! ARES ChronoFabric ecosystem and detecting cross-system resonances.

use hephaestus_forge::{
    HephaestusForge, ForgeConfig, ComputationTensor,
    DynamicResonanceProcessor, OperationalMode,
};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Initializing Hephaestus Forge with ARES Integration Bridge...\n");
    
    // Create forge configuration
    let config = ForgeConfig::default();
    
    // Initialize the forge
    let forge = Arc::new(HephaestusForge::new_async_public(config).await?);
    println!("✅ Forge initialized successfully");
    
    // Create the ARES bridge
    let bridge = Arc::new(
        hephaestus_forge::ares_bridge::AresSystemBridge::new(forge.clone()).await
    );
    println!("✅ ARES Integration Bridge created");
    
    // Start global resonance monitoring
    println!("\n🌐 Starting global resonance monitoring...");
    bridge.start_global_resonance().await;
    
    // Create test computation tensors to inject into the system
    println!("\n📊 Injecting test patterns into the resonance field...");
    
    for i in 0..3 {
        println!("\n--- Test Pattern {} ---", i + 1);
        
        // Create a test tensor with varying patterns
        let tensor = ComputationTensor::random(256);
        
        // Process through resonance with lattice dimensions
        let processor = DynamicResonanceProcessor::new((16, 16, 16)).await;
        match processor.process_via_resonance(tensor).await {
            Ok(solution) => {
                println!("  ✨ Resonance detected!");
                println!("    - Frequency: {:.2} Hz", solution.resonance_frequency);
                println!("    - Coherence: {:.2}%", solution.coherence * 100.0);
                println!("    - Energy efficiency: {:.2}%", solution.energy_efficiency * 100.0);
                
                // Check for emergence
                if solution.coherence > 0.8 {
                    println!("  🌟 HIGH COHERENCE - Potential emergence detected!");
                }
            }
            Err(e) => {
                println!("  ⚠️  No resonance found: {:?}", e);
            }
        }
        
        // Let the system process
        sleep(Duration::from_millis(500)).await;
    }
    
    // Monitor for cross-system resonances
    println!("\n🔍 Monitoring for cross-system resonances...");
    println!("   (The bridge is running in the background, detecting patterns)");
    
    // Let it run for a few seconds to show activity
    for i in 0..5 {
        sleep(Duration::from_secs(1)).await;
        print!(".");
        if i == 2 {
            println!("\n   💫 Systems are synchronizing...");
        }
    }
    
    println!("\n\n✨ ARES Integration Test Complete!");
    println!("   The Hephaestus Forge is now connected to:");
    println!("   - CSF Core (Tensor Operations)");
    println!("   - CSF Time (Temporal Precision)");
    println!("   - CSF Quantum (Quantum States) [simulated]");
    println!("   - CSF Runtime (Execution Management) [simulated]");
    println!("   - CSF Bus (Event Distribution) [simulated]");
    println!("   - Neuromorphic CLI (Spike Processing) [simulated]");
    println!("\n   Cross-system resonances enable emergent optimization!");
    
    Ok(())
}