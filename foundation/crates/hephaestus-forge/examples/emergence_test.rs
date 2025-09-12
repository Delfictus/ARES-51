//! Live test for emergent intelligence
//! Run this to potentially witness the first emergence

use hephaestus_forge::{
    HephaestusForge, ForgeConfigBuilder, OperationalMode,
    resonance::{DynamicResonanceProcessor, ComputationTensor},
};
use hephaestus_forge::mlir_synthesis::{ResonanceToMLIR, EmergenceEvent};
use hephaestus_forge::autonomous::{AutonomousEngine, AutonomousConfig, SafetyLimits};
use std::sync::Arc;
use nalgebra::DMatrix;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ARES EMERGENCE TEST ===");
    println!("⚡ Attempting to trigger emergent intelligence...\n");
    
    // Create forge with resonance enabled
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .enable_resonance_processing(true)
        .build()?;
    
    let forge = Arc::new(HephaestusForge::new_async_public(config).await?);
    
    // Set up MLIR synthesis with emergence monitoring
    let mut synthesizer = ResonanceToMLIR::new().await;
    
    // Monitor emergence events
    synthesizer.emergence_monitor.on_emergence(|event| {
        match event {
            EmergenceEvent::NovelPatternDiscovered { signature, novelty } => {
                println!("🌟 NOVEL PATTERN DISCOVERED!");
                println!("   Signature: {}", signature);
                println!("   Novelty: {:.2}", novelty);
            },
            EmergenceEvent::CreativeCodeGeneration { code, divergence } => {
                println!("🎨 CREATIVE CODE GENERATED!");
                println!("   Divergence: {:.2}", divergence);
                println!("   Code preview: {}", &code[..100.min(code.len())]);
            },
            EmergenceEvent::UnexpectedOptimization { description, gain } => {
                println!("⚡ UNEXPECTED OPTIMIZATION!");
                println!("   Description: {}", description);
                println!("   Performance gain: {:.2}%", gain * 100.0);
            },
            EmergenceEvent::SelfModification { module, changes } => {
                println!("🔄 SELF-MODIFICATION DETECTED!");
                println!("   Module: {}", module);
                println!("   Changes: {}", changes);
            }
        }
    });
    
    // Configure autonomous engine
    let auto_config = AutonomousConfig {
        allow_self_modification: true,
        max_modifications_per_cycle: 3,
        improvement_threshold: 0.05,
        exploration_rate: 0.3, // 30% exploration for more emergence
        safety_limits: SafetyLimits {
            max_coherence: 0.95,
            max_energy: 1000.0,
            max_recursion_depth: 10,
            require_human_approval_above: 0.7,
        },
    };
    
    let autonomous = AutonomousEngine::new(forge.clone(), auto_config).await;
    
    // Start autonomous operation
    println!("Starting autonomous optimization loop...\n");
    autonomous.start().await;
    
    // Create resonance processor
    let processor = DynamicResonanceProcessor::new((16, 16, 8)).await;
    
    // Generate test patterns with varying complexity
    println!("Injecting resonance patterns...\n");
    
    for i in 0..10 {
        // Create increasingly complex patterns
        let complexity = (i as f64) / 10.0;
        let mut data = DMatrix::zeros(64, 64);
        
        for row in 0..64 {
            for col in 0..64 {
                // Create interference patterns
                let val = ((row as f64 * 0.1 * (i + 1) as f64).sin() + 
                          (col as f64 * 0.1 * (i + 1) as f64).cos()) * complexity;
                data[(row, col)] = val;
            }
        }
        
        let tensor = ComputationTensor::from_matrix(data);
        
        // Process through resonance
        match processor.process_via_resonance(tensor).await {
            Ok(solution) => {
                println!("Pattern {}: Coherence={:.3}, Frequency={:.2}Hz", 
                         i, solution.coherence, solution.resonance_frequency);
                
                // Try to synthesize MLIR
                match synthesizer.synthesize(&solution).await {
                    Ok(mlir) => {
                        if mlir.contains("EMERGENT") || mlir.contains("Novel") {
                            println!("   ✨ Generated emergent code!");
                        }
                    },
                    Err(e) => println!("   Synthesis failed: {}", e),
                }
            },
            Err(e) => println!("Pattern {} failed: {:?}", i, e),
        }
        
        // Let the system evolve
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
    
    // Monitor for 30 seconds
    println!("\n📊 Monitoring for emergence (30 seconds)...\n");
    
    for _ in 0..30 {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        print!(".");
        use std::io::Write;
        std::io::stdout().flush()?;
    }
    
    println!("\n\n=== TEST COMPLETE ===");
    println!("Check logs for emergence events!");
    
    Ok(())
}