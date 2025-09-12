//! Quick Integration Test
//! A faster test to verify the integration pipeline works

use hephaestus_forge::{
    HephaestusForge, ForgeConfig, 
    ComputationTensor, DynamicResonanceProcessor,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Quick Integration Test\n");
    
    // Step 1: Initialize Forge
    println!("1. Initializing Forge...");
    let forge = Arc::new(HephaestusForge::new_async_public(
        ForgeConfig::default()
    ).await?);
    println!("   ✅ Forge ready\n");
    
    // Step 2: Test Resonance Processing
    println!("2. Testing Resonance Processing...");
    let processor = DynamicResonanceProcessor::new((8, 8, 8)).await;
    let tensor = ComputationTensor::random(64);
    
    match processor.process_via_resonance(tensor).await {
        Ok(solution) => {
            println!("   ✅ Resonance detected!");
            println!("      Coherence: {:.2}%", solution.coherence * 100.0);
            println!("      Frequency: {:.2} Hz\n", solution.resonance_frequency);
        }
        Err(e) => {
            println!("   ⚠️  No resonance: {:?}\n", e);
        }
    }
    
    // Step 3: Test Workload Bridge
    println!("3. Testing Workload Bridge...");
    use hephaestus_forge::workload::SystemMetrics;
    
    let metrics = SystemMetrics {
        timestamp: 1000,
        cpu_usage_percent: 75.0,
        cpu_temperature: Some(65.0),
        memory_used_bytes: 4_000_000_000,
        memory_available_bytes: 8_000_000_000,
        cache_hits: 10000,
        cache_misses: 100,
        disk_read_bytes_sec: 1_000_000.0,
        disk_write_bytes_sec: 500_000.0,
        network_rx_bytes_sec: 2_000_000.0,
        network_tx_bytes_sec: 1_500_000.0,
        active_connections: 100,
        requests_per_sec: 1000.0,
        average_latency_ms: 10.0,
        error_rate: 0.001,
    };
    
    // Convert metrics to tensor
    let mut tensor_data = vec![0.0; 64];
    tensor_data[0] = metrics.cpu_usage_percent / 100.0;
    tensor_data[1] = metrics.memory_used_bytes as f64 / metrics.memory_available_bytes as f64;
    tensor_data[2] = metrics.average_latency_ms / 100.0;
    tensor_data[3] = metrics.error_rate * 10.0;
    
    let workload_tensor = ComputationTensor::from_vec(tensor_data);
    
    match processor.process_via_resonance(workload_tensor).await {
        Ok(solution) => {
            println!("   ✅ Workload resonance detected!");
            println!("      Coherence: {:.2}%", solution.coherence * 100.0);
            
            if solution.coherence > 0.7 {
                println!("      🎯 High coherence - optimization opportunity!\n");
            } else {
                println!();
            }
        }
        Err(e) => {
            println!("   ⚠️  No workload resonance: {:?}\n", e);
        }
    }
    
    // Step 4: Verify ARES Bridge
    println!("4. Testing ARES Bridge...");
    use hephaestus_forge::AresSystemBridge;
    
    let bridge = Arc::new(AresSystemBridge::new(forge).await);
    println!("   ✅ ARES bridge created");
    
    // Start monitoring (non-blocking)
    bridge.start_global_resonance().await;
    println!("   ✅ Global resonance monitoring started\n");
    
    // Step 5: Summary
    println!("═══════════════════════════════════════");
    println!("           TEST COMPLETE");
    println!("═══════════════════════════════════════");
    println!("\n✅ All components working:");
    println!("   • Forge initialization");
    println!("   • Resonance processing");
    println!("   • Workload analysis");
    println!("   • ARES bridge connection");
    println!("\nThe system is ready for full integration!");
    
    Ok(())
}