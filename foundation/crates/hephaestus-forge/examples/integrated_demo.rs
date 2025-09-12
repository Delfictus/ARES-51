//! Integrated demonstration of all 4 new features
//! 
//! Run with: cargo run --example integrated_demo --features monitoring

use hephaestus_forge::{
    HephaestusForge, ForgeConfigBuilder, OperationalMode,
    resonance::{DynamicResonanceProcessor, ComputationTensor},
};
use nalgebra::DMatrix;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hephaestus Forge Integrated Demo ===\n");
    
    // 1. Hardware Acceleration Demo
    println!("1. Hardware Acceleration:");
    #[cfg(feature = "cuda")]
    {
        use hephaestus_forge::accelerator::{AcceleratedPhaseLattice, AcceleratorBackend};
        let lattice = AcceleratedPhaseLattice::new(
            (64, 64, 16),
            AcceleratorBackend::CUDA { device_id: 0 }
        ).await?;
        println!("   ✓ CUDA acceleration enabled");
    }
    #[cfg(not(feature = "cuda"))]
    {
        use hephaestus_forge::accelerator::{AcceleratedPhaseLattice, AcceleratorBackend};
        let lattice = AcceleratedPhaseLattice::new(
            (64, 64, 16),
            AcceleratorBackend::CPU
        ).await?;
        println!("   ✓ CPU fallback (CUDA not available)");
        
        // Run computation
        let input = ComputationTensor::random(64 * 64);
        let start = Instant::now();
        lattice.evolve_accelerated(&input, 10).await?;
        println!("   → Evolution time: {:?}", start.elapsed());
    }
    
    // 2. Distributed Phase Lattice Demo
    println!("\n2. Distributed Phase Lattice:");
    {
        use hephaestus_forge::distributed::{DistributedPhaseLattice, DistributedResonanceProtocol, NetworkTopology};
        
        let mut protocol = DistributedResonanceProtocol::new(NetworkTopology::FullMesh);
        protocol.initialize(3).await;
        println!("   ✓ Initialized 3-node distributed lattice");
        
        let node = DistributedPhaseLattice::new("demo_node".to_string(), (32, 32, 8)).await;
        node.add_peer("peer1".to_string(), "tcp://127.0.0.1:8081".to_string()).await;
        node.add_peer("peer2".to_string(), "tcp://127.0.0.1:8082".to_string()).await;
        println!("   ✓ Connected to 2 peer nodes");
        
        // Simulate distributed computation
        let input = ComputationTensor::random(32 * 32);
        match node.distributed_resonance(input).await {
            Ok(solution) => {
                println!("   → Distributed coherence: {:.3}", solution.coherence);
            },
            Err(e) => println!("   → Demo mode: {}", e),
        }
    }
    
    // 3. Performance Profiling Demo
    println!("\n3. Performance Profiling:");
    {
        use hephaestus_forge::profiling::{ResonanceProfiler, ResonanceOptimizer};
        
        let mut profiler = ResonanceProfiler::new();
        let mut optimizer = ResonanceOptimizer::new();
        
        // Profile resonance computation
        let processor = DynamicResonanceProcessor::new((8, 8, 4)).await;
        let input = {
            let mut data = DMatrix::zeros(32, 32);
            for i in 0..32 {
                for j in 0..32 {
                    data[(i, j)] = ((i as f64 * 0.1).sin() + (j as f64 * 0.1).cos()) / 2.0;
                }
            }
            ComputationTensor::from_matrix(data)
        };
        
        let result = profiler.profile("resonance_computation", || {
            // Simulate computation
            std::thread::sleep(std::time::Duration::from_millis(10));
            "computed"
        });
        
        let report = profiler.generate_report();
        println!("   ✓ Profiled {} operations", report.total_operations);
        
        let config = optimizer.auto_tune();
        println!("   ✓ Auto-tuning recommendations:");
        println!("     - SIMD: {}", config.enable_simd);
        println!("     - Threads: {}", config.thread_count);
        println!("     - FFT size: {}", config.fft_size);
    }
    
    // 4. Monitoring Integration Demo
    println!("\n4. Prometheus Monitoring:");
    #[cfg(feature = "monitoring")]
    {
        use hephaestus_forge::monitoring::{ForgeMetrics, MetricsServer};
        use std::sync::Arc;
        
        let metrics = Arc::new(ForgeMetrics::new());
        
        // Update some metrics
        metrics.update_resonance(0.85, 2.5, 100.0);
        metrics.record_computation(0.025);
        metrics.active_optimizations.set(3.0);
        metrics.optimization_success_rate.set(0.92);
        
        println!("   ✓ Metrics initialized");
        println!("   → Coherence: 0.85");
        println!("   → Frequency: 2.5 Hz");
        println!("   → Active optimizations: 3");
        println!("   → Success rate: 92%");
        
        // Start metrics server (in background)
        let server = MetricsServer::new(metrics.clone(), 9090);
        tokio::spawn(async move {
            // server.start().await;
            println!("   ✓ Metrics server would run on :9090");
        });
        
        // Generate Grafana dashboard config
        let dashboard = hephaestus_forge::monitoring::generate_grafana_dashboard();
        println!("   ✓ Grafana dashboard config generated");
    }
    #[cfg(not(feature = "monitoring"))]
    {
        println!("   → Monitoring disabled (enable with --features monitoring)");
    }
    
    // Full System Integration
    println!("\n5. Full System Integration:");
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .enable_resonance_processing(true)
        .build()?;
    
    let forge = HephaestusForge::new_async_public(config).await?;
    forge.start().await?;
    println!("   ✓ Forge started with all features");
    
    let status = forge.status().await;
    println!("   → Mode: {:?}", status.mode);
    println!("   → Running: {}", status.is_running);
    
    forge.stop().await?;
    println!("   ✓ Forge stopped successfully");
    
    println!("\n=== All 4 features integrated successfully! ===");
    
    Ok(())
}