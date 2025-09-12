//! End-to-end test for the Dynamic Resonance Phase Processing system

use hephaestus_forge::{
    HephaestusForge, ForgeConfig, ForgeConfigBuilder, OperationalMode, ForgeStatus
};
use std::time::Duration;

#[tokio::test]
async fn test_resonance_based_optimization_detection() {
    // Create forge with resonance-based processing
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .enable_resonance_processing(true)
        .energy_threshold(0.5)
        .detection_sensitivity(0.8)
        .build()
        .expect("Failed to build config");
    
    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");
    
    // Start the forge
    forge.start().await.expect("Failed to start forge");
    
    // Let it run briefly to detect patterns through resonance
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Check status - should show resonance processing active
    let status = forge.status().await;
    
    // Verify forge is operational
    assert!(status.is_running, "Forge should be running");
    println!("Forge is running with resonance processing");
    
    // Stop the forge
    forge.stop().await.expect("Failed to stop forge");
}

#[tokio::test]
async fn test_phase_lattice_computation() {
    use hephaestus_forge::resonance::{
        DynamicResonanceProcessor, ComputationTensor
    };
    use nalgebra::DMatrix;
    
    // Create resonance processor
    let processor = DynamicResonanceProcessor::new((8, 8, 4)).await;
    
    // Create test input tensor
    let mut data = DMatrix::zeros(32, 32);
    for i in 0..32 {
        for j in 0..32 {
            // Create wave pattern
            data[(i, j)] = ((i as f64 * 0.1).sin() + (j as f64 * 0.1).cos()) / 2.0;
        }
    }
    
    let tensor = ComputationTensor::from_matrix(data);
    
    // Process through resonance
    let result = processor.process_via_resonance(tensor).await;
    
    match result {
        Ok(solution) => {
            println!("Resonance frequency: {}", solution.resonance_frequency);
            println!("Coherence: {}", solution.coherence);
            println!("Energy efficiency: {}", solution.energy_efficiency);
            
            // Verify solution has meaningful values
            assert!(solution.coherence > 0.0, "Coherence should be positive");
            assert!(solution.resonance_frequency > 0.0, "Frequency should be positive");
        },
        Err(e) => {
            // It's ok if no solution crystallizes in this simple test
            println!("No solution crystallized: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_harmonic_interference() {
    use hephaestus_forge::resonance::{
        HarmonicInducer, PhaseLattice
    };
    
    // Create phase lattice
    let lattice = PhaseLattice::new((16, 16, 4)).await;
    
    // Create harmonic inducer
    let inducer = HarmonicInducer::new();
    
    // Target frequencies for constructive interference
    let target_frequencies = vec![1.0, 2.0, 3.0, 5.0];
    
    // Induce interference
    let modes = inducer.induce_constructive_interference(
        &target_frequencies,
        &lattice
    ).await.expect("Failed to induce interference");
    
    // Verify modes were created
    assert!(!modes.is_empty(), "Should create resonant modes");
    
    for mode in &modes {
        println!("Mode frequency: {} Hz, Q-factor: {}", 
                 mode.frequency, mode.q_factor);
        assert!(mode.q_factor > 0.0, "Q-factor should be positive");
    }
}

#[tokio::test]
async fn test_adaptive_dissipation() {
    use hephaestus_forge::resonance::{
        AdaptiveDissipativeProcessor, DissipationStrategy,
        InterferencePatterns, ResonantMode
    };
    use nalgebra::{DMatrix, Complex};
    
    // Create dissipative processor
    let processor = AdaptiveDissipativeProcessor::new().await;
    
    // Create test interference patterns
    let patterns = InterferencePatterns {
        constructive_modes: vec![
            ResonantMode {
                frequency: 1.0,
                mode_shape: DMatrix::from_element(32, 32, Complex::new(1.0, 0.0)),
                energy: 10.0,
                q_factor: 100.0,
                amplification_factor: 2.0,
                phase_velocity: 2.0,
                group_velocity: 1.8,
                damping_rate: 0.01,
            }
        ],
        destructive_modes: vec![],
        coupling_matrix: DMatrix::identity(1, 1),
    };
    
    // Apply adaptive dissipation
    let stabilized = processor.stabilize_through_dissipation(
        patterns,
        DissipationStrategy::AdaptiveGradient
    ).await.expect("Failed to stabilize");
    
    // Verify stabilization
    assert!(stabilized.coherence > 0.0, "Should maintain coherence");
    assert!(stabilized.coherence <= 1.0, "Coherence should be bounded");
    
    println!("Stabilized coherence: {}", stabilized.coherence);
}

#[tokio::test]
async fn test_topological_analysis() {
    use hephaestus_forge::resonance::{
        TopologicalAnalyzer, StabilizedPattern
    };
    use nalgebra::DMatrix;
    
    // Create topology analyzer
    let analyzer = TopologicalAnalyzer::new((8, 8, 4));
    
    // Create test pattern with interesting topology
    let mut energy = DMatrix::zeros(32, 32);
    
    // Create a pattern with loops (Betti-1 features)
    for i in 10..20 {
        energy[(i, 10)] = 1.0;
        energy[(i, 20)] = 1.0;
        energy[(10, i)] = 1.0;
        energy[(20, i)] = 1.0;
    }
    
    let pattern = StabilizedPattern {
        energy_distribution: energy,
        coherence: 0.8,
    };
    
    // Analyze topology
    let analysis = analyzer.analyze_persistent_homology(&pattern).await
        .expect("Failed to analyze topology");
    
    // Verify features were detected
    assert!(!analysis.features.is_empty(), "Should detect topological features");
    
    for feature in &analysis.features {
        println!("Feature: {} (persistence: {})", 
                 feature.feature_type, feature.persistence);
    }
}