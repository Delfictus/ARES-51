//! PHASE 2C INTEGRATION TEST
//! Complete validation of Phase 2C Network Processing Layer
//! Tests all advanced network processing components working together

use crate::coupling_adaptation::{CouplingAdaptationEngine, AdaptationStrategy, CouplingPerformance};
use crate::pattern_routing::{PatternRoutingEngine, RoutingStrategy, PatternData, RouteDestination};
use crate::cross_module_optimizer::{CrossModuleOptimizer, CommunicationStats};
use crate::realtime_adaptation::{RealTimeAdaptationEngine, AdaptationResult, ParameterType};
use crate::drpp::{DynamicResonancePatternProcessor, DrppConfig, DrppState, Pattern, PatternType};
use crate::adp::{AdaptiveDecisionProcessor, AdpConfig};
use crate::drpp_adp_bridge::{DrppAdpChannel, DrppPatternMessage};
use crate::multi_timeframe::{MultiTimeframeNetwork, TimeHorizon};
use crate::phase_coherence::{PhaseCoherenceAnalyzer, FrequencyBand, MarketRegime};
use crate::transfer_entropy::TransferEntropyEngine;
use crate::spike_encoding::{Spike, NeuronType};
use crate::test_pattern_performance::{MarketPatternGenerator, MarketPatternType};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::time::{Duration, Instant};
use anyhow::Result;
use std::collections::HashMap;

/// Complete Phase 2C integration test result
#[derive(Debug)]
pub struct Phase2CResult {
    /// Coupling adaptation operational
    pub coupling_adaptation_operational: bool,
    /// Pattern routing system working
    pub pattern_routing_operational: bool,
    /// Cross-module optimization active
    pub cross_module_optimization_operational: bool,
    /// Real-time parameter adaptation working
    pub realtime_adaptation_operational: bool,
    /// End-to-end network processing latency
    pub end_to_end_latency_us: f64,
    /// Overall network processing efficiency
    pub network_efficiency: f64,
    /// System adaptation effectiveness
    pub adaptation_effectiveness: f64,
    /// Communication optimization ratio
    pub communication_optimization: f64,
    /// Total patterns processed
    pub total_patterns_processed: usize,
    /// Network stability score
    pub network_stability: f64,
}

/// Complete Phase 2C integration test
pub async fn test_phase_2c_full_integration() -> Result<Phase2CResult> {
    tracing::info!("🚀 STARTING PHASE 2C NETWORK PROCESSING LAYER INTEGRATION TEST");
    tracing::info!("================================================================");
    
    let start_time = Instant::now();
    
    // Initialize all Phase 2C components
    tracing::info!("🔧 Initializing Phase 2C components...");
    
    // 1. Initialize DRPP-ADP bridge for communication
    let bridge = Arc::new(DrppAdpChannel::new().await?);
    
    // 2. Coupling Adaptation Engine
    let mut coupling_engine = CouplingAdaptationEngine::new(
        128, // 128 oscillators
        0.01, // Learning rate
        AdaptationStrategy::Hybrid(vec![
            AdaptationStrategy::Hebbian,
            AdaptationStrategy::MarketAware,
            AdaptationStrategy::SpikeTiming,
        ])
    );
    
    // 3. Pattern Routing Engine
    let mut pattern_router = PatternRoutingEngine::new(RoutingStrategy::Hybrid(vec![
        RoutingStrategy::LatencyOptimized,
        RoutingStrategy::ContentAware,
        RoutingStrategy::LoadAware,
    ]));
    
    // 4. Cross-Module Optimizer
    let cross_optimizer = CrossModuleOptimizer::new(Arc::clone(&bridge))?;
    
    // 5. Real-Time Adaptation Engine
    let mut adaptation_engine = RealTimeAdaptationEngine::new()?;
    
    // 6. Supporting neuromorphic components
    let mut multi_timeframe = MultiTimeframeNetwork::new(5000).await?;
    let frequency_bands = vec![
        FrequencyBand::new("Alpha", 8.0, 13.0),
        FrequencyBand::new("Beta", 13.0, 30.0),
        FrequencyBand::new("Gamma", 30.0, 100.0),
    ];
    let mut coherence_analyzer = PhaseCoherenceAnalyzer::new(128, frequency_bands)?;
    
    tracing::info!("✅ All Phase 2C components initialized");
    
    // Generate comprehensive test dataset
    let mut generator = MarketPatternGenerator::new(2024, 20.0, 0.02, 0.9);
    
    // Create complex multi-scenario test
    let test_scenarios = vec![
        (MarketPatternType::TrendingBullish, 2000, 30000),
        (MarketPatternType::RegimeShift, 1500, 25000),
        (MarketPatternType::HighVolatility, 2500, 35000),
        (MarketPatternType::FlashCrash, 1000, 10000),
        (MarketPatternType::Ranging, 1800, 28000),
    ];
    
    let mut all_test_spikes = Vec::new();
    for (scenario, count, duration_ms) in test_scenarios {
        let scenario_spikes = generator.generate_market_spikes(scenario, count, duration_ms);
        all_test_spikes.extend(scenario_spikes);
    }
    
    all_test_spikes.sort_by_key(|s| s.timestamp_ns);
    tracing::info!("📊 Generated {} test spikes across 5 market scenarios", all_test_spikes.len());
    
    let integration_start = Instant::now();
    
    // TEST 1: Coupling Adaptation Engine
    tracing::info!("🧪 Testing coupling adaptation engine...");
    let coupling_test_start = Instant::now();
    
    // Create mock DRPP state for coupling adaptation
    let mock_drpp_state = DrppState {
        patterns: generate_test_patterns(&all_test_spikes[0..500]),
        oscillator_phases: generate_test_phases(128),
        coherence: 0.7,
        novelty: 0.6,
    };
    
    // Create mock multi-timeframe result
    let mock_multi_result = multi_timeframe.process_multi_timeframe(&all_test_spikes[0..1000]).await?;
    
    // Analyze phase coherence
    let phases: Vec<f64> = all_test_spikes[0..1000].iter()
        .map(|s| (s.timestamp_ns as f64 / 1_000_000.0) % (2.0 * std::f64::consts::PI))
        .collect();
    let coherence_patterns = coherence_analyzer.analyze_coherence(&phases)?;
    
    // Test coupling adaptation
    coupling_engine.adapt_coupling(
        &mock_drpp_state,
        &coherence_patterns,
        &mock_multi_result,
        &all_test_spikes[0..1000],
    ).await?;
    
    let coupling_duration = coupling_test_start.elapsed();
    let coupling_metrics = coupling_engine.calculate_network_metrics();
    
    let coupling_operational = coupling_duration < Duration::from_millis(500) &&
                              coupling_metrics.clustering_coefficient > 0.1 &&
                              coupling_metrics.connection_density > 0.05;
    
    if coupling_operational {
        tracing::info!("   ✅ Coupling adaptation: {:.2}ms, clustering={:.3}, density={:.3}",
                      coupling_duration.as_millis(), 
                      coupling_metrics.clustering_coefficient,
                      coupling_metrics.connection_density);
    } else {
        tracing::error!("   ❌ Coupling adaptation failed: {:.2}ms, clustering={:.3}",
                       coupling_duration.as_millis(), coupling_metrics.clustering_coefficient);
    }
    
    // TEST 2: Pattern Routing Engine
    tracing::info!("🧪 Testing pattern routing engine...");
    let routing_test_start = Instant::now();
    
    // Add routes for different pattern types
    pattern_router.add_route(PatternType::Emergent, RouteDestination::AdpProcessor(0)).await?;
    pattern_router.add_route(PatternType::Synchronous, RouteDestination::DrppProcessor(0)).await?;
    pattern_router.add_route(PatternType::Chaotic, RouteDestination::MultiTimeframe(TimeHorizon::High)).await?;
    
    let mut successful_routes = 0;
    let mut total_routing_time = Duration::ZERO;
    
    // Test routing for different pattern types
    for pattern in &mock_drpp_state.patterns[0..10] {
        let pattern_data = PatternData::Drpp(pattern.clone());
        let route_start = Instant::now();
        
        if let Ok(_destination) = pattern_router.route_pattern(pattern_data, 150).await {
            successful_routes += 1;
            total_routing_time += route_start.elapsed();
        }
    }
    
    let routing_duration = routing_test_start.elapsed();
    let avg_routing_latency = if successful_routes > 0 {
        total_routing_time.as_micros() as f64 / successful_routes as f64
    } else {
        0.0
    };
    
    let routing_operational = successful_routes >= 8 && // At least 80% success
                             avg_routing_latency < 100.0 && // <100μs average
                             routing_duration < Duration::from_millis(50);
    
    if routing_operational {
        tracing::info!("   ✅ Pattern routing: {:.2}ms, success={}/{}, avg_latency={:.1}μs",
                      routing_duration.as_millis(), successful_routes, 10, avg_routing_latency);
    } else {
        tracing::error!("   ❌ Pattern routing failed: success={}/{}, latency={:.1}μs",
                       successful_routes, 10, avg_routing_latency);
    }
    
    // TEST 3: Cross-Module Optimizer
    tracing::info!("🧪 Testing cross-module optimizer...");
    let optimizer_test_start = Instant::now();
    
    let mut optimizer_success_count = 0;
    let mut total_optimizer_latency = Duration::ZERO;
    
    // Test optimized pattern sending
    for pattern in &mock_drpp_state.patterns[0..5] {
        let pattern_msg = DrppPatternMessage {
            pattern: pattern.clone(),
            confidence: pattern.strength,
            priority: 180,
            timestamp: Instant::now(),
            feature_vector: vec![pattern.strength, pattern.phase_coherence],
        };
        
        let send_start = Instant::now();
        if let Ok(_) = cross_optimizer.send_pattern_optimized(pattern_msg, Some(1000)).await {
            optimizer_success_count += 1;
            total_optimizer_latency += send_start.elapsed();
        }
    }
    
    let optimizer_duration = optimizer_test_start.elapsed();
    let avg_optimizer_latency = if optimizer_success_count > 0 {
        total_optimizer_latency.as_micros() as f64 / optimizer_success_count as f64
    } else {
        0.0
    };
    
    // Get communication stats
    let comm_stats = cross_optimizer.get_stats();
    
    let optimizer_operational = optimizer_success_count >= 4 && // At least 80% success
                               avg_optimizer_latency < 500.0 && // <500μs average
                               optimizer_duration < Duration::from_millis(100);
    
    if optimizer_operational {
        tracing::info!("   ✅ Cross-module optimizer: {:.2}ms, success={}/{}, avg_latency={:.1}μs",
                      optimizer_duration.as_millis(), optimizer_success_count, 5, avg_optimizer_latency);
        tracing::info!("     Messages sent: {}, avg latency: {:.1}μs", 
                      comm_stats.messages_sent, comm_stats.average_latency_ns / 1000.0);
    } else {
        tracing::error!("   ❌ Cross-module optimizer failed: success={}/{}, latency={:.1}μs",
                       optimizer_success_count, 5, avg_optimizer_latency);
    }
    
    // TEST 4: Real-Time Parameter Adaptation
    tracing::info!("🧪 Testing real-time parameter adaptation...");
    let adaptation_test_start = Instant::now();
    
    // Create mock coupling performance
    let coupling_performance = CouplingPerformance {
        timestamp: Instant::now(),
        pattern_accuracy: 0.75,
        coherence_score: 0.68,
        sync_efficiency: 0.72,
        info_flow_rate: 0.45,
        coupling_energy: 0.3,
        market_regime: MarketRegime::Trending,
    };
    
    // Test parameter adaptation
    let adaptation_result = adaptation_engine.adapt_parameters(
        &mock_drpp_state,
        &mock_multi_result,
        &coherence_patterns,
        &coupling_performance,
    ).await?;
    
    let adaptation_duration = adaptation_test_start.elapsed();
    
    let adaptation_operational = match adaptation_result {
        AdaptationResult::Success { adapted_parameters, performance_improvement, .. } => {
            adaptation_duration < Duration::from_millis(200) &&
            !adapted_parameters.is_empty() &&
            performance_improvement > 0.0
        },
        AdaptationResult::NoAdaptationNeeded => {
            adaptation_duration < Duration::from_millis(50)
        },
        _ => false,
    };
    
    if adaptation_operational {
        tracing::info!("   ✅ Real-time adaptation: {:.2}ms", adaptation_duration.as_millis());
        match adaptation_result {
            AdaptationResult::Success { adapted_parameters, performance_improvement, regime, .. } => {
                tracing::info!("     Adapted {} parameters for regime {:?}, improvement: {:.1}%",
                              adapted_parameters.len(), regime, performance_improvement * 100.0);
            },
            AdaptationResult::NoAdaptationNeeded => {
                tracing::info!("     No adaptation needed - system performing well");
            },
            _ => {}
        }
    } else {
        tracing::error!("   ❌ Real-time adaptation failed: {:.2}ms", adaptation_duration.as_millis());
    }
    
    // TEST 5: End-to-End Network Processing Pipeline
    tracing::info!("🧪 Testing end-to-end network processing pipeline...");
    let pipeline_test_start = Instant::now();
    
    let mut processed_patterns = 0;
    let mut pipeline_successes = 0;
    let mut total_pipeline_latency = Duration::ZERO;
    
    // Process batches of spikes through the complete pipeline
    for batch_start in (0..all_test_spikes.len()).step_by(500).take(5) {
        let batch_end = (batch_start + 500).min(all_test_spikes.len());
        let batch = &all_test_spikes[batch_start..batch_end];
        
        let batch_process_start = Instant::now();
        
        // 1. Multi-timeframe processing
        let multi_result = multi_timeframe.process_multi_timeframe(batch).await?;
        
        // 2. Coherence analysis
        let batch_phases: Vec<f64> = batch.iter()
            .map(|s| (s.timestamp_ns as f64 / 1_000_000.0) % (2.0 * std::f64::consts::PI))
            .collect();
        let batch_coherence = coherence_analyzer.analyze_coherence(&batch_phases)?;
        
        // 3. Create patterns for routing
        let batch_patterns = generate_test_patterns(batch);
        
        // 4. Route patterns
        let mut routed_patterns = 0;
        for pattern in &batch_patterns {
            let pattern_data = PatternData::Drpp(pattern.clone());
            if pattern_router.route_pattern(pattern_data, 120).await.is_ok() {
                routed_patterns += 1;
            }
        }
        
        let batch_duration = batch_process_start.elapsed();
        
        processed_patterns += batch_patterns.len();
        if routed_patterns >= batch_patterns.len() / 2 { // At least 50% routing success
            pipeline_successes += 1;
            total_pipeline_latency += batch_duration;
        }
    }
    
    let pipeline_duration = pipeline_test_start.elapsed();
    let avg_pipeline_latency = if pipeline_successes > 0 {
        total_pipeline_latency.as_micros() as f64 / pipeline_successes as f64
    } else {
        0.0
    };
    
    let pipeline_operational = pipeline_successes >= 4 && // At least 80% success
                              avg_pipeline_latency < 10000.0 && // <10ms per batch
                              pipeline_duration < Duration::from_secs(5);
    
    if pipeline_operational {
        tracing::info!("   ✅ End-to-end pipeline: {:.2}ms total, {:.1}ms avg per batch",
                      pipeline_duration.as_millis(), avg_pipeline_latency / 1000.0);
        tracing::info!("     Processed {} patterns in {} batches", processed_patterns, pipeline_successes);
    } else {
        tracing::error!("   ❌ End-to-end pipeline failed: success={}/5, avg_latency={:.1}ms",
                       pipeline_successes, avg_pipeline_latency / 1000.0);
    }
    
    // Calculate overall metrics
    let total_integration_time = integration_start.elapsed();
    let end_to_end_latency = total_integration_time.as_micros() as f64;
    
    // Network efficiency (patterns processed per second)
    let network_efficiency = processed_patterns as f64 / total_integration_time.as_secs_f64();
    
    // Adaptation effectiveness (based on successful adaptations)
    let adaptation_effectiveness = match adaptation_result {
        AdaptationResult::Success { performance_improvement, .. } => performance_improvement,
        _ => 0.0,
    };
    
    // Communication optimization ratio
    let communication_optimization = if comm_stats.messages_sent > 0 {
        1.0 - (comm_stats.average_latency_ns / 1000000.0) / 1000.0 // Normalized improvement
    } else {
        0.0
    };
    
    // Network stability (based on coupling metrics)
    let network_stability = coupling_metrics.clustering_coefficient * 
                           coupling_metrics.connection_density * 
                           (1.0 - coupling_metrics.small_worldness.clamp(0.0, 1.0));
    
    let integration_result = Phase2CResult {
        coupling_adaptation_operational,
        pattern_routing_operational: routing_operational,
        cross_module_optimization_operational: optimizer_operational,
        realtime_adaptation_operational: adaptation_operational,
        end_to_end_latency_us: end_to_end_latency,
        network_efficiency,
        adaptation_effectiveness,
        communication_optimization,
        total_patterns_processed: processed_patterns,
        network_stability,
    };
    
    // Final validation
    let all_components_operational = coupling_operational && 
                                    routing_operational &&
                                    optimizer_operational &&
                                    adaptation_operational &&
                                    pipeline_operational;
    
    let total_duration = start_time.elapsed();
    
    tracing::info!("🏁 PHASE 2C INTEGRATION TEST COMPLETE");
    tracing::info!("=====================================");
    tracing::info!("⏱️  Total test duration: {:.2}ms", total_duration.as_millis());
    tracing::info!("⚡ End-to-end latency: {:.1}μs", end_to_end_latency);
    tracing::info!("🎯 Network efficiency: {:.0} patterns/sec", network_efficiency);
    tracing::info!("📈 Adaptation effectiveness: {:.1}%", adaptation_effectiveness * 100.0);
    tracing::info!("🔧 Communication optimization: {:.1}%", communication_optimization * 100.0);
    tracing::info!("🌐 Network stability: {:.3}", network_stability);
    tracing::info!("📊 Total patterns processed: {}", processed_patterns);
    
    if all_components_operational {
        tracing::info!("✅ ALL PHASE 2C COMPONENTS OPERATIONAL");
        tracing::info!("   ✓ Coupling Adaptation: Adaptive neural oscillator networks");
        tracing::info!("   ✓ Pattern Routing: Intelligent multi-destination routing");
        tracing::info!("   ✓ Cross-Module Optimizer: Zero-copy communication with compression");
        tracing::info!("   ✓ Real-time Adaptation: AI-driven parameter optimization");
        tracing::info!("   ✓ End-to-End Pipeline: Complete network processing layer");
        tracing::info!("🚀 NEUROMORPHIC TRADING SYSTEM PHASE 2C COMPLETE");
        tracing::info!("🎯 Network Processing Layer fully operational");
        tracing::info!("🎯 Ready for Phase 3 implementation");
    } else {
        tracing::error!("❌ PHASE 2C INTEGRATION FAILURES DETECTED");
        if !coupling_operational { tracing::error!("   ❌ Coupling Adaptation Engine failed"); }
        if !routing_operational { tracing::error!("   ❌ Pattern Routing Engine failed"); }
        if !optimizer_operational { tracing::error!("   ❌ Cross-Module Optimizer failed"); }
        if !adaptation_operational { tracing::error!("   ❌ Real-time Adaptation Engine failed"); }
        if !pipeline_operational { tracing::error!("   ❌ End-to-End Pipeline failed"); }
        tracing::error!("🔧 System requires optimization before Phase 3");
    }
    
    Ok(integration_result)
}

/// Generate test patterns from spikes
fn generate_test_patterns(spikes: &[Spike]) -> Vec<Pattern> {
    let mut patterns = Vec::new();
    
    if spikes.is_empty() {
        return patterns;
    }
    
    // Create patterns based on spike characteristics
    let chunk_size = spikes.len() / 5;
    
    for (i, chunk) in spikes.chunks(chunk_size.max(1)).enumerate() {
        if chunk.is_empty() {
            continue;
        }
        
        let avg_strength = chunk.iter().map(|s| s.strength as f64).sum::<f64>() / chunk.len() as f64;
        let time_span = chunk.last().unwrap().timestamp_ns - chunk.first().unwrap().timestamp_ns;
        
        let pattern_type = match i % 5 {
            0 => PatternType::Emergent,
            1 => PatternType::Synchronous,
            2 => PatternType::Traveling,
            3 => PatternType::Standing,
            _ => PatternType::Chaotic,
        };
        
        patterns.push(Pattern {
            id: i as u64,
            pattern_type,
            strength: avg_strength,
            timestamp: std::time::SystemTime::now(),
            oscillators: chunk.iter().map(|s| s.neuron_id).collect(),
            phase_coherence: avg_strength * 0.8,
            frequency_content: vec![10.0 + i as f64 * 5.0], // Simplified frequency
        });
    }
    
    patterns
}

/// Generate test oscillator phases
fn generate_test_phases(n_oscillators: usize) -> Vec<f64> {
    let mut phases = Vec::with_capacity(n_oscillators);
    let mut rng = rand::thread_rng();
    
    for i in 0..n_oscillators {
        let base_phase = (i as f64 / n_oscillators as f64) * 2.0 * std::f64::consts::PI;
        let noise = (rng.gen::<f64>() - 0.5) * 0.5; // ±0.25 radians noise
        phases.push(base_phase + noise);
    }
    
    phases
}

/// Quick smoke test for Phase 2C components
pub async fn smoke_test_phase_2c() -> Result<bool> {
    tracing::info!("💨 Running Phase 2C smoke test...");
    
    // Quick test of each component
    let bridge = Arc::new(DrppAdpChannel::new().await?);
    let coupling_engine = CouplingAdaptationEngine::new(32, 0.01, AdaptationStrategy::Hebbian);
    let pattern_router = PatternRoutingEngine::new(RoutingStrategy::ContentAware);
    let cross_optimizer = CrossModuleOptimizer::new(bridge)?;
    let adaptation_engine = RealTimeAdaptationEngine::new()?;
    
    // All components created successfully
    tracing::info!("✅ Phase 2C smoke test passed - all components initialized");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_smoke_test() {
        let result = smoke_test_phase_2c().await;
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
    
    #[test]
    fn test_generate_test_patterns() {
        let spikes = vec![
            Spike { neuron_id: 0, timestamp_ns: 1000, strength: 0.5 },
            Spike { neuron_id: 1, timestamp_ns: 2000, strength: 0.8 },
            Spike { neuron_id: 2, timestamp_ns: 3000, strength: 0.6 },
        ];
        
        let patterns = generate_test_patterns(&spikes);
        assert!(!patterns.is_empty());
        
        for pattern in &patterns {
            assert!(pattern.strength > 0.0);
            assert!(pattern.strength <= 1.0);
        }
    }
    
    #[test]
    fn test_generate_test_phases() {
        let phases = generate_test_phases(10);
        assert_eq!(phases.len(), 10);
        
        for phase in &phases {
            assert!(*phase >= 0.0);
            assert!(*phase < 2.0 * std::f64::consts::PI);
        }
    }
}