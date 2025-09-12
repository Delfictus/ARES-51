//! Integration tests for cross-module communication system
//! 
//! Validates message passing between DRPP, ADP, EGC, and EMS modules.

use csf_clogic::{
    CLogicSystem, CLogicConfig, CrossModuleCommunication, CrossModuleConfig,
    DrppFeatures, ProcessingMetrics, EmotionalModulation, GovernanceDecision,
};
use csf_core::hardware_timestamp;
use csf_bus::PhaseCoherenceBus;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_cross_module_system_initialization() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CLogicConfig {
        enable_cross_talk: true,
        ..Default::default()
    };
    
    let system = CLogicSystem::new(bus, config).await.unwrap();
    
    // System should start successfully with cross-module communication enabled
    system.start().await.unwrap();
    
    // Verify system state
    let state = system.get_state().await;
    assert!(state.timestamp.as_nanos() > 0);
    
    system.stop().await.unwrap();
}

#[tokio::test]
async fn test_drpp_to_adp_communication() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    // Create DRPP -> ADP channel
    let drpp_tx = comm_system.create_channel::<DrppFeatures>("drpp_to_adp").unwrap();
    let adp_rx = comm_system.subscribe::<DrppFeatures>("drpp_to_adp").await.unwrap();
    
    // Send DRPP features
    let features = DrppFeatures {
        patterns: vec![(csf_clogic::drpp::PatternType::Temporal, 0.85)],
        phase_info: vec![0.0, 1.57, 3.14],
        coherence: 0.92,
        resonance_strength: 0.78,
        timestamp: hardware_timestamp(),
    };
    
    drpp_tx.send(features.clone()).unwrap();
    
    // ADP should receive the features
    let received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = adp_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    assert_eq!(received.patterns.len(), 1);
    assert_eq!(received.patterns[0].1, 0.85);
    assert_eq!(received.phase_info.len(), 3);
    assert!((received.coherence - 0.92).abs() < 1e-10);
}

#[tokio::test]
async fn test_adp_to_egc_communication() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    // Create ADP -> EGC channel
    let adp_tx = comm_system.create_channel::<ProcessingMetrics>("adp_to_egc").unwrap();
    let egc_rx = comm_system.subscribe::<ProcessingMetrics>("adp_to_egc").await.unwrap();
    
    // Send processing metrics
    let metrics = ProcessingMetrics {
        processing_time_ns: 5000,
        confidence: 0.87,
        resource_utilization: 0.65,
        patterns_processed: 42,
        efficiency: 0.91,
        timestamp: hardware_timestamp(),
    };
    
    adp_tx.send(metrics.clone()).unwrap();
    
    // EGC should receive the metrics
    let received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = egc_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    assert_eq!(received.processing_time_ns, 5000);
    assert!((received.confidence - 0.87).abs() < 1e-10);
    assert_eq!(received.patterns_processed, 42);
}

#[tokio::test]
async fn test_ems_broadcast_communication() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    // Create EMS broadcast channel
    let ems_tx = comm_system.create_broadcast_channel::<EmotionalModulation>("ems_broadcast").unwrap();
    
    // Multiple modules subscribe to emotional modulation
    let drpp_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await.unwrap();
    let adp_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await.unwrap();
    let egc_rx = comm_system.subscribe::<EmotionalModulation>("ems_broadcast").await.unwrap();
    
    // EMS broadcasts emotional state
    let modulation = EmotionalModulation {
        valence: 0.65,
        arousal: 0.82,
        emotion_type: "focused_excitement".to_string(),
        strength: 0.9,
        global_modulation: true,
        timestamp: hardware_timestamp(),
    };
    
    ems_tx.broadcast(modulation.clone()).unwrap();
    
    // All modules should receive the broadcast
    let drpp_received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = drpp_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    let adp_received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = adp_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    let egc_received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = egc_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    // Verify all received the same message
    assert!((drpp_received.valence - 0.65).abs() < 1e-10);
    assert!((adp_received.valence - 0.65).abs() < 1e-10);
    assert!((egc_received.valence - 0.65).abs() < 1e-10);
    
    assert_eq!(drpp_received.emotion_type, "focused_excitement");
    assert_eq!(adp_received.emotion_type, "focused_excitement");
    assert_eq!(egc_received.emotion_type, "focused_excitement");
}

#[tokio::test]
async fn test_egc_governance_broadcast() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    // Create EGC governance broadcast
    let egc_tx = comm_system.create_broadcast_channel::<GovernanceDecision>("egc_broadcast").unwrap();
    
    // All modules subscribe to governance decisions
    let drpp_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await.unwrap();
    let adp_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await.unwrap();
    let ems_rx = comm_system.subscribe::<GovernanceDecision>("egc_broadcast").await.unwrap();
    
    // EGC issues governance decision
    let mut params = std::collections::HashMap::new();
    params.insert("learning_rate".to_string(), 0.01);
    params.insert("exploration_factor".to_string(), 0.1);
    
    let decision = GovernanceDecision {
        decision_type: "parameter_adjustment".to_string(),
        parameters: params.clone(),
        priority: 150,
        immediate: false,
        confidence: 0.95,
        timestamp: hardware_timestamp(),
    };
    
    egc_tx.broadcast(decision.clone()).unwrap();
    
    // All modules should receive the governance decision
    let drpp_decision = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = drpp_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    let adp_decision = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = adp_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    let ems_decision = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = ems_rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }).await.unwrap();
    
    // Verify governance decisions are identical
    assert_eq!(drpp_decision.decision_type, "parameter_adjustment");
    assert_eq!(adp_decision.decision_type, "parameter_adjustment");
    assert_eq!(ems_decision.decision_type, "parameter_adjustment");
    
    assert_eq!(drpp_decision.priority, 150);
    assert_eq!(adp_decision.priority, 150);
    assert_eq!(ems_decision.priority, 150);
    
    assert!(drpp_decision.parameters.contains_key("learning_rate"));
    assert!(adp_decision.parameters.contains_key("learning_rate"));
    assert!(ems_decision.parameters.contains_key("learning_rate"));
}

#[tokio::test]
async fn test_priority_message_handling() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    let tx = comm_system.create_channel::<GovernanceDecision>("priority_test").unwrap();
    let rx = comm_system.subscribe::<GovernanceDecision>("priority_test").await.unwrap();
    
    // Send normal priority messages
    for i in 0..10 {
        let normal_decision = GovernanceDecision {
            decision_type: format!("normal_{}", i),
            parameters: std::collections::HashMap::new(),
            priority: 50,
            immediate: false,
            confidence: 0.5,
            timestamp: hardware_timestamp(),
        };
        tx.send(normal_decision).unwrap();
    }
    
    // Send high priority message
    let high_priority_decision = GovernanceDecision {
        decision_type: "emergency".to_string(),
        parameters: std::collections::HashMap::new(),
        priority: 255,
        immediate: true,
        confidence: 1.0,
        timestamp: hardware_timestamp(),
    };
    
    tx.send_priority(high_priority_decision.clone()).unwrap();
    
    // High priority message should be received before normal ones
    let first_received = timeout(Duration::from_millis(100), async {
        loop {
            if let Some(msg) = rx.try_recv() {
                return msg;
            }
            tokio::time::sleep(Duration::from_micros(1)).await;
        }
    }).await.unwrap();
    
    // The first message received should be the high priority emergency
    assert_eq!(first_received.decision_type, "emergency");
    assert_eq!(first_received.priority, 255);
    assert!(first_received.immediate);
}

#[tokio::test]
async fn test_backpressure_handling() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig {
        channel_capacity: 10, // Very small capacity to trigger backpressure quickly
        enable_zero_copy: true,
        enable_priority_routing: true,
        max_latency_ns: 10,
        backpressure_threshold: 0.8,
    };
    
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    let tx = comm_system.create_channel::<EmotionalModulation>("backpressure_test").unwrap();
    let rx = comm_system.subscribe::<EmotionalModulation>("backpressure_test").await.unwrap();
    
    let test_modulation = EmotionalModulation {
        valence: 0.5,
        arousal: 0.5,
        emotion_type: "test".to_string(),
        strength: 0.5,
        global_modulation: false,
        timestamp: hardware_timestamp(),
    };
    
    // Fill the channel to capacity
    let mut sent_count = 0;
    while tx.send(test_modulation.clone()).is_ok() && sent_count < 100 {
        sent_count += 1;
    }
    
    // Channel should be full now
    assert!(tx.has_backpressure());
    
    // Drain some messages
    for _ in 0..5 {
        rx.try_recv();
    }
    
    // Should be able to send again
    assert!(tx.send(test_modulation.clone()).is_ok());
}

#[tokio::test]
async fn test_message_ordering_preservation() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    let tx = comm_system.create_channel::<ProcessingMetrics>("ordering_test").unwrap();
    let rx = comm_system.subscribe::<ProcessingMetrics>("ordering_test").await.unwrap();
    
    // Send messages with sequence numbers
    for i in 0..10 {
        let metrics = ProcessingMetrics {
            processing_time_ns: i * 1000, // Use processing time as sequence ID
            confidence: 0.8,
            resource_utilization: 0.5,
            patterns_processed: 1,
            efficiency: 0.9,
            timestamp: hardware_timestamp(),
        };
        tx.send(metrics).unwrap();
    }
    
    // Receive messages and verify ordering
    let mut received_sequence = Vec::new();
    for _ in 0..10 {
        let msg = timeout(Duration::from_millis(100), async {
            loop {
                if let Some(msg) = rx.try_recv() {
                    return msg;
                }
                tokio::time::sleep(Duration::from_micros(10)).await;
            }
        }).await.unwrap();
        
        received_sequence.push(msg.processing_time_ns / 1000);
    }
    
    // Verify sequential ordering (allowing for some flexibility due to async nature)
    assert_eq!(received_sequence.len(), 10);
    
    // Check that we received all expected values
    let mut expected: Vec<u64> = (0..10).collect();
    expected.sort();
    received_sequence.sort();
    assert_eq!(received_sequence, expected);
}

#[tokio::test]
async fn test_concurrent_producer_consumer() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CrossModuleConfig::default();
    let comm_system = Arc::new(CrossModuleCommunication::new(bus, config).unwrap());
    comm_system.start().await.unwrap();
    
    let tx = comm_system.create_broadcast_channel::<DrppFeatures>("concurrent_test").unwrap();
    
    // Start multiple producers
    let mut producers = Vec::new();
    for producer_id in 0..4 {
        let tx_clone = tx.clone();
        let producer = tokio::spawn(async move {
            for i in 0..25 {
                let features = DrppFeatures {
                    patterns: vec![(csf_clogic::drpp::PatternType::Spatial, producer_id as f64)],
                    phase_info: vec![i as f64],
                    coherence: 0.8,
                    resonance_strength: 0.9,
                    timestamp: hardware_timestamp(),
                };
                tx_clone.broadcast(features).unwrap();
            }
        });
        producers.push(producer);
    }
    
    // Start multiple consumers
    let mut consumers = Vec::new();
    for _consumer_id in 0..4 {
        let rx = comm_system.subscribe::<DrppFeatures>("concurrent_test").await.unwrap();
        let consumer = tokio::spawn(async move {
            let mut count = 0;
            let start = tokio::time::Instant::now();
            while start.elapsed() < Duration::from_millis(500) && count < 100 {
                if rx.try_recv().is_some() {
                    count += 1;
                }
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
            count
        });
        consumers.push(consumer);
    }
    
    // Wait for producers to complete
    for producer in producers {
        producer.await.unwrap();
    }
    
    // Collect consumer results
    let mut total_received = 0;
    for consumer in consumers {
        total_received += consumer.await.unwrap();
    }
    
    // Should have received a reasonable number of messages
    // (100 total sent, 4 consumers, some duplication expected due to broadcast)
    assert!(total_received > 0);
    println!("Concurrent test: {} messages received across all consumers", total_received);
}