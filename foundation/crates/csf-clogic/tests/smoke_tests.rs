//! Smoke tests to validate core csf-clogic functionality after major fixes

use csf_bus::PhaseCoherenceBus;
use csf_clogic::*;
use std::sync::Arc;

#[tokio::test]
async fn smoke_test_all_modules_instantiate() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = CLogicConfig::default();

    // Test that all modules can be created without panics
    let system = CLogicSystem::new(bus, config)
        .await
        .expect("System creation failed");

    // Verify all modules start successfully
    system.start().await.expect("System start failed");

    // Get state to verify internal consistency
    let state = system.get_state().await;
    assert!(state.timestamp.as_nanos() > 0, "Invalid timestamp");

    // Clean shutdown
    system.stop().await.expect("System stop failed");
}

#[test]
fn smoke_test_pattern_detector_thread_safety() {
    use csf_clogic::drpp::{DrppConfig, PatternDetector};

    let config = DrppConfig::default();
    let detector = PatternDetector::new(&config);

    // Test concurrent access doesn't panic
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    let oscillators = vec![]; // Empty for smoke test
                    detector.detect(&oscillators); // Should not panic
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });
}

// Test disabled - transfer_entropy module not currently public
// #[test]
// fn smoke_test_transfer_entropy_array_handling() {
//     use csf_clogic::drpp::transfer_entropy::{TransferEntropyEngine, TeConfig};
//     use ndarray::Array2;
//
//     let config = TeConfig::default();
//     let mut engine = TransferEntropyEngine::new(config).expect("Engine creation failed");
//
//     // Test basic array operations don't panic
//     let data = Array2::<f32>::zeros((3, 100));
//
//     // This should not panic after our CircularBuffer fixes
//     for _ in 0..5 {
//         engine.history_buffer.write().push(data.clone());
//     }
//
//     // Verify history can be retrieved
//     let history = engine.history_buffer.read()
//         .get_history_array(3);
//     assert!(history.is_ok(), "History retrieval failed");
// }

#[tokio::test]
async fn smoke_test_egc_consensus_basic_flow() {
    use csf_clogic::egc::{DecisionOption, DecisionType, EgcConfig, EmergentGovernanceController};

    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = EgcConfig::default();

    let egc = EmergentGovernanceController::new(bus, config)
        .await
        .expect("EGC creation failed");

    // Test basic decision submission doesn't panic
    let options = vec![DecisionOption {
        id: "allow".to_string(),
        description: "Allow operation".to_string(),
        impact: 0.5,
    }];

    let decision_id = egc
        .submit_decision(
            DecisionType::SystemConfiguration,
            "Test decision".to_string(),
            options,
        )
        .await
        .expect("Decision submission failed");

    // Verify state is consistent
    let state = egc.get_state().await;
    assert!(!state.pending_decisions.is_empty(), "Decision not recorded");
}
