//! Integration tests for CSF

use csf_bus::packet::PhasePacket;
use csf_core::prelude::*;
use csf_core::NanoTime;
use csf_time::global_time_source;

#[test]
fn test_basic_packet_flow() {
    // This test would normally interact with the full system
    // For now, we just test the core types

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct SensorData {
        timestamp: u64,
        values: Vec<f32>,
    }

    // Create a sensor packet
    let sensor_data = SensorData {
        timestamp: 12345,
        values: vec![1.0, 2.0, 3.0],
    };

    let packet = PhasePacket::new(sensor_data, ComponentId::custom(1))
        .with_priority(Priority::High)
        .with_deadline(
            global_time_source().now_ns().unwrap_or(NanoTime::ZERO) + NanoTime::from_millis(1),
        ); // 1ms deadline

    // Verify packet properties
    assert_eq!(packet.routing_metadata.source_id, ComponentId::custom(1));
    assert_eq!(packet.routing_metadata.priority, Priority::High);
    assert!(packet.routing_metadata.deadline_ns.is_some());
}

#[test]
fn test_core_types() {
    // Test ComponentId
    let id1 = ComponentId::custom(100);
    let id2 = ComponentId::DRPP;
    assert_ne!(id1, id2);

    // Test Priority ordering
    assert!(Priority::Critical > Priority::High);
    assert!(Priority::High > Priority::Normal);
    assert!(Priority::Normal > Priority::Low);

    // Test NanoTime
    let time1 = NanoTime::from_millis(1000);
    let time2 = NanoTime::from_secs(1);
    assert_eq!(time1, time2);
}
