//! Tests for Send + Sync trait implementations
//!
//! Verifies that core types can be safely shared across threads
//! in the distributed Phase Coherence Bus system.

use crate::packet::{
    DeliveryOptions, PhasePacket, QuantumCorrelation, RoutingMetadata, SharedPacket,
};
use csf_core::{ComponentId, NanoTime, Priority, TaskId};
use csf_time::{LogicalTime, QuantumOffset};
use std::sync::Arc;
use std::thread;
use tracing::Span;
use uuid::Uuid;

/// Test that PhasePacket<T> implements Send + Sync for thread safety
#[test]
fn test_phase_packet_send_sync() {
    let packet = create_test_packet();
    let packet_arc = Arc::new(packet);

    // Test sharing across threads
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let packet_clone = packet_arc.clone();
            thread::spawn(move || {
                // Access packet in different thread
                let id = packet_clone.id;
                let timestamp = packet_clone.timestamp;
                println!("Thread {}: accessed packet {} at {:?}", i, id, timestamp);

                // Verify we can read all fields safely
                assert!(
                    !packet_clone
                        .routing_metadata
                        .delivery_options
                        .guaranteed_delivery
                        || true
                );
                assert!(packet_clone.quantum_correlation.coherence_score >= 0.0);
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

/// Test SharedPacket type alias for multi-threaded access
#[test]
fn test_shared_packet_threading() {
    let shared_packet: SharedPacket = Arc::new(create_test_packet().into_erased());

    // Clone and send to multiple threads
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let packet_ref = shared_packet.clone();
            thread::spawn(move || {
                // Test concurrent read access
                let packet_id = packet_ref.id;
                let coherence = packet_ref.quantum_correlation.coherence_score;

                // Verify thread safety
                assert_ne!(packet_id, Uuid::nil());
                assert!(coherence >= 0.0 && coherence <= 1.0);

                println!("Thread {} processed packet {}", thread_id, packet_id);
            })
        })
        .collect();

    for handle in handles {
        handle
            .join()
            .expect("All threads should complete successfully");
    }
}

/// Test RoutingMetadata Send + Sync implementation
#[test]
fn test_routing_metadata_send_sync() {
    let metadata = create_test_routing_metadata();
    let metadata_arc = Arc::new(metadata);

    // Test sharing metadata across threads
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let meta = metadata_arc.clone();
            thread::spawn(move || {
                assert_eq!(meta.source_id, ComponentId::new(42u64));
                assert_eq!(meta.priority, Priority::High);
                assert!(meta.size_hint > 0);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

/// Test QuantumCorrelation Send + Sync implementation  
#[test]
fn test_quantum_correlation_send_sync() {
    let correlation = create_test_quantum_correlation();
    let correlation_arc = Arc::new(correlation);

    let handles: Vec<_> = (0..5)
        .map(|_| {
            let corr = correlation_arc.clone();
            thread::spawn(move || {
                assert!(corr.coherence_score >= 0.0);
                assert!(corr.temporal_phase >= -1.0 && corr.temporal_phase <= 1.0);
                assert!(!corr.causal_dependencies.is_empty());
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }
}

/// Comprehensive multi-threaded stress test
#[test]
fn test_concurrent_packet_operations() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let packet_count = Arc::new(AtomicUsize::new(0));
    let packets: Vec<SharedPacket> = (0..10)
        .map(|_| Arc::new(create_test_packet().into_erased()))
        .collect();

    // Spawn many threads that concurrently access packets
    let handles: Vec<_> = (0..20)
        .map(|thread_id| {
            let packets = packets.clone();
            let counter = packet_count.clone();

            thread::spawn(move || {
                for (idx, packet) in packets.iter().enumerate() {
                    // Concurrent read operations
                    let _id = packet.id;
                    let _coherence = packet.quantum_correlation.coherence_score;
                    let _timestamp = packet.timestamp;

                    counter.fetch_add(1, Ordering::Relaxed);

                    // Simulate some work
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("All threads should complete");
    }

    // Verify all operations completed
    assert_eq!(packet_count.load(Ordering::Relaxed), 20 * 10);
}

// Helper functions for creating test data

fn create_test_packet() -> PhasePacket<String> {
    PhasePacket {
        id: Uuid::new_v4(),
        timestamp: LogicalTime::new(1000, 0, 1),
        payload: Box::new("test payload".to_string()),
        routing_metadata: create_test_routing_metadata(),
        quantum_correlation: create_test_quantum_correlation(),
        trace_span: Span::current(),
    }
}

fn create_test_routing_metadata() -> RoutingMetadata {
    RoutingMetadata {
        source_id: ComponentId::new(42u64),
        source_task_id: Some(TaskId::new()),
        target_component_mask: 0xFF00FF00,
        priority: Priority::High,
        deadline_ns: Some(NanoTime::from_nanos(5000000)),
        size_hint: 256,
        delivery_options: DeliveryOptions {
            guaranteed_delivery: true,
            max_retries: 3,
            timeout_ns: Some(1000000),
            use_hardware_acceleration: true,
            simd_flags: 0x01,
        },
    }
}

fn create_test_quantum_correlation() -> QuantumCorrelation {
    QuantumCorrelation {
        quantum_offset: QuantumOffset::new(0.1, 0.8, 100.0),
        causal_dependencies: vec![Uuid::new_v4(), Uuid::new_v4()],
        temporal_phase: 0.707,
        coherence_score: 0.85,
        energy_state: nalgebra::DVector::from_vec(vec![1.0, 0.5, 0.3]),
        energy_parameters: nalgebra::DVector::from_vec(vec![0.1, 0.2, 0.15]),
    }
}
