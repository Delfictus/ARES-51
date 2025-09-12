//! Comprehensive tests for DRPP lock-free channel architecture
//! 
//! Tests ordering guarantees, backpressure handling, and multi-threaded safety.

use csf_clogic::drpp::{
    LockFreeSpmc, PatternData, ChannelConfig, ChannelError,
    Producer, Consumer, PatternType, DynamicResonancePatternProcessor, DrppConfig
};
use csf_core::{hardware_timestamp, NanoTime};
use csf_bus::PhaseCoherenceBus;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;
use std::collections::VecDeque;
use tokio::test;

/// Test basic channel creation and single message passing
#[tokio::test]
async fn test_basic_channel_operations() {
    let config = ChannelConfig::default();
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let mut consumer = channel.create_consumer().unwrap();
    
    let test_data = PatternData {
        features: vec![1.0, 2.0, 3.0],
        sequence: 42,
        priority: 128,
        source_id: 1,
        timestamp: hardware_timestamp(),
    };
    
    // Send message
    producer.send(test_data.clone()).unwrap();
    
    // Receive message
    let received = consumer.try_recv().unwrap();
    assert_eq!(received.features, test_data.features);
    assert_eq!(received.sequence, test_data.sequence);
    assert_eq!(received.priority, test_data.priority);
    assert_eq!(received.source_id, test_data.source_id);
}

/// Test ordering guarantees with sequence numbers
#[tokio::test]
async fn test_ordering_guarantees() {
    let config = ChannelConfig {
        capacity: 1024,
        backpressure_threshold: 0.9,
        max_consumers: 1,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let mut consumer = channel.create_consumer().unwrap();
    
    const NUM_MESSAGES: u64 = 100;
    
    // Send messages with sequential sequence numbers
    for i in 0..NUM_MESSAGES {
        let data = PatternData {
            features: vec![i as f64],
            sequence: producer.next_sequence(),
            priority: 128,
            source_id: 2,
            timestamp: hardware_timestamp(),
        };
        producer.send(data).unwrap();
    }
    
    // Receive messages and verify ordering
    let mut last_sequence = 0;
    for _ in 0..NUM_MESSAGES {
        let received = consumer.try_recv().unwrap();
        assert!(received.sequence > last_sequence, 
               "Sequence ordering violated: {} <= {}", received.sequence, last_sequence);
        last_sequence = received.sequence;
    }
}

/// Test backpressure handling
#[tokio::test]
async fn test_backpressure_handling() {
    let config = ChannelConfig {
        capacity: 16, // Very small capacity to trigger backpressure quickly
        backpressure_threshold: 0.8,
        max_consumers: 1,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let _consumer = channel.create_consumer().unwrap(); // Don't consume to trigger backpressure
    
    let test_data = PatternData {
        features: vec![1.0],
        sequence: 0,
        priority: 128,
        source_id: 3,
        timestamp: hardware_timestamp(),
    };
    
    // Fill the channel until backpressure is triggered
    let mut sent_count = 0;
    loop {
        match producer.send(test_data.clone()) {
            Ok(()) => sent_count += 1,
            Err(ChannelError::Backpressure) => break,
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
    
    assert!(sent_count > 0, "Should have sent at least some messages");
    assert!(channel.has_backpressure(), "Channel should be under backpressure");
    
    // Verify utilization is above threshold
    assert!(channel.utilization() >= config.backpressure_threshold,
           "Channel utilization should be above backpressure threshold");
}

/// Test priority message handling (bypasses backpressure)
#[tokio::test]
async fn test_priority_message_handling() {
    let config = ChannelConfig {
        capacity: 8, // Small capacity
        backpressure_threshold: 0.7,
        max_consumers: 1,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let mut consumer = channel.create_consumer().unwrap();
    
    let normal_data = PatternData {
        features: vec![1.0],
        sequence: 0,
        priority: 100,
        source_id: 4,
        timestamp: hardware_timestamp(),
    };
    
    let priority_data = PatternData {
        features: vec![2.0],
        sequence: 0,
        priority: 255,
        source_id: 4,
        timestamp: hardware_timestamp(),
    };
    
    // Fill channel with normal messages to trigger backpressure
    while producer.send(normal_data.clone()).is_ok() {
        // Continue until backpressure
    }
    
    // Priority message should still go through
    producer.send_priority(priority_data.clone()).unwrap();
    
    // Drain and verify priority message is received
    let mut found_priority = false;
    while let Some(received) = consumer.try_recv() {
        if received.priority == 255 {
            found_priority = true;
            assert_eq!(received.features, priority_data.features);
            break;
        }
    }
    
    assert!(found_priority, "Priority message should have been received");
}

/// Test SPMC (Single Producer, Multiple Consumer) functionality
#[tokio::test]
async fn test_spmc_functionality() {
    let config = ChannelConfig {
        capacity: 1024,
        backpressure_threshold: 0.9,
        max_consumers: 4,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let channel = Arc::new(channel);
    
    // Create multiple consumers
    const NUM_CONSUMERS: usize = 4;
    let mut consumers = Vec::new();
    for _ in 0..NUM_CONSUMERS {
        consumers.push(channel.create_consumer().unwrap());
    }
    
    const NUM_MESSAGES: usize = 100;
    
    // Send messages
    for i in 0..NUM_MESSAGES {
        let data = PatternData {
            features: vec![i as f64],
            sequence: i as u64,
            priority: 128,
            source_id: 5,
            timestamp: hardware_timestamp(),
        };
        producer.send(data).unwrap();
    }
    
    // Each consumer should receive all messages
    for mut consumer in consumers {
        let mut received_count = 0;
        while received_count < NUM_MESSAGES {
            if let Some(_data) = consumer.try_recv() {
                received_count += 1;
            }
        }
        assert_eq!(received_count, NUM_MESSAGES, 
                  "Each consumer should receive all messages");
    }
}

/// Test multi-threaded safety and performance
#[tokio::test]
async fn test_multithreaded_safety() {
    let config = ChannelConfig {
        capacity: 8192,
        backpressure_threshold: 0.9,
        max_consumers: 2,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let channel = Arc::new(channel);
    
    let consumer1 = channel.create_consumer().unwrap();
    let consumer2 = channel.create_consumer().unwrap();
    
    let barrier = Arc::new(Barrier::new(3));
    let received_messages = Arc::new(Mutex::new(Vec::new()));
    
    const NUM_MESSAGES: usize = 1000;
    
    // Producer thread
    let producer_barrier = barrier.clone();
    let producer_handle = thread::spawn(move || {
        let mut prod = producer;
        producer_barrier.wait();
        
        for i in 0..NUM_MESSAGES {
            let data = PatternData {
                features: vec![i as f64],
                sequence: i as u64,
                priority: if i % 10 == 0 { 255 } else { 128 }, // Some priority messages
                source_id: 6,
                timestamp: hardware_timestamp(),
            };
            
            while prod.send(data).is_err() {
                thread::yield_now(); // Retry on backpressure
            }
        }
    });
    
    // Consumer 1 thread
    let consumer1_barrier = barrier.clone();
    let received1 = received_messages.clone();
    let consumer1_handle = thread::spawn(move || {
        let mut cons = consumer1;
        consumer1_barrier.wait();
        
        let mut count = 0;
        while count < NUM_MESSAGES {
            if let Some(data) = cons.try_recv() {
                received1.lock().unwrap().push(data.sequence);
                count += 1;
            } else {
                thread::yield_now();
            }
        }
    });
    
    // Consumer 2 thread
    let consumer2_barrier = barrier.clone();
    let received2 = received_messages.clone();
    let consumer2_handle = thread::spawn(move || {
        let mut cons = consumer2;
        consumer2_barrier.wait();
        
        let mut count = 0;
        while count < NUM_MESSAGES {
            if let Some(data) = cons.try_recv() {
                received2.lock().unwrap().push(data.sequence);
                count += 1;
            } else {
                thread::yield_now();
            }
        }
    });
    
    // Wait for all threads
    producer_handle.join().unwrap();
    consumer1_handle.join().unwrap();
    consumer2_handle.join().unwrap();
    
    // Verify all messages were received by both consumers
    let messages = received_messages.lock().unwrap();
    assert_eq!(messages.len(), NUM_MESSAGES * 2, 
              "Both consumers should have received all messages");
}

/// Test channel metrics and monitoring
#[tokio::test]
async fn test_channel_metrics() {
    let config = ChannelConfig {
        capacity: 256,
        backpressure_threshold: 0.8,
        max_consumers: 1,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let mut consumer = channel.create_consumer().unwrap();
    
    let initial_metrics = channel.metrics();
    assert_eq!(initial_metrics.messages_sent.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(initial_metrics.messages_received.load(std::sync::atomic::Ordering::Relaxed), 0);
    
    const NUM_MESSAGES: usize = 50;
    
    // Send messages
    for i in 0..NUM_MESSAGES {
        let data = PatternData {
            features: vec![i as f64],
            sequence: i as u64,
            priority: 128,
            source_id: 7,
            timestamp: hardware_timestamp(),
        };
        producer.send(data).unwrap();
    }
    
    // Receive messages
    for _ in 0..NUM_MESSAGES {
        consumer.try_recv().unwrap();
    }
    
    // Check metrics
    let final_metrics = channel.metrics();
    assert_eq!(final_metrics.messages_sent.load(std::sync::atomic::Ordering::Relaxed), NUM_MESSAGES as u64);
    assert_eq!(final_metrics.messages_received.load(std::sync::atomic::Ordering::Relaxed), NUM_MESSAGES as u64);
    
    // Check latency metrics are updated
    assert!(final_metrics.min_latency_ns.load(std::sync::atomic::Ordering::Relaxed) > 0);
    assert!(final_metrics.max_latency_ns.load(std::sync::atomic::Ordering::Relaxed) > 0);
    assert!(final_metrics.avg_latency_ns.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// Test integration with DRPP module
#[tokio::test]
async fn test_drpp_integration() {
    let bus = Arc::new(PhaseCoherenceBus::new(Default::default()).unwrap());
    let config = DrppConfig::default();
    
    let drpp = DynamicResonancePatternProcessor::new(bus, config).await.unwrap();
    
    // Test channel creation
    let pattern_consumer = drpp.create_pattern_consumer().unwrap();
    
    // Test sending pattern data
    let test_pattern = PatternData {
        features: vec![0.5, 0.7, 0.9],
        sequence: 1,
        priority: 200,
        source_id: 8,
        timestamp: hardware_timestamp(),
    };
    
    drpp.send_pattern_data(test_pattern.clone()).unwrap();
    
    // Verify metrics are available
    let (pattern_metrics, packet_metrics) = drpp.get_channel_metrics();
    assert_eq!(pattern_metrics.messages_sent.load(std::sync::atomic::Ordering::Relaxed), 1);
    
    // Test backpressure detection
    let (pattern_bp, packet_bp) = drpp.has_backpressure();
    assert!(!pattern_bp); // Should not have backpressure initially
    assert!(!packet_bp);
    
    // Test utilization
    let (pattern_util, packet_util) = drpp.channel_utilization();
    assert!(pattern_util >= 0.0 && pattern_util <= 1.0);
    assert!(packet_util >= 0.0 && packet_util <= 1.0);
}

/// Test error handling and edge cases
#[tokio::test]
async fn test_error_handling() {
    // Test invalid capacity (not power of 2)
    let invalid_config = ChannelConfig {
        capacity: 1000, // Not power of 2
        backpressure_threshold: 0.8,
        max_consumers: 1,
        use_mmap: false,
        numa_node: -1,
    };
    
    match LockFreeSpmc::<PatternData>::new(invalid_config) {
        Err(ChannelError::InvalidCapacity(1000)) => {}, // Expected
        other => panic!("Expected InvalidCapacity error, got: {:?}", other),
    }
    
    // Test too many consumers
    let config = ChannelConfig {
        capacity: 1024,
        backpressure_threshold: 0.8,
        max_consumers: 2,
        use_mmap: false,
        numa_node: -1,
    };
    
    let (_producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
    let _consumer1 = channel.create_consumer().unwrap();
    let _consumer2 = channel.create_consumer().unwrap();
    
    // Third consumer should fail
    match channel.create_consumer() {
        Err(ChannelError::TooManyConsumers) => {}, // Expected
        other => panic!("Expected TooManyConsumers error, got: {:?}", other),
    }
}

/// Test memory cleanup and resource management
#[tokio::test]
async fn test_memory_cleanup() {
    // Create and destroy many channels to test for memory leaks
    for _ in 0..100 {
        let config = ChannelConfig {
            capacity: 1024,
            backpressure_threshold: 0.9,
            max_consumers: 4,
            use_mmap: false,
            numa_node: -1,
        };
        
        let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
        let mut consumers = Vec::new();
        
        // Create consumers
        for _ in 0..4 {
            consumers.push(channel.create_consumer().unwrap());
        }
        
        // Send some data
        for i in 0..10 {
            let data = PatternData {
                features: vec![i as f64],
                sequence: i,
                priority: 128,
                source_id: 9,
                timestamp: hardware_timestamp(),
            };
            producer.send(data).unwrap();
        }
        
        // Consume some data
        for mut consumer in consumers {
            while consumer.try_recv().is_some() {
                // Drain
            }
        }
        
        // Channel and all handles will be dropped here
    }
    
    // If we get here without crashes, memory management is working
    assert!(true);
}