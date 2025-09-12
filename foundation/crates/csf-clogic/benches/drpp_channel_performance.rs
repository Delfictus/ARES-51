//! High-performance benchmarks for DRPP lock-free channel architecture
//! 
//! This benchmark suite validates the <10ns latency requirement for
//! the lock-free SPMC channels used in the DRPP module.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
    measurement::WallTime,
};
use csf_clogic::drpp::{
    LockFreeSpmc, PatternData, ChannelConfig, ChannelError,
    Producer, Consumer, PatternType
};
use csf_core::{hardware_timestamp, NanoTime};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Benchmark single-threaded send/receive latency
fn bench_single_thread_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("drpp_single_thread_latency");
    
    // Test with different channel capacities
    for capacity in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("send_recv_ns", capacity),
            capacity,
            |b, &capacity| {
                b.iter_custom(|iters| {
                    // Create high-performance channel configuration
                    let config = ChannelConfig {
                        capacity,
                        backpressure_threshold: 0.95,
                        max_consumers: 1,
                        use_mmap: false,
                        numa_node: -1,
                    };
                    
                    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
                    let mut consumer = channel.create_consumer().unwrap();
                    
                    let test_data = PatternData {
                        features: vec![1.0, 2.0, 3.0],
                        sequence: 0,
                        priority: 128,
                        source_id: 0,
                        timestamp: hardware_timestamp(),
                    };
                    
                    let mut total_duration = Duration::ZERO;
                    
                    for i in 0..iters {
                        let mut data = test_data.clone();
                        data.sequence = i;
                        
                        // Measure pure send latency
                        let start = Instant::now();
                        producer.send(data).unwrap();
                        let send_time = start.elapsed();
                        
                        // Measure receive latency
                        let recv_start = Instant::now();
                        while consumer.try_recv().is_none() {
                            // Busy wait for minimum overhead
                        }
                        let recv_time = recv_start.elapsed();
                        
                        total_duration += send_time + recv_time;
                    }
                    
                    total_duration
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark zero-copy send operation latency
fn bench_zero_copy_send(c: &mut Criterion) {
    c.bench_function("drpp_zero_copy_send", |b| {
        let config = ChannelConfig {
            capacity: 32768,
            backpressure_threshold: 0.9,
            max_consumers: 1,
            use_mmap: true, // Enable memory mapping for zero-copy
            numa_node: -1,
        };
        
        let (mut producer, _channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
        
        let test_data = PatternData {
            features: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            sequence: 0,
            priority: 200,
            source_id: 1,
            timestamp: hardware_timestamp(),
        };
        
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            
            for i in 0..iters {
                let mut data = test_data.clone();
                data.sequence = i;
                
                let start = Instant::now();
                black_box(producer.send(data).unwrap());
                total_duration += start.elapsed();
            }
            
            total_duration
        });
    });
}

/// Benchmark priority send performance
fn bench_priority_send(c: &mut Criterion) {
    c.bench_function("drpp_priority_send", |b| {
        let config = ChannelConfig {
            capacity: 16384,
            backpressure_threshold: 0.8,
            max_consumers: 1,
            use_mmap: false,
            numa_node: -1,
        };
        
        let (mut producer, _channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
        
        let priority_data = PatternData {
            features: vec![1.0],
            sequence: 0,
            priority: 255, // Maximum priority
            source_id: 2,
            timestamp: hardware_timestamp(),
        };
        
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            
            for i in 0..iters {
                let mut data = priority_data.clone();
                data.sequence = i;
                
                let start = Instant::now();
                black_box(producer.send_priority(data).unwrap());
                total_duration += start.elapsed();
            }
            
            total_duration
        });
    });
}

/// Benchmark multi-consumer (SPMC) performance
fn bench_spmc_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("drpp_spmc_performance");
    
    for num_consumers in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("consumers", num_consumers),
            num_consumers,
            |b, &num_consumers| {
                b.iter_custom(|iters| {
                    let config = ChannelConfig {
                        capacity: 65536,
                        backpressure_threshold: 0.9,
                        max_consumers: num_consumers,
                        use_mmap: false,
                        numa_node: -1,
                    };
                    
                    let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
                    let channel = Arc::new(channel);
                    
                    // Create consumers
                    let mut consumers = Vec::new();
                    for _ in 0..num_consumers {
                        consumers.push(channel.create_consumer().unwrap());
                    }
                    
                    let test_data = PatternData {
                        features: vec![1.0, 2.0],
                        sequence: 0,
                        priority: 128,
                        source_id: 3,
                        timestamp: hardware_timestamp(),
                    };
                    
                    let start = Instant::now();
                    
                    // Send messages
                    for i in 0..iters {
                        let mut data = test_data.clone();
                        data.sequence = i;
                        producer.send(data).unwrap();
                    }
                    
                    // Wait for all consumers to receive all messages
                    for mut consumer in consumers {
                        for _ in 0..iters {
                            while consumer.try_recv().is_none() {
                                // Busy wait
                            }
                        }
                    }
                    
                    start.elapsed()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark throughput under backpressure
fn bench_backpressure_throughput(c: &mut Criterion) {
    c.bench_function("drpp_backpressure_throughput", |b| {
        let config = ChannelConfig {
            capacity: 1024, // Small capacity to trigger backpressure
            backpressure_threshold: 0.8,
            max_consumers: 1,
            use_mmap: false,
            numa_node: -1,
        };
        
        let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
        let mut consumer = channel.create_consumer().unwrap();
        
        let test_data = PatternData {
            features: vec![0.5],
            sequence: 0,
            priority: 100,
            source_id: 4,
            timestamp: hardware_timestamp(),
        };
        
        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut sent = 0u64;
            let mut received = 0u64;
            
            for i in 0..iters {
                let mut data = test_data.clone();
                data.sequence = i;
                
                // Try to send (may fail due to backpressure)
                if producer.send(data).is_ok() {
                    sent += 1;
                }
                
                // Try to receive to relieve backpressure
                if consumer.try_recv().is_some() {
                    received += 1;
                }
            }
            
            black_box(sent);
            black_box(received);
            start.elapsed()
        });
    });
}

/// Benchmark concurrent producer/consumer in separate threads
fn bench_concurrent_threads(c: &mut Criterion) {
    c.bench_function("drpp_concurrent_threads", |b| {
        b.iter_custom(|iters| {
            let config = ChannelConfig {
                capacity: 32768,
                backpressure_threshold: 0.9,
                max_consumers: 1,
                use_mmap: true,
                numa_node: -1,
            };
            
            let (producer, channel) = LockFreeSpmc::<PatternData>::new(config).unwrap();
            let consumer = channel.create_consumer().unwrap();
            
            let barrier = Arc::new(Barrier::new(2));
            let barrier_clone = barrier.clone();
            
            let producer_barrier = barrier.clone();
            let producer_handle = thread::spawn(move || {
                let mut prod = producer;
                producer_barrier.wait();
                
                let start = Instant::now();
                
                for i in 0..iters {
                    let data = PatternData {
                        features: vec![i as f64],
                        sequence: i,
                        priority: 128,
                        source_id: 5,
                        timestamp: hardware_timestamp(),
                    };
                    
                    while prod.send(data).is_err() {
                        // Retry on backpressure
                        thread::yield_now();
                    }
                }
                
                start.elapsed()
            });
            
            let consumer_handle = thread::spawn(move || {
                let mut cons = consumer;
                barrier_clone.wait();
                
                let start = Instant::now();
                let mut received = 0u64;
                
                while received < iters {
                    if cons.try_recv().is_some() {
                        received += 1;
                    } else {
                        thread::yield_now();
                    }
                }
                
                start.elapsed()
            });
            
            let producer_time = producer_handle.join().unwrap();
            let consumer_time = consumer_handle.join().unwrap();
            
            // Return the maximum time (bottleneck)
            producer_time.max(consumer_time)
        });
    });
}

/// Benchmark memory utilization and allocation patterns
fn bench_memory_efficiency(c: &mut Criterion) {
    c.bench_function("drpp_memory_efficiency", |b| {
        b.iter_custom(|iters| {
            let config = ChannelConfig {
                capacity: 16384,
                backpressure_threshold: 0.9,
                max_consumers: 4,
                use_mmap: true, // Test memory-mapped allocation
                numa_node: -1,
            };
            
            let start = Instant::now();
            
            // Create and destroy channels to test allocation efficiency
            for _ in 0..iters.min(1000) { // Limit iterations for memory test
                let (mut producer, channel) = LockFreeSpmc::<PatternData>::new(config.clone()).unwrap();
                let mut consumer = channel.create_consumer().unwrap();
                
                // Send some data
                let data = PatternData {
                    features: vec![1.0, 2.0, 3.0],
                    sequence: 0,
                    priority: 128,
                    source_id: 6,
                    timestamp: hardware_timestamp(),
                };
                
                producer.send(data).unwrap();
                consumer.try_recv();
                
                // Channel will be dropped here, testing cleanup efficiency
            }
            
            start.elapsed()
        });
    });
}

criterion_group!(
    drpp_channel_benches,
    bench_single_thread_latency,
    bench_zero_copy_send,
    bench_priority_send,
    bench_spmc_performance,
    bench_backpressure_throughput,
    bench_concurrent_threads,
    bench_memory_efficiency
);
criterion_main!(drpp_channel_benches);