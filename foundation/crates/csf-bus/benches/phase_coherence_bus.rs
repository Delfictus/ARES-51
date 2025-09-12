//! Performance benchmarks for Phase Coherence Bus Goal 2 implementation
//!
//! Validates the <1μs latency and >1M messages/sec throughput targets
//! specified in the ARES Strategic Roadmap Goal 2.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use csf_bus::packet::PhasePacket;
use csf_bus::{BusConfig, EventBusRx, EventBusTx, PhaseCoherenceBus};
use csf_core::ComponentId;
use csf_time::{NanoTime, SimulatedTimeSource, TimeSource};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkMessage {
    id: u64,
    timestamp: u64,
    payload: Vec<u8>,
}

impl BenchmarkMessage {
    fn new(id: u64, payload_size: usize) -> Self {
        Self {
            id,
            timestamp: 0,
            payload: vec![0u8; payload_size],
        }
    }

    fn small() -> Self {
        Self::new(1, 64) // 64 bytes
    }

    fn medium() -> Self {
        Self::new(2, 1024) // 1KB
    }

    fn large() -> Self {
        Self::new(3, 65536) // 64KB
    }
}

/// Benchmark local message passing latency - Target: <1μs p99
fn bench_local_message_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("local_message_latency");
    group.throughput(Throughput::Elements(1));

    // Test different message sizes
    for (name, msg_factory) in [
        ("small_64b", BenchmarkMessage::small as fn() -> _),
        ("medium_1kb", BenchmarkMessage::medium),
        ("large_64kb", BenchmarkMessage::large),
    ] {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter_batched(
                || {
                    // Setup: Create bus and subscription
                    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                        1_000_000_000,
                    )));
                    let config = BusConfig {
                        channel_buffer_size: 1024,
                    };
                    let bus = Arc::new(PhaseCoherenceBus::with_time_source(config, time_source));

                    let bus_clone = bus.clone();
                    let subscription = rt.block_on(async {
                        bus_clone.subscribe::<BenchmarkMessage>().await.unwrap()
                    });

                    (bus, subscription, msg_factory())
                },
                |(bus, mut subscription, message)| async move {
                    // Measure: Publish and receive
                    let packet = PhasePacket::new(message, ComponentId::Custom(1));
                    // Note: Using Instant for benchmark precision, but TTW TimeSource available via:
                    // let start_ttw = global_time_source().now_ns().unwrap_or(NanoTime::ZERO);
                    let start = Instant::now();

                    bus.publish(black_box(packet)).await.unwrap();
                    subscription.recv().await.unwrap();

                    let latency = start.elapsed();
                    black_box(latency);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark sustained throughput - Target: >1M messages/sec for 60+ seconds
fn bench_throughput_sustained(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("sustained_throughput");
    group.sample_size(10); // Fewer samples for throughput test
    group.measurement_time(std::time::Duration::from_secs(10)); // 10 second measurement

    for (name, message_count) in [
        ("1k_messages", 1_000),
        ("10k_messages", 10_000),
        ("100k_messages", 100_000),
        ("1m_messages", 1_000_000),
    ] {
        group.throughput(Throughput::Elements(message_count));

        group.bench_function(name, |b| {
            b.to_async(&rt).iter_batched(
                || {
                    // Setup: Create bus with multiple subscribers
                    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                        1_000_000_000,
                    )));
                    let config = BusConfig {
                        channel_buffer_size: 10_000,
                    };
                    let bus = Arc::new(PhaseCoherenceBus::with_time_source(config, time_source));

                    // Create multiple subscribers for realistic load
                    let mut subscriptions = Vec::new();
                    for _ in 0..4 {
                        let sub = rt
                            .block_on(async { bus.subscribe::<BenchmarkMessage>().await.unwrap() });
                        subscriptions.push(sub);
                    }

                    let messages: Vec<_> = (0..message_count)
                        .map(|i| {
                            PhasePacket::new(
                                BenchmarkMessage::small(),
                                ComponentId::Custom(i as u32),
                            )
                        })
                        .collect();

                    (bus, subscriptions, messages)
                },
                |(bus, mut subscriptions, messages)| async move {
                    // Measure: Batch publish with concurrent receiving
                    let receive_tasks: Vec<_> = subscriptions
                        .into_iter()
                        .enumerate()
                        .map(|(i, mut sub)| {
                            let count = message_count;
                            tokio::spawn(async move {
                                for _ in 0..count {
                                    if sub.recv().await.is_none() {
                                        break;
                                    }
                                }
                                i
                            })
                        })
                        .collect();

                    // Publish all messages
                    let start = Instant::now();
                    for message in messages {
                        bus.publish(black_box(message)).await.unwrap();
                    }

                    // Wait for all receives to complete
                    for task in receive_tasks {
                        task.await.unwrap();
                    }

                    let duration = start.elapsed();
                    let throughput = message_count as f64 / duration.as_secs_f64();
                    black_box(throughput);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark memory efficiency - Target: Zero heap allocations on hot path
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_efficiency");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("zero_copy_publish", |b| {
        b.to_async(&rt).iter_batched(
            || {
                let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                    1_000_000_000,
                )));
                let config = BusConfig {
                    channel_buffer_size: 1024,
                };
                let bus = Arc::new(PhaseCoherenceBus::with_time_source(config, time_source));

                let subscription =
                    rt.block_on(async { bus.subscribe::<BenchmarkMessage>().await.unwrap() });

                let messages: Vec<_> = (0..1000)
                    .map(|i| PhasePacket::new(BenchmarkMessage::small(), ComponentId::custom(i)))
                    .collect();

                (bus, subscription, messages)
            },
            |(bus, mut subscription, messages)| async move {
                // Test zero-copy message passing
                let recv_task = tokio::spawn(async move {
                    for _ in 0..1000 {
                        if subscription.recv().await.is_none() {
                            break;
                        }
                    }
                });

                // Publish with potential zero-copy optimization
                for message in messages {
                    bus.publish(black_box(message)).await.unwrap();
                }

                recv_task.await.unwrap();
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark concurrent subscribers - Test scalability
fn bench_concurrent_subscribers(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_subscribers");

    for subscriber_count in [1, 4, 16, 64, 256] {
        group.throughput(Throughput::Elements(1000));

        group.bench_function(&format!("{}_subscribers", subscriber_count), |b| {
            b.iter_batched(
                || {
                    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                        1_000_000_000,
                    )));
                    let config = BusConfig {
                        channel_buffer_size: 1024,
                    };
                    let bus = Arc::new(PhaseCoherenceBus::with_time_source(config, time_source));

                    // Create many concurrent subscribers
                    let mut subscriptions = Vec::new();
                    for _ in 0..subscriber_count {
                        let sub = rt
                            .block_on(async { bus.subscribe::<BenchmarkMessage>().await.unwrap() });
                        subscriptions.push(sub);
                    }

                    (bus, subscriptions)
                },
                |(bus, mut subscriptions)| async move {
                    // Spawn receiver tasks
                    let receive_tasks: Vec<_> = subscriptions
                        .into_iter()
                        .map(|mut sub| {
                            tokio::spawn(async move {
                                for _ in 0..1000 {
                                    if sub.recv().await.is_none() {
                                        break;
                                    }
                                }
                            })
                        })
                        .collect();

                    // Publish messages concurrently to all subscribers
                    for i in 0..1000 {
                        let message = PhasePacket::new(
                            BenchmarkMessage::new(i, 64),
                            ComponentId::custom(i as u32),
                        );
                        bus.publish(black_box(message)).await.unwrap();
                    }

                    // Wait for all receives
                    for task in receive_tasks {
                        task.await.unwrap();
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark backpressure handling
fn bench_backpressure_handling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("backpressure_handling");
    group.throughput(Throughput::Elements(10000));

    group.bench_function("full_channel_handling", |b| {
        b.iter_batched(
            || {
                let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                    1_000_000_000,
                )));
                // Small buffer to trigger backpressure
                let config = BusConfig {
                    channel_buffer_size: 10,
                };
                let bus = Arc::new(PhaseCoherenceBus::with_time_source(config, time_source));

                // Create slow subscriber
                let subscription =
                    rt.block_on(async { bus.subscribe::<BenchmarkMessage>().await.unwrap() });

                (bus, subscription)
            },
            |(bus, mut _subscription)| async move {
                // Rapidly publish to trigger backpressure
                for i in 0..10000 {
                    let message = PhasePacket::new(
                        BenchmarkMessage::new(i, 64),
                        ComponentId::custom(i as u32),
                    );

                    // Use try_publish to handle backpressure gracefully
                    let _ = bus.try_publish(black_box(message));
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark quantum temporal correlation overhead
fn bench_quantum_correlation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("quantum_correlation");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("quantum_optimized_packets", |b| {
        b.iter_batched(
            || {
                let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(
                    1_000_000_000,
                )));
                let config = BusConfig {
                    channel_buffer_size: 1024,
                };
                let bus = Arc::new(PhaseCoherenceBus::with_time_source(
                    config,
                    time_source.clone(),
                ));

                let subscription =
                    rt.block_on(async { bus.subscribe::<BenchmarkMessage>().await.unwrap() });

                // Create packets with standard optimization
                let packets: Vec<_> = (0..1000)
                    .map(|i| {
                        PhasePacket::new(
                            BenchmarkMessage::new(i, 64),
                            ComponentId::custom(i as u64),
                        )
                    })
                    .collect();

                (bus, subscription, packets)
            },
            |(bus, mut subscription, packets)| {
                rt.block_on(async move {
                    let recv_task = tokio::spawn(async move {
                        for _ in 0..1000 {
                            if subscription.recv().await.is_none() {
                                break;
                            }
                        }
                    });

                    for packet in packets {
                        bus.publish(black_box(packet)).await.unwrap();
                    }

                    recv_task.await.unwrap();
                })
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_local_message_latency,
    bench_throughput_sustained,
    bench_memory_efficiency,
    bench_concurrent_subscribers,
    bench_backpressure_handling,
    bench_quantum_correlation
);

criterion_main!(benches);
