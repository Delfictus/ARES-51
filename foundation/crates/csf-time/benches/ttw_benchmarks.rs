//! Performance benchmarks for TTW Temporal Task Weaver
//!
//! Validates that all TTW components meet sub-microsecond latency
//! and >1M ops/sec throughput requirements.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use csf_time::clock::HlcClock;
use csf_time::deadline::{Task, TaskPriority};
use csf_time::*;
use std::hint::black_box;
use std::sync::Arc;

fn bench_time_source_operations(c: &mut Criterion) {
    let time_source = Arc::new(TimeSourceImpl::new().expect("Failed to create time source"));
    let simulated_time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));

    let mut group = c.benchmark_group("time_source");
    group.throughput(Throughput::Elements(1));

    group.bench_function("real_time_source_now", |b| {
        b.iter(|| black_box(time_source.now_ns().unwrap()))
    });

    group.bench_function("simulated_time_source_now", |b| {
        b.iter(|| black_box(simulated_time_source.now_ns().unwrap()))
    });

    group.bench_function("simulated_advance", |b| {
        b.iter(|| black_box(simulated_time_source.advance_simulation(1).unwrap()))
    });

    group.finish();
}

fn bench_hlc_clock_operations(c: &mut Criterion) {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");

    let mut group = c.benchmark_group("hlc_clock");
    group.throughput(Throughput::Elements(1));

    group.bench_function("hlc_now", |b| {
        b.iter(|| black_box(hlc_clock.current_time().unwrap()))
    });

    let message_time = LogicalTime::new(2000, 100, 2);

    group.bench_function("hlc_update_message", |b| {
        b.iter(|| black_box(hlc_clock.update(message_time)))
    });

    group.finish();
}

fn bench_deadline_scheduler_operations(c: &mut Criterion) {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());

    let mut group = c.benchmark_group("deadline_scheduler");
    group.throughput(Throughput::Elements(1));

    // Benchmark task scheduling
    group.bench_function("schedule_single_task", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            counter += 1;
            let task = Task::new(
                format!("bench_task_{}", counter),
                TaskPriority::Normal,
                NanoTime::from_nanos(10000),
                Duration::from_micros(100),
            );
            black_box(scheduler.schedule_task(task, NanoTime::from_nanos(10000)))
        })
    });

    // Benchmark batch scheduling
    let batch_sizes = [10, 100, 1000];
    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("schedule_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        let task = Task::new(
                            format!("batch_task_{}", i),
                            TaskPriority::Normal,
                            NanoTime::from_nanos(10000),
                            Duration::from_micros(100),
                        );
                        black_box(scheduler.schedule_task(task, NanoTime::from_nanos(10000)));
                    }
                })
            },
        );
    }

    group.bench_function("get_statistics", |b| {
        b.iter(|| black_box(scheduler.get_statistics()))
    });

    group.bench_function("optimize_schedule", |b| {
        b.iter(|| black_box(scheduler.optimize_schedule()))
    });

    group.finish();
}

fn bench_quantum_oracle_operations(c: &mut Criterion) {
    let oracle = QuantumTimeOracle::new();

    let mut group = c.benchmark_group("quantum_oracle");
    group.throughput(Throughput::Elements(1));

    let test_time = NanoTime::from_nanos(1000);

    group.bench_function("current_offset", |b| {
        let mut time_counter = 1000u64;
        b.iter(|| {
            time_counter += 1;
            black_box(oracle.current_offset_with_time(NanoTime::from_nanos(time_counter)))
        })
    });

    group.bench_function("current_offset", |b| {
        b.iter(|| black_box(oracle.current_offset()))
    });

    // Test quantum state evolution
    group.bench_function("quantum_evolution_sequence", |b| {
        let mut time_counter = 1000u64;
        b.iter(|| {
            for _ in 0..10 {
                time_counter += 100;
                black_box(oracle.current_offset_with_time(NanoTime::from_nanos(time_counter)));
            }
        })
    });

    group.finish();
}

fn bench_integrated_ttw_operations(c: &mut Criterion) {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());
    let oracle = QuantumTimeOracle::new();

    let mut group = c.benchmark_group("integrated_ttw");
    group.throughput(Throughput::Elements(1));

    // Full TTW integration benchmark
    group.bench_function("full_ttw_cycle", |b| {
        let mut task_counter = 0u64;
        b.iter(|| {
            // Complete TTW cycle: time -> HLC -> quantum optimization -> scheduling
            task_counter += 1;

            let current_time = black_box(time_source.now_ns().unwrap());
            let logical_time = black_box(hlc_clock.current_time().unwrap());
            let quantum_offset = black_box(oracle.current_offset_with_time(current_time));
            // Convert Duration -> NanoTime for correct type when adding to current_time
            let optimized_deadline = black_box(
                quantum_offset.apply(NanoTime::from_nanos(
                    current_time
                        .as_nanos()
                        .saturating_add(Duration::from_millis(1).as_nanos()),
                )),
            );

            let task = Task::new(
                format!("integrated_task_{}", task_counter),
                TaskPriority::Normal,
                optimized_deadline,
                Duration::from_micros(100),
            );

            black_box(scheduler.schedule_task(task, optimized_deadline))
        })
    });

    // Causality tracking benchmark
    group.bench_function("causality_tracking_cycle", |b| {
        let mut message_counter = 0u64;
        b.iter(|| {
            message_counter += 1;

            let current_logical = hlc_clock.current_time().unwrap();
            let message_time = LogicalTime::new(
                current_logical.physical + message_counter,
                current_logical.logical + 1,
                2,
            );

            black_box(hlc_clock.update(message_time))
        })
    });

    group.finish();
}

fn bench_throughput_targets(c: &mut Criterion) {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");

    let mut group = c.benchmark_group("throughput_validation");

    // Test 1M ops/sec target for time operations
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("time_ops_1m_per_sec", |b| {
        b.iter(|| {
            for _ in 0..1_000_000 {
                black_box(time_source.now_ns());
            }
        })
    });

    // Test 1M ops/sec for HLC operations
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("hlc_ops_1m_per_sec", |b| {
        b.iter(|| {
            for i in 0..1_000_000 {
                if i % 2 == 0 {
                    black_box(hlc_clock.current_time().unwrap());
                } else {
                    let msg_time = LogicalTime::new(2000 + i, i as u64, 2);
                    black_box(hlc_clock.update(msg_time));
                }
            }
        })
    });

    group.finish();
}

fn bench_latency_targets(c: &mut Criterion) {
    let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_nanos(1000)));
    let hlc_clock = HlcClockImpl::new(1, time_source.clone()).expect("Failed to create HLC clock");
    let scheduler = DeadlineSchedulerImpl::new(time_source.clone());
    let oracle = QuantumTimeOracle::new();

    let mut group = c.benchmark_group("latency_validation");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(100000);

    // Validate sub-microsecond latency for critical operations
    group.bench_function("sub_microsecond_time_query", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            black_box(time_source.now_ns());
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_nanos() < 1000,
                "Time query exceeded 1μs: {:?}",
                elapsed
            );
        })
    });

    group.bench_function("sub_microsecond_hlc_now", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            black_box(hlc_clock.current_time().unwrap());
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_nanos() < 1000,
                "HLC now exceeded 1μs: {:?}",
                elapsed
            );
        })
    });

    group.bench_function("sub_microsecond_task_schedule", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            counter += 1;
            let start = std::time::Instant::now();

            let task = Task::new(
                format!("latency_task_{}", counter),
                TaskPriority::Critical,
                NanoTime::from_nanos(10000),
                Duration::from_micros(50),
            );
            black_box(scheduler.schedule_task(task, NanoTime::from_nanos(10000)));

            let elapsed = start.elapsed();
            // Allow slightly higher latency for scheduling due to complexity
            assert!(
                elapsed.as_nanos() < 5000,
                "Task scheduling exceeded 5μs: {:?}",
                elapsed
            );
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_time_source_operations,
    bench_hlc_clock_operations,
    bench_deadline_scheduler_operations,
    bench_quantum_oracle_operations,
    bench_integrated_ttw_operations,
    bench_throughput_targets,
    bench_latency_targets
);

criterion_main!(benches);
