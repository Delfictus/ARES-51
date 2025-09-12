//! Throughput benchmarks for CSF

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use csf_core::prelude::*;

fn benchmark_packet_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_creation");

    // Small payload
    group.throughput(Throughput::Elements(1));
    group.bench_function("small_payload", |b| {
        b.iter(|| {
            let packet = PhasePacket::new(black_box(42u32), black_box(ComponentId::DRPP));
            black_box(packet);
        });
    });

    // Medium payload
    group.throughput(Throughput::Bytes(1024));
    group.bench_function("medium_payload", |b| {
        let payload = vec![0u8; 1024];
        b.iter(|| {
            let packet = PhasePacket::new(black_box(payload.clone()), black_box(ComponentId::DRPP));
            black_box(packet);
        });
    });

    // Large payload
    group.throughput(Throughput::Bytes(65536));
    group.bench_function("large_payload", |b| {
        let payload = vec![0u8; 65536];
        b.iter(|| {
            let packet = PhasePacket::new(black_box(payload.clone()), black_box(ComponentId::DRPP));
            black_box(packet);
        });
    });

    group.finish();
}

fn benchmark_packet_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_routing");

    let packet = PhasePacket::new((), ComponentId::DRPP).with_targets(0xFFFFFFFFFFFFFFFF); // All targets

    group.bench_function("check_single_target", |b| {
        b.iter(|| {
            black_box(packet.targets(black_box(ComponentId::ADP)));
        });
    });

    group.bench_function("check_all_targets", |b| {
        b.iter(|| {
            let mut count = 0;
            for i in 0..64 {
                if packet.targets(ComponentId::Sensor(i)) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_packet_creation, benchmark_packet_routing);
criterion_main!(benches);
