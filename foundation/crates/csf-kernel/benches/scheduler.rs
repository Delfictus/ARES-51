use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_scheduler_baseline(c: &mut Criterion) {
    c.bench_function("scheduler_baseline", |b| {
        b.iter(|| {
            // Placeholder benchmark to satisfy workspace formatting and build.
            // Replace with real scheduling microbenchmarks.
            black_box(())
        })
    });
}

criterion_group!(benches, bench_scheduler_baseline);
criterion_main!(benches);
