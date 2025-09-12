use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use neuromorphic_trading::{
    reservoir::{LiquidStateMachine, ReservoirConfig, PatternType},
    spike_encoding::Spike,
};

fn benchmark_reservoir(c: &mut Criterion) {
    let config = ReservoirConfig {
        size: 5000,
        spectral_radius: 0.95,
        connection_probability: 0.2,
        leak_rate: 0.1,
    };
    
    let mut reservoir = LiquidStateMachine::new(config);
    
    // Create test spike data
    let spikes: Vec<Spike> = (0..100)
        .map(|i| Spike {
            timestamp_ns: 1_000_000_000 + i * 1_000_000,
            neuron_id: i % 100,
            strength: 1.0,
        })
        .collect();
    
    let mut group = c.benchmark_group("reservoir");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("process_100_spikes", |b| {
        b.iter(|| {
            let state = reservoir.process(black_box(&spikes));
            black_box(state);
        });
    });
    
    group.bench_function("pattern_detection", |b| {
        // First process some spikes to get state
        reservoir.process(&spikes);
        
        b.iter(|| {
            let patterns = reservoir.detect_patterns();
            black_box(patterns);
        });
    });
    
    group.bench_function("compute_separation", |b| {
        let spikes2: Vec<Spike> = (0..100)
            .map(|i| Spike {
                timestamp_ns: 1_000_000_000 + i * 1_000_000,
                neuron_id: (i + 50) % 100,
                strength: 0.8,
            })
            .collect();
        
        b.iter(|| {
            let separation = reservoir.compute_separation(black_box(&spikes), black_box(&spikes2));
            black_box(separation);
        });
    });
    
    // Benchmark continuous processing
    group.bench_function("continuous_1000_timesteps", |b| {
        b.iter(|| {
            for t in 0..1000 {
                let dynamic_spikes: Vec<Spike> = (0..10)
                    .map(|i| Spike {
                        timestamp_ns: t * 1_000_000 + i * 100_000,
                        neuron_id: (t as u32 + i) % 100,
                        strength: 0.5 + (t as f32 * 0.001),
                    })
                    .collect();
                
                let state = reservoir.process(&dynamic_spikes);
                black_box(state);
            }
        });
    });
    
    group.finish();
}

fn benchmark_large_reservoir(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_reservoir");
    
    // Test with different reservoir sizes
    for size in &[1000, 2500, 5000, 10000] {
        let config = ReservoirConfig {
            size: *size,
            spectral_radius: 0.95,
            connection_probability: 0.1, // Lower for larger networks
            leak_rate: 0.1,
        };
        
        let mut reservoir = LiquidStateMachine::new(config);
        
        let spikes: Vec<Spike> = (0..100)
            .map(|i| Spike {
                timestamp_ns: 1_000_000_000 + i * 1_000_000,
                neuron_id: i % 100,
                strength: 1.0,
            })
            .collect();
        
        group.bench_function(&format!("size_{}", size), |b| {
            b.iter(|| {
                let state = reservoir.process(black_box(&spikes));
                black_box(state);
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_reservoir, benchmark_large_reservoir);
criterion_main!(benches);