use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use neuromorphic_trading::{
    spike_encoding::{SpikeEncoder, EncoderConfig, EncodingScheme},
    event_bus::{TradeData, Symbol, Side},
    MarketData,
};

fn benchmark_spike_encoding(c: &mut Criterion) {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![
            EncodingScheme::RateCoding,
            EncodingScheme::TemporalCoding,
            EncodingScheme::PopulationCoding,
        ],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    let trade = TradeData {
        symbol: Symbol::AAPL,
        price: 150.0,
        quantity: 100,
        timestamp_ns: 1_000_000_000,
        exchange_timestamp: 1_000_000_000,
        aggressor_side: Side::Buy,
        trade_id: 1,
    };
    
    let market_data = MarketData {
        trade: Some(trade),
        quote: None,
        order_book: None,
        timestamp_ns: 1_000_000_000,
    };
    
    let mut group = c.benchmark_group("spike_encoding");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("encode_trade", |b| {
        b.iter(|| {
            let spikes = encoder.encode_trade(black_box(&trade));
            black_box(spikes);
        });
    });
    
    group.bench_function("encode_market_data", |b| {
        b.iter(|| {
            let spikes = encoder.encode(black_box(&market_data));
            black_box(spikes);
        });
    });
    
    group.bench_function("encode_10k_events", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                let spikes = encoder.encode(black_box(&market_data));
                black_box(spikes);
            }
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_spike_encoding);
criterion_main!(benches);