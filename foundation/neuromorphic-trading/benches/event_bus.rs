use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use neuromorphic_trading::event_bus::{MarketDataBus, TradeData, QuoteData, OrderBookData, Symbol, Side, BusConfig};
use std::sync::Arc;
use std::thread;

fn benchmark_event_bus(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_bus");
    
    // Single publisher benchmarks
    {
        let bus = MarketDataBus::new(BusConfig::default());
        
        let trade = TradeData {
            symbol: Symbol::AAPL,
            price: 150.0,
            quantity: 100,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
            aggressor_side: Side::Buy,
            trade_id: 1,
        };
        
        group.throughput(Throughput::Elements(1));
        
        group.bench_function("publish_trade", |b| {
            b.iter(|| {
                bus.publish_trade(black_box(trade.clone())).unwrap();
            });
        });
        
        let quote = QuoteData {
            symbol: Symbol::AAPL,
            bid_price: 149.99,
            bid_size: 100,
            ask_price: 150.01,
            ask_size: 100,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
        };
        
        group.bench_function("publish_quote", |b| {
            b.iter(|| {
                bus.publish_quote(black_box(quote.clone())).unwrap();
            });
        });
    }
    
    // Throughput benchmark
    {
        let bus = Arc::new(MarketDataBus::new(BusConfig::default()));
        
        group.bench_function("throughput_1m_trades", |b| {
            b.iter(|| {
                for i in 0..1_000_000 {
                    let trade = TradeData {
                        symbol: Symbol::AAPL,
                        price: 150.0 + (i as f64 * 0.01),
                        quantity: 100,
                        timestamp_ns: i as u64,
                        exchange_timestamp: i as u64,
                        aggressor_side: if i % 2 == 0 { Side::Buy } else { Side::Sell },
                        trade_id: i as u64,
                    };
                    bus.publish_trade(trade).unwrap();
                }
            });
        });
    }
    
    // Multi-subscriber benchmark
    {
        let bus = Arc::new(MarketDataBus::new(BusConfig::default()));
        
        group.bench_function("multi_subscriber", |b| {
            // Create 10 subscribers
            let subscribers: Vec<_> = (0..10).map(|_| bus.subscribe()).collect();
            
            b.iter(|| {
                let trade = TradeData {
                    symbol: Symbol::AAPL,
                    price: 150.0,
                    quantity: 100,
                    timestamp_ns: 1_000_000_000,
                    exchange_timestamp: 1_000_000_000,
                    aggressor_side: Side::Buy,
                    trade_id: 1,
                };
                
                bus.publish_trade(trade).unwrap();
                
                // Each subscriber reads
                for &sub_id in &subscribers {
                    let _ = bus.read_trade(sub_id);
                }
            });
        });
    }
    
    group.finish();
}

fn benchmark_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_bus_latency");
    
    let bus = MarketDataBus::new(BusConfig::default());
    let subscriber = bus.subscribe();
    
    group.bench_function("end_to_end_latency", |b| {
        b.iter(|| {
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price: 150.0,
                quantity: 100,
                timestamp_ns: 1_000_000_000,
                exchange_timestamp: 1_000_000_000,
                aggressor_side: Side::Buy,
                trade_id: 1,
            };
            
            bus.publish_trade(trade).unwrap();
            let _ = bus.read_trade(subscriber);
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_event_bus, benchmark_latency);
criterion_main!(benches);