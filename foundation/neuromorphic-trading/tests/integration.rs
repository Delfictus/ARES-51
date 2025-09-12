use neuromorphic_trading::{
    NeuromorphicTradingSystem,
    SystemConfig,
    MarketData,
    event_bus::{TradeData, QuoteData, OrderBookData, Symbol, Side},
};
use std::time::Instant;

#[tokio::test]
async fn test_full_system_integration() {
    // Initialize system
    let config = SystemConfig {
        num_neurons: 1000,
        reservoir_size: 500,
        enable_brian2: false, // Disable for testing without Python
        enable_cuda: false,
    };
    
    let mut system = NeuromorphicTradingSystem::new(config)
        .expect("Failed to create trading system");
    
    // Create test market data
    let trade = TradeData {
        symbol: Symbol::AAPL,
        price: 150.0,
        quantity: 100,
        timestamp_ns: 1_000_000_000,
        exchange_timestamp: 1_000_000_000,
        aggressor_side: Side::Buy,
        trade_id: 1,
    };
    
    let quote = QuoteData {
        symbol: Symbol::AAPL,
        bid_price: 149.99,
        bid_size: 100,
        ask_price: 150.01,
        ask_size: 100,
        timestamp_ns: 1_000_000_000,
        exchange_timestamp: 1_000_000_000,
    };
    
    let market_data = MarketData {
        trade: Some(trade),
        quote: Some(quote),
        order_book: None,
        timestamp_ns: 1_000_000_000,
    };
    
    // Process market data
    let start = Instant::now();
    let signal = system.process_market_data(market_data).await
        .expect("Failed to process market data");
    let latency = start.elapsed();
    
    // Verify signal
    assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    assert!(signal.risk_score >= 0.0 && signal.risk_score <= 1.0);
    
    println!("Signal: {:?}", signal.action);
    println!("Confidence: {:.2}%", signal.confidence * 100.0);
    println!("Risk Score: {:.2}", signal.risk_score);
    println!("Processing Latency: {:?}", latency);
    
    // Latency should be reasonable
    assert!(latency.as_millis() < 100, "Processing took too long: {:?}", latency);
}

#[tokio::test]
async fn test_high_throughput() {
    let config = SystemConfig {
        num_neurons: 100,
        reservoir_size: 50,
        enable_brian2: false,
        enable_cuda: false,
    };
    
    let mut system = NeuromorphicTradingSystem::new(config)
        .expect("Failed to create trading system");
    
    let start = Instant::now();
    let mut signals = Vec::new();
    
    // Process 1000 market events
    for i in 0..1000 {
        let trade = TradeData {
            symbol: Symbol::AAPL,
            price: 150.0 + (i as f64 * 0.01),
            quantity: 100,
            timestamp_ns: 1_000_000_000 + i * 1_000_000,
            exchange_timestamp: 1_000_000_000 + i * 1_000_000,
            aggressor_side: if i % 2 == 0 { Side::Buy } else { Side::Sell },
            trade_id: i as u64,
        };
        
        let market_data = MarketData {
            trade: Some(trade),
            quote: None,
            order_book: None,
            timestamp_ns: 1_000_000_000 + i * 1_000_000,
        };
        
        let signal = system.process_market_data(market_data).await
            .expect("Failed to process market data");
        
        signals.push(signal);
    }
    
    let elapsed = start.elapsed();
    let throughput = 1000.0 / elapsed.as_secs_f64();
    
    println!("Processed 1000 events in {:?}", elapsed);
    println!("Throughput: {:.0} events/second", throughput);
    
    // Should process at least 100 events/second
    assert!(throughput > 100.0, "Throughput too low: {:.0} events/sec", throughput);
}

#[tokio::test]
async fn test_pattern_detection() {
    let config = SystemConfig::default();
    let mut system = NeuromorphicTradingSystem::new(config)
        .expect("Failed to create trading system");
    
    // Simulate a momentum pattern
    let mut momentum_detected = false;
    
    for i in 0..100 {
        let price = 100.0 + (i as f64 * 0.5); // Steady upward momentum
        
        let trade = TradeData {
            symbol: Symbol::AAPL,
            price,
            quantity: 1000 + i * 10, // Increasing volume
            timestamp_ns: 1_000_000_000 + i * 10_000_000,
            exchange_timestamp: 1_000_000_000 + i * 10_000_000,
            aggressor_side: Side::Buy, // All buys
            trade_id: i as u64,
        };
        
        let market_data = MarketData {
            trade: Some(trade),
            quote: None,
            order_book: None,
            timestamp_ns: 1_000_000_000 + i * 10_000_000,
        };
        
        let signal = system.process_market_data(market_data).await
            .expect("Failed to process market data");
        
        // Check if system detected momentum
        if let neuromorphic_trading::signal_fusion::TradeAction::Buy = signal.action {
            if signal.confidence > 0.6 {
                momentum_detected = true;
                println!("Momentum detected at event {} with confidence {:.2}%", 
                         i, signal.confidence * 100.0);
                break;
            }
        }
    }
    
    assert!(momentum_detected, "Failed to detect obvious momentum pattern");
}

#[test]
fn test_memory_pool() {
    use neuromorphic_trading::memory_pool::{MemoryPool, PoolConfig};
    
    let pool = MemoryPool::new(PoolConfig::default())
        .expect("Failed to create memory pool");
    
    // Allocate various sizes
    let ptr1 = pool.allocate(64).expect("Failed to allocate 64 bytes");
    let ptr2 = pool.allocate(256).expect("Failed to allocate 256 bytes");
    let ptr3 = pool.allocate(1024).expect("Failed to allocate 1024 bytes");
    
    // Deallocate
    pool.deallocate(ptr1, 64);
    pool.deallocate(ptr2, 256);
    pool.deallocate(ptr3, 1024);
    
    // Check stats
    let stats = pool.get_stats();
    println!("Memory Pool Stats:");
    println!("  Allocations: {}", stats.allocations);
    println!("  Deallocations: {}", stats.deallocations);
    println!("  Cache Hits: {}", stats.cache_hits);
    println!("  Cache Misses: {}", stats.cache_misses);
    
    // Reallocation should hit cache
    let ptr4 = pool.allocate(64).expect("Failed to reallocate");
    pool.deallocate(ptr4, 64);
    
    let final_stats = pool.get_stats();
    assert!(final_stats.cache_hits > 0, "Cache should have hits");
}

#[test]
fn test_hardware_clock() {
    use neuromorphic_trading::time_source::HardwareClock;
    use std::time::Duration;
    use std::thread;
    
    let clock = HardwareClock::new()
        .expect("Failed to create hardware clock");
    
    let t1 = clock.now_ns();
    thread::sleep(Duration::from_millis(10));
    let t2 = clock.now_ns();
    
    let elapsed_ns = t2 - t1;
    let elapsed_ms = elapsed_ns as f64 / 1_000_000.0;
    
    println!("Hardware Clock Test:");
    println!("  TSC Frequency: {} Hz", clock.get_frequency());
    println!("  Elapsed: {:.3} ms", elapsed_ms);
    
    // Should be approximately 10ms
    assert!(elapsed_ms > 9.0 && elapsed_ms < 15.0, 
            "Timing inaccurate: {:.3}ms", elapsed_ms);
}