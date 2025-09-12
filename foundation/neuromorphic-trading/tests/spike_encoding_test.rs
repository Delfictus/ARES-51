//! Comprehensive tests for spike encoding implementations

use neuromorphic_trading::{
    spike_encoding::{SpikeEncoder, EncoderConfig, EncodingScheme},
    event_bus::{TradeData, QuoteData, OrderBookData, Symbol, Side},
    MarketData,
};

#[test]
fn test_quote_encoding() {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![EncodingScheme::RateCoding],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    // Test basic quote
    let quote = QuoteData {
        symbol: Symbol::AAPL,
        bid_price: 149.99,
        bid_size: 1000,
        ask_price: 150.01,
        ask_size: 1000,
        timestamp_ns: 1_000_000_000,
        exchange_timestamp: 1_000_000_000,
    };
    
    let spikes = encoder.encode_quote(&quote);
    
    // Should generate spikes for:
    // - Bid price neurons
    // - Ask price neurons
    // - Spread neurons
    // - Imbalance neurons
    assert!(!spikes.is_empty(), "Quote encoding should generate spikes");
    
    // Verify spike properties
    for spike in &spikes {
        assert!(spike.neuron_id < 10_000, "Neuron ID should be in range");
        assert!(spike.strength > 0.0 && spike.strength <= 2.0, "Strength should be reasonable");
        assert_eq!(spike.timestamp_ns, 1_000_000_000, "Timestamp should match");
    }
    
    // Test wide spread
    let wide_quote = QuoteData {
        symbol: Symbol::AAPL,
        bid_price: 149.50,
        bid_size: 1000,
        ask_price: 150.50,
        ask_size: 1000,
        timestamp_ns: 2_000_000_000,
        exchange_timestamp: 2_000_000_000,
    };
    
    let wide_spikes = encoder.encode_quote(&wide_quote);
    assert!(wide_spikes.len() > 0, "Wide spread should generate more spread neurons");
    
    // Test imbalanced quote (bid heavy)
    let bid_heavy_quote = QuoteData {
        symbol: Symbol::AAPL,
        bid_price: 150.00,
        bid_size: 5000,
        ask_price: 150.01,
        ask_size: 500,
        timestamp_ns: 3_000_000_000,
        exchange_timestamp: 3_000_000_000,
    };
    
    let imbalanced_spikes = encoder.encode_quote(&bid_heavy_quote);
    
    // Check for imbalance neurons (high neuron IDs)
    let has_imbalance_neurons = imbalanced_spikes.iter()
        .any(|s| s.neuron_id >= 9950); // Imbalance neurons in high range
    assert!(has_imbalance_neurons, "Imbalanced quote should activate imbalance neurons");
}

#[test]
fn test_orderbook_encoding() {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![EncodingScheme::PopulationCoding],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    // Create realistic order book
    let book = OrderBookData {
        symbol: Symbol::AAPL,
        bids: [
            (150.00, 1000),
            (149.99, 1500),
            (149.98, 2000),
            (149.97, 2500),
            (149.96, 3000),
            (149.95, 2000),
            (149.94, 1500),
            (149.93, 1000),
            (149.92, 500),
            (149.91, 200),
        ],
        asks: [
            (150.01, 1000),
            (150.02, 1500),
            (150.03, 2000),
            (150.04, 2500),
            (150.05, 3000),
            (150.06, 2000),
            (150.07, 1500),
            (150.08, 1000),
            (150.09, 500),
            (150.10, 200),
        ],
        timestamp_ns: 1_000_000_000,
        sequence_number: 1,
    };
    
    let spikes = encoder.encode_orderbook(&book);
    
    assert!(!spikes.is_empty(), "Order book should generate spikes");
    
    // Should encode:
    // - All 10 bid levels
    // - All 10 ask levels
    // - Imbalance signals
    // - Liquidity concentration
    // - Pressure signals
    
    // Verify we have spikes across different neuron ranges
    let price_neurons = spikes.iter().filter(|s| s.neuron_id < 9700).count();
    let imbalance_neurons = spikes.iter().filter(|s| s.neuron_id >= 9800 && s.neuron_id < 9900).count();
    let liquidity_neurons = spikes.iter().filter(|s| s.neuron_id >= 9750 && s.neuron_id < 9800).count();
    
    assert!(price_neurons > 0, "Should have price level neurons");
    assert!(imbalance_neurons > 0, "Should have imbalance neurons");
    
    // Test imbalanced order book
    let mut imbalanced_book = book.clone();
    // Make bids much larger
    for i in 0..10 {
        imbalanced_book.bids[i].1 *= 5;
    }
    
    let imbalanced_spikes = encoder.encode_orderbook(&imbalanced_book);
    
    // Should have more bid-side imbalance neurons activated
    let bid_imbalance_count = imbalanced_spikes.iter()
        .filter(|s| s.neuron_id >= 9800 && s.neuron_id < 9820)
        .count();
    
    assert!(bid_imbalance_count > 5, "Heavy bid book should activate bid imbalance neurons");
}

#[test]
fn test_spread_encoding() {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![EncodingScheme::RateCoding],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    // Test different spread scenarios
    let spreads = vec![
        (149.99, 150.01, 2),   // 2 bps spread
        (149.95, 150.05, 10),  // 10 bps spread
        (149.90, 150.10, 20),  // 20 bps spread
    ];
    
    for (bid, ask, expected_bps) in spreads {
        let quote = QuoteData {
            symbol: Symbol::AAPL,
            bid_price: bid,
            bid_size: 1000,
            ask_price: ask,
            ask_size: 1000,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
        };
        
        let spikes = encoder.encode_quote(&quote);
        
        // Count spread neurons (high neuron IDs)
        let spread_neuron_count = spikes.iter()
            .filter(|s| s.neuron_id >= 9900 && s.neuron_id < 10000)
            .count();
        
        // More spread neurons for wider spreads
        assert!(spread_neuron_count > 0, "Should have spread neurons for {} bps", expected_bps);
        
        if expected_bps > 10 {
            assert!(spread_neuron_count >= 5, "Wide spread should activate more neurons");
        }
    }
}

#[test]
fn test_liquidity_concentration() {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![EncodingScheme::PopulationCoding],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    // Test tight liquidity (concentrated near touch)
    let mut tight_book = OrderBookData {
        symbol: Symbol::AAPL,
        bids: [(0.0, 0); 10],
        asks: [(0.0, 0); 10],
        timestamp_ns: 1_000_000_000,
        sequence_number: 1,
    };
    
    // Concentrate liquidity at top levels
    tight_book.bids[0] = (150.00, 10000);
    tight_book.bids[1] = (149.99, 8000);
    tight_book.bids[2] = (149.98, 6000);
    for i in 3..10 {
        tight_book.bids[i] = (150.00 - (i as f64 * 0.01), 100);
    }
    
    tight_book.asks[0] = (150.01, 10000);
    tight_book.asks[1] = (150.02, 8000);
    tight_book.asks[2] = (150.03, 6000);
    for i in 3..10 {
        tight_book.asks[i] = (150.01 + (i as f64 * 0.01), 100);
    }
    
    let tight_spikes = encoder.encode_orderbook(&tight_book);
    
    // Check for tight market liquidity pattern
    let liquidity_neurons = tight_spikes.iter()
        .filter(|s| s.neuron_id >= 9750 && s.neuron_id < 9760)
        .count();
    
    assert!(liquidity_neurons > 0, "Tight market should activate liquidity concentration neurons");
    
    // Test wide liquidity (distributed)
    let mut wide_book = tight_book.clone();
    // Move liquidity to deeper levels
    wide_book.bids[0] = (150.00, 100);
    wide_book.bids[5] = (149.95, 10000);
    wide_book.bids[6] = (149.94, 8000);
    
    wide_book.asks[0] = (150.01, 100);
    wide_book.asks[5] = (150.06, 10000);
    wide_book.asks[6] = (150.07, 8000);
    
    let wide_spikes = encoder.encode_orderbook(&wide_book);
    
    // Different liquidity pattern for wide market
    let wide_liquidity_neurons = wide_spikes.iter()
        .filter(|s| s.neuron_id >= 9755 && s.neuron_id < 9765)
        .count();
    
    assert!(wide_liquidity_neurons > 0, "Wide market should activate different liquidity neurons");
}

#[test]
fn test_orderbook_pressure() {
    let config = EncoderConfig {
        num_neurons: 10_000,
        encoding_schemes: vec![EncodingScheme::PopulationCoding],
        window_size_ms: 1000,
    };
    
    let mut encoder = SpikeEncoder::new(config);
    
    // Create book with strong bid pressure
    let mut bid_pressure_book = OrderBookData {
        symbol: Symbol::AAPL,
        bids: [
            (150.00, 5000),
            (149.99, 4000),
            (149.98, 3000),
            (149.97, 2000),
            (149.96, 1000),
            (149.95, 1000),
            (149.94, 1000),
            (149.93, 1000),
            (149.92, 1000),
            (149.91, 1000),
        ],
        asks: [
            (150.01, 500),
            (150.02, 400),
            (150.03, 300),
            (150.04, 200),
            (150.05, 100),
            (150.06, 100),
            (150.07, 100),
            (150.08, 100),
            (150.09, 100),
            (150.10, 100),
        ],
        timestamp_ns: 1_000_000_000,
        sequence_number: 1,
    };
    
    let bid_spikes = encoder.encode_orderbook(&bid_pressure_book);
    
    // Check pressure neurons (base 9700)
    let pressure_neurons = bid_spikes.iter()
        .filter(|s| s.neuron_id >= 9700 && s.neuron_id < 9710)
        .count();
    
    assert!(pressure_neurons > 5, "Strong bid pressure should activate many pressure neurons");
    
    // Test balanced book
    for i in 0..10 {
        bid_pressure_book.asks[i].1 = bid_pressure_book.bids[i].1;
    }
    
    let balanced_spikes = encoder.encode_orderbook(&bid_pressure_book);
    
    let balanced_pressure = balanced_spikes.iter()
        .filter(|s| s.neuron_id >= 9700 && s.neuron_id < 9710)
        .count();
    
    // Balanced book should have moderate pressure signal
    assert!(balanced_pressure < pressure_neurons, "Balanced book should have less pressure signal");
}

#[test]
fn test_full_market_data_encoding() {
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
    
    // Create complete market data
    let market_data = MarketData {
        trade: Some(TradeData {
            symbol: Symbol::AAPL,
            price: 150.00,
            quantity: 1000,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
            aggressor_side: Side::Buy,
            trade_id: 1,
        }),
        quote: Some(QuoteData {
            symbol: Symbol::AAPL,
            bid_price: 149.99,
            bid_size: 2000,
            ask_price: 150.01,
            ask_size: 1500,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
        }),
        order_book: Some(OrderBookData {
            symbol: Symbol::AAPL,
            bids: [
                (149.99, 2000),
                (149.98, 1500),
                (149.97, 1000),
                (149.96, 800),
                (149.95, 600),
                (149.94, 400),
                (149.93, 300),
                (149.92, 200),
                (149.91, 100),
                (149.90, 50),
            ],
            asks: [
                (150.01, 1500),
                (150.02, 1200),
                (150.03, 900),
                (150.04, 700),
                (150.05, 500),
                (150.06, 350),
                (150.07, 250),
                (150.08, 150),
                (150.09, 75),
                (150.10, 25),
            ],
            timestamp_ns: 1_000_000_000,
            sequence_number: 1,
        }),
        timestamp_ns: 1_000_000_000,
    };
    
    let spikes = encoder.encode(&market_data);
    
    // Should have spikes from all three data types
    assert!(spikes.len() > 50, "Full market data should generate many spikes");
    
    // Verify spike diversity
    let unique_neurons: std::collections::HashSet<_> = spikes.iter()
        .map(|s| s.neuron_id)
        .collect();
    
    assert!(unique_neurons.len() > 20, "Should activate diverse set of neurons");
    
    // Check we have spikes across different ranges
    let price_range = spikes.iter().filter(|s| s.neuron_id < 8000).count();
    let feature_range = spikes.iter().filter(|s| s.neuron_id >= 9000).count();
    
    assert!(price_range > 0, "Should have price-sensitive neurons");
    assert!(feature_range > 0, "Should have feature-encoding neurons");
}