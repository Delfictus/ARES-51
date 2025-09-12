# Quick Start Guide - Historical Market Data System

## 5-Minute Setup

### 1. Add Dependencies

Ensure your `Cargo.toml` includes:

```toml
[dependencies]
csf-core = { path = "../csf-core" }
tokio = { version = "1.0", features = ["full"] }
chrono = { version = "0.4", features = ["serde"] }
```

### 2. Basic Usage

```rust
use csf_core::prelude::*;
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure data fetching
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec!["AAPL".to_string()],
        start_date: Utc::now() - Duration::days(30),
        end_date: Utc::now(),
        interval: TimeInterval::Daily,
        playback_speed: 1.0,
        max_retries: 3,
        rate_limit_ms: 1000,
    };

    // Fetch data
    let mut fetcher = HistoricalDataFetcher::new(config);
    let data = fetcher.fetch_symbol_data("AAPL").await?;

    println!("✅ Fetched {} data points for AAPL", data.len());

    // Process data
    for point in data.iter().take(5) {
        println!("📊 {}: ${:.2}", 
            point.timestamp.format("%Y-%m-%d"), 
            point.close
        );
    }

    Ok(())
}
```

### 3. Run the Example

```bash
# Run the complete validation example
cargo run --example historical_data_validation

# Or create your own file and run it
cargo run --bin your_program
```

## Common Use Cases

### Case 1: Model Backtesting

```rust
// Fetch 1 year of data for backtesting
let config = HistoricalDataConfig {
    data_source: DataSource::YahooFinance,
    symbols: vec!["AAPL".to_string(), "GOOGL".to_string()],
    start_date: Utc::now() - Duration::days(365),
    end_date: Utc::now(),
    interval: TimeInterval::Daily,
    playback_speed: 1000.0, // Very fast for backtesting
    max_retries: 3,
    rate_limit_ms: 500,
};

let mut fetcher = HistoricalDataFetcher::new(config);
let all_data = fetcher.fetch_all_data().await?;

for (symbol, data) in all_data {
    // Run your model on historical data
    let predictions = run_quantum_model(data).await?;
    println!("Model accuracy for {}: {:.2}%", symbol, calculate_accuracy(&predictions));
}
```

### Case 2: Real-Time Simulation

```rust
// Stream historical data as if it's live
let data = fetcher.fetch_symbol_data("AAPL").await?;

fetcher.replay_data_with_timing(data, |stream_data| {
    // Process each point as it would arrive live
    println!("🔴 LIVE: Processing ${:.2}", extract_close_price(&stream_data));
    
    // Send to your quantum temporal correlation system
    process_real_time_data(stream_data);
}).await?;
```

### Case 3: Multi-Timeframe Analysis

```rust
// Analyze same symbol across different timeframes
let timeframes = vec![
    TimeInterval::OneHour,
    TimeInterval::Daily,
    TimeInterval::Weekly,
];

for interval in timeframes {
    let config = HistoricalDataConfig {
        interval,
        // ... other config
    };
    
    let fetcher = HistoricalDataFetcher::new(config);
    let data = fetcher.fetch_symbol_data("AAPL").await?;
    
    analyze_timeframe(interval, data);
}
```

## Data Sources Quick Reference

| Source | Free | API Key | Rate Limit | Best For |
|--------|------|---------|------------|----------|
| **Yahoo Finance** | ✅ | ❌ | ~5/sec | Development, backtesting |
| **Alpha Vantage** | ✅ (limited) | ✅ | 5/min (free) | Production, comprehensive data |
| **Twelve Data** | ✅ (limited) | ✅ | Variable | Enterprise, global markets |

### Yahoo Finance (Recommended Start)

```rust
let config = HistoricalDataConfig {
    data_source: DataSource::YahooFinance,
    rate_limit_ms: 200, // 5 calls per second
    // ... other config
};
```

### Alpha Vantage (Production Ready)

```rust
// Get free API key from: https://www.alphavantage.co/support/#api-key
let config = HistoricalDataConfig {
    data_source: DataSource::AlphaVantage { 
        api_key: std::env::var("ALPHA_VANTAGE_API_KEY")
            .expect("Set ALPHA_VANTAGE_API_KEY environment variable")
    },
    rate_limit_ms: 12000, // 5 calls per minute (free tier)
    // ... other config
};
```

## Time Intervals

```rust
// High-frequency (requires more API calls)
interval: TimeInterval::OneMinute,    // 1-minute bars
interval: TimeInterval::FiveMinutes,  // 5-minute bars

// Standard analysis
interval: TimeInterval::OneHour,      // Hourly data
interval: TimeInterval::Daily,        // Daily OHLCV

// Long-term analysis
interval: TimeInterval::Weekly,       // Weekly aggregates
interval: TimeInterval::Monthly,      // Monthly aggregates
```

## Error Handling Patterns

```rust
match fetcher.fetch_symbol_data("AAPL").await {
    Ok(data) => {
        println!("✅ Success: {} points", data.len());
    }
    Err(HistoricalDataError::NetworkError(msg)) => {
        eprintln!("🌐 Network issue: {}", msg);
        // Retry or use cached data
    }
    Err(HistoricalDataError::RateLimitExceeded) => {
        eprintln!("⏱️ Rate limited, waiting...");
        tokio::time::sleep(Duration::from_secs(60)).await;
        // Retry with longer delays
    }
    Err(HistoricalDataError::NoDataFound(symbol)) => {
        eprintln!("📊 No data for {}", symbol);
        // Skip this symbol or try alternative
    }
    Err(e) => {
        eprintln!("❌ Error: {}", e);
    }
}
```

## Performance Tips

### Batch Processing
```rust
// Efficient: Fetch all symbols in one go
let all_data = fetcher.fetch_all_data().await?;

// Less efficient: Individual fetches
for symbol in &config.symbols {
    let data = fetcher.fetch_symbol_data(symbol).await?;
}
```

### Caching
```rust
// First call hits API
let data1 = fetcher.fetch_symbol_data("AAPL").await?;

// Second call uses cache (instant)
let data2 = fetcher.fetch_symbol_data("AAPL").await?;

// Clear cache to force fresh data
fetcher.clear_cache();
```

### Rate Limiting
```rust
// Conservative (safe for all sources)
rate_limit_ms: 1000,

// Aggressive (only for premium APIs)
rate_limit_ms: 100,

// Alpha Vantage free tier
rate_limit_ms: 12000, // 12 seconds between calls
```

## Integration with Quantum System

### Convert to MarketDataPoint

```rust
let historical_data = fetcher.fetch_symbol_data("AAPL").await?;

let market_data: Vec<MarketDataPoint> = historical_data
    .into_iter()
    .map(|point| MarketDataPoint {
        timestamp: point.timestamp.timestamp_millis() as u64,
        symbol: point.symbol,
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        volume: point.volume,
        bid: point.close - 0.01,
        ask: point.close + 0.01,
    })
    .collect();

// Process through existing quantum system
let prediction = process_market_data(market_data).await?;
```

### Stream Processing

```rust
use futures::StreamExt;

let data = fetcher.fetch_symbol_data("AAPL").await?;
let mut stream = fetcher.create_replay_stream(data).await;

while let Some(stream_data) = stream.next().await {
    // Send to streaming processor
    streaming_processor.process(stream_data).await;
}
```

## Troubleshooting

### "No data found"
- Check symbol format: "AAPL" not "Apple"
- Verify date range is valid
- Try different data source

### Rate limit errors
- Increase `rate_limit_ms`
- Reduce number of symbols
- Get premium API key

### Network timeouts
- Check internet connection
- Increase `max_retries`
- Switch to different data source

### Compilation errors
- Ensure all dependencies in Cargo.toml
- Check feature flags are enabled
- Update dependency versions

## Next Steps

1. **Read Full Documentation**: `docs/HISTORICAL_DATA_SYSTEM.md`
2. **API Reference**: `docs/API_REFERENCE.md`
3. **Run Examples**: `cargo run --example historical_data_validation`
4. **Set Up Production API Keys**: Get Alpha Vantage key for production use

## Support

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory  
- **Issues**: Check common troubleshooting patterns above

---

**Author**: Ididia Serfaty  
**Contact**: IS@delfictus.com  
**Project**: ARES ChronoFabric System