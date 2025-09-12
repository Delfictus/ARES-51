# ARES ChronoFabric Historical Data System - API Reference

## Table of Contents
- [Core Types](#core-types)
- [Configuration](#configuration)
- [HistoricalDataFetcher](#historicaldatafetcher)
- [Data Structures](#data-structures)
- [Error Handling](#error-handling)
- [Integration Types](#integration-types)

## Core Types

### `HistoricalDataFetcher`

Primary interface for fetching historical market data.

```rust
pub struct HistoricalDataFetcher {
    config: HistoricalDataConfig,
    client: Client,
    cache: HashMap<String, Vec<HistoricalDataPoint>>,
}
```

#### Constructor

```rust
impl HistoricalDataFetcher {
    /// Create new historical data fetcher with specified configuration
    pub fn new(config: HistoricalDataConfig) -> Self
}
```

**Parameters:**
- `config`: Configuration specifying data sources, symbols, and behavior

**Returns:** New `HistoricalDataFetcher` instance

**Example:**
```rust
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
let fetcher = HistoricalDataFetcher::new(config);
```

#### Methods

##### `fetch_all_data`

```rust
pub async fn fetch_all_data(&mut self) -> Result<HashMap<String, Vec<HistoricalDataPoint>>, HistoricalDataError>
```

Fetch historical data for all configured symbols.

**Returns:** 
- `Ok(HashMap<String, Vec<HistoricalDataPoint>>)`: Map of symbol → data points
- `Err(HistoricalDataError)`: Various error conditions

**Side Effects:**
- Caches fetched data for subsequent calls
- Respects rate limiting configuration
- Updates internal state

**Example:**
```rust
let mut fetcher = HistoricalDataFetcher::new(config);
match fetcher.fetch_all_data().await {
    Ok(all_data) => {
        for (symbol, data_points) in all_data {
            println!("Symbol: {} has {} points", symbol, data_points.len());
        }
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

##### `fetch_symbol_data`

```rust
pub async fn fetch_symbol_data(&mut self, symbol: &str) -> Result<Vec<HistoricalDataPoint>, HistoricalDataError>
```

Fetch historical data for a single symbol.

**Parameters:**
- `symbol`: Stock symbol (e.g., "AAPL", "GOOGL")

**Returns:**
- `Ok(Vec<HistoricalDataPoint>)`: Vector of historical data points
- `Err(HistoricalDataError)`: Error fetching data

**Behavior:**
- Checks cache first to avoid redundant API calls
- Automatically caches successful results
- Respects rate limiting

**Example:**
```rust
let data = fetcher.fetch_symbol_data("AAPL").await?;
println!("Fetched {} data points for AAPL", data.len());
```

##### `create_replay_stream`

```rust
pub async fn create_replay_stream(&self, data: Vec<HistoricalDataPoint>) -> impl futures::Stream<Item = StreamData>
```

Create a streaming replay of historical data compatible with the streaming processor.

**Parameters:**
- `data`: Historical data points to convert to stream

**Returns:** Stream of `StreamData` items

**Usage:**
```rust
use futures::StreamExt;

let data = fetcher.fetch_symbol_data("AAPL").await?;
let mut stream = fetcher.create_replay_stream(data).await;

while let Some(stream_data) = stream.next().await {
    // Process streaming data
    println!("Stream data: {:?}", stream_data.timestamp);
}
```

##### `replay_data_with_timing`

```rust
pub async fn replay_data_with_timing<F>(&self, data: Vec<HistoricalDataPoint>, mut callback: F) -> Result<(), HistoricalDataError>
where
    F: FnMut(StreamData) + Send,
```

Replay historical data with accurate timing control and playback speed.

**Parameters:**
- `data`: Historical data points to replay
- `callback`: Function called for each data point

**Behavior:**
- Maintains original time intervals between data points
- Applies playback speed multiplier from configuration
- Calls callback function for each replayed data point

**Example:**
```rust
let data = fetcher.fetch_symbol_data("AAPL").await?;

fetcher.replay_data_with_timing(data, |stream_data| {
    // Process each data point with accurate timing
    println!("Processing: {}", stream_data.timestamp);
    
    // Send to quantum temporal correlation system
    process_market_data(stream_data);
}).await?;
```

##### `get_cached_data`

```rust
pub fn get_cached_data(&self, symbol: &str) -> Option<&Vec<HistoricalDataPoint>>
```

Retrieve cached data for a symbol without making API calls.

**Parameters:**
- `symbol`: Symbol to retrieve from cache

**Returns:**
- `Some(&Vec<HistoricalDataPoint>)`: Cached data if available
- `None`: No cached data for this symbol

**Example:**
```rust
if let Some(cached) = fetcher.get_cached_data("AAPL") {
    println!("Found {} cached points", cached.len());
} else {
    println!("No cached data for AAPL");
}
```

##### `clear_cache`

```rust
pub fn clear_cache(&mut self)
```

Clear all cached data to force fresh API calls.

**Example:**
```rust
fetcher.clear_cache();
// Next fetch_symbol_data call will hit the API
```

## Configuration

### `HistoricalDataConfig`

Configuration structure for historical data fetching behavior.

```rust
#[derive(Debug, Clone)]
pub struct HistoricalDataConfig {
    pub data_source: DataSource,
    pub symbols: Vec<String>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub interval: TimeInterval,
    pub playback_speed: f64,
    pub max_retries: u32,
    pub rate_limit_ms: u64,
}
```

#### Fields

##### `data_source: DataSource`
Specifies which data provider to use.

**Options:**
```rust
pub enum DataSource {
    YahooFinance,                           // Free, no API key
    AlphaVantage { api_key: String },      // Requires API key
    TwelveData { api_key: String },        // Requires API key  
}
```

**Example:**
```rust
// Free option
let source = DataSource::YahooFinance;

// Premium option
let source = DataSource::AlphaVantage { 
    api_key: "your_api_key".to_string() 
};
```

##### `symbols: Vec<String>`
List of stock symbols to fetch data for.

**Format:** Standard stock ticker symbols
**Example:**
```rust
let symbols = vec![
    "AAPL".to_string(),    // Apple Inc.
    "GOOGL".to_string(),   // Alphabet Inc.
    "MSFT".to_string(),    // Microsoft Corporation
    "^GSPC".to_string(),   // S&P 500 Index
    "EURUSD=X".to_string(), // EUR/USD forex pair
];
```

##### `start_date: DateTime<Utc>` / `end_date: DateTime<Utc>`
Date range for historical data fetching.

**Constraints:**
- `start_date` must be before `end_date`
- Range depends on data source limitations
- Be aware of market holidays and weekends

**Example:**
```rust
let start_date = Utc::now() - Duration::days(365); // 1 year ago
let end_date = Utc::now();                          // Now
```

##### `interval: TimeInterval`
Time interval between data points.

```rust
pub enum TimeInterval {
    OneMinute,
    FiveMinutes, 
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    Daily,
    Weekly,
    Monthly,
}
```

**Data Source Support:**
- Yahoo Finance: All intervals supported
- Alpha Vantage: All intervals supported
- Twelve Data: All intervals supported

**Example:**
```rust
// For high-frequency analysis
let interval = TimeInterval::OneMinute;

// For daily analysis
let interval = TimeInterval::Daily;
```

##### `playback_speed: f64`
Speed multiplier for data replay.

**Values:**
- `1.0`: Real-time playback
- `< 1.0`: Slower than real-time
- `> 1.0`: Faster than real-time
- `1000.0`: Very fast for backtesting

**Example:**
```rust
let playback_speed = 10.0; // 10x real-time for quick validation
```

##### `max_retries: u32`
Maximum number of retry attempts for failed API calls.

**Recommended Values:**
- `3`: Good balance of reliability and speed
- `5`: For unreliable networks
- `1`: For fast failure in production

##### `rate_limit_ms: u64`
Milliseconds to wait between API calls.

**Recommended Values by Source:**
```rust
// Yahoo Finance
rate_limit_ms: 200,     // 5 calls per second

// Alpha Vantage Free
rate_limit_ms: 12000,   // 5 calls per minute

// Alpha Vantage Premium  
rate_limit_ms: 100,     // 10 calls per second
```

## Data Structures

### `HistoricalDataPoint`

Individual market data point containing OHLCV data.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataPoint {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub adjusted_close: Option<f64>,
}
```

#### Fields

##### `timestamp: DateTime<Utc>`
UTC timestamp for this data point.

**Notes:**
- Always in UTC regardless of market timezone
- Aligned to interval boundaries (e.g., daily data at market close)

##### `symbol: String`
Stock symbol this data point represents.

##### `open: f64` / `high: f64` / `low: f64` / `close: f64`
Standard OHLC price data in USD (or native currency).

**Validation:**
- All values must be positive
- `high >= max(open, close)`
- `low <= min(open, close)`

##### `volume: f64`
Trading volume for this time period.

**Units:** Number of shares traded
**Note:** Can be 0 for some instruments or time periods

##### `adjusted_close: Option<f64>`
Split/dividend-adjusted closing price.

**Availability:**
- Yahoo Finance: Usually available
- Alpha Vantage: Available for daily data
- Other sources: May not be available

**Example:**
```rust
let point = HistoricalDataPoint {
    timestamp: Utc::now(),
    symbol: "AAPL".to_string(),
    open: 150.00,
    high: 152.50,
    low: 149.75,
    close: 151.25,
    volume: 50000000.0,
    adjusted_close: Some(151.25),
};
```

### `TimeInterval` Methods

#### `to_yahoo_string`

```rust
impl TimeInterval {
    pub fn to_yahoo_string(&self) -> &'static str
}
```

Convert time interval to Yahoo Finance API format.

**Returns:** String representation for Yahoo Finance API

**Mapping:**
```rust
TimeInterval::OneMinute     => "1m"
TimeInterval::FiveMinutes   => "5m"
TimeInterval::FifteenMinutes => "15m"
TimeInterval::ThirtyMinutes => "30m"
TimeInterval::OneHour       => "1h"
TimeInterval::Daily         => "1d"
TimeInterval::Weekly        => "1wk"
TimeInterval::Monthly       => "1mo"
```

## Error Handling

### `HistoricalDataError`

Comprehensive error type for all data fetching operations.

```rust
#[derive(Debug, Error)]
pub enum HistoricalDataError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("No data found for symbol: {0}")]
    NoDataFound(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}
```

#### Error Types

##### `NetworkError(String)`
Network-related failures (timeouts, DNS resolution, etc.).

**Common Causes:**
- Internet connection issues
- API server downtime
- Firewall blocking requests

**Handling:**
```rust
match error {
    HistoricalDataError::NetworkError(msg) => {
        eprintln!("Network issue: {}", msg);
        // Retry with exponential backoff
        // Or fall back to cached data
    }
    _ => {}
}
```

##### `ApiError(String)`
API-specific errors from data providers.

**Common Causes:**
- Invalid API key
- Rate limit exceeded
- Invalid symbol format
- Unsupported date range

**Handling:**
```rust
match error {
    HistoricalDataError::ApiError(msg) if msg.contains("rate limit") => {
        // Implement backoff strategy
        sleep(Duration::from_secs(60)).await;
        // Retry request
    }
    HistoricalDataError::ApiError(msg) if msg.contains("Invalid symbol") => {
        // Skip this symbol and continue
        eprintln!("Skipping invalid symbol");
    }
    _ => {}
}
```

##### `ParseError(String)`
JSON parsing or data format errors.

**Common Causes:**
- API format changes
- Corrupted response data
- Unexpected data types

##### `NoDataFound(String)`
No data available for the requested symbol/date range.

**Common Causes:**
- Symbol doesn't exist
- Date range predates data availability
- Market was closed during entire range

**Handling:**
```rust
match error {
    HistoricalDataError::NoDataFound(symbol) => {
        eprintln!("No data for {}, trying alternative source", symbol);
        // Try different data source
        // Or skip this symbol
    }
    _ => {}
}
```

##### `RateLimitExceeded`
Hit rate limits for the data source.

**Handling:**
```rust
match error {
    HistoricalDataError::RateLimitExceeded => {
        // Wait before retrying
        sleep(Duration::from_secs(60)).await;
        // Reduce request frequency
        // Or switch to different data source
    }
    _ => {}
}
```

## Integration Types

### StreamData Integration

The historical data system produces `StreamData` compatible with the streaming processor.

```rust
pub struct StreamData {
    pub stream_id: StreamId,
    pub sequence_number: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data_type: DataType,
    pub payload: DataPayload,
    pub metadata: HashMap<String, String>,
}
```

#### Conversion from HistoricalDataPoint

```rust
// From historical_fetcher.rs:312
let stream_data = StreamData {
    stream_id: uuid::Uuid::new_v4(),
    sequence_number: index as u64,
    timestamp: point.timestamp,
    data_type: DataType::PhaseSpaceTrajectory,
    payload: DataPayload::Points {
        points: vec![vec![
            point.open,
            point.high,
            point.low,
            point.close,
            point.volume,
        ]],
        dimension: 5,
    },
    metadata,
};
```

### MarketDataPoint Integration

Direct compatibility with existing market data processing.

```rust
// Conversion example
let market_point = MarketDataPoint {
    timestamp: historical_point.timestamp.timestamp_millis() as u64,
    symbol: historical_point.symbol,
    open: historical_point.open,
    high: historical_point.high,
    low: historical_point.low,
    close: historical_point.close,
    volume: historical_point.volume,
    bid: historical_point.close - 0.01, // Simulate bid/ask spread
    ask: historical_point.close + 0.01,
};
```

## Usage Patterns

### Pattern 1: Simple Data Fetching

```rust
async fn simple_fetch() -> Result<(), Box<dyn std::error::Error>> {
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
    
    let mut fetcher = HistoricalDataFetcher::new(config);
    let data = fetcher.fetch_symbol_data("AAPL").await?;
    
    println!("Fetched {} data points", data.len());
    Ok(())
}
```

### Pattern 2: Multi-Symbol Batch Processing

```rust
async fn batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
        start_date: Utc::now() - Duration::days(90),
        end_date: Utc::now(),
        interval: TimeInterval::Daily,
        playback_speed: 1.0,
        max_retries: 3,
        rate_limit_ms: 500,
    };
    
    let mut fetcher = HistoricalDataFetcher::new(config);
    let all_data = fetcher.fetch_all_data().await?;
    
    for (symbol, data) in all_data {
        println!("Processing {} with {} points", symbol, data.len());
        // Process each symbol's data
    }
    
    Ok(())
}
```

### Pattern 3: Streaming Replay Integration

```rust
async fn streaming_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec!["AAPL".to_string()],
        start_date: Utc::now() - Duration::days(7),
        end_date: Utc::now(),
        interval: TimeInterval::OneHour,
        playback_speed: 10.0, // 10x speed
        max_retries: 3,
        rate_limit_ms: 1000,
    };
    
    let mut fetcher = HistoricalDataFetcher::new(config);
    let data = fetcher.fetch_symbol_data("AAPL").await?;
    
    // Stream the data with timing
    fetcher.replay_data_with_timing(data, |stream_data| {
        // Process each data point as it arrives
        println!("Processing: {:?}", stream_data);
        // Send to quantum system...
    }).await?;
    
    Ok(())
}
```

### Pattern 4: Error Handling and Fallbacks

```rust
async fn robust_fetching() -> Result<Vec<HistoricalDataPoint>, Box<dyn std::error::Error>> {
    // Primary data source
    let primary_config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec!["AAPL".to_string()],
        start_date: Utc::now() - Duration::days(30),
        end_date: Utc::now(),
        interval: TimeInterval::Daily,
        playback_speed: 1.0,
        max_retries: 2,
        rate_limit_ms: 500,
    };
    
    let mut primary_fetcher = HistoricalDataFetcher::new(primary_config);
    
    match primary_fetcher.fetch_symbol_data("AAPL").await {
        Ok(data) => Ok(data),
        Err(e) => {
            eprintln!("Primary source failed: {}, trying fallback", e);
            
            // Fallback to Alpha Vantage
            let fallback_config = HistoricalDataConfig {
                data_source: DataSource::AlphaVantage { 
                    api_key: std::env::var("ALPHA_VANTAGE_API_KEY")? 
                },
                symbols: vec!["AAPL".to_string()],
                start_date: Utc::now() - Duration::days(30),
                end_date: Utc::now(),
                interval: TimeInterval::Daily,
                playback_speed: 1.0,
                max_retries: 2,
                rate_limit_ms: 12000, // Alpha Vantage free tier
            };
            
            let mut fallback_fetcher = HistoricalDataFetcher::new(fallback_config);
            fallback_fetcher.fetch_symbol_data("AAPL").await
                .map_err(|e| format!("Both sources failed. Last error: {}", e).into())
        }
    }
}
```

## Performance Considerations

### Memory Usage
- Each `HistoricalDataPoint` is approximately 72 bytes
- 1 year of daily data ≈ 18KB per symbol
- 1 year of minute data ≈ 18MB per symbol
- Cache grows with number of symbols and time ranges

### Network Usage
- Yahoo Finance responses vary by time range and interval
- Alpha Vantage responses are typically larger and more detailed
- Enable compression in HTTP client for bandwidth efficiency

### Rate Limiting Best Practices
```rust
// Conservative approach for free APIs
let config = HistoricalDataConfig {
    rate_limit_ms: 1000,  // 1 second between calls
    max_retries: 3,
    // ...
};

// Aggressive approach for premium APIs  
let config = HistoricalDataConfig {
    rate_limit_ms: 100,   // 10 calls per second
    max_retries: 5,
    // ...
};
```

---

**Author**: Ididia Serfaty  
**Contact**: IS@delfictus.com  
**Project**: ARES ChronoFabric System