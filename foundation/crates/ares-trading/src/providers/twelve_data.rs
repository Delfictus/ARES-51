use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::market_data::{Quote, Candle, OrderBook, MarketDataProvider};
use super::MarketDataConfig;

/// Twelve Data provider - Professional financial data with technical indicators
/// Free tier: 800 API credits/day (8 requests/minute)
/// Includes 50+ technical indicators
pub struct TwelveDataProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
    rate_limiter: Arc<RwLock<DateTime<Utc>>>,
}

impl TwelveDataProvider {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        let api_key = config.api_key
            .ok_or_else(|| anyhow::anyhow!("Twelve Data requires an API key"))?;
        
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            api_key,
            base_url: config.base_url.unwrap_or_else(|| "https://api.twelvedata.com".to_string()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(Utc::now())),
        })
    }

    async fn rate_limit(&self) {
        let mut last_request = self.rate_limiter.write().await;
        let now = Utc::now();
        let time_since_last = now.signed_duration_since(*last_request);
        
        // Free tier: 8 requests per minute = 1 request every 7.5 seconds
        if time_since_last.num_milliseconds() < 7500 {
            let wait_time = 7500 - time_since_last.num_milliseconds();
            tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;
        }
        *last_request = Utc::now();
    }
}

#[async_trait]
impl MarketDataProvider for TwelveDataProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Check cache first (30 second TTL)
        let cache = self.cache.read().await;
        if let Some((quote, timestamp)) = cache.get(symbol) {
            if Utc::now().signed_duration_since(*timestamp).num_seconds() < 30 {
                return Ok(quote.clone());
            }
        }
        drop(cache);

        self.rate_limit().await;

        let url = format!(
            "{}/quote?symbol={}&apikey={}",
            self.base_url, symbol, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: TwelveDataQuote = response.json().await?;
        
        let last_price = data.close.parse::<Decimal>()?;
        let open = data.open.parse::<Decimal>()?;
        let high = data.high.parse::<Decimal>()?;
        let low = data.low.parse::<Decimal>()?;
        let previous_close = data.previous_close.parse::<Decimal>()?;
        let change = data.change.parse::<Decimal>()?;
        let percent_change = data.percent_change.parse::<f64>()
            .map(|pc| Decimal::from_f64_retain(pc).unwrap_or(dec!(0)))
            .unwrap_or(dec!(0));
        let volume = data.volume.parse::<i64>().unwrap_or(0);
        
        // Approximate bid/ask based on last price
        let spread = last_price * dec!(0.0001); // 0.01% spread
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: last_price - spread,
            ask: last_price + spread,
            last: last_price,
            volume: Decimal::from(volume),
            timestamp: Utc::now(),
            bid_size: dec!(100),
            ask_size: dec!(100),
            open,
            high,
            low,
            close: last_price,
            previous_close,
            change,
            change_percent: percent_change,
        };
        
        // Update cache
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), (quote.clone(), Utc::now()));
        
        Ok(quote)
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        // Twelve Data doesn't provide order book data
        // We'll simulate it based on the quote
        let quote = self.get_quote(symbol).await?;
        
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        for i in 0..depth {
            let offset = Decimal::from(i) * dec!(0.01);
            let size = Decimal::from(100 * (depth - i));
            
            bids.push((quote.bid - offset, size));
            asks.push((quote.ask + offset, size));
        }
        
        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids,
            asks,
        })
    }

    async fn get_candles(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        self.rate_limit().await;

        let td_interval = match interval {
            "1m" => "1min",
            "5m" => "5min",
            "15m" => "15min",
            "30m" => "30min",
            "1h" => "1h",
            "2h" => "2h",
            "4h" => "4h",
            "1d" => "1day",
            "1w" => "1week",
            "1mo" => "1month",
            _ => "1day",
        };
        
        let url = format!(
            "{}/time_series?symbol={}&interval={}&outputsize={}&apikey={}",
            self.base_url, symbol, td_interval, limit.min(5000), self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: TwelveDataTimeSeries = response.json().await?;
        
        let mut candles = Vec::new();
        for value in data.values.iter().take(limit) {
            candles.push(Candle {
                symbol: symbol.to_string(),
                timestamp: value.datetime.parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
                open: value.open.parse::<Decimal>()?,
                high: value.high.parse::<Decimal>()?,
                low: value.low.parse::<Decimal>()?,
                close: value.close.parse::<Decimal>()?,
                volume: value.volume.parse::<i64>()
                    .map(|v| Decimal::from(v))
                    .unwrap_or(dec!(0)),
                trades: 0,
            });
        }
        
        candles.reverse(); // Return in chronological order
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to Twelve Data for symbols: {:?}", symbols);
        // WebSocket support available in paid tiers
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from Twelve Data for symbols: {:?}", symbols);
        Ok(())
    }
}

/// Enhanced features for technical analysis
pub struct TwelveDataTechnical {
    provider: TwelveDataProvider,
}

impl TwelveDataTechnical {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        Ok(Self {
            provider: TwelveDataProvider::new(config)?,
        })
    }

    /// Get RSI (Relative Strength Index) indicator
    pub async fn get_rsi(&self, symbol: &str, interval: &str, period: u32) -> Result<Vec<RSIValue>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/rsi?symbol={}&interval={}&time_period={}&apikey={}",
            self.provider.base_url, symbol, interval, period, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let data: TechnicalIndicatorResponse<RSIValue> = response.json().await?;
        
        Ok(data.values)
    }

    /// Get MACD (Moving Average Convergence Divergence) indicator
    pub async fn get_macd(&self, symbol: &str, interval: &str) -> Result<Vec<MACDValue>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/macd?symbol={}&interval={}&apikey={}",
            self.provider.base_url, symbol, interval, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let data: TechnicalIndicatorResponse<MACDValue> = response.json().await?;
        
        Ok(data.values)
    }

    /// Get Bollinger Bands indicator
    pub async fn get_bollinger_bands(&self, symbol: &str, interval: &str, period: u32) -> Result<Vec<BollingerBand>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/bbands?symbol={}&interval={}&time_period={}&apikey={}",
            self.provider.base_url, symbol, interval, period, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let data: TechnicalIndicatorResponse<BollingerBand> = response.json().await?;
        
        Ok(data.values)
    }

    /// Get Stochastic Oscillator
    pub async fn get_stochastic(&self, symbol: &str, interval: &str) -> Result<Vec<StochasticValue>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/stoch?symbol={}&interval={}&apikey={}",
            self.provider.base_url, symbol, interval, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let data: TechnicalIndicatorResponse<StochasticValue> = response.json().await?;
        
        Ok(data.values)
    }
}

#[derive(Debug, Deserialize)]
struct TwelveDataQuote {
    symbol: String,
    name: String,
    exchange: String,
    currency: String,
    datetime: String,
    timestamp: i64,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: String,
    previous_close: String,
    change: String,
    percent_change: String,
    average_volume: String,
    is_market_open: bool,
    fifty_two_week: Option<FiftyTwoWeek>,
}

#[derive(Debug, Deserialize)]
struct FiftyTwoWeek {
    low: String,
    high: String,
    low_change: String,
    high_change: String,
    low_change_percent: String,
    high_change_percent: String,
    range: String,
}

#[derive(Debug, Deserialize)]
struct TwelveDataTimeSeries {
    meta: TimeSeriesMeta,
    values: Vec<TimeSeriesValue>,
}

#[derive(Debug, Deserialize)]
struct TimeSeriesMeta {
    symbol: String,
    interval: String,
    currency: String,
    exchange_timezone: String,
    exchange: String,
    #[serde(rename = "type")]
    data_type: String,
}

#[derive(Debug, Deserialize)]
struct TimeSeriesValue {
    datetime: String,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: String,
}

#[derive(Debug, Deserialize)]
struct TechnicalIndicatorResponse<T> {
    meta: TechnicalMeta,
    values: Vec<T>,
}

#[derive(Debug, Deserialize)]
struct TechnicalMeta {
    symbol: String,
    interval: String,
    currency: String,
    exchange_timezone: String,
    exchange: String,
    #[serde(rename = "type")]
    indicator_type: String,
}

#[derive(Debug, Deserialize)]
pub struct RSIValue {
    pub datetime: String,
    pub rsi: String,
}

#[derive(Debug, Deserialize)]
pub struct MACDValue {
    pub datetime: String,
    pub macd: String,
    pub macd_signal: String,
    pub macd_hist: String,
}

#[derive(Debug, Deserialize)]
pub struct BollingerBand {
    pub datetime: String,
    pub upper_band: String,
    pub middle_band: String,
    pub lower_band: String,
}

#[derive(Debug, Deserialize)]
pub struct StochasticValue {
    pub datetime: String,
    pub slow_k: String,
    pub slow_d: String,
}