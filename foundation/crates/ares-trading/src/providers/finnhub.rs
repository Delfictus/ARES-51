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

/// Finnhub provider - Real-time market data and financial news
/// Free tier: 60 API calls/minute
/// WebSocket support for real-time data
pub struct FinnhubProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
    rate_limiter: Arc<RwLock<DateTime<Utc>>>,
}

impl FinnhubProvider {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        let api_key = config.api_key
            .ok_or_else(|| anyhow::anyhow!("Finnhub requires an API key"))?;
        
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            api_key,
            base_url: config.base_url.unwrap_or_else(|| "https://finnhub.io/api/v1".to_string()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(Utc::now())),
        })
    }

    async fn rate_limit(&self) {
        let mut last_request = self.rate_limiter.write().await;
        let now = Utc::now();
        let time_since_last = now.signed_duration_since(*last_request);
        
        // 60 requests per minute = 1 request per second
        if time_since_last.num_milliseconds() < 1000 {
            let wait_time = 1000 - time_since_last.num_milliseconds();
            tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;
        }
        *last_request = Utc::now();
    }

    fn get_resolution(&self, interval: &str) -> &str {
        match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "4h" => "240",
            "1d" => "D",
            "1w" => "W",
            "1mo" => "M",
            _ => "D",
        }
    }
}

#[async_trait]
impl MarketDataProvider for FinnhubProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Check cache first (10 second TTL)
        let cache = self.cache.read().await;
        if let Some((quote, timestamp)) = cache.get(symbol) {
            if Utc::now().signed_duration_since(*timestamp).num_seconds() < 10 {
                return Ok(quote.clone());
            }
        }
        drop(cache);

        self.rate_limit().await;

        let url = format!(
            "{}/quote?symbol={}&token={}",
            self.base_url, symbol, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: FinnhubQuote = response.json().await?;
        
        let last_price = Decimal::from_f64_retain(data.current_price).unwrap_or(dec!(0));
        let change = Decimal::from_f64_retain(data.change).unwrap_or(dec!(0));
        let change_percent = Decimal::from_f64_retain(data.percent_change).unwrap_or(dec!(0));
        
        // Finnhub doesn't provide bid/ask in the quote endpoint, so we approximate
        let spread = last_price * dec!(0.0001); // 0.01% spread approximation
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: last_price - spread,
            ask: last_price + spread,
            last: last_price,
            volume: dec!(0), // Volume not provided in quote endpoint
            timestamp: DateTime::from_timestamp(data.timestamp, 0)
                .unwrap_or_else(|| Utc::now()),
            bid_size: dec!(100),
            ask_size: dec!(100),
            open: Decimal::from_f64_retain(data.open_price).unwrap_or(dec!(0)),
            high: Decimal::from_f64_retain(data.high_price).unwrap_or(dec!(0)),
            low: Decimal::from_f64_retain(data.low_price).unwrap_or(dec!(0)),
            close: last_price,
            previous_close: Decimal::from_f64_retain(data.previous_close).unwrap_or(dec!(0)),
            change,
            change_percent,
        };
        
        // Update cache
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), (quote.clone(), Utc::now()));
        
        Ok(quote)
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        // Finnhub doesn't provide order book data in the free tier
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

        let resolution = self.get_resolution(interval);
        let to = Utc::now().timestamp();
        let from = to - (86400 * 30); // 30 days of data
        
        let url = format!(
            "{}/stock/candle?symbol={}&resolution={}&from={}&to={}&token={}",
            self.base_url, symbol, resolution, from, to, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: FinnhubCandleResponse = response.json().await?;
        
        if data.status != "ok" {
            return Err(anyhow::anyhow!("Failed to fetch candles: {:?}", data.status));
        }
        
        let mut candles = Vec::new();
        let count = data.timestamps.len().min(limit);
        
        for i in 0..count {
            candles.push(Candle {
                symbol: symbol.to_string(),
                timestamp: DateTime::from_timestamp(data.timestamps[i], 0)
                    .unwrap_or_else(|| Utc::now()),
                open: Decimal::from_f64_retain(data.open[i]).unwrap_or(dec!(0)),
                high: Decimal::from_f64_retain(data.high[i]).unwrap_or(dec!(0)),
                low: Decimal::from_f64_retain(data.low[i]).unwrap_or(dec!(0)),
                close: Decimal::from_f64_retain(data.close[i]).unwrap_or(dec!(0)),
                volume: Decimal::from_f64_retain(data.volume[i]).unwrap_or(dec!(0)),
                trades: 0,
            });
        }
        
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to Finnhub for symbols: {:?}", symbols);
        // In production, this would establish WebSocket connections
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from Finnhub for symbols: {:?}", symbols);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct FinnhubQuote {
    #[serde(rename = "c")]
    current_price: f64,
    #[serde(rename = "d")]
    change: f64,
    #[serde(rename = "dp")]
    percent_change: f64,
    #[serde(rename = "h")]
    high_price: f64,
    #[serde(rename = "l")]
    low_price: f64,
    #[serde(rename = "o")]
    open_price: f64,
    #[serde(rename = "pc")]
    previous_close: f64,
    #[serde(rename = "t")]
    timestamp: i64,
}

#[derive(Debug, Deserialize)]
struct FinnhubCandleResponse {
    #[serde(rename = "c")]
    close: Vec<f64>,
    #[serde(rename = "h")]
    high: Vec<f64>,
    #[serde(rename = "l")]
    low: Vec<f64>,
    #[serde(rename = "o")]
    open: Vec<f64>,
    #[serde(rename = "s")]
    status: String,
    #[serde(rename = "t")]
    timestamps: Vec<i64>,
    #[serde(rename = "v")]
    volume: Vec<f64>,
}

/// Additional Finnhub features for enhanced market intelligence
pub struct FinnhubEnhanced {
    provider: FinnhubProvider,
}

impl FinnhubEnhanced {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        Ok(Self {
            provider: FinnhubProvider::new(config)?,
        })
    }

    /// Get company news and sentiment
    pub async fn get_company_news(&self, symbol: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<CompanyNews>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/company-news?symbol={}&from={}&to={}&token={}",
            self.provider.base_url,
            symbol,
            from.format("%Y-%m-%d"),
            to.format("%Y-%m-%d"),
            self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let news: Vec<CompanyNews> = response.json().await?;
        
        Ok(news)
    }

    /// Get basic financials (P/E ratio, market cap, etc.)
    pub async fn get_basic_financials(&self, symbol: &str) -> Result<BasicFinancials> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/stock/metric?symbol={}&metric=all&token={}",
            self.provider.base_url, symbol, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let data: BasicFinancialsResponse = response.json().await?;
        
        Ok(data.metric)
    }

    /// Get recommendation trends from analysts
    pub async fn get_recommendations(&self, symbol: &str) -> Result<Vec<Recommendation>> {
        self.provider.rate_limit().await;
        
        let url = format!(
            "{}/stock/recommendation?symbol={}&token={}",
            self.provider.base_url, symbol, self.provider.api_key
        );
        
        let response = self.provider.client.get(&url).send().await?;
        let recommendations: Vec<Recommendation> = response.json().await?;
        
        Ok(recommendations)
    }
}

#[derive(Debug, Deserialize)]
pub struct CompanyNews {
    pub category: String,
    pub datetime: i64,
    pub headline: String,
    pub id: i64,
    pub image: String,
    pub related: String,
    pub source: String,
    pub summary: String,
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct BasicFinancialsResponse {
    pub metric: BasicFinancials,
    pub series: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BasicFinancials {
    #[serde(rename = "10DayAverageTradingVolume")]
    pub ten_day_avg_volume: Option<f64>,
    #[serde(rename = "52WeekHigh")]
    pub fifty_two_week_high: Option<f64>,
    #[serde(rename = "52WeekLow")]
    pub fifty_two_week_low: Option<f64>,
    pub beta: Option<f64>,
    pub dividend_yield_indication_annual: Option<f64>,
    pub market_capitalization: Option<f64>,
    pub pe_annual: Option<f64>,
    pub pe_ttm: Option<f64>,
    pub eps_annual: Option<f64>,
    pub eps_ttm: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Recommendation {
    pub buy: i32,
    pub hold: i32,
    pub period: String,
    pub sell: i32,
    pub strong_buy: i32,
    pub strong_sell: i32,
    pub symbol: String,
}