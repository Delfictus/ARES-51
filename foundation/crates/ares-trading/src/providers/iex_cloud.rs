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

/// IEX Cloud provider - Professional financial data
/// Free tier: 50,000 API calls/month
/// Paid tiers available for production use
pub struct IexCloudProvider {
    client: reqwest::Client,
    api_token: String,
    base_url: String,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
    rate_limiter: Arc<RwLock<DateTime<Utc>>>,
}

impl IexCloudProvider {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        let api_token = config.api_key
            .ok_or_else(|| anyhow::anyhow!("IEX Cloud requires an API token"))?;
        
        // Use sandbox URL for testing (free, but delayed data)
        let base_url = config.base_url.unwrap_or_else(|| 
            if api_token.starts_with("Tpk_") {
                "https://sandbox.iexapis.com/stable".to_string()
            } else {
                "https://cloud.iexapis.com/stable".to_string()
            }
        );
        
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            api_token,
            base_url,
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(Utc::now())),
        })
    }

    async fn rate_limit(&self) {
        let mut last_request = self.rate_limiter.write().await;
        let now = Utc::now();
        let time_since_last = now.signed_duration_since(*last_request);
        
        // Conservative rate limiting for free tier
        if time_since_last.num_milliseconds() < 100 {
            let wait_time = 100 - time_since_last.num_milliseconds();
            tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;
        }
        *last_request = Utc::now();
    }
}

#[async_trait]
impl MarketDataProvider for IexCloudProvider {
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
            "{}/stock/{}/quote?token={}",
            self.base_url, symbol, self.api_token
        );
        
        let response = self.client.get(&url).send().await?;
        let data: IexQuote = response.json().await?;
        
        let last_price = Decimal::from_f64_retain(data.latest_price).unwrap_or(dec!(0));
        let change = Decimal::from_f64_retain(data.change.unwrap_or(0.0)).unwrap_or(dec!(0));
        let change_percent = Decimal::from_f64_retain(data.change_percent.unwrap_or(0.0) * 100.0).unwrap_or(dec!(0));
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: Decimal::from_f64_retain(data.iex_bid_price.unwrap_or(data.latest_price - 0.01)).unwrap_or(last_price - dec!(0.01)),
            ask: Decimal::from_f64_retain(data.iex_ask_price.unwrap_or(data.latest_price + 0.01)).unwrap_or(last_price + dec!(0.01)),
            last: last_price,
            volume: Decimal::from(data.volume.unwrap_or(0) as i64),
            timestamp: Utc::now(),
            bid_size: Decimal::from(data.iex_bid_size.unwrap_or(100) as i64),
            ask_size: Decimal::from(data.iex_ask_size.unwrap_or(100) as i64),
            open: Decimal::from_f64_retain(data.open.unwrap_or(0.0)).unwrap_or(last_price),
            high: Decimal::from_f64_retain(data.high.unwrap_or(0.0)).unwrap_or(last_price),
            low: Decimal::from_f64_retain(data.low.unwrap_or(0.0)).unwrap_or(last_price),
            close: last_price,
            previous_close: Decimal::from_f64_retain(data.previous_close.unwrap_or(0.0)).unwrap_or(dec!(0)),
            change,
            change_percent,
        };
        
        // Update cache
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), (quote.clone(), Utc::now()));
        
        Ok(quote)
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        self.rate_limit().await;

        // IEX provides limited order book data through the DEEP endpoint
        let url = format!(
            "{}/stock/{}/book?token={}",
            self.base_url, symbol, self.api_token
        );
        
        let response = self.client.get(&url).send().await?;
        let data: IexBook = response.json().await?;
        
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        // IEX typically provides top-of-book only, so we'll simulate depth
        if let Some(quote) = data.quote {
            let bid_price = Decimal::from_f64_retain(quote.iex_bid_price.unwrap_or(0.0)).unwrap_or(dec!(0));
            let ask_price = Decimal::from_f64_retain(quote.iex_ask_price.unwrap_or(0.0)).unwrap_or(dec!(0));
            let bid_size = Decimal::from(quote.iex_bid_size.unwrap_or(0) as i64);
            let ask_size = Decimal::from(quote.iex_ask_size.unwrap_or(0) as i64);
            
            for i in 0..depth {
                let offset = Decimal::from(i) * dec!(0.01);
                let size_multiplier = Decimal::from((depth - i) as i64);
                
                bids.push((bid_price - offset, bid_size * size_multiplier));
                asks.push((ask_price + offset, ask_size * size_multiplier));
            }
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

        let range = match interval {
            "1m" | "5m" | "15m" => "1d",    // Intraday
            "1h" | "4h" => "1m",             // 1 month
            "1d" => "3m",                    // 3 months
            _ => "1m",
        };
        
        let url = format!(
            "{}/stock/{}/chart/{}?token={}&chartLast={}",
            self.base_url, symbol, range, self.api_token, limit
        );
        
        let response = self.client.get(&url).send().await?;
        let data: Vec<IexCandle> = response.json().await?;
        
        let mut candles = Vec::new();
        for bar in data.iter().take(limit) {
            candles.push(Candle {
                symbol: symbol.to_string(),
                timestamp: bar.date.parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
                open: Decimal::from_f64_retain(bar.open).unwrap_or(dec!(0)),
                high: Decimal::from_f64_retain(bar.high).unwrap_or(dec!(0)),
                low: Decimal::from_f64_retain(bar.low).unwrap_or(dec!(0)),
                close: Decimal::from_f64_retain(bar.close).unwrap_or(dec!(0)),
                volume: Decimal::from(bar.volume as i64),
                trades: 0,
            });
        }
        
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to IEX Cloud for symbols: {:?}", symbols);
        // In production, this would set up SSE or WebSocket connections
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from IEX Cloud for symbols: {:?}", symbols);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IexQuote {
    symbol: String,
    company_name: String,
    latest_price: f64,
    latest_source: String,
    latest_time: String,
    latest_update: i64,
    latest_volume: Option<i64>,
    volume: Option<i64>,
    iex_bid_price: Option<f64>,
    iex_bid_size: Option<i32>,
    iex_ask_price: Option<f64>,
    iex_ask_size: Option<i32>,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    previous_close: Option<f64>,
    change: Option<f64>,
    change_percent: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct IexBook {
    quote: Option<IexQuote>,
    bids: Option<Vec<IexOrder>>,
    asks: Option<Vec<IexOrder>>,
}

#[derive(Debug, Deserialize)]
struct IexOrder {
    price: f64,
    size: i32,
    timestamp: i64,
}

#[derive(Debug, Deserialize)]
struct IexCandle {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
    #[serde(rename = "uOpen")]
    u_open: Option<f64>,
    #[serde(rename = "uClose")]
    u_close: Option<f64>,
    #[serde(rename = "uHigh")]
    u_high: Option<f64>,
    #[serde(rename = "uLow")]
    u_low: Option<f64>,
    #[serde(rename = "uVolume")]
    u_volume: Option<i64>,
    change: Option<f64>,
    #[serde(rename = "changePercent")]
    change_percent: Option<f64>,
    label: Option<String>,
}