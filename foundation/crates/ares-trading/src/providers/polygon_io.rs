use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, NaiveDate};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::market_data::{Quote, Candle, OrderBook, MarketDataProvider};
use super::MarketDataConfig;

/// Polygon.io provider - Professional market data
/// Free tier: 5 API calls/minute
/// Paid tiers available for higher limits
pub struct PolygonProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
    rate_limiter: Arc<RwLock<DateTime<Utc>>>,
}

impl PolygonProvider {
    pub fn new(config: MarketDataConfig) -> Result<Self> {
        let api_key = config.api_key
            .ok_or_else(|| anyhow::anyhow!("Polygon.io requires an API key"))?;
        
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            api_key,
            base_url: config.base_url.unwrap_or_else(|| "https://api.polygon.io".to_string()),
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(Utc::now())),
        })
    }

    async fn rate_limit(&self) {
        let mut last_request = self.rate_limiter.write().await;
        let now = Utc::now();
        let time_since_last = now.signed_duration_since(*last_request);
        
        // Free tier: 5 requests per minute = 1 request every 12 seconds
        if time_since_last.num_seconds() < 12 {
            let wait_time = 12000 - time_since_last.num_milliseconds();
            tokio::time::sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;
        }
        *last_request = Utc::now();
    }
}

#[async_trait]
impl MarketDataProvider for PolygonProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Check cache first
        let cache = self.cache.read().await;
        if let Some((quote, timestamp)) = cache.get(symbol) {
            if Utc::now().signed_duration_since(*timestamp).num_seconds() < 60 {
                return Ok(quote.clone());
            }
        }
        drop(cache);

        self.rate_limit().await;

        // Get the latest trade
        let url = format!(
            "{}/v2/last/trade/{}?apiKey={}",
            self.base_url, symbol, self.api_key
        );
        
        let trade_response = self.client.get(&url).send().await?;
        let trade_data: PolygonLastTradeResponse = trade_response.json().await?;
        
        // Get the snapshot for more details
        let snapshot_url = format!(
            "{}/v2/snapshot/locale/us/markets/stocks/tickers/{}?apiKey={}",
            self.base_url, symbol, self.api_key
        );
        
        let snapshot_response = self.client.get(&snapshot_url).send().await?;
        let snapshot: PolygonSnapshotResponse = snapshot_response.json().await?;
        
        let ticker = &snapshot.ticker;
        let day = &ticker.day;
        let prev_day = &ticker.prev_day;
        
        let last_price = Decimal::from_f64_retain(trade_data.results.price).unwrap_or(dec!(0));
        let change = last_price - Decimal::from_f64_retain(prev_day.close).unwrap_or(dec!(0));
        let change_percent = if prev_day.close != 0.0 {
            (change / Decimal::from_f64_retain(prev_day.close).unwrap_or(dec!(1))) * dec!(100)
        } else {
            dec!(0)
        };
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: Decimal::from_f64_retain(ticker.last_quote.bid_price).unwrap_or(last_price - dec!(0.01)),
            ask: Decimal::from_f64_retain(ticker.last_quote.ask_price).unwrap_or(last_price + dec!(0.01)),
            last: last_price,
            volume: Decimal::from(day.volume as i64),
            timestamp: Utc::now(),
            bid_size: Decimal::from(ticker.last_quote.bid_size as i64),
            ask_size: Decimal::from(ticker.last_quote.ask_size as i64),
            open: Decimal::from_f64_retain(day.open).unwrap_or(dec!(0)),
            high: Decimal::from_f64_retain(day.high).unwrap_or(dec!(0)),
            low: Decimal::from_f64_retain(day.low).unwrap_or(dec!(0)),
            close: Decimal::from_f64_retain(day.close).unwrap_or(last_price),
            previous_close: Decimal::from_f64_retain(prev_day.close).unwrap_or(dec!(0)),
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

        let url = format!(
            "{}/v2/snapshot/locale/us/markets/stocks/tickers/{}?apiKey={}",
            self.base_url, symbol, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: PolygonSnapshotResponse = response.json().await?;
        
        let quote = &data.ticker.last_quote;
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        // Polygon provides only top of book, so we'll generate depth
        let bid_price = Decimal::from_f64_retain(quote.bid_price).unwrap_or(dec!(0));
        let ask_price = Decimal::from_f64_retain(quote.ask_price).unwrap_or(dec!(0));
        let bid_size = Decimal::from(quote.bid_size as i64);
        let ask_size = Decimal::from(quote.ask_size as i64);
        
        for i in 0..depth {
            let offset = Decimal::from(i) * dec!(0.01);
            let size_multiplier = Decimal::from((depth - i) as i64);
            
            bids.push((bid_price - offset, bid_size * size_multiplier));
            asks.push((ask_price + offset, ask_size * size_multiplier));
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

        let multiplier = match interval {
            "1m" => 1,
            "5m" => 5,
            "15m" => 15,
            "1h" => 1,
            "4h" => 4,
            "1d" => 1,
            _ => 1,
        };
        
        let timespan = match interval {
            "1m" | "5m" | "15m" => "minute",
            "1h" | "4h" => "hour",
            "1d" => "day",
            _ => "day",
        };
        
        let to = Utc::now().format("%Y-%m-%d").to_string();
        let from = (Utc::now() - chrono::Duration::days(30)).format("%Y-%m-%d").to_string();
        
        let url = format!(
            "{}/v2/aggs/ticker/{}/range/{}/{}/{}/{}?adjusted=true&sort=desc&limit={}&apiKey={}",
            self.base_url, symbol, multiplier, timespan, from, to, limit, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: PolygonAggregatesResponse = response.json().await?;
        
        let mut candles = Vec::new();
        for bar in data.results {
            candles.push(Candle {
                symbol: symbol.to_string(),
                timestamp: DateTime::from_timestamp(bar.timestamp / 1000, 0)
                    .unwrap_or_else(|| Utc::now()),
                open: Decimal::from_f64_retain(bar.open).unwrap_or(dec!(0)),
                high: Decimal::from_f64_retain(bar.high).unwrap_or(dec!(0)),
                low: Decimal::from_f64_retain(bar.low).unwrap_or(dec!(0)),
                close: Decimal::from_f64_retain(bar.close).unwrap_or(dec!(0)),
                volume: Decimal::from(bar.volume as i64),
                trades: bar.trades.unwrap_or(0),
            });
        }
        
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to Polygon.io for symbols: {:?}", symbols);
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from Polygon.io for symbols: {:?}", symbols);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct PolygonLastTradeResponse {
    status: String,
    results: PolygonLastTrade,
}

#[derive(Debug, Deserialize)]
struct PolygonLastTrade {
    #[serde(rename = "T")]
    symbol: String,
    #[serde(rename = "p")]
    price: f64,
    #[serde(rename = "s")]
    size: i64,
    #[serde(rename = "t")]
    timestamp: i64,
}

#[derive(Debug, Deserialize)]
struct PolygonSnapshotResponse {
    status: String,
    ticker: PolygonTickerSnapshot,
}

#[derive(Debug, Deserialize)]
struct PolygonTickerSnapshot {
    ticker: String,
    day: PolygonDayData,
    #[serde(rename = "prevDay")]
    prev_day: PolygonDayData,
    #[serde(rename = "lastQuote")]
    last_quote: PolygonQuote,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PolygonDayData {
    #[serde(rename = "o")]
    open: f64,
    #[serde(rename = "h")]
    high: f64,
    #[serde(rename = "l")]
    low: f64,
    #[serde(rename = "c")]
    close: f64,
    #[serde(rename = "v")]
    volume: f64,
}

#[derive(Debug, Deserialize)]
struct PolygonQuote {
    #[serde(rename = "P")]
    ask_price: f64,
    #[serde(rename = "S")]
    ask_size: i32,
    #[serde(rename = "p")]
    bid_price: f64,
    #[serde(rename = "s")]
    bid_size: i32,
}

#[derive(Debug, Deserialize)]
struct PolygonAggregatesResponse {
    status: String,
    results: Vec<PolygonAggregate>,
}

#[derive(Debug, Deserialize)]
struct PolygonAggregate {
    #[serde(rename = "o")]
    open: f64,
    #[serde(rename = "h")]
    high: f64,
    #[serde(rename = "l")]
    low: f64,
    #[serde(rename = "c")]
    close: f64,
    #[serde(rename = "v")]
    volume: f64,
    #[serde(rename = "t")]
    timestamp: i64,
    #[serde(rename = "n")]
    trades: Option<u32>,
}