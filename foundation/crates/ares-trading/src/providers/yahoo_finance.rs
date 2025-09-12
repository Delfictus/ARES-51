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

/// Yahoo Finance provider - FREE, no API key required
pub struct YahooFinanceProvider {
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
    rate_limiter: Arc<RwLock<DateTime<Utc>>>,
}

impl YahooFinanceProvider {
    pub fn new(_config: MarketDataConfig) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            cache: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(Utc::now())),
        })
    }

    async fn fetch_quote_from_yahoo(&self, symbol: &str) -> Result<YahooQuoteResponse> {
        // Rate limiting: 1 request per second
        let mut last_request = self.rate_limiter.write().await;
        let now = Utc::now();
        let time_since_last = now.signed_duration_since(*last_request);
        if time_since_last.num_milliseconds() < 1000 {
            tokio::time::sleep(tokio::time::Duration::from_millis(
                (1000 - time_since_last.num_milliseconds()) as u64
            )).await;
        }
        *last_request = Utc::now();
        drop(last_request);

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}",
            symbol
        );
        
        let response = self.client.get(&url).send().await?;
        let data: YahooChartResponse = response.json().await?;
        
        if let Some(result) = data.chart.result.get(0) {
            Ok(result.clone())
        } else {
            Err(anyhow::anyhow!("No data returned for symbol: {}", symbol))
        }
    }

    async fn fetch_historical(&self, symbol: &str, period: &str) -> Result<Vec<Candle>> {
        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval=1d",
            symbol, period
        );
        
        let response = self.client.get(&url).send().await?;
        let data: YahooChartResponse = response.json().await?;
        
        if let Some(result) = data.chart.result.get(0) {
            if let (Some(timestamps), Some(indicators)) = (&result.timestamp, &result.indicators) {
                if let Some(quote) = &indicators.quote.get(0) {
                    let mut candles = Vec::new();
                    
                    for i in 0..timestamps.len() {
                        if let (Some(open), Some(high), Some(low), Some(close), Some(volume)) = (
                            quote.open.as_ref().and_then(|o| o.get(i)),
                            quote.high.as_ref().and_then(|h| h.get(i)),
                            quote.low.as_ref().and_then(|l| l.get(i)),
                            quote.close.as_ref().and_then(|c| c.get(i)),
                            quote.volume.as_ref().and_then(|v| v.get(i)),
                        ) {
                            candles.push(Candle {
                                symbol: symbol.to_string(),
                                timestamp: DateTime::from_timestamp(timestamps[i], 0)
                                    .unwrap_or_else(|| Utc::now()),
                                open: Decimal::from_f64_retain(*open).unwrap_or(dec!(0)),
                                high: Decimal::from_f64_retain(*high).unwrap_or(dec!(0)),
                                low: Decimal::from_f64_retain(*low).unwrap_or(dec!(0)),
                                close: Decimal::from_f64_retain(*close).unwrap_or(dec!(0)),
                                volume: Decimal::from(*volume as i64),
                                trades: 0,
                            });
                        }
                    }
                    
                    return Ok(candles);
                }
            }
        }
        
        Err(anyhow::anyhow!("Failed to parse historical data"))
    }
}

#[async_trait]
impl MarketDataProvider for YahooFinanceProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Check cache first (1 minute TTL)
        let cache = self.cache.read().await;
        if let Some((quote, timestamp)) = cache.get(symbol) {
            if Utc::now().signed_duration_since(*timestamp).num_seconds() < 60 {
                return Ok(quote.clone());
            }
        }
        drop(cache);

        let yahoo_data = self.fetch_quote_from_yahoo(symbol).await?;
        let meta = &yahoo_data.meta;
        
        let price = Decimal::from_f64_retain(meta.regular_market_price).unwrap_or(dec!(0));
        let previous_close = Decimal::from_f64_retain(meta.previous_close.unwrap_or(0.0)).unwrap_or(dec!(0));
        let change = price - previous_close;
        let change_percent = if previous_close != dec!(0) {
            (change / previous_close) * dec!(100)
        } else {
            dec!(0)
        };
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: price - dec!(0.01), // Approximate bid
            ask: price + dec!(0.01), // Approximate ask
            last: price,
            volume: Decimal::from(meta.regular_market_volume.unwrap_or(0) as i64),
            timestamp: Utc::now(),
            bid_size: dec!(100),
            ask_size: dec!(100),
            open: Decimal::from_f64_retain(meta.regular_market_open.unwrap_or(0.0)).unwrap_or(price),
            high: Decimal::from_f64_retain(meta.regular_market_day_high.unwrap_or(0.0)).unwrap_or(price),
            low: Decimal::from_f64_retain(meta.regular_market_day_low.unwrap_or(0.0)).unwrap_or(price),
            close: price,
            previous_close,
            change,
            change_percent,
        };
        
        // Update cache
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), (quote.clone(), Utc::now()));
        
        Ok(quote)
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        // Yahoo doesn't provide order book data, so we'll simulate it based on the quote
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
        let period = match interval {
            "1m" | "5m" | "15m" => "1d",  // Last day for minute intervals
            "1h" | "4h" => "5d",           // Last 5 days for hourly
            "1d" => "1mo",                 // Last month for daily
            _ => "1mo",
        };
        
        let candles = self.fetch_historical(symbol, period).await?;
        Ok(candles.into_iter().rev().take(limit).rev().collect())
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to Yahoo Finance for symbols: {:?}", symbols);
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from Yahoo Finance for symbols: {:?}", symbols);
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct YahooChartResponse {
    chart: YahooChart,
}

#[derive(Debug, Clone, Deserialize)]
struct YahooChart {
    result: Vec<YahooQuoteResponse>,
}

#[derive(Debug, Clone, Deserialize)]
struct YahooQuoteResponse {
    meta: YahooMeta,
    timestamp: Option<Vec<i64>>,
    indicators: Option<YahooIndicators>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YahooMeta {
    currency: Option<String>,
    symbol: String,
    regular_market_price: f64,
    regular_market_volume: Option<i64>,
    regular_market_open: Option<f64>,
    regular_market_day_high: Option<f64>,
    regular_market_day_low: Option<f64>,
    previous_close: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuoteData>,
}

#[derive(Debug, Clone, Deserialize)]
struct YahooQuoteData {
    open: Option<Vec<f64>>,
    high: Option<Vec<f64>>,
    low: Option<Vec<f64>>,
    close: Option<Vec<f64>>,
    volume: Option<Vec<i64>>,
}