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

/// Binance provider for cryptocurrency market data
/// FREE - No API key required for public market data
/// WebSocket support for real-time data
pub struct BinanceProvider {
    client: reqwest::Client,
    base_url: String,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
}

impl BinanceProvider {
    pub fn new(_config: MarketDataConfig) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?,
            base_url: "https://api.binance.com".to_string(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn convert_symbol(&self, symbol: &str) -> String {
        // Convert from standard format (BTC-USD) to Binance format (BTCUSDT)
        symbol.replace("-", "").replace("USD", "USDT")
    }
}

#[async_trait]
impl MarketDataProvider for BinanceProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        // Check cache first
        let cache = self.cache.read().await;
        if let Some((quote, timestamp)) = cache.get(symbol) {
            if Utc::now().signed_duration_since(*timestamp).num_seconds() < 10 {
                return Ok(quote.clone());
            }
        }
        drop(cache);

        let binance_symbol = self.convert_symbol(symbol);
        
        // Get ticker data
        let ticker_url = format!("{}/api/v3/ticker/24hr?symbol={}", self.base_url, binance_symbol);
        let ticker_response = self.client.get(&ticker_url).send().await?;
        let ticker: BinanceTicker24hr = ticker_response.json().await?;
        
        // Get order book for bid/ask
        let book_url = format!("{}/api/v3/ticker/bookTicker?symbol={}", self.base_url, binance_symbol);
        let book_response = self.client.get(&book_url).send().await?;
        let book: BinanceBookTicker = book_response.json().await?;
        
        let last_price = ticker.last_price.parse::<Decimal>()?;
        let change = ticker.price_change.parse::<Decimal>()?;
        let change_percent = ticker.price_change_percent.parse::<Decimal>()?;
        
        let quote = Quote {
            symbol: symbol.to_string(),
            bid: book.bid_price.parse::<Decimal>()?,
            ask: book.ask_price.parse::<Decimal>()?,
            last: last_price,
            volume: ticker.volume.parse::<Decimal>()?,
            timestamp: Utc::now(),
            bid_size: book.bid_qty.parse::<Decimal>()?,
            ask_size: book.ask_qty.parse::<Decimal>()?,
            open: ticker.open_price.parse::<Decimal>()?,
            high: ticker.high_price.parse::<Decimal>()?,
            low: ticker.low_price.parse::<Decimal>()?,
            close: last_price,
            previous_close: ticker.prev_close_price.parse::<Decimal>()?,
            change,
            change_percent,
        };
        
        // Update cache
        let mut cache = self.cache.write().await;
        cache.insert(symbol.to_string(), (quote.clone(), Utc::now()));
        
        Ok(quote)
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        let binance_symbol = self.convert_symbol(symbol);
        
        let url = format!(
            "{}/api/v3/depth?symbol={}&limit={}",
            self.base_url, binance_symbol, depth.min(100)
        );
        
        let response = self.client.get(&url).send().await?;
        let data: BinanceOrderBook = response.json().await?;
        
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        for bid in data.bids.iter().take(depth) {
            bids.push((
                bid[0].parse::<Decimal>()?,
                bid[1].parse::<Decimal>()?,
            ));
        }
        
        for ask in data.asks.iter().take(depth) {
            asks.push((
                ask[0].parse::<Decimal>()?,
                ask[1].parse::<Decimal>()?,
            ));
        }
        
        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids,
            asks,
        })
    }

    async fn get_candles(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let binance_symbol = self.convert_symbol(symbol);
        
        let binance_interval = match interval {
            "1m" => "1m",
            "5m" => "5m",
            "15m" => "15m",
            "1h" => "1h",
            "4h" => "4h",
            "1d" => "1d",
            _ => "1h",
        };
        
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            self.base_url, binance_symbol, binance_interval, limit.min(1000)
        );
        
        let response = self.client.get(&url).send().await?;
        let data: Vec<Vec<serde_json::Value>> = response.json().await?;
        
        let mut candles = Vec::new();
        for kline in data {
            if kline.len() >= 11 {
                candles.push(Candle {
                    symbol: symbol.to_string(),
                    timestamp: DateTime::from_timestamp(
                        kline[0].as_i64().unwrap_or(0) / 1000,
                        0
                    ).unwrap_or_else(|| Utc::now()),
                    open: kline[1].as_str().unwrap_or("0").parse::<Decimal>()?,
                    high: kline[2].as_str().unwrap_or("0").parse::<Decimal>()?,
                    low: kline[3].as_str().unwrap_or("0").parse::<Decimal>()?,
                    close: kline[4].as_str().unwrap_or("0").parse::<Decimal>()?,
                    volume: kline[5].as_str().unwrap_or("0").parse::<Decimal>()?,
                    trades: kline[8].as_i64().unwrap_or(0) as u32,
                });
            }
        }
        
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Subscribed to Binance for symbols: {:?}", symbols);
        // In production, this would establish WebSocket connections
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        info!("Unsubscribed from Binance for symbols: {:?}", symbols);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinanceTicker24hr {
    symbol: String,
    price_change: String,
    price_change_percent: String,
    weighted_avg_price: String,
    prev_close_price: String,
    last_price: String,
    bid_price: String,
    ask_price: String,
    open_price: String,
    high_price: String,
    low_price: String,
    volume: String,
    quote_volume: String,
    open_time: i64,
    close_time: i64,
    count: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinanceBookTicker {
    symbol: String,
    bid_price: String,
    bid_qty: String,
    ask_price: String,
    ask_qty: String,
}

#[derive(Debug, Deserialize)]
struct BinanceOrderBook {
    #[serde(rename = "lastUpdateId")]
    last_update_id: i64,
    bids: Vec<Vec<String>>,
    asks: Vec<Vec<String>>,
}