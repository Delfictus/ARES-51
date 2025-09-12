use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub last: Decimal,
    pub volume: Decimal,
    pub timestamp: DateTime<Utc>,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub previous_close: Decimal,
    pub change: Decimal,
    pub change_percent: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub trades: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<(Decimal, Decimal)>, // (price, size)
    pub asks: Vec<(Decimal, Decimal)>, // (price, size)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub size: Decimal,
    pub side: TradeSide,
    pub trade_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    Quote(Quote),
    Trade(Trade),
    OrderBook(OrderBook),
    Candle(Candle),
}

#[async_trait::async_trait]
pub trait MarketDataProvider: Send + Sync {
    async fn get_quote(&self, symbol: &str) -> Result<Quote>;
    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook>;
    async fn get_candles(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>>;
    async fn subscribe(&self, symbols: Vec<String>) -> Result<()>;
    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()>;
}

// Simulated market data provider for paper trading
pub struct SimulatedMarketDataProvider {
    quotes: Arc<RwLock<HashMap<String, Quote>>>,
    order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
    subscriptions: Arc<RwLock<Vec<String>>>,
    base_prices: HashMap<String, Decimal>,
}

impl SimulatedMarketDataProvider {
    pub fn new() -> Self {
        let mut base_prices = HashMap::new();
        // Initialize with some common stock prices
        base_prices.insert("AAPL".to_string(), rust_decimal_macros::dec!(175.50));
        base_prices.insert("GOOGL".to_string(), rust_decimal_macros::dec!(140.25));
        base_prices.insert("MSFT".to_string(), rust_decimal_macros::dec!(370.80));
        base_prices.insert("AMZN".to_string(), rust_decimal_macros::dec!(130.60));
        base_prices.insert("TSLA".to_string(), rust_decimal_macros::dec!(245.30));
        base_prices.insert("META".to_string(), rust_decimal_macros::dec!(310.45));
        base_prices.insert("NVDA".to_string(), rust_decimal_macros::dec!(450.20));
        base_prices.insert("SPY".to_string(), rust_decimal_macros::dec!(440.50));
        base_prices.insert("QQQ".to_string(), rust_decimal_macros::dec!(365.25));
        base_prices.insert("BTC-USD".to_string(), rust_decimal_macros::dec!(42500.00));
        base_prices.insert("ETH-USD".to_string(), rust_decimal_macros::dec!(2250.00));

        Self {
            quotes: Arc::new(RwLock::new(HashMap::new())),
            order_books: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            base_prices,
        }
    }

    pub async fn start_simulation(&self) {
        let quotes = self.quotes.clone();
        let base_prices = self.base_prices.clone();
        let subscriptions = self.subscriptions.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
            
            loop {
                interval.tick().await;
                
                let subs = subscriptions.read().await.clone();
                for symbol in subs {
                    if let Some(base_price) = base_prices.get(&symbol) {
                        let quote = generate_random_quote(&symbol, *base_price);
                        quotes.write().await.insert(symbol.clone(), quote);
                    }
                }
            }
        });
    }

    pub async fn get_latest_quotes(&self) -> HashMap<String, Quote> {
        self.quotes.read().await.clone()
    }
}

fn generate_random_quote(symbol: &str, base_price: Decimal) -> Quote {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate random walk with mean reversion
    let volatility = rust_decimal_macros::dec!(0.002); // 0.2% volatility
    let change_factor = Decimal::from_f64_retain(
        1.0 + rng.gen_range(-0.002..0.002)
    ).unwrap_or(rust_decimal_macros::dec!(1));
    
    let last = base_price * change_factor;
    let spread = last * rust_decimal_macros::dec!(0.0001); // 0.01% spread
    let bid = last - spread;
    let ask = last + spread;
    
    let volume = Decimal::from(rng.gen_range(1000000..10000000));
    let bid_size = Decimal::from(rng.gen_range(100..10000));
    let ask_size = Decimal::from(rng.gen_range(100..10000));
    
    Quote {
        symbol: symbol.to_string(),
        bid,
        ask,
        last,
        volume,
        timestamp: Utc::now(),
        bid_size,
        ask_size,
        open: base_price,
        high: last.max(base_price),
        low: last.min(base_price),
        close: last,
        previous_close: base_price,
        change: last - base_price,
        change_percent: ((last - base_price) / base_price) * rust_decimal_macros::dec!(100),
    }
}

#[async_trait::async_trait]
impl MarketDataProvider for SimulatedMarketDataProvider {
    async fn get_quote(&self, symbol: &str) -> Result<Quote> {
        let quotes = self.quotes.read().await;
        quotes.get(symbol)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Quote not found for symbol: {}", symbol))
    }

    async fn get_order_book(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        let quote = self.get_quote(symbol).await?;
        
        // Generate simulated order book
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        for i in 0..depth {
            let price_offset = Decimal::from(i) * rust_decimal_macros::dec!(0.01);
            let size = Decimal::from(rand::random::<u32>() % 10000 + 100);
            
            bids.push((quote.bid - price_offset, size));
            asks.push((quote.ask + price_offset, size));
        }
        
        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids,
            asks,
        })
    }

    async fn get_candles(&self, symbol: &str, _interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let quote = self.get_quote(symbol).await?;
        let mut candles = Vec::new();
        
        // Generate historical candles
        let mut current_time = Utc::now() - chrono::Duration::minutes(limit as i64);
        let mut current_price = quote.last;
        
        for _ in 0..limit {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            let open = current_price;
            let change = Decimal::from_f64_retain(rng.gen_range(-0.01..0.01))
                .unwrap_or(rust_decimal_macros::dec!(0));
            let close = open * (rust_decimal_macros::dec!(1) + change);
            let high = open.max(close) * rust_decimal_macros::dec!(1.001);
            let low = open.min(close) * rust_decimal_macros::dec!(0.999);
            let volume = Decimal::from(rng.gen_range(100000..1000000));
            
            candles.push(Candle {
                symbol: symbol.to_string(),
                timestamp: current_time,
                open,
                high,
                low,
                close,
                volume,
                trades: rng.gen_range(100..1000),
            });
            
            current_price = close;
            current_time = current_time + chrono::Duration::minutes(1);
        }
        
        Ok(candles)
    }

    async fn subscribe(&self, symbols: Vec<String>) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        for symbol in symbols {
            if !subs.contains(&symbol) {
                info!("Subscribed to {}", symbol);
                subs.push(symbol);
            }
        }
        Ok(())
    }

    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        subs.retain(|s| !symbols.contains(s));
        for symbol in symbols {
            info!("Unsubscribed from {}", symbol);
        }
        Ok(())
    }
}

// Alpha Vantage provider for real market data (free tier)
pub struct AlphaVantageProvider {
    api_key: String,
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, (Quote, DateTime<Utc>)>>>,
}

impl AlphaVantageProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn fetch_quote(&self, symbol: &str) -> Result<Quote> {
        let url = format!(
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={}&apikey={}",
            symbol, self.api_key
        );
        
        let response = self.client.get(&url).send().await?;
        let data: serde_json::Value = response.json().await?;
        
        if let Some(quote_data) = data.get("Global Quote") {
            Ok(parse_alpha_vantage_quote(symbol, quote_data)?)
        } else {
            Err(anyhow::anyhow!("Invalid response from Alpha Vantage"))
        }
    }
}

fn parse_alpha_vantage_quote(symbol: &str, data: &serde_json::Value) -> Result<Quote> {
    let parse_decimal = |field: &str| -> Result<Decimal> {
        data.get(field)
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .ok_or_else(|| anyhow::anyhow!("Failed to parse field: {}", field))
    };
    
    let price = parse_decimal("05. price")?;
    let volume = parse_decimal("06. volume")?;
    let previous_close = parse_decimal("08. previous close")?;
    let change = parse_decimal("09. change")?;
    let change_percent_str = data.get("10. change percent")
        .and_then(|v| v.as_str())
        .unwrap_or("0%");
    let change_percent = change_percent_str.trim_end_matches('%')
        .parse::<Decimal>()
        .unwrap_or(rust_decimal_macros::dec!(0));
    
    Ok(Quote {
        symbol: symbol.to_string(),
        bid: price - rust_decimal_macros::dec!(0.01),
        ask: price + rust_decimal_macros::dec!(0.01),
        last: price,
        volume,
        timestamp: Utc::now(),
        bid_size: rust_decimal_macros::dec!(100),
        ask_size: rust_decimal_macros::dec!(100),
        open: parse_decimal("02. open").unwrap_or(price),
        high: parse_decimal("03. high").unwrap_or(price),
        low: parse_decimal("04. low").unwrap_or(price),
        close: price,
        previous_close,
        change,
        change_percent,
    })
}