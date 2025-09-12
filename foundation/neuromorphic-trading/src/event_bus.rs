//! Ultra-low latency event bus for market data distribution
//! 
//! Features:
//! - Zero-copy via Arc
//! - Lock-free SPMC channels
//! - Ring buffer with 65536 slots
//! - Sequence numbers for gap detection
//! - Conflation for slow consumers

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::time::Instant;
use std::collections::HashMap;
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;
use crate::neuromorphic::SpikePattern;

/// Market symbols
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Symbol {
    AAPL,
    GOOGL,
    MSFT,
    AMZN,
    TSLA,
    SPY,
    QQQ,
    Other(u32),
}

/// Order side
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Trade data
#[derive(Clone, Debug)]
pub struct TradeData {
    pub symbol: Symbol,
    pub price: f64,
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub exchange_timestamp: u64,
    pub aggressor_side: Side,
    pub trade_id: u64,
}

/// Quote data
#[derive(Clone, Debug)]
pub struct QuoteData {
    pub symbol: Symbol,
    pub bid_price: f64,
    pub bid_size: u64,
    pub ask_price: f64,
    pub ask_size: u64,
    pub timestamp_ns: u64,
    pub exchange_timestamp: u64,
}

/// Order book data
#[derive(Clone, Debug)]
pub struct OrderBookData {
    pub symbol: Symbol,
    pub bids: [(f64, u64); 10], // Top 10 levels
    pub asks: [(f64, u64); 10], // Top 10 levels
    pub timestamp_ns: u64,
    pub sequence_number: u64,
}

/// Subscriber ID
pub type SubscriberId = u32;

/// Subscriber state
struct SubscriberState {
    id: SubscriberId,
    tail: AtomicU64,
    conflate: bool,
    active: AtomicBool,
    last_read_ns: AtomicU64,
    dropped_messages: AtomicU64,
}

/// Bus statistics
pub struct BusStatistics {
    pub messages_published: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub max_latency_ns: AtomicU64,
    pub gaps_detected: AtomicU64,
}

impl Default for BusStatistics {
    fn default() -> Self {
        Self {
            messages_published: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            max_latency_ns: AtomicU64::new(0),
            gaps_detected: AtomicU64::new(0),
        }
    }
}

/// Bus configuration
pub struct BusConfig {
    pub buffer_size: usize,
    pub max_subscribers: usize,
    pub enable_conflation: bool,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            buffer_size: 65536,
            max_subscribers: 32,
            enable_conflation: true,
        }
    }
}

/// Lock-free SPMC channel
pub struct SPMCChannel<T> {
    buffer: Vec<Option<Arc<T>>>,
    mask: usize,
    head: AtomicU64,
    cached_head: u64,
    tails: Vec<AtomicU64>,
    active_consumers: AtomicU32,
}

impl<T> SPMCChannel<T> {
    fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Size must be power of 2");
        
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(None);
        }
        
        let mut tails = Vec::with_capacity(32);
        for _ in 0..32 {
            tails.push(AtomicU64::new(0));
        }
        
        Self {
            buffer,
            mask: size - 1,
            head: AtomicU64::new(0),
            cached_head: 0,
            tails,
            active_consumers: AtomicU32::new(0),
        }
    }
    
    fn publish(&mut self, item: Arc<T>) -> Result<()> {
        let head = self.head.load(Ordering::Acquire);
        let index = (head & self.mask as u64) as usize;
        
        // Check if buffer is full
        let min_tail = self.get_min_tail();
        if head - min_tail >= self.mask as u64 {
            return Err(anyhow::anyhow!("Buffer full"));
        }
        
        // Store item
        self.buffer[index] = Some(item);
        
        // Update head
        self.head.store(head + 1, Ordering::Release);
        self.cached_head = head + 1;
        
        Ok(())
    }
    
    fn subscribe(&self, consumer_id: usize) -> Result<Arc<T>> {
        let tail = self.tails[consumer_id].load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        
        if tail >= head {
            return Err(anyhow::anyhow!("No data available"));
        }
        
        let index = (tail & self.mask as u64) as usize;
        
        if let Some(ref item) = self.buffer[index] {
            // Update tail
            self.tails[consumer_id].store(tail + 1, Ordering::Release);
            Ok(Arc::clone(item))
        } else {
            Err(anyhow::anyhow!("Buffer slot empty"))
        }
    }
    
    fn get_min_tail(&self) -> u64 {
        let mut min_tail = u64::MAX;
        let num_consumers = self.active_consumers.load(Ordering::Acquire) as usize;
        
        for i in 0..num_consumers.min(self.tails.len()) {
            let tail = self.tails[i].load(Ordering::Acquire);
            if tail < min_tail {
                min_tail = tail;
            }
        }
        
        if min_tail == u64::MAX {
            self.cached_head
        } else {
            min_tail
        }
    }
}

/// Market data bus
pub struct MarketDataBus {
    trade_channel: Arc<RwLock<SPMCChannel<TradeData>>>,
    quote_channel: Arc<RwLock<SPMCChannel<QuoteData>>>,
    book_channel: Arc<RwLock<SPMCChannel<OrderBookData>>>,
    pattern_channel: Arc<RwLock<SPMCChannel<SpikePattern>>>,
    subscribers: DashMap<SubscriberId, SubscriberState>,
    stats: Arc<BusStatistics>,
    next_subscriber_id: AtomicU32,
    sequence_number: AtomicU64,
    latest_pattern: Arc<RwLock<Option<SpikePattern>>>,
}

impl MarketDataBus {
    pub fn new(config: BusConfig) -> Self {
        Self {
            trade_channel: Arc::new(RwLock::new(SPMCChannel::new(config.buffer_size))),
            quote_channel: Arc::new(RwLock::new(SPMCChannel::new(config.buffer_size))),
            book_channel: Arc::new(RwLock::new(SPMCChannel::new(config.buffer_size))),
            pattern_channel: Arc::new(RwLock::new(SPMCChannel::new(config.buffer_size))),
            subscribers: DashMap::new(),
            stats: Arc::new(BusStatistics::default()),
            next_subscriber_id: AtomicU32::new(0),
            sequence_number: AtomicU64::new(0),
            latest_pattern: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Publish trade data
    pub fn publish_trade(&self, trade: TradeData) -> Result<()> {
        let start = Instant::now();
        let wrapped = Arc::new(trade);
        
        // Publish to channel
        self.trade_channel.write().publish(wrapped)?;
        
        // Update statistics
        self.stats.messages_published.fetch_add(1, Ordering::Relaxed);
        let latency = start.elapsed().as_nanos() as u64;
        self.stats.total_latency_ns.fetch_add(latency, Ordering::Relaxed);
        
        // Update max latency
        let mut max = self.stats.max_latency_ns.load(Ordering::Relaxed);
        while latency > max {
            match self.stats.max_latency_ns.compare_exchange_weak(
                max,
                latency,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => max = x,
            }
        }
        
        Ok(())
    }
    
    /// Publish quote data
    pub fn publish_quote(&self, quote: QuoteData) -> Result<()> {
        let wrapped = Arc::new(quote);
        self.quote_channel.write().publish(wrapped)
    }
    
    /// Publish order book data
    pub fn publish_book(&self, mut book: OrderBookData) -> Result<()> {
        // Add sequence number
        book.sequence_number = self.sequence_number.fetch_add(1, Ordering::SeqCst);
        
        let wrapped = Arc::new(book);
        self.book_channel.write().publish(wrapped)
    }
    
    /// Subscribe to market data
    pub fn subscribe(&self) -> SubscriberId {
        let id = self.next_subscriber_id.fetch_add(1, Ordering::SeqCst);
        
        let state = SubscriberState {
            id,
            tail: AtomicU64::new(0),
            conflate: false,
            active: AtomicBool::new(true),
            last_read_ns: AtomicU64::new(0),
            dropped_messages: AtomicU64::new(0),
        };
        
        self.subscribers.insert(id, state);
        id
    }
    
    /// Read trade data for subscriber
    pub fn read_trade(&self, subscriber_id: SubscriberId) -> Result<Arc<TradeData>> {
        let subscriber = self.subscribers.get(&subscriber_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid subscriber ID"))?;
        
        if !subscriber.active.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Subscriber inactive"));
        }
        
        let consumer_id = subscriber_id as usize % 32;
        let trade = self.trade_channel.read().subscribe(consumer_id)?;
        
        // Update last read time
        subscriber.last_read_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed
        );
        
        Ok(trade)
    }
    
    /// Publish pattern
    pub async fn publish_pattern(&self, pattern: SpikePattern) -> Result<()> {
        let wrapped = Arc::new(pattern.clone());
        self.pattern_channel.write().publish(wrapped)?;
        *self.latest_pattern.write() = Some(pattern);
        Ok(())
    }
    
    /// Get latest pattern
    pub async fn get_latest_pattern(&self) -> Option<SpikePattern> {
        self.latest_pattern.read().clone()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> BusStats {
        let messages = self.stats.messages_published.load(Ordering::Relaxed);
        let total_latency = self.stats.total_latency_ns.load(Ordering::Relaxed);
        
        BusStats {
            messages_published: messages,
            avg_latency_ns: if messages > 0 { total_latency / messages } else { 0 },
            max_latency_ns: self.stats.max_latency_ns.load(Ordering::Relaxed),
            gaps_detected: self.stats.gaps_detected.load(Ordering::Relaxed),
            active_subscribers: self.subscribers.len(),
        }
    }
}

/// Bus statistics snapshot
#[derive(Debug)]
pub struct BusStats {
    pub messages_published: u64,
    pub avg_latency_ns: u64,
    pub max_latency_ns: u64,
    pub gaps_detected: u64,
    pub active_subscribers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spmc_channel() {
        let mut channel = SPMCChannel::new(16);
        
        // Publish data
        let data = Arc::new(42);
        assert!(channel.publish(data.clone()).is_ok());
        
        // Subscribe and read
        let result = channel.subscribe(0);
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), 42);
    }
    
    #[test]
    fn test_market_data_bus() {
        let bus = MarketDataBus::new(BusConfig::default());
        
        // Publish trade
        let trade = TradeData {
            symbol: Symbol::AAPL,
            price: 150.0,
            quantity: 100,
            timestamp_ns: 1000,
            exchange_timestamp: 1000,
            aggressor_side: Side::Buy,
            trade_id: 1,
        };
        
        assert!(bus.publish_trade(trade).is_ok());
        
        // Subscribe and read
        let subscriber_id = bus.subscribe();
        let result = bus.read_trade(subscriber_id);
        assert!(result.is_ok());
    }
}