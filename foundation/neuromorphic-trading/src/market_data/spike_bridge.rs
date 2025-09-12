//! Bridge between market data and spike encoding

use super::unified_feed::{UnifiedMarketEvent, UnifiedMarketFeed};
use crate::spike_encoding::{SpikeEncoder, QuoteData, OrderBookData, Spike};
use crate::market_state::{MarketState, MarketStateTracker};
use crate::exchanges::{Symbol, Exchange};
use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use std::time::{Duration, Instant};

/// Spike generation statistics
#[derive(Default, Clone, Debug)]
pub struct SpikeStatistics {
    pub events_processed: u64,
    pub spikes_generated: u64,
    pub encoding_latency_us: f64,
    pub last_spike_time: Option<Instant>,
    pub spike_rate_hz: f64,
}

/// Configuration for spike bridge
pub struct SpikeBridgeConfig {
    pub neuron_count: usize,
    pub spike_buffer_size: usize,
    pub batch_size: usize,
    pub encoding_timeout: Duration,
    pub enable_adaptive_encoding: bool,
}

impl Default for SpikeBridgeConfig {
    fn default() -> Self {
        Self {
            neuron_count: 10000,
            spike_buffer_size: 100000,
            batch_size: 100,
            encoding_timeout: Duration::from_millis(10),
            enable_adaptive_encoding: true,
        }
    }
}

/// Bridge between market data and spike encoding
pub struct MarketDataSpikeBridge {
    encoders: DashMap<Symbol, Arc<SpikeEncoder>>,
    state_trackers: DashMap<Symbol, Arc<MarketStateTracker>>,
    spike_sender: mpsc::UnboundedSender<Vec<Spike>>,
    spike_receiver: Option<mpsc::UnboundedReceiver<Vec<Spike>>>,
    statistics: DashMap<Symbol, SpikeStatistics>,
    config: SpikeBridgeConfig,
    spike_buffer: DashMap<Symbol, Vec<Spike>>,
}

impl MarketDataSpikeBridge {
    pub fn new(config: SpikeBridgeConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        Self {
            encoders: DashMap::new(),
            state_trackers: DashMap::new(),
            spike_sender: tx,
            spike_receiver: Some(rx),
            statistics: DashMap::new(),
            config,
            spike_buffer: DashMap::new(),
        }
    }
    
    /// Initialize encoder for a symbol
    pub fn initialize_symbol(&self, symbol: Symbol) {
        if !self.encoders.contains_key(&symbol) {
            let mut encoder = SpikeEncoder::new(self.config.neuron_count);
            
            // Configure encoder based on symbol characteristics
            if symbol.as_str().contains("BTC") || symbol.as_str().contains("ETH") {
                encoder.set_sensitivity(1.5); // Higher sensitivity for major pairs
            }
            
            self.encoders.insert(symbol.clone(), Arc::new(encoder));
            self.state_trackers.insert(
                symbol.clone(), 
                Arc::new(MarketStateTracker::new())
            );
            self.statistics.insert(symbol.clone(), SpikeStatistics::default());
            self.spike_buffer.insert(symbol, Vec::new());
        }
    }
    
    /// Process unified market event
    pub async fn process_event(&self, event: UnifiedMarketEvent) -> Result<()> {
        let start = Instant::now();
        
        match event {
            UnifiedMarketEvent::Trade(trade) => {
                self.process_trade(trade).await?;
            }
            UnifiedMarketEvent::Quote(quote) => {
                self.process_quote(quote).await?;
            }
            UnifiedMarketEvent::OrderBook(book) => {
                self.process_orderbook(book).await?;
            }
            UnifiedMarketEvent::Heartbeat { .. } => {
                // Heartbeats don't generate spikes
            }
            UnifiedMarketEvent::Error { exchange, error } => {
                println!("Exchange {} error: {}", exchange, error);
            }
        }
        
        Ok(())
    }
    
    /// Process trade and generate spikes
    async fn process_trade(&self, trade: crate::exchanges::UniversalTrade) -> Result<()> {
        self.initialize_symbol(trade.symbol.clone());
        
        let encoder = self.encoders.get(&trade.symbol).unwrap();
        let mut encoder = Arc::clone(&encoder.value());
        let encoder_mut = Arc::get_mut(&mut encoder).unwrap();
        
        // Update market state
        if let Some(tracker) = self.state_trackers.get(&trade.symbol) {
            let mut tracker = Arc::clone(&tracker.value());
            let tracker_mut = Arc::get_mut(&mut tracker).unwrap();
            tracker_mut.update_from_trade(trade.price, trade.quantity);
        }
        
        // Generate spikes from trade
        let spikes = encoder_mut.encode_trade(
            trade.price,
            trade.quantity,
            trade.side == crate::exchanges::Side::Buy
        );
        
        self.send_spikes(trade.symbol, spikes).await?;
        
        Ok(())
    }
    
    /// Process quote and generate spikes
    async fn process_quote(&self, quote: crate::exchanges::UniversalQuote) -> Result<()> {
        self.initialize_symbol(quote.symbol.clone());
        
        let encoder = self.encoders.get(&quote.symbol).unwrap();
        let mut encoder = Arc::clone(&encoder.value());
        let encoder_mut = Arc::get_mut(&mut encoder).unwrap();
        
        // Update market state
        if let Some(tracker) = self.state_trackers.get(&quote.symbol) {
            let mut tracker = Arc::clone(&tracker.value());
            let tracker_mut = Arc::get_mut(&mut tracker).unwrap();
            tracker_mut.update_from_quote(
                quote.bid_price,
                quote.ask_price,
                quote.bid_size + quote.ask_size
            );
        }
        
        // Create QuoteData
        let quote_data = QuoteData {
            bid_price: quote.bid_price,
            ask_price: quote.ask_price,
            bid_size: quote.bid_size,
            ask_size: quote.ask_size,
            timestamp: quote.timestamp_local,
        };
        
        // Generate spikes from quote
        let spikes = encoder_mut.encode_quote(&quote_data);
        
        self.send_spikes(quote.symbol, spikes).await?;
        
        Ok(())
    }
    
    /// Process order book and generate spikes
    async fn process_orderbook(&self, book: crate::exchanges::UniversalOrderBook) -> Result<()> {
        self.initialize_symbol(book.symbol.clone());
        
        let encoder = self.encoders.get(&book.symbol).unwrap();
        let mut encoder = Arc::clone(&encoder.value());
        let encoder_mut = Arc::get_mut(&mut encoder).unwrap();
        
        // Update market state
        if let Some(tracker) = self.state_trackers.get(&book.symbol) {
            if !book.bids.is_empty() && !book.asks.is_empty() {
                let mut tracker = Arc::clone(&tracker.value());
                let tracker_mut = Arc::get_mut(&mut tracker).unwrap();
                
                let best_bid = book.bids[0].0;
                let best_ask = book.asks[0].0;
                let total_volume = book.bids.iter().map(|(_, q)| q).sum::<f64>() +
                                 book.asks.iter().map(|(_, q)| q).sum::<f64>();
                
                tracker_mut.update_from_quote(best_bid, best_ask, total_volume);
            }
        }
        
        // Create OrderBookData
        let book_data = OrderBookData {
            bids: book.bids.clone(),
            asks: book.asks.clone(),
            timestamp: book.timestamp_local,
        };
        
        // Generate spikes from order book
        let spikes = encoder_mut.encode_orderbook(&book_data);
        
        self.send_spikes(book.symbol, spikes).await?;
        
        Ok(())
    }
    
    /// Send spikes and update statistics
    async fn send_spikes(&self, symbol: Symbol, spikes: Vec<Spike>) -> Result<()> {
        if spikes.is_empty() {
            return Ok(());
        }
        
        // Buffer spikes if batching is enabled
        if self.config.batch_size > 1 {
            let mut buffer = self.spike_buffer.get_mut(&symbol).unwrap();
            buffer.extend(spikes.clone());
            
            if buffer.len() >= self.config.batch_size {
                let batch = buffer.drain(..).collect::<Vec<_>>();
                self.spike_sender.send(batch)?;
            }
        } else {
            self.spike_sender.send(spikes.clone())?;
        }
        
        // Update statistics
        if let Some(mut stats) = self.statistics.get_mut(&symbol) {
            stats.events_processed += 1;
            stats.spikes_generated += spikes.len() as u64;
            
            if let Some(last_time) = stats.last_spike_time {
                let elapsed = last_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    stats.spike_rate_hz = spikes.len() as f64 / elapsed;
                }
            }
            stats.last_spike_time = Some(Instant::now());
        }
        
        Ok(())
    }
    
    /// Flush buffered spikes for a symbol
    pub async fn flush_symbol(&self, symbol: &Symbol) -> Result<()> {
        if let Some(mut buffer) = self.spike_buffer.get_mut(symbol) {
            if !buffer.is_empty() {
                let batch = buffer.drain(..).collect::<Vec<_>>();
                self.spike_sender.send(batch)?;
            }
        }
        Ok(())
    }
    
    /// Flush all buffered spikes
    pub async fn flush_all(&self) -> Result<()> {
        for entry in self.spike_buffer.iter() {
            let symbol = entry.key().clone();
            drop(entry); // Release the lock
            self.flush_symbol(&symbol).await?;
        }
        Ok(())
    }
    
    /// Subscribe to spike stream
    pub fn subscribe(&mut self) -> Option<mpsc::UnboundedReceiver<Vec<Spike>>> {
        self.spike_receiver.take()
    }
    
    /// Get statistics for a symbol
    pub fn get_statistics(&self, symbol: &Symbol) -> Option<SpikeStatistics> {
        self.statistics.get(symbol).map(|s| s.clone())
    }
    
    /// Get market state for a symbol
    pub fn get_market_state(&self, symbol: &Symbol) -> Option<MarketState> {
        self.state_trackers.get(symbol).map(|tracker| {
            let tracker = Arc::clone(&tracker.value());
            // This would need proper accessor methods in real implementation
            MarketState::default()
        })
    }
    
    /// Adaptive encoding adjustment based on market conditions
    pub fn adjust_encoding(&self, symbol: &Symbol) {
        if !self.config.enable_adaptive_encoding {
            return;
        }
        
        if let Some(tracker) = self.state_trackers.get(symbol) {
            if let Some(encoder) = self.encoders.get(symbol) {
                let mut encoder = Arc::clone(&encoder.value());
                if let Some(encoder_mut) = Arc::get_mut(&mut encoder) {
                    // Adjust sensitivity based on volatility
                    let state = self.get_market_state(symbol).unwrap_or_default();
                    
                    if state.volatility > 0.02 {
                        encoder_mut.set_sensitivity(2.0); // High volatility
                    } else if state.volatility > 0.01 {
                        encoder_mut.set_sensitivity(1.5); // Medium volatility
                    } else {
                        encoder_mut.set_sensitivity(1.0); // Low volatility
                    }
                }
            }
        }
    }
}

/// Integration handler for connecting unified feed to spike bridge
pub struct MarketSpikeIntegration {
    feed: Arc<UnifiedMarketFeed>,
    bridge: Arc<MarketDataSpikeBridge>,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl MarketSpikeIntegration {
    pub fn new(
        mut feed: UnifiedMarketFeed,
        bridge: MarketDataSpikeBridge
    ) -> Self {
        Self {
            feed: Arc::new(feed),
            bridge: Arc::new(bridge),
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }
    
    /// Start processing market events
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = true;
        drop(running);
        
        // Get event receiver from feed
        let mut feed = Arc::get_mut(&mut self.feed).unwrap();
        let mut receiver = feed.subscribe()
            .ok_or_else(|| anyhow::anyhow!("Failed to subscribe to feed"))?;
        
        let bridge = self.bridge.clone();
        let running = self.running.clone();
        
        // Spawn processing task
        tokio::spawn(async move {
            while *running.read().await {
                tokio::select! {
                    Some(event) = receiver.recv() => {
                        if let Err(e) = bridge.process_event(event).await {
                            eprintln!("Error processing event: {}", e);
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        // Periodic flush
                        if let Err(e) = bridge.flush_all().await {
                            eprintln!("Error flushing spikes: {}", e);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop processing
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        
        // Final flush
        self.bridge.flush_all().await?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_spike_bridge() {
        let config = SpikeBridgeConfig::default();
        let mut bridge = MarketDataSpikeBridge::new(config);
        
        let quote = crate::exchanges::UniversalQuote {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC-USD"),
            bid_price: 50000.0,
            bid_size: 1.0,
            ask_price: 50001.0,
            ask_size: 1.0,
            timestamp_exchange: 0,
            timestamp_local: 0,
        };
        
        let event = UnifiedMarketEvent::Quote(quote);
        bridge.process_event(event).await.unwrap();
        
        let stats = bridge.get_statistics(&Symbol::new("BTC-USD"));
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().events_processed, 1);
    }
}