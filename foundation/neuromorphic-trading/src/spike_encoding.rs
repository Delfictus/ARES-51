//! Spike Encoding Module
//! 
//! Converts market data to neural spikes using multiple encoding schemes:
//! - Rate coding: price changes → spike frequency
//! - Temporal coding: volume → inter-spike intervals
//! - Population coding: order book → spatial patterns

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;
use rand::Rng;
use rand_distr::{Normal, Distribution};

use crate::event_bus::{TradeData, QuoteData, OrderBookData};
use crate::MarketData;

/// Spike encoder configuration
pub struct EncoderConfig {
    pub num_neurons: usize,
    pub encoding_schemes: Vec<EncodingScheme>,
    pub window_size_ms: u32,
}

/// Encoding schemes
#[derive(Clone, Debug)]
pub enum EncodingScheme {
    RateCoding,
    TemporalCoding,
    PopulationCoding,
}

/// Single spike event
#[derive(Clone, Debug)]
pub struct Spike {
    pub timestamp_ns: u64,
    pub neuron_id: u32,
    pub strength: f32,
}

/// Neuron types
#[derive(Clone, Debug)]
pub enum NeuronType {
    Excitatory,  // 80% - positive signals
    Inhibitory,  // 20% - negative signals
    Modulatory,  // Special - volatility/regime changes
}

/// Receptive field for a neuron
#[derive(Clone)]
pub struct ReceptiveField {
    pub data_type: MarketDataType,
    pub price_range: (f64, f64),
    pub volume_range: (u64, u64),
    pub temporal_window: u32,
}

/// Market data types that neurons respond to
#[derive(Clone, Debug)]
pub enum MarketDataType {
    Price,
    Volume,
    Spread,
    Imbalance,
    Volatility,
}

/// Individual neuron
struct Neuron {
    id: u32,
    neuron_type: NeuronType,
    receptive_field: ReceptiveField,
    threshold: f32,
    refractory_period: u32,
    last_spike_time: u64,
}

impl Neuron {
    fn should_fire(&self, value: f64, current_time: u64) -> bool {
        // Check refractory period
        if current_time - self.last_spike_time < self.refractory_period as u64 {
            return false;
        }
        
        // Check if value is in receptive field
        match self.receptive_field.data_type {
            MarketDataType::Price => {
                let in_range = value >= self.receptive_field.price_range.0 
                    && value <= self.receptive_field.price_range.1;
                    
                // Gaussian activation based on distance from center
                if in_range {
                    let center = (self.receptive_field.price_range.0 + self.receptive_field.price_range.1) / 2.0;
                    let width = (self.receptive_field.price_range.1 - self.receptive_field.price_range.0) / 2.0;
                    let distance = (value - center).abs() / width;
                    let activation = (-distance * distance).exp();
                    activation > self.threshold
                } else {
                    false
                }
            },
            MarketDataType::Volume => {
                // Volume-based activation
                if value >= self.receptive_field.volume_range.0 as f64 &&
                   value <= self.receptive_field.volume_range.1 as f64 {
                    let range = (self.receptive_field.volume_range.1 - self.receptive_field.volume_range.0) as f64;
                    let normalized = (value - self.receptive_field.volume_range.0 as f64) / range;
                    normalized > self.threshold as f64
                } else {
                    false
                }
            },
            MarketDataType::Spread => {
                // Spread-sensitive neurons
                let normalized_spread = value / 0.01; // Normalize to cents
                normalized_spread > self.threshold as f64
            },
            MarketDataType::Imbalance => {
                // Imbalance detection (-1 to 1)
                value.abs() > self.threshold as f64
            },
            MarketDataType::Volatility => {
                // Volatility threshold
                value > self.threshold as f64 * 0.15 // 15% base volatility
            }
        }
    }
    
    fn compute_strength(&self, value: f64) -> f32 {
        // Compute spike strength based on receptive field
        match self.receptive_field.data_type {
            MarketDataType::Price => {
                let center = (self.receptive_field.price_range.0 + self.receptive_field.price_range.1) / 2.0;
                let width = (self.receptive_field.price_range.1 - self.receptive_field.price_range.0) / 2.0;
                let distance = (value - center).abs() / width;
                ((-distance * distance).exp() * 100.0) as f32
            },
            _ => 1.0
        }
    }
}

/// Time window for spike aggregation
struct TimeWindow {
    spikes: VecDeque<Spike>,
    window_size_ms: u32,
    current_time: u64,
}

impl TimeWindow {
    fn new(window_size_ms: u32) -> Self {
        Self {
            spikes: VecDeque::with_capacity(10000),
            window_size_ms,
            current_time: 0,
        }
    }
    
    fn add_spike(&mut self, spike: Spike) {
        self.current_time = spike.timestamp_ns;
        self.spikes.push_back(spike);
        
        // Remove old spikes outside window
        let cutoff = self.current_time.saturating_sub(self.window_size_ms as u64 * 1_000_000);
        while let Some(front) = self.spikes.front() {
            if front.timestamp_ns < cutoff {
                self.spikes.pop_front();
            } else {
                break;
            }
        }
    }
    
    fn add_spikes(&mut self, spikes: &[Spike]) {
        for spike in spikes {
            self.add_spike(spike.clone());
        }
    }
    
    fn get_spikes_in_window(&self) -> Vec<Spike> {
        self.spikes.iter().cloned().collect()
    }
}

/// Network topology organization
struct NetworkTopology {
    price_neurons: Vec<usize>,      // Indices of price-sensitive neurons
    volume_neurons: Vec<usize>,     // Indices of volume-sensitive neurons
    orderbook_neurons: Vec<usize>,  // Indices of order book neurons
    excitatory_ratio: f32,          // Ratio of excitatory neurons
}

/// Main spike encoder
pub struct SpikeEncoder {
    neurons: Vec<Neuron>,
    time_window: Arc<RwLock<TimeWindow>>,
    topology: NetworkTopology,
    config: EncoderConfig,
}

impl SpikeEncoder {
    pub fn new(config: EncoderConfig) -> Self {
        let neurons = Self::create_neurons(&config);
        let topology = Self::create_topology(&neurons);
        
        Self {
            neurons,
            time_window: Arc::new(RwLock::new(TimeWindow::new(config.window_size_ms))),
            topology,
            config,
        }
    }
    
    fn create_neurons(config: &EncoderConfig) -> Vec<Neuron> {
        let mut neurons = Vec::with_capacity(config.num_neurons);
        let mut rng = rand::thread_rng();
        
        for i in 0..config.num_neurons {
            // 80% excitatory, 20% inhibitory
            let neuron_type = if rng.gen::<f32>() < 0.8 {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };
            
            // Distribute neurons across price ranges
            let price_min = 100.0 + (i as f64 * 0.1);
            let price_max = price_min + 1.0;
            
            neurons.push(Neuron {
                id: i as u32,
                neuron_type,
                receptive_field: ReceptiveField {
                    data_type: MarketDataType::Price,
                    price_range: (price_min, price_max),
                    volume_range: (0, 1_000_000),
                    temporal_window: 100,
                },
                threshold: 0.5,
                refractory_period: 2, // 2ms
                last_spike_time: 0,
            });
        }
        
        neurons
    }
    
    fn create_topology(neurons: &[Neuron]) -> NetworkTopology {
        let mut price_neurons = Vec::new();
        let mut volume_neurons = Vec::new();
        let mut orderbook_neurons = Vec::new();
        
        for (i, neuron) in neurons.iter().enumerate() {
            match neuron.receptive_field.data_type {
                MarketDataType::Price => price_neurons.push(i),
                MarketDataType::Volume => volume_neurons.push(i),
                _ => orderbook_neurons.push(i),
            }
        }
        
        NetworkTopology {
            price_neurons,
            volume_neurons,
            orderbook_neurons,
            excitatory_ratio: 0.8,
        }
    }
    
    /// Encode market data to spikes
    pub fn encode(&mut self, data: &MarketData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Encode trade data
        if let Some(trade) = &data.trade {
            spikes.extend(self.encode_trade(trade));
        }
        
        // Encode quote data
        if let Some(quote) = &data.quote {
            spikes.extend(self.encode_quote(quote));
        }
        
        // Encode order book
        if let Some(book) = &data.order_book {
            spikes.extend(self.encode_orderbook(book));
        }
        
        // Add all spikes to time window
        let mut window = self.time_window.write();
        for spike in &spikes {
            window.add_spike(spike.clone());
        }
        
        spikes
    }
    
    /// Encode trade data using rate coding
    pub fn encode_trade(&mut self, trade: &TradeData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Rate coding for price
        for &neuron_idx in &self.topology.price_neurons {
            let neuron = &mut self.neurons[neuron_idx];
            
            if neuron.should_fire(trade.price, trade.timestamp_ns) {
                let spike = Spike {
                    timestamp_ns: trade.timestamp_ns,
                    neuron_id: neuron.id,
                    strength: neuron.compute_strength(trade.price),
                };
                
                spikes.push(spike);
                neuron.last_spike_time = trade.timestamp_ns;
            }
        }
        
        // Temporal and population coding are implemented in spike_encoding_enhanced.rs
        
        spikes
    }
    
    /// Encode quote data
    pub fn encode_quote(&mut self, quote: &QuoteData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        let timestamp = quote.timestamp_ns;
        
        // 1. Encode bid price
        let bid_neurons = self.find_neurons_for_price(quote.bid_price);
        for neuron_id in bid_neurons {
            if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                // Check refractory period
                if timestamp > neuron.last_spike_time + (neuron.refractory_period as u64 * 1_000_000) {
                    // Strength based on bid size
                    let size_factor = (quote.bid_size as f32 / 10000.0).min(2.0);
                    let strength = 0.7 * size_factor; // Bid side strength
                    
                    spikes.push(Spike {
                        timestamp_ns: timestamp,
                        neuron_id: neuron.id,
                        strength,
                    });
                    
                    neuron.last_spike_time = timestamp;
                }
            }
        }
        
        // 2. Encode ask price
        let ask_neurons = self.find_neurons_for_price(quote.ask_price);
        for neuron_id in ask_neurons {
            if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                if timestamp > neuron.last_spike_time + (neuron.refractory_period as u64 * 1_000_000) {
                    let size_factor = (quote.ask_size as f32 / 10000.0).min(2.0);
                    let strength = 0.7 * size_factor; // Ask side strength
                    
                    spikes.push(Spike {
                        timestamp_ns: timestamp,
                        neuron_id: neuron.id,
                        strength,
                    });
                    
                    neuron.last_spike_time = timestamp;
                }
            }
        }
        
        // 3. Encode spread as special pattern
        let spread = quote.ask_price - quote.bid_price;
        let spread_bps = (spread / quote.bid_price * 10000.0) as u32; // Basis points
        
        // Activate spread-sensitive neurons (using high neuron IDs)
        let spread_neuron_base = self.config.num_neurons - 100;
        for i in 0..spread_bps.min(20) {
            let neuron_id = spread_neuron_base + i as usize;
            if neuron_id < self.config.num_neurons {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: neuron_id as u32,
                    strength: 0.5 + (spread_bps as f32 / 100.0),
                });
            }
        }
        
        // 4. Encode bid/ask imbalance
        let total_size = quote.bid_size + quote.ask_size;
        if total_size > 0 {
            let bid_ratio = quote.bid_size as f32 / total_size as f32;
            let imbalance_neurons = if bid_ratio > 0.6 {
                // Bid pressure
                (self.config.num_neurons - 50)..self.config.num_neurons - 30
            } else if bid_ratio < 0.4 {
                // Ask pressure  
                (self.config.num_neurons - 30)..self.config.num_neurons - 10
            } else {
                // Balanced
                (self.config.num_neurons - 40)..self.config.num_neurons - 35
            };
            
            for neuron_id in imbalance_neurons {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: neuron_id as u32,
                    strength: (bid_ratio - 0.5).abs(),
                });
            }
        }
        
        // Update time window
        self.time_window.write().add_spikes(&spikes);
        
        spikes
    }
    
    /// Encode order book using population coding
    pub fn encode_orderbook(&mut self, book: &OrderBookData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        let timestamp = book.timestamp_ns;
        
        // Calculate mid price for reference
        let best_bid = book.bids[0].0;
        let best_ask = book.asks[0].0;
        let mid_price = (best_bid + best_ask) / 2.0;
        
        // 1. Encode bid levels with depth-weighted activation
        for (level, &(price, size)) in book.bids.iter().enumerate() {
            if price <= 0.0 || size == 0 {
                continue;
            }
            
            // Find neurons for this price level
            let neurons = self.find_neurons_for_price(price);
            
            // Weight by level (closer levels more important)
            let level_weight = 1.0 / (1.0 + level as f32 * 0.2);
            
            // Weight by size relative to level 1
            let size_weight = (size as f32 / book.bids[0].1.max(1) as f32).min(2.0);
            
            for neuron_id in neurons {
                if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                    if timestamp > neuron.last_spike_time + (neuron.refractory_period as u64 * 1_000_000) {
                        let strength = 0.6 * level_weight * size_weight;
                        
                        spikes.push(Spike {
                            timestamp_ns: timestamp,
                            neuron_id: neuron.id,
                            strength,
                        });
                        
                        neuron.last_spike_time = timestamp;
                    }
                }
            }
        }
        
        // 2. Encode ask levels
        for (level, &(price, size)) in book.asks.iter().enumerate() {
            if price <= 0.0 || size == 0 {
                continue;
            }
            
            let neurons = self.find_neurons_for_price(price);
            let level_weight = 1.0 / (1.0 + level as f32 * 0.2);
            let size_weight = (size as f32 / book.asks[0].1.max(1) as f32).min(2.0);
            
            for neuron_id in neurons {
                if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                    if timestamp > neuron.last_spike_time + (neuron.refractory_period as u64 * 1_000_000) {
                        let strength = 0.6 * level_weight * size_weight;
                        
                        spikes.push(Spike {
                            timestamp_ns: timestamp,
                            neuron_id: neuron.id,
                            strength,
                        });
                        
                        neuron.last_spike_time = timestamp;
                    }
                }
            }
        }
        
        // 3. Encode order book imbalance across all levels
        let total_bid_volume: u64 = book.bids.iter().map(|&(_, size)| size).sum();
        let total_ask_volume: u64 = book.asks.iter().map(|&(_, size)| size).sum();
        
        if total_bid_volume + total_ask_volume > 0 {
            let bid_ratio = total_bid_volume as f32 / (total_bid_volume + total_ask_volume) as f32;
            
            // Use dedicated neurons for imbalance
            let imbalance_base = self.config.num_neurons - 200;
            let imbalance_neurons = ((bid_ratio * 20.0) as usize).min(20);
            
            for i in 0..imbalance_neurons {
                let neuron_id = imbalance_base + i;
                if neuron_id < self.config.num_neurons {
                    spikes.push(Spike {
                        timestamp_ns: timestamp,
                        neuron_id: neuron_id as u32,
                        strength: bid_ratio,
                    });
                }
            }
        }
        
        // 4. Encode liquidity concentration (where is most volume)
        let mut max_bid_level = 0;
        let mut max_bid_size = 0u64;
        for (level, &(_, size)) in book.bids.iter().enumerate() {
            if size > max_bid_size {
                max_bid_size = size;
                max_bid_level = level;
            }
        }
        
        let mut max_ask_level = 0;
        let mut max_ask_size = 0u64;
        for (level, &(_, size)) in book.asks.iter().enumerate() {
            if size > max_ask_size {
                max_ask_size = size;
                max_ask_level = level;
            }
        }
        
        // Encode liquidity concentration pattern
        let liquidity_base = self.config.num_neurons - 250;
        if max_bid_level < 3 && max_ask_level < 3 {
            // Liquidity near touch - tight market
            for i in 0..5 {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: (liquidity_base + i) as u32,
                    strength: 1.0,
                });
            }
        } else if max_bid_level > 5 || max_ask_level > 5 {
            // Liquidity far from touch - wide market
            for i in 5..10 {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: (liquidity_base + i) as u32,
                    strength: 0.8,
                });
            }
        }
        
        // 5. Encode orderbook pressure (weighted price levels)
        let bid_pressure: f64 = book.bids.iter()
            .enumerate()
            .map(|(level, &(price, size))| {
                let distance = (mid_price - price).abs() / mid_price;
                size as f64 * (1.0 - distance.min(0.01) * 100.0)
            })
            .sum();
            
        let ask_pressure: f64 = book.asks.iter()
            .enumerate()
            .map(|(level, &(price, size))| {
                let distance = (price - mid_price).abs() / mid_price;
                size as f64 * (1.0 - distance.min(0.01) * 100.0)
            })
            .sum();
        
        let pressure_ratio = bid_pressure / (bid_pressure + ask_pressure + 1.0);
        let pressure_base = self.config.num_neurons - 300;
        let pressure_neurons = ((pressure_ratio * 10.0) as usize).min(10);
        
        for i in 0..pressure_neurons {
            spikes.push(Spike {
                timestamp_ns: timestamp,
                neuron_id: (pressure_base + i) as u32,
                strength: pressure_ratio as f32,
            });
        }
        
        // Update time window
        self.time_window.write().add_spikes(&spikes);
        
        spikes
    }
    
    /// Helper function to find neurons sensitive to a price
    fn find_neurons_for_price(&self, price: f64) -> Vec<usize> {
        let mut neurons = Vec::new();
        
        for (idx, neuron) in self.neurons.iter().enumerate() {
            if price >= neuron.receptive_field.price_range.0 && 
               price <= neuron.receptive_field.price_range.1 {
                neurons.push(idx);
            }
        }
        
        // If no exact match, find nearest neurons
        if neurons.is_empty() && !self.neurons.is_empty() {
            // Find closest neuron
            let mut min_distance = f64::MAX;
            let mut closest_idx = 0;
            
            for (idx, neuron) in self.neurons.iter().enumerate() {
                let mid_range = (neuron.receptive_field.price_range.0 + 
                                neuron.receptive_field.price_range.1) / 2.0;
                let distance = (price - mid_range).abs();
                
                if distance < min_distance {
                    min_distance = distance;
                    closest_idx = idx;
                }
            }
            
            neurons.push(closest_idx);
            
            // Add neighboring neurons for better coverage
            if closest_idx > 0 {
                neurons.push(closest_idx - 1);
            }
            if closest_idx < self.neurons.len() - 1 {
                neurons.push(closest_idx + 1);
            }
        }
        
        neurons
    }
    
    /// Get current spike train
    pub fn get_spike_train(&self) -> Vec<Spike> {
        self.time_window.read().get_spikes_in_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::Side;
    
    #[test]
    fn test_spike_encoder_creation() {
        let config = EncoderConfig {
            num_neurons: 1000,
            encoding_schemes: vec![EncodingScheme::RateCoding],
            window_size_ms: 100,
        };
        
        let encoder = SpikeEncoder::new(config);
        assert_eq!(encoder.neurons.len(), 1000);
    }
    
    #[test]
    fn test_spike_encoding() {
        let config = EncoderConfig {
            num_neurons: 100,
            encoding_schemes: vec![EncodingScheme::RateCoding],
            window_size_ms: 100,
        };
        
        let mut encoder = SpikeEncoder::new(config);
        
        let trade = TradeData {
            symbol: crate::event_bus::Symbol::AAPL,
            price: 150.0,
            quantity: 100,
            timestamp_ns: 1000000,
            exchange_timestamp: 1000000,
            aggressor_side: Side::Buy,
            trade_id: 1,
        };
        
        let spikes = encoder.encode_trade(&trade);
        assert!(!spikes.is_empty(), "Should generate some spikes");
    }
}