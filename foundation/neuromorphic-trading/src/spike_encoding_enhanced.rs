//! Enhanced Spike Encoding with Temporal and Population Coding
//! 
//! Complete implementation of all three encoding schemes

use crate::spike_encoding::{Spike, NeuronType, MarketDataType};
use crate::event_bus::{TradeData, QuoteData, OrderBookData};
use rand::Rng;
use rand_distr::{Poisson, Distribution};
use std::collections::VecDeque;

/// Temporal coding for volume
pub struct TemporalEncoder {
    base_rate: f32,
    max_isi: f32, // Maximum inter-spike interval in ms
    min_isi: f32, // Minimum inter-spike interval in ms
    volume_neurons: Vec<VolumeNeuron>,
}

struct VolumeNeuron {
    id: u32,
    volume_threshold: u64,
    last_spike_time: u64,
    refractory_period: u32,
}

impl TemporalEncoder {
    pub fn new(num_neurons: usize) -> Self {
        let mut volume_neurons = Vec::with_capacity(num_neurons);
        
        for i in 0..num_neurons {
            volume_neurons.push(VolumeNeuron {
                id: 1000 + i as u32, // IDs 1000-1999 for volume neurons
                volume_threshold: 100 * (i + 1) as u64, // Different thresholds
                last_spike_time: 0,
                refractory_period: 2, // 2ms
            });
        }
        
        Self {
            base_rate: 100.0, // 100 Hz base rate
            max_isi: 50.0,    // 50ms max ISI
            min_isi: 1.0,     // 1ms min ISI
            volume_neurons,
        }
    }
    
    /// Encode volume using temporal coding (ISI inversely proportional to volume)
    pub fn encode_volume(&mut self, volume: u64, timestamp_ns: u64) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Calculate ISI based on volume (higher volume = shorter ISI = higher rate)
        let normalized_volume = (volume as f32 / 10000.0).min(1.0);
        let isi = self.max_isi - (normalized_volume * (self.max_isi - self.min_isi));
        let isi_ns = (isi * 1_000_000.0) as u64;
        
        for neuron in &mut self.volume_neurons {
            // Check if neuron should fire based on volume threshold
            if volume >= neuron.volume_threshold {
                // Check refractory period
                if timestamp_ns - neuron.last_spike_time > (neuron.refractory_period as u64 * 1_000_000) {
                    // Generate spike train with calculated ISI
                    let num_spikes = ((1000.0 / isi) as usize).min(10); // Up to 10 spikes
                    
                    for s in 0..num_spikes {
                        spikes.push(Spike {
                            timestamp_ns: timestamp_ns + (s as u64 * isi_ns),
                            neuron_id: neuron.id,
                            strength: 1.0 - (s as f32 * 0.1), // Decaying strength
                        });
                    }
                    
                    neuron.last_spike_time = timestamp_ns;
                }
            }
        }
        
        spikes
    }
    
    /// Generate Poisson spike train for continuous volume
    pub fn poisson_encode(&self, volume: u64, duration_ms: u32) -> Vec<Spike> {
        let mut spikes = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Rate proportional to volume
        let rate = (volume as f32 / 1000.0).min(500.0); // Max 500 Hz
        let lambda = rate * (duration_ms as f32 / 1000.0);
        
        if lambda > 0.0 {
            let poisson = Poisson::new(lambda).unwrap();
            let num_spikes = poisson.sample(&mut rng) as usize;
            
            for i in 0..num_spikes {
                let time_offset = rng.gen_range(0..duration_ms) as u64 * 1_000_000;
                let neuron_idx = rng.gen_range(0..self.volume_neurons.len());
                
                spikes.push(Spike {
                    timestamp_ns: time_offset,
                    neuron_id: self.volume_neurons[neuron_idx].id,
                    strength: rng.gen_range(0.5..1.0),
                });
            }
        }
        
        spikes
    }
}

/// Population coding for order book
pub struct PopulationEncoder {
    price_levels: usize,
    neurons_per_level: usize,
    bid_neurons: Vec<Vec<u32>>, // Neurons for each bid level
    ask_neurons: Vec<Vec<u32>>, // Neurons for each ask level
    imbalance_neurons: Vec<u32>, // Special neurons for imbalance detection
}

impl PopulationEncoder {
    pub fn new(levels: usize, neurons_per_level: usize) -> Self {
        let mut bid_neurons = Vec::with_capacity(levels);
        let mut ask_neurons = Vec::with_capacity(levels);
        let mut neuron_id = 2000u32; // IDs 2000+ for order book neurons
        
        // Create neuron populations for each price level
        for _ in 0..levels {
            let mut bid_level = Vec::with_capacity(neurons_per_level);
            let mut ask_level = Vec::with_capacity(neurons_per_level);
            
            for _ in 0..neurons_per_level {
                bid_level.push(neuron_id);
                neuron_id += 1;
                ask_level.push(neuron_id);
                neuron_id += 1;
            }
            
            bid_neurons.push(bid_level);
            ask_neurons.push(ask_level);
        }
        
        // Imbalance detection neurons
        let imbalance_neurons: Vec<u32> = (neuron_id..neuron_id + 100).collect();
        
        Self {
            price_levels: levels,
            neurons_per_level,
            bid_neurons,
            ask_neurons,
            imbalance_neurons,
        }
    }
    
    /// Encode order book using population coding
    pub fn encode_orderbook(&self, book: &OrderBookData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        let timestamp = book.timestamp_ns;
        
        // Encode bid levels
        for (level, &(price, size)) in book.bids.iter().enumerate() {
            if level >= self.price_levels || size == 0 {
                continue;
            }
            
            let normalized_size = (size as f32 / 10000.0).min(1.0);
            let active_neurons = (normalized_size * self.neurons_per_level as f32) as usize;
            
            for i in 0..active_neurons.min(self.neurons_per_level) {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: self.bid_neurons[level][i],
                    strength: normalized_size * (1.0 - (i as f32 / self.neurons_per_level as f32) * 0.5),
                });
            }
        }
        
        // Encode ask levels
        for (level, &(price, size)) in book.asks.iter().enumerate() {
            if level >= self.price_levels || size == 0 {
                continue;
            }
            
            let normalized_size = (size as f32 / 10000.0).min(1.0);
            let active_neurons = (normalized_size * self.neurons_per_level as f32) as usize;
            
            for i in 0..active_neurons.min(self.neurons_per_level) {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: self.ask_neurons[level][i],
                    strength: normalized_size * (1.0 - (i as f32 / self.neurons_per_level as f32) * 0.5),
                });
            }
        }
        
        // Encode imbalance
        let imbalance_spikes = self.encode_imbalance(book);
        spikes.extend(imbalance_spikes);
        
        spikes
    }
    
    /// Detect and encode order book imbalance
    fn encode_imbalance(&self, book: &OrderBookData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Calculate total bid and ask volumes
        let total_bid_volume: u64 = book.bids.iter().map(|(_, size)| size).sum();
        let total_ask_volume: u64 = book.asks.iter().map(|(_, size)| size).sum();
        
        if total_bid_volume + total_ask_volume == 0 {
            return spikes;
        }
        
        // Calculate imbalance ratio (-1 to 1, negative = more asks, positive = more bids)
        let imbalance = (total_bid_volume as f32 - total_ask_volume as f32) / 
                       (total_bid_volume as f32 + total_ask_volume as f32);
        
        // Map imbalance to neuron population
        let center = self.imbalance_neurons.len() / 2;
        let spread = (imbalance.abs() * center as f32) as usize;
        
        if imbalance > 0.0 {
            // Bid pressure - activate neurons above center
            for i in 0..spread.min(center) {
                spikes.push(Spike {
                    timestamp_ns: book.timestamp_ns,
                    neuron_id: self.imbalance_neurons[center + i],
                    strength: imbalance * (1.0 - (i as f32 / center as f32) * 0.5),
                });
            }
        } else {
            // Ask pressure - activate neurons below center
            for i in 0..spread.min(center) {
                spikes.push(Spike {
                    timestamp_ns: book.timestamp_ns,
                    neuron_id: self.imbalance_neurons[center - i],
                    strength: imbalance.abs() * (1.0 - (i as f32 / center as f32) * 0.5),
                });
            }
        }
        
        spikes
    }
    
    /// Encode microstructure patterns
    pub fn encode_microstructure(&self, book: &OrderBookData) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Detect patterns like:
        // 1. Tight spreads (market maker presence)
        // 2. Large orders at specific levels (support/resistance)
        // 3. Rapid changes in book depth
        
        // Spread encoding
        if book.asks[0].0 > 0.0 && book.bids[0].0 > 0.0 {
            let spread = book.asks[0].0 - book.bids[0].0;
            let normalized_spread = (spread / book.bids[0].0).min(0.01); // Max 1% spread
            
            // Tight spread = high activity in special neurons
            if normalized_spread < 0.001 { // Less than 0.1% spread
                for i in 0..10 {
                    spikes.push(Spike {
                        timestamp_ns: book.timestamp_ns + i * 100_000, // Burst pattern
                        neuron_id: self.imbalance_neurons[90 + i as usize],
                        strength: 1.0 - normalized_spread * 100.0,
                    });
                }
            }
        }
        
        spikes
    }
}

/// Stochastic resonance for weak signal amplification
pub struct StochasticResonance {
    noise_level: f32,
    bistable_threshold: f32,
    history: VecDeque<f32>,
}

impl StochasticResonance {
    pub fn new(noise_level: f32) -> Self {
        Self {
            noise_level,
            bistable_threshold: 0.5,
            history: VecDeque::with_capacity(100),
        }
    }
    
    /// Add calibrated noise to enhance weak signals
    pub fn enhance_spikes(&mut self, spikes: &mut Vec<Spike>) {
        let mut rng = rand::thread_rng();
        
        for spike in spikes.iter_mut() {
            // Add noise to strength
            let noise = rng.gen_range(-self.noise_level..self.noise_level);
            let enhanced_strength = spike.strength + noise;
            
            // Bistable system - push toward 0 or 1
            if enhanced_strength > self.bistable_threshold {
                spike.strength = (enhanced_strength * 1.2).min(1.0);
            } else if enhanced_strength > 0.0 {
                spike.strength = enhanced_strength * 0.8;
            }
            
            self.history.push_back(spike.strength);
            if self.history.len() > 100 {
                self.history.pop_front();
            }
        }
    }
    
    /// Detect subthreshold signals using stochastic resonance
    pub fn detect_weak_patterns(&self, signal: &[f32]) -> Vec<usize> {
        let mut patterns = Vec::new();
        let mut rng = rand::thread_rng();
        
        for (i, &value) in signal.iter().enumerate() {
            let noise = rng.gen_range(-self.noise_level..self.noise_level);
            let enhanced = value + noise;
            
            // Signal crosses threshold with noise
            if value < self.bistable_threshold && enhanced > self.bistable_threshold {
                patterns.push(i);
            }
        }
        
        patterns
    }
}

/// Predictive spike caching
pub struct PredictiveCache {
    cache: HashMap<u64, Vec<Spike>>,
    pattern_memory: VecDeque<MarketPattern>,
    prediction_horizon: u32, // ms
}

use std::collections::HashMap;

#[derive(Clone)]
struct MarketPattern {
    price_pattern: Vec<f64>,
    volume_pattern: Vec<u64>,
    timestamp: u64,
}

impl PredictiveCache {
    pub fn new(horizon_ms: u32) -> Self {
        Self {
            cache: HashMap::new(),
            pattern_memory: VecDeque::with_capacity(1000),
            prediction_horizon: horizon_ms,
        }
    }
    
    /// Pre-generate spikes for likely future scenarios
    pub fn pregenerate_spikes(&mut self, current_state: &MarketPattern, encoder: &mut impl SpikeGenerator) {
        // Find similar historical patterns
        let similar = self.find_similar_patterns(current_state, 5);
        
        // Generate spikes for likely next states
        for pattern in similar {
            let future_hash = self.hash_pattern(&pattern);
            if !self.cache.contains_key(&future_hash) {
                let spikes = encoder.generate_spikes(&pattern);
                self.cache.insert(future_hash, spikes);
            }
        }
        
        // Store current pattern
        self.pattern_memory.push_back(current_state.clone());
        if self.pattern_memory.len() > 1000 {
            self.pattern_memory.pop_front();
        }
        
        // Clean old cache entries
        self.clean_cache();
    }
    
    /// Get cached spikes if available
    pub fn get_cached(&self, pattern_hash: u64) -> Option<&Vec<Spike>> {
        self.cache.get(&pattern_hash)
    }
    
    fn find_similar_patterns(&self, current: &MarketPattern, n: usize) -> Vec<MarketPattern> {
        let mut similarities: Vec<(f32, MarketPattern)> = Vec::new();
        
        for pattern in &self.pattern_memory {
            let similarity = self.compute_similarity(current, pattern);
            similarities.push((similarity, pattern.clone()));
        }
        
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities.into_iter()
            .take(n)
            .map(|(_, p)| p)
            .collect()
    }
    
    fn compute_similarity(&self, p1: &MarketPattern, p2: &MarketPattern) -> f32 {
        // Simple Euclidean distance for now
        let price_diff: f64 = p1.price_pattern.iter()
            .zip(&p2.price_pattern)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        1.0 / (1.0 + price_diff as f32)
    }
    
    fn hash_pattern(&self, pattern: &MarketPattern) -> u64 {
        // Simple hash based on price levels
        let mut hash = 0u64;
        for (i, &price) in pattern.price_pattern.iter().enumerate() {
            hash ^= ((price * 1000.0) as u64).rotate_left((i * 7) as u32);
        }
        hash
    }
    
    fn clean_cache(&mut self) {
        if self.cache.len() > 10000 {
            // Remove oldest entries
            let to_remove = self.cache.len() / 2;
            let keys: Vec<u64> = self.cache.keys().take(to_remove).cloned().collect();
            for key in keys {
                self.cache.remove(&key);
            }
        }
    }
}

/// Trait for spike generation
trait SpikeGenerator {
    fn generate_spikes(&mut self, pattern: &MarketPattern) -> Vec<Spike>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temporal_encoding() {
        let mut encoder = TemporalEncoder::new(10);
        
        let spikes = encoder.encode_volume(5000, 1_000_000_000);
        assert!(!spikes.is_empty());
        
        // Higher volume should generate more spikes
        let spikes_high = encoder.encode_volume(10000, 2_000_000_000);
        let spikes_low = encoder.encode_volume(1000, 3_000_000_000);
        assert!(spikes_high.len() >= spikes_low.len());
    }
    
    #[test]
    fn test_population_encoding() {
        let encoder = PopulationEncoder::new(10, 10);
        
        let book = OrderBookData {
            symbol: crate::event_bus::Symbol::AAPL,
            bids: [
                (150.00, 1000), (149.99, 2000), (149.98, 3000),
                (149.97, 1500), (149.96, 1000), (149.95, 500),
                (149.94, 300), (149.93, 200), (149.92, 100), (149.91, 50),
            ],
            asks: [
                (150.01, 1000), (150.02, 2000), (150.03, 3000),
                (150.04, 1500), (150.05, 1000), (150.06, 500),
                (150.07, 300), (150.08, 200), (150.09, 100), (150.10, 50),
            ],
            timestamp_ns: 1_000_000_000,
            sequence_number: 1,
        };
        
        let spikes = encoder.encode_orderbook(&book);
        assert!(!spikes.is_empty());
        
        // Should encode both bid and ask sides
        let bid_spikes: Vec<_> = spikes.iter()
            .filter(|s| s.neuron_id >= 2000 && s.neuron_id < 3000)
            .collect();
        let ask_spikes: Vec<_> = spikes.iter()
            .filter(|s| s.neuron_id >= 3000 && s.neuron_id < 4000)
            .collect();
        
        assert!(!bid_spikes.is_empty());
        assert!(!ask_spikes.is_empty());
    }
    
    #[test]
    fn test_stochastic_resonance() {
        let mut sr = StochasticResonance::new(0.1);
        
        let mut weak_spikes = vec![
            Spike { timestamp_ns: 1000, neuron_id: 0, strength: 0.4 },
            Spike { timestamp_ns: 2000, neuron_id: 1, strength: 0.45 },
            Spike { timestamp_ns: 3000, neuron_id: 2, strength: 0.3 },
        ];
        
        sr.enhance_spikes(&mut weak_spikes);
        
        // Some spikes should be enhanced
        let enhanced_count = weak_spikes.iter()
            .filter(|s| s.strength > 0.5)
            .count();
        
        assert!(enhanced_count > 0);
    }
}