//! SIMD-optimized spike encoding
//! 
//! Features:
//! - AVX2/AVX512 vectorized operations
//! - Batch processing of neurons
//! - Cache-aligned data structures
//! - Parallel spike generation

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem;
use std::sync::Arc;
use parking_lot::RwLock;
use aligned_vec::{AVec, avec};
use rayon::prelude::*;
use anyhow::Result;

use crate::spike_encoding::{Spike, NeuronType};
use crate::event_bus::{TradeData, QuoteData, OrderBookData};
use crate::MarketData;

/// SIMD lane width
#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 8;  // AVX2 = 256 bits / 32 bits = 8 floats

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Aligned neuron data for SIMD processing
#[repr(align(64))]  // Cache-line aligned
#[derive(Clone)]
struct AlignedNeuronData {
    thresholds: AVec<f32>,     // AVec is automatically aligned
    activations: AVec<f32>,    
    refractory: AVec<u8>,      
    last_spike: AVec<u64>,     
}

impl AlignedNeuronData {
    fn new(size: usize) -> Self {
        // Round up to multiple of SIMD width
        let aligned_size = ((size + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        Self {
            thresholds: avec![0.5; aligned_size],
            activations: avec![0.0; aligned_size],
            refractory: avec![0; aligned_size],
            last_spike: avec![0; aligned_size],
        }
    }
}

/// SIMD-optimized spike encoder
pub struct SIMDSpikeEncoder {
    num_neurons: usize,
    aligned_size: usize,
    neurons: Arc<RwLock<AlignedNeuronData>>,
    receptive_fields: Arc<Vec<ReceptiveField>>,
    use_avx512: bool,
    use_avx2: bool,
}

/// Receptive field for price encoding
#[derive(Clone)]
struct ReceptiveField {
    center: f32,
    width: f32,
    neuron_indices: Vec<usize>,
}

impl SIMDSpikeEncoder {
    pub fn new(num_neurons: usize) -> Self {
        let aligned_size = ((num_neurons + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        // Detect CPU features
        let use_avx512 = is_x86_feature_detected!("avx512f");
        let use_avx2 = is_x86_feature_detected!("avx2");
        
        println!("SIMD features: AVX512={}, AVX2={}", use_avx512, use_avx2);
        
        // Create receptive fields
        let mut receptive_fields = Vec::new();
        let price_range = 50.0;  // $100-$200 range
        let base_price = 150.0;
        
        for i in 0..num_neurons {
            let center = base_price - price_range/2.0 + (i as f32 / num_neurons as f32) * price_range;
            let width = price_range / num_neurons as f32 * 2.0;  // Overlapping fields
            
            receptive_fields.push(ReceptiveField {
                center,
                width,
                neuron_indices: vec![i],
            });
        }
        
        Self {
            num_neurons,
            aligned_size,
            neurons: Arc::new(RwLock::new(AlignedNeuronData::new(aligned_size))),
            receptive_fields: Arc::new(receptive_fields),
            use_avx512,
            use_avx2,
        }
    }
    
    /// Encode market data using SIMD operations
    pub fn encode_simd(&self, data: &MarketData) -> Vec<Spike> {
        let timestamp = data.timestamp_ns;
        let mut spikes = Vec::new();
        
        // Process trade data
        if let Some(ref trade) = data.trade {
            let trade_spikes = self.encode_trade_simd(trade, timestamp);
            spikes.extend(trade_spikes);
        }
        
        // Process quote data
        if let Some(ref quote) = data.quote {
            let quote_spikes = self.encode_quote_simd(quote, timestamp);
            spikes.extend(quote_spikes);
        }
        
        // Process order book
        if let Some(ref book) = data.order_book {
            let book_spikes = self.encode_orderbook_simd(book, timestamp);
            spikes.extend(book_spikes);
        }
        
        spikes
    }
    
    /// Encode trade data with SIMD
    fn encode_trade_simd(&self, trade: &TradeData, timestamp: u64) -> Vec<Spike> {
        let mut neurons = self.neurons.write();
        let mut spikes = Vec::new();
        
        // Prepare input activation based on price
        let price_activation = self.compute_price_activation_simd(trade.price);
        
        // Volume modulation
        let volume_factor = (trade.quantity as f32 / 1000.0).min(2.0);
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if self.use_avx2 {
                self.process_neurons_avx2(
                    &mut neurons,
                    &price_activation,
                    volume_factor,
                    timestamp,
                    &mut spikes,
                );
            } else {
                self.process_neurons_scalar(
                    &mut neurons,
                    &price_activation,
                    volume_factor,
                    timestamp,
                    &mut spikes,
                );
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        self.process_neurons_scalar(
            &mut neurons,
            &price_activation,
            volume_factor,
            timestamp,
            &mut spikes,
        );
        
        spikes
    }
    
    /// Compute price activation using SIMD
    fn compute_price_activation_simd(&self, price: f64) -> AVec<f32> {
        let mut activations = avec![0.0f32; self.aligned_size];
        let price_f32 = price as f32;
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if self.use_avx2 {
                self.compute_gaussian_activation_avx2(&mut activations, price_f32);
            } else {
                self.compute_gaussian_activation_scalar(&mut activations, price_f32);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        self.compute_gaussian_activation_scalar(&mut activations, price_f32);
        
        activations
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn compute_gaussian_activation_avx2(&self, activations: &mut [f32], price: f32) {
        let price_vec = _mm256_set1_ps(price);
        
        for (i, field) in self.receptive_fields.iter().enumerate() {
            if i >= self.num_neurons {
                break;
            }
            
            let center_vec = _mm256_set1_ps(field.center);
            let width_vec = _mm256_set1_ps(field.width);
            
            // Gaussian: exp(-(price - center)^2 / (2 * width^2))
            let diff = _mm256_sub_ps(price_vec, center_vec);
            let diff_sq = _mm256_mul_ps(diff, diff);
            let width_sq = _mm256_mul_ps(width_vec, width_vec);
            let two_width_sq = _mm256_mul_ps(_mm256_set1_ps(2.0), width_sq);
            let exponent = _mm256_div_ps(diff_sq, two_width_sq);
            let neg_exponent = _mm256_sub_ps(_mm256_setzero_ps(), exponent);
            
            // Approximate exp using polynomial (faster than _mm256_exp_ps)
            let activation = self.fast_exp_avx2(neg_exponent);
            
            // Store result
            if i < activations.len() {
                let current = _mm256_loadu_ps(&activations[i]);
                let result = _mm256_add_ps(current, activation);
                _mm256_storeu_ps(&mut activations[i], result);
            }
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn fast_exp_avx2(&self, x: __m256) -> __m256 {
        // Fast exponential approximation using Taylor series
        // exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
        
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let sixth = _mm256_set1_ps(1.0 / 6.0);
        let twentyfourth = _mm256_set1_ps(1.0 / 24.0);
        
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x4 = _mm256_mul_ps(x3, x);
        
        let term2 = _mm256_mul_ps(x2, half);
        let term3 = _mm256_mul_ps(x3, sixth);
        let term4 = _mm256_mul_ps(x4, twentyfourth);
        
        let sum1 = _mm256_add_ps(one, x);
        let sum2 = _mm256_add_ps(sum1, term2);
        let sum3 = _mm256_add_ps(sum2, term3);
        let result = _mm256_add_ps(sum3, term4);
        
        // Clamp to [0, 1]
        let zero = _mm256_setzero_ps();
        _mm256_min_ps(_mm256_max_ps(result, zero), one)
    }
    
    fn compute_gaussian_activation_scalar(&self, activations: &mut [f32], price: f32) {
        for (i, field) in self.receptive_fields.iter().enumerate() {
            if i >= self.num_neurons {
                break;
            }
            
            let diff = price - field.center;
            let activation = (-diff * diff / (2.0 * field.width * field.width)).exp();
            activations[i] = activation;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn process_neurons_avx2(
        &self,
        neurons: &mut AlignedNeuronData,
        input_activation: &[f32],
        volume_factor: f32,
        timestamp: u64,
        spikes: &mut Vec<Spike>,
    ) {
        let volume_vec = _mm256_set1_ps(volume_factor);
        let leak_rate = _mm256_set1_ps(0.95);  // Leak factor
        
        // Process neurons in SIMD chunks
        for i in (0..self.aligned_size).step_by(SIMD_WIDTH) {
            // Load current activations
            let current = _mm256_load_ps(&neurons.activations[i]);
            
            // Load input
            let input = _mm256_load_ps(&input_activation[i]);
            let scaled_input = _mm256_mul_ps(input, volume_vec);
            
            // Leaky integration
            let leaked = _mm256_mul_ps(current, leak_rate);
            let new_activation = _mm256_add_ps(leaked, scaled_input);
            
            // Store updated activation
            _mm256_store_ps(&mut neurons.activations[i], new_activation);
            
            // Check thresholds
            let thresholds = _mm256_load_ps(&neurons.thresholds[i]);
            let mask = _mm256_cmp_ps(new_activation, thresholds, _CMP_GE_OQ);
            let mask_int = _mm256_movemask_ps(mask);
            
            // Generate spikes for neurons that crossed threshold
            if mask_int != 0 {
                for j in 0..SIMD_WIDTH {
                    if mask_int & (1 << j) != 0 {
                        let neuron_id = i + j;
                        if neuron_id < self.num_neurons {
                            // Check refractory period
                            if neurons.refractory[neuron_id] == 0 {
                                spikes.push(Spike {
                                    timestamp_ns: timestamp,
                                    neuron_id: neuron_id as u32,
                                    strength: neurons.activations[neuron_id],
                                });
                                
                                // Reset neuron
                                neurons.activations[neuron_id] = 0.0;
                                neurons.refractory[neuron_id] = 5;  // 5ms refractory
                                neurons.last_spike[neuron_id] = timestamp;
                            }
                        }
                    }
                }
            }
            
            // Update refractory periods
            for j in i..i.min(i + SIMD_WIDTH).min(self.num_neurons) {
                if neurons.refractory[j] > 0 {
                    neurons.refractory[j] -= 1;
                }
            }
        }
    }
    
    fn process_neurons_scalar(
        &self,
        neurons: &mut AlignedNeuronData,
        input_activation: &[f32],
        volume_factor: f32,
        timestamp: u64,
        spikes: &mut Vec<Spike>,
    ) {
        for i in 0..self.num_neurons {
            // Leaky integration
            neurons.activations[i] = neurons.activations[i] * 0.95 + 
                                     input_activation[i] * volume_factor;
            
            // Check threshold
            if neurons.activations[i] >= neurons.thresholds[i] {
                if neurons.refractory[i] == 0 {
                    spikes.push(Spike {
                        timestamp_ns: timestamp,
                        neuron_id: i as u32,
                        strength: neurons.activations[i],
                    });
                    
                    // Reset
                    neurons.activations[i] = 0.0;
                    neurons.refractory[i] = 5;
                    neurons.last_spike[i] = timestamp;
                }
            }
            
            // Update refractory
            if neurons.refractory[i] > 0 {
                neurons.refractory[i] -= 1;
            }
        }
    }
    
    /// Encode quote data with SIMD
    fn encode_quote_simd(&self, quote: &QuoteData, timestamp: u64) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Encode bid/ask spread
        let spread = ((quote.ask_price - quote.bid_price) * 10000.0) as f32;
        let mid_price = (quote.bid_price + quote.ask_price) / 2.0;
        
        // Process bid side
        let bid_activation = self.compute_price_activation_simd(quote.bid_price);
        let bid_spikes = self.generate_spikes_from_activation(&bid_activation, timestamp, spread);
        spikes.extend(bid_spikes);
        
        // Process ask side
        let ask_activation = self.compute_price_activation_simd(quote.ask_price);
        let ask_spikes = self.generate_spikes_from_activation(&ask_activation, timestamp, spread);
        spikes.extend(ask_spikes);
        
        spikes
    }
    
    /// Encode order book with SIMD
    fn encode_orderbook_simd(&self, book: &OrderBookData, timestamp: u64) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        // Parallel processing of order book levels
        let bid_spikes: Vec<Spike> = book.bids
            .par_iter()
            .enumerate()
            .flat_map(|(level, &(price, size))| {
                let activation = self.compute_level_activation(price, size, level);
                self.generate_spikes_from_activation(&activation, timestamp, level as f32)
            })
            .collect();
        
        let ask_spikes: Vec<Spike> = book.asks
            .par_iter()
            .enumerate()
            .flat_map(|(level, &(price, size))| {
                let activation = self.compute_level_activation(price, size, level);
                self.generate_spikes_from_activation(&activation, timestamp, level as f32)
            })
            .collect();
        
        spikes.extend(bid_spikes);
        spikes.extend(ask_spikes);
        
        spikes
    }
    
    fn compute_level_activation(&self, price: f64, size: u64, level: usize) -> AVec<f32> {
        let mut activation = self.compute_price_activation_simd(price);
        
        // Weight by size and level
        let size_factor = (size as f32 / 1000.0).min(2.0);
        let level_factor = 1.0 / (1.0 + level as f32);
        let combined_factor = size_factor * level_factor;
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if self.use_avx2 {
                let factor_vec = _mm256_set1_ps(combined_factor);
                for i in (0..self.aligned_size).step_by(SIMD_WIDTH) {
                    let current = _mm256_load_ps(&activation[i]);
                    let scaled = _mm256_mul_ps(current, factor_vec);
                    _mm256_store_ps(&mut activation[i], scaled);
                }
            } else {
                for i in 0..self.num_neurons {
                    activation[i] *= combined_factor;
                }
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        for i in 0..self.num_neurons {
            activation[i] *= combined_factor;
        }
        
        activation
    }
    
    fn generate_spikes_from_activation(
        &self,
        activation: &[f32],
        timestamp: u64,
        strength_multiplier: f32,
    ) -> Vec<Spike> {
        let mut spikes = Vec::new();
        
        for i in 0..self.num_neurons {
            if activation[i] > 0.5 {
                spikes.push(Spike {
                    timestamp_ns: timestamp,
                    neuron_id: i as u32,
                    strength: activation[i] * strength_multiplier,
                });
            }
        }
        
        spikes
    }
    
    /// Batch encode multiple market data samples
    pub fn batch_encode_simd(&self, data_batch: &[MarketData]) -> Vec<Vec<Spike>> {
        data_batch
            .par_iter()
            .map(|data| self.encode_simd(data))
            .collect()
    }
    
    /// Get encoding statistics
    pub fn get_stats(&self) -> EncodingStats {
        let neurons = self.neurons.read();
        
        let mut active_neurons = 0;
        let mut total_activation = 0.0;
        let mut max_activation = 0.0;
        
        for i in 0..self.num_neurons {
            if neurons.activations[i] > 0.0 {
                active_neurons += 1;
                total_activation += neurons.activations[i];
                max_activation = max_activation.max(neurons.activations[i]);
            }
        }
        
        EncodingStats {
            total_neurons: self.num_neurons,
            active_neurons,
            avg_activation: if active_neurons > 0 {
                total_activation / active_neurons as f32
            } else {
                0.0
            },
            max_activation,
            simd_enabled: self.use_avx2 || self.use_avx512,
        }
    }
}

/// Encoding statistics
#[derive(Debug)]
pub struct EncodingStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub avg_activation: f32,
    pub max_activation: f32,
    pub simd_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_bus::Symbol;
    
    #[test]
    fn test_simd_encoder() {
        let encoder = SIMDSpikeEncoder::new(1000);
        
        let trade = TradeData {
            symbol: Symbol::AAPL,
            price: 150.0,
            quantity: 1000,
            timestamp_ns: 1_000_000_000,
            exchange_timestamp: 1_000_000_000,
            aggressor_side: crate::event_bus::Side::Buy,
            trade_id: 1,
        };
        
        let data = MarketData {
            trade: Some(trade),
            quote: None,
            order_book: None,
            timestamp_ns: 1_000_000_000,
        };
        
        let spikes = encoder.encode_simd(&data);
        assert!(!spikes.is_empty());
        
        let stats = encoder.get_stats();
        println!("SIMD Encoding stats: {:?}", stats);
    }
    
    #[test]
    fn test_batch_encoding() {
        let encoder = SIMDSpikeEncoder::new(1000);
        
        let mut batch = Vec::new();
        for i in 0..100 {
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price: 150.0 + i as f64 * 0.1,
                quantity: 1000,
                timestamp_ns: 1_000_000_000 + i * 1_000_000,
                exchange_timestamp: 1_000_000_000 + i * 1_000_000,
                aggressor_side: if i % 2 == 0 {
                    crate::event_bus::Side::Buy
                } else {
                    crate::event_bus::Side::Sell
                },
                trade_id: i as u64,
            };
            
            batch.push(MarketData {
                trade: Some(trade),
                quote: None,
                order_book: None,
                timestamp_ns: 1_000_000_000 + i * 1_000_000,
            });
        }
        
        let start = std::time::Instant::now();
        let results = encoder.batch_encode_simd(&batch);
        let elapsed = start.elapsed();
        
        assert_eq!(results.len(), 100);
        println!("Batch encoding 100 samples: {:?}", elapsed);
        println!("Throughput: {:.0} samples/sec", 
                 100.0 / elapsed.as_secs_f64());
    }
    
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx_operations() {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let a = _mm256_set1_ps(1.0);
                let b = _mm256_set1_ps(2.0);
                let c = _mm256_add_ps(a, b);
                
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), c);
                
                assert_eq!(result[0], 3.0);
            }
        }
    }
}