//! Market pattern training module
//! 
//! Trains the neuromorphic system to recognize:
//! 1. Momentum - Sustained directional movement
//! 2. Reversal - Trend changes
//! 3. Breakout - Breaking support/resistance
//! 4. Consolidation - Sideways movement
//! 5. Volatility Spike - Sudden volatility increase
//! 6. Trend - Long-term directional bias

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use rand::Rng;
use anyhow::Result;

use crate::spike_encoding::{Spike, SpikeEncoder, EncoderConfig, EncodingScheme};
use crate::event_bus::{TradeData, QuoteData, OrderBookData, Symbol, Side};
use crate::reservoir::PatternType;
use crate::reservoir_stdp::{STDPReservoir, STDPConfig};
use crate::MarketData;

/// Pattern generator for synthetic training data
pub struct PatternGenerator {
    base_price: f64,
    base_volume: u64,
    timestamp: u64,
    noise_level: f64,
}

impl PatternGenerator {
    pub fn new() -> Self {
        Self {
            base_price: 150.0,
            base_volume: 1000,
            timestamp: 1_000_000_000,
            noise_level: 0.01,
        }
    }
    
    /// Generate momentum pattern data
    pub fn generate_momentum(&mut self, steps: usize, direction: f64) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..steps {
            // Steady price increase/decrease with accelerating volume
            let price_change = direction * (i as f64 / steps as f64) * 5.0;
            let noise = rng.gen_range(-self.noise_level..self.noise_level) * self.base_price;
            let price = self.base_price + price_change + noise;
            
            // Volume increases with momentum
            let volume = self.base_volume * (1 + i / 10) as u64;
            
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price,
                quantity: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
                aggressor_side: if direction > 0.0 { Side::Buy } else { Side::Sell },
                trade_id: i as u64,
            };
            
            let quote = QuoteData {
                symbol: Symbol::AAPL,
                bid_price: price - 0.01,
                bid_size: volume / 2,
                ask_price: price + 0.01,
                ask_size: volume / 2,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
            };
            
            data.push(MarketData {
                trade: Some(trade),
                quote: Some(quote),
                order_book: None,
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000; // 1ms
        }
        
        data
    }
    
    /// Generate reversal pattern data
    pub fn generate_reversal(&mut self, steps: usize) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        let peak_point = steps / 2;
        
        for i in 0..steps {
            // Price rises then falls
            let price = if i < peak_point {
                self.base_price + (i as f64 / peak_point as f64) * 5.0
            } else {
                self.base_price + 5.0 - ((i - peak_point) as f64 / peak_point as f64) * 5.0
            };
            
            let noise = rng.gen_range(-self.noise_level..self.noise_level) * price;
            let final_price = price + noise;
            
            // Volume spikes at reversal point
            let volume_multiplier = if (i as i32 - peak_point as i32).abs() < 5 {
                3.0
            } else {
                1.0
            };
            let volume = (self.base_volume as f64 * volume_multiplier) as u64;
            
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price: final_price,
                quantity: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
                aggressor_side: if i < peak_point { Side::Buy } else { Side::Sell },
                trade_id: i as u64,
            };
            
            data.push(MarketData {
                trade: Some(trade),
                quote: None,
                order_book: None,
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000;
        }
        
        data
    }
    
    /// Generate breakout pattern data
    pub fn generate_breakout(&mut self, steps: usize, resistance: f64) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        let breakout_point = steps * 2 / 3;
        
        for i in 0..steps {
            let price = if i < breakout_point {
                // Consolidation near resistance
                resistance - rng.gen_range(0.0..1.0)
            } else {
                // Breakout with strong momentum
                resistance + ((i - breakout_point) as f64 / 10.0) * 2.0
            };
            
            // Volume surge on breakout
            let volume = if i >= breakout_point {
                self.base_volume * 5
            } else {
                self.base_volume
            };
            
            // Order book shows accumulation
            let mut bids = [(0.0, 0); 10];
            let mut asks = [(0.0, 0); 10];
            
            for j in 0..10 {
                bids[j] = (price - 0.01 * (j + 1) as f64, volume / (j + 1) as u64);
                asks[j] = (price + 0.01 * (j + 1) as f64, volume / (j + 1) as u64);
            }
            
            // Imbalance favors breakout direction
            if i >= breakout_point {
                for j in 0..5 {
                    bids[j].1 *= 2;  // Strong bid support
                }
            }
            
            let order_book = OrderBookData {
                symbol: Symbol::AAPL,
                bids,
                asks,
                timestamp_ns: self.timestamp,
                sequence_number: i as u64,
            };
            
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price,
                quantity: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
                aggressor_side: if i >= breakout_point { Side::Buy } else { Side::Sell },
                trade_id: i as u64,
            };
            
            data.push(MarketData {
                trade: Some(trade),
                quote: None,
                order_book: Some(order_book),
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000;
        }
        
        data
    }
    
    /// Generate consolidation pattern data
    pub fn generate_consolidation(&mut self, steps: usize, range: f64) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..steps {
            // Price oscillates within range
            let oscillation = (i as f64 * 0.2).sin() * range / 2.0;
            let noise = rng.gen_range(-self.noise_level..self.noise_level) * self.base_price;
            let price = self.base_price + oscillation + noise;
            
            // Low, steady volume
            let volume = self.base_volume / 2 + rng.gen_range(0..200);
            
            // Balanced order book
            let mut bids = [(0.0, 0); 10];
            let mut asks = [(0.0, 0); 10];
            
            for j in 0..10 {
                let size = volume / 10;
                bids[j] = (price - 0.01 * (j + 1) as f64, size);
                asks[j] = (price + 0.01 * (j + 1) as f64, size);
            }
            
            let quote = QuoteData {
                symbol: Symbol::AAPL,
                bid_price: price - 0.01,
                bid_size: volume / 2,
                ask_price: price + 0.01,
                ask_size: volume / 2,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
            };
            
            let order_book = OrderBookData {
                symbol: Symbol::AAPL,
                bids,
                asks,
                timestamp_ns: self.timestamp,
                sequence_number: i as u64,
            };
            
            data.push(MarketData {
                trade: None,
                quote: Some(quote),
                order_book: Some(order_book),
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000;
        }
        
        data
    }
    
    /// Generate volatility spike pattern data
    pub fn generate_volatility_spike(&mut self, steps: usize) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        let spike_start = steps / 3;
        let spike_end = steps * 2 / 3;
        
        for i in 0..steps {
            // Volatility increases dramatically
            let volatility = if i >= spike_start && i < spike_end {
                5.0  // High volatility period
            } else {
                0.5  // Normal volatility
            };
            
            let price_change = rng.gen_range(-volatility..volatility);
            let price = self.base_price + price_change;
            
            // Volume surges with volatility
            let volume = if i >= spike_start && i < spike_end {
                self.base_volume * 10
            } else {
                self.base_volume
            };
            
            // Wide spreads during volatility
            let spread = if i >= spike_start && i < spike_end {
                0.10
            } else {
                0.01
            };
            
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price,
                quantity: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
                aggressor_side: if rng.gen_bool(0.5) { Side::Buy } else { Side::Sell },
                trade_id: i as u64,
            };
            
            let quote = QuoteData {
                symbol: Symbol::AAPL,
                bid_price: price - spread,
                bid_size: volume,
                ask_price: price + spread,
                ask_size: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
            };
            
            data.push(MarketData {
                trade: Some(trade),
                quote: Some(quote),
                order_book: None,
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000;
        }
        
        data
    }
    
    /// Generate trend pattern data
    pub fn generate_trend(&mut self, steps: usize, slope: f64) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..steps {
            // Linear trend with pullbacks
            let trend_price = self.base_price + slope * i as f64;
            
            // Periodic pullbacks
            let pullback = if i % 20 < 5 {
                -slope * 3.0  // Pullback against trend
            } else {
                0.0
            };
            
            let noise = rng.gen_range(-self.noise_level..self.noise_level) * self.base_price;
            let price = trend_price + pullback + noise;
            
            // Volume confirms trend
            let volume = if pullback < 0.0 {
                self.base_volume / 2  // Lower volume on pullbacks
            } else {
                self.base_volume * 2  // Higher volume on trend moves
            };
            
            // Order book shows trend bias
            let mut bids = [(0.0, 0); 10];
            let mut asks = [(0.0, 0); 10];
            
            for j in 0..10 {
                if slope > 0.0 {
                    // Uptrend: more aggressive bids
                    bids[j] = (price - 0.01 * (j + 1) as f64, volume * 2 / (j + 1) as u64);
                    asks[j] = (price + 0.01 * (j + 1) as f64, volume / (j + 2) as u64);
                } else {
                    // Downtrend: more aggressive asks
                    bids[j] = (price - 0.01 * (j + 1) as f64, volume / (j + 2) as u64);
                    asks[j] = (price + 0.01 * (j + 1) as f64, volume * 2 / (j + 1) as u64);
                }
            }
            
            let trade = TradeData {
                symbol: Symbol::AAPL,
                price,
                quantity: volume,
                timestamp_ns: self.timestamp,
                exchange_timestamp: self.timestamp,
                aggressor_side: if slope > 0.0 { Side::Buy } else { Side::Sell },
                trade_id: i as u64,
            };
            
            let order_book = OrderBookData {
                symbol: Symbol::AAPL,
                bids,
                asks,
                timestamp_ns: self.timestamp,
                sequence_number: i as u64,
            };
            
            data.push(MarketData {
                trade: Some(trade),
                quote: None,
                order_book: Some(order_book),
                timestamp_ns: self.timestamp,
            });
            
            self.timestamp += 1_000_000;
        }
        
        data
    }
}

/// Pattern trainer for the neuromorphic system
pub struct PatternTrainer {
    encoder: SpikeEncoder,
    reservoir: Arc<RwLock<STDPReservoir>>,
    generator: PatternGenerator,
    training_history: Vec<TrainingRecord>,
}

/// Training record
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub pattern: PatternType,
    pub accuracy: f32,
    pub epochs: u32,
    pub final_loss: f32,
}

impl PatternTrainer {
    pub fn new(encoder_config: EncoderConfig, reservoir_size: usize) -> Self {
        let encoder = SpikeEncoder::new(encoder_config);
        let reservoir = Arc::new(RwLock::new(
            STDPReservoir::new(reservoir_size, STDPConfig::default())
        ));
        
        Self {
            encoder,
            reservoir,
            generator: PatternGenerator::new(),
            training_history: Vec::new(),
        }
    }
    
    /// Train all 6 market patterns
    pub fn train_all_patterns(&mut self, epochs_per_pattern: u32) -> Result<Vec<TrainingRecord>> {
        let patterns = vec![
            PatternType::Momentum,
            PatternType::Reversal,
            PatternType::Breakout,
            PatternType::Consolidation,
            PatternType::Volatility,
            PatternType::Trend,
        ];
        
        for pattern in patterns {
            println!("Training pattern: {:?}", pattern);
            let record = self.train_pattern(pattern, epochs_per_pattern)?;
            self.training_history.push(record.clone());
            println!("  Accuracy: {:.2}%", record.accuracy * 100.0);
        }
        
        Ok(self.training_history.clone())
    }
    
    /// Train a specific pattern
    pub fn train_pattern(&mut self, pattern: PatternType, epochs: u32) -> Result<TrainingRecord> {
        let mut best_accuracy = 0.0;
        let mut total_loss = 0.0;
        
        for epoch in 0..epochs {
            // Generate training data for this pattern
            let market_data = self.generate_pattern_data(pattern, 100);
            
            // Convert to spikes
            let mut training_data = Vec::new();
            for data in market_data {
                let spikes = self.encoder.encode(&data);
                training_data.push((spikes, pattern));
            }
            
            // Train reservoir
            self.reservoir.write().train_patterns(training_data.clone())?;
            
            // Evaluate accuracy
            let accuracy = self.evaluate_pattern(pattern, &training_data);
            best_accuracy = best_accuracy.max(accuracy);
            
            // Calculate loss (1 - accuracy)
            let loss = 1.0 - accuracy;
            total_loss += loss;
            
            if epoch % 10 == 0 {
                println!("  Epoch {}/{}: accuracy={:.3}, loss={:.3}", 
                         epoch, epochs, accuracy, loss);
            }
            
            // Early stopping if accuracy is good
            if accuracy > 0.9 {
                break;
            }
        }
        
        Ok(TrainingRecord {
            pattern,
            accuracy: best_accuracy,
            epochs,
            final_loss: total_loss / epochs as f32,
        })
    }
    
    fn generate_pattern_data(&mut self, pattern: PatternType, samples: usize) -> Vec<MarketData> {
        match pattern {
            PatternType::Momentum => self.generator.generate_momentum(samples, 1.0),
            PatternType::Reversal => self.generator.generate_reversal(samples),
            PatternType::Breakout => self.generator.generate_breakout(samples, 155.0),
            PatternType::Consolidation => self.generator.generate_consolidation(samples, 2.0),
            PatternType::Volatility => self.generator.generate_volatility_spike(samples),
            PatternType::Trend => self.generator.generate_trend(samples, 0.1),
        }
    }
    
    fn evaluate_pattern(&mut self, target: PatternType, data: &[(Vec<Spike>, PatternType)]) -> f32 {
        let mut correct = 0;
        let mut total = 0;
        
        for (spikes, expected) in data {
            let state = self.reservoir.write().process_with_learning(spikes);
            
            // Find strongest detected pattern
            let detected = self.detect_strongest_pattern(&state.activations);
            
            if detected == Some(*expected) {
                correct += 1;
            }
            total += 1;
        }
        
        correct as f32 / total as f32
    }
    
    fn detect_strongest_pattern(&self, activations: &[f32]) -> Option<PatternType> {
        // Simple pattern detection based on activation patterns
        // In production, this would use the full pattern detector
        
        let sum: f32 = activations.iter().sum();
        let mean = sum / activations.len() as f32;
        let variance: f32 = activations.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / activations.len() as f32;
        
        // Heuristic pattern detection
        if variance > 0.5 {
            Some(PatternType::Volatility)
        } else if mean > 0.5 {
            Some(PatternType::Momentum)
        } else if mean < -0.5 {
            Some(PatternType::Reversal)
        } else {
            Some(PatternType::Consolidation)
        }
    }
    
    /// Test trained patterns
    pub fn test_patterns(&mut self) -> Result<f32> {
        let mut total_accuracy = 0.0;
        let patterns = vec![
            PatternType::Momentum,
            PatternType::Reversal,
            PatternType::Breakout,
            PatternType::Consolidation,
            PatternType::Volatility,
            PatternType::Trend,
        ];
        
        for pattern in patterns {
            // Generate test data
            let test_data = self.generate_pattern_data(pattern, 50);
            let mut spikes_data = Vec::new();
            
            for data in test_data {
                let spikes = self.encoder.encode(&data);
                spikes_data.push((spikes, pattern));
            }
            
            // Evaluate
            let accuracy = self.evaluate_pattern(pattern, &spikes_data);
            total_accuracy += accuracy;
            
            println!("Test {:?}: {:.2}%", pattern, accuracy * 100.0);
        }
        
        Ok(total_accuracy / 6.0)
    }
    
    /// Get training summary
    pub fn get_summary(&self) -> TrainingSummary {
        let avg_accuracy = self.training_history.iter()
            .map(|r| r.accuracy)
            .sum::<f32>() / self.training_history.len().max(1) as f32;
        
        let avg_loss = self.training_history.iter()
            .map(|r| r.final_loss)
            .sum::<f32>() / self.training_history.len().max(1) as f32;
        
        TrainingSummary {
            patterns_trained: self.training_history.len(),
            average_accuracy: avg_accuracy,
            average_loss: avg_loss,
            best_pattern: self.training_history.iter()
                .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap())
                .map(|r| r.pattern),
            worst_pattern: self.training_history.iter()
                .min_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap())
                .map(|r| r.pattern),
        }
    }
}

/// Training summary
#[derive(Debug)]
pub struct TrainingSummary {
    pub patterns_trained: usize,
    pub average_accuracy: f32,
    pub average_loss: f32,
    pub best_pattern: Option<PatternType>,
    pub worst_pattern: Option<PatternType>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_generation() {
        let mut generator = PatternGenerator::new();
        
        let momentum_data = generator.generate_momentum(10, 1.0);
        assert_eq!(momentum_data.len(), 10);
        
        let reversal_data = generator.generate_reversal(20);
        assert_eq!(reversal_data.len(), 20);
        
        let breakout_data = generator.generate_breakout(15, 150.0);
        assert_eq!(breakout_data.len(), 15);
    }
    
    #[test]
    fn test_pattern_training() {
        let encoder_config = EncoderConfig {
            num_neurons: 100,
            encoding_schemes: vec![EncodingScheme::RateCoding],
            window_size_ms: 100,
        };
        
        let mut trainer = PatternTrainer::new(encoder_config, 100);
        
        // Train single pattern
        let record = trainer.train_pattern(PatternType::Momentum, 10).unwrap();
        assert!(record.accuracy >= 0.0);
        assert!(record.accuracy <= 1.0);
    }
    
    #[test]
    fn test_all_patterns() {
        let encoder_config = EncoderConfig {
            num_neurons: 100,
            encoding_schemes: vec![
                EncodingScheme::RateCoding,
                EncodingScheme::TemporalCoding,
            ],
            window_size_ms: 100,
        };
        
        let mut trainer = PatternTrainer::new(encoder_config, 200);
        
        // Train all patterns with fewer epochs for testing
        let records = trainer.train_all_patterns(5).unwrap();
        assert_eq!(records.len(), 6);
        
        // Test patterns
        let test_accuracy = trainer.test_patterns().unwrap();
        println!("Overall test accuracy: {:.2}%", test_accuracy * 100.0);
    }
}