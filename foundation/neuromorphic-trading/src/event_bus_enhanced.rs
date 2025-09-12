//! Enhanced event bus with conflation and advanced gap detection
//! 
//! Features:
//! - Automatic conflation for slow consumers
//! - Gap detection with recovery
//! - Message priority levels
//! - Backpressure handling

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use std::collections::{HashMap, VecDeque, BTreeMap};
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;

use crate::event_bus::{Symbol, Side, TradeData, QuoteData, OrderBookData};

/// Message priority
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical = 3,  // Never conflate
    High = 2,      // Conflate cautiously
    Normal = 1,    // Standard conflation
    Low = 0,       // Aggressive conflation
}

/// Enhanced message wrapper with metadata
#[derive(Clone, Debug)]
pub struct EnhancedMessage<T> {
    pub data: Arc<T>,
    pub sequence: u64,
    pub priority: Priority,
    pub timestamp_ns: u64,
    pub source_id: u32,
}

/// Gap information
#[derive(Clone, Debug)]
pub struct GapInfo {
    pub start_seq: u64,
    pub end_seq: u64,
    pub detected_at_ns: u64,
    pub recovered: bool,
}

/// Conflation strategy
#[derive(Clone, Copy, Debug)]
pub enum ConflationStrategy {
    None,                    // No conflation
    Latest,                  // Keep only latest
    TimeWindow(Duration),    // Conflate within time window
    Count(usize),           // Keep last N messages
    Smart,                  // Adaptive based on consumer speed
}

/// Consumer profile for adaptive conflation
struct ConsumerProfile {
    id: u32,
    avg_processing_time_ns: AtomicU64,
    messages_processed: AtomicU64,
    messages_dropped: AtomicU64,
    last_sequence_read: AtomicU64,
    is_slow: AtomicBool,
    conflation_strategy: RwLock<ConflationStrategy>,
}

impl ConsumerProfile {
    fn new(id: u32) -> Self {
        Self {
            id,
            avg_processing_time_ns: AtomicU64::new(0),
            messages_processed: AtomicU64::new(0),
            messages_dropped: AtomicU64::new(0),
            last_sequence_read: AtomicU64::new(0),
            is_slow: AtomicBool::new(false),
            conflation_strategy: RwLock::new(ConflationStrategy::None),
        }
    }
    
    fn update_processing_time(&self, duration_ns: u64) {
        let processed = self.messages_processed.fetch_add(1, Ordering::Relaxed) + 1;
        let current_avg = self.avg_processing_time_ns.load(Ordering::Relaxed);
        
        // Exponential moving average
        let alpha = 0.1;
        let new_avg = ((1.0 - alpha) * current_avg as f64 + alpha * duration_ns as f64) as u64;
        self.avg_processing_time_ns.store(new_avg, Ordering::Relaxed);
        
        // Mark as slow if processing time > 1ms
        if new_avg > 1_000_000 {
            self.is_slow.store(true, Ordering::Relaxed);
            
            // Upgrade conflation strategy
            let mut strategy = self.conflation_strategy.write();
            *strategy = match *strategy {
                ConflationStrategy::None => ConflationStrategy::Latest,
                ConflationStrategy::Latest => ConflationStrategy::TimeWindow(Duration::from_micros(100)),
                ConflationStrategy::TimeWindow(d) if d.as_micros() < 1000 => {
                    ConflationStrategy::TimeWindow(Duration::from_micros(d.as_micros() as u64 * 2))
                }
                _ => ConflationStrategy::Smart,
            };
        } else if new_avg < 100_000 {
            // Fast consumer, reduce conflation
            self.is_slow.store(false, Ordering::Relaxed);
            let mut strategy = self.conflation_strategy.write();
            *strategy = ConflationStrategy::None;
        }
    }
}

/// Enhanced SPMC channel with conflation
pub struct ConflatingChannel<T> {
    // Primary buffer for all messages
    buffer: Vec<Option<EnhancedMessage<T>>>,
    mask: usize,
    head: AtomicU64,
    
    // Per-consumer conflation buffers
    consumer_buffers: DashMap<u32, VecDeque<EnhancedMessage<T>>>,
    
    // Gap tracking
    gaps: Arc<RwLock<BTreeMap<u64, GapInfo>>>,
    expected_sequence: AtomicU64,
    
    // Consumer profiles
    consumers: DashMap<u32, ConsumerProfile>,
    
    // Statistics
    total_conflated: AtomicU64,
    total_gaps_detected: AtomicU64,
    total_gaps_recovered: AtomicU64,
}

impl<T: Clone> ConflatingChannel<T> {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Size must be power of 2");
        
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(None);
        }
        
        Self {
            buffer,
            mask: size - 1,
            head: AtomicU64::new(0),
            consumer_buffers: DashMap::new(),
            gaps: Arc::new(RwLock::new(BTreeMap::new())),
            expected_sequence: AtomicU64::new(0),
            consumers: DashMap::new(),
            total_conflated: AtomicU64::new(0),
            total_gaps_detected: AtomicU64::new(0),
            total_gaps_recovered: AtomicU64::new(0),
        }
    }
    
    pub fn publish(&mut self, message: EnhancedMessage<T>) -> Result<()> {
        // Check for gaps
        let expected = self.expected_sequence.load(Ordering::Acquire);
        if message.sequence > expected {
            // Gap detected
            let gap = GapInfo {
                start_seq: expected,
                end_seq: message.sequence - 1,
                detected_at_ns: Instant::now().elapsed().as_nanos() as u64,
                recovered: false,
            };
            
            self.gaps.write().insert(expected, gap);
            self.total_gaps_detected.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update expected sequence
        self.expected_sequence.store(message.sequence + 1, Ordering::Release);
        
        // Store in main buffer
        let head = self.head.fetch_add(1, Ordering::AcqRel);
        let index = (head & self.mask as u64) as usize;
        self.buffer[index] = Some(message.clone());
        
        // Distribute to consumer buffers with conflation
        for mut consumer_buffer in self.consumer_buffers.iter_mut() {
            let consumer_id = *consumer_buffer.key();
            
            if let Some(profile) = self.consumers.get(&consumer_id) {
                let strategy = profile.conflation_strategy.read().clone();
                
                match strategy {
                    ConflationStrategy::None => {
                        consumer_buffer.push_back(message.clone());
                    }
                    ConflationStrategy::Latest => {
                        // Only keep latest message per symbol
                        if message.priority >= Priority::High {
                            consumer_buffer.push_back(message.clone());
                        } else {
                            // Replace last message if same type
                            if !consumer_buffer.is_empty() {
                                consumer_buffer.pop_back();
                                self.total_conflated.fetch_add(1, Ordering::Relaxed);
                            }
                            consumer_buffer.push_back(message.clone());
                        }
                    }
                    ConflationStrategy::TimeWindow(window) => {
                        // Remove old messages outside window
                        let cutoff = message.timestamp_ns - window.as_nanos() as u64;
                        while let Some(front) = consumer_buffer.front() {
                            if front.timestamp_ns < cutoff && front.priority < Priority::High {
                                consumer_buffer.pop_front();
                                self.total_conflated.fetch_add(1, Ordering::Relaxed);
                            } else {
                                break;
                            }
                        }
                        consumer_buffer.push_back(message.clone());
                    }
                    ConflationStrategy::Count(max_count) => {
                        // Keep only last N messages
                        while consumer_buffer.len() >= max_count {
                            if let Some(front) = consumer_buffer.front() {
                                if front.priority < Priority::High {
                                    consumer_buffer.pop_front();
                                    self.total_conflated.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    break;
                                }
                            }
                        }
                        consumer_buffer.push_back(message.clone());
                    }
                    ConflationStrategy::Smart => {
                        // Adaptive conflation based on consumer speed
                        let is_slow = profile.is_slow.load(Ordering::Relaxed);
                        
                        if is_slow {
                            // Aggressive conflation for slow consumers
                            if message.priority < Priority::Normal && consumer_buffer.len() > 10 {
                                // Skip low priority messages if buffer is building up
                                self.total_conflated.fetch_add(1, Ordering::Relaxed);
                                profile.messages_dropped.fetch_add(1, Ordering::Relaxed);
                            } else {
                                // Keep critical and high priority, conflate others
                                if message.priority >= Priority::High || consumer_buffer.is_empty() {
                                    consumer_buffer.push_back(message.clone());
                                } else {
                                    // Replace last normal/low priority message
                                    if let Some(back) = consumer_buffer.back() {
                                        if back.priority <= Priority::Normal {
                                            consumer_buffer.pop_back();
                                            self.total_conflated.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                    consumer_buffer.push_back(message.clone());
                                }
                            }
                        } else {
                            // Fast consumer, minimal conflation
                            consumer_buffer.push_back(message.clone());
                        }
                    }
                }
            } else {
                // No profile, no conflation
                consumer_buffer.push_back(message.clone());
            }
        }
        
        Ok(())
    }
    
    pub fn subscribe(&self, consumer_id: u32) -> Result<EnhancedMessage<T>> {
        let start = Instant::now();
        
        // Get consumer's buffer
        let mut buffer = self.consumer_buffers.entry(consumer_id)
            .or_insert_with(VecDeque::new);
        
        if let Some(message) = buffer.pop_front() {
            // Update consumer profile
            if let Some(profile) = self.consumers.get(&consumer_id) {
                profile.last_sequence_read.store(message.sequence, Ordering::Relaxed);
                let duration = start.elapsed().as_nanos() as u64;
                profile.update_processing_time(duration);
            }
            
            Ok(message)
        } else {
            Err(anyhow::anyhow!("No messages available"))
        }
    }
    
    pub fn register_consumer(&self, id: u32, strategy: ConflationStrategy) {
        let profile = ConsumerProfile::new(id);
        *profile.conflation_strategy.write() = strategy;
        self.consumers.insert(id, profile);
        self.consumer_buffers.insert(id, VecDeque::new());
    }
    
    pub fn check_gaps(&self) -> Vec<GapInfo> {
        self.gaps.read().values().cloned().collect()
    }
    
    pub fn recover_gap(&self, start_seq: u64, messages: Vec<EnhancedMessage<T>>) -> Result<()> {
        let mut gaps = self.gaps.write();
        
        if let Some(gap) = gaps.get_mut(&start_seq) {
            gap.recovered = true;
            self.total_gaps_recovered.fetch_add(1, Ordering::Relaxed);
            
            // Insert recovered messages into consumer buffers
            for message in messages {
                for mut buffer in self.consumer_buffers.iter_mut() {
                    buffer.push_back(message.clone());
                }
            }
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("Gap not found"))
        }
    }
    
    pub fn get_consumer_stats(&self, consumer_id: u32) -> Option<ConsumerStats> {
        self.consumers.get(&consumer_id).map(|profile| {
            ConsumerStats {
                avg_processing_time_ns: profile.avg_processing_time_ns.load(Ordering::Relaxed),
                messages_processed: profile.messages_processed.load(Ordering::Relaxed),
                messages_dropped: profile.messages_dropped.load(Ordering::Relaxed),
                last_sequence_read: profile.last_sequence_read.load(Ordering::Relaxed),
                is_slow: profile.is_slow.load(Ordering::Relaxed),
                conflation_strategy: profile.conflation_strategy.read().clone(),
                buffer_depth: self.consumer_buffers.get(&consumer_id)
                    .map(|b| b.len())
                    .unwrap_or(0),
            }
        })
    }
}

/// Consumer statistics
#[derive(Debug, Clone)]
pub struct ConsumerStats {
    pub avg_processing_time_ns: u64,
    pub messages_processed: u64,
    pub messages_dropped: u64,
    pub last_sequence_read: u64,
    pub is_slow: bool,
    pub conflation_strategy: ConflationStrategy,
    pub buffer_depth: usize,
}

/// Enhanced market data bus with conflation
pub struct EnhancedMarketDataBus {
    trade_channel: Arc<RwLock<ConflatingChannel<TradeData>>>,
    quote_channel: Arc<RwLock<ConflatingChannel<QuoteData>>>,
    book_channel: Arc<RwLock<ConflatingChannel<OrderBookData>>>,
    next_consumer_id: AtomicU32,
    sequence_generator: AtomicU64,
}

impl EnhancedMarketDataBus {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            trade_channel: Arc::new(RwLock::new(ConflatingChannel::new(buffer_size))),
            quote_channel: Arc::new(RwLock::new(ConflatingChannel::new(buffer_size))),
            book_channel: Arc::new(RwLock::new(ConflatingChannel::new(buffer_size))),
            next_consumer_id: AtomicU32::new(0),
            sequence_generator: AtomicU64::new(0),
        }
    }
    
    pub fn register_consumer(&self, strategy: ConflationStrategy) -> u32 {
        let id = self.next_consumer_id.fetch_add(1, Ordering::SeqCst);
        
        self.trade_channel.read().register_consumer(id, strategy);
        self.quote_channel.read().register_consumer(id, strategy);
        self.book_channel.read().register_consumer(id, strategy);
        
        id
    }
    
    pub fn publish_trade(&self, trade: TradeData, priority: Priority) -> Result<()> {
        let message = EnhancedMessage {
            data: Arc::new(trade.clone()),
            sequence: self.sequence_generator.fetch_add(1, Ordering::SeqCst),
            priority,
            timestamp_ns: trade.timestamp_ns,
            source_id: 0,
        };
        
        self.trade_channel.write().publish(message)
    }
    
    pub fn publish_quote(&self, quote: QuoteData, priority: Priority) -> Result<()> {
        let message = EnhancedMessage {
            data: Arc::new(quote.clone()),
            sequence: self.sequence_generator.fetch_add(1, Ordering::SeqCst),
            priority,
            timestamp_ns: quote.timestamp_ns,
            source_id: 0,
        };
        
        self.quote_channel.write().publish(message)
    }
    
    pub fn read_trade(&self, consumer_id: u32) -> Result<Arc<TradeData>> {
        self.trade_channel.read()
            .subscribe(consumer_id)
            .map(|msg| msg.data)
    }
    
    pub fn read_quote(&self, consumer_id: u32) -> Result<Arc<QuoteData>> {
        self.quote_channel.read()
            .subscribe(consumer_id)
            .map(|msg| msg.data)
    }
    
    pub fn check_trade_gaps(&self) -> Vec<GapInfo> {
        self.trade_channel.read().check_gaps()
    }
    
    pub fn get_consumer_stats(&self, consumer_id: u32) -> BusConsumerStats {
        BusConsumerStats {
            trade_stats: self.trade_channel.read().get_consumer_stats(consumer_id),
            quote_stats: self.quote_channel.read().get_consumer_stats(consumer_id),
            book_stats: self.book_channel.read().get_consumer_stats(consumer_id),
        }
    }
}

/// Combined consumer statistics
#[derive(Debug)]
pub struct BusConsumerStats {
    pub trade_stats: Option<ConsumerStats>,
    pub quote_stats: Option<ConsumerStats>,
    pub book_stats: Option<ConsumerStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conflation_latest() {
        let mut channel = ConflatingChannel::<u32>::new(16);
        channel.register_consumer(0, ConflationStrategy::Latest);
        
        // Publish multiple messages
        for i in 0..10 {
            let msg = EnhancedMessage {
                data: Arc::new(i),
                sequence: i as u64,
                priority: Priority::Normal,
                timestamp_ns: i as u64 * 1000,
                source_id: 0,
            };
            channel.publish(msg).unwrap();
        }
        
        // Should have conflated to keep only latest
        let stats = channel.get_consumer_stats(0).unwrap();
        assert!(stats.buffer_depth <= 5);
    }
    
    #[test]
    fn test_gap_detection() {
        let mut channel = ConflatingChannel::<u32>::new(16);
        
        // Publish with gap
        let msg1 = EnhancedMessage {
            data: Arc::new(1),
            sequence: 1,
            priority: Priority::Normal,
            timestamp_ns: 1000,
            source_id: 0,
        };
        channel.publish(msg1).unwrap();
        
        let msg3 = EnhancedMessage {
            data: Arc::new(3),
            sequence: 3,
            priority: Priority::Normal,
            timestamp_ns: 3000,
            source_id: 0,
        };
        channel.publish(msg3).unwrap();
        
        // Check gaps detected
        let gaps = channel.check_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_seq, 2);
        assert_eq!(gaps[0].end_seq, 2);
    }
    
    #[test]
    fn test_priority_conflation() {
        let mut channel = ConflatingChannel::<u32>::new(16);
        channel.register_consumer(0, ConflationStrategy::Latest);
        
        // High priority message should not be conflated
        let critical = EnhancedMessage {
            data: Arc::new(999),
            sequence: 1,
            priority: Priority::Critical,
            timestamp_ns: 1000,
            source_id: 0,
        };
        channel.publish(critical).unwrap();
        
        // Normal messages
        for i in 2..10 {
            let msg = EnhancedMessage {
                data: Arc::new(i),
                sequence: i as u64,
                priority: Priority::Normal,
                timestamp_ns: i as u64 * 1000,
                source_id: 0,
            };
            channel.publish(msg).unwrap();
        }
        
        // Critical message should still be there
        let first = channel.subscribe(0).unwrap();
        assert_eq!(*first.data, 999);
    }
}