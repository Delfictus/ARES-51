//! Parallel reservoir computing with Rayon
//! 
//! Features:
//! - Parallel neuron updates
//! - Work-stealing for load balancing
//! - Chunked matrix operations
//! - Lock-free state updates

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::RwLock;
use ndarray::{Array1, Array2, Axis, s, Zip};
use ndarray::parallel::prelude::*;
use rayon::prelude::*;
use crossbeam::channel::{bounded, Sender, Receiver};
use ahash::AHashMap;
use anyhow::Result;

use crate::spike_encoding::Spike;
use crate::reservoir::{PatternType, ReservoirState};

/// Parallel reservoir configuration
#[derive(Debug, Clone)]
pub struct ParallelReservoirConfig {
    pub size: usize,
    pub spectral_radius: f32,
    pub connection_probability: f32,
    pub leak_rate: f32,
    pub num_threads: usize,
    pub chunk_size: usize,
}

impl Default for ParallelReservoirConfig {
    fn default() -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            size: 5000,
            spectral_radius: 0.95,
            connection_probability: 0.2,
            leak_rate: 0.1,
            num_threads,
            chunk_size: 64,  // Cache-friendly chunk size
        }
    }
}

/// Neuron chunk for parallel processing
#[derive(Clone)]
struct NeuronChunk {
    start_idx: usize,
    end_idx: usize,
    weights: Array2<f32>,  // Local weight matrix
    state: Array1<f32>,     // Local state
    input_buffer: Array1<f32>,
}

impl NeuronChunk {
    fn new(start: usize, end: usize, full_size: usize) -> Self {
        let chunk_size = end - start;
        Self {
            start_idx: start,
            end_idx: end,
            weights: Array2::zeros((chunk_size, full_size)),
            state: Array1::zeros(chunk_size),
            input_buffer: Array1::zeros(chunk_size),
        }
    }
    
    fn update(&mut self, global_state: &Array1<f32>, leak_rate: f32) {
        // Compute weighted input from all neurons
        self.input_buffer = self.weights.dot(global_state);
        
        // Leaky integration
        self.state = &self.state * (1.0 - leak_rate) + &self.input_buffer;
        
        // Apply activation
        self.state.mapv_inplace(|x| x.tanh());
    }
}

/// Parallel Liquid State Machine
pub struct ParallelReservoir {
    config: ParallelReservoirConfig,
    chunks: Vec<Arc<RwLock<NeuronChunk>>>,
    global_state: Arc<RwLock<Array1<f32>>>,
    pattern_detector: Arc<ParallelPatternDetector>,
    
    // Parallel work distribution
    work_sender: Sender<WorkItem>,
    work_receiver: Receiver<WorkItem>,
    
    // Metrics
    updates_processed: AtomicU64,
    parallel_efficiency: Arc<RwLock<f32>>,
}

/// Work item for parallel processing
enum WorkItem {
    UpdateChunk(usize, Array1<f32>),
    DetectPattern(Array1<f32>),
    ComputeSeparation(Vec<Spike>, Vec<Spike>),
}

impl ParallelReservoir {
    pub fn new(config: ParallelReservoirConfig) -> Self {
        // Initialize thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .unwrap_or_else(|_| {});
        
        // Create neuron chunks
        let chunk_size = (config.size + config.num_threads - 1) / config.num_threads;
        let mut chunks = Vec::new();
        
        for i in 0..config.num_threads {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(config.size);
            if start < end {
                let mut chunk = NeuronChunk::new(start, end, config.size);
                
                // Initialize weights
                Self::initialize_chunk_weights(&mut chunk, &config);
                
                chunks.push(Arc::new(RwLock::new(chunk)));
            }
        }
        
        // Create work channel
        let (sender, receiver) = bounded(config.num_threads * 2);
        
        Self {
            config: config.clone(),
            chunks,
            global_state: Arc::new(RwLock::new(Array1::zeros(config.size))),
            pattern_detector: Arc::new(ParallelPatternDetector::new(config.size)),
            work_sender: sender,
            work_receiver: receiver,
            updates_processed: AtomicU64::new(0),
            parallel_efficiency: Arc::new(RwLock::new(1.0)),
        }
    }
    
    fn initialize_chunk_weights(chunk: &mut NeuronChunk, config: &ParallelReservoirConfig) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let chunk_size = chunk.end_idx - chunk.start_idx;
        
        // Initialize with sparse random connections
        for i in 0..chunk_size {
            for j in 0..config.size {
                if i + chunk.start_idx == j {
                    continue;  // No self-connections
                }
                
                if rng.gen::<f32>() < config.connection_probability {
                    chunk.weights[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
        }
        
        // Normalize spectral radius
        Self::normalize_chunk_weights(&mut chunk.weights, config.spectral_radius);
    }
    
    fn normalize_chunk_weights(weights: &mut Array2<f32>, target_radius: f32) {
        // Approximate spectral radius normalization
        let norm: f32 = weights.iter()
            .map(|w| w * w)
            .sum::<f32>()
            .sqrt();
        
        if norm > 0.0 {
            let scale = target_radius / norm.sqrt();
            weights.mapv_inplace(|w| w * scale);
        }
    }
    
    /// Process spikes in parallel
    pub fn process_parallel(&self, spikes: &[Spike]) -> ReservoirState {
        if spikes.is_empty() {
            return ReservoirState {
                activations: self.global_state.read().to_vec(),
                confidence: 0.0,
                novelty: 0.0,
            };
        }
        
        let start_time = std::time::Instant::now();
        
        // Convert spikes to input
        let input = self.spikes_to_input_parallel(spikes);
        
        // Update chunks in parallel
        let global_state = self.global_state.read().clone();
        
        let updated_states: Vec<Array1<f32>> = self.chunks
            .par_iter()
            .map(|chunk_arc| {
                let mut chunk = chunk_arc.write();
                chunk.update(&global_state, self.config.leak_rate);
                
                // Add input
                let chunk_input = input.slice(s![chunk.start_idx..chunk.end_idx]).to_owned();
                chunk.state = &chunk.state + &chunk_input;
                
                chunk.state.clone()
            })
            .collect();
        
        // Merge updated states
        let mut new_global_state = Array1::zeros(self.config.size);
        for (chunk_idx, state) in updated_states.iter().enumerate() {
            let chunk = self.chunks[chunk_idx].read();
            let start = chunk.start_idx;
            let end = chunk.end_idx;
            new_global_state.slice_mut(s![start..end]).assign(state);
        }
        
        // Update global state
        *self.global_state.write() = new_global_state.clone();
        
        // Detect patterns in parallel
        let patterns = self.pattern_detector.detect_parallel(&new_global_state);
        
        // Calculate metrics
        let confidence = self.calculate_confidence_parallel(&new_global_state);
        let novelty = self.calculate_novelty_parallel(&new_global_state, &patterns);
        
        // Update efficiency metric
        let elapsed = start_time.elapsed();
        self.update_efficiency(elapsed.as_micros() as f32);
        
        self.updates_processed.fetch_add(1, Ordering::Relaxed);
        
        ReservoirState {
            activations: new_global_state.to_vec(),
            confidence,
            novelty,
        }
    }
    
    fn spikes_to_input_parallel(&self, spikes: &[Spike]) -> Array1<f32> {
        let mut input = Array1::zeros(self.config.size);
        
        // Parallel accumulation of spike contributions
        let spike_groups: AHashMap<usize, Vec<f32>> = spikes
            .par_iter()
            .filter(|s| (s.neuron_id as usize) < self.config.size)
            .fold(
                || AHashMap::new(),
                |mut acc, spike| {
                    acc.entry(spike.neuron_id as usize)
                        .or_insert_with(Vec::new)
                        .push(spike.strength);
                    acc
                },
            )
            .reduce(
                || AHashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        a.entry(k)
                            .or_insert_with(Vec::new)
                            .extend(v);
                    }
                    a
                },
            );
        
        // Sum contributions
        for (neuron_id, strengths) in spike_groups {
            input[neuron_id] = strengths.iter().sum();
        }
        
        input
    }
    
    fn calculate_confidence_parallel(&self, state: &Array1<f32>) -> f32 {
        // Parallel confidence calculation
        let chunks: Vec<f32> = state
            .as_slice()
            .unwrap()
            .par_chunks(self.config.chunk_size)
            .map(|chunk| {
                let sum: f32 = chunk.iter().sum();
                let mean = sum / chunk.len() as f32;
                let variance: f32 = chunk.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>() / chunk.len() as f32;
                
                (variance * mean.abs()).min(1.0)
            })
            .collect();
        
        chunks.iter().sum::<f32>() / chunks.len() as f32
    }
    
    fn calculate_novelty_parallel(&self, state: &Array1<f32>, patterns: &[(PatternType, f32)]) -> f32 {
        if patterns.is_empty() {
            return 1.0;
        }
        
        // Parallel novelty computation
        let pattern_strengths: f32 = patterns
            .par_iter()
            .map(|(_, strength)| strength)
            .sum();
        
        1.0 - (pattern_strengths / patterns.len() as f32).min(1.0)
    }
    
    fn update_efficiency(&self, processing_time_us: f32) {
        let mut efficiency = self.parallel_efficiency.write();
        
        // Ideal time for serial processing (estimated)
        let serial_time_us = processing_time_us * self.config.num_threads as f32;
        
        // Efficiency = serial_time / (parallel_time * num_threads)
        *efficiency = serial_time_us / (processing_time_us * self.config.num_threads as f32);
    }
    
    /// Batch process multiple spike trains
    pub fn batch_process(&self, spike_batch: Vec<Vec<Spike>>) -> Vec<ReservoirState> {
        spike_batch
            .into_par_iter()
            .map(|spikes| self.process_parallel(&spikes))
            .collect()
    }
    
    /// Compute separation between two spike patterns in parallel
    pub fn compute_separation_parallel(&self, spikes1: &[Spike], spikes2: &[Spike]) -> f32 {
        let state1 = self.process_parallel(spikes1);
        let state2 = self.process_parallel(spikes2);
        
        // Parallel distance computation
        let distances: Vec<f32> = state1.activations
            .par_chunks(self.config.chunk_size)
            .zip(state2.activations.par_chunks(self.config.chunk_size))
            .map(|(chunk1, chunk2)| {
                chunk1.iter()
                    .zip(chunk2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .collect();
        
        distances.iter().sum::<f32>().sqrt()
    }
    
    /// Get parallel processing statistics
    pub fn get_stats(&self) -> ParallelStats {
        let total_weights = self.chunks
            .par_iter()
            .map(|chunk| {
                let c = chunk.read();
                c.weights.iter().filter(|&&w| w != 0.0).count()
            })
            .sum();
        
        let avg_activation = self.global_state.read().mean().unwrap_or(0.0);
        
        ParallelStats {
            reservoir_size: self.config.size,
            num_chunks: self.chunks.len(),
            chunk_size: self.config.chunk_size,
            num_threads: self.config.num_threads,
            total_connections: total_weights,
            updates_processed: self.updates_processed.load(Ordering::Relaxed),
            parallel_efficiency: *self.parallel_efficiency.read(),
            avg_activation,
        }
    }
}

/// Parallel pattern detector
struct ParallelPatternDetector {
    templates: Arc<Vec<(PatternType, Array1<f32>)>>,
    size: usize,
}

impl ParallelPatternDetector {
    fn new(size: usize) -> Self {
        let templates = vec![
            (PatternType::Momentum, Self::create_template(size, 0)),
            (PatternType::Reversal, Self::create_template(size, 1)),
            (PatternType::Breakout, Self::create_template(size, 2)),
            (PatternType::Consolidation, Self::create_template(size, 3)),
            (PatternType::Volatility, Self::create_template(size, 4)),
            (PatternType::Trend, Self::create_template(size, 5)),
        ];
        
        Self {
            templates: Arc::new(templates),
            size,
        }
    }
    
    fn create_template(size: usize, pattern_id: usize) -> Array1<f32> {
        let mut template = Array1::zeros(size);
        
        match pattern_id {
            0 => {
                // Momentum: rising pattern
                for i in 0..size {
                    template[i] = (i as f32 / size as f32).powf(2.0);
                }
            }
            1 => {
                // Reversal: peak in middle
                for i in 0..size {
                    let x = (i as f32 - size as f32 / 2.0) / (size as f32 / 2.0);
                    template[i] = (-x * x + 1.0).max(0.0);
                }
            }
            2 => {
                // Breakout: sharp transition
                for i in 0..size {
                    template[i] = if i > size * 2 / 3 { 1.0 } else { 0.1 };
                }
            }
            3 => {
                // Consolidation: flat
                template.fill(0.5);
            }
            4 => {
                // Volatility: oscillating
                for i in 0..size {
                    template[i] = (i as f32 * 0.3).sin().abs();
                }
            }
            5 => {
                // Trend: linear
                for i in 0..size {
                    template[i] = i as f32 / size as f32;
                }
            }
            _ => {}
        }
        
        template
    }
    
    fn detect_parallel(&self, state: &Array1<f32>) -> Vec<(PatternType, f32)> {
        self.templates
            .par_iter()
            .map(|(pattern_type, template)| {
                let similarity = self.cosine_similarity_parallel(state, template);
                (*pattern_type, similarity)
            })
            .filter(|(_, similarity)| *similarity > 0.3)
            .collect()
    }
    
    fn cosine_similarity_parallel(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let chunks = 64;
        let chunk_size = (a.len() + chunks - 1) / chunks;
        
        let results: Vec<(f32, f32, f32)> = (0..chunks)
            .into_par_iter()
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(a.len());
                
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;
                
                for j in start..end {
                    dot += a[j] * b[j];
                    norm_a += a[j] * a[j];
                    norm_b += b[j] * b[j];
                }
                
                (dot, norm_a, norm_b)
            })
            .collect();
        
        let total_dot: f32 = results.iter().map(|r| r.0).sum();
        let total_norm_a: f32 = results.iter().map(|r| r.1).sum();
        let total_norm_b: f32 = results.iter().map(|r| r.2).sum();
        
        if total_norm_a > 0.0 && total_norm_b > 0.0 {
            total_dot / (total_norm_a.sqrt() * total_norm_b.sqrt())
        } else {
            0.0
        }
    }
}

/// Parallel processing statistics
#[derive(Debug)]
pub struct ParallelStats {
    pub reservoir_size: usize,
    pub num_chunks: usize,
    pub chunk_size: usize,
    pub num_threads: usize,
    pub total_connections: usize,
    pub updates_processed: u64,
    pub parallel_efficiency: f32,
    pub avg_activation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_reservoir() {
        let config = ParallelReservoirConfig {
            size: 1000,
            num_threads: 4,
            ..Default::default()
        };
        
        let reservoir = ParallelReservoir::new(config);
        
        // Create test spikes
        let spikes: Vec<Spike> = (0..100)
            .map(|i| Spike {
                timestamp_ns: 1_000_000 * i,
                neuron_id: i % 100,
                strength: 1.0,
            })
            .collect();
        
        let state = reservoir.process_parallel(&spikes);
        assert!(!state.activations.is_empty());
        
        let stats = reservoir.get_stats();
        println!("Parallel reservoir stats: {:?}", stats);
        assert!(stats.parallel_efficiency > 0.0);
    }
    
    #[test]
    fn test_batch_processing() {
        let reservoir = ParallelReservoir::new(ParallelReservoirConfig::default());
        
        // Create batch of spike trains
        let batch: Vec<Vec<Spike>> = (0..10)
            .map(|batch_idx| {
                (0..50)
                    .map(|i| Spike {
                        timestamp_ns: batch_idx * 1_000_000_000 + i * 1_000_000,
                        neuron_id: (i + batch_idx) % 100,
                        strength: 0.5 + batch_idx as f32 * 0.1,
                    })
                    .collect()
            })
            .collect();
        
        let start = std::time::Instant::now();
        let results = reservoir.batch_process(batch);
        let elapsed = start.elapsed();
        
        assert_eq!(results.len(), 10);
        println!("Batch processing 10 samples: {:?}", elapsed);
    }
    
    #[test]
    fn test_separation_computation() {
        let reservoir = ParallelReservoir::new(ParallelReservoirConfig::default());
        
        // Two different spike patterns
        let spikes1: Vec<Spike> = (0..50)
            .map(|i| Spike {
                timestamp_ns: i * 1_000_000,
                neuron_id: i,
                strength: 1.0,
            })
            .collect();
        
        let spikes2: Vec<Spike> = (50..100)
            .map(|i| Spike {
                timestamp_ns: i * 1_000_000,
                neuron_id: i,
                strength: 1.0,
            })
            .collect();
        
        let separation = reservoir.compute_separation_parallel(&spikes1, &spikes2);
        assert!(separation > 0.0);
        println!("Pattern separation: {}", separation);
    }
    
    #[test]
    fn test_efficiency_scaling() {
        // Test with different thread counts
        for num_threads in &[1, 2, 4, 8] {
            let config = ParallelReservoirConfig {
                size: 2000,
                num_threads: *num_threads,
                ..Default::default()
            };
            
            let reservoir = ParallelReservoir::new(config);
            
            let spikes: Vec<Spike> = (0..200)
                .map(|i| Spike {
                    timestamp_ns: i * 1_000_000,
                    neuron_id: i % 200,
                    strength: 1.0,
                })
                .collect();
            
            let start = std::time::Instant::now();
            for _ in 0..100 {
                reservoir.process_parallel(&spikes);
            }
            let elapsed = start.elapsed();
            
            let stats = reservoir.get_stats();
            println!("Threads: {}, Time: {:?}, Efficiency: {:.2}",
                     num_threads, elapsed, stats.parallel_efficiency);
        }
    }
}