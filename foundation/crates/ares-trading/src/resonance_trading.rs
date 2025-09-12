// Quantum Resonance Trading System
// Integrates DRPP (Dynamic Resonance Phase Processing) for market analysis

use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use rustfft::{FftPlanner, Fft};

use crate::market_data::{Quote, Candle};

/// Quantum Resonance Market Analyzer
/// Uses phase correlations and oscillator networks to detect market patterns
pub struct ResonanceMarketAnalyzer {
    /// Network of market oscillators (one per symbol)
    market_oscillators: HashMap<String, MarketOscillator>,
    
    /// Cross-market correlation matrix
    correlation_matrix: Arc<RwLock<DMatrix<f64>>>,
    
    /// Resonance detection parameters
    params: ResonanceParameters,
    
    /// Pattern memory for temporal correlation
    pattern_memory: PatternMemory,
    
    /// FFT planner for frequency analysis
    fft_planner: Arc<RwLock<FftPlanner<f64>>>,
}

#[derive(Debug, Clone)]
pub struct ResonanceParameters {
    /// Minimum resonance strength to trigger signal (0.0 - 1.0)
    pub min_resonance_strength: f64,
    
    /// Phase coherence threshold for pattern detection
    pub coherence_threshold: f64,
    
    /// Temporal window for analysis (in candles)
    pub temporal_window: usize,
    
    /// Frequency bands for decomposition
    pub frequency_bands: Vec<(f64, f64)>,
    
    /// Coupling strength between correlated assets
    pub coupling_strength: f64,
    
    /// Nonlinearity factor for phase dynamics
    pub nonlinearity: f64,
}

impl Default for ResonanceParameters {
    fn default() -> Self {
        Self {
            min_resonance_strength: 0.75,
            coherence_threshold: 0.8,
            temporal_window: 100,
            frequency_bands: vec![
                (0.001, 0.01),  // Ultra-low frequency (trend)
                (0.01, 0.1),    // Low frequency (swing)
                (0.1, 0.5),     // Medium frequency (day trading)
                (0.5, 1.0),     // High frequency (scalping)
            ],
            coupling_strength: 0.15,
            nonlinearity: 0.05,
        }
    }
}

/// Market-specific oscillator that tracks price dynamics
pub struct MarketOscillator {
    pub symbol: String,
    
    /// Phase and amplitude of price oscillation
    pub phase: f64,
    pub amplitude: f64,
    pub frequency: f64,
    
    /// Price history for phase extraction
    pub price_history: VecDeque<f64>,
    pub volume_history: VecDeque<f64>,
    
    /// Phase velocity and acceleration
    pub phase_velocity: f64,
    pub phase_acceleration: f64,
    
    /// Resonance strength with other oscillators
    pub resonance_map: HashMap<String, f64>,
    
    /// Spectral components
    pub spectral_power: Vec<f64>,
    
    /// Hilbert transform for instantaneous phase
    pub hilbert_phase: f64,
}

impl MarketOscillator {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            phase: 0.0,
            amplitude: 1.0,
            frequency: 0.0,
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            phase_velocity: 0.0,
            phase_acceleration: 0.0,
            resonance_map: HashMap::new(),
            spectral_power: Vec::new(),
            hilbert_phase: 0.0,
        }
    }
    
    /// Update oscillator with new price data
    pub fn update(&mut self, price: f64, volume: f64) {
        self.price_history.push_back(price);
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        
        self.volume_history.push_back(volume);
        if self.volume_history.len() > 1000 {
            self.volume_history.pop_front();
        }
        
        // Calculate instantaneous phase using Hilbert transform approximation
        if self.price_history.len() >= 4 {
            let prices: Vec<f64> = self.price_history.iter().copied().collect();
            self.hilbert_phase = self.calculate_hilbert_phase(&prices);
            
            // Update phase dynamics
            let new_phase = self.hilbert_phase;
            let dt = 1.0; // Unit time step
            
            self.phase_acceleration = (new_phase - self.phase - self.phase_velocity * dt) / (dt * dt);
            self.phase_velocity += self.phase_acceleration * dt;
            self.phase = new_phase;
            
            // Calculate amplitude from price volatility
            let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
            let variance = prices.iter().map(|p| (p - mean_price).powi(2)).sum::<f64>() / prices.len() as f64;
            self.amplitude = variance.sqrt() / mean_price;
            
            // Estimate frequency from phase velocity
            self.frequency = self.phase_velocity.abs() / (2.0 * std::f64::consts::PI);
        }
    }
    
    /// Calculate Hilbert phase using discrete Hilbert transform
    fn calculate_hilbert_phase(&self, signal: &[f64]) -> f64 {
        if signal.len() < 4 {
            return 0.0;
        }
        
        // Simple Hilbert transform approximation using finite differences
        let n = signal.len();
        let mut hilbert = vec![0.0; n];
        
        for i in 1..n-1 {
            hilbert[i] = (signal[i+1] - signal[i-1]) / 2.0;
        }
        
        // Calculate instantaneous phase
        let last_idx = n - 1;
        let real_part = signal[last_idx];
        let imag_part = hilbert[last_idx];
        
        imag_part.atan2(real_part)
    }
}

/// Pattern memory for storing and recognizing resonance patterns
pub struct PatternMemory {
    /// Stored resonance patterns
    patterns: Vec<ResonancePattern>,
    
    /// Pattern matching threshold
    match_threshold: f64,
    
    /// Maximum patterns to store
    max_patterns: usize,
}

#[derive(Clone, Debug)]
pub struct ResonancePattern {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub symbols: Vec<String>,
    pub phase_signature: Vec<f64>,
    pub frequency_signature: Vec<f64>,
    pub strength: f64,
    pub outcome: PatternOutcome,
}

#[derive(Clone, Debug)]
pub enum PatternOutcome {
    Bullish { magnitude: f64 },
    Bearish { magnitude: f64 },
    Neutral,
    Unknown,
}

/// Trading signal generated from resonance analysis
#[derive(Debug, Clone)]
pub struct ResonanceSignal {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub signal_type: ResonanceSignalType,
    pub strength: f64,
    pub confidence: f64,
    pub correlated_symbols: Vec<String>,
    pub frequency_band: (f64, f64),
    pub phase_alignment: f64,
}

#[derive(Debug, Clone)]
pub enum ResonanceSignalType {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
    PhaseTransition,
    ResonanceBreakout,
}

impl ResonanceMarketAnalyzer {
    pub fn new(params: ResonanceParameters) -> Self {
        Self {
            market_oscillators: HashMap::new(),
            correlation_matrix: Arc::new(RwLock::new(DMatrix::zeros(0, 0))),
            params,
            pattern_memory: PatternMemory {
                patterns: Vec::new(),
                match_threshold: 0.85,
                max_patterns: 1000,
            },
            fft_planner: Arc::new(RwLock::new(FftPlanner::new())),
        }
    }
    
    /// Analyze market data for resonance patterns
    pub async fn analyze(&mut self, quotes: &HashMap<String, Quote>) -> Vec<ResonanceSignal> {
        let mut signals = Vec::new();
        
        // Update oscillators with new price data
        for (symbol, quote) in quotes {
            let oscillator = self.market_oscillators.entry(symbol.clone())
                .or_insert_with(|| MarketOscillator::new(symbol.clone()));
            
            oscillator.update(
                quote.last.to_f64().unwrap_or(0.0),
                quote.volume.to_f64().unwrap_or(0.0)
            );
        }
        
        // Calculate cross-market correlations
        self.update_correlation_matrix().await;
        
        // Detect resonance patterns
        let resonances = self.detect_resonances().await;
        
        // Generate trading signals from resonances
        for resonance in resonances {
            if resonance.strength >= self.params.min_resonance_strength {
                signals.push(self.generate_signal_from_resonance(resonance));
            }
        }
        
        signals
    }
    
    /// Update correlation matrix using phase synchronization
    async fn update_correlation_matrix(&self) {
        let symbols: Vec<String> = self.market_oscillators.keys().cloned().collect();
        let n = symbols.len();
        
        if n == 0 {
            return;
        }
        
        let mut matrix = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let osc1 = &self.market_oscillators[&symbols[i]];
                    let osc2 = &self.market_oscillators[&symbols[j]];
                    
                    // Calculate phase synchronization index
                    let phase_diff = (osc1.phase - osc2.phase).abs();
                    let sync_index = (1.0 - (phase_diff / std::f64::consts::PI).min(1.0)) 
                        * osc1.amplitude.min(osc2.amplitude);
                    
                    matrix[(i, j)] = sync_index;
                }
            }
        }
        
        let mut correlation = self.correlation_matrix.write().await;
        *correlation = matrix;
    }
    
    /// Detect resonance patterns in the oscillator network
    async fn detect_resonances(&self) -> Vec<ResonanceDetection> {
        let mut detections = Vec::new();
        
        for (symbol, oscillator) in &self.market_oscillators {
            // Check for resonance with other oscillators
            let mut resonating_symbols = Vec::new();
            let mut total_resonance = 0.0;
            
            for (other_symbol, other_osc) in &self.market_oscillators {
                if symbol != other_symbol {
                    // Calculate resonance strength using phase coherence
                    let phase_coherence = self.calculate_phase_coherence(oscillator, other_osc);
                    
                    if phase_coherence > self.params.coherence_threshold {
                        resonating_symbols.push(other_symbol.clone());
                        total_resonance += phase_coherence;
                    }
                }
            }
            
            if !resonating_symbols.is_empty() {
                let avg_resonance = total_resonance / resonating_symbols.len() as f64;
                
                detections.push(ResonanceDetection {
                    primary_symbol: symbol.clone(),
                    resonating_symbols,
                    strength: avg_resonance,
                    phase: oscillator.phase,
                    frequency: oscillator.frequency,
                    timestamp: Utc::now(),
                });
            }
        }
        
        detections
    }
    
    /// Calculate phase coherence between two oscillators
    fn calculate_phase_coherence(&self, osc1: &MarketOscillator, osc2: &MarketOscillator) -> f64 {
        // Phase locking value (PLV) calculation
        let min_len = osc1.price_history.len().min(osc2.price_history.len());
        
        if min_len < 10 {
            return 0.0;
        }
        
        let mut phase_diffs = Vec::new();
        
        for i in 0..min_len {
            let phase_diff = (osc1.phase - osc2.phase + 
                             i as f64 * 0.01 * (osc1.frequency - osc2.frequency))
                             .rem_euclid(2.0 * std::f64::consts::PI);
            phase_diffs.push(phase_diff);
        }
        
        // Calculate circular variance
        let mean_cos: f64 = phase_diffs.iter().map(|p| p.cos()).sum::<f64>() / phase_diffs.len() as f64;
        let mean_sin: f64 = phase_diffs.iter().map(|p| p.sin()).sum::<f64>() / phase_diffs.len() as f64;
        
        (mean_cos.powi(2) + mean_sin.powi(2)).sqrt()
    }
    
    /// Generate trading signal from resonance detection
    fn generate_signal_from_resonance(&self, detection: ResonanceDetection) -> ResonanceSignal {
        // Determine signal type based on phase and frequency
        let signal_type = if detection.phase > 0.0 && detection.frequency > 0.05 {
            if detection.strength > 0.9 {
                ResonanceSignalType::StrongBuy
            } else {
                ResonanceSignalType::Buy
            }
        } else if detection.phase < 0.0 && detection.frequency > 0.05 {
            if detection.strength > 0.9 {
                ResonanceSignalType::StrongSell
            } else {
                ResonanceSignalType::Sell
            }
        } else if detection.frequency < 0.01 {
            ResonanceSignalType::PhaseTransition
        } else {
            ResonanceSignalType::Hold
        };
        
        ResonanceSignal {
            symbol: detection.primary_symbol,
            timestamp: detection.timestamp,
            signal_type,
            strength: detection.strength,
            confidence: detection.strength * 0.8, // Adjust confidence based on strength
            correlated_symbols: detection.resonating_symbols,
            frequency_band: (detection.frequency - 0.01, detection.frequency + 0.01),
            phase_alignment: detection.phase,
        }
    }
    
    /// Perform spectral analysis on price data
    pub async fn spectral_analysis(&self, symbol: &str) -> Result<Vec<(f64, f64)>> {
        let oscillator = self.market_oscillators.get(symbol)
            .ok_or_else(|| anyhow::anyhow!("Symbol not found"))?;
        
        if oscillator.price_history.len() < 64 {
            return Ok(Vec::new());
        }
        
        // Prepare data for FFT
        let prices: Vec<f64> = oscillator.price_history.iter().copied().collect();
        let n = prices.len();
        
        // Apply Hanning window to reduce spectral leakage
        let windowed: Vec<Complex<f64>> = prices.iter().enumerate()
            .map(|(i, &p)| {
                let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                Complex::new(p * window, 0.0)
            })
            .collect();
        
        // Perform FFT
        let mut fft_data = windowed.clone();
        let mut planner = self.fft_planner.write().await;
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut fft_data);
        
        // Calculate power spectrum
        let spectrum: Vec<(f64, f64)> = fft_data.iter().enumerate()
            .take(n / 2)
            .map(|(i, c)| {
                let freq = i as f64 / n as f64;
                let power = (c.re * c.re + c.im * c.im).sqrt() / n as f64;
                (freq, power)
            })
            .collect();
        
        Ok(spectrum)
    }
    
    /// Get oscillator data for a specific symbol (for demo purposes)
    pub fn get_oscillator(&self, symbol: &str) -> Option<&MarketOscillator> {
        self.market_oscillators.get(symbol)
    }
    
    /// Check if symbol has oscillator data
    pub fn has_oscillator(&self, symbol: &str) -> bool {
        self.market_oscillators.contains_key(symbol)
    }
}

struct ResonanceDetection {
    primary_symbol: String,
    resonating_symbols: Vec<String>,
    strength: f64,
    phase: f64,
    frequency: f64,
    timestamp: DateTime<Utc>,
}

/// Integration with main trading engine
pub struct ResonanceTradingStrategy {
    analyzer: ResonanceMarketAnalyzer,
    signal_threshold: f64,
    position_sizing_factor: f64,
}

impl ResonanceTradingStrategy {
    pub fn new() -> Self {
        Self {
            analyzer: ResonanceMarketAnalyzer::new(ResonanceParameters::default()),
            signal_threshold: 0.7,
            position_sizing_factor: 0.1,
        }
    }
    
    /// Generate trading decisions from market data
    pub async fn analyze_market(&mut self, quotes: HashMap<String, Quote>) -> Vec<TradingDecision> {
        let signals = self.analyzer.analyze(&quotes).await;
        let mut decisions = Vec::new();
        
        for signal in signals {
            if signal.strength >= self.signal_threshold {
                let decision = self.convert_signal_to_decision(signal, &quotes);
                decisions.push(decision);
            }
        }
        
        decisions
    }
    
    fn convert_signal_to_decision(&self, signal: ResonanceSignal, quotes: &HashMap<String, Quote>) -> TradingDecision {
        let current_price = quotes.get(&signal.symbol)
            .map(|q| q.last)
            .unwrap_or(dec!(0));
        
        let (action, quantity) = match signal.signal_type {
            ResonanceSignalType::StrongBuy => {
                (TradingAction::Buy, self.calculate_position_size(signal.strength * 1.5))
            },
            ResonanceSignalType::Buy => {
                (TradingAction::Buy, self.calculate_position_size(signal.strength))
            },
            ResonanceSignalType::StrongSell => {
                (TradingAction::Sell, self.calculate_position_size(signal.strength * 1.5))
            },
            ResonanceSignalType::Sell => {
                (TradingAction::Sell, self.calculate_position_size(signal.strength))
            },
            ResonanceSignalType::ResonanceBreakout => {
                (TradingAction::Buy, self.calculate_position_size(signal.strength * 2.0))
            },
            _ => (TradingAction::Hold, 0),
        };
        
        TradingDecision {
            symbol: signal.symbol,
            action,
            quantity,
            price: current_price,
            confidence: signal.confidence,
            reason: format!("Resonance: {:?} @ {:.2} strength", signal.signal_type, signal.strength),
            correlated_assets: signal.correlated_symbols,
        }
    }
    
    fn calculate_position_size(&self, strength: f64) -> u32 {
        ((strength * 100.0 * self.position_sizing_factor) as u32).max(1).min(1000)
    }
}

#[derive(Debug, Clone)]
pub struct TradingDecision {
    pub symbol: String,
    pub action: TradingAction,
    pub quantity: u32,
    pub price: Decimal,
    pub confidence: f64,
    pub reason: String,
    pub correlated_assets: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
}