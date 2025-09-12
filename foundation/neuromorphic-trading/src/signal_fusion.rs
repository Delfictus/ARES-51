//! Signal Fusion Module
//! 
//! Combines deterministic and neuromorphic signals for optimal trading decisions

use std::collections::HashMap;
use anyhow::Result;

use crate::{DeterministicSignal, NeuromorphicSignal, MarketState};
use crate::reservoir::PatternType;

/// Trading action
#[derive(Clone, Debug, PartialEq)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
    CancelAll,
}

/// Signal source
#[derive(Clone, Debug)]
pub enum SignalSource {
    Deterministic {
        strategy: String,
        latency_ns: u64,
    },
    Neuromorphic {
        patterns: Vec<PatternType>,
        novelty_score: f32,
    },
    Combined {
        det_weight: f32,
        neuro_weight: f32,
    },
}

/// Trading signal
#[derive(Clone, Debug)]
pub struct TradingSignal {
    pub action: TradeAction,
    pub confidence: f32,
    pub source: SignalSource,
    pub risk_score: f32,
    pub expected_pnl: f64,
    pub time_horizon_ms: u64,
}

impl TradingSignal {
    pub fn hold() -> Self {
        Self {
            action: TradeAction::Hold,
            confidence: 0.0,
            source: SignalSource::Combined {
                det_weight: 0.5,
                neuro_weight: 0.5,
            },
            risk_score: 0.0,
            expected_pnl: 0.0,
            time_horizon_ms: 0,
        }
    }
}

/// Adaptive weights for signal combination
struct AdaptiveWeights {
    deterministic: f32,
    neuromorphic: f32,
    learning_rate: f32,
    performance_history: Vec<f32>,
}

impl AdaptiveWeights {
    fn new() -> Self {
        Self {
            deterministic: 0.5,
            neuromorphic: 0.5,
            learning_rate: 0.01,
            performance_history: Vec::new(),
        }
    }
    
    fn get_current(&self, market_state: &MarketState) -> (f32, f32) {
        // Adjust weights based on market conditions
        let volatility_factor = (market_state.volatility / 0.02).min(2.0) as f32;
        
        // Higher volatility -> more weight to neuromorphic
        let neuro_boost = volatility_factor * 0.1;
        let det_weight = (self.deterministic - neuro_boost).max(0.2);
        let neuro_weight = (self.neuromorphic + neuro_boost).min(0.8);
        
        // Normalize
        let total = det_weight + neuro_weight;
        (det_weight / total, neuro_weight / total)
    }
    
    fn update(&mut self, performance: f32) {
        self.performance_history.push(performance);
        
        // Simple gradient update based on performance
        if performance > 0.0 {
            // Good performance - increase weight of current mix
            self.deterministic *= 1.0 + self.learning_rate;
            self.neuromorphic *= 1.0 + self.learning_rate;
        } else {
            // Poor performance - adjust weights
            self.deterministic *= 1.0 - self.learning_rate;
            self.neuromorphic *= 1.0 - self.learning_rate;
        }
        
        // Normalize
        let total = self.deterministic + self.neuromorphic;
        self.deterministic /= total;
        self.neuromorphic /= total;
    }
}

/// Conflict resolver for disagreeing signals
struct ConflictResolver {
    resolution_history: HashMap<String, f32>,
}

impl ConflictResolver {
    fn new() -> Self {
        Self {
            resolution_history: HashMap::new(),
        }
    }
    
    fn has_conflict(&self, det: &DeterministicSignal, neuro: &NeuromorphicSignal) -> bool {
        // Check if signals disagree on direction
        let det_bullish = det.action == TradeAction::Buy;
        let neuro_bullish = neuro.patterns.iter()
            .any(|(p, _)| matches!(p, PatternType::MomentumIgnition));
        
        det_bullish != neuro_bullish
    }
    
    fn resolve(&self, det: &DeterministicSignal, neuro: &NeuromorphicSignal, weights: (f32, f32)) -> TradingSignal {
        // Resolution based on confidence and weights
        let det_score = det.confidence * weights.0;
        let neuro_score = neuro.confidence * weights.1;
        
        if det_score > neuro_score {
            TradingSignal {
                action: det.action.clone(),
                confidence: det.confidence * 0.7, // Reduce confidence due to conflict
                source: SignalSource::Deterministic {
                    strategy: det.strategy.clone(),
                    latency_ns: det.latency_ns,
                },
                risk_score: 0.7, // Higher risk due to conflict
                expected_pnl: 0.0,
                time_horizon_ms: 100,
            }
        } else {
            TradingSignal {
                action: if neuro.patterns.iter().any(|(p, _)| matches!(p, PatternType::MomentumIgnition)) {
                    TradeAction::Buy
                } else {
                    TradeAction::Sell
                },
                confidence: neuro.confidence * 0.7,
                source: SignalSource::Neuromorphic {
                    patterns: neuro.patterns.clone(),
                    novelty_score: neuro.novelty_score,
                },
                risk_score: 0.7,
                expected_pnl: 0.0,
                time_horizon_ms: 1000,
            }
        }
    }
}

/// Performance tracker
struct PerformanceTracker {
    decisions: Vec<DecisionRecord>,
    total_pnl: f64,
}

#[derive(Clone)]
struct DecisionRecord {
    timestamp: u64,
    action: TradeAction,
    source: DecisionSource,
    pnl: f64,
}

#[derive(Clone)]
enum DecisionSource {
    Deterministic,
    Neuromorphic,
    Combined,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            decisions: Vec::new(),
            total_pnl: 0.0,
        }
    }
    
    fn record_decision(&mut self, source: DecisionSource) {
        self.decisions.push(DecisionRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            action: TradeAction::Hold,
            source,
            pnl: 0.0,
        });
    }
    
    fn update_pnl(&mut self, pnl: f64) {
        if let Some(last) = self.decisions.last_mut() {
            last.pnl = pnl;
            self.total_pnl += pnl;
        }
    }
}

/// Signal fusion engine
pub struct SignalFusion {
    weights: AdaptiveWeights,
    conflict_resolver: ConflictResolver,
    performance_tracker: PerformanceTracker,
}

impl SignalFusion {
    pub fn new() -> Self {
        Self {
            weights: AdaptiveWeights::new(),
            conflict_resolver: ConflictResolver::new(),
            performance_tracker: PerformanceTracker::new(),
        }
    }
    
    /// Fuse deterministic and neuromorphic signals
    pub fn fuse(
        &mut self,
        deterministic: Option<DeterministicSignal>,
        neuromorphic: Option<NeuromorphicSignal>,
        market_state: &MarketState,
    ) -> TradingSignal {
        match (deterministic, neuromorphic) {
            (Some(det), Some(neuro)) => {
                // Fast path: high confidence deterministic
                if det.confidence > 0.95 && det.latency_ns < 1000 {
                    self.performance_tracker.record_decision(DecisionSource::Deterministic);
                    return self.create_signal_from_deterministic(det);
                }
                
                // Smart path: novel pattern detected
                if neuro.novelty_score > 0.8 && !neuro.patterns.is_empty() {
                    self.performance_tracker.record_decision(DecisionSource::Neuromorphic);
                    return self.create_signal_from_neuromorphic(neuro);
                }
                
                // Combined path
                let weights = self.weights.get_current(market_state);
                
                // Check for conflicts
                if self.conflict_resolver.has_conflict(&det, &neuro) {
                    return self.conflict_resolver.resolve(&det, &neuro, weights);
                }
                
                // Weighted combination
                self.performance_tracker.record_decision(DecisionSource::Combined);
                self.combine_signals(det, neuro, weights)
            },
            (Some(det), None) => {
                self.performance_tracker.record_decision(DecisionSource::Deterministic);
                self.create_signal_from_deterministic(det)
            },
            (None, Some(neuro)) => {
                self.performance_tracker.record_decision(DecisionSource::Neuromorphic);
                self.create_signal_from_neuromorphic(neuro)
            },
            (None, None) => TradingSignal::hold(),
        }
    }
    
    fn create_signal_from_deterministic(&self, det: DeterministicSignal) -> TradingSignal {
        TradingSignal {
            action: det.action,
            confidence: det.confidence,
            source: SignalSource::Deterministic {
                strategy: det.strategy,
                latency_ns: det.latency_ns,
            },
            risk_score: 0.3,
            expected_pnl: 0.0,
            time_horizon_ms: 10,
        }
    }
    
    fn create_signal_from_neuromorphic(&self, neuro: NeuromorphicSignal) -> TradingSignal {
        let action = if neuro.patterns.iter().any(|(p, _)| matches!(p, PatternType::MomentumIgnition | PatternType::InstitutionalFootprint)) {
            TradeAction::Buy
        } else if neuro.patterns.iter().any(|(p, _)| matches!(p, PatternType::FlashCrashPrecursor | PatternType::LiquidityWithdrawal)) {
            TradeAction::Sell
        } else {
            TradeAction::Hold
        };
        
        TradingSignal {
            action,
            confidence: neuro.confidence,
            source: SignalSource::Neuromorphic {
                patterns: neuro.patterns,
                novelty_score: neuro.novelty_score,
            },
            risk_score: 0.5,
            expected_pnl: 0.0,
            time_horizon_ms: 100,
        }
    }
    
    fn combine_signals(&self, det: DeterministicSignal, neuro: NeuromorphicSignal, weights: (f32, f32)) -> TradingSignal {
        let combined_confidence = weights.0 * det.confidence + weights.1 * neuro.confidence;
        let combined_risk = weights.0 * 0.3 + weights.1 * 0.5;
        
        // Vote on action
        let action = if det.action == TradeAction::Buy && neuro.confidence > 0.3 {
            TradeAction::Buy
        } else if det.action == TradeAction::Sell && neuro.confidence > 0.3 {
            TradeAction::Sell
        } else {
            TradeAction::Hold
        };
        
        TradingSignal {
            action,
            confidence: combined_confidence,
            source: SignalSource::Combined {
                det_weight: weights.0,
                neuro_weight: weights.1,
            },
            risk_score: combined_risk,
            expected_pnl: 0.0,
            time_horizon_ms: 50,
        }
    }
    
    /// Update performance based on trade results
    pub fn update_performance(&mut self, pnl: f64) {
        self.performance_tracker.update_pnl(pnl);
        self.weights.update(pnl as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_fusion() {
        let mut fusion = SignalFusion::new();
        
        let det = DeterministicSignal {
            action: TradeAction::Buy,
            confidence: 0.8,
            latency_ns: 500,
            strategy: "momentum".to_string(),
        };
        
        let neuro = NeuromorphicSignal {
            patterns: vec![(PatternType::MomentumIgnition, 0.7)],
            confidence: 0.6,
            novelty_score: 0.4,
        };
        
        let market = MarketState {
            volatility: 0.02,
            spread: 0.01,
            volume: 1000000,
        };
        
        let signal = fusion.fuse(Some(det), Some(neuro), &market);
        assert!(signal.confidence > 0.0);
    }
}