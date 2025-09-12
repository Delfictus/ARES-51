//! DRPP Integration Module
//! Imports and re-exports DRPP functionality for neuromorphic trading

// Re-export DRPP core components
pub use csf_clogic::drpp::{
    DynamicResonancePatternProcessor,
    DrppConfig, 
    DrppState,
    Pattern,
    PatternType,
    PatternData,
    LockFreeSpmc,
    ChannelConfig,
    Consumer,
    Producer,
};

// Transfer entropy engine
pub use csf_clogic::drpp::{
    TransferEntropyEngine,
    TeConfig,
};

// Neural oscillator components
pub use csf_clogic::drpp::{
    NeuralOscillator,
    PatternDetector as DrppPatternDetector,
};

/// Market-optimized DRPP configuration
pub fn create_market_config() -> DrppConfig {
    DrppConfig {
        num_oscillators: 128, // Optimized for market data
        coupling_strength: 0.3,
        pattern_threshold: 0.7,
        frequency_range: (0.1, 100.0), // Market frequencies
        time_window_ms: 1000,
        adaptive_tuning: true,
        channel_config: ChannelConfig {
            capacity: 32768, // High-frequency processing
            backpressure_threshold: 0.85,
            max_consumers: 16,
            use_mmap: true,
            numa_node: -1,
        },
    }
}

/// Revolutionary pattern types for trading
pub fn pattern_to_signal_strength(pattern: &Pattern) -> f64 {
    match pattern.pattern_type {
        PatternType::Emergent => pattern.strength * 2.0,     // Highest value
        PatternType::Synchronous => pattern.strength * 1.5,  // High value
        PatternType::Traveling => pattern.strength * 1.2,    // Medium-high
        PatternType::Standing => pattern.strength * 0.8,     // Medium
        PatternType::Chaotic => pattern.strength * 0.3,      // Low value
    }
}