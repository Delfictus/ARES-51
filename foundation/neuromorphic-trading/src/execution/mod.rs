//! Execution and performance tracking

pub mod signal_bridge;
pub mod algorithms;
pub mod performance;

pub use signal_bridge::{
    NeuromorphicSignalBridge, SignalConverterConfig, ConfidenceThresholds,
    PatternActionMap
};

pub use algorithms::{
    ExecutionEngine, ExecutionParams, ExecutionProgress, AlgorithmType,
    ExecutionAlgorithm, MarketContext, OrderSlice,
    TWAPAlgorithm, VWAPAlgorithm, IcebergAlgorithm, POVAlgorithm, SniperAlgorithm
};

pub use performance::{
    PerformanceAnalyzer, PerformanceMetrics, PerformanceReporter,
    Period, EquityPoint, TradeRecord, CurrentStats
};