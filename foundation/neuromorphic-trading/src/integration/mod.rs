//! Full system integration

use crate::neuromorphic::{SpikeProcessor, ReservoirComputer, DrppResonanceAnalyzer};
use crate::spike_encoding::SpikeEncoder;
use crate::event_bus::EventBus;
use crate::exchanges::{ExchangeConnector, Symbol, Exchange};
use crate::market_data::{UnifiedMarketFeed, MarketDataSpikeBridge, UnifiedFeedConfig, SpikeBridgeConfig};
use crate::paper_trading::{PaperTradingEngine, PaperTradingConfig};
use crate::execution::{NeuromorphicSignalBridge, SignalConverterConfig, ExecutionEngine, PerformanceAnalyzer};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use std::time::Duration;

/// Full system configuration
pub struct SystemConfig {
    pub neuron_count: usize,
    pub reservoir_size: usize,
    pub initial_capital: f64,
    pub exchanges: Vec<Exchange>,
    pub symbols: Vec<Symbol>,
    pub enable_paper_trading: bool,
    pub enable_live_trading: bool,
    pub update_interval: Duration,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            neuron_count: 10000,
            reservoir_size: 1000,
            initial_capital: 100000.0,
            exchanges: vec![Exchange::Binance, Exchange::Coinbase],
            symbols: vec![
                Symbol::new("BTC-USD"),
                Symbol::new("ETH-USD"),
                Symbol::new("SOL-USD"),
            ],
            enable_paper_trading: true,
            enable_live_trading: false,
            update_interval: Duration::from_millis(100),
        }
    }
}

/// Integrated trading system
pub struct IntegratedTradingSystem {
    // Neuromorphic components
    spike_processor: Arc<SpikeProcessor>,
    reservoir: Arc<ReservoirComputer>,
    pattern_detector: Arc<PatternDetector>,
    event_bus: Arc<EventBus>,
    
    // Market data components
    unified_feed: Arc<UnifiedMarketFeed>,
    spike_bridge: Arc<MarketDataSpikeBridge>,
    
    // Trading components
    paper_trading: Arc<PaperTradingEngine>,
    signal_bridge: Arc<NeuromorphicSignalBridge>,
    execution_engine: Arc<ExecutionEngine>,
    performance_analyzer: Arc<PerformanceAnalyzer>,
    
    // Configuration
    config: SystemConfig,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl IntegratedTradingSystem {
    pub fn new(config: SystemConfig) -> Result<Self> {
        // Initialize neuromorphic components
        let spike_processor = Arc::new(SpikeProcessor::new(config.neuron_count));
        let reservoir = Arc::new(ReservoirComputer::new(config.reservoir_size, config.neuron_count));
        let pattern_detector = Arc::new(PatternDetector::new());
        let event_bus = Arc::new(EventBus::new(65536));
        
        // Initialize market data components
        let unified_feed = Arc::new(UnifiedMarketFeed::new(UnifiedFeedConfig::default()));
        let spike_bridge = Arc::new(MarketDataSpikeBridge::new(SpikeBridgeConfig::default()));
        
        // Initialize trading components
        let paper_config = PaperTradingConfig {
            initial_capital: config.initial_capital,
            ..PaperTradingConfig::default()
        };
        let paper_trading = Arc::new(PaperTradingEngine::new(paper_config));
        
        let signal_bridge = Arc::new(NeuromorphicSignalBridge::new(SignalConverterConfig::default()));
        
        let (order_tx, _order_rx) = mpsc::unbounded_channel();
        let execution_engine = Arc::new(ExecutionEngine::new(order_tx));
        
        let performance_analyzer = Arc::new(PerformanceAnalyzer::new(config.initial_capital));
        
        Ok(Self {
            spike_processor,
            reservoir,
            pattern_detector,
            event_bus,
            unified_feed,
            spike_bridge,
            paper_trading,
            signal_bridge,
            execution_engine,
            performance_analyzer,
            config,
            running: Arc::new(tokio::sync::RwLock::new(false)),
        })
    }
    
    /// Start the integrated system
    pub async fn start(&mut self) -> Result<()> {
        println!("Starting Integrated Trading System...");
        
        // Set running flag
        let mut running = self.running.write().await;
        *running = true;
        drop(running);
        
        // Start paper trading engine
        if self.config.enable_paper_trading {
            self.paper_trading.clone().start().await?;
            println!("✓ Paper trading engine started");
        }
        
        // Start market data processing pipeline
        self.start_market_data_pipeline().await?;
        println!("✓ Market data pipeline started");
        
        // Start neuromorphic processing pipeline
        self.start_neuromorphic_pipeline().await?;
        println!("✓ Neuromorphic processing started");
        
        // Start signal processing pipeline
        self.start_signal_pipeline().await?;
        println!("✓ Signal processing started");
        
        // Start performance monitoring
        self.start_performance_monitoring().await?;
        println!("✓ Performance monitoring started");
        
        println!("System fully operational!");
        
        Ok(())
    }
    
    /// Stop the integrated system
    pub async fn stop(&self) -> Result<()> {
        println!("Stopping Integrated Trading System...");
        
        let mut running = self.running.write().await;
        *running = false;
        
        if self.config.enable_paper_trading {
            self.paper_trading.stop().await?;
        }
        
        println!("System stopped");
        
        Ok(())
    }
    
    /// Start market data processing pipeline
    async fn start_market_data_pipeline(&self) -> Result<()> {
        let unified_feed = self.unified_feed.clone();
        let spike_bridge = self.spike_bridge.clone();
        let running = self.running.clone();
        
        // Process market events and generate spikes
        tokio::spawn(async move {
            let mut feed = unified_feed.clone();
            let mut receiver = Arc::get_mut(&mut feed)
                .and_then(|f| f.subscribe());
            
            if let Some(mut receiver) = receiver {
                while *running.read().await {
                    tokio::select! {
                        Some(event) = receiver.recv() => {
                            if let Err(e) = spike_bridge.process_event(event).await {
                                eprintln!("Error processing market event: {}", e);
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start neuromorphic processing pipeline
    async fn start_neuromorphic_pipeline(&self) -> Result<()> {
        let spike_processor = self.spike_processor.clone();
        let reservoir = self.reservoir.clone();
        let pattern_detector = self.pattern_detector.clone();
        let event_bus = self.event_bus.clone();
        let spike_bridge = self.spike_bridge.clone();
        let running = self.running.clone();
        
        // Process spikes through neuromorphic pipeline
        tokio::spawn(async move {
            let mut bridge = spike_bridge.clone();
            let mut receiver = Arc::get_mut(&mut bridge)
                .and_then(|b| b.subscribe());
            
            if let Some(mut receiver) = receiver {
                while *running.read().await {
                    tokio::select! {
                        Some(spikes) = receiver.recv() => {
                            // Process through spike processor
                            let processed = spike_processor.process_batch(&spikes);
                            
                            // Feed to reservoir
                            reservoir.process_spikes(&processed);
                            
                            // Detect patterns
                            if let Some(pattern) = pattern_detector.detect(&processed) {
                                // Publish pattern to event bus
                                event_bus.publish_pattern(pattern).await;
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_millis(1)) => {}
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start signal processing pipeline
    async fn start_signal_pipeline(&self) -> Result<()> {
        let signal_bridge = self.signal_bridge.clone();
        let paper_trading = self.paper_trading.clone();
        let event_bus = self.event_bus.clone();
        let running = self.running.clone();
        let symbols = self.config.symbols.clone();
        let exchange = self.config.exchanges[0]; // Use first exchange for now
        
        // Convert patterns to trading signals
        tokio::spawn(async move {
            let mut signal_bridge = signal_bridge.clone();
            let mut receiver = Arc::get_mut(&mut signal_bridge)
                .and_then(|b| b.subscribe());
            
            if receiver.is_some() {
                while *running.read().await {
                    // Check for patterns from event bus
                    if let Some(pattern) = event_bus.get_latest_pattern().await {
                        // Process pattern for each symbol
                        for symbol in &symbols {
                            if let Ok(Some(signal)) = signal_bridge.process_spike_pattern(
                                &pattern,
                                symbol.clone(),
                                exchange
                            ) {
                                // Send to paper trading
                                if let Err(e) = paper_trading.process_signal(signal).await {
                                    eprintln!("Error processing signal: {}", e);
                                }
                            }
                        }
                    }
                    
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        });
        
        Ok(())
    }
    
    /// Start performance monitoring
    async fn start_performance_monitoring(&self) -> Result<()> {
        let performance = self.performance_analyzer.clone();
        let paper_trading = self.paper_trading.clone();
        let running = self.running.clone();
        
        // Monitor and update performance metrics
        tokio::spawn(async move {
            while *running.read().await {
                // Get current statistics from paper trading
                let stats = paper_trading.get_statistics();
                
                // Update performance analyzer
                performance.update_equity(stats.capital);
                
                // Record completed trades
                let positions = paper_trading.position_manager().get_open_positions();
                for position in positions {
                    if position.status == crate::paper_trading::PositionStatus::Closed {
                        performance.record_trade(&position);
                    }
                }
                
                // Print summary every minute
                static mut COUNTER: u64 = 0;
                unsafe {
                    COUNTER += 1;
                    if COUNTER % 600 == 0 { // Every minute at 10ms intervals
                        let metrics = performance.get_metrics(crate::execution::Period::AllTime);
                        println!("\n=== Performance Update ===");
                        println!("Capital: ${:.2}", stats.capital);
                        println!("Total P&L: ${:.2} ({:.2}%)", 
                            stats.total_pnl, stats.total_return_pct);
                        println!("Win Rate: {:.1}%", metrics.win_rate);
                        println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
                        println!("Trades: {}", metrics.total_trades);
                        println!("========================\n");
                    }
                }
                
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        
        Ok(())
    }
    
    /// Get system statistics
    pub fn get_statistics(&self) -> SystemStatistics {
        let trading_stats = self.paper_trading.get_statistics();
        let performance_metrics = self.performance_analyzer.get_metrics(crate::execution::Period::AllTime);
        let current_stats = self.performance_analyzer.get_current_stats();
        
        SystemStatistics {
            current_capital: trading_stats.capital,
            total_pnl: trading_stats.total_pnl,
            total_return_pct: trading_stats.total_return_pct,
            win_rate: performance_metrics.win_rate,
            sharpe_ratio: performance_metrics.sharpe_ratio,
            max_drawdown: performance_metrics.max_drawdown,
            total_trades: performance_metrics.total_trades,
            signals_processed: trading_stats.signals_processed,
            signals_executed: trading_stats.signals_executed,
            current_drawdown: current_stats.current_drawdown,
        }
    }
    
    /// Run backtesting
    pub async fn run_backtest(&mut self, duration: Duration) -> Result<BacktestResults> {
        println!("Starting backtest for {:?}...", duration);
        
        self.start().await?;
        
        // Run for specified duration
        tokio::time::sleep(duration).await;
        
        self.stop().await?;
        
        // Collect results
        let stats = self.get_statistics();
        let equity_curve = self.performance_analyzer.get_equity_curve(
            crate::execution::Period::AllTime
        );
        
        Ok(BacktestResults {
            statistics: stats,
            equity_curve,
            duration,
        })
    }
}

/// System statistics
#[derive(Clone, Debug)]
pub struct SystemStatistics {
    pub current_capital: f64,
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_trades: u64,
    pub signals_processed: u64,
    pub signals_executed: u64,
    pub current_drawdown: f64,
}

/// Backtest results
#[derive(Clone, Debug)]
pub struct BacktestResults {
    pub statistics: SystemStatistics,
    pub equity_curve: Vec<crate::execution::EquityPoint>,
    pub duration: Duration,
}

impl BacktestResults {
    /// Print summary
    pub fn print_summary(&self) {
        println!("\n========== BACKTEST RESULTS ==========");
        println!("Duration: {:?}", self.duration);
        println!("Final Capital: ${:.2}", self.statistics.current_capital);
        println!("Total Return: ${:.2} ({:.2}%)", 
            self.statistics.total_pnl, self.statistics.total_return_pct);
        println!("Total Trades: {}", self.statistics.total_trades);
        println!("Win Rate: {:.1}%", self.statistics.win_rate);
        println!("Sharpe Ratio: {:.3}", self.statistics.sharpe_ratio);
        println!("Max Drawdown: {:.2}%", self.statistics.max_drawdown * 100.0);
        println!("Signals: {} processed, {} executed", 
            self.statistics.signals_processed, self.statistics.signals_executed);
        println!("======================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let config = SystemConfig::default();
        let system = IntegratedTradingSystem::new(config);
        assert!(system.is_ok());
    }
    
    #[tokio::test]
    async fn test_system_start_stop() {
        let config = SystemConfig {
            enable_paper_trading: true,
            enable_live_trading: false,
            ..SystemConfig::default()
        };
        
        let mut system = IntegratedTradingSystem::new(config).unwrap();
        
        assert!(system.start().await.is_ok());
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(system.stop().await.is_ok());
    }
}