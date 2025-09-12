//! Performance tracking and analytics

use crate::paper_trading::{Position, PositionStatistics};
use crate::exchanges::Symbol;
use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Performance period
#[derive(Clone, Debug, PartialEq)]
pub enum Period {
    Minute1,
    Minute5,
    Minute15,
    Hour1,
    Day1,
    Week1,
    Month1,
    Year1,
    AllTime,
}

impl Period {
    pub fn duration(&self) -> Duration {
        match self {
            Period::Minute1 => Duration::from_secs(60),
            Period::Minute5 => Duration::from_secs(300),
            Period::Minute15 => Duration::from_secs(900),
            Period::Hour1 => Duration::from_secs(3600),
            Period::Day1 => Duration::from_secs(86400),
            Period::Week1 => Duration::from_secs(604800),
            Period::Month1 => Duration::from_secs(2592000),
            Period::Year1 => Duration::from_secs(31536000),
            Period::AllTime => Duration::from_secs(u64::MAX),
        }
    }
}

/// Performance metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub total_return_pct: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: Duration,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub consecutive_wins: u32,
    pub consecutive_losses: u32,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub total_volume: f64,
    pub total_commission: f64,
    pub total_slippage: f64,
}

/// Equity curve point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: u64,
    pub equity: f64,
    pub drawdown: f64,
    pub daily_return: f64,
}

/// Performance analyzer
pub struct PerformanceAnalyzer {
    equity_curve: Arc<RwLock<Vec<EquityPoint>>>,
    metrics_by_period: DashMap<Period, PerformanceMetrics>,
    metrics_by_symbol: DashMap<Symbol, PerformanceMetrics>,
    daily_returns: Arc<RwLock<Vec<f64>>>,
    trade_history: Arc<RwLock<Vec<TradeRecord>>>,
    initial_capital: f64,
    current_capital: Arc<RwLock<f64>>,
    peak_equity: Arc<RwLock<f64>>,
    drawdown_start: Arc<RwLock<Option<u64>>>,
}

/// Trade record for history
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: Symbol,
    pub entry_time: u64,
    pub exit_time: u64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub commission: f64,
    pub slippage: f64,
}

impl PerformanceAnalyzer {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            equity_curve: Arc::new(RwLock::new(Vec::new())),
            metrics_by_period: DashMap::new(),
            metrics_by_symbol: DashMap::new(),
            daily_returns: Arc::new(RwLock::new(Vec::new())),
            trade_history: Arc::new(RwLock::new(Vec::new())),
            initial_capital,
            current_capital: Arc::new(RwLock::new(initial_capital)),
            peak_equity: Arc::new(RwLock::new(initial_capital)),
            drawdown_start: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Update with new equity value
    pub fn update_equity(&self, equity: f64) {
        let mut current = self.current_capital.write();
        let prev_equity = *current;
        *current = equity;
        drop(current);
        
        // Calculate daily return
        let daily_return = if prev_equity > 0.0 {
            (equity - prev_equity) / prev_equity
        } else {
            0.0
        };
        
        // Update peak and drawdown
        let mut peak = self.peak_equity.write();
        if equity > *peak {
            *peak = equity;
            *self.drawdown_start.write() = None;
        }
        
        let drawdown = (*peak - equity) / *peak;
        
        // Record equity point
        let point = EquityPoint {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            equity,
            drawdown,
            daily_return,
        };
        
        self.equity_curve.write().push(point);
        
        // Update daily returns
        if daily_return != 0.0 {
            self.daily_returns.write().push(daily_return);
        }
        
        // Start tracking drawdown duration if needed
        if drawdown > 0.0 && self.drawdown_start.read().is_none() {
            *self.drawdown_start.write() = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            );
        }
    }
    
    /// Record completed trade
    pub fn record_trade(&self, position: &Position) {
        if position.exit_price.is_none() {
            return; // Not closed yet
        }
        
        let record = TradeRecord {
            symbol: position.symbol.clone(),
            entry_time: position.entry_time,
            exit_time: position.exit_time.unwrap_or(0),
            entry_price: position.entry_price,
            exit_price: position.exit_price.unwrap_or(0.0),
            quantity: position.quantity,
            pnl: position.realized_pnl,
            return_pct: position.roi(),
            commission: position.commission,
            slippage: position.slippage,
        };
        
        self.trade_history.write().push(record);
        
        // Update symbol-specific metrics
        self.update_symbol_metrics(&position.symbol);
    }
    
    /// Calculate metrics for a period
    pub fn calculate_metrics(&self, period: Period) -> PerformanceMetrics {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 - period.duration().as_millis() as u64;
        
        let trades: Vec<TradeRecord> = self.trade_history
            .read()
            .iter()
            .filter(|t| t.exit_time >= cutoff)
            .cloned()
            .collect();
        
        let equity_points: Vec<EquityPoint> = self.equity_curve
            .read()
            .iter()
            .filter(|p| p.timestamp >= cutoff)
            .cloned()
            .collect();
        
        self.calculate_metrics_from_data(&trades, &equity_points, period)
    }
    
    /// Calculate metrics from data
    fn calculate_metrics_from_data(
        &self,
        trades: &[TradeRecord],
        equity_points: &[EquityPoint],
        period: Period,
    ) -> PerformanceMetrics {
        let mut metrics = PerformanceMetrics::default();
        
        if trades.is_empty() && equity_points.is_empty() {
            return metrics;
        }
        
        // Basic statistics
        metrics.total_trades = trades.len() as u64;
        metrics.winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count() as u64;
        metrics.losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count() as u64;
        
        // Win rate
        if metrics.total_trades > 0 {
            metrics.win_rate = (metrics.winning_trades as f64 / metrics.total_trades as f64) * 100.0;
        }
        
        // P&L statistics
        let wins: Vec<f64> = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).collect();
        let losses: Vec<f64> = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).collect();
        
        if !wins.is_empty() {
            metrics.avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
            metrics.largest_win = wins.iter().cloned().fold(f64::MIN, f64::max);
        }
        
        if !losses.is_empty() {
            metrics.avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
            metrics.largest_loss = losses.iter().cloned().fold(f64::MIN, f64::max);
        }
        
        // Profit factor
        if metrics.avg_loss > 0.0 && metrics.win_rate > 0.0 {
            let win_expectancy = metrics.avg_win * (metrics.win_rate / 100.0);
            let loss_expectancy = metrics.avg_loss * (1.0 - metrics.win_rate / 100.0);
            if loss_expectancy > 0.0 {
                metrics.profit_factor = win_expectancy / loss_expectancy;
            }
        }
        
        // Consecutive wins/losses
        let mut current_streak = 0i32;
        let mut max_wins = 0u32;
        let mut max_losses = 0u32;
        
        for trade in trades {
            if trade.pnl > 0.0 {
                if current_streak >= 0 {
                    current_streak += 1;
                    max_wins = max_wins.max(current_streak as u32);
                } else {
                    current_streak = 1;
                }
            } else if trade.pnl < 0.0 {
                if current_streak <= 0 {
                    current_streak -= 1;
                    max_losses = max_losses.max((-current_streak) as u32);
                } else {
                    current_streak = -1;
                }
            }
        }
        
        metrics.consecutive_wins = max_wins;
        metrics.consecutive_losses = max_losses;
        
        // Volume and costs
        metrics.total_volume = trades.iter().map(|t| t.quantity * t.entry_price).sum();
        metrics.total_commission = trades.iter().map(|t| t.commission).sum();
        metrics.total_slippage = trades.iter().map(|t| t.slippage).sum();
        
        // Returns and drawdown from equity curve
        if !equity_points.is_empty() {
            let start_equity = equity_points.first().unwrap().equity;
            let end_equity = equity_points.last().unwrap().equity;
            
            metrics.total_return = end_equity - start_equity;
            metrics.total_return_pct = if start_equity > 0.0 {
                ((end_equity - start_equity) / start_equity) * 100.0
            } else {
                0.0
            };
            
            // Annualized return
            let days = period.duration().as_secs() as f64 / 86400.0;
            if days > 0.0 {
                let years = days / 365.0;
                metrics.annualized_return = ((end_equity / start_equity).powf(1.0 / years) - 1.0) * 100.0;
            }
            
            // Max drawdown
            metrics.max_drawdown = equity_points
                .iter()
                .map(|p| p.drawdown)
                .fold(0.0, f64::max);
            
            // Volatility (standard deviation of returns)
            let returns: Vec<f64> = equity_points.windows(2)
                .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
                .collect();
            
            if returns.len() > 1 {
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                metrics.volatility = variance.sqrt() * (252.0_f64.sqrt()); // Annualized
                
                // Sharpe ratio (assume 0% risk-free rate)
                if metrics.volatility > 0.0 {
                    metrics.sharpe_ratio = (metrics.annualized_return / 100.0) / metrics.volatility;
                }
                
                // Sortino ratio (downside deviation)
                let downside_returns: Vec<f64> = returns.iter()
                    .filter(|&&r| r < 0.0)
                    .copied()
                    .collect();
                
                if !downside_returns.is_empty() {
                    let downside_variance = downside_returns.iter()
                        .map(|r| r.powi(2))
                        .sum::<f64>() / downside_returns.len() as f64;
                    let downside_dev = downside_variance.sqrt() * (252.0_f64.sqrt());
                    
                    if downside_dev > 0.0 {
                        metrics.sortino_ratio = (metrics.annualized_return / 100.0) / downside_dev;
                    }
                }
                
                // Calmar ratio
                if metrics.max_drawdown > 0.0 {
                    metrics.calmar_ratio = (metrics.annualized_return / 100.0) / metrics.max_drawdown;
                }
            }
        }
        
        // Cache the metrics
        self.metrics_by_period.insert(period, metrics.clone());
        
        metrics
    }
    
    /// Update symbol-specific metrics
    fn update_symbol_metrics(&self, symbol: &Symbol) {
        let trades: Vec<TradeRecord> = self.trade_history
            .read()
            .iter()
            .filter(|t| &t.symbol == symbol)
            .cloned()
            .collect();
        
        if trades.is_empty() {
            return;
        }
        
        let mut metrics = PerformanceMetrics::default();
        
        // Calculate symbol-specific metrics
        metrics.total_trades = trades.len() as u64;
        metrics.winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count() as u64;
        metrics.losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count() as u64;
        
        if metrics.total_trades > 0 {
            metrics.win_rate = (metrics.winning_trades as f64 / metrics.total_trades as f64) * 100.0;
        }
        
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let total_return_pct: f64 = trades.iter().map(|t| t.return_pct).sum();
        
        metrics.total_return = total_pnl;
        metrics.total_return_pct = total_return_pct / trades.len() as f64;
        
        self.metrics_by_symbol.insert(symbol.clone(), metrics);
    }
    
    /// Get metrics for a specific period
    pub fn get_metrics(&self, period: Period) -> PerformanceMetrics {
        self.metrics_by_period
            .get(&period)
            .map(|m| m.clone())
            .unwrap_or_else(|| self.calculate_metrics(period))
    }
    
    /// Get metrics for a specific symbol
    pub fn get_symbol_metrics(&self, symbol: &Symbol) -> Option<PerformanceMetrics> {
        self.metrics_by_symbol.get(symbol).map(|m| m.clone())
    }
    
    /// Get equity curve
    pub fn get_equity_curve(&self, period: Period) -> Vec<EquityPoint> {
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64 - period.duration().as_millis() as u64;
        
        self.equity_curve
            .read()
            .iter()
            .filter(|p| p.timestamp >= cutoff)
            .cloned()
            .collect()
    }
    
    /// Get trade history
    pub fn get_trade_history(&self, limit: Option<usize>) -> Vec<TradeRecord> {
        let history = self.trade_history.read();
        
        if let Some(limit) = limit {
            history.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        } else {
            history.clone()
        }
    }
    
    /// Get current statistics
    pub fn get_current_stats(&self) -> CurrentStats {
        CurrentStats {
            current_equity: *self.current_capital.read(),
            peak_equity: *self.peak_equity.read(),
            current_drawdown: (*self.peak_equity.read() - *self.current_capital.read()) / *self.peak_equity.read(),
            daily_returns_count: self.daily_returns.read().len(),
            total_trades: self.trade_history.read().len(),
        }
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        self.equity_curve.write().clear();
        self.metrics_by_period.clear();
        self.metrics_by_symbol.clear();
        self.daily_returns.write().clear();
        self.trade_history.write().clear();
        *self.current_capital.write() = self.initial_capital;
        *self.peak_equity.write() = self.initial_capital;
        *self.drawdown_start.write() = None;
    }
}

/// Current statistics snapshot
#[derive(Clone, Debug)]
pub struct CurrentStats {
    pub current_equity: f64,
    pub peak_equity: f64,
    pub current_drawdown: f64,
    pub daily_returns_count: usize,
    pub total_trades: usize,
}

/// Performance report generator
pub struct PerformanceReporter {
    analyzer: Arc<PerformanceAnalyzer>,
}

impl PerformanceReporter {
    pub fn new(analyzer: Arc<PerformanceAnalyzer>) -> Self {
        Self { analyzer }
    }
    
    /// Generate text report
    pub fn generate_text_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== PERFORMANCE REPORT ===\n\n");
        
        // All-time metrics
        let all_time = self.analyzer.get_metrics(Period::AllTime);
        report.push_str(&format!("Total Return: ${:.2} ({:.2}%)\n", 
            all_time.total_return, all_time.total_return_pct));
        report.push_str(&format!("Annualized Return: {:.2}%\n", all_time.annualized_return));
        report.push_str(&format!("Sharpe Ratio: {:.3}\n", all_time.sharpe_ratio));
        report.push_str(&format!("Max Drawdown: {:.2}%\n", all_time.max_drawdown * 100.0));
        report.push_str(&format!("Win Rate: {:.1}%\n", all_time.win_rate));
        report.push_str(&format!("Profit Factor: {:.2}\n", all_time.profit_factor));
        report.push_str(&format!("Total Trades: {}\n", all_time.total_trades));
        
        report.push_str("\n=== PERIOD BREAKDOWN ===\n");
        
        // Period metrics
        for period in &[Period::Day1, Period::Week1, Period::Month1] {
            let metrics = self.analyzer.get_metrics(period.clone());
            report.push_str(&format!("\n{:?}:\n", period));
            report.push_str(&format!("  Return: {:.2}%\n", metrics.total_return_pct));
            report.push_str(&format!("  Trades: {}\n", metrics.total_trades));
            report.push_str(&format!("  Win Rate: {:.1}%\n", metrics.win_rate));
        }
        
        report.push_str("\n=== TOP SYMBOLS ===\n");
        
        // Symbol performance (would need to track all symbols)
        // This is a simplified version
        
        report.push_str("\n=== RISK METRICS ===\n");
        report.push_str(&format!("Volatility: {:.2}%\n", all_time.volatility * 100.0));
        report.push_str(&format!("Sortino Ratio: {:.3}\n", all_time.sortino_ratio));
        report.push_str(&format!("Calmar Ratio: {:.3}\n", all_time.calmar_ratio));
        report.push_str(&format!("Largest Win: ${:.2}\n", all_time.largest_win));
        report.push_str(&format!("Largest Loss: ${:.2}\n", all_time.largest_loss));
        report.push_str(&format!("Max Consecutive Wins: {}\n", all_time.consecutive_wins));
        report.push_str(&format!("Max Consecutive Losses: {}\n", all_time.consecutive_losses));
        
        report
    }
    
    /// Generate JSON report
    pub fn generate_json_report(&self) -> Result<String> {
        let report = PerformanceReport {
            all_time: self.analyzer.get_metrics(Period::AllTime),
            daily: self.analyzer.get_metrics(Period::Day1),
            weekly: self.analyzer.get_metrics(Period::Week1),
            monthly: self.analyzer.get_metrics(Period::Month1),
            current_stats: self.analyzer.get_current_stats(),
            recent_trades: self.analyzer.get_trade_history(Some(10)),
        };
        
        Ok(serde_json::to_string_pretty(&report)?)
    }
}

/// Complete performance report structure
#[derive(Serialize, Deserialize)]
struct PerformanceReport {
    all_time: PerformanceMetrics,
    daily: PerformanceMetrics,
    weekly: PerformanceMetrics,
    monthly: PerformanceMetrics,
    current_stats: CurrentStatsJson,
    recent_trades: Vec<TradeRecord>,
}

#[derive(Serialize, Deserialize)]
struct CurrentStatsJson {
    current_equity: f64,
    peak_equity: f64,
    current_drawdown: f64,
    daily_returns_count: usize,
    total_trades: usize,
}

impl From<CurrentStats> for CurrentStatsJson {
    fn from(stats: CurrentStats) -> Self {
        Self {
            current_equity: stats.current_equity,
            peak_equity: stats.peak_equity,
            current_drawdown: stats.current_drawdown,
            daily_returns_count: stats.daily_returns_count,
            total_trades: stats.total_trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_analyzer() {
        let analyzer = PerformanceAnalyzer::new(10000.0);
        
        // Simulate equity updates
        analyzer.update_equity(10100.0);
        analyzer.update_equity(10200.0);
        analyzer.update_equity(10050.0);
        
        let stats = analyzer.get_current_stats();
        assert_eq!(stats.current_equity, 10050.0);
        assert_eq!(stats.peak_equity, 10200.0);
        
        let metrics = analyzer.calculate_metrics(Period::AllTime);
        assert!(metrics.total_return_pct > 0.0);
    }
    
    #[test]
    fn test_performance_reporter() {
        let analyzer = Arc::new(PerformanceAnalyzer::new(10000.0));
        let reporter = PerformanceReporter::new(analyzer.clone());
        
        analyzer.update_equity(11000.0);
        
        let report = reporter.generate_text_report();
        assert!(report.contains("PERFORMANCE REPORT"));
    }
}