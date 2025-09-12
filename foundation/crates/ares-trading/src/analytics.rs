use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::portfolio::Portfolio;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAnalytics {
    pub daily_returns: Vec<(DateTime<Utc>, Decimal)>,
    pub cumulative_returns: Vec<(DateTime<Utc>, Decimal)>,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub drawdown_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub win_loss_ratio: f64,
    pub average_trade_duration: chrono::Duration,
    pub best_day: (DateTime<Utc>, Decimal),
    pub worst_day: (DateTime<Utc>, Decimal),
}

pub struct AnalyticsEngine {
    history: Vec<Portfolio>,
    timestamps: Vec<DateTime<Utc>>,
}

impl AnalyticsEngine {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn record_snapshot(&mut self, portfolio: Portfolio) {
        self.timestamps.push(Utc::now());
        self.history.push(portfolio);
    }

    pub fn generate_analytics(&self) -> TradingAnalytics {
        let mut daily_returns = Vec::new();
        let mut cumulative_returns = Vec::new();
        let mut equity_curve = Vec::new();
        let mut drawdown_curve = Vec::new();
        
        let mut cumulative = rust_decimal_macros::dec!(0);
        let mut peak = rust_decimal_macros::dec!(0);
        
        for (i, portfolio) in self.history.iter().enumerate() {
            let timestamp = self.timestamps[i];
            let equity = portfolio.total_market_value() + portfolio.total_realized_pnl;
            equity_curve.push((timestamp, equity));
            
            if i > 0 {
                let prev_equity = equity_curve[i - 1].1;
                let daily_return = if prev_equity != rust_decimal_macros::dec!(0) {
                    (equity - prev_equity) / prev_equity
                } else {
                    rust_decimal_macros::dec!(0)
                };
                daily_returns.push((timestamp, daily_return));
                
                cumulative += daily_return;
                cumulative_returns.push((timestamp, cumulative));
            }
            
            peak = peak.max(equity);
            let drawdown = if peak != rust_decimal_macros::dec!(0) {
                (equity - peak) / peak
            } else {
                rust_decimal_macros::dec!(0)
            };
            drawdown_curve.push((timestamp, drawdown));
        }
        
        let best_day = daily_returns.iter()
            .max_by_key(|(_, r)| *r)
            .cloned()
            .unwrap_or((Utc::now(), rust_decimal_macros::dec!(0)));
        
        let worst_day = daily_returns.iter()
            .min_by_key(|(_, r)| *r)
            .cloned()
            .unwrap_or((Utc::now(), rust_decimal_macros::dec!(0)));
        
        TradingAnalytics {
            daily_returns,
            cumulative_returns,
            equity_curve,
            drawdown_curve,
            win_loss_ratio: 0.0, // Would calculate from trades
            average_trade_duration: chrono::Duration::hours(0),
            best_day,
            worst_day,
        }
    }

    pub fn export_csv(&self, path: &str) -> anyhow::Result<()> {
        // CSV export implementation would go here
        Ok(())
    }

    pub fn generate_report(&self) -> String {
        let analytics = self.generate_analytics();
        format!(
            "Trading Performance Report\n\
             ==========================\n\
             Total Snapshots: {}\n\
             Best Day: {:?}\n\
             Worst Day: {:?}\n\
             Win/Loss Ratio: {:.2}\n",
            self.history.len(),
            analytics.best_day,
            analytics.worst_day,
            analytics.win_loss_ratio
        )
    }
}