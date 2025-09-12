use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::portfolio::{Portfolio, Position};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk: Decimal,
    pub sharpe_ratio: f64,
    pub max_drawdown: Decimal,
    pub position_concentration: f64,
    pub margin_usage: Decimal,
    pub exposure: Decimal,
}

pub struct RiskManager {
    max_position_size: Decimal,
    max_portfolio_risk: Decimal,
    max_drawdown_limit: Decimal,
}

impl RiskManager {
    pub fn new() -> Self {
        Self {
            max_position_size: dec!(10000),
            max_portfolio_risk: dec!(0.1), // 10% max risk
            max_drawdown_limit: dec!(0.2),  // 20% max drawdown
        }
    }

    pub fn calculate_risk_metrics(&self, portfolio: &Portfolio) -> RiskMetrics {
        let total_value = portfolio.total_market_value();
        let exposure = portfolio.positions.values()
            .map(|p| p.market_value().abs())
            .sum();
        
        RiskMetrics {
            value_at_risk: self.calculate_var(portfolio),
            sharpe_ratio: self.calculate_sharpe_ratio(portfolio),
            max_drawdown: self.calculate_max_drawdown(portfolio),
            position_concentration: self.calculate_concentration(portfolio),
            margin_usage: dec!(0), // Would be calculated from account
            exposure,
        }
    }

    fn calculate_var(&self, portfolio: &Portfolio) -> Decimal {
        // Simplified VaR calculation
        portfolio.total_market_value() * dec!(0.05)
    }

    fn calculate_sharpe_ratio(&self, portfolio: &Portfolio) -> f64 {
        // Simplified Sharpe ratio
        let returns = portfolio.total_pnl() / portfolio.total_cost_basis().max(dec!(1));
        returns.to_f64().unwrap_or(0.0) / 0.15 // Assuming 15% volatility
    }

    fn calculate_max_drawdown(&self, portfolio: &Portfolio) -> Decimal {
        // Simplified max drawdown
        portfolio.positions.values()
            .map(|p| p.unrealized_pnl.min(dec!(0)))
            .sum::<Decimal>()
            .abs()
    }

    fn calculate_concentration(&self, portfolio: &Portfolio) -> f64 {
        if portfolio.positions.is_empty() {
            return 0.0;
        }
        
        let total = portfolio.total_market_value();
        if total == dec!(0) {
            return 0.0;
        }
        
        let max_position = portfolio.positions.values()
            .map(|p| p.market_value())
            .max()
            .unwrap_or(dec!(0));
        
        (max_position / total).to_f64().unwrap_or(0.0)
    }

    pub fn check_risk_limits(&self, position: &Position) -> bool {
        position.market_value() <= self.max_position_size
    }
}