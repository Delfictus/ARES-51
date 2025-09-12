pub mod yahoo_finance;
pub mod polygon_io;
pub mod iex_cloud;
pub mod binance;
pub mod finnhub;
pub mod twelve_data;

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use chrono::{DateTime, Utc};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::market_data::{Quote, Candle, OrderBook, MarketDataProvider};

/// Configuration for different market data providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub provider: ProviderType,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub base_url: Option<String>,
    pub rate_limit_per_minute: Option<u32>,
    pub enable_websocket: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderType {
    YahooFinance,
    PolygonIo,
    IexCloud,
    Binance,
    Finnhub,
    TwelveData,
    AlphaVantage,
}

/// Factory for creating market data providers
pub struct MarketDataProviderFactory;

impl MarketDataProviderFactory {
    pub async fn create(config: MarketDataConfig) -> Result<Box<dyn MarketDataProvider>> {
        match config.provider {
            ProviderType::YahooFinance => {
                Ok(Box::new(yahoo_finance::YahooFinanceProvider::new(config)?))
            }
            ProviderType::PolygonIo => {
                Ok(Box::new(polygon_io::PolygonProvider::new(config)?))
            }
            ProviderType::IexCloud => {
                Ok(Box::new(iex_cloud::IexCloudProvider::new(config)?))
            }
            ProviderType::Binance => {
                Ok(Box::new(binance::BinanceProvider::new(config)?))
            }
            ProviderType::Finnhub => {
                Ok(Box::new(finnhub::FinnhubProvider::new(config)?))
            }
            ProviderType::TwelveData => {
                Ok(Box::new(twelve_data::TwelveDataProvider::new(config)?))
            }
            ProviderType::AlphaVantage => {
                Err(anyhow::anyhow!("AlphaVantage provider already implemented in market_data.rs"))
            }
        }
    }
}

/// Common utilities for parsing market data responses
pub mod utils {
    use super::*;
    
    pub fn parse_decimal(value: &serde_json::Value, field: &str) -> Result<Decimal> {
        if let Some(v) = value.get(field) {
            if let Some(s) = v.as_str() {
                s.parse::<Decimal>()
                    .map_err(|_| anyhow::anyhow!("Failed to parse field: {}", field))
            } else if let Some(f) = v.as_f64() {
                Ok(Decimal::from_f64_retain(f).unwrap_or_default())
            } else {
                Err(anyhow::anyhow!("Failed to parse field: {}", field))
            }
        } else {
            Err(anyhow::anyhow!("Field not found: {}", field))
        }
    }
    
    pub fn parse_timestamp(value: &serde_json::Value, field: &str) -> Result<DateTime<Utc>> {
        value.get(field)
            .and_then(|v| v.as_i64())
            .and_then(|ts| DateTime::from_timestamp(ts, 0))
            .ok_or_else(|| anyhow::anyhow!("Failed to parse timestamp: {}", field))
    }
}