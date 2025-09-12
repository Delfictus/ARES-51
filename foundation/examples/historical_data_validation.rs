//! Historical Market Data Validation Example for ARES ChronoFabric
//!
//! Demonstrates how to fetch historical data and replay it through the system
//! for model validation and backtesting.

use chrono::{Duration, Utc};
use csf_core::prelude::*;
use std::collections::HashMap;
use tokio::time::{sleep, Duration as TokioDuration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 ARES ChronoFabric Historical Data Validation");
    println!("===============================================");

    // Configuration for historical data fetching
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec![
            "AAPL".to_string(),
            "GOOGL".to_string(),
            "MSFT".to_string(),
            "TSLA".to_string(),
        ],
        start_date: Utc::now() - Duration::days(30), // Last 30 days
        end_date: Utc::now(),
        interval: TimeInterval::Daily,
        playback_speed: 10.0, // 10x speed for demonstration
        max_retries: 3,
        rate_limit_ms: 1000, // 1 second between API calls
    };

    println!("📊 Fetching historical data for {} symbols...", config.symbols.len());
    println!("   Symbols: {:?}", config.symbols);
    println!("   Date range: {} to {}", config.start_date.format("%Y-%m-%d"), config.end_date.format("%Y-%m-%d"));
    println!("   Interval: {:?}", config.interval);

    // Create historical data fetcher
    let mut fetcher = HistoricalDataFetcher::new(config.clone());

    // Fetch all historical data
    match fetcher.fetch_all_data().await {
        Ok(all_data) => {
            println!("✅ Successfully fetched historical data for {} symbols", all_data.len());
            
            // Process each symbol's data
            for (symbol, data_points) in &all_data {
                println!("\n📈 Symbol: {} ({} data points)", symbol, data_points.len());
                
                if !data_points.is_empty() {
                    let first_point = &data_points[0];
                    let last_point = &data_points[data_points.len() - 1];
                    
                    println!("   First data point: {}", first_point.timestamp.format("%Y-%m-%d"));
                    println!("   Last data point: {}", last_point.timestamp.format("%Y-%m-%d"));
                    println!("   Price range: ${:.2} - ${:.2}", 
                        data_points.iter().map(|p| p.low).fold(f64::INFINITY, f64::min),
                        data_points.iter().map(|p| p.high).fold(0.0, f64::max)
                    );
                    println!("   Average volume: {:.0}", 
                        data_points.iter().map(|p| p.volume).sum::<f64>() / data_points.len() as f64
                    );

                    // Demonstrate streaming replay
                    println!("   🔄 Starting streaming replay at {}x speed...", config.playback_speed);
                    
                    let symbol_data = data_points.clone();
                    let mut processed_count = 0;
                    let mut total_value = 0.0;
                    
                    // Replay data with timing control
                    fetcher.replay_data_with_timing(symbol_data, |stream_data| {
                        // Process the streaming data (this would normally go to your quantum system)
                        if let csf_core::hpc::streaming_processor::DataPayload::Points { points, .. } = &stream_data.payload {
                            if let Some(point_data) = points.first() {
                                if point_data.len() >= 4 {
                                    let close_price = point_data[3]; // close price
                                    total_value += close_price;
                                    processed_count += 1;
                                    
                                    if processed_count % 5 == 0 || processed_count <= 3 {
                                        println!("     📊 Processed point {}: Close=${:.2} ({})", 
                                            processed_count, 
                                            close_price,
                                            stream_data.timestamp.format("%Y-%m-%d")
                                        );
                                    }
                                }
                            }
                        }
                    }).await?;
                    
                    if processed_count > 0 {
                        println!("   ✅ Replay complete! Processed {} data points", processed_count);
                        println!("   📊 Average close price: ${:.2}", total_value / processed_count as f64);
                    }
                } else {
                    println!("   ⚠️  No data points found for {}", symbol);
                }
            }

            // Demonstrate using alternative data source (Alpha Vantage)
            println!("\n🔄 Testing Alpha Vantage data source...");
            println!("   (Note: Requires API key - demo will show error handling)");
            
            let alpha_config = HistoricalDataConfig {
                data_source: DataSource::AlphaVantage { 
                    api_key: "demo".to_string() // Invalid key for demo
                },
                symbols: vec!["AAPL".to_string()],
                start_date: Utc::now() - Duration::days(7),
                end_date: Utc::now(),
                interval: TimeInterval::Daily,
                playback_speed: 1.0,
                max_retries: 1,
                rate_limit_ms: 0,
            };
            
            let mut alpha_fetcher = HistoricalDataFetcher::new(alpha_config);
            
            match alpha_fetcher.fetch_symbol_data("AAPL").await {
                Ok(data) => {
                    println!("   ✅ Alpha Vantage data fetched: {} points", data.len());
                }
                Err(e) => {
                    println!("   ❌ Alpha Vantage error (expected with demo key): {}", e);
                }
            }

            // Show integration with existing market data system
            println!("\n🔗 Integration with existing market data system:");
            
            if let Some(aapl_data) = all_data.get("AAPL") {
                if let Some(latest_point) = aapl_data.last() {
                    // Convert to existing MarketDataPoint format from market_data_integration.rs
                    let market_point = crate::examples::market_data_integration::MarketDataPoint {
                        timestamp: latest_point.timestamp.timestamp_millis() as u64,
                        symbol: latest_point.symbol.clone(),
                        open: latest_point.open,
                        high: latest_point.high,
                        low: latest_point.low,
                        close: latest_point.close,
                        volume: latest_point.volume,
                        bid: latest_point.close - 0.01, // Simulate bid/ask spread
                        ask: latest_point.close + 0.01,
                    };
                    
                    println!("   📊 Latest AAPL converted to MarketDataPoint:");
                    println!("      Symbol: {}", market_point.symbol);
                    println!("      Close: ${:.2}", market_point.close);
                    println!("      Volume: {:.0}", market_point.volume);
                    
                    // Process through existing market data system
                    let market_data_vec = vec![market_point];
                    
                    // This would normally process through your quantum temporal system
                    match crate::examples::market_data_integration::process_market_data(market_data_vec).await {
                        Ok(prediction) => {
                            println!("   🔮 Quantum prediction generated:");
                            println!("      Predicted price: ${:.2}", prediction.predicted_price);
                            println!("      Confidence: {:.2}%", prediction.confidence * 100.0);
                            println!("      Trend: {:?}", prediction.trend_direction);
                        }
                        Err(e) => {
                            println!("   ❌ Prediction processing error: {}", e);
                        }
                    }
                }
            }

            println!("\n✨ Validation complete! Historical data system ready for:");
            println!("   • Model backtesting with real market data");
            println!("   • Performance validation across different time periods");
            println!("   • Multi-symbol correlation analysis");
            println!("   • Risk assessment using historical volatility");
            
        }
        Err(e) => {
            println!("❌ Error fetching historical data: {}", e);
            println!("\nTroubleshooting tips:");
            println!("• Check internet connection");
            println!("• Yahoo Finance may have rate limits");
            println!("• Try with fewer symbols or shorter date range");
            println!("• Consider using Alpha Vantage with API key for production");
        }
    }

    Ok(())
}

// Module path adjustments for example compilation
use crate::examples;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_historical_data_config() {
        let config = HistoricalDataConfig {
            data_source: DataSource::YahooFinance,
            symbols: vec!["AAPL".to_string()],
            start_date: Utc::now() - Duration::days(7),
            end_date: Utc::now(),
            interval: TimeInterval::Daily,
            playback_speed: 1.0,
            max_retries: 3,
            rate_limit_ms: 1000,
        };

        let fetcher = HistoricalDataFetcher::new(config);
        assert!(fetcher.get_cached_data("AAPL").is_none());
    }

    #[test]
    fn test_time_interval_conversions() {
        assert_eq!(TimeInterval::Daily.to_yahoo_string(), "1d");
        assert_eq!(TimeInterval::OneHour.to_yahoo_string(), "1h");
        assert_eq!(TimeInterval::FiveMinutes.to_yahoo_string(), "5m");
    }
}