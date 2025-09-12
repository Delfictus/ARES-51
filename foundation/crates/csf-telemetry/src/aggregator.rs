//! Metrics aggregation utilities

use hdrhistogram::Histogram;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Metrics aggregator for time-series data
pub struct Aggregator {
    windows: RwLock<HashMap<String, TimeWindow>>,
    config: AggregatorConfig,
}

#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Window duration (ms)
    pub window_duration_ms: u64,

    /// Maximum windows to keep
    pub max_windows: usize,

    /// Aggregation functions
    pub functions: Vec<AggregationFunction>,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregationFunction {
    Min,
    Max,
    Mean,
    Sum,
    Count,
    Percentile(f64),
    StdDev,
}

struct TimeWindow {
    start_time: u64,
    end_time: u64,
    histogram: Option<Histogram<u64>>,
    sum: f64,
    count: u64,
    min: f64,
    max: f64,
}

impl Aggregator {
    /// Create new aggregator
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            windows: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Add a value to aggregation
    pub fn add(&self, metric_name: &str, value: f64, timestamp: u64) {
        let window_start = (timestamp / (self.config.window_duration_ms * 1_000_000))
            * (self.config.window_duration_ms * 1_000_000);

        let mut windows = self.windows.write();
        let window = windows
            .entry(format!("{}:{}", metric_name, window_start))
            .or_insert_with(|| TimeWindow {
                start_time: window_start,
                end_time: window_start + self.config.window_duration_ms * 1_000_000,
                histogram: Histogram::new(3).ok().or_else(|| Histogram::new(1).ok()),
                sum: 0.0,
                count: 0,
                min: f64::MAX,
                max: f64::MIN,
            });

        // Update aggregates
        window.sum += value;
        window.count += 1;
        window.min = window.min.min(value);
        window.max = window.max.max(value);
        if let Some(h) = window.histogram.as_mut() {
            h.record((value * 1000.0) as u64).ok();
        }

        // Clean old windows
        if windows.len() > self.config.max_windows {
            let oldest = windows
                .keys()
                .min_by_key(|k| {
                    k.split(':')
                        .nth(1)
                        .unwrap_or("0")
                        .parse::<u64>()
                        .unwrap_or(0)
                })
                .cloned();

            if let Some(key) = oldest {
                windows.remove(&key);
            }
        }
    }

    /// Get aggregated values
    pub fn get_aggregates(&self, metric_name: &str) -> Vec<AggregatedMetric> {
        let windows = self.windows.read();
        let mut results = Vec::new();

        for (key, window) in windows.iter() {
            if key.starts_with(&format!("{}:", metric_name)) {
                for func in &self.config.functions {
                    let value = match func {
                        AggregationFunction::Min => window.min,
                        AggregationFunction::Max => window.max,
                        AggregationFunction::Mean => {
                            if window.count > 0 {
                                window.sum / window.count as f64
                            } else {
                                0.0
                            }
                        }
                        AggregationFunction::Sum => window.sum,
                        AggregationFunction::Count => window.count as f64,
                        AggregationFunction::Percentile(p) => window
                            .histogram
                            .as_ref()
                            .map(|h| h.value_at_percentile(*p) as f64 / 1000.0)
                            .unwrap_or(0.0),
                        AggregationFunction::StdDev => {
                            if let Some(h) = window.histogram.as_ref() {
                                // hdrhistogram stdev is over stored values
                                h.stdev() / 1000.0
                            } else {
                                0.0
                            }
                        }
                    };

                    results.push(AggregatedMetric {
                        name: metric_name.to_string(),
                        function: *func,
                        value,
                        window_start: window.start_time,
                        window_end: window.end_time,
                    });
                }
            }
        }

        results
    }
}

#[derive(Debug, Clone)]
pub struct AggregatedMetric {
    pub name: String,
    pub function: AggregationFunction,
    pub value: f64,
    pub window_start: u64,
    pub window_end: u64,
}
