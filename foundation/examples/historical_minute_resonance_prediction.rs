//! Historical minute-level simulation through resonance with 2-min ahead predictions
//!
//! - Fetches a few days of 1-minute bars (Yahoo)
//! - Streams through DynamicResonanceProcessor using a sliding window
//! - After history, simulates next-day timeline producing 2-minute-ahead predictions
//! - Writes CSV suitable for animation: artifacts/market_predictions.csv

use chrono::{Duration, Utc, TimeZone, NaiveDateTime};
use csf_core::prelude::*;
use hephaestus_forge::resonance::{ComputationTensor, DynamicResonanceProcessor};
use std::fs;
use std::io::{self, Write};
use std::time::Instant;
use tokio::time::sleep;
use std::time::Duration as StdDuration;

const SYMBOL: &str = "AAPL";
const WINDOW_SIZE: usize = 256; // sliding window length for resonance input
const HISTORY_DAYS: i64 = 3;    // number of past days to fetch
const PREDICT_MINUTES: usize = 390; // ~one trading day of minutes

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Minute-level history → resonance → 2-min predictions");

    // 1) Configure historical fetch for minute data
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec![SYMBOL.to_string()],
        start_date: Utc::now() - Duration::days(HISTORY_DAYS),
        end_date: Utc::now(),
        interval: TimeInterval::OneMinute,
        playback_speed: 10000.0,
        max_retries: 3,
        rate_limit_ms: 400,
    };

    let mut fetcher = HistoricalDataFetcher::new(config.clone());

    println!("🌐 Fetching {} days of 1m bars for {}…", HISTORY_DAYS, SYMBOL);
    io::stdout().flush().ok();

    let offline = std::env::var("ARES_OFFLINE").ok().map(|v| v == "1").unwrap_or(false);
    let history = if offline {
        eprintln!("⚠️  ARES_OFFLINE=1 → using synthetic data");
        generate_synthetic_minutes(SYMBOL, HISTORY_DAYS)
    } else {
        match tokio::time::timeout(std::time::Duration::from_secs(6), fetcher.fetch_symbol_data(SYMBOL)).await {
            Ok(Ok(v)) if !v.is_empty() => v,
            Ok(Ok(_)) | Ok(Err(_)) | Err(_) => {
                eprintln!("⚠️  Network slow/unavailable → using synthetic data");
                generate_synthetic_minutes(SYMBOL, HISTORY_DAYS)
            }
        }
    };

    println!("✅ History points: {} ({} → {})",
        history.len(),
        history.first().map(|p| p.timestamp).unwrap_or(Utc::now())
            .format("%Y-%m-%d %H:%M"),
        history.last().map(|p| p.timestamp).unwrap_or(Utc::now())
            .format("%Y-%m-%d %H:%M")
    );

    // 2) Init resonance processor
    let processor = DynamicResonanceProcessor::new((16, 16, 8)).await;

    // 3) Prepare output CSV
    fs::create_dir_all("artifacts").ok();
    let mut file = fs::File::create("artifacts/market_predictions.csv")?;
    writeln!(file, "phase,timestamp,symbol,close,predicted_2m,confidence")?;

    // 4) Stream history minute-by-minute through resonance
    let mut window: Vec<f64> = Vec::with_capacity(WINDOW_SIZE);
    let mut closes: Vec<(i64, f64)> = history
        .iter()
        .map(|p| (p.timestamp.timestamp(), p.close))
        .collect();

    // Optional throttles for CI/sandbox: ARES_MAX_HISTORY, ARES_MAX_FORECAST
    if let Ok(env_val) = std::env::var("ARES_MAX_HISTORY") {
        if let Ok(max_hist) = env_val.parse::<usize>() {
            if closes.len() > max_hist { closes.truncate(max_hist); }
            println!("⚙️  Limiting history to {} bars via ARES_MAX_HISTORY", closes.len());
        }
    }

    let mut prev_ts: Option<i64> = None;
    let total_hist = closes.len();
    let hist_start = Instant::now();
    println!("📡 Processing history: {} bars at {:.0}x speed", total_hist, config.playback_speed);
    for (idx, (ts_sec, close)) in closes.iter().enumerate() {
        if let Some(prev) = prev_ts {
            let dt_ms = (*ts_sec - prev).max(0) as u64 * 1000;
            let sleep_ms = (dt_ms as f64 / config.playback_speed) as u64;
            if sleep_ms > 0 { sleep(StdDuration::from_millis(sleep_ms)).await; }
        }
        prev_ts = Some(*ts_sec);
        push_window(&mut window, *close, WINDOW_SIZE);
        if window.len() < WINDOW_SIZE { continue; }

        // Normalize window for stable resonance processing
        let input = normalize(&window);
        let tensor = ComputationTensor::from_vec(input);
        let result = processor.process_via_resonance(tensor).await;

        // Derive predicted delta from solution tensor energy (demo heuristic)
        let (pred, conf) = result
            .ok()
            .map(|sol| predict_from_solution(*close, &sol.solution_tensor))
            .unwrap_or((*close, 0.0));

        // Prediction is for t+2 minutes
        let pred_ts = *ts_sec + 120; // 2 minutes ahead
        writeln!(
            file,
            "history,{},{},{:.6},{:.6},{:.4}",
            ts_sec_to_iso(pred_ts),
            SYMBOL,
            *close,
            pred,
            conf
        )?;

        if idx % 100 == 0 || idx + 1 == total_hist {
            let pct = ((idx + 1) as f64 / total_hist.max(1) as f64) * 100.0;
            print!("   ▸ history {:>5}/{:>5} ({:>5.1}%)\r", idx + 1, total_hist, pct);
            io::stdout().flush().ok();
        }
    }
    println!("");
    println!("✅ History complete in {:.2}s", hist_start.elapsed().as_secs_f64());

    // 5) Forecast next day: roll buffer forward with last known close
    let (last_ts, last_close) = closes
        .last()
        .copied()
        .unwrap_or((Utc::now().timestamp(), *window.last().unwrap_or(&100.0)));
    let mut sim_ts = last_ts;
    let mut sim_close = last_close;
    let mut rng = oorandom::Rand32::new(42);

    let mut forecast_steps = PREDICT_MINUTES;
    if let Ok(env_val) = std::env::var("ARES_MAX_FORECAST") {
        if let Ok(max_fc) = env_val.parse::<usize>() {
            forecast_steps = max_fc;
            println!("⚙️  Limiting forecast to {} minutes via ARES_MAX_FORECAST", forecast_steps);
        }
    }

    println!("🔮 Forecasting next day: {} minutes (t+2m predictions)", forecast_steps);
    let fc_start = Instant::now();
    for i in 0..forecast_steps {
        // Drift the close slightly to emulate plausible evolution
        let noise = ((rng.rand_float() as f64) - 0.5) * 0.04 * sim_close / 100.0;
        sim_close = (sim_close + noise).max(0.01);
        sim_ts += 60; // advance by one minute

        push_window(&mut window, sim_close, WINDOW_SIZE);
        let input = normalize(&window);
        let tensor = ComputationTensor::from_vec(input);
        let result = processor.process_via_resonance(tensor).await;

        let (pred, conf) = result
            .ok()
            .map(|sol| predict_from_solution(sim_close, &sol.solution_tensor))
            .unwrap_or((sim_close, 0.0));

        let pred_ts = sim_ts + 120; // 2 minutes ahead of buffer head
        writeln!(
            file,
            "forecast,{},{},,{:.6},{:.4}",
            ts_sec_to_iso(pred_ts),
            SYMBOL,
            pred,
            conf
        )?;
        if i % 50 == 0 || i + 1 == forecast_steps {
            let pct = ((i + 1) as f64 / forecast_steps as f64) * 100.0;
            print!("   ▸ forecast {:>4}/{:>4} ({:>5.1}%)\r", i + 1, forecast_steps, pct);
            io::stdout().flush().ok();
        }
    }
    println!("");
    println!("✅ Forecast complete in {:.2}s", fc_start.elapsed().as_secs_f64());

    println!("\n📝 Wrote predictions: artifacts/market_predictions.csv");
    println!("   Next: run scripts/animate_market_predictions.py to create GIF");
    Ok(())
}

fn push_window(buf: &mut Vec<f64>, v: f64, max_len: usize) {
    if buf.len() == max_len { buf.remove(0); }
    buf.push(v);
}

fn normalize(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    let mean = x.iter().copied().sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = var.sqrt().max(1e-9);
    x.iter().map(|v| (v - mean) / std).collect()
}

fn predict_from_solution(last_close: f64, tensor: &ComputationTensor) -> (f64, f64) {
    let s = tensor.as_slice();
    if s.is_empty() { return (last_close, 0.0); }
    let avg = s.iter().copied().sum::<f64>() / s.len() as f64;
    let conf = (avg.abs().min(1.0)).clamp(0.0, 1.0);
    // Map resonance response to a modest delta
    let delta = avg.tanh() * last_close * 0.002; // ±0.2% sway
    (last_close + delta, conf)
}

fn ts_sec_to_iso(ts: i64) -> String {
    let ndt = NaiveDateTime::from_timestamp_opt(ts, 0).unwrap_or(NaiveDateTime::UNIX_EPOCH);
    Utc.from_utc_datetime(&ndt).format("%Y-%m-%d %H:%M:%S").to_string()
}

fn generate_synthetic_minutes(symbol: &str, days: i64) -> Vec<HistoricalDataPoint> {
    let mut out = Vec::new();
    let start = Utc::now() - Duration::days(days);
    let mut ts = start;
    let mut price: f64 = 100.0;
    let mut rng = oorandom::Rand32::new(123);
    let total = (days * 24 * 60) as i64;
    for _ in 0..total {
        let drift = ((rng.rand_float() as f64) - 0.5) * 0.05; // ±5c approx
        price = (price + drift).max(0.01);
        out.push(HistoricalDataPoint {
            timestamp: ts,
            symbol: symbol.to_string(),
            open: price,
            high: price * 1.001,
            low: price * 0.999,
            close: price,
            volume: 10_000.0,
            adjusted_close: None,
        });
        ts += Duration::minutes(1);
    }
    out
}
