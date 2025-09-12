// Minimal, offline-only historical minute simulation with 2m-ahead predictions.
// - Synthetic data only (no network)
// - Clear progress output that starts immediately
// - Writes artifacts/market_predictions.csv compatible with the animator script

use std::fs;
use std::io::{self, Write, BufWriter};
use std::fs::File;
use serde_json::json;
use std::time::Instant;

// Optional resonance processor (enabled via RESONANCE=1)
use hephaestus_forge::resonance::DynamicResonanceProcessor;

const SYMBOL: &str = "AAPL";
const DEFAULT_MINUTES: usize = 1440; // 1 day of minutes
const WINDOW_SIZE: usize = 60;       // sliding window length for baseline predictor
const FORECAST_STEPS: usize = 390;   // approx. one trading day for forecast

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Starting minimal minute-level simulation (offline)");

    // 1) Determine size from env (optional)
    let minutes = std::env::var("HIST_MINUTES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(DEFAULT_MINUTES);
    let forecast_steps = std::env::var("FORECAST_MINUTES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(FORECAST_STEPS);

    println!("🧪 Generating synthetic data: {} minutes for {}", minutes, SYMBOL);
    io::stdout().flush().ok();
    let (timestamps, closes) = generate_synthetic_minutes(minutes, 100.0, 0.05);
    println!("✅ History generated: {} bars", closes.len());

    fs::create_dir_all("artifacts").ok();
    let csv_file = fs::File::create("artifacts/market_predictions.csv")?;
    let mut w = BufWriter::new(csv_file);
    let mut logger = JsonlLogger::new("artifacts/run_log.jsonl")?;
    logger.log("start", json!({
        "symbol": SYMBOL,
        "hist_minutes": minutes,
        "forecast_minutes": forecast_steps,
        "window": WINDOW_SIZE,
    }))?;
    writeln!(w, "phase,timestamp,symbol,close,predicted_2m,confidence")?;

    // 2) Optional resonance initialization
    let use_resonance = std::env::var("RESONANCE").ok().map(|v| v == "1").unwrap_or(false);
    if use_resonance { println!("🔧 Resonance mode: ON (RESONANCE=1)"); } else { println!("🔧 Resonance mode: OFF"); }
    let processor = if use_resonance {
        Some(hephaestus_forge::resonance::DynamicResonanceProcessor::new((8, 8, 4)).await)
    } else { None };

    // 3) History pass: produce 2m-ahead predictions for each bar
    let mut window: Vec<f64> = Vec::with_capacity(WINDOW_SIZE);
    let total_hist = closes.len();
    let hist_start = Instant::now();
    println!("📡 Processing history: {} bars", total_hist);
    for i in 0..total_hist {
        let ts = timestamps[i];
        let close = closes[i];
        push_window(&mut window, close, WINDOW_SIZE);
        if window.len() < WINDOW_SIZE { continue; }

        let (pred, conf) = if let Some(proc_ref) = &processor {
            predict_two_minutes_ahead_resonance(&window, proc_ref).await.unwrap_or_else(|| predict_two_minutes_ahead(&window))
        } else {
            predict_two_minutes_ahead(&window)
        };
        // Use current bar time + 120s to mark the 2m-ahead prediction timestamp
        let pred_ts = ts + 120;
        writeln!(w, "history,{},{},{:.6},{:.6},{:.4}", pred_ts, SYMBOL, close, pred, conf)?;

        if i % 200 == 0 || i + 1 == total_hist {
            let pct = ((i + 1) as f64 / total_hist.max(1) as f64) * 100.0;
            print!("   ▸ history {:>5}/{:>5} ({:>5.1}%)\r", i + 1, total_hist, pct);
            io::stdout().flush().ok();
            logger.log("history_progress", json!({
                "i": i+1, "total": total_hist, "pct": pct,
                "ts": ts, "close": close, "pred": pred, "conf": conf
            }))?;
        }
    }
    println!("");
    println!("✅ History complete in {:.2}s", hist_start.elapsed().as_secs_f64());
    logger.log("history_complete", json!({
        "bars": total_hist,
        "elapsed_sec": hist_start.elapsed().as_secs_f64()
    }))?;

    // 4) Forecast next day based on last close
    let mut sim_ts = timestamps.last().copied().unwrap_or(0);
    let mut sim_close = *closes.last().unwrap_or(&100.0);
    let fc_start = Instant::now();
    println!("🔮 Forecasting next day: {} minutes (t+2m predictions)", forecast_steps);
    for i in 0..forecast_steps {
        // Simple random walk continuation
        let noise = ((fastrand::f64() - 0.5) * 0.04) * sim_close / 100.0;
        sim_close = (sim_close + noise).max(0.01);
        sim_ts += 60;

        push_window(&mut window, sim_close, WINDOW_SIZE);
        let (pred, conf) = if window.len() >= WINDOW_SIZE {
            if let Some(proc_ref) = &processor {
                predict_two_minutes_ahead_resonance(&window, proc_ref).await.unwrap_or_else(|| predict_two_minutes_ahead(&window))
            } else {
                predict_two_minutes_ahead(&window)
            }
        } else { (sim_close, 0.0) };

        let pred_ts = sim_ts + 120;
        writeln!(w, "forecast,{},{},,{:.6},{:.4}", pred_ts, SYMBOL, pred, conf)?;

        if i % 50 == 0 || i + 1 == forecast_steps {
            let pct = ((i + 1) as f64 / forecast_steps as f64) * 100.0;
            print!("   ▸ forecast {:>4}/{:>4} ({:>5.1}%)\r", i + 1, forecast_steps, pct);
            io::stdout().flush().ok();
            logger.log("forecast_progress", json!({
                "i": i+1, "total": forecast_steps, "pct": pct,
                "sim_ts": sim_ts, "sim_close": sim_close, "pred": pred, "conf": conf
            }))?;
        }
    }
    println!("");
    println!("✅ Forecast complete in {:.2}s", fc_start.elapsed().as_secs_f64());
    logger.log("forecast_complete", json!({
        "steps": forecast_steps,
        "elapsed_sec": fc_start.elapsed().as_secs_f64()
    }))?;

    w.flush()?;
    logger.log("done", json!({ "csv": "artifacts/market_predictions.csv" }))?;
    println!("\n📝 Wrote predictions: artifacts/market_predictions.csv");
    println!("   Next: python3 scripts/animate_market_predictions.py");
    Ok(())
}

fn push_window(buf: &mut Vec<f64>, v: f64, max_len: usize) {
    if buf.len() == max_len { buf.remove(0); }
    buf.push(v);
}

// Very simple baseline predictor using SMA + trend extrapolation
fn predict_two_minutes_ahead(window: &[f64]) -> (f64, f64) {
    let n = window.len();
    let mean = window.iter().copied().sum::<f64>() / n as f64;
    let last = window[n - 1];
    let prev = window[n - 2];
    let slope = last - prev; // last minute delta
    let pred = last + 2.0 * slope * 0.5 + 0.5 * (last - mean); // modest drift + mean reversion
    let conf = (1.0 - (last - mean).abs() / (mean.abs() + 1e-6)).clamp(0.0, 1.0);
    (pred, conf)
}

async fn predict_two_minutes_ahead_resonance(
    window: &[f64],
    processor: &DynamicResonanceProcessor,
) -> Option<(f64, f64)> {
    // Normalize window to zero-mean, unit-variance
    let n = window.len();
    if n == 0 { return None; }
    let mean = window.iter().copied().sum::<f64>() / n as f64;
    let var = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-9);
    let norm: Vec<f64> = window.iter().map(|v| (v - mean) / std).collect();

    let input = hephaestus_forge::resonance::ComputationTensor::from_vec(norm);
    let result = processor.process_via_resonance(input).await.ok()?;
    let s = result.solution_tensor.as_slice();
    if s.is_empty() { return None; }
    let avg = s.iter().copied().sum::<f64>() / s.len() as f64;
    let last = *window.last().unwrap();
    let conf = (avg.abs().min(1.0)).clamp(0.0, 1.0);
    let delta = avg.tanh() * last * 0.002; // modest sway
    Some((last + delta, conf))
}

fn generate_synthetic_minutes(minutes: usize, start_price: f64, tick: f64) -> (Vec<i64>, Vec<f64>) {
    let mut out_ts = Vec::with_capacity(minutes);
    let mut out_close = Vec::with_capacity(minutes);
    let mut price = start_price;
    let mut ts: i64 = 1_700_000_000; // fixed epoch baseline for reproducibility

    for _ in 0..minutes {
        // Daily-like cycle + noise
        let phase = (ts % 86400) as f64 / 86400.0;
        let daily = (std::f64::consts::TAU * phase).sin() * tick;
        let noise = (fastrand::f64() - 0.5) * tick;
        price = (price + daily + noise).max(0.01);

        out_ts.push(ts);
        out_close.push(price);
        ts += 60;
    }
    (out_ts, out_close)
}

struct JsonlLogger {
    w: BufWriter<File>,
}

impl JsonlLogger {
    fn new(path: &str) -> io::Result<Self> {
        let f = File::create(path)?;
        Ok(Self { w: BufWriter::new(f) })
    }
    fn log(&mut self, event: &str, value: serde_json::Value) -> io::Result<()> {
        let rec = json!({
            "ts_ms": now_ms(),
            "event": event,
            "data": value,
        });
        writeln!(self.w, "{}", rec.to_string())?;
        self.w.flush()
    }
}

fn now_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}
