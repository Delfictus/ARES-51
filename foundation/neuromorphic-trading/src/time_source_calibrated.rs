//! Calibrated hardware clock with drift compensation
//! 
//! Features:
//! - TSC frequency calibration
//! - Drift detection and compensation
//! - NTP synchronization
//! - Temperature-based adjustments

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, Duration, Instant};
use std::thread;
use std::fs;
use parking_lot::RwLock;
use anyhow::{Result, Context};

/// TSC calibration data
#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub tsc_frequency: u64,
    pub reference_tsc: u64,
    pub reference_wall_ns: u64,
    pub drift_ppb: i64,  // Parts per billion
    pub temperature_coefficient: f64,  // Hz per degree C
    pub last_calibration: SystemTime,
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self {
            tsc_frequency: 3_000_000_000,  // 3 GHz default
            reference_tsc: 0,
            reference_wall_ns: 0,
            drift_ppb: 0,
            temperature_coefficient: 0.0,
            last_calibration: SystemTime::now(),
        }
    }
}

/// Drift compensation model
struct DriftModel {
    samples: Vec<(u64, i64)>,  // (timestamp_ns, drift_ns)
    linear_coefficient: f64,
    quadratic_coefficient: f64,
    last_update: Instant,
}

impl DriftModel {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(1000),
            linear_coefficient: 0.0,
            quadratic_coefficient: 0.0,
            last_update: Instant::now(),
        }
    }
    
    fn add_sample(&mut self, timestamp_ns: u64, drift_ns: i64) {
        self.samples.push((timestamp_ns, drift_ns));
        
        // Keep last 1000 samples
        if self.samples.len() > 1000 {
            self.samples.remove(0);
        }
        
        // Recompute model every 100 samples
        if self.samples.len() % 100 == 0 {
            self.compute_model();
        }
    }
    
    fn compute_model(&mut self) {
        if self.samples.len() < 10 {
            return;
        }
        
        // Least squares fitting for drift model
        // drift = a * t + b * t^2
        
        let n = self.samples.len() as f64;
        let mut sum_t = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t3 = 0.0;
        let mut sum_t4 = 0.0;
        let mut sum_d = 0.0;
        let mut sum_td = 0.0;
        let mut sum_t2d = 0.0;
        
        let t0 = self.samples[0].0 as f64;
        
        for &(timestamp, drift) in &self.samples {
            let t = (timestamp as f64 - t0) / 1e9;  // Convert to seconds
            let d = drift as f64;
            
            sum_t += t;
            sum_t2 += t * t;
            sum_t3 += t * t * t;
            sum_t4 += t * t * t * t;
            sum_d += d;
            sum_td += t * d;
            sum_t2d += t * t * d;
        }
        
        // Solve normal equations
        let det = n * sum_t2 * sum_t4 + sum_t * sum_t3 * sum_t2 + sum_t2 * sum_t * sum_t3
                - sum_t2 * sum_t2 * sum_t2 - sum_t * sum_t * sum_t4 - n * sum_t3 * sum_t3;
        
        if det.abs() > 1e-10 {
            self.linear_coefficient = (sum_td * (n * sum_t4 - sum_t2 * sum_t2) 
                                     + sum_d * (sum_t2 * sum_t3 - sum_t * sum_t4)
                                     + sum_t2d * (sum_t * sum_t2 - n * sum_t3)) / det;
            
            self.quadratic_coefficient = (sum_t2d * (n * sum_t2 - sum_t * sum_t)
                                        + sum_td * (sum_t * sum_t2 - n * sum_t3)
                                        + sum_d * (sum_t * sum_t3 - sum_t2 * sum_t2)) / det;
        }
        
        self.last_update = Instant::now();
    }
    
    fn predict_drift(&self, timestamp_ns: u64) -> i64 {
        if self.samples.is_empty() {
            return 0;
        }
        
        let t0 = self.samples[0].0 as f64;
        let t = (timestamp_ns as f64 - t0) / 1e9;
        
        (self.linear_coefficient * t + self.quadratic_coefficient * t * t) as i64
    }
}

/// Temperature sensor reader
struct TemperatureSensor {
    cpu_temp_path: Option<String>,
    last_temp: AtomicI64,  // Millidegrees Celsius
}

impl TemperatureSensor {
    fn new() -> Self {
        // Find CPU temperature sensor
        let mut cpu_temp_path = None;
        
        // Try common hwmon paths
        for i in 0..10 {
            let path = format!("/sys/class/hwmon/hwmon{}/temp1_input", i);
            if fs::metadata(&path).is_ok() {
                cpu_temp_path = Some(path);
                break;
            }
        }
        
        // Try thermal zone
        if cpu_temp_path.is_none() {
            let path = "/sys/class/thermal/thermal_zone0/temp";
            if fs::metadata(&path).is_ok() {
                cpu_temp_path = Some(path.to_string());
            }
        }
        
        Self {
            cpu_temp_path,
            last_temp: AtomicI64::new(25000),  // Default 25°C
        }
    }
    
    fn read_temperature(&self) -> i64 {
        if let Some(ref path) = self.cpu_temp_path {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(temp) = content.trim().parse::<i64>() {
                    self.last_temp.store(temp, Ordering::Relaxed);
                    return temp;
                }
            }
        }
        
        self.last_temp.load(Ordering::Relaxed)
    }
    
    fn get_temperature_celsius(&self) -> f64 {
        self.read_temperature() as f64 / 1000.0
    }
}

/// Calibrated hardware clock
pub struct CalibratedClock {
    calibration: Arc<RwLock<CalibrationData>>,
    drift_model: Arc<RwLock<DriftModel>>,
    temperature_sensor: Arc<TemperatureSensor>,
    reference_temp: AtomicI64,  // Reference temperature during calibration
    calibration_thread: Option<thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
    ntp_sync_enabled: AtomicBool,
}

impl CalibratedClock {
    pub fn new() -> Result<Self> {
        let mut clock = Self {
            calibration: Arc::new(RwLock::new(CalibrationData::default())),
            drift_model: Arc::new(RwLock::new(DriftModel::new())),
            temperature_sensor: Arc::new(TemperatureSensor::new()),
            reference_temp: AtomicI64::new(25000),  // 25°C
            calibration_thread: None,
            running: Arc::new(AtomicBool::new(false)),
            ntp_sync_enabled: AtomicBool::new(false),
        };
        
        // Initial calibration
        clock.calibrate_frequency()?;
        
        // Start background calibration thread
        clock.start_calibration_thread();
        
        Ok(clock)
    }
    
    /// Calibrate TSC frequency
    fn calibrate_frequency(&mut self) -> Result<()> {
        // Method 1: Read from kernel
        if let Ok(freq) = self.read_kernel_tsc_frequency() {
            self.calibration.write().tsc_frequency = freq;
            return Ok(());
        }
        
        // Method 2: Measure against wall clock
        let freq = self.measure_tsc_frequency()?;
        self.calibration.write().tsc_frequency = freq;
        
        Ok(())
    }
    
    fn read_kernel_tsc_frequency(&self) -> Result<u64> {
        // Try to read from /proc/cpuinfo
        let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
        
        for line in cpuinfo.lines() {
            if line.starts_with("cpu MHz") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() == 2 {
                    if let Ok(mhz) = parts[1].trim().parse::<f64>() {
                        return Ok((mhz * 1_000_000.0) as u64);
                    }
                }
            }
        }
        
        // Try to read from sysfs
        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") {
            if let Ok(khz) = content.trim().parse::<u64>() {
                return Ok(khz * 1000);
            }
        }
        
        Err(anyhow::anyhow!("Could not read TSC frequency from kernel"))
    }
    
    fn measure_tsc_frequency(&self) -> Result<u64> {
        // Measure TSC ticks over a known time interval
        let duration = Duration::from_millis(100);
        
        let start_wall = Instant::now();
        let start_tsc = self.read_tsc();
        
        thread::sleep(duration);
        
        let end_tsc = self.read_tsc();
        let elapsed = start_wall.elapsed();
        
        let tsc_delta = end_tsc - start_tsc;
        let time_ns = elapsed.as_nanos() as u64;
        
        let frequency = (tsc_delta as f64 * 1_000_000_000.0 / time_ns as f64) as u64;
        
        Ok(frequency)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn read_tsc(&self) -> u64 {
        unsafe {
            std::arch::x86_64::_rdtsc()
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn read_tsc(&self) -> u64 {
        // Fallback to system time
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    /// Get current time with drift compensation
    pub fn now_ns(&self) -> u64 {
        let tsc = self.read_tsc();
        let cal = self.calibration.read();
        
        // Basic TSC to nanoseconds conversion
        let base_ns = ((tsc - cal.reference_tsc) as f64 * 1_000_000_000.0 
                      / cal.tsc_frequency as f64) as u64;
        
        let timestamp = cal.reference_wall_ns + base_ns;
        
        // Apply drift correction
        let drift_correction = self.drift_model.read().predict_drift(timestamp);
        
        // Apply temperature compensation
        let temp_correction = self.calculate_temperature_correction(&cal);
        
        // Apply all corrections
        let corrected_ns = timestamp as i64 + drift_correction + temp_correction;
        
        corrected_ns.max(0) as u64
    }
    
    fn calculate_temperature_correction(&self, cal: &CalibrationData) -> i64 {
        let current_temp = self.temperature_sensor.get_temperature_celsius();
        let ref_temp = self.reference_temp.load(Ordering::Relaxed) as f64 / 1000.0;
        
        let temp_delta = current_temp - ref_temp;
        let freq_delta = cal.temperature_coefficient * temp_delta;
        
        // Convert frequency change to time correction
        let correction_ppb = (freq_delta / cal.tsc_frequency as f64 * 1e9) as i64;
        
        // Apply correction based on time since calibration
        let elapsed = SystemTime::now()
            .duration_since(cal.last_calibration)
            .unwrap_or(Duration::ZERO);
        
        (elapsed.as_nanos() as i64 * correction_ppb / 1_000_000_000)
    }
    
    /// Start background calibration thread
    fn start_calibration_thread(&mut self) {
        self.running.store(true, Ordering::Relaxed);
        
        let calibration = Arc::clone(&self.calibration);
        let drift_model = Arc::clone(&self.drift_model);
        let temperature_sensor = Arc::clone(&self.temperature_sensor);
        let running = Arc::clone(&self.running);
        let ntp_enabled = self.ntp_sync_enabled.clone();
        
        self.calibration_thread = Some(thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                // Calibrate every minute
                thread::sleep(Duration::from_secs(60));
                
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                
                // Update drift model
                Self::update_drift_model_static(&calibration, &drift_model);
                
                // Update temperature coefficient
                Self::update_temperature_coefficient_static(&calibration, &temperature_sensor);
                
                // NTP sync if enabled
                if ntp_enabled.load(Ordering::Relaxed) {
                    Self::ntp_sync_static(&calibration);
                }
            }
        }));
    }
    
    fn update_drift_model_static(
        calibration: &Arc<RwLock<CalibrationData>>,
        drift_model: &Arc<RwLock<DriftModel>>,
    ) {
        let wall_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Calculate actual drift
        let cal = calibration.read();
        let expected_ns = cal.reference_wall_ns + 
            ((wall_ns - cal.reference_wall_ns) as f64 * 
             (1.0 + cal.drift_ppb as f64 / 1e9)) as u64;
        
        let drift_ns = wall_ns as i64 - expected_ns as i64;
        
        // Add to drift model
        drift_model.write().add_sample(wall_ns, drift_ns);
    }
    
    fn update_temperature_coefficient_static(
        calibration: &Arc<RwLock<CalibrationData>>,
        temperature_sensor: &Arc<TemperatureSensor>,
    ) {
        // Simple linear model: frequency change per degree
        let temp = temperature_sensor.get_temperature_celsius();
        
        // Estimate based on typical CPU behavior (100-200 Hz per degree)
        let mut cal = calibration.write();
        cal.temperature_coefficient = 150.0;  // Hz per degree C
    }
    
    fn ntp_sync_static(calibration: &Arc<RwLock<CalibrationData>>) {
        // Simplified NTP sync (would need actual NTP client in production)
        // For now, just resync with system time
        
        let wall_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let mut cal = calibration.write();
        
        #[cfg(target_arch = "x86_64")]
        {
            cal.reference_tsc = unsafe { std::arch::x86_64::_rdtsc() };
        }
        
        cal.reference_wall_ns = wall_ns;
        cal.last_calibration = SystemTime::now();
    }
    
    /// Enable NTP synchronization
    pub fn enable_ntp_sync(&self, enable: bool) {
        self.ntp_sync_enabled.store(enable, Ordering::Relaxed);
    }
    
    /// Get calibration data
    pub fn get_calibration(&self) -> CalibrationData {
        self.calibration.read().clone()
    }
    
    /// Get drift statistics
    pub fn get_drift_stats(&self) -> DriftStats {
        let model = self.drift_model.read();
        
        let mut min_drift = i64::MAX;
        let mut max_drift = i64::MIN;
        let mut avg_drift = 0i64;
        
        for &(_, drift) in &model.samples {
            min_drift = min_drift.min(drift);
            max_drift = max_drift.max(drift);
            avg_drift += drift;
        }
        
        if !model.samples.is_empty() {
            avg_drift /= model.samples.len() as i64;
        }
        
        DriftStats {
            samples: model.samples.len(),
            min_drift_ns: min_drift,
            max_drift_ns: max_drift,
            avg_drift_ns: avg_drift,
            linear_coefficient: model.linear_coefficient,
            quadratic_coefficient: model.quadratic_coefficient,
        }
    }
    
    /// Force recalibration
    pub fn recalibrate(&mut self) -> Result<()> {
        self.calibrate_frequency()
    }
}

impl Drop for CalibratedClock {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.calibration_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Drift statistics
#[derive(Debug)]
pub struct DriftStats {
    pub samples: usize,
    pub min_drift_ns: i64,
    pub max_drift_ns: i64,
    pub avg_drift_ns: i64,
    pub linear_coefficient: f64,
    pub quadratic_coefficient: f64,
}

/// Precision timer using calibrated clock
pub struct PrecisionTimer {
    clock: Arc<CalibratedClock>,
    start_ns: u64,
}

impl PrecisionTimer {
    pub fn new(clock: Arc<CalibratedClock>) -> Self {
        let start_ns = clock.now_ns();
        Self { clock, start_ns }
    }
    
    pub fn elapsed_ns(&self) -> u64 {
        self.clock.now_ns() - self.start_ns
    }
    
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_ns())
    }
    
    pub fn reset(&mut self) {
        self.start_ns = self.clock.now_ns();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calibrated_clock() {
        let clock = CalibratedClock::new().unwrap();
        
        let t1 = clock.now_ns();
        thread::sleep(Duration::from_millis(10));
        let t2 = clock.now_ns();
        
        let elapsed = t2 - t1;
        assert!(elapsed >= 9_000_000 && elapsed <= 15_000_000);
        
        // Get calibration info
        let cal = clock.get_calibration();
        println!("TSC Frequency: {} Hz", cal.tsc_frequency);
        println!("Drift: {} ppb", cal.drift_ppb);
    }
    
    #[test]
    fn test_drift_model() {
        let mut model = DriftModel::new();
        
        // Add synthetic drift samples (linear drift)
        for i in 0..100 {
            let timestamp = i * 1_000_000_000;  // 1 second intervals
            let drift = i * 100;  // 100ns drift per second
            model.add_sample(timestamp, drift);
        }
        
        // Predict drift
        let predicted = model.predict_drift(50_000_000_000);
        assert!(predicted > 0);
    }
    
    #[test]
    fn test_temperature_sensor() {
        let sensor = TemperatureSensor::new();
        let temp = sensor.get_temperature_celsius();
        
        // CPU temperature should be reasonable
        assert!(temp > 0.0 && temp < 150.0);
        println!("CPU Temperature: {:.1}°C", temp);
    }
    
    #[test]
    fn test_precision_timer() {
        let clock = Arc::new(CalibratedClock::new().unwrap());
        let mut timer = PrecisionTimer::new(Arc::clone(&clock));
        
        thread::sleep(Duration::from_millis(100));
        
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 90 && elapsed.as_millis() <= 110);
        
        timer.reset();
        let elapsed2 = timer.elapsed_ns();
        assert!(elapsed2 < 1_000_000);  // Less than 1ms after reset
    }
}