//! Enterprise-grade validation and hardening for the resonance system
//! 
//! Implements comprehensive security, stability, and reliability measures

use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use std::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Hardening error types
#[derive(Debug, Error)]
pub enum HardeningError {
    #[error("Input validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),
    
    #[error("Timeout exceeded: {0}")]
    TimeoutExceeded(String),
    
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
}

/// Input validator for resonance processing
pub struct InputValidator {
    /// Maximum tensor size
    max_tensor_size: usize,
    
    /// Valid frequency range
    frequency_range: (f64, f64),
    
    /// Maximum amplitude
    max_amplitude: f64,
    
    /// Sanitization rules
    sanitization_rules: Vec<SanitizationRule>,
}

/// Sanitization rule for input data
#[derive(Clone)]
pub struct SanitizationRule {
    pub name: String,
    pub validator: Arc<dyn Fn(&[f64]) -> bool + Send + Sync>,
}

impl std::fmt::Debug for SanitizationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SanitizationRule")
            .field("name", &self.name)
            .finish()
    }
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    /// Current state
    state: Arc<RwLock<CircuitState>>,
    
    /// Configuration
    config: CircuitBreakerConfig,
    
    /// Failure counter
    failure_count: Arc<RwLock<usize>>,
    
    /// Last failure time
    last_failure: Arc<RwLock<Option<Instant>>>,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening
    pub failure_threshold: usize,
    
    /// Success threshold for closing
    pub success_threshold: usize,
    
    /// Timeout before attempting half-open
    pub timeout: Duration,
    
    /// Reset timeout
    pub reset_timeout: Duration,
}

/// Rate limiter for API protection
pub struct RateLimiter {
    /// Semaphore for rate limiting
    semaphore: Arc<Semaphore>,
    
    /// Token bucket
    token_bucket: Arc<RwLock<TokenBucket>>,
    
    /// Configuration
    config: RateLimiterConfig,
}

/// Token bucket for rate limiting
struct TokenBucket {
    /// Current tokens
    tokens: f64,
    
    /// Maximum tokens
    max_tokens: f64,
    
    /// Refill rate (tokens per second)
    refill_rate: f64,
    
    /// Last refill time
    last_refill: Instant,
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Maximum requests per second
    pub max_rps: f64,
    
    /// Burst capacity
    pub burst_capacity: usize,
    
    /// Refill interval
    pub refill_interval: Duration,
}

/// Resource monitor for preventing exhaustion
pub struct ResourceMonitor {
    /// Memory threshold (bytes)
    memory_threshold: usize,
    
    /// CPU threshold (percentage)
    cpu_threshold: f64,
    
    /// Current metrics
    metrics: Arc<RwLock<ResourceMetrics>>,
}

/// Resource metrics
#[derive(Debug, Clone)]
struct ResourceMetrics {
    /// Memory usage (bytes)
    memory_used: usize,
    
    /// CPU usage (0-100)
    cpu_percent: f64,
    
    /// Active operations
    active_operations: usize,
    
    /// Queue depth
    queue_depth: usize,
}

/// Chaos engineering for resilience testing
pub struct ChaosEngine {
    /// Chaos mode enabled
    enabled: Arc<RwLock<bool>>,
    
    /// Chaos configuration
    config: Arc<RwLock<ChaosConfig>>,
    
    /// Injection points
    injection_points: Arc<RwLock<Vec<InjectionPoint>>>,
}

/// Chaos configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosConfig {
    /// Failure probability (0-1)
    pub failure_probability: f64,
    
    /// Latency injection (ms)
    pub latency_ms: Option<u64>,
    
    /// Memory pressure (MB)
    pub memory_pressure_mb: Option<usize>,
    
    /// CPU stress (threads)
    pub cpu_stress_threads: Option<usize>,
    
    /// Network partition
    pub network_partition: bool,
}

/// Injection point for chaos
#[derive(Debug, Clone)]
struct InjectionPoint {
    name: String,
    location: String,
    fault_type: FaultType,
}

/// Types of faults to inject
#[derive(Debug, Clone)]
enum FaultType {
    Latency(Duration),
    Error(String),
    Panic(String),
    ResourceExhaustion,
    NetworkPartition,
}

/// Audit logger for security and compliance
pub struct AuditLogger {
    /// Log destination
    destination: Arc<RwLock<AuditDestination>>,
    
    /// Encryption key for sensitive data
    encryption_key: Option<Vec<u8>>,
    
    /// Buffer for batch writing
    buffer: Arc<RwLock<Vec<AuditEntry>>>,
    
    /// Flush interval
    flush_interval: Duration,
}

/// Audit log destination
enum AuditDestination {
    File(String),
    Database(String),
    Syslog(String),
    CloudWatch(String),
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Operation type
    pub operation: String,
    
    /// User/system identifier
    pub principal: String,
    
    /// Resource accessed
    pub resource: String,
    
    /// Result (success/failure)
    pub result: AuditResult,
    
    /// Additional metadata
    pub metadata: serde_json::Value,
}

/// Audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure(String),
    Denied(String),
}

impl InputValidator {
    pub fn new() -> Self {
        Self {
            max_tensor_size: 1_000_000,
            frequency_range: (0.1, 1000.0),
            max_amplitude: 100.0,
            sanitization_rules: Self::default_rules(),
        }
    }
    
    /// Validate input tensor
    pub fn validate_tensor(&self, data: &[f64]) -> Result<(), HardeningError> {
        // Check size
        if data.len() > self.max_tensor_size {
            return Err(HardeningError::ValidationFailed(
                format!("Tensor size {} exceeds maximum {}", data.len(), self.max_tensor_size)
            ));
        }
        
        // Check for NaN or Inf
        if data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(HardeningError::ValidationFailed(
                "Input contains NaN or Inf values".to_string()
            ));
        }
        
        // Check amplitude bounds
        if let Some(max) = data.iter().fold(None, |max, &x| {
            match max {
                None => Some(x.abs()),
                Some(m) => Some(m.max(x.abs())),
            }
        }) {
            if max > self.max_amplitude {
                return Err(HardeningError::ValidationFailed(
                    format!("Amplitude {} exceeds maximum {}", max, self.max_amplitude)
                ));
            }
        }
        
        // Apply sanitization rules
        for rule in &self.sanitization_rules {
            if !(rule.validator)(data) {
                return Err(HardeningError::ValidationFailed(
                    format!("Sanitization rule '{}' failed", rule.name)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Default sanitization rules
    fn default_rules() -> Vec<SanitizationRule> {
        vec![
            SanitizationRule {
                name: "no_extreme_outliers".to_string(),
                validator: Arc::new(|data| {
                    if data.is_empty() { return true; }
                    let mean = data.iter().sum::<f64>() / data.len() as f64;
                    let std_dev = (data.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / data.len() as f64)
                        .sqrt();
                    
                    // No values beyond 10 standard deviations
                    !data.iter().any(|x| (x - mean).abs() > 10.0 * std_dev)
                }),
            },
            SanitizationRule {
                name: "reasonable_frequency_content".to_string(),
                validator: Arc::new(|data| {
                    // Check that data has reasonable frequency content
                    // (not all same value, not pure noise)
                    if data.len() < 2 { return true; }
                    
                    let variance = data.windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .sum::<f64>() / (data.len() - 1) as f64;
                    
                    variance > 0.001 && variance < 100.0
                }),
            },
        ]
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Execute operation with circuit breaker protection
    pub async fn execute<F, T>(&self, operation: F) -> Result<T, HardeningError>
    where
        F: FnOnce() -> Result<T, HardeningError>,
    {
        let state = self.state.read().await.clone();
        
        match state {
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(last) = *self.last_failure.read().await {
                    if last.elapsed() > self.config.timeout {
                        *self.state.write().await = CircuitState::HalfOpen;
                        *self.failure_count.write().await = 0;
                    } else {
                        return Err(HardeningError::CircuitBreakerOpen(
                            "Circuit breaker is open".to_string()
                        ));
                    }
                }
            },
            CircuitState::HalfOpen => {
                // Allow limited traffic through
            },
            CircuitState::Closed => {
                // Normal operation
            }
        }
        
        // Execute the operation
        match operation() {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            },
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }
    
    /// Handle successful operation
    async fn on_success(&self) {
        let mut state = self.state.write().await;
        let mut count = self.failure_count.write().await;
        
        match *state {
            CircuitState::HalfOpen => {
                *count = 0;
                *state = CircuitState::Closed;
            },
            _ => {
                *count = 0;
            }
        }
    }
    
    /// Handle failed operation
    async fn on_failure(&self) {
        let mut state = self.state.write().await;
        let mut count = self.failure_count.write().await;
        let mut last = self.last_failure.write().await;
        
        *count += 1;
        *last = Some(Instant::now());
        
        if *count >= self.config.failure_threshold {
            *state = CircuitState::Open;
        }
    }
}

impl RateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.burst_capacity));
        let token_bucket = Arc::new(RwLock::new(TokenBucket {
            tokens: config.max_rps,
            max_tokens: config.max_rps * 2.0,
            refill_rate: config.max_rps,
            last_refill: Instant::now(),
        }));
        
        Self {
            semaphore,
            token_bucket,
            config,
        }
    }
    
    /// Acquire permission to proceed
    pub async fn acquire(&self) -> Result<(), HardeningError> {
        // Refill tokens
        self.refill_tokens().await;
        
        // Try to acquire token
        let mut bucket = self.token_bucket.write().await;
        
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            Err(HardeningError::RateLimitExceeded(
                "Rate limit exceeded".to_string()
            ))
        }
    }
    
    /// Refill token bucket
    async fn refill_tokens(&self) {
        let mut bucket = self.token_bucket.write().await;
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        
        let tokens_to_add = elapsed * bucket.refill_rate;
        bucket.tokens = (bucket.tokens + tokens_to_add).min(bucket.max_tokens);
        bucket.last_refill = now;
    }
}

impl ResourceMonitor {
    pub fn new(memory_threshold: usize, cpu_threshold: f64) -> Self {
        Self {
            memory_threshold,
            cpu_threshold,
            metrics: Arc::new(RwLock::new(ResourceMetrics {
                memory_used: 0,
                cpu_percent: 0.0,
                active_operations: 0,
                queue_depth: 0,
            })),
        }
    }
    
    /// Check if resources are available
    pub async fn check_resources(&self) -> Result<(), HardeningError> {
        let metrics = self.metrics.read().await;
        
        if metrics.memory_used > self.memory_threshold {
            return Err(HardeningError::ResourceExhaustion(
                format!("Memory usage {} exceeds threshold {}", 
                        metrics.memory_used, self.memory_threshold)
            ));
        }
        
        if metrics.cpu_percent > self.cpu_threshold {
            return Err(HardeningError::ResourceExhaustion(
                format!("CPU usage {:.1}% exceeds threshold {:.1}%", 
                        metrics.cpu_percent, self.cpu_threshold)
            ));
        }
        
        Ok(())
    }
    
    /// Update resource metrics
    pub async fn update_metrics(&self, memory: usize, cpu: f64, operations: usize, queue: usize) {
        let mut metrics = self.metrics.write().await;
        metrics.memory_used = memory;
        metrics.cpu_percent = cpu;
        metrics.active_operations = operations;
        metrics.queue_depth = queue;
    }
}

impl ChaosEngine {
    pub fn new(config: ChaosConfig) -> Self {
        Self {
            enabled: Arc::new(RwLock::new(false)),
            config: Arc::new(RwLock::new(config)),
            injection_points: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Enable chaos engineering
    pub async fn enable(&self) {
        *self.enabled.write().await = true;
    }
    
    /// Disable chaos engineering
    pub async fn disable(&self) {
        *self.enabled.write().await = false;
    }

    /// Set the failure probability for chaos engineering
    pub async fn set_failure_probability(&self, probability: f64) {
        self.config.write().await.failure_probability = probability;
    }
    
    /// Maybe inject fault
    pub async fn maybe_inject_fault(&self, point: &str) -> Result<(), HardeningError> {
        if !*self.enabled.read().await {
            return Ok(());
        }
        
        let config = self.config.read().await;

        // Random chance of failure
        if rand::random::<f64>() < config.failure_probability {
            // Inject latency if configured
            if let Some(latency_ms) = config.latency_ms {
                tokio::time::sleep(Duration::from_millis(latency_ms)).await;
            }
            
            // Random fault type
            if rand::random::<f64>() < 0.5 {
                return Err(HardeningError::ValidationFailed(
                    format!("Chaos injection at {}", point)
                ));
            }
        }
        
        Ok(())
    }
}

impl AuditLogger {
    pub async fn new(destination: &str) -> Self {
        let dest = if destination.starts_with("file://") {
            AuditDestination::File(destination[7..].to_string())
        } else if destination.starts_with("db://") {
            AuditDestination::Database(destination[5..].to_string())
        } else {
            AuditDestination::Syslog(destination.to_string())
        };
        
        Self {
            destination: Arc::new(RwLock::new(dest)),
            encryption_key: None,
            buffer: Arc::new(RwLock::new(Vec::new())),
            flush_interval: Duration::from_secs(5),
        }
    }
    
    /// Log audit entry
    pub async fn log(&self, entry: AuditEntry) {
        let mut buffer = self.buffer.write().await;
        buffer.push(entry);
        
        // Flush if buffer is large
        if buffer.len() > 100 {
            self.flush().await;
        }
    }
    
    /// Flush buffer to destination
    pub async fn flush(&self) {
        let mut buffer = self.buffer.write().await;
        if buffer.is_empty() {
            return;
        }
        
        let entries = buffer.drain(..).collect::<Vec<_>>();
        
        // Write to destination
        match &*self.destination.read().await {
            AuditDestination::File(path) => {
                // Write to file
                if let Ok(json) = serde_json::to_string_pretty(&entries) {
                    let _ = tokio::fs::write(path, json).await;
                }
            },
            _ => {
                // Other destinations would be implemented here
            }
        }
    }
}

// Re-export for convenience
use rand;