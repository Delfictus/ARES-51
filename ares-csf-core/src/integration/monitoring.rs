//! Monitoring and health checking for CSF components.

use crate::error::{Error, Result};
use crate::types::{ComponentId, Timestamp};
use serde::{Deserialize, Serialize};

/// Health level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthLevel {
    /// System is operating optimally
    Healthy,
    /// System is operating but with degraded performance
    Degraded,
    /// System is experiencing significant issues
    Unhealthy,
    /// System is non-functional
    Critical,
}

impl HealthLevel {
    /// Convert to numeric score (0-100)
    pub fn score(&self) -> u8 {
        match self {
            Self::Healthy => 100,
            Self::Degraded => 75,
            Self::Unhealthy => 25,
            Self::Critical => 0,
        }
    }

    /// Get health level from score
    pub fn from_score(score: u8) -> Self {
        match score {
            90..=100 => Self::Healthy,
            50..=89 => Self::Degraded,
            1..=49 => Self::Unhealthy,
            0 => Self::Critical,
            _ => Self::Critical, // fallback for any other values
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Monitoring interval in milliseconds
    pub interval_ms: u64,
    /// Health check timeout in milliseconds
    pub timeout_ms: u64,
    /// CPU usage threshold for warnings
    pub cpu_warning_threshold: f64,
    /// Memory usage threshold for warnings
    pub memory_warning_threshold: f64,
    /// Error rate threshold for warnings
    pub error_rate_threshold: f64,
    /// Enable detailed logging
    pub detailed_logging: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            interval_ms: 5000,
            timeout_ms: 1000,
            cpu_warning_threshold: 80.0,
            memory_warning_threshold: 80.0,
            error_rate_threshold: 0.1,
            detailed_logging: false,
        }
    }
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    /// Overall health level
    pub overall_health: HealthLevel,
    /// Component health statuses
    pub component_health: std::collections::HashMap<ComponentId, HealthLevel>,
    /// Health check timestamp
    pub timestamp: Timestamp,
    /// Detailed health report
    pub report: HealthReport,
}

/// Detailed health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// CPU usage statistics
    pub cpu_stats: CpuStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Network statistics
    pub network_stats: NetworkStats,
    /// Error statistics
    pub error_stats: ErrorStats,
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Current CPU usage percentage
    pub current_usage: f64,
    /// Average CPU usage over last minute
    pub avg_usage_1m: f64,
    /// Average CPU usage over last 5 minutes
    pub avg_usage_5m: f64,
    /// Peak CPU usage
    pub peak_usage: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Memory usage percentage
    pub usage_percentage: f64,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Bytes received per second
    pub bytes_received_per_sec: u64,
    /// Bytes sent per second
    pub bytes_sent_per_sec: u64,
    /// Active connections
    pub active_connections: u32,
    /// Failed connections
    pub failed_connections: u32,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total errors in last minute
    pub errors_1m: u32,
    /// Total errors in last hour
    pub errors_1h: u32,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Most common error types
    pub top_errors: Vec<ErrorCount>,
}

/// Error count by type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCount {
    /// Error type
    pub error_type: String,
    /// Number of occurrences
    pub count: u32,
}

/// Health monitor
pub struct HealthMonitor {
    config: MonitorConfig,
    start_time: Timestamp,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            start_time: Timestamp::now(),
        }
    }

    /// Check system health
    pub async fn check_health(&self) -> Result<SystemHealthStatus> {
        let timestamp = Timestamp::now();
        
        // Collect basic system stats
        let cpu_stats = self.collect_cpu_stats().await?;
        let memory_stats = self.collect_memory_stats().await?;
        let network_stats = self.collect_network_stats().await?;
        let error_stats = self.collect_error_stats().await?;
        
        let report = HealthReport {
            cpu_stats,
            memory_stats,
            network_stats,
            error_stats,
        };
        
        // Calculate overall health
        let overall_health = self.calculate_overall_health(&report);
        
        Ok(SystemHealthStatus {
            overall_health,
            component_health: std::collections::HashMap::new(),
            timestamp,
            report,
        })
    }

    async fn collect_cpu_stats(&self) -> Result<CpuStats> {
        // Placeholder implementation
        Ok(CpuStats {
            current_usage: 25.0,
            avg_usage_1m: 30.0,
            avg_usage_5m: 28.0,
            peak_usage: 45.0,
        })
    }

    async fn collect_memory_stats(&self) -> Result<MemoryStats> {
        let sys = sysinfo::System::new_all();
        let total_memory = sys.total_memory() * 1024; // Convert KB to bytes
        let used_memory = sys.used_memory() * 1024;
        let available_memory = total_memory - used_memory;
        let usage_percentage = (used_memory as f64 / total_memory as f64) * 100.0;
        
        Ok(MemoryStats {
            current_usage: used_memory,
            peak_usage: used_memory, // Placeholder
            available_memory,
            usage_percentage,
        })
    }

    async fn collect_network_stats(&self) -> Result<NetworkStats> {
        // Placeholder implementation
        Ok(NetworkStats {
            bytes_received_per_sec: 1024,
            bytes_sent_per_sec: 512,
            active_connections: 10,
            failed_connections: 0,
        })
    }

    async fn collect_error_stats(&self) -> Result<ErrorStats> {
        // Placeholder implementation
        Ok(ErrorStats {
            errors_1m: 0,
            errors_1h: 2,
            error_rate: 0.0,
            top_errors: vec![],
        })
    }

    fn calculate_overall_health(&self, report: &HealthReport) -> HealthLevel {
        let mut score = 100u8;
        
        // Reduce score based on CPU usage
        if report.cpu_stats.current_usage > self.config.cpu_warning_threshold {
            score = score.saturating_sub(20);
        }
        
        // Reduce score based on memory usage
        if report.memory_stats.usage_percentage > self.config.memory_warning_threshold {
            score = score.saturating_sub(20);
        }
        
        // Reduce score based on error rate
        if report.error_stats.error_rate > self.config.error_rate_threshold {
            score = score.saturating_sub(30);
        }
        
        HealthLevel::from_score(score)
    }

    /// Get uptime since monitor started
    pub fn uptime(&self) -> std::time::Duration {
        let now = Timestamp::now();
        now.duration_since(self.start_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_level_scoring() {
        assert_eq!(HealthLevel::Healthy.score(), 100);
        assert_eq!(HealthLevel::Degraded.score(), 75);
        assert_eq!(HealthLevel::Unhealthy.score(), 25);
        assert_eq!(HealthLevel::Critical.score(), 0);
        
        assert_eq!(HealthLevel::from_score(95), HealthLevel::Healthy);
        assert_eq!(HealthLevel::from_score(60), HealthLevel::Degraded);
        assert_eq!(HealthLevel::from_score(30), HealthLevel::Unhealthy);
        assert_eq!(HealthLevel::from_score(0), HealthLevel::Critical);
    }

    #[tokio::test]
    async fn test_health_monitor() {
        let config = MonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        
        let health = monitor.check_health().await.unwrap();
        
        // Should have valid health status
        assert!(matches!(health.overall_health, HealthLevel::Healthy | HealthLevel::Degraded));
        assert!(health.report.memory_stats.current_usage > 0);
    }
}