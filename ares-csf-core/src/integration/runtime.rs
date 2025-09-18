//! Runtime management for CSF components.

use crate::error::{Error, Result};
use crate::types::{ComponentId, Timestamp, Priority};
use serde::{Deserialize, Serialize};

/// DRPP runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrppConfig {
    /// Maximum number of worker threads
    pub max_workers: usize,
    /// Task queue size
    pub queue_size: usize,
    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for DrppConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get(),
            queue_size: 1000,
            task_timeout_ms: 30000,
            enable_metrics: true,
        }
    }
}

/// DRPP runtime for task execution
pub struct DrppRuntime {
    config: DrppConfig,
    stats: RuntimeStats,
}

impl DrppRuntime {
    /// Create new DRPP runtime
    pub fn new(config: DrppConfig) -> Self {
        Self {
            config,
            stats: RuntimeStats::default(),
        }
    }

    /// Get runtime statistics
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }
}

/// Runtime event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeEvent {
    /// Component started
    ComponentStarted {
        component_id: ComponentId,
        timestamp: Timestamp,
    },
    /// Component stopped
    ComponentStopped {
        component_id: ComponentId,
        timestamp: Timestamp,
        reason: String,
    },
    /// Task completed
    TaskCompleted {
        task_id: String,
        duration_ms: u64,
        timestamp: Timestamp,
    },
    /// Error occurred
    ErrorOccurred {
        component_id: ComponentId,
        error_message: String,
        timestamp: Timestamp,
    },
}

/// Runtime statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStats {
    /// Total tasks executed
    pub total_tasks: u64,
    /// Tasks currently running
    pub running_tasks: u32,
    /// Failed tasks
    pub failed_tasks: u64,
    /// Average task duration in milliseconds
    pub avg_task_duration_ms: f64,
    /// Runtime uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
}

impl RuntimeStats {
    /// Calculate task success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_tasks > 0 {
            let successful_tasks = self.total_tasks - self.failed_tasks;
            (successful_tasks as f64 / self.total_tasks as f64) * 100.0
        } else {
            0.0
        }
    }
}