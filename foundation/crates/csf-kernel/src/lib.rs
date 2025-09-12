//! Chronos Kernel - Real-time scheduling and execution

#![warn(missing_docs)]

pub mod executor;
pub mod memory;
pub mod scheduler;
pub mod task;
pub mod time;

use csf_core::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// CPU cores dedicated to scheduling
    pub scheduler_cores: Vec<u32>,

    /// Maximum number of concurrent tasks
    pub max_tasks: usize,

    /// Scheduling quantum in microseconds
    pub quantum_us: u64,

    /// Memory pool size in bytes
    pub memory_pool_size: usize,

    /// Enable deadline monitoring
    pub enable_deadline_monitoring: bool,
}

/// The main Chronos kernel
#[allow(dead_code)]
pub struct ChronosKernel {
    /// Configuration
    config: KernelConfig,

    /// Schedulers (one per core)
    schedulers: Vec<scheduler::TemporalTaskWeaver>,

    /// For round-robin selection
    next_scheduler: AtomicUsize,

    /// Shutdown signal for all schedulers
    shutdown: Arc<AtomicBool>,
}

impl ChronosKernel {
    /// Create a new kernel instance
    pub fn new(config: KernelConfig) -> Result<Self, Error> {
        // Validate configuration
        if config.scheduler_cores.is_empty() {
            return Err(Error::InvalidConfig("No scheduler cores specified".into()));
        }

        // Create schedulers
        let mut schedulers = Vec::new();
        for &core_id in &config.scheduler_cores {
            let scheduler = scheduler::TemporalTaskWeaver::new(core_id, config.clone())?;
            schedulers.push(scheduler);
        }

        Ok(Self {
            config,
            schedulers,
            next_scheduler: AtomicUsize::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start the kernel
    /// This will spawn a thread for each scheduler and pin it to the configured core.
    pub fn start(mut self) -> Result<KernelHandle, Error> {
        let handles = Vec::new();

        for scheduler in &mut self.schedulers {
            scheduler.start()?;
        }

        // For now, we don't spawn additional threads since the schedulers start their own threads
        // In the future, we might want to spawn monitoring threads here

        Ok(KernelHandle {
            thread_handles: handles,
        })
    }

    /// Submit a task to the kernel
    pub fn submit_task(&self, task: task::Task) -> Result<TaskId, Error> {
        // Select scheduler with least load
        // Fallback to round-robin if load-based selection is not decisive
        let scheduler = self
            .select_scheduler_by_load()
            .or_else(|_| self.select_scheduler_round_robin())?;

        scheduler.submit(task)
    }

    /// Get kernel statistics
    pub fn stats(&self) -> KernelStats {
        KernelStats {
            active_tasks: self.count_active_tasks(),
            completed_tasks: self.count_completed_tasks(),
            deadline_misses: self.count_deadline_misses(),
        }
    }

    /// Selects the scheduler with the minimum number of active tasks.
    fn select_scheduler_by_load(&self) -> Result<&scheduler::TemporalTaskWeaver, Error> {
        self.schedulers
            .iter()
            .min_by_key(|s| s.active_task_count())
            .ok_or(Error::NoSchedulerAvailable)
    }

    /// Selects a scheduler using a simple round-robin strategy.
    /// This is a good fallback and useful for non-load-sensitive tasks.
    fn select_scheduler_round_robin(&self) -> Result<&scheduler::TemporalTaskWeaver, Error> {
        if self.schedulers.is_empty() {
            return Err(Error::NoSchedulerAvailable);
        }
        // Use fetch_add to atomically increment and get the previous value
        let index = self.next_scheduler.fetch_add(1, Ordering::Relaxed);
        // Modulo to wrap around
        // Relaxed ordering is fine because we don't need to synchronize memory with other operations,
        // we just need an atomic counter.
        Ok(&self.schedulers[index % self.schedulers.len()])
    }

    fn count_active_tasks(&self) -> usize {
        self.schedulers.iter().map(|s| s.active_task_count()).sum()
    }

    fn count_completed_tasks(&self) -> u64 {
        self.schedulers
            .iter()
            .map(|s| s.completed_task_count())
            .sum()
    }

    fn count_deadline_misses(&self) -> u64 {
        self.schedulers
            .iter()
            .map(|s| s.deadline_miss_count())
            .sum()
    }
}

/// A handle to the running kernel, which keeps the scheduler threads alive.
/// When this handle is dropped, the kernel will shut down (though graceful shutdown logic is not yet implemented).
pub struct KernelHandle {
    thread_handles: Vec<thread::JoinHandle<Result<(), Error>>>,
}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        println!("Shutting down ChronosKernel...");
        for handle in self.thread_handles.drain(..) {
            let _ = handle.join(); // Wait for scheduler threads to exit
        }
        println!("Kernel shutdown complete.");
    }
}

/// Kernel statistics
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Number of active tasks
    pub active_tasks: usize,

    /// Total completed tasks
    pub completed_tasks: u64,

    /// Total deadline misses
    pub deadline_misses: u64,
}

/// Kernel error types
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No scheduler available
    #[error("No scheduler available")]
    NoSchedulerAvailable,

    /// System error
    #[error("System error: {0}")]
    System(#[from] nix::Error),

    /// Other error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
