//! Temporal Task Weaver - Causality-aware real-time scheduler

use csf_core::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

pub mod causality;
pub mod priority;

use crate::executor::{ExecutorConfig, TaskExecutor};
use crate::task::Task;
use causality::CausalityGraph;

/// The Temporal Task Weaver scheduler
pub struct TemporalTaskWeaver {
    /// Core ID this scheduler runs on
    core_id: u32,

    /// Task executor
    executor: Arc<TaskExecutor>,

    /// Causality tracking
    causality_graph: Arc<CausalityGraph>,

    /// Statistics
    stats: Arc<SchedulerStats>,

    /// Scheduler thread handle (minimal monitoring thread)
    thread: Option<thread::JoinHandle<()>>,
}

/// Scheduler statistics
#[derive(Default)]
pub struct SchedulerStats {
    active_tasks: AtomicUsize,
    completed_tasks: AtomicU64,
    deadline_misses: AtomicU64,
}

impl TemporalTaskWeaver {
    /// Create a new scheduler for a specific core
    pub fn new(core_id: u32, _config: crate::KernelConfig) -> Result<Self, crate::Error> {
        // Create executor configuration optimized for sub-microsecond latency
        let executor_config = ExecutorConfig {
            worker_threads: 4, // Dedicated execution threads
            max_queue_depth: 10_000,
            target_latency_ns: 2_000, // 2μs target for sub-microsecond performance
            enable_timing_metrics: true,
            executor_cores: vec![], // Let executor auto-assign cores
        };

        let executor = TaskExecutor::new(executor_config).map_err(|e| {
            crate::Error::Other(anyhow::anyhow!("Failed to create executor: {}", e))
        })?;

        Ok(Self {
            core_id,
            executor: Arc::new(executor),
            causality_graph: Arc::new(CausalityGraph::new()),
            stats: Arc::new(SchedulerStats::default()),
            thread: None,
        })
    }

    /// Start the scheduler (minimal monitoring since executor handles execution)
    pub fn start(&mut self) -> Result<(), crate::Error> {
        let stats = self.stats.clone();

        let thread = thread::spawn(move || {
            // Minimal monitoring loop - executor handles actual task execution
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1000));
                // Could add monitoring metrics here if needed
                let active = stats.active_tasks.load(Ordering::Relaxed);
                tracing::trace!("TTW monitoring: {} active tasks", active);
            }
        });

        self.thread = Some(thread);
        tracing::info!("Temporal Task Weaver started with executor-based task execution");
        Ok(())
    }

    /// Submit a task
    pub fn submit(&self, task: Task) -> Result<TaskId, crate::Error> {
        let task_id = task.id();

        // Submit directly to executor - no need to store in scheduler
        self.executor.submit_task(task).map_err(|e| {
            crate::Error::Other(anyhow::anyhow!("Failed to submit task to executor: {}", e))
        })?;

        // Add to causality graph for dependency tracking
        self.causality_graph.add_task(task_id, vec![]); // Dependencies handled by executor

        self.stats.active_tasks.fetch_add(1, Ordering::Relaxed);

        Ok(task_id)
    }

    /// Get active task count
    pub fn active_task_count(&self) -> usize {
        self.stats.active_tasks.load(Ordering::Relaxed)
    }

    /// Get completed task count
    pub fn completed_task_count(&self) -> u64 {
        self.stats.completed_tasks.load(Ordering::Relaxed)
    }

    /// Get deadline miss count
    pub fn deadline_miss_count(&self) -> u64 {
        self.stats.deadline_misses.load(Ordering::Relaxed)
    }

    /// Get the core ID this scheduler runs on
    pub fn core_id(&self) -> u32 {
        self.core_id
    }
}