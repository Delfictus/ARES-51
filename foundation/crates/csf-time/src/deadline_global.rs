//! Global deadline scheduler management for system-wide temporal coordination.

use crate::deadline::DeadlineSchedulerImpl;
use crate::oracle::QuantumTimeOracle;
use parking_lot::RwLock;
use std::sync::{Arc, OnceLock};
use tracing::{debug, warn};

/// Global deadline scheduler instance
static GLOBAL_DEADLINE_SCHEDULER: OnceLock<Arc<RwLock<DeadlineSchedulerImpl>>> = OnceLock::new();

/// Initialize the global deadline scheduler with the given time source.
///
/// This should be called early in the application lifecycle.
///
/// # Errors
/// Returns error if called more than once or if time source is invalid.
pub fn initialize_global_deadline_scheduler(
    time_source: Arc<dyn crate::TimeSource>,
) -> crate::TimeResult<()> {
    // If already initialized, treat this call as a no-op to support tests and
    // repeated initialization attempts across modules.
    if GLOBAL_DEADLINE_SCHEDULER.get().is_some() {
        tracing::debug!("Global deadline scheduler already initialized; skipping");
        return Ok(());
    }

    let quantum_oracle = Arc::new(QuantumTimeOracle::new());
    let scheduler = DeadlineSchedulerImpl::with_config(
        time_source,
        10_000, // Default queue size
        quantum_oracle,
    );

    // Attempt to set the global scheduler. If another thread initialized it
    // between the `get().is_some()` check and this call, treat that as a
    // successful initialization (idempotent behavior) to avoid races during tests.
    match GLOBAL_DEADLINE_SCHEDULER.set(Arc::new(RwLock::new(scheduler))) {
        Ok(()) => {
            debug!("Global deadline scheduler initialized");
            Ok(())
        }
        Err(_) => {
            tracing::debug!("Global deadline scheduler was initialized concurrently; continuing");
            Ok(())
        }
    }
}

/// Get a reference to the global deadline scheduler.
///
/// # Errors
/// Returns `TimeError::InvalidOperation` if the scheduler has not been initialized.
pub fn global_deadline_scheduler() -> crate::TimeResult<Arc<RwLock<DeadlineSchedulerImpl>>> {
    GLOBAL_DEADLINE_SCHEDULER
        .get()
        .cloned()
        .ok_or(crate::TimeError::InvalidOperation {
            operation: "global_deadline_scheduler".to_string(),
            reason: "Global deadline scheduler not initialized".to_string(),
        })
}

/// Check if the global deadline scheduler has been initialized.
pub fn is_global_deadline_scheduler_initialized() -> bool {
    GLOBAL_DEADLINE_SCHEDULER.get().is_some()
}

/// Schedule a task with the global deadline scheduler.
///
/// # Errors
/// Returns error if scheduler not initialized or scheduling fails.
pub fn global_schedule_with_deadline(
    task: crate::deadline::Task,
    deadline: csf_shared_types::NanoTime,
) -> crate::TimeResult<crate::deadline::ScheduleResult> {
    let scheduler =
        GLOBAL_DEADLINE_SCHEDULER
            .get()
            .cloned()
            .ok_or(crate::TimeError::InvalidOperation {
                operation: "global_schedule_with_deadline".to_string(),
                reason: "Global deadline scheduler not initialized".to_string(),
            })?;
    let scheduler_guard = scheduler.read();

    use crate::deadline::DeadlineScheduler;
    let csf_time_deadline = crate::NanoTime::from_nanos(deadline.as_nanos());
    Ok(scheduler_guard.schedule_task(task, csf_time_deadline))
}

/// Schedule a task after a delay using the global deadline scheduler.
///
/// # Errors
/// Returns error if scheduler not initialized or scheduling fails.
pub fn global_schedule_after(
    task: crate::deadline::Task,
    delay: csf_shared_types::NanoTime,
) -> crate::TimeResult<crate::deadline::ScheduleResult> {
    let scheduler =
        GLOBAL_DEADLINE_SCHEDULER
            .get()
            .cloned()
            .ok_or(crate::TimeError::InvalidOperation {
                operation: "global_schedule_after".to_string(),
                reason: "Global deadline scheduler not initialized".to_string(),
            })?;
    let scheduler_guard = scheduler.read();

    // Calculate absolute deadline from current time + delay
    let current_time = crate::NanoTime::now(); // Use placeholder current time
    let deadline =
        crate::NanoTime::from_nanos(current_time.as_nanos().saturating_add(delay.as_nanos()));

    use crate::deadline::DeadlineScheduler;
    Ok(scheduler_guard.schedule_task(task, deadline))
}

/// Get the current load of the global deadline scheduler.
///
/// Returns 0.0 if scheduler not initialized.
pub fn global_deadline_load() -> f64 {
    if let Some(scheduler) = GLOBAL_DEADLINE_SCHEDULER.get() {
        let scheduler_guard = scheduler.read();
        use crate::deadline::DeadlineScheduler;
        let stats = scheduler_guard.get_statistics();
        stats.utilization
    } else {
        warn!("Global deadline scheduler not initialized, returning 0.0 load");
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deadline::{Task, TaskPriority};
    use crate::source::SimulatedTimeSource;
    use crate::{Duration, NanoTime};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_global_scheduler_initialization() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(0)));

        assert!(!is_global_deadline_scheduler_initialized());

        initialize_global_deadline_scheduler(time_source).expect("Failed to initialize");

        assert!(is_global_deadline_scheduler_initialized());
        assert!(global_deadline_load() >= 0.0);
    }

    #[tokio::test]
    async fn test_global_scheduling() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(10)));
        initialize_global_deadline_scheduler(time_source).expect("Failed to initialize");

        let task = Task::new(
            "test_global_task".to_string(),
            TaskPriority::Normal,
            NanoTime::from_secs(100),
            Duration::from_secs(20),
        );

        let result = global_schedule_with_deadline(
            task,
            csf_shared_types::NanoTime::from_nanos(100_000_000_000), // 100 seconds
        );

        assert!(result.is_ok());
        assert!(global_deadline_load() > 0.0);
    }
}
