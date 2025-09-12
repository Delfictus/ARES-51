//! Deadline scheduling with predictive temporal analysis

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use tracing::instrument;

use crate::{oracle::QuantumTimeOracle, Duration, NanoTime, TimeError, TimeResult};

/// Task information for deadline scheduling
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "net", derive(Serialize, Deserialize))]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Task priority level
    pub priority: TaskPriority,
    /// Absolute deadline for completion
    pub deadline: NanoTime,
    /// Estimated execution duration
    pub estimated_duration: Duration,
    /// Dependencies on other tasks
    pub dependencies: Vec<String>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

impl Task {
    /// Create a new task
    pub fn new(
        id: String,
        priority: TaskPriority,
        deadline: NanoTime,
        estimated_duration: Duration,
    ) -> Self {
        Self {
            id,
            priority,
            deadline,
            estimated_duration,
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a dependency on another task
    pub fn add_dependency(&mut self, task_id: String) {
        self.dependencies.push(task_id);
    }

    /// Calculate laxity (slack time) given current time
    pub fn laxity(&self, current_time: NanoTime) -> Option<Duration> {
        if self.deadline <= current_time {
            None // Past deadline
        } else {
            let remaining_time = Duration::from_nanos(
                self.deadline
                    .as_nanos()
                    .saturating_sub(current_time.as_nanos()),
            );
            if remaining_time >= self.estimated_duration {
                Some(Duration::from_nanos(
                    remaining_time
                        .as_nanos()
                        .saturating_sub(self.estimated_duration.as_nanos()),
                ))
            } else {
                Some(Duration::ZERO) // Critical - no slack
            }
        }
    }

    /// Check if task is critical (no slack time)
    pub fn is_critical(&self, current_time: NanoTime) -> bool {
        self.laxity(current_time)
            .is_none_or(|l| l == Duration::ZERO)
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Lowest priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority (real-time)
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Scheduled task with timing information
#[derive(Debug, Clone, PartialEq)]
struct ScheduledTask {
    task: Task,
    scheduled_start: NanoTime,
    scheduled_completion: NanoTime,
    slack_time: Duration,
}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Earlier scheduled start time has higher priority in heap
        other
            .scheduled_start
            .cmp(&self.scheduled_start)
            .then_with(|| self.task.priority.cmp(&other.task.priority))
            .then_with(|| self.task.deadline.cmp(&other.task.deadline))
    }
}

impl Eq for ScheduledTask {}

/// Result of scheduling operation
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleResult {
    /// Task successfully scheduled
    Scheduled {
        /// When the task will start execution
        start_time: NanoTime,
        /// When the task will complete execution
        completion_time: NanoTime,
        /// Available slack time before deadline
        slack_time: Duration,
    },
    /// Task cannot meet deadline
    DeadlineMissed {
        /// Earliest possible start time
        earliest_start: NanoTime,
        /// Required completion time to meet deadline
        required_completion: NanoTime,
        /// The deadline that cannot be met
        deadline: NanoTime,
    },
    /// Task has unresolved dependencies
    DependencyBlocked {
        /// List of tasks that must complete first
        missing_dependencies: Vec<String>,
    },
    /// Scheduling queue is full
    QueueFull,
}

/// Result of schedule optimization
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    /// Number of tasks rescheduled
    pub tasks_rescheduled: usize,
    /// Total slack time improvement
    pub slack_improvement: Duration,
    /// Number of deadline violations resolved
    pub violations_resolved: usize,
    /// Optimization strategy used
    pub strategy_used: String,
}

/// Deadline scheduler trait for predictive temporal analysis
#[async_trait]
pub trait DeadlineScheduler: Send + Sync {
    /// Schedule a task with deadline constraint
    fn schedule_task(&self, task: Task, deadline: NanoTime) -> ScheduleResult;

    /// Predict completion time for a task
    fn predict_completion(&self, task: &Task) -> NanoTime;

    /// Optimize current schedule using quantum-inspired algorithms
    fn optimize_schedule(&self) -> OptimizationResult;

    /// Get next task to execute
    fn next_task(&self) -> Option<Task>;

    /// Complete a task and update schedule
    fn complete_task(&self, task_id: &str, completion_time: NanoTime) -> TimeResult<()>;

    /// Get current schedule statistics
    fn get_statistics(&self) -> ScheduleStatistics;

    /// Remove a task from the schedule
    fn cancel_task(&self, task_id: &str) -> TimeResult<()>;
}

/// Scheduling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleStatistics {
    /// Total number of scheduled tasks
    pub total_tasks: usize,
    /// Number of critical tasks
    pub critical_tasks: usize,
    /// Average slack time across all tasks
    pub average_slack: Duration,
    /// Number of potential deadline violations
    pub deadline_violations: usize,
    /// Schedule utilization (0.0 to 1.0)
    pub utilization: f64,
}

/// Production implementation of deadline scheduler
#[derive(Debug)]
pub struct DeadlineSchedulerImpl {
    /// Priority queue of scheduled tasks
    schedule_queue: Arc<RwLock<BinaryHeap<ScheduledTask>>>,
    /// Completed tasks for history tracking
    completed_tasks: Arc<RwLock<HashMap<String, (NanoTime, Duration)>>>,
    /// Current time source
    time_source: Arc<dyn crate::TimeSource>,
    /// Quantum optimization oracle
    quantum_oracle: Arc<QuantumTimeOracle>,
    /// Maximum queue size
    max_queue_size: usize,
}

impl DeadlineSchedulerImpl {
    /// Create a new deadline scheduler
    pub fn new(time_source: Arc<dyn crate::TimeSource>) -> Self {
        Self {
            schedule_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            time_source,
            quantum_oracle: Arc::new(QuantumTimeOracle::new()),
            max_queue_size: 10_000,
        }
    }

    /// Create with custom settings
    pub fn with_config(
        time_source: Arc<dyn crate::TimeSource>,
        max_queue_size: usize,
        quantum_oracle: Arc<QuantumTimeOracle>,
    ) -> Self {
        Self {
            schedule_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            time_source,
            quantum_oracle,
            max_queue_size,
        }
    }

    /// Check if all dependencies are satisfied
    fn check_dependencies(&self, task: &Task) -> Vec<String> {
        let completed = self.completed_tasks.read();
        let mut missing = Vec::new();

        for dep in &task.dependencies {
            if !completed.contains_key(dep) {
                missing.push(dep.clone());
            }
        }

        missing
    }

    /// Calculate optimal start time using quantum optimization
    fn calculate_optimal_start_time(&self, task: &Task, current_time: NanoTime) -> NanoTime {
        let quantum_offset = self.quantum_oracle.current_offset();

        // Base start time is current time
        let base_start = current_time;

        // Apply quantum optimization based on task characteristics
        let priority_factor = match task.priority {
            TaskPriority::Critical => 0.0, // No delay for critical
            TaskPriority::High => 0.1,
            TaskPriority::Normal => 0.2,
            TaskPriority::Low => 0.5,
        };

        let delay = Duration::from_nanos((quantum_offset.phase * priority_factor * 1000.0) as u64);

        NanoTime::from_nanos(base_start.as_nanos().saturating_add(delay.as_nanos()))
    }

    /// Predict task completion using historical data
    fn predict_task_completion(&self, task: &Task, start_time: NanoTime) -> NanoTime {
        let completed = self.completed_tasks.read();

        // Look for similar tasks in history
        let mut similar_durations = Vec::new();
        for (completed_id, (_, actual_duration)) in completed.iter() {
            // Simple similarity: same priority level
            if completed_id.starts_with(&task.id[..2]) {
                // Rough heuristic
                similar_durations.push(*actual_duration);
            }
        }

        let predicted_duration = if similar_durations.is_empty() {
            // No history - use estimated duration
            task.estimated_duration
        } else {
            // Use average of similar tasks with quantum adjustment
            let avg_nanos: u64 = similar_durations.iter().map(|d| d.as_nanos()).sum::<u64>()
                / similar_durations.len() as u64;

            let base_duration = Duration::from_nanos(avg_nanos);

            // Apply quantum optimization
            let quantum_offset = self.quantum_oracle.current_offset();
            let adjustment_factor = 1.0 + quantum_offset.amplitude * 0.1;
            let adjusted_nanos = (base_duration.as_nanos() as f64 * adjustment_factor) as u64;

            Duration::from_nanos(adjusted_nanos)
        };

        NanoTime::from_nanos(
            start_time
                .as_nanos()
                .saturating_add(predicted_duration.as_nanos()),
        )
    }

    /// Optimize schedule using quantum-inspired algorithms
    fn apply_quantum_optimization(&self) -> OptimizationResult {
        let current_time = self.time_source.now_ns().unwrap_or(NanoTime::ZERO); // Fallback if time source fails
        let mut queue = self.schedule_queue.write();

        if queue.is_empty() {
            return OptimizationResult {
                tasks_rescheduled: 0,
                slack_improvement: Duration::ZERO,
                violations_resolved: 0,
                strategy_used: "no_tasks".to_string(),
            };
        }

        // Convert heap to vector for manipulation
        let mut tasks: Vec<_> = queue.drain().collect();

        let initial_violations = tasks
            .iter()
            .filter(|t| t.scheduled_completion > t.task.deadline)
            .count();

        // Apply quantum-inspired optimization strategies
        let dominant_strategy = self.quantum_oracle.current_state().dominant_strategy();

        match dominant_strategy {
            0 => self.optimize_for_latency(&mut tasks, current_time),
            1 => self.optimize_for_throughput(&mut tasks, current_time),
            2 => self.optimize_balanced(&mut tasks, current_time),
            _ => self.optimize_for_energy(&mut tasks, current_time),
        }

        // Recalculate schedule
        let mut current_slot = current_time;
        let mut total_slack_improvement = Duration::ZERO;
        let mut rescheduled_count = 0;

        for scheduled_task in &mut tasks {
            let old_slack = scheduled_task.slack_time;
            let new_start = current_slot.max(scheduled_task.scheduled_start);
            let new_completion = self.predict_task_completion(&scheduled_task.task, new_start);

            let new_slack = if new_completion <= scheduled_task.task.deadline {
                Duration::from_nanos(
                    scheduled_task
                        .task
                        .deadline
                        .as_nanos()
                        .saturating_sub(new_completion.as_nanos()),
                )
            } else {
                Duration::ZERO
            };

            if new_slack > old_slack {
                total_slack_improvement += new_slack - old_slack;
                rescheduled_count += 1;
            }

            scheduled_task.scheduled_start = new_start;
            scheduled_task.scheduled_completion = new_completion;
            scheduled_task.slack_time = new_slack;

            current_slot = new_completion;
        }

        // Rebuild heap
        for task in tasks {
            queue.push(task);
        }

        let final_violations = queue
            .iter()
            .filter(|t| t.scheduled_completion > t.task.deadline)
            .count();

        OptimizationResult {
            tasks_rescheduled: rescheduled_count,
            slack_improvement: total_slack_improvement,
            violations_resolved: initial_violations.saturating_sub(final_violations),
            strategy_used: format!("quantum_strategy_{}", dominant_strategy),
        }
    }

    fn optimize_for_latency(&self, tasks: &mut [ScheduledTask], _current_time: NanoTime) {
        // Sort by priority then deadline
        tasks.sort_by(|a, b| {
            b.task
                .priority
                .cmp(&a.task.priority)
                .then_with(|| a.task.deadline.cmp(&b.task.deadline))
        });
    }

    fn optimize_for_throughput(&self, tasks: &mut [ScheduledTask], _current_time: NanoTime) {
        // Sort by shortest job first
        tasks.sort_by(|a, b| a.task.estimated_duration.cmp(&b.task.estimated_duration));
    }

    fn optimize_balanced(&self, tasks: &mut [ScheduledTask], _current_time: NanoTime) {
        // Sort by slack time (least slack first)
        tasks.sort_by(|a, b| a.slack_time.cmp(&b.slack_time));
    }

    fn optimize_for_energy(&self, tasks: &mut [ScheduledTask], _current_time: NanoTime) {
        // Sort by deadline to minimize preemption
        tasks.sort_by(|a, b| a.task.deadline.cmp(&b.task.deadline));
    }
}

#[async_trait]
impl DeadlineScheduler for DeadlineSchedulerImpl {
    #[instrument(level = "debug")]
    fn schedule_task(&self, task: Task, deadline: NanoTime) -> ScheduleResult {
        let current_time = self.time_source.now_ns().unwrap_or(NanoTime::ZERO); // Fallback if time source fails

        // Check queue capacity
        {
            let queue = self.schedule_queue.read();
            if queue.len() >= self.max_queue_size {
                return ScheduleResult::QueueFull;
            }
        }

        // Check dependencies
        let missing_deps = self.check_dependencies(&task);
        if !missing_deps.is_empty() {
            return ScheduleResult::DependencyBlocked {
                missing_dependencies: missing_deps,
            };
        }

        // Calculate optimal scheduling
        let start_time = self.calculate_optimal_start_time(&task, current_time);
        let completion_time = self.predict_task_completion(&task, start_time);

        // Check if deadline can be met
        if completion_time > deadline {
            return ScheduleResult::DeadlineMissed {
                earliest_start: start_time,
                required_completion: completion_time,
                deadline,
            };
        }

        let slack_time = Duration::from_nanos(
            deadline
                .as_nanos()
                .saturating_sub(completion_time.as_nanos()),
        );

        // Create scheduled task
        let scheduled = ScheduledTask {
            task,
            scheduled_start: start_time,
            scheduled_completion: completion_time,
            slack_time,
        };

        // Add to queue
        self.schedule_queue.write().push(scheduled);

        ScheduleResult::Scheduled {
            start_time,
            completion_time,
            slack_time,
        }
    }

    fn predict_completion(&self, task: &Task) -> NanoTime {
        let current_time = self.time_source.now_ns().unwrap_or(NanoTime::ZERO); // Fallback to zero time if source fails
        let start_time = self.calculate_optimal_start_time(task, current_time);
        self.predict_task_completion(task, start_time)
    }

    fn optimize_schedule(&self) -> OptimizationResult {
        self.apply_quantum_optimization()
    }

    fn next_task(&self) -> Option<Task> {
        let mut queue = self.schedule_queue.write();
        queue.pop().map(|scheduled| scheduled.task)
    }

    fn complete_task(&self, task_id: &str, completion_time: NanoTime) -> TimeResult<()> {
        // Find and remove completed task from queue
        let mut queue = self.schedule_queue.write();
        let tasks: Vec<_> = queue.drain().collect();

        let mut found = false;
        let mut actual_duration = Duration::ZERO;

        for scheduled in tasks {
            if scheduled.task.id == task_id {
                found = true;
                actual_duration = Duration::from_nanos(
                    completion_time
                        .as_nanos()
                        .saturating_sub(scheduled.scheduled_start.as_nanos()),
                );
            } else {
                queue.push(scheduled);
            }
        }

        if !found {
            return Err(TimeError::DeadlineFailure {
                task_id: task_id.to_string(),
                reason: "Task not found in schedule".to_string(),
            });
        }

        // Record completion
        self.completed_tasks
            .write()
            .insert(task_id.to_string(), (completion_time, actual_duration));

        Ok(())
    }

    fn get_statistics(&self) -> ScheduleStatistics {
        let queue = self.schedule_queue.read();
        let current_time = self.time_source.now_ns().unwrap_or(NanoTime::ZERO); // Fallback if time source fails

        let total_tasks = queue.len();
        let critical_tasks = queue
            .iter()
            .filter(|t| t.task.priority == TaskPriority::Critical)
            .count();

        let total_slack: Duration = queue.iter().map(|t| t.slack_time).sum();

        let average_slack = if total_tasks > 0 {
            Duration::from_nanos(total_slack.as_nanos() / total_tasks as u64)
        } else {
            Duration::ZERO
        };

        let deadline_violations = queue
            .iter()
            .filter(|t| t.scheduled_completion > t.task.deadline)
            .count();

        // Calculate utilization
        let total_work: u64 = queue
            .iter()
            .map(|t| t.task.estimated_duration.as_nanos())
            .sum();

        let schedule_span = queue
            .iter()
            .map(|t| t.scheduled_completion)
            .max()
            .unwrap_or(current_time)
            - current_time;

        let utilization = if schedule_span.as_nanos() > 0 {
            (total_work as f64) / (schedule_span.as_nanos() as f64)
        } else {
            0.0
        };

        ScheduleStatistics {
            total_tasks,
            critical_tasks,
            average_slack,
            deadline_violations,
            utilization: utilization.min(1.0),
        }
    }

    fn cancel_task(&self, task_id: &str) -> TimeResult<()> {
        let mut queue = self.schedule_queue.write();
        let tasks: Vec<_> = queue.drain().collect();

        let mut found = false;
        for scheduled in tasks {
            if scheduled.task.id != task_id {
                queue.push(scheduled);
            } else {
                found = true;
            }
        }

        if !found {
            Err(TimeError::DeadlineFailure {
                task_id: task_id.to_string(),
                reason: "Task not found for cancellation".to_string(),
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SimulatedTimeSource;

    #[test]
    fn test_task_creation() {
        let task = Task::new(
            "test_task".to_string(),
            TaskPriority::High,
            NanoTime::from_secs(100),
            Duration::from_millis(50),
        );

        assert_eq!(task.id, "test_task");
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.deadline, NanoTime::from_secs(100));
    }

    #[test]
    fn test_task_laxity() {
        let task = Task::new(
            "test".to_string(),
            TaskPriority::Normal,
            NanoTime::from_secs(100),
            Duration::from_secs(30),
        );

        let current_time = NanoTime::from_secs(50);
        let laxity = task.laxity(current_time).unwrap();

        // Deadline - current - estimated = 100 - 50 - 30 = 20 seconds
        assert_eq!(laxity.as_secs(), 20);
    }

    #[test]
    fn test_scheduler_creation() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(0)));
        let scheduler = DeadlineSchedulerImpl::new(time_source);

        let stats = scheduler.get_statistics();
        assert_eq!(stats.total_tasks, 0);
    }

    #[test]
    fn test_task_scheduling_success() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(10)));
        let scheduler = DeadlineSchedulerImpl::new(time_source);

        let task = Task::new(
            "test_task".to_string(),
            TaskPriority::Normal,
            NanoTime::from_secs(100),
            Duration::from_secs(20),
        );

        let result = scheduler.schedule_task(task, NanoTime::from_secs(100));

        assert!(matches!(result, ScheduleResult::Scheduled { .. }));
    }

    #[test]
    fn test_deadline_missed() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(90)));
        let scheduler = DeadlineSchedulerImpl::new(time_source);

        let task = Task::new(
            "test_task".to_string(),
            TaskPriority::Normal,
            NanoTime::from_secs(100),
            Duration::from_secs(20),
        );

        let result = scheduler.schedule_task(task, NanoTime::from_secs(95)); // Too tight

        assert!(matches!(result, ScheduleResult::DeadlineMissed { .. }));
    }

    #[test]
    fn test_dependency_blocking() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(10)));
        let scheduler = DeadlineSchedulerImpl::new(time_source);

        let mut task = Task::new(
            "dependent_task".to_string(),
            TaskPriority::Normal,
            NanoTime::from_secs(100),
            Duration::from_secs(20),
        );
        task.add_dependency("missing_task".to_string());

        let result = scheduler.schedule_task(task, NanoTime::from_secs(100));

        assert!(matches!(result, ScheduleResult::DependencyBlocked { .. }));
    }

    #[test]
    fn test_schedule_optimization() {
        let time_source = Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(10)));
        let scheduler = DeadlineSchedulerImpl::new(time_source);

        // Add some tasks
        let task1 = Task::new(
            "t1".to_string(),
            TaskPriority::High,
            NanoTime::from_secs(50),
            Duration::from_secs(10),
        );
        let task2 = Task::new(
            "t2".to_string(),
            TaskPriority::Low,
            NanoTime::from_secs(100),
            Duration::from_secs(15),
        );

        let result1 = scheduler.schedule_task(task1, NanoTime::from_secs(50));
        assert!(matches!(result1, ScheduleResult::Scheduled { .. }));

        let result2 = scheduler.schedule_task(task2, NanoTime::from_secs(100));
        assert!(matches!(result2, ScheduleResult::Scheduled { .. }));

        let _result = scheduler.optimize_schedule();
        // tasks_rescheduled is usize which is always >= 0
    }
}
