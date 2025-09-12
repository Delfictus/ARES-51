//! Defines the core Task structure for the Chronos Kernel.

use csf_core::prelude::*;
use std::fmt;

/// Represents a unit of work to be executed by the kernel's scheduler.
///
/// A Task encapsulates an executable closure, along with metadata that guides
/// the scheduling process, such as priority, deadlines, and resource requirements.
pub struct Task {
    /// Unique identifier for the task.
    id: TaskId,

    /// A descriptive name for debugging and telemetry.
    name: String,

    /// The executable logic of the task.
    /// It's boxed to allow for different closures to be stored in a collection.
    runnable: Box<dyn FnOnce() -> Result<(), anyhow::Error> + Send + 'static>,

    /// Static priority level.
    priority: Priority,

    /// The absolute time by which the task must complete, in nanoseconds.
    deadline_ns: Option<NanoTime>,

    /// The timestamp when the task was created, in nanoseconds.
    created_at_ns: NanoTime,

    /// A list of tasks that must be completed before this task can run.
    dependencies: Vec<TaskId>,

    /// An estimate of how long the task will take to run, in nanoseconds.
    /// Used for calculating laxity.
    estimated_duration_ns: u64,
}

impl Task {
    /// Creates a new task.
    pub fn new<F>(name: impl Into<String>, priority: Priority, runnable: F) -> Self
    where
        F: FnOnce() -> Result<(), anyhow::Error> + Send + 'static,
    {
        Self {
            id: TaskId::new(),
            name: name.into(),
            runnable: Box::new(runnable),
            priority,
            deadline_ns: None,
            created_at_ns: crate::time::hardware_clock::now(),
            dependencies: Vec::new(),
            estimated_duration_ns: 0, // Default, should be set if known
        }
    }

    /// Sets a deadline for the task.
    pub fn with_deadline(mut self, deadline_ns: NanoTime) -> Self {
        self.deadline_ns = Some(deadline_ns);
        self
    }

    /// Sets an estimated duration for the task.
    pub fn with_estimated_duration(mut self, duration_ns: u64) -> Self {
        self.estimated_duration_ns = duration_ns;
        self
    }

    /// Sets the dependencies for this task.
    pub fn with_dependencies(mut self, dependencies: Vec<TaskId>) -> Self {
        self.dependencies = dependencies;
        self
    }

    // --- Accessors ---

    /// Returns the task ID.
    pub fn id(&self) -> TaskId {
        self.id
    }
    /// Returns the task name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the task priority.
    pub fn priority(&self) -> Priority {
        self.priority
    }
    /// Returns the task deadline in nanoseconds.
    pub fn deadline_ns(&self) -> Option<NanoTime> {
        self.deadline_ns
    }
    /// Returns when the task was created in nanoseconds.
    pub fn created_at_ns(&self) -> NanoTime {
        self.created_at_ns
    }
    /// Returns the estimated duration in nanoseconds.
    pub fn estimated_duration_ns(&self) -> u64 {
        self.estimated_duration_ns
    }
    /// Returns the task dependencies.
    pub fn dependencies(&self) -> &[TaskId] {
        &self.dependencies
    }

    /// Executes the task.
    pub fn run(self) -> Result<(), anyhow::Error> {
        (self.runnable)()
    }
}

impl fmt::Debug for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("priority", &self.priority)
            .field("deadline_ns", &self.deadline_ns)
            .finish()
    }
}
