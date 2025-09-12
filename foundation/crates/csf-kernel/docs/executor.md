# Task Executor API Documentation

The task executor provides high-performance task execution with sub-microsecond latency targeting and seamless integration with the Temporal Task Weaver (TTW).

## Overview

The `TaskExecutor` is designed to achieve the NovaCore ChronoSynclastic Fabric's performance goals:
- **Sub-microsecond latency**: Target latency of 2-5μs per task execution
- **High throughput**: Supports >1M messages/sec processing
- **Memory safety**: Zero-copy operations where possible, comprehensive error handling
- **Hardware optimization**: CPU affinity and real-time scheduling support

## Core Components

### ExecutorConfig

Configuration structure for tuning executor performance:

```rust
pub struct ExecutorConfig {
    /// Number of worker threads for task execution
    pub worker_threads: usize,
    
    /// Maximum queue depth before backpressure
    pub max_queue_depth: usize,
    
    /// Target execution latency in nanoseconds (2-5μs target)
    pub target_latency_ns: u64,
    
    /// Enable detailed task timing
    pub enable_timing_metrics: bool,
    
    /// CPU cores dedicated to execution (separate from scheduler)
    pub executor_cores: Vec<u32>,
}
```

**Default Configuration:**
- Worker threads: Number of available CPU cores
- Max queue depth: 10,000 tasks
- Target latency: 5,000ns (5μs)
- Timing metrics: Enabled
- Executor cores: Auto-assigned

### TaskExecutor

Main execution engine implementing channel-based task distribution:

```rust
impl TaskExecutor {
    /// Create a new task executor with specified configuration
    pub fn new(config: ExecutorConfig) -> Result<Self, crate::Error>
    
    /// Submit a task for asynchronous execution
    pub fn submit_task(&self, task: Task) -> Result<TaskId, crate::Error>
    
    /// Submit task and wait for completion with timeout
    pub fn submit_and_wait(&self, task: Task, timeout: Duration) -> Result<TaskResult, crate::Error>
    
    /// Wait for specific task completion
    pub fn wait_for_completion(&self, task_id: TaskId, timeout: Duration) -> Result<TaskResult, crate::Error>
    
    /// Get current execution statistics
    pub fn stats(&self) -> ExecutorStats
    
    /// Check if latency targets are being met
    pub fn is_meeting_latency_target(&self) -> bool
    
    /// Graceful shutdown of all workers
    pub fn shutdown(self) -> Result<(), crate::Error>
}
```

### TaskResult

Comprehensive execution results with timing information:

```rust
pub struct TaskResult {
    /// Task identifier
    pub task_id: TaskId,
    
    /// Time when task was submitted
    pub submitted_at: NanoTime,
    
    /// Time when execution started
    pub started_at: NanoTime,
    
    /// Time when execution completed
    pub completed_at: NanoTime,
    
    /// Execution time in nanoseconds
    pub execution_latency_ns: u64,
    
    /// Total latency (including queuing) in nanoseconds
    pub total_latency_ns: u64,
    
    /// Whether execution succeeded
    pub success: bool,
    
    /// Error message if execution failed
    pub error: Option<String>,
}
```

### ExecutorStats

Real-time performance metrics:

```rust
pub struct ExecutorStats {
    /// Total tasks executed
    pub tasks_executed: AtomicU64,
    
    /// Total failed tasks
    pub tasks_failed: AtomicU64,
    
    /// Total execution time across all tasks
    pub total_execution_time_ns: AtomicU64,
    
    /// Minimum observed latency
    pub min_latency_ns: AtomicU64,
    
    /// Maximum observed latency
    pub max_latency_ns: AtomicU64,
    
    /// Running average latency
    pub avg_latency_ns: AtomicU64,
    
    /// Number of active worker threads
    pub active_workers: AtomicUsize,
}
```

## Architecture

### Channel-Based Design

The executor uses unbounded channels for task distribution to avoid blocking the TTW scheduler:

```
TTW Scheduler → submit_task() → UnboundedSender → Worker Threads → Task Execution
                                     ↓
                              Response Channel ← oneshot::Receiver ← TaskResult
```

### Worker Thread Model

Each worker thread:
1. **CPU Affinity**: Pinned to specific cores when configured
2. **Task Loop**: Continuously processes tasks from the channel
3. **Timing**: Precise timing using `global_time_source()`
4. **Statistics**: Lock-free atomic updates for metrics
5. **Error Handling**: Comprehensive error propagation

### Memory Safety

- **No unwrap/expect**: All operations use proper Result<T, E> patterns
- **Channel-based communication**: Avoids shared Task storage across threads
- **Atomic statistics**: Thread-safe metrics collection
- **Resource cleanup**: Proper shutdown and thread joining

## Integration with TTW

The Temporal Task Weaver integrates with the executor for optimal performance:

```rust
// TTW creates executor during initialization
let executor_config = ExecutorConfig {
    worker_threads: 4,
    max_queue_depth: 10_000,
    target_latency_ns: 2_000, // 2μs target
    enable_timing_metrics: true,
    executor_cores: vec![],
};

let executor = TaskExecutor::new(executor_config)?;

// TTW submits tasks directly to executor
pub fn submit(&self, task: Task) -> Result<TaskId, crate::Error> {
    let task_id = task.id();
    
    // Submit directly to executor
    self.executor.submit_task(task)?;
    
    // Track in causality graph
    self.causality_graph.add_task(task_id, vec![]);
    
    Ok(task_id)
}
```

## Performance Characteristics

### Latency Targets

- **Target**: 2-5μs execution latency
- **Measurement**: Hardware timestamp counters when available
- **Monitoring**: Real-time latency tracking and alerting
- **Optimization**: CPU affinity, real-time scheduling, zero-copy operations

### Throughput

- **Design target**: >1M messages/sec
- **Scaling**: Worker thread pool based on core count
- **Backpressure**: Queue depth limiting prevents memory exhaustion
- **Batching**: Future optimization for batch processing

### Error Handling

All operations return `Result<T, crate::Error>` with specific error types:
- **Queue full**: Backpressure protection
- **Task execution failure**: Captured and propagated
- **Timeout**: Configurable timeouts for completion waiting
- **Runtime errors**: Tokio runtime integration issues

## Usage Examples

### Basic Task Execution

```rust
use csf_kernel::executor::{TaskExecutor, ExecutorConfig};
use csf_kernel::task::Task;
use csf_core::Priority;

// Create executor
let config = ExecutorConfig::default();
let executor = TaskExecutor::new(config)?;

// Create and submit task
let task = Task::new("example_task", Priority::Normal, move || {
    // Task logic here
    println!("Task executing!");
    Ok(())
});

let task_id = executor.submit_task(task)?;

// Wait for completion with timeout
let result = executor.wait_for_completion(task_id, Duration::from_millis(100))?;

println!("Task completed in {}ns", result.execution_latency_ns);
```

### Performance Monitoring

```rust
// Check performance metrics
let stats = executor.stats();
let avg_latency = stats.avg_latency_ns.load(Ordering::Relaxed);
let tasks_executed = stats.tasks_executed.load(Ordering::Relaxed);

if !executor.is_meeting_latency_target() {
    eprintln!("Warning: Average latency {}ns exceeds target", avg_latency);
}

println!("Processed {} tasks, avg latency: {}ns", tasks_executed, avg_latency);
```

### Synchronized Execution

```rust
// Submit and wait in one call
let task = Task::new("sync_task", Priority::High, || {
    // Critical task logic
    Ok(())
});

let result = executor.submit_and_wait(task, Duration::from_millis(10))?;

if result.success {
    println!("Task completed successfully in {}ns", result.execution_latency_ns);
} else {
    eprintln!("Task failed: {:?}", result.error);
}
```

## Future Enhancements

1. **Hardware Acceleration**: TSC-based timing, SIMD optimizations
2. **Adaptive Scheduling**: Dynamic worker count based on load
3. **Priority Queues**: Support for task prioritization
4. **Batch Processing**: Grouped task execution for improved throughput
5. **NUMA Awareness**: Worker thread placement based on memory topology

## Thread Safety

All public APIs are thread-safe:
- `TaskExecutor` can be safely shared via `Arc<TaskExecutor>`
- Statistics use atomic operations for lock-free updates
- Channel operations are inherently thread-safe
- No shared mutable state between worker threads