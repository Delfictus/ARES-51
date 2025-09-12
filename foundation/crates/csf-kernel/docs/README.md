# CSF-Kernel Documentation

The CSF-Kernel crate implements the core temporal task execution engine for the ARES ChronoSynclastic Fabric system.

## Architecture Overview

The kernel consists of several key components working together to achieve sub-microsecond task execution:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CSF-Kernel Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │  Temporal Task      │    │   Task Executor     │                │
│  │  Weaver (TTW)       │────│   (Sub-μs latency)  │                │
│  │                     │    │                     │                │
│  │  • Causality-aware  │    │  • Channel-based    │                │
│  │  • Priority scheduling │  │  • Worker threads   │                │
│  │  • Deadline tracking│    │  • CPU affinity     │                │
│  └─────────────────────┘    └─────────────────────┘                │
│           │                            │                            │
│           ▼                            ▼                            │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │   Task Objects      │    │   Performance       │                │
│  │                     │    │   Monitoring        │                │
│  │  • FnOnce closures  │    │                     │                │
│  │  • Metadata         │    │  • Real-time stats  │                │
│  │  • Dependencies     │    │  • Latency tracking │                │
│  └─────────────────────┘    └─────────────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### [Task Executor](executor.md)
High-performance task execution engine with sub-microsecond latency targeting.

Key features:
- 2-5μs execution latency target
- Channel-based task distribution
- CPU affinity and real-time scheduling
- Comprehensive performance monitoring
- TTW integration

### Temporal Task Weaver (TTW)
Causality-aware scheduler that coordinates task execution:
- Dependency graph management
- Priority-based scheduling
- Deadline monitoring
- Integration with executor for optimal performance

### Task Objects
Encapsulated work units with rich metadata:
- `FnOnce` closures for zero-overhead execution
- Priority levels and deadline tracking
- Dependency relationships
- Creation timestamps and duration estimates

### Memory Management
Efficient memory pool allocation:
- Pre-allocated pools for common sizes
- Arena-based allocation patterns
- NUMA-aware allocation strategies

### Time Management
Precise temporal coordination:
- Hardware timestamp counter (TSC) integration
- Rate limiting and deadline scheduling
- Global time source abstraction

## Performance Targets

The CSF-Kernel is designed to meet aggressive performance requirements:

| Metric | Target | Implementation |
|--------|---------|----------------|
| Task Latency | <5μs | Channel-based executor, CPU affinity |
| Throughput | >1M msg/sec | Multi-threaded workers, zero-copy |
| Memory Safety | 100% | No unwrap/expect, comprehensive error handling |
| Real-time | Hard deadlines | Priority scheduling, deadline monitoring |

## API Quick Reference

### Creating and Configuring Executor

```rust
use csf_kernel::executor::{TaskExecutor, ExecutorConfig};

let config = ExecutorConfig {
    worker_threads: 8,
    target_latency_ns: 2_000, // 2μs target
    max_queue_depth: 10_000,
    enable_timing_metrics: true,
    executor_cores: vec![4, 5, 6, 7], // Dedicated cores
};

let executor = TaskExecutor::new(config)?;
```

### Task Creation and Submission

```rust
use csf_kernel::task::Task;
use csf_core::Priority;

let task = Task::new("critical_task", Priority::High, move || {
    // Task logic here
    Ok(())
});

let task_id = executor.submit_task(task)?;
let result = executor.wait_for_completion(task_id, Duration::from_millis(10))?;
```

### Performance Monitoring

```rust
let stats = executor.stats();
let avg_latency = stats.avg_latency_ns.load(Ordering::Relaxed);

if !executor.is_meeting_latency_target() {
    eprintln!("Latency target exceeded: {}ns", avg_latency);
}
```

## Integration with ARES CSF

The CSF-Kernel integrates with other ARES components:

- **CSF-Bus**: Receives tasks from Phase Coherence Bus
- **CSF-Time**: Uses global time source for precise timing
- **CSF-Core**: Implements core traits and error types
- **CSF-Network**: Distributes tasks across nodes when scaling

## Development Guidelines

### Memory Safety
- All public APIs use `Result<T, Error>` patterns
- No `unwrap()` or `expect()` in library code
- Comprehensive error propagation and handling

### Performance
- Zero-allocation hot paths where possible
- Lock-free data structures for statistics
- CPU affinity for deterministic scheduling
- Hardware timestamp counters for precise timing

### Testing
- Property-based testing with `proptest`
- Concurrency testing with `Loom`
- Fuzz testing for error handling
- Integration tests with deterministic time sources

### Documentation
- All public APIs documented
- Architecture decision records
- Performance benchmarks and analysis
- Usage examples and best practices

## Error Handling

The kernel uses structured error types for different failure modes:

```rust
pub enum Error {
    InvalidConfig(String),
    NoSchedulerAvailable,
    System(nix::Error),
    Other(anyhow::Error),
}
```

Common error scenarios:
- Configuration validation failures
- Resource exhaustion (queue full)
- Task execution timeouts
- System-level scheduling errors

## Future Roadmap

Planned enhancements:
1. **SIMD Optimizations**: Vectorized task processing
2. **GPU Offloading**: CUDA/OpenCL integration for parallel tasks
3. **Adaptive Scheduling**: ML-based load prediction
4. **Distributed Execution**: Multi-node task distribution
5. **Hardware Acceleration**: FPGA integration for ultra-low latency

## Contributing

When contributing to CSF-Kernel:

1. Maintain sub-microsecond performance requirements
2. Ensure memory safety (no unsafe code without justification)
3. Add comprehensive tests for new functionality
4. Update documentation for API changes
5. Follow the existing architectural patterns

See the main ARES project guidelines for detailed contribution instructions.