---
sidebar_position: 3
title: "Core Components"
description: "Detailed overview of ARES ChronoFabric core system components"
---

# Core Components

ARES ChronoFabric is built around several core components that work together to provide sub-microsecond messaging and temporal coherence.

## Component Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                    │
├─────────────────────────────────────────────────────┤
│  Phase Coherence Bus  │  Temporal Task Weaver      │
│  (Message Routing)    │  (Time Management)         │
├─────────────────────────────────────────────────────┤
│  Hardware Router      │  Quantum Time Oracle       │
│  (SIMD Optimization)  │  (Predictive Scheduling)   │
├─────────────────────────────────────────────────────┤
│              CSF Core Runtime                       │
│         (Foundation Types & Error Handling)         │
└─────────────────────────────────────────────────────┘
```

## CSF-Core: Foundation Layer

**Purpose**: Provides foundational types, traits, and error handling for all system components.

**Key Features**:
- Strongly typed identifiers (`ComponentId`, `MessageId`, `TaskId`)
- Comprehensive error handling with `thiserror`
- Configuration management
- Port traits for hexagonal architecture

**API Overview**:
```rust
// Core types
pub use types::{ComponentId, NanoTime, PacketId, Priority, TaskId};

// Port traits
pub use ports::{
    Consensus, DeadlineScheduler, EventBusRx, EventBusTx, 
    HlcClock, SecureImmutableLedger, TimeSource
};
```

## Phase Coherence Bus (PCB)

**Purpose**: High-performance message routing with zero-copy semantics and hardware acceleration.

**Key Features**:
- Sub-microsecond routing latency
- SIMD-optimized subscriber matching
- Zero-copy message passing with `Arc<PhasePacket<T>>`
- Temporal coherence validation
- Backpressure and flow control

**Performance Targets**:
- Latency: < 1μs per message
- Throughput: > 1M messages/second
- Memory: Zero-copy operations

## Temporal Task Weaver (TTW)

**Purpose**: Provides precise timing, causality tracking, and deadline scheduling.

**Key Features**:
- Hybrid Logical Clocks (HLC) for causality
- Sub-microsecond time precision with TSC
- Deadline-aware task scheduling
- Quantum-inspired temporal optimization
- Distributed time synchronization

**Time Sources**:
- System time with TSC calibration
- Mock time for deterministic testing
- Network-synchronized time for clusters

## Quantum Time Oracle

**Purpose**: Provides quantum-inspired optimization for temporal scheduling and message routing.

**Key Features**:
- Predictive scheduling algorithms
- Temporal phase alignment
- Coherence score calculation
- Adaptive optimization strategies
- Performance feedback loops

## Hardware Router

**Purpose**: Hardware-accelerated message routing with SIMD optimization.

**Key Features**:
- TSC-based timing calibration
- SIMD subscriber matching
- Temporal message queuing
- Performance monitoring
- Dynamic routing optimization

## Component Interaction

### Message Flow
1. Application creates `PhasePacket<T>` with payload
2. PCB receives packet and validates temporal coherence
3. Hardware Router performs SIMD-optimized routing
4. TTW schedules delivery based on deadlines
5. Quantum Oracle provides optimization hints

### Time Management
1. Time Source provides nanosecond precision timing
2. HLC Clock tracks causality relationships
3. TTW manages deadline scheduling
4. Quantum Oracle optimizes temporal behavior

### Error Handling
1. All components use `thiserror` for structured errors
2. Errors propagate through the system consistently  
3. Performance monitoring tracks error rates
4. Graceful degradation under fault conditions

## Configuration

All components are configured through the unified configuration system in `csf-core::config`:

```toml
[bus]
queue_capacity = 1024
enable_quantum_optimization = true

[time]
time_source = "system"
enable_hardware_timing = true
quantum_optimization_level = "balanced"

[routing]
enable_simd = true
subscriber_cache_size = 10000
```

## Integration Patterns

### Publisher Pattern
```rust
let packet = PhasePacket::new(data, source_id)
    .with_priority(Priority::High)
    .with_deadline(deadline_time);
    
bus.publish(packet).await?;
```

### Subscriber Pattern  
```rust
let mut rx = bus.subscribe::<MyMessageType>().await?;
while let Some(packet) = rx.recv().await {
    process_message(packet.payload).await;
}
```

### Time-Critical Processing
```rust
let deadline = time_source.now_ns()? + Duration::from_micros(100);
let packet = packet.with_deadline(deadline);
bus.publish_with_deadline(packet, deadline).await?;
```

## Monitoring and Observability

All core components provide comprehensive metrics:

- **Performance Metrics**: Latency, throughput, error rates
- **Resource Usage**: Memory, CPU, network bandwidth  
- **Temporal Metrics**: Clock drift, coherence scores
- **System Health**: Component status, fault detection

See the [Observability Guide](../operations/observability.md) for complete monitoring setup.