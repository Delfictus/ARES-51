# ARES Goal 2: Phase Coherence Bus (PCB) Complete Implementation

## Primary Objective
Complete the zero-copy, lock-free Phase Coherence Bus (PCB) implementation with hardware-accelerated routing to achieve <1μs local message passing latency and >1M messages/sec throughput as specified in ARES Strategic Roadmap Goal 2.

## Context Overview

You are working on the **NovaCore ARES ChronoSynclastic Fabric (CSF)** - a next-generation real-time computing platform. The Phase Coherence Bus is the central communication backbone that enables distributed processing with sub-microsecond latency.

**Foundation Status**: Goal 1 (TTW foundation) is COMPLETE with csf-time crate providing TimeSource, HlcClock, and DeadlineScheduler implementations. Build on this foundation.

## Critical Architecture Constraints

1. **Zero-Copy Semantics**: Use `Arc<PhasePacket<T>>` and `bytes::Bytes` for all payload handling - never clone large data
2. **Lock-Free Operation**: Use `DashMap`, `crossbeam`, and atomic operations - no `Mutex` on hot paths  
3. **Hardware Acceleration**: Integrate with TSC (Time Stamp Counter) and SIMD optimizations
4. **Temporal Coherence**: All bus operations must use `TimeSource` from csf-time, never `Instant::now()`
5. **Sub-microsecond Latency**: Target <1μs for local message passing, measure with high-precision timing
6. **Million Message Throughput**: Design for >1M messages/sec sustained throughput
7. **Backpressure**: Bounded channels with configurable limits, fail fast on overload
8. **Type Safety**: Strongly-typed pub/sub with compile-time guarantees

## Implementation Priorities

### Phase 2.1a: Core Bus Infrastructure (Week 1-2)
**Complete EventBusTx/EventBusRx trait implementations**

```rust
// Target API in csf-bus/src/traits.rs
#[async_trait]
pub trait EventBusTx: Send + Sync {
    async fn publish<T: Any + Send + Sync + Clone + 'static>(&self, packet: PhasePacket<T>) -> BusResult<MessageId>;
    fn try_publish<T: Any + Send + Sync + Clone + 'static>(&self, packet: PhasePacket<T>) -> BusResult<MessageId>;
    fn publish_batch<T: Any + Send + Sync + Clone + 'static>(&self, packets: Vec<PhasePacket<T>>) -> BusResult<Vec<MessageId>>;
    fn get_stats(&self) -> BusStats;
}

#[async_trait] 
pub trait EventBusRx: Send + Sync {
    async fn subscribe<T: Any + Send + Sync + Clone + 'static>(&self) -> BusResult<Receiver<PhasePacket<T>>>;
    fn subscribe_filtered<T, F>(&self, filter: F) -> BusResult<Receiver<PhasePacket<T>>>
    where
        F: Fn(&PhasePacket<T>) -> bool + Send + Sync + 'static;
    fn unsubscribe<T: Any + Send + Sync + Clone + 'static>(&self, subscription_id: SubscriptionId) -> BusResult<()>;
}
```

**Key Focus**:
- Implement zero-copy message routing with `Arc<PhasePacket<dyn Any>>`
- Add message batching for throughput optimization  
- Use `DashMap<TypeId, Vec<BusSender>>` for lock-free subscription management
- Integrate temporal coherence with `TimeSource` for all timing operations

### Phase 2.1b: Hardware-Accelerated Routing (Week 2-3)
**Implement sub-microsecond message routing with hardware optimization**

```rust
// Target implementation in csf-bus/src/routing.rs
pub struct HardwareRouter {
    routing_table: DashMap<TypeId, RouteEntry>,
    tsc_calibration: Arc<TscCalibration>,
    simd_optimizer: SimdMessageOptimizer,
}

impl HardwareRouter {
    pub fn route_message<T>(&self, packet: Arc<PhasePacket<T>>) -> RouteResult {
        let start_tsc = self.read_tsc();
        // SIMD-optimized subscriber lookup
        // Zero-copy message distribution  
        // Hardware-accelerated serialization
        let end_tsc = self.read_tsc();
        self.record_latency(end_tsc - start_tsc);
    }
}
```

**Key Focus**:
- Use TSC (Time Stamp Counter) for sub-nanosecond latency measurement
- Implement SIMD-optimized subscriber matching
- Add CPU cache-line aligned data structures for performance
- Integrate with csf-time for temporal correlation

### Phase 2.1c: Complete PhasePacket System (Week 3-4)
**Finalize zero-copy packet handling with quantum temporal correlation**

```rust
// Enhanced PhasePacket in csf-bus/src/packet.rs
pub struct PhasePacket<T> {
    pub id: MessageId,
    pub timestamp: LogicalTime,  // From csf-time HlcClock
    pub payload: T,
    pub routing_metadata: RoutingMetadata,
    pub quantum_correlation: QuantumCorrelation,
    pub trace_span: tracing::Span,
}

impl<T> PhasePacket<T> {
    pub fn with_quantum_optimization(payload: T, quantum_offset: QuantumOffset) -> Self;
    pub fn add_temporal_correlation(&mut self, causal_deps: Vec<MessageId>);
    pub fn serialize_zero_copy(&self) -> bytes::Bytes where T: Serialize;
}
```

**Key Focus**:
- Integrate quantum optimization from csf-time oracle
- Add causal dependency tracking for temporal coherence
- Implement efficient serialization with `bytes::Bytes`
- Add comprehensive tracing for observability

### Phase 2.1d: Performance Validation (Week 4)
**Achieve and validate performance targets**

```rust
// Performance tests in csf-bus/benches/
#[bench]
fn bench_local_message_latency(b: &mut Bencher) {
    // Target: <1μs p99 latency for local messages
}

#[bench] 
fn bench_throughput_sustained(b: &mut Bencher) {
    // Target: >1M messages/sec sustained
}
```

**Key Focus**:
- Implement high-precision latency measurement with TSC
- Add throughput benchmarks with realistic payloads
- Memory allocation profiling to ensure zero-copy
- Integration tests with csf-time temporal validation

## Implementation Commands & Workflow

### Search and Analysis
```bash
# Analyze current PCB implementation
rg -A5 -B5 "EventBus|PhasePacket|publish|subscribe" crates/csf-bus/
rg "unwrap\|expect\|panic" crates/csf-bus/ # Find unsafe code to fix
rg "Instant::now\|SystemTime::now" crates/csf-bus/ # Find time violations
```

### Code Generation
```bash
# Generate performance-critical code sections
fd -e rs . crates/csf-bus/src/ -x wc -l {} + # Assess current implementation size
cargo expand --package csf-bus # Review macro expansions for optimization
```

### Testing and Validation  
```bash
# Run comprehensive testing
cargo test -p csf-bus --all-features
cargo test -p csf-bus -- --ignored # Run performance tests  
cargo bench -p csf-bus # Run benchmarks
miri cargo test -p csf-bus # Check for UB in unsafe code
```

## Success Metrics (Must Achieve)

### Performance KPIs
- [ ] **Local Message Latency**: <1μs p99 latency measured with TSC
- [ ] **Sustained Throughput**: >1M messages/sec for 60+ seconds  
- [ ] **Memory Efficiency**: Zero heap allocations on message hot path
- [ ] **CPU Efficiency**: <10% CPU utilization at target throughput
- [ ] **Temporal Coherence**: 100% causality preservation with csf-time integration

### Code Quality KPIs  
- [ ] **Zero Unsafe Code**: All unsafe isolated to `unsafe/` modules with documentation
- [ ] **Zero Panics**: No `unwrap`/`expect`/`panic` in library code
- [ ] **Zero Time Violations**: All timing via `TimeSource`, no `Instant::now()`
- [ ] **Complete API Coverage**: All EventBusTx/EventBusRx methods implemented
- [ ] **Comprehensive Tests**: >90% line coverage including property tests

### Integration KPIs
- [ ] **csf-time Integration**: All bus operations use TTW TimeSource  
- [ ] **csf-kernel Integration**: Scheduler uses bus for task communication
- [ ] **Hardware Acceleration**: TSC and SIMD optimizations functional
- [ ] **Observability**: Complete tracing spans and Prometheus metrics
- [ ] **Documentation**: All public APIs documented with examples

## Architecture Guidelines

### Zero-Copy Message Flow
```
Publisher → PhasePacket<T> → Arc::new() → TypeId Lookup → 
SIMD Route Match → Hardware Send → Zero-Copy Delivery → Subscriber
```

### Lock-Free Subscription Management
```
DashMap<TypeId, Vec<Sender>> → Atomic Updates → 
Lock-Free Publisher Lookup → Concurrent Subscriber Addition
```

### Hardware-Accelerated Path
```  
TSC Timestamp → SIMD Subscriber Match → CPU Cache Optimization →
Zero-Copy Buffer → Hardware Send → Latency Recording
```

## Error Handling Patterns

```rust
// Use thiserror for structured errors
#[derive(Error, Debug)]
pub enum BusError {
    #[error("Subscription failed for type {type_name}: {reason}")]
    SubscriptionFailed { type_name: String, reason: String },
    
    #[error("Message publish timeout after {timeout_ms}ms")]
    PublishTimeout { timeout_ms: u64 },
    
    #[error("Hardware acceleration unavailable: {details}")]
    HardwareUnavailable { details: String },
}

// Never use unwrap/expect - always return proper errors
pub fn publish_message<T>(&self, packet: PhasePacket<T>) -> BusResult<MessageId> {
    self.validate_packet(&packet)?;  // Not: packet.validate().unwrap()
    self.route_with_timeout(packet)  // Not: route().expect("routing failed")
}
```

## Dependencies and Integration

### Required Crates
```toml
[dependencies]
# Foundation (already implemented)
csf-core = { path = "../csf-core" }
csf-time = { path = "../csf-time" }  # Use for all timing

# Performance optimization
crossbeam = "0.8"         # Lock-free data structures  
dashmap = "5.5"           # Concurrent hash maps
bytes = "1.5"             # Zero-copy buffers
parking_lot = "0.12"      # Fast synchronization

# Hardware acceleration  
rayon = "1.8"             # Parallel processing
simd-json = "0.13"        # SIMD JSON processing

# Observability
tracing = "0.1"           # Distributed tracing
metrics = "0.21"          # Prometheus metrics

# Testing
proptest = "1.3"          # Property-based testing
criterion = "0.5"         # Benchmarking
```

### Integration Points
```rust
// csf-time integration (CRITICAL)
use csf_time::{TimeSource, global_time_source, LogicalTime};

// csf-kernel integration  
use csf_kernel::scheduler::TaskScheduler;

// csf-network integration (future)
use csf_network::transport::QuicTransport;
```

## Quality Gates (All Must Pass)

### Before Committing Code
```bash
# Format and lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Core functionality
cargo test -p csf-bus --all-features

# Performance validation
cargo bench -p csf-bus -- --measurement-time 10

# Memory safety
miri cargo test -p csf-bus

# Build verification
cargo build --all-targets --all-features
```

### Before Milestone Completion
```bash
# Integration testing
cargo test --workspace --all-features

# Performance profiling  
perf record --call-graph dwarf cargo bench -p csf-bus
flamegraph --output bus-profile.svg perf.data

# Security audit
cargo audit
cargo deny check

# Documentation build
cargo doc --no-deps --open
```

## Implementation Notes

1. **Start with Traits**: Implement EventBusTx/EventBusRx traits completely before optimizing
2. **Measure Early**: Add TSC timing instrumentation from the beginning  
3. **Test Driven**: Write property tests for message ordering and delivery guarantees
4. **Profile Guided**: Use `perf` and `flamegraph` to identify hot paths for optimization
5. **Memory Conscious**: Use `heaptrack` to verify zero-copy operation
6. **Integration First**: Ensure csf-time integration works before hardware acceleration
7. **Document Trade-offs**: Explain performance vs complexity decisions in code comments

## Next Phase Dependencies

This Goal 2 implementation enables:
- **Goal 3**: ChronoSynclastic deterministic operation across distributed nodes
- **Goal 4**: Byzantine consensus layer using PCB for coordination  
- **Goal 5**: Production observability and monitoring of bus performance
- **Goal 6**: C-LOGIC neuromorphic modules using PCB for communication

## Critical Success Factors

1. **Zero-Copy Architecture**: Every memory allocation must be justified and measured
2. **Hardware Integration**: TSC timing and SIMD optimization are non-negotiable for latency targets  
3. **Temporal Coherence**: Perfect integration with csf-time TimeSource - no exceptions
4. **Performance Validation**: Continuous benchmarking and profiling throughout development
5. **Production Quality**: No technical debt - every line must be production-grade from start

This implementation will establish the NovaCore CSF as the highest-performance real-time messaging system in the Rust ecosystem, enabling the advanced distributed computing capabilities required for Goals 3-7.