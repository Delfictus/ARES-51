# ARES ChronoFabric API Reference Guide

**Version**: 1.0  
**Last Updated**: August 26, 2025  
**Document Type**: API Reference  

> **⚡ High-Performance APIs**: This reference covers production-grade APIs with sub-microsecond latency requirements and thread-safety guarantees.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [CSF-Bus APIs](#csf-bus-apis)  
3. [CSF-Time APIs](#csf-time-apis)
4. [CSF-Core APIs](#csf-core-apis)
5. [Hardware Router APIs](#hardware-router-apis)
6. [C-LOGIC Module APIs](#c-logic-module-apis)
7. [MLIR Runtime APIs](#mlir-runtime-apis)
8. [Error Handling](#error-handling)
9. [Type Definitions](#type-definitions)
10. [Usage Examples](#usage-examples)

---

## Quick Reference

### Essential Imports

```rust
// Core functionality
use csf_core::prelude::*;
use csf_bus::{PhaseCoherenceBus, PhasePacket, BusConfig};
use csf_time::{TimeSource, HlcClock, NanoTime, QuantumOffset};

// Advanced features  
use csf_clogic::CLogicSystem;
use csf_mlir::{MlirRuntime, Backend};
use csf_hardware::HardwareRouter;
```

### Key Types at a Glance

```rust
// Message types
PhasePacket<T>                    // Primary message container
SharedPacket                      // Arc<PhasePacket<dyn Any + Send + Sync>>
MessageId                         // Uuid for message tracking

// Bus interfaces
PhaseCoherenceBus                 // Main event bus implementation  
EventBusTx/EventBusRx            // Pub/sub trait interfaces
BusConfig                         // Bus configuration options

// Time management
NanoTime                          // High-precision time representation
LogicalTime                       // HLC with causality tracking
QuantumOffset                     // Quantum optimization parameters

// Component identification
ComponentId                       // Typed component identifiers
Priority                          // Message priority levels
TaskId                           // Task identification
```

---

## CSF-Bus APIs

### PhaseCoherenceBus

The main event bus for zero-copy, hardware-accelerated message passing.

#### Construction

```rust
impl PhaseCoherenceBus {
    /// Create new bus with configuration
    pub fn new(config: BusConfig) -> BusResult<Self>;
    
    /// Create with custom time source (for testing)
    pub fn with_time_source(
        config: BusConfig, 
        time_source: Arc<dyn TimeSource>
    ) -> Self;
}

/// Bus configuration options
#[derive(Debug, Clone)]
pub struct BusConfig {
    /// Buffer size for each message type channel
    pub channel_buffer_size: usize,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self { channel_buffer_size: 1024 }
    }
}
```

#### Publishing Messages (EventBusTx)

```rust
#[async_trait::async_trait]
impl EventBusTx for PhaseCoherenceBus {
    /// Publish single message with hardware acceleration
    async fn publish<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
    ) -> BusResult<MessageId>;

    /// Publish multiple messages efficiently
    async fn publish_batch<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packets: Vec<PhasePacket<T>>,
    ) -> BusResult<Vec<MessageId>>;

    /// Non-blocking publish attempt
    fn try_publish<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
    ) -> BusResult<MessageId>;

    /// Get comprehensive bus statistics
    fn get_stats(&self) -> BusStats;

    /// Count active subscribers for message type
    fn subscriber_count<T: Any + Send + Sync + Clone + 'static>(&self) -> usize;

    /// Check bus operational health
    fn is_healthy(&self) -> bool;
}
```

#### Subscribing to Messages (EventBusRx)

```rust
#[async_trait::async_trait]
impl EventBusRx for PhaseCoherenceBus {
    /// Subscribe to messages of specific type
    async fn subscribe<T: Any + Send + Sync + Clone + 'static>(
        &self
    ) -> BusResult<Receiver<T>>;

    /// Subscribe with message filtering predicate
    fn subscribe_filtered<T, F>(&self, filter: F) -> BusResult<Receiver<T>>
    where
        T: Any + Send + Sync + Clone + 'static,
        F: Fn(&PhasePacket<T>) -> bool + Send + Sync + 'static;

    /// Remove subscription by ID
    fn unsubscribe<T: Any + Send + Sync + Clone + 'static>(
        &self,
        subscription_id: SubscriptionId,
    ) -> BusResult<()>;

    /// Get all active subscription IDs
    fn active_subscriptions(&self) -> Vec<SubscriptionId>;

    /// Get total subscription count across all types
    fn subscription_count(&self) -> usize;
}
```

#### Advanced TTW Integration

```rust
impl PhaseCoherenceBus {
    /// Publish with temporal deadline scheduling
    pub async fn publish_with_deadline<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
        deadline: NanoTime,
    ) -> BusResult<MessageId>;

    /// Publish with causal dependencies
    pub async fn publish_with_dependencies<T: Any + Send + Sync + Clone + 'static>(
        &self,
        mut packet: PhasePacket<T>,
        causal_dependencies: Vec<MessageId>,
    ) -> BusResult<MessageId>;

    /// Process pending temporal messages
    pub fn process_temporal_queue(&self) -> usize;

    /// Get quantum routing optimization hints
    pub fn get_routing_optimization_hints(&self) -> OptimizationHint;

    /// Get comprehensive temporal coherence metrics
    pub fn get_temporal_metrics(&self) -> TemporalCoherenceMetrics;

    /// Optimize temporal routing based on current workload
    pub fn optimize_temporal_routing(&self) -> OptimizationResult;

    /// Enable/disable quantum-optimized routing  
    pub fn set_quantum_optimization(&self, enabled: bool);
}
```

### PhasePacket<T>

The fundamental message container with quantum temporal correlation.

#### Construction and Builder Pattern

```rust
impl<T> PhasePacket<T> where T: Sized {
    /// Create new packet with HLC temporal coherence
    pub fn new(payload: T, source_id: ComponentId) -> Self;

    /// Create with quantum optimization
    pub fn with_quantum_optimization(
        payload: T, 
        quantum_offset: QuantumOffset
    ) -> Self;

    /// Set message priority (builder pattern)
    pub fn with_priority(self, priority: Priority) -> Self;

    /// Set processing deadline
    pub fn with_deadline(self, deadline_ns: NanoTime) -> Self;

    /// Set target component bitmask for SIMD routing
    pub fn with_targets(self, targets: u64) -> Self;

    /// Set source task for causal tracking
    pub fn with_source_task(self, task_id: TaskId) -> Self;

    /// Configure delivery options
    pub fn with_delivery_options(self, options: DeliveryOptions) -> Self;

    /// Enable guaranteed delivery with retries
    pub fn with_guaranteed_delivery(self, max_retries: u8) -> Self;

    /// Set delivery timeout
    pub fn with_timeout(self, timeout_ns: u64) -> Self;
}
```

#### Data Access and Manipulation

```rust
impl<T> PhasePacket<T> {
    /// Add temporal correlation with causal dependencies
    pub fn add_temporal_correlation(&mut self, causal_deps: Vec<MessageId>);

    /// Serialize to zero-copy bytes using quantum-optimized encoding
    pub fn serialize_zero_copy(&self) -> BusResult<Bytes>
    where T: Serialize;

    /// Get message size for memory optimization
    pub fn message_size(&self) -> usize;

    /// Check temporal coherence requirements
    pub fn is_temporally_coherent(&self) -> bool;

    /// Get quantum-optimized timestamp for scheduling
    pub fn quantum_timestamp(&self) -> LogicalTime;
}
```

#### Type Erasure Support

```rust
impl<T: Any + Send + Sync> PhasePacket<T> {
    /// Convert to type-erased packet for dynamic dispatch
    pub fn into_erased(self) -> PhasePacket<dyn Any + Send + Sync>;
}

/// Production-grade shared packet utilities
impl PhasePacket<dyn Any + Send + Sync> {
    /// Create shared packet from any Send + Sync payload
    pub fn new_shared<T: Any + Send + Sync>(
        payload: T, 
        source_id: ComponentId
    ) -> Arc<Self>;
    
    /// Create shared packet with quantum optimization
    pub fn new_shared_with_quantum<T: Any + Send + Sync>(
        payload: T, 
        quantum_offset: QuantumOffset
    ) -> Arc<Self>;
    
    /// Safely downcast payload to specific type
    pub fn downcast_payload<T: Any + Send + Sync>(&self) -> Option<&T>;
    
    /// Check concurrent processing safety
    pub fn is_concurrent_safe(&self) -> bool;
}
```

### Message Receiver

```rust
/// Type-safe message receiver with async support
pub struct Receiver<T> {
    inner: mpsc::Receiver<PhasePacket<T>>,
}

impl<T> Receiver<T> {
    /// Receive next message (async)
    pub async fn recv(&mut self) -> Option<PhasePacket<T>>;

    /// Try to receive without blocking
    pub fn try_recv(&mut self) -> Result<PhasePacket<T>, TryRecvError>;

    /// Close receiver to stop receiving messages
    pub fn close(&mut self);
    
    /// Check if receiver is closed
    pub fn is_closed(&self) -> bool;
}
```

---

## CSF-Time APIs

### TimeSource Trait

Core time management interface with nanosecond precision.

```rust
/// Deterministic time source for ChronoSynclastic coherence
pub trait TimeSource: Send + Sync {
    /// Get current time with nanosecond precision
    fn now_ns(&self) -> TimeResult<NanoTime>;

    /// Get quantum offset for optimization
    fn quantum_offset(&self) -> QuantumOffset;

    /// Check hardware timing availability
    fn has_hardware_timing(&self) -> bool;

    /// Create temporal checkpoint for coherence validation
    fn create_checkpoint(&self, name: &str) -> TimeResult<TimeCheckpoint>;

    /// Validate temporal coherence between checkpoints
    fn validate_coherence(
        &self, 
        from: &TimeCheckpoint, 
        to: &TimeCheckpoint
    ) -> bool;
}
```

### TimeSourceImpl

Production time source implementation.

```rust
impl TimeSourceImpl {
    /// Create new time source with hardware calibration
    pub fn new() -> TimeResult<Self>;
    
    /// Get current quantum state for optimization
    pub fn quantum_state(&self) -> QuantumState;
    
    /// Force recalibration of hardware timing
    pub fn recalibrate(&self) -> TimeResult<()>;
}
```

### NanoTime

High-precision time representation.

```rust
/// Nanosecond-precision time with arithmetic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NanoTime(u64);

impl NanoTime {
    /// Zero time constant  
    pub const ZERO: NanoTime = NanoTime(0);
    
    /// Create from nanoseconds
    pub fn from_nanos(nanos: u64) -> Self;
    
    /// Create from duration
    pub fn from_duration(duration: Duration) -> Self;
    
    /// Get nanoseconds value
    pub fn as_nanos(&self) -> u64;
    
    /// Convert to standard Duration
    pub fn as_duration(&self) -> Duration;
    
    /// Add duration (saturating)
    pub fn saturating_add(&self, duration: Duration) -> Self;
    
    /// Subtract duration (saturating)
    pub fn saturating_sub(&self, duration: Duration) -> Self;
    
    /// Get elapsed time since this timestamp
    pub fn elapsed(&self) -> TimeResult<Duration>;
}

// Arithmetic trait implementations
impl Add<Duration> for NanoTime { /* ... */ }
impl Sub<Duration> for NanoTime { /* ... */ }
impl Sub<NanoTime> for NanoTime { /* ... */ }
```

### HLC Clock Implementation

Hybrid Logical Clock with causality tracking.

```rust
/// HLC Clock trait interface
pub trait HlcClock: Send + Sync {
    /// Get current logical time
    fn now(&self) -> TimeResult<LogicalTime>;
    
    /// Update clock with received logical time
    fn update(&self, received_time: LogicalTime) -> TimeResult<CausalityResult>;
    
    /// Check causality relationship
    fn happens_before(&self, t1: LogicalTime, t2: LogicalTime) -> bool;
}

/// Production HLC implementation
pub struct HlcClockImpl {
    node_id: u16,
    logical_time: Arc<RwLock<LogicalTime>>,
    time_source: Arc<dyn TimeSource>,
    quantum_oracle: Arc<QuantumTimeOracle>,
}

impl HlcClockImpl {
    /// Create new HLC clock
    pub fn new(node_id: u16, time_source: Arc<dyn TimeSource>) -> TimeResult<Self>;
    
    /// Create with custom configuration
    pub fn with_config(
        node_id: u16,
        initial_time: LogicalTime,
        time_source: Arc<dyn TimeSource>,
        quantum_oracle: Arc<QuantumTimeOracle>,
        buffer_size: usize,
    ) -> Self;
}
```

### LogicalTime

Logical timestamp with causality information.

```rust
/// Logical time with causality tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LogicalTime {
    /// Physical timestamp (nanoseconds)
    pub physical: u64,
    /// Logical counter for causality
    pub logical: u64,
    /// Node identifier  
    pub node_id: u16,
}

impl LogicalTime {
    /// Create new logical time
    pub fn new(physical: u64, logical: u64, node_id: u16) -> Self;
    
    /// Zero time for node
    pub fn zero(node_id: u16) -> Self;
    
    /// Increment logical counter
    pub fn increment(&mut self);
    
    /// Update with physical time
    pub fn update_physical(&mut self, physical_time: u64);
    
    /// Check if this time happens before another
    pub fn happens_before(&self, other: &LogicalTime) -> bool;
    
    /// Check if times are concurrent (neither happens before the other)
    pub fn concurrent_with(&self, other: &LogicalTime) -> bool;
}
```

### Global Time Functions

Convenient global access to time services.

```rust
/// Initialize global time source
pub fn initialize_global_time_source() -> TimeResult<()>;

/// Initialize simulated time source for testing
pub fn initialize_simulated_time_source(start_time: NanoTime);

/// Get global time source instance
pub fn global_time_source() -> &'static dyn TimeSource;

/// Get current time from global source
pub fn now() -> TimeResult<NanoTime>;

/// Global HLC functions
pub fn initialize_global_hlc(
    time_source: Arc<dyn TimeSource>, 
    node_id: u16
) -> TimeResult<()>;

pub fn global_hlc_now() -> TimeResult<LogicalTime>;
pub fn global_hlc_update(received_time: LogicalTime) -> TimeResult<CausalityResult>;
pub fn is_global_hlc_initialized() -> bool;
```

### Deadline Scheduler APIs

```rust
/// Global deadline scheduling functions
pub fn initialize_global_deadline_scheduler(
    time_source: Arc<dyn TimeSource>
) -> TimeResult<()>;

/// Schedule task with deadline
pub async fn global_schedule_with_deadline(
    task: Task, 
    deadline: NanoTime
) -> TimeResult<()>;

/// Schedule task after delay
pub async fn global_schedule_after(
    task: Task, 
    delay: Duration
) -> TimeResult<()>;

/// Get current scheduler load (0.0 to 1.0)
pub fn global_deadline_load() -> f64;

/// Check if global scheduler is initialized
pub fn is_global_deadline_scheduler_initialized() -> bool;
```

### QuantumTimeOracle

Quantum-inspired optimization for temporal operations.

```rust
/// Quantum time oracle for optimization hints
pub struct QuantumTimeOracle {
    state: Arc<RwLock<QuantumState>>,
    strategies: Vec<OptimizationStrategy>,
}

impl QuantumTimeOracle {
    /// Create new quantum oracle
    pub fn new() -> Self;
    
    /// Get current quantum offset
    pub fn current_offset(&self) -> QuantumOffset;
    
    /// Get quantum offset for specific time
    pub fn current_offset_with_time(&self, time: NanoTime) -> QuantumOffset;
    
    /// Get optimization hint for current state
    pub fn get_optimization_hint(&self, current_time: NanoTime) -> OptimizationHint;
    
    /// Enable or disable quantum optimization
    pub fn set_enabled(&self, enabled: bool);
    
    /// Update quantum state based on performance metrics
    pub fn update_state(&self, metrics: &PerformanceMetrics);
}

/// Quantum offset for temporal optimization
#[derive(Debug, Clone, Copy)]
pub struct QuantumOffset {
    /// Amplitude of quantum optimization
    pub amplitude: f64,
    /// Frequency for periodic optimization
    pub frequency: f64,  
    /// Phase shift for temporal alignment
    pub phase: f64,
}

impl QuantumOffset {
    /// Create new quantum offset
    pub fn new(amplitude: f64, frequency: f64, phase: f64) -> Self;
    
    /// Apply offset to time value
    pub fn apply(&self, time: NanoTime) -> NanoTime;
    
    /// Calculate optimization score (0.0 to 1.0)
    pub fn optimization_score(&self) -> f64;
}
```

---

## CSF-Core APIs

### Core Types

#### ComponentId

Strongly-typed component identification.

```rust
/// Component identifier with type safety
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentId {
    // System components
    DRPP,
    ADP, 
    EGC,
    EMS,
    // Custom components
    Custom(u32),
    // Service components
    Service(Uuid),
}

impl ComponentId {
    /// Create custom component ID
    pub fn custom(id: u32) -> Self;
    
    /// Create service component ID  
    pub fn service(uuid: Uuid) -> Self;
    
    /// Get numeric representation for bitmasks
    pub fn as_u32(&self) -> u32;
    
    /// Check if component is system component
    pub fn is_system_component(&self) -> bool;
}
```

#### Priority

Message priority levels for scheduling.

```rust
/// Message priority for scheduling and routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1, 
    High = 2,
    Critical = 3,
}

impl Priority {
    /// Convert to numeric value
    pub fn as_u8(&self) -> u8;
    
    /// Create from numeric value
    pub fn from_u8(value: u8) -> Option<Self>;
    
    /// Check if priority is time-critical
    pub fn is_critical(&self) -> bool;
}
```

### Port Traits (Hexagonal Architecture)

Core interfaces defining the system boundaries.

```rust
/// Event bus transmitter interface
#[async_trait::async_trait]
pub trait EventBusTx: Send + Sync {
    async fn publish<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
    ) -> BusResult<MessageId>;
    
    // ... (other methods as shown above)
}

/// Event bus receiver interface  
#[async_trait::async_trait]
pub trait EventBusRx: Send + Sync {
    async fn subscribe<T: Any + Send + Sync + Clone + 'static>(
        &self
    ) -> BusResult<Receiver<T>>;
    
    // ... (other methods as shown above)
}

/// Time source interface
pub trait TimeSource: Send + Sync {
    fn now_ns(&self) -> TimeResult<NanoTime>;
    fn quantum_offset(&self) -> QuantumOffset;
    fn has_hardware_timing(&self) -> bool;
}

/// HLC clock interface
pub trait HlcClock: Send + Sync {
    fn now(&self) -> TimeResult<LogicalTime>;
    fn update(&self, received_time: LogicalTime) -> TimeResult<CausalityResult>;
}

/// Deadline scheduler interface
#[async_trait::async_trait]
pub trait DeadlineScheduler: Send + Sync {
    async fn schedule(&self, task: Task, deadline: NanoTime) -> TimeResult<()>;
    fn current_load(&self) -> f64;
    fn process_ready_tasks(&self) -> usize;
}
```

---

## Hardware Router APIs

### HardwareRouter

SIMD-optimized message routing with TSC timing.

```rust
/// Hardware-accelerated router with TTW temporal coherence
pub struct HardwareRouter {
    // Internal fields...
}

impl HardwareRouter {
    /// Create new hardware router with TTW integration
    pub fn new() -> BusResult<Self>;
    
    /// Create with specific time source
    pub fn with_time_source(time_source: Arc<dyn TimeSource>) -> BusResult<Self>;
    
    /// Route message to all subscribers
    pub fn route_message<T: Any + Send + Sync + 'static>(
        &self,
        packet: Arc<PhasePacket<T>>,
    ) -> RouteResult;
    
    /// Route with temporal coherence and causality tracking
    pub fn route_with_temporal_coherence(
        &self,
        packet: Arc<PhasePacket<dyn Any + Send + Sync + 'static>>,
    ) -> anyhow::Result<()>;
    
    /// Add route for message type
    pub fn add_route(&self, type_id: TypeId) -> Arc<RouteEntry>;
    
    /// Remove route for message type
    pub fn remove_route(&self, type_id: &TypeId) -> bool;
    
    /// Get route for message type
    pub fn get_route(&self, type_id: &TypeId) -> Option<Arc<RouteEntry>>;
}

// Performance and monitoring
impl HardwareRouter {
    /// Get comprehensive router statistics
    pub fn get_stats(&self) -> BusStats;
    
    /// Check if router meets performance targets
    pub fn is_healthy(&self) -> bool;
    
    /// Read hardware timestamp counter
    pub fn read_tsc(&self) -> u64;
    
    /// Record latency metric
    pub fn record_latency(&self, latency_tsc: u64);
}

// TTW integration methods
impl HardwareRouter {
    /// Process pending temporal messages
    pub fn process_pending_messages(&self) -> usize;
    
    /// Get quantum routing optimization hints
    pub fn get_quantum_routing_hints(&self) -> OptimizationHint;
    
    /// Schedule message delivery with deadline
    pub async fn schedule_message_delivery<T: Any + Send + Sync + 'static>(
        &self,
        packet: Arc<PhasePacket<T>>,
        deadline: NanoTime,
    ) -> BusResult<()>;
    
    /// Update temporal coherence metrics
    pub fn update_temporal_metrics(&self);
    
    /// Enable/disable quantum optimization
    pub fn set_quantum_optimization(&self, enabled: bool);
}
```

### Routing Metrics and Statistics

```rust
/// Metrics for a single routing operation
#[derive(Debug, Clone)]
pub struct RouteMetrics {
    /// TSC timestamp when routing started
    pub start_tsc: u64,
    /// TSC timestamp when routing completed
    pub end_tsc: u64,
    /// Number of subscribers reached
    pub subscribers_reached: usize,
    /// Number of delivery failures
    pub delivery_failures: usize,
    /// Message size in bytes
    pub message_size: usize,
}

impl RouteMetrics {
    /// Calculate latency in nanoseconds
    pub fn latency_ns(&self) -> u64;
    
    /// Check if meets <1μs latency target
    pub fn meets_latency_target(&self) -> bool;
}

/// Overall bus performance statistics
#[derive(Debug, Clone)]
pub struct BusStats {
    pub packets_published: u64,
    pub packets_delivered: u64, 
    pub packets_dropped: u64,
    pub active_subscriptions: u64,
    pub peak_latency_ns: u64,
    pub avg_latency_ns: u64,
    pub throughput_mps: u64, // messages per second
}
```

### TSC Calibration

```rust
/// TSC calibration for accurate timing
pub struct TscCalibration {
    pub frequency_hz: AtomicU64,
    pub calibrated_at: AtomicU64,
    pub is_calibrated: AtomicU64,
}

impl TscCalibration {
    /// Create and calibrate TSC
    pub fn new() -> Self;
    
    /// Recalibrate against system time
    pub fn calibrate(&self);
    
    /// Read TSC counter (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    pub fn read_tsc() -> u64;
    
    /// Convert TSC ticks to nanoseconds
    pub fn tsc_to_ns(&self, tsc_ticks: u64) -> u64;
    
    /// Check if calibration is valid
    pub fn is_valid(&self) -> bool;
}
```

---

## C-LOGIC Module APIs

### CLogicSystem

Coordinator for all cognitive logic modules.

```rust
/// C-LOGIC system coordinator
pub struct CLogicSystem {
    drpp: Arc<DynamicResonancePatternProcessor>,
    adp: Arc<AdaptiveDistributedProcessor>,
    egc: Arc<EmergentGovernanceController>,
    ems: Arc<EmotionalModelingSystem>,
    bus: Arc<Bus>,
    config: CLogicConfig,
}

impl CLogicSystem {
    /// Create new C-LOGIC system
    pub async fn new(bus: Arc<Bus>, config: CLogicConfig) -> Result<Self>;
    
    /// Start all modules
    pub async fn start(&self) -> Result<()>;
    
    /// Stop all modules
    pub async fn stop(&self) -> Result<()>;
    
    /// Get combined system state
    pub async fn get_state(&self) -> CLogicState;
}
```

### Common Module Trait

```rust
/// Common interface for all C-LOGIC modules
#[async_trait::async_trait]
pub trait CLogicModule: Send + Sync {
    /// Start the module
    async fn start(&self) -> Result<()>;
    
    /// Stop the module
    async fn stop(&self) -> Result<()>;
    
    /// Process input packet
    async fn process(&self, input: &PhasePacket) -> Result<PhasePacket>;
    
    /// Get module name
    fn name(&self) -> &str;
    
    /// Get module performance metrics
    async fn metrics(&self) -> ModuleMetrics;
}

/// Module performance metrics
#[derive(Debug, Clone, Default)]
pub struct ModuleMetrics {
    pub processed_packets: u64,
    pub processing_time_ns: u64,
    pub error_count: u64,
    pub last_update: NanoTime,
}
```

### Configuration Types

```rust
/// C-LOGIC system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLogicConfig {
    pub drpp: DrppConfig,
    pub adp: AdpConfig,
    pub egc: EgcConfig,
    pub ems: EmsConfig,
    pub enable_cross_talk: bool,
    pub update_frequency: f64,
}

/// DRPP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrppConfig {
    pub oscillator_count: usize,
    pub detection_threshold: f64,
    pub frequency_range: (f64, f64),
    pub update_rate_hz: f64,
}

/// Similar configuration structs for ADP, EGC, EMS...
```

---

## MLIR Runtime APIs

### MlirRuntime

Multi-backend hardware acceleration runtime.

```rust
/// MLIR Runtime for hardware acceleration
pub struct MlirRuntime {
    compiler: Arc<MlirCompiler>,
    execution_engine: Arc<ExecutionEngine>,
    memory_manager: Arc<MemoryManager>,
    backend_selector: Arc<BackendSelector>,
}

impl MlirRuntime {
    /// Create new MLIR runtime
    pub async fn new(config: RuntimeConfig) -> Result<Arc<Self>>;
    
    /// Compile MLIR module for target backend
    pub async fn compile(
        &self,
        module: MlirModule,
        backend: Backend,
    ) -> Result<CompiledArtifact>;
    
    /// Execute compiled module with inputs
    pub async fn execute(
        &self,
        artifact: CompiledArtifact,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>>;
}
```

### Backend Support

```rust
/// Supported hardware backends
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    CPU,      // Native CPU execution
    CUDA,     // NVIDIA GPU 
    HIP,      // AMD GPU
    Vulkan,   // Cross-platform GPU compute
    WebGPU,   // Browser/edge deployment
    TPU,      // Tensor Processing Units
    FPGA,     // Field-Programmable Gate Arrays
}

/// Hardware abstraction layer
#[async_trait::async_trait]
pub trait HardwareAbstraction: Send + Sync {
    /// Get available backends on this system
    fn available_backends(&self) -> Vec<Backend>;
    
    /// Select optimal backend for workload
    async fn select_backend(&self, module: &MlirModule) -> Result<Backend>;
    
    /// Allocate compute resources
    async fn allocate_resources(
        &self,
        requirements: &ResourceRequirements,
    ) -> Result<ResourceHandle>;
    
    /// Release allocated resources
    async fn release_resources(&self, handle: ResourceHandle) -> Result<()>;
}
```

### Quantum-Classical Interface

```rust
/// Interface for quantum-classical hybrid computing
#[async_trait::async_trait]
pub trait QuantumClassicalInterface: Send + Sync {
    /// Execute quantum circuit
    async fn execute_quantum(
        &self,
        circuit: &QuantumCircuit,
    ) -> Result<QuantumResult>;
    
    /// Transfer classical data to quantum
    async fn classical_to_quantum(&self, data: &[f64]) -> Result<QuantumState>;
    
    /// Transfer quantum results to classical
    async fn quantum_to_classical(
        &self,
        state: &QuantumState,
    ) -> Result<Vec<f64>>;
}
```

---

## Error Handling

### Error Types

Comprehensive error handling across all modules.

```rust
/// Primary bus error types
#[derive(Error, Debug)]
pub enum BusError {
    #[error("Subscription not found: {subscription_id}")]
    SubscriptionNotFound { subscription_id: SubscriptionId },

    #[error("Message delivery failed: {details}")]
    DeliveryFailed { details: String },

    #[error("Temporal violation: {details}")]
    TemporalViolation { details: String },

    #[error("Resource exhausted: {resource} - {details}")]
    ResourceExhausted { resource: String, details: String },

    #[error("Serialization failed: {details}")]
    SerializationFailed { details: String },

    #[error("Initialization failed: {component} - {reason}")]
    InitializationFailed { component: String, reason: String },

    #[error("Internal error: {details}")]
    Internal { details: String },

    #[error("Time source error: {details}")]
    TimeSourceError { details: String },
}

/// Time management error types
#[derive(Error, Debug)]
pub enum TimeError {
    #[error("Causality violation: expected {expected:?}, got {actual:?}")]
    CausalityViolation {
        expected: LogicalTime,
        actual: LogicalTime,
    },

    #[error("Clock sync failed: {reason}")]
    SyncFailure { reason: String },

    #[error("Quantum optimization error: {details}")]
    QuantumError { details: String },

    #[error("Deadline miss: task {task_id} missed deadline by {overage_ns}ns")]
    DeadlineMiss { task_id: String, overage_ns: u64 },

    #[error("System time error: {details}")]
    SystemTimeError { details: String },

    #[error("Hardware timing unavailable: {details}")]
    HardwareUnavailable { details: String },

    #[error("Time arithmetic overflow")]
    ArithmeticOverflow,
}
```

### Result Types

```rust
/// Result types for consistent error handling
pub type BusResult<T> = std::result::Result<T, BusError>;
pub type TimeResult<T> = std::result::Result<T, TimeError>;
pub type RouteResult = BusResult<RouteMetrics>;

/// Helper functions for error creation
impl BusError {
    pub fn subscription_not_found(id: SubscriptionId) -> Self {
        Self::SubscriptionNotFound { subscription_id: id }
    }
    
    pub fn delivery_failed(details: String) -> Self {
        Self::DeliveryFailed { details }
    }
    
    pub fn temporal_violation(details: &str) -> Self {
        Self::TemporalViolation { details: details.to_string() }
    }
    
    pub fn serialization_failed(details: String) -> Self {
        Self::SerializationFailed { details }
    }
    
    pub fn time_source_error(details: String) -> Self {
        Self::TimeSourceError { details }
    }
}
```

---

## Type Definitions

### Core Type Aliases

```rust
/// Message identification
pub type MessageId = Uuid;
pub type SubscriptionId = Uuid;
pub type TaskId = String;
pub type PacketId = Uuid;

/// Shared packet for concurrent access
pub type SharedPacket = Arc<PhasePacket<dyn Any + Send + Sync>>;
pub type DynamicPacket = PhasePacket<dyn Any + Send + Sync>;

/// Time representations
pub type Duration = std::time::Duration;
```

### Configuration Structures

```rust
/// Delivery options for fine-grained routing control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryOptions {
    /// Guarantee delivery (vs best-effort)
    pub guaranteed_delivery: bool,
    /// Maximum retry attempts
    pub max_retries: u8,
    /// Delivery timeout in nanoseconds
    pub timeout_ns: Option<u64>,
    /// Use hardware acceleration if available
    pub use_hardware_acceleration: bool,
    /// SIMD optimization flags
    pub simd_flags: u32,
}

/// Routing metadata for hardware-accelerated delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetadata {
    pub source_id: ComponentId,
    pub source_task_id: Option<TaskId>,
    /// SIMD-optimized component bitmask
    pub target_component_mask: u64,
    pub priority: Priority,
    pub deadline_ns: Option<NanoTime>,
    pub size_hint: usize,
    pub delivery_options: DeliveryOptions,
}

/// Quantum correlation for temporal optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCorrelation {
    pub quantum_offset: QuantumOffset,
    pub causal_dependencies: Vec<MessageId>,
    pub temporal_phase: f64,
    pub coherence_score: f32,
}
```

---

## Usage Examples

### Complete Application Example

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket, BusConfig};
use csf_core::prelude::*;
use csf_time::{initialize_global_time_source, NanoTime};
use std::sync::Arc;
use tokio;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorReading {
    sensor_id: u32,
    value: f64,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]  
struct ProcessedData {
    source_sensor: u32,
    processed_value: f64,
    processing_time_ns: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize global time source
    initialize_global_time_source()?;
    
    // Create bus with optimal configuration
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig {
        channel_buffer_size: 2048, // Larger buffer for high throughput
    })?);
    
    // Start sensor data producer
    let producer_bus = bus.clone();
    let producer_handle = tokio::spawn(async move {
        let mut counter = 0u32;
        loop {
            let reading = SensorReading {
                sensor_id: 1,
                value: (counter as f64 * 0.1).sin(), // Sine wave
                timestamp: csf_time::now()?.as_nanos(),
            };
            
            let packet = PhasePacket::new(reading, ComponentId::custom(1))
                .with_priority(Priority::High)
                .with_targets(0xFF) // Broadcast to all processors
                .with_timeout(1_000_000); // 1ms timeout
            
            producer_bus.publish(packet).await?;
            
            counter += 1;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            if counter >= 100 { break; }
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });
    
    // Start data processor
    let processor_bus = bus.clone();
    let processor_handle = tokio::spawn(async move {
        let mut sensor_rx = processor_bus.subscribe::<SensorReading>().await?;
        let processing_start = csf_time::now()?;
        
        while let Some(sensor_packet) = sensor_rx.recv().await {
            let process_start = csf_time::now()?;
            
            // Simulate processing
            let processed_value = sensor_packet.payload.value * 2.0 + 1.0;
            
            let process_end = csf_time::now()?;
            let processing_time = process_end.as_nanos() - process_start.as_nanos();
            
            let processed = ProcessedData {
                source_sensor: sensor_packet.payload.sensor_id,
                processed_value,
                processing_time_ns: processing_time,
            };
            
            // Publish processed data with causal dependency
            let processed_packet = PhasePacket::new(processed, ComponentId::custom(2))
                .with_priority(Priority::Normal);
            
            let mut packet_with_deps = processed_packet;
            packet_with_deps.add_temporal_correlation(vec![sensor_packet.id]);
            
            processor_bus.publish(packet_with_deps).await?;
        }
        
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });
    
    // Start results collector  
    let collector_bus = bus.clone();
    let collector_handle = tokio::spawn(async move {
        let mut results_rx = collector_bus.subscribe::<ProcessedData>().await?;
        let mut count = 0;
        
        while let Some(result_packet) = results_rx.recv().await {
            println!(
                "Result {}: sensor={}, value={:.2}, processing_time={}ns",
                count,
                result_packet.payload.source_sensor,
                result_packet.payload.processed_value,
                result_packet.payload.processing_time_ns
            );
            
            count += 1;
            if count >= 100 { break; }
        }
        
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });
    
    // Wait for completion and check performance
    let (producer_result, processor_result, collector_result) = tokio::join!(
        producer_handle,
        processor_handle, 
        collector_handle
    );
    
    producer_result??;
    processor_result??;
    collector_result??;
    
    // Print final statistics
    let stats = bus.get_stats();
    println!("\nFinal Performance Statistics:");
    println!("Messages published: {}", stats.packets_published);
    println!("Messages delivered: {}", stats.packets_delivered);
    println!("Messages dropped: {}", stats.packets_dropped);
    println!("Average latency: {}ns", stats.avg_latency_ns);
    println!("Peak latency: {}ns", stats.peak_latency_ns);
    println!("Throughput: {} msg/sec", stats.throughput_mps);
    
    // Check if performance targets were met
    let health = bus.health_check();
    if health.is_healthy {
        println!("✅ All performance targets met!");
    } else {
        println!("⚠️  Performance warnings:");
        for warning in &health.warnings {
            println!("  - {}", warning);
        }
    }
    
    Ok(())
}
```

### Quantum-Optimized Processing Example

```rust
use csf_time::{QuantumOffset, QuantumTimeOracle};

async fn quantum_optimized_processing(
    bus: &PhaseCoherenceBus,
    oracle: &QuantumTimeOracle,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get quantum optimization hint
    let hint = oracle.get_optimization_hint(csf_time::now()?);
    
    let quantum_offset = match hint {
        OptimizationHint::MinimizeLatency => {
            // High frequency, low amplitude for speed
            QuantumOffset::new(0.3, 0.8, 1200.0)
        }
        OptimizationHint::MaximizeThroughput => {
            // Low frequency, high amplitude for batching
            QuantumOffset::new(0.9, 0.2, 800.0)
        }
        OptimizationHint::Balanced => {
            // Balanced parameters
            QuantumOffset::new(0.6, 0.5, 1000.0)
        }
    };
    
    // Create quantum-optimized packet
    let data = HighPriorityData { 
        content: "time-critical processing".to_string(),
        urgency: 0.95,
    };
    
    let packet = PhasePacket::with_quantum_optimization(data, quantum_offset)
        .with_priority(Priority::Critical)
        .with_guaranteed_delivery(3);
    
    // Publish with deadline scheduling
    let deadline = csf_time::now()? + Duration::from_micros(500); // 500μs deadline
    bus.publish_with_deadline(packet, deadline).await?;
    
    Ok(())
}
```

---

This API reference provides comprehensive coverage of all major interfaces in the ARES ChronoFabric system. For implementation details and advanced usage patterns, refer to the individual crate documentation and the comprehensive system architecture guide.

**Document Information**:
- **Generated**: August 26, 2025  
- **Generator**: Claude Code (Sonnet 4)
- **Project**: ARES ChronoFabric v0.1.0
- **License**: Proprietary - ARES Systems