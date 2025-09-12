//! Production-grade Hybrid Logical Clock implementation for causality tracking
//!
//! This module provides the core HLC implementation for the Temporal Task Weaver (TTW),
//! ensuring causality tracking, dependency management, and quantum-optimized temporal ordering.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument, warn};

use crate::{oracle::QuantumTimeOracle, NanoTime, TimeError};

/// Logical timestamp combining physical and logical components for causality tracking
///
/// This represents a point in logical time that respects causality constraints
/// across distributed systems using Hybrid Logical Clock semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LogicalTime {
    /// Physical time component (nanoseconds since epoch)
    pub physical: u64,
    /// Logical counter for causality ordering within the same physical time
    pub logical: u64,
    /// Node identifier for distributed systems disambiguation
    pub node_id: u64,
}

impl LogicalTime {
    /// Create a new logical time
    pub const fn new(physical: u64, logical: u64, node_id: u64) -> Self {
        Self {
            physical,
            logical,
            node_id,
        }
    }

    /// Create from NanoTime with zero logical component
    pub fn from_nano_time(time: NanoTime, node_id: u64) -> Self {
        Self::new(time.as_nanos(), 0, node_id)
    }

    /// Convert to NanoTime (physical component only)
    pub fn to_nano_time(self) -> NanoTime {
        NanoTime::from_nanos(self.physical)
    }

    /// Check if this time happens before another (causality ordering)
    ///
    /// Returns true if this event must have happened before the other event
    /// based on HLC semantics.
    pub fn happens_before(self, other: LogicalTime) -> bool {
        self.physical < other.physical
            || (self.physical == other.physical && self.logical < other.logical)
    }

    /// Check if this time is concurrent with another
    ///
    /// Two events are concurrent if neither happens before the other
    /// and they are from different nodes or have different logical times.
    pub fn is_concurrent_with(self, other: LogicalTime) -> bool {
        !self.happens_before(other)
            && !other.happens_before(self)
            && (self.node_id != other.node_id || self != other)
    }

    /// Maximum of two logical times following HLC semantics
    pub fn max(self, other: LogicalTime) -> LogicalTime {
        if self.happens_before(other) {
            other
        } else if other.happens_before(self) {
            self
        } else {
            // Concurrent events - use the one with higher node_id as tiebreaker
            if self.node_id >= other.node_id {
                self
            } else {
                other
            }
        }
    }

    /// Create a successor logical time
    ///
    /// This creates the next logical time that happens after this one
    /// for the same node.
    pub fn successor(self) -> LogicalTime {
        LogicalTime::new(self.physical, self.logical + 1, self.node_id)
    }

    /// Check if this time could be a valid successor to another
    pub fn is_valid_successor_of(self, predecessor: LogicalTime) -> bool {
        // Must be from the same node and happen after
        self.node_id == predecessor.node_id && predecessor.happens_before(self)
    }

    /// Calculate causal distance between two logical times
    ///
    /// Returns None if the events are concurrent
    pub fn causal_distance(self, other: LogicalTime) -> Option<u64> {
        if self.happens_before(other) {
            if self.physical == other.physical {
                Some(other.logical - self.logical)
            } else {
                Some((other.physical - self.physical) + other.logical)
            }
        } else if other.happens_before(self) {
            if other.physical == self.physical {
                Some(self.logical - other.logical)
            } else {
                Some((self.physical - other.physical) + self.logical)
            }
        } else {
            None // Concurrent events have no causal distance
        }
    }
}

impl std::fmt::Display for LogicalTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}.{}", self.node_id, self.physical, self.logical)
    }
}

/// Zero logical time constant
impl LogicalTime {
    /// Zero logical time for the given node
    pub const fn zero(node_id: u64) -> Self {
        Self::new(0, 0, node_id)
    }
}

/// Causal dependency relationship between events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalDependency {
    /// The event that must happen first
    pub cause: LogicalTime,
    /// The event that depends on the cause
    pub effect: LogicalTime,
    /// Strength of the causal relationship (0.0 to 1.0)
    pub strength: f64,
    /// Type of dependency
    pub dependency_type: DependencyType,
}

/// Types of causal dependencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Direct message passing dependency
    DirectMessage,
    /// Transitive dependency through intermediate events
    Transitive,
    /// Task scheduling dependency
    TaskScheduling,
    /// Resource access dependency
    ResourceAccess,
    /// Quantum optimization dependency
    QuantumOptimization,
}

/// Result of causality checking operations
#[derive(Debug, Clone, PartialEq)]
pub enum CausalityResult {
    /// Event respects causality constraints
    Valid {
        /// Dependencies that were satisfied
        satisfied_dependencies: Vec<CausalDependency>,
    },
    /// Event violates causality constraints
    Violation {
        /// Expected logical time for causality
        expected: LogicalTime,
        /// Actual logical time that caused violation
        actual: LogicalTime,
        /// Dependencies that were violated
        violated_dependencies: Vec<CausalDependency>,
    },
    /// Event is concurrent (no causality constraint)
    Concurrent {
        /// Other concurrent events
        concurrent_events: Vec<LogicalTime>,
    },
    /// Dependencies are missing or incomplete
    IncompleteDependencies {
        /// Missing dependencies that need to be resolved
        missing_dependencies: Vec<CausalDependency>,
    },
}

/// Event for causality tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    /// Logical timestamp of the event
    pub timestamp: LogicalTime,
    /// Event identifier
    pub event_id: String,
    /// Event type for categorization
    pub event_type: String,
    /// Payload data (optional)
    pub data: Option<Vec<u8>>,
    /// Dependencies this event has on other events
    pub dependencies: Vec<LogicalTime>,
}

impl Event {
    /// Create a new event
    pub fn new(timestamp: LogicalTime, event_id: String, event_type: String) -> Self {
        Self {
            timestamp,
            event_id,
            event_type,
            data: None,
            dependencies: Vec::new(),
        }
    }

    /// Create an event with dependencies
    pub fn with_dependencies(
        timestamp: LogicalTime,
        event_id: String,
        event_type: String,
        dependencies: Vec<LogicalTime>,
    ) -> Self {
        Self {
            timestamp,
            event_id,
            event_type,
            data: None,
            dependencies,
        }
    }

    /// Add a dependency to this event
    pub fn add_dependency(&mut self, dependency: LogicalTime) {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
        }
    }

    /// Check if this event depends on another
    pub fn depends_on(&self, other: &LogicalTime) -> bool {
        self.dependencies.contains(other)
            || self
                .dependencies
                .iter()
                .any(|dep| dep.happens_before(*other))
    }
}

/// Production-grade Hybrid Logical Clock trait for causality tracking
///
/// This trait provides advanced causality tracking, dependency management,
/// and quantum-optimized temporal coordination for distributed systems.
pub trait HlcClock: Send + Sync + std::fmt::Debug {
    /// Advance the clock and return new logical time
    ///
    /// # Errors
    /// Returns `TimeError` if the underlying time source fails
    fn tick(&self) -> Result<LogicalTime, TimeError>;

    /// Update clock with remote logical time and return new local time
    ///
    /// This implements the HLC update algorithm for processing messages
    /// from remote nodes while maintaining causality.
    ///
    /// # Errors
    /// Returns `TimeError` if time source fails or causality is violated
    fn update(&self, remote_time: LogicalTime) -> Result<CausalityResult, TimeError>;

    /// Check causality constraints for an event
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if causality validation fails.
    fn validate_causality(&self, event: &Event) -> Result<bool, TimeError>;

    /// Get causal dependencies for a logical time
    ///
    /// Returns all events that must happen before the given time
    /// to maintain causality.
    fn get_causal_dependencies(&self, time: LogicalTime) -> Vec<LogicalTime>;

    /// Get current logical time without advancing
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if the time source fails.
    fn current_time(&self) -> Result<LogicalTime, TimeError>;

    /// Get node identifier
    fn node_id(&self) -> u64;

    /// Reset clock to initial state
    ///
    /// # Errors
    /// Returns `TimeError` if reset fails
    fn reset(&self, initial_time: LogicalTime) -> Result<(), TimeError>;

    /// Add a causal dependency relationship
    ///
    /// This creates a constraint that one event must happen before another
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if adding the dependency fails.
    fn add_causal_dependency(&self, dependency: CausalDependency) -> Result<(), TimeError>;

    /// Remove causal dependencies that are no longer needed
    ///
    /// This helps prevent memory growth by cleaning up old dependencies
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if garbage collection fails.
    fn gc_dependencies(&self, before_time: LogicalTime) -> Result<usize, TimeError>;

    /// Create a checkpoint of the current causal state
    ///
    /// This captures the complete causality state for debugging and replay
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if creating the checkpoint fails.
    fn create_causal_checkpoint(&self) -> Result<CausalCheckpoint, TimeError>;

    /// Restore from a causal checkpoint
    ///
    /// This restores the complete causality state from a checkpoint
    ///
    /// # Errors
    ///
    /// Returns `TimeError` if restoring from the checkpoint fails.
    fn restore_causal_checkpoint(&self, checkpoint: &CausalCheckpoint) -> Result<(), TimeError>;
}

/// Causal checkpoint for deterministic replay
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalCheckpoint {
    /// Current logical time
    pub current_time: LogicalTime,
    /// All active causal dependencies
    pub dependencies: Vec<CausalDependency>,
    /// Recent events for dependency tracking
    pub recent_events: Vec<Event>,
    /// Node identifier
    pub node_id: u64,
    /// Checkpoint timestamp
    pub checkpoint_time: LogicalTime,
    /// Checkpoint unique identifier
    pub checkpoint_id: u64,
}

/// Enterprise distributed coordination state for multi-node determinism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCoordinationState {
    /// Active peer nodes in the distributed system
    pub peer_nodes: HashMap<u64, NodeState>,
    /// Global logical time vector for distributed consensus
    pub global_time_vector: HashMap<u64, LogicalTime>,
    /// Distributed barrier synchronization points
    pub synchronization_barriers: Vec<DistributedBarrier>,
    /// Enterprise determinism epoch for reproducible execution
    pub determinism_epoch: u64,
    /// Last global synchronization timestamp
    pub last_global_sync: LogicalTime,
}

/// State information for a peer node in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    /// Node identifier
    pub node_id: u64,
    /// Last known logical time from this node
    pub last_seen_time: LogicalTime,
    /// Node health status
    pub status: NodeStatus,
    /// Network round-trip time for coordination
    pub rtt_nanos: u64,
    /// Last successful heartbeat timestamp
    pub last_heartbeat: LogicalTime,
}

/// Node health status for distributed coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is actively participating in distributed coordination
    Active,
    /// Node is temporarily unavailable but expected to return
    Degraded,
    /// Node has failed and cannot participate
    Failed,
    /// Node is leaving the cluster gracefully
    Leaving,
}

/// Distributed barrier for enterprise deterministic synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBarrier {
    /// Unique barrier identifier
    pub barrier_id: u64,
    /// Target logical time for barrier synchronization
    pub target_time: LogicalTime,
    /// Nodes that must reach this barrier
    pub required_nodes: Vec<u64>,
    /// Nodes that have reached this barrier
    pub reached_nodes: Vec<u64>,
    /// Barrier creation timestamp
    pub created_at: LogicalTime,
    /// Barrier timeout for enterprise SLA compliance
    pub timeout_ns: u64,
}

/// Production implementation of Hybrid Logical Clock with advanced causality tracking
#[derive(Debug)]
pub struct HlcClockImpl {
    /// Current physical time component (atomic for lock-free access)
    physical_time: AtomicU64,
    /// Logical counter (atomic for lock-free increment)
    logical_counter: AtomicU64,
    /// Node identifier for this clock instance
    node_id: u64,
    /// Time source for physical time
    time_source: Arc<dyn crate::TimeSource>,
    /// Quantum oracle for optimization
    quantum_oracle: Arc<QuantumTimeOracle>,
    /// Causal dependency graph (protected by RwLock)
    dependencies: RwLock<HashMap<LogicalTime, Vec<CausalDependency>>>,
    /// Recent events for dependency tracking (bounded circular buffer)
    recent_events: RwLock<VecDeque<Event>>,
    /// Maximum events to keep in memory
    max_events: usize,
    /// Causality violation count (for monitoring)
    violation_count: AtomicU64,
    /// Checkpoint counter for unique IDs
    checkpoint_counter: AtomicU64,
    /// Enterprise distributed coordination state
    distributed_state: RwLock<DistributedCoordinationState>,
}

impl HlcClockImpl {
    /// Create a new HLC clock with given node ID
    ///
    /// # Errors
    /// Returns `TimeError` if time source initialization fails
    pub fn new(node_id: u64, time_source: Arc<dyn crate::TimeSource>) -> Result<Self, TimeError> {
        let current_time = time_source.now_ns().map(|t| t.as_nanos()).unwrap_or(0); // Safe fallback for initialization

        Ok(Self {
            physical_time: AtomicU64::new(current_time),
            logical_counter: AtomicU64::new(0),
            node_id,
            time_source,
            quantum_oracle: Arc::new(QuantumTimeOracle::new()),
            dependencies: RwLock::new(HashMap::new()),
            recent_events: RwLock::new(VecDeque::new()),
            max_events: 10000, // Configurable limit
            violation_count: AtomicU64::new(0),
            checkpoint_counter: AtomicU64::new(0),
            distributed_state: RwLock::new(DistributedCoordinationState {
                peer_nodes: HashMap::new(),
                global_time_vector: HashMap::new(),
                synchronization_barriers: Vec::new(),
                determinism_epoch: 0,
                last_global_sync: LogicalTime::new(current_time, 0, node_id),
            }),
        })
    }

    /// Create with specific initial time and quantum oracle
    pub fn with_config(
        node_id: u64,
        initial_time: LogicalTime,
        time_source: Arc<dyn crate::TimeSource>,
        quantum_oracle: Arc<QuantumTimeOracle>,
        max_events: usize,
    ) -> Self {
        Self {
            physical_time: AtomicU64::new(initial_time.physical),
            logical_counter: AtomicU64::new(initial_time.logical),
            node_id,
            time_source,
            quantum_oracle,
            dependencies: RwLock::new(HashMap::new()),
            recent_events: RwLock::new(VecDeque::new()),
            max_events,
            violation_count: AtomicU64::new(0),
            checkpoint_counter: AtomicU64::new(0),
            distributed_state: RwLock::new(DistributedCoordinationState {
                peer_nodes: HashMap::new(),
                global_time_vector: HashMap::new(),
                synchronization_barriers: Vec::new(),
                determinism_epoch: 0,
                last_global_sync: initial_time,
            }),
        }
    }

    /// Apply quantum optimization to logical time
    fn apply_quantum_optimization(&self, time: LogicalTime) -> LogicalTime {
        let quantum_offset = self.quantum_oracle.current_offset();
        let optimized_physical = quantum_offset.apply(NanoTime::from_nanos(time.physical));

        LogicalTime::new(optimized_physical.as_nanos(), time.logical, time.node_id)
    }

    /// Ensure monotonic progression with quantum optimization
    fn ensure_monotonic_progression(&self, new_physical: u64) -> (u64, u64) {
        let old_physical = self.physical_time.load(Ordering::Acquire);
        let _old_logical = self.logical_counter.load(Ordering::Acquire);

        if new_physical > old_physical {
            // Physical time advanced - update and reset logical
            self.physical_time.store(new_physical, Ordering::Release);
            self.logical_counter.store(0, Ordering::Release);
            (new_physical, 0)
        } else if new_physical == old_physical {
            // Same physical time - increment logical
            let new_logical = self.logical_counter.fetch_add(1, Ordering::AcqRel) + 1;
            (old_physical, new_logical)
        } else {
            // Physical time went backwards - increment logical counter
            let new_logical = self.logical_counter.fetch_add(1, Ordering::AcqRel) + 1;
            (old_physical, new_logical)
        }
    }

    /// Validate causal dependencies for an event
    fn validate_dependencies(&self, event: &Event) -> Result<CausalityResult, TimeError> {
        let dependencies = self.dependencies.read();
        let mut violated_deps = Vec::new();
        let mut satisfied_deps = Vec::new();
        let mut missing_deps = Vec::new();

        // Check direct dependencies
        for dep_time in &event.dependencies {
            if let Some(causal_deps) = dependencies.get(dep_time) {
                for causal_dep in causal_deps {
                    if causal_dep.effect == event.timestamp {
                        if dep_time.happens_before(event.timestamp) {
                            satisfied_deps.push(causal_dep.clone());
                        } else {
                            violated_deps.push(causal_dep.clone());
                        }
                    }
                }
            } else {
                // Dependency not found - might be missing
                missing_deps.push(CausalDependency {
                    cause: *dep_time,
                    effect: event.timestamp,
                    strength: 1.0,
                    dependency_type: DependencyType::DirectMessage,
                });
            }
        }

        if !violated_deps.is_empty() {
            // Record violation for monitoring
            self.violation_count.fetch_add(1, Ordering::Relaxed);

            warn!(
                node_id = self.node_id,
                event_id = %event.event_id,
                violations = violated_deps.len(),
                "Causality violation detected"
            );

            Ok(CausalityResult::Violation {
                expected: violated_deps[0].cause,
                actual: event.timestamp,
                violated_dependencies: violated_deps,
            })
        } else if !missing_deps.is_empty() {
            debug!(
                node_id = self.node_id,
                event_id = %event.event_id,
                missing = missing_deps.len(),
                "Incomplete dependencies detected"
            );

            Ok(CausalityResult::IncompleteDependencies {
                missing_dependencies: missing_deps,
            })
        } else {
            Ok(CausalityResult::Valid {
                satisfied_dependencies: satisfied_deps,
            })
        }
    }

    /// Add event to recent events buffer
    fn add_recent_event(&self, event: Event) {
        let mut events = self.recent_events.write();

        // Maintain bounded buffer
        if events.len() >= self.max_events {
            events.pop_front(); // Remove oldest event
        }

        events.push_back(event);
    }

    /// Clean up old dependencies to prevent memory growth
    fn cleanup_old_dependencies(&self, before_time: LogicalTime) -> usize {
        let mut dependencies = self.dependencies.write();
        let mut removed_count = 0;

        // Remove dependencies that are older than the specified time
        dependencies.retain(|&time, deps| {
            let should_keep = !time.happens_before(before_time);
            if !should_keep {
                removed_count += deps.len();
            }
            should_keep
        });

        removed_count
    }

    /// Enterprise distributed coordination methods
    /// Register a peer node for distributed synchronization
    pub fn register_peer_node(&self, peer_node_id: u64, initial_time: LogicalTime) -> Result<(), TimeError> {
        let mut state = self.distributed_state.write();
        
        let node_state = NodeState {
            node_id: peer_node_id,
            last_seen_time: initial_time,
            status: NodeStatus::Active,
            rtt_nanos: 0,
            last_heartbeat: initial_time,
        };
        
        state.peer_nodes.insert(peer_node_id, node_state);
        state.global_time_vector.insert(peer_node_id, initial_time);
        
        debug!(
            node_id = self.node_id,
            peer_node_id = peer_node_id,
            initial_time = %initial_time,
            "Registered peer node for distributed coordination"
        );
        
        Ok(())
    }

    /// Update peer node state from received message
    pub fn update_peer_node(&self, peer_node_id: u64, peer_time: LogicalTime) -> Result<(), TimeError> {
        let mut state = self.distributed_state.write();
        
        if let Some(node_state) = state.peer_nodes.get_mut(&peer_node_id) {
            node_state.last_seen_time = peer_time;
            node_state.last_heartbeat = peer_time;
            node_state.status = NodeStatus::Active;
            
            // Update global time vector
            state.global_time_vector.insert(peer_node_id, peer_time);
            
            debug!(
                node_id = self.node_id,
                peer_node_id = peer_node_id,
                peer_time = %peer_time,
                "Updated peer node state"
            );
        } else {
            warn!(
                node_id = self.node_id,
                peer_node_id = peer_node_id,
                "Received update from unregistered peer node"
            );
        }
        
        Ok(())
    }

    /// Create distributed synchronization barrier
    pub fn create_synchronization_barrier(&self, required_nodes: Vec<u64>, timeout_ns: u64) -> Result<u64, TimeError> {
        let current_time = self.current_time()?;
        let barrier_id = self.checkpoint_counter.fetch_add(1, Ordering::AcqRel);
        
        let barrier = DistributedBarrier {
            barrier_id,
            target_time: current_time,
            required_nodes: required_nodes.clone(),
            reached_nodes: vec![self.node_id], // This node reaches immediately
            created_at: current_time,
            timeout_ns,
        };
        
        let mut state = self.distributed_state.write();
        state.synchronization_barriers.push(barrier);
        
        debug!(
            node_id = self.node_id,
            barrier_id = barrier_id,
            required_nodes = ?required_nodes,
            target_time = %current_time,
            "Created distributed synchronization barrier"
        );
        
        Ok(barrier_id)
    }

    /// Signal that this node has reached a synchronization barrier
    pub fn reach_synchronization_barrier(&self, barrier_id: u64) -> Result<bool, TimeError> {
        let mut state = self.distributed_state.write();
        
        if let Some(barrier) = state.synchronization_barriers.iter_mut().find(|b| b.barrier_id == barrier_id) {
            if !barrier.reached_nodes.contains(&self.node_id) {
                barrier.reached_nodes.push(self.node_id);
            }
            
            let all_reached = barrier.required_nodes.iter().all(|&node| barrier.reached_nodes.contains(&node));
            
            debug!(
                node_id = self.node_id,
                barrier_id = barrier_id,
                reached_count = barrier.reached_nodes.len(),
                required_count = barrier.required_nodes.len(),
                all_reached = all_reached,
                "Node reached synchronization barrier"
            );
            
            Ok(all_reached)
        } else {
            Err(TimeError::SystemTimeError {
                details: format!("Synchronization barrier {} not found", barrier_id),
            })
        }
    }

    /// Check if all required nodes have reached the barrier
    pub fn is_barrier_synchronized(&self, barrier_id: u64) -> Result<bool, TimeError> {
        let state = self.distributed_state.read();
        
        if let Some(barrier) = state.synchronization_barriers.iter().find(|b| b.barrier_id == barrier_id) {
            let all_reached = barrier.required_nodes.iter().all(|&node| barrier.reached_nodes.contains(&node));
            Ok(all_reached)
        } else {
            Err(TimeError::SystemTimeError {
                details: format!("Synchronization barrier {} not found", barrier_id),
            })
        }
    }

    /// Perform enterprise global time synchronization across all peers
    pub fn enterprise_global_sync(&self) -> Result<LogicalTime, TimeError> {
        let mut state = self.distributed_state.write();
        
        // Calculate maximum logical time across all known nodes
        let mut max_time = self.current_time()?;
        
        for peer_time in state.global_time_vector.values() {
            max_time = max_time.max(*peer_time);
        }
        
        // Advance determinism epoch
        state.determinism_epoch += 1;
        state.last_global_sync = max_time;
        
        // Update our local clock to the synchronized time
        self.physical_time.store(max_time.physical, Ordering::Release);
        self.logical_counter.store(max_time.logical + 1, Ordering::Release);
        
        let new_sync_time = LogicalTime::new(max_time.physical, max_time.logical + 1, self.node_id);
        
        debug!(
            node_id = self.node_id,
            epoch = state.determinism_epoch,
            sync_time = %new_sync_time,
            peer_count = state.peer_nodes.len(),
            "Performed enterprise global time synchronization"
        );
        
        Ok(new_sync_time)
    }

    /// Get current distributed coordination state snapshot
    pub fn get_distributed_state_snapshot(&self) -> DistributedCoordinationState {
        self.distributed_state.read().clone()
    }

    /// Cleanup expired synchronization barriers
    pub fn cleanup_expired_barriers(&self) -> Result<usize, TimeError> {
        let current_time = self.current_time()?;
        let mut state = self.distributed_state.write();
        
        let original_count = state.synchronization_barriers.len();
        
        state.synchronization_barriers.retain(|barrier| {
            let elapsed_ns = current_time.physical.saturating_sub(barrier.created_at.physical);
            elapsed_ns < barrier.timeout_ns
        });
        
        let removed_count = original_count - state.synchronization_barriers.len();
        
        if removed_count > 0 {
            debug!(
                node_id = self.node_id,
                removed_count = removed_count,
                current_time = %current_time,
                "Cleaned up expired synchronization barriers"
            );
        }
        
        Ok(removed_count)
    }
}

impl HlcClock for HlcClockImpl {
    #[instrument(level = "trace", skip(self))]
    fn tick(&self) -> Result<LogicalTime, TimeError> {
        // Get current physical time with error handling
        let current_physical = self.time_source.now_ns()?;

        // Apply quantum optimization
        let optimized_time = self.apply_quantum_optimization(LogicalTime::from_nano_time(
            current_physical,
            self.node_id,
        ));

        // Ensure monotonic progression
        let (physical, logical) = self.ensure_monotonic_progression(optimized_time.physical);

        let new_time = LogicalTime::new(physical, logical, self.node_id);

        debug!(
            node_id = self.node_id,
            physical = physical,
            logical = logical,
            "HLC tick"
        );

        Ok(new_time)
    }

    #[instrument(level = "trace", skip(self))]
    fn update(&self, remote_time: LogicalTime) -> Result<CausalityResult, TimeError> {
        // Get current physical time
        let current_physical = self.time_source.now_ns()?.as_nanos();
        let old_physical = self.physical_time.load(Ordering::Acquire);
        let old_logical = self.logical_counter.load(Ordering::Acquire);

        // HLC update algorithm with quantum optimization
        let max_physical = current_physical.max(remote_time.physical).max(old_physical);

        let new_logical = if max_physical == old_physical && max_physical == remote_time.physical {
            // Same physical time - take max logical + 1
            old_logical.max(remote_time.logical) + 1
        } else if max_physical == remote_time.physical {
            // Remote time is ahead - use remote logical + 1
            remote_time.logical + 1
        } else {
            // Local time is ahead - increment from current logical
            old_logical + 1
        };

        // Apply quantum optimization
        let optimized_time = self.apply_quantum_optimization(LogicalTime::new(
            max_physical,
            new_logical,
            self.node_id,
        ));

        // Update atomic values with optimized time
        self.physical_time
            .store(optimized_time.physical, Ordering::Release);
        self.logical_counter
            .store(optimized_time.logical, Ordering::Release);

        debug!(
            node_id = self.node_id,
            remote_node = remote_time.node_id,
            local_time = %LogicalTime::new(old_physical, old_logical, self.node_id),
            remote_time = %remote_time,
            new_time = %optimized_time,
            "HLC update"
        );

        // HLC update should always succeed - it's designed to maintain causality
        // The key property is that the new local time happens after both
        // the old local time and the remote time
        Ok(CausalityResult::Valid {
            satisfied_dependencies: vec![],
        })
    }

    #[instrument(level = "debug", skip(self, event))]
    fn validate_causality(&self, event: &Event) -> Result<bool, TimeError> {
        let result = self.validate_dependencies(event)?;

        match result {
            CausalityResult::Valid { .. } => {
                debug!(
                    node_id = self.node_id,
                    event_id = %event.event_id,
                    "Event causality validation passed"
                );

                // Add event to recent events for future dependency tracking
                self.add_recent_event(event.clone());
                Ok(true)
            }
            CausalityResult::Violation {
                violated_dependencies,
                ..
            } => {
                warn!(
                    node_id = self.node_id,
                    event_id = %event.event_id,
                    violations = violated_dependencies.len(),
                    "Event causality validation failed"
                );
                Ok(false)
            }
            CausalityResult::Concurrent { .. } => {
                debug!(
                    node_id = self.node_id,
                    event_id = %event.event_id,
                    "Event is concurrent - no causality constraints"
                );

                self.add_recent_event(event.clone());
                Ok(true)
            }
            CausalityResult::IncompleteDependencies {
                missing_dependencies,
            } => {
                debug!(
                    node_id = self.node_id,
                    event_id = %event.event_id,
                    missing = missing_dependencies.len(),
                    "Event has incomplete dependencies"
                );
                Ok(false)
            }
        }
    }

    fn get_causal_dependencies(&self, time: LogicalTime) -> Vec<LogicalTime> {
        let dependencies = self.dependencies.read();
        let mut deps = Vec::new();

        // Find all dependencies that must happen before this time
        for (&dep_time, causal_deps) in dependencies.iter() {
            for causal_dep in causal_deps {
                if causal_dep.effect == time && dep_time.happens_before(time) {
                    deps.push(dep_time);
                }
            }
        }

        // Also check recent events for transitive dependencies
        let events = self.recent_events.read();
        for event in events.iter() {
            if event.timestamp.happens_before(time) {
                for &event_dep in &event.dependencies {
                    if event_dep.happens_before(time) && !deps.contains(&event_dep) {
                        deps.push(event_dep);
                    }
                }
            }
        }

        deps.sort();
        deps.dedup();
        deps
    }

    fn current_time(&self) -> Result<LogicalTime, TimeError> {
        let physical = self.physical_time.load(Ordering::Acquire);
        let logical = self.logical_counter.load(Ordering::Acquire);
        Ok(LogicalTime::new(physical, logical, self.node_id))
    }

    fn node_id(&self) -> u64 {
        self.node_id
    }

    #[instrument(level = "debug", skip(self))]
    fn reset(&self, initial_time: LogicalTime) -> Result<(), TimeError> {
        debug!(
            node_id = self.node_id,
            initial_time = %initial_time,
            "Resetting HLC clock"
        );

        // Update atomic values
        self.physical_time
            .store(initial_time.physical, Ordering::Release);
        self.logical_counter
            .store(initial_time.logical, Ordering::Release);

        // Clear dependencies and events
        {
            let mut dependencies = self.dependencies.write();
            dependencies.clear();
        }

        {
            let mut events = self.recent_events.write();
            events.clear();
        }

        // Reset violation counter
        self.violation_count.store(0, Ordering::Release);

        Ok(())
    }

    #[instrument(level = "debug", skip(self, dependency))]
    fn add_causal_dependency(&self, dependency: CausalDependency) -> Result<(), TimeError> {
        let mut dependencies = self.dependencies.write();

        dependencies
            .entry(dependency.cause)
            .or_default()
            .push(dependency.clone());

        debug!(
            node_id = self.node_id,
            cause = %dependency.cause,
            effect = %dependency.effect,
            dep_type = ?dependency.dependency_type,
            "Added causal dependency"
        );

        Ok(())
    }

    #[instrument(level = "debug", skip(self))]
    fn gc_dependencies(&self, before_time: LogicalTime) -> Result<usize, TimeError> {
        let removed_count = self.cleanup_old_dependencies(before_time);

        // Also clean up old events
        let mut events = self.recent_events.write();
        let original_len = events.len();
        events.retain(|event| !event.timestamp.happens_before(before_time));
        let events_removed = original_len - events.len();

        debug!(
            node_id = self.node_id,
            before_time = %before_time,
            deps_removed = removed_count,
            events_removed = events_removed,
            "Garbage collected dependencies"
        );

        Ok(removed_count + events_removed)
    }

    #[instrument(level = "debug", skip(self))]
    fn create_causal_checkpoint(&self) -> Result<CausalCheckpoint, TimeError> {
        let current_time = self.current_time()?;
        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::AcqRel);

        // Capture current state
        let dependencies: Vec<CausalDependency> = {
            let deps = self.dependencies.read();
            deps.values().flatten().cloned().collect()
        };

        let recent_events: Vec<Event> = {
            let events = self.recent_events.read();
            events.iter().cloned().collect()
        };

        let checkpoint = CausalCheckpoint {
            current_time,
            dependencies,
            recent_events,
            node_id: self.node_id,
            checkpoint_time: current_time,
            checkpoint_id,
        };

        debug!(
            node_id = self.node_id,
            checkpoint_id = checkpoint_id,
            current_time = %current_time,
            deps_count = checkpoint.dependencies.len(),
            events_count = checkpoint.recent_events.len(),
            "Created causal checkpoint"
        );

        Ok(checkpoint)
    }

    #[instrument(level = "debug", skip(self, checkpoint))]
    fn restore_causal_checkpoint(&self, checkpoint: &CausalCheckpoint) -> Result<(), TimeError> {
        debug!(
            node_id = self.node_id,
            checkpoint_id = checkpoint.checkpoint_id,
            checkpoint_time = %checkpoint.checkpoint_time,
            "Restoring causal checkpoint"
        );

        // Restore clock state
        self.physical_time
            .store(checkpoint.current_time.physical, Ordering::Release);
        self.logical_counter
            .store(checkpoint.current_time.logical, Ordering::Release);

        // Restore dependencies
        {
            let mut dependencies = self.dependencies.write();
            dependencies.clear();

            for dep in &checkpoint.dependencies {
                dependencies.entry(dep.cause).or_default().push(dep.clone());
            }
        }

        // Restore recent events
        {
            let mut events = self.recent_events.write();
            events.clear();
            events.extend(checkpoint.recent_events.iter().cloned());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SimulatedTimeSource;

    fn create_test_time_source() -> Arc<dyn crate::TimeSource> {
        Arc::new(SimulatedTimeSource::new(NanoTime::from_secs(1000)))
    }

    #[test]
    fn test_logical_time_ordering() {
        let time1 = LogicalTime::new(100, 5, 1);
        let time2 = LogicalTime::new(100, 6, 1);
        let time3 = LogicalTime::new(101, 0, 1);

        assert!(time1.happens_before(time2));
        assert!(time2.happens_before(time3));
        assert!(time1.happens_before(time3));

        assert!(!time2.happens_before(time1));
    }

    #[test]
    fn test_logical_time_concurrency() {
        let time1 = LogicalTime::new(100, 5, 1);
        let time2 = LogicalTime::new(100, 5, 2); // Different node, same time

        assert!(time1.is_concurrent_with(time2));
        assert!(time2.is_concurrent_with(time1));
    }

    #[test]
    fn test_causal_distance() {
        let time1 = LogicalTime::new(100, 5, 1);
        let time2 = LogicalTime::new(100, 8, 1);
        let time3 = LogicalTime::new(102, 0, 1);

        assert_eq!(time1.causal_distance(time2), Some(3));
        assert_eq!(time1.causal_distance(time3), Some(2));

        let concurrent1 = LogicalTime::new(100, 5, 1);
        let concurrent2 = LogicalTime::new(100, 5, 2);
        assert_eq!(concurrent1.causal_distance(concurrent2), None);
    }

    #[tokio::test]
    async fn test_hlc_clock_basic_operations() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        // Test tick
        let time1 = clock.tick().expect("Tick should work");
        let time2 = clock.tick().expect("Tick should work");

        assert!(time1.happens_before(time2));
        assert_eq!(time1.node_id, 1);
        assert_eq!(time2.node_id, 1);
    }

    #[tokio::test]
    async fn test_hlc_clock_update() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        // Get initial clock state
        let initial_time = clock.current_time().expect("Current time should work");
        println!("Initial clock time: {}", initial_time);

        // Simulate receiving a message from node 2 with a future time
        let remote_time = LogicalTime::new(1050000000000, 10, 2);
        println!("Remote time: {}", remote_time);

        let result = clock.update(remote_time).expect("Update should work");
        let current = clock.current_time().expect("Current time should work");
        println!("After update - current time: {}", current);
        println!("Update result: {:?}", result);

        // Basic HLC property: the updated local time should advance
        assert!(initial_time.happens_before(current) || initial_time == current);

        // The new local time should incorporate the remote time information
        // This means it should be at least as recent as the remote time
        assert!(current.physical >= remote_time.physical);

        match result {
            CausalityResult::Valid { .. } => {
                // Valid update - this is expected for HLC
            }
            CausalityResult::Concurrent { .. } => {
                // Concurrent is also acceptable for HLC updates
            }
            _ => panic!(
                "Expected valid or concurrent causality result, got: {:?}",
                result
            ),
        }
    }

    #[tokio::test]
    async fn test_causal_dependency_tracking() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        let time1 = LogicalTime::new(1000000000000, 1, 1);
        let time2 = LogicalTime::new(1000000000000, 2, 1);

        let dependency = CausalDependency {
            cause: time1,
            effect: time2,
            strength: 1.0,
            dependency_type: DependencyType::DirectMessage,
        };

        clock
            .add_causal_dependency(dependency)
            .expect("Add dependency should work");

        let dependencies = clock.get_causal_dependencies(time2);
        assert!(dependencies.contains(&time1));
    }

    #[tokio::test]
    async fn test_event_causality_validation() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        let time1 = LogicalTime::new(1000000000000, 1, 1);
        let time2 = LogicalTime::new(1000000000000, 2, 1);

        // Create event with dependency
        let event = Event::with_dependencies(
            time2,
            "test_event".to_string(),
            "test".to_string(),
            vec![time1],
        );

        // Add the dependency to the clock
        let dependency = CausalDependency {
            cause: time1,
            effect: time2,
            strength: 1.0,
            dependency_type: DependencyType::DirectMessage,
        };
        clock
            .add_causal_dependency(dependency)
            .expect("Add dependency should work");

        // Validate causality
        let is_valid = clock
            .validate_causality(&event)
            .expect("Validation should work");
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_causal_checkpoint_restore() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        // Advance clock and add dependencies
        let _time1 = clock.tick().expect("Tick should work");
        let time2 = clock.tick().expect("Tick should work");

        let dependency = CausalDependency {
            cause: LogicalTime::new(1000000000000, 1, 1),
            effect: time2,
            strength: 1.0,
            dependency_type: DependencyType::TaskScheduling,
        };
        clock
            .add_causal_dependency(dependency)
            .expect("Add dependency should work");

        // Create checkpoint
        let checkpoint = clock
            .create_causal_checkpoint()
            .expect("Checkpoint should work");

        // Advance clock more
        let _time3 = clock.tick().expect("Tick should work");

        // Restore checkpoint
        clock
            .restore_causal_checkpoint(&checkpoint)
            .expect("Restore should work");

        // Verify state was restored
        let current = clock.current_time().expect("Current time should work");
        assert_eq!(current, checkpoint.current_time);
    }

    #[tokio::test]
    async fn test_dependency_garbage_collection() {
        let time_source = create_test_time_source();
        let clock = HlcClockImpl::new(1, time_source).expect("Clock creation should work");

        // Add several dependencies
        for i in 0..10 {
            let dependency = CausalDependency {
                cause: LogicalTime::new(1000000000000 + i * 1000, 0, 1),
                effect: LogicalTime::new(1000000000000 + i * 1000 + 500, 0, 1),
                strength: 1.0,
                dependency_type: DependencyType::DirectMessage,
            };
            clock
                .add_causal_dependency(dependency)
                .expect("Add dependency should work");
        }

        // Garbage collect dependencies before a certain time
        let gc_time = LogicalTime::new(1000000000000 + 5000, 0, 1);
        let removed = clock.gc_dependencies(gc_time).expect("GC should work");

        assert!(removed > 0);
    }
}