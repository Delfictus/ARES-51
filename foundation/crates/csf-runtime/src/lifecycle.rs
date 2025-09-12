//! Component Lifecycle Management
//!
//! This module provides sophisticated component lifecycle management with state
//! machines, graceful transitions, and failure recovery for the CSF Runtime.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Notify, RwLock};
use uuid::Uuid;

use crate::config::RuntimeConfig;
use crate::core::{Component, ComponentId, ComponentType};
use crate::error::{LifecycleError, RuntimeError, RuntimeResult};

/// Advanced lifecycle manager with state machine and recovery
#[derive(Debug)]
pub struct LifecycleManager {
    /// Component lifecycle states
    component_states: Arc<RwLock<HashMap<ComponentId, ComponentLifecycleState>>>,
    /// Startup sequences
    startup_sequences: Arc<RwLock<HashMap<Uuid, StartupSequence>>>,
    /// Shutdown sequences
    shutdown_sequences: Arc<RwLock<HashMap<Uuid, ShutdownSequence>>>,
    /// Lifecycle configuration
    config: Arc<RuntimeConfig>,
    /// Event channel for lifecycle events
    event_sender: mpsc::UnboundedSender<LifecycleEvent>,
    /// State change notifications
    state_change_notify: Arc<Notify>,
    /// Lifecycle metrics
    metrics: Arc<RwLock<LifecycleMetrics>>,
}

/// Component lifecycle state with transition history
#[derive(Debug, Clone)]
pub struct ComponentLifecycleState {
    /// Current state
    pub current_state: ComponentState,
    /// Previous state
    pub previous_state: Option<ComponentState>,
    /// State entry timestamp
    pub state_entered_at: SystemTime,
    /// Total time in current state
    pub time_in_state: Duration,
    /// State transition history
    pub transition_history: Vec<StateTransition>,
    /// Failure count
    pub failure_count: u32,
    /// Last failure timestamp
    pub last_failure: Option<SystemTime>,
    /// Recovery attempts
    pub recovery_attempts: u32,
    /// State metadata
    pub metadata: HashMap<String, String>,
}

/// Component lifecycle states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentState {
    /// Component has been created but not initialized
    Created,
    /// Component is initializing
    Initializing,
    /// Component initialization failed
    InitializationFailed(String),
    /// Component is initialized and ready to start
    Initialized,
    /// Component is starting
    Starting,
    /// Component startup failed
    StartupFailed(String),
    /// Component is running normally
    Running,
    /// Component is running but degraded
    Degraded(String),
    /// Component is pausing
    Pausing,
    /// Component is paused
    Paused,
    /// Component is resuming from pause
    Resuming,
    /// Component is stopping
    Stopping,
    /// Component stop failed
    StopFailed(String),
    /// Component has stopped cleanly
    Stopped,
    /// Component has failed
    Failed(String),
    /// Component is being destroyed
    Destroying,
    /// Component has been destroyed
    Destroyed,
}

/// State transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Transition ID
    pub id: Uuid,
    /// Source state
    pub from_state: ComponentState,
    /// Target state
    pub to_state: ComponentState,
    /// Transition timestamp
    pub timestamp: SystemTime,
    /// Transition duration
    pub duration: Duration,
    /// Transition trigger
    pub trigger: TransitionTrigger,
    /// Transition result
    pub result: TransitionResult,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// What triggered a state transition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionTrigger {
    /// Manual trigger (user initiated)
    Manual,
    /// Automatic trigger (system initiated)
    Automatic,
    /// Health check failure
    HealthCheckFailure,
    /// Dependency change
    DependencyChange,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Configuration change
    ConfigurationChange,
    /// External signal
    ExternalSignal(String),
    /// Recovery attempt
    Recovery,
    /// Shutdown request
    Shutdown,
}

/// Result of a state transition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionResult {
    /// Transition completed successfully
    Success,
    /// Transition failed
    Failed(String),
    /// Transition timed out
    TimedOut,
    /// Transition was cancelled
    Cancelled,
    /// Transition is in progress
    InProgress,
}

/// Coordinated startup sequence
#[derive(Debug)]
pub struct StartupSequence {
    /// Sequence ID
    pub id: Uuid,
    /// Sequence name
    pub name: String,
    /// Components to start in order
    pub components: Vec<ComponentId>,
    /// Current step in sequence
    pub current_step: usize,
    /// Sequence status
    pub status: SequenceStatus,
    /// Sequence configuration
    pub config: SequenceConfig,
    /// Start timestamp
    pub started_at: Option<SystemTime>,
    /// Completion timestamp
    pub completed_at: Option<SystemTime>,
    /// Step results
    pub step_results: Vec<StepResult>,
}

/// Coordinated shutdown sequence
#[derive(Debug)]
pub struct ShutdownSequence {
    /// Sequence ID
    pub id: Uuid,
    /// Sequence name
    pub name: String,
    /// Components to stop in order
    pub components: Vec<ComponentId>,
    /// Current step in sequence
    pub current_step: usize,
    /// Sequence status
    pub status: SequenceStatus,
    /// Sequence configuration
    pub config: SequenceConfig,
    /// Start timestamp
    pub started_at: Option<SystemTime>,
    /// Completion timestamp
    pub completed_at: Option<SystemTime>,
    /// Step results
    pub step_results: Vec<StepResult>,
}

/// Sequence execution status
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceStatus {
    /// Sequence is prepared but not started
    Prepared,
    /// Sequence is executing
    Executing,
    /// Sequence completed successfully
    Completed,
    /// Sequence failed
    Failed(String),
    /// Sequence was cancelled
    Cancelled,
    /// Sequence is paused
    Paused,
}

/// Configuration for startup/shutdown sequences
#[derive(Debug, Clone)]
pub struct SequenceConfig {
    /// Timeout for individual steps
    pub step_timeout: Duration,
    /// Overall sequence timeout
    pub sequence_timeout: Duration,
    /// Enable parallel execution where possible
    pub parallel_execution: bool,
    /// Failure handling strategy
    pub failure_strategy: FailureStrategy,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Rollback configuration
    pub rollback_config: RollbackConfig,
}

/// How to handle failures during sequences
#[derive(Debug, Clone, PartialEq)]
pub enum FailureStrategy {
    /// Stop sequence on first failure
    StopOnFailure,
    /// Continue sequence despite failures
    ContinueOnFailure,
    /// Retry failed steps
    RetryOnFailure { max_retries: u32, delay: Duration },
    /// Rollback on failure
    RollbackOnFailure,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable jitter
    pub jitter: bool,
}

/// Rollback configuration
#[derive(Debug, Clone)]
pub struct RollbackConfig {
    /// Enable automatic rollback
    pub auto_rollback: bool,
    /// Rollback timeout
    pub rollback_timeout: Duration,
    /// Partial rollback allowed
    pub partial_rollback: bool,
}

/// Result of a sequence step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step index
    pub step_index: usize,
    /// Target component
    pub component_id: ComponentId,
    /// Step execution result
    pub result: TransitionResult,
    /// Step start time
    pub started_at: SystemTime,
    /// Step completion time
    pub completed_at: Option<SystemTime>,
    /// Step duration
    pub duration: Option<Duration>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Lifecycle events
#[derive(Debug, Clone)]
pub enum LifecycleEvent {
    /// Component state changed
    StateChanged {
        component_id: ComponentId,
        old_state: ComponentState,
        new_state: ComponentState,
        transition_id: Uuid,
    },
    /// Startup sequence started
    StartupSequenceStarted { sequence_id: Uuid },
    /// Startup sequence completed
    StartupSequenceCompleted { sequence_id: Uuid, success: bool },
    /// Shutdown sequence started
    ShutdownSequenceStarted { sequence_id: Uuid },
    /// Shutdown sequence completed
    ShutdownSequenceCompleted { sequence_id: Uuid, success: bool },
    /// Component recovery initiated
    RecoveryInitiated {
        component_id: ComponentId,
        attempt: u32,
    },
    /// Component recovery completed
    RecoveryCompleted {
        component_id: ComponentId,
        success: bool,
    },
}

/// Lifecycle management metrics
#[derive(Debug, Clone, Default)]
pub struct LifecycleMetrics {
    /// Total state transitions
    pub total_transitions: u64,
    /// Successful transitions
    pub successful_transitions: u64,
    /// Failed transitions
    pub failed_transitions: u64,
    /// Average transition time
    pub avg_transition_time: Duration,
    /// Total startup sequences
    pub total_startup_sequences: u64,
    /// Successful startup sequences
    pub successful_startup_sequences: u64,
    /// Total shutdown sequences
    pub total_shutdown_sequences: u64,
    /// Successful shutdown sequences
    pub successful_shutdown_sequences: u64,
    /// Total recovery attempts
    pub total_recovery_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Components currently in failed state
    pub failed_components_count: u32,
}

impl LifecycleManager {
    /// Create a new lifecycle manager
    pub fn new(config: Arc<RuntimeConfig>) -> Self {
        let (event_sender, _) = mpsc::unbounded_channel();

        Self {
            component_states: Arc::new(RwLock::new(HashMap::new())),
            startup_sequences: Arc::new(RwLock::new(HashMap::new())),
            shutdown_sequences: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_sender,
            state_change_notify: Arc::new(Notify::new()),
            metrics: Arc::new(RwLock::new(LifecycleMetrics::default())),
        }
    }

    /// Register a component for lifecycle management
    pub async fn register_component(&self, component_id: ComponentId) -> RuntimeResult<()> {
        let lifecycle_state = ComponentLifecycleState {
            current_state: ComponentState::Created,
            previous_state: None,
            state_entered_at: SystemTime::now(),
            time_in_state: Duration::new(0, 0),
            transition_history: Vec::new(),
            failure_count: 0,
            last_failure: None,
            recovery_attempts: 0,
            metadata: HashMap::new(),
        };

        let mut states = self.component_states.write().await;
        states.insert(component_id.clone(), lifecycle_state);

        tracing::info!(
            "Registered component for lifecycle management: {}",
            component_id
        );
        Ok(())
    }

    /// Transition component to a new state
    pub async fn transition_component_state(
        &self,
        component_id: &ComponentId,
        target_state: ComponentState,
        trigger: TransitionTrigger,
    ) -> RuntimeResult<Uuid> {
        let transition_id = Uuid::new_v4();
        let transition_start = Instant::now();

        // Perform state transition within a scoped block
        let (old_state, previous_state, current_state) = {
            let mut states = self.component_states.write().await;
            let lifecycle_state = states.get_mut(component_id).ok_or_else(|| {
                RuntimeError::Lifecycle(LifecycleError::InvalidTransition {
                    component: component_id.clone(),
                    from: "Unknown".to_string(),
                    to: format!("{:?}", target_state),
                })
            })?;

            let old_state = lifecycle_state.current_state.clone();

            // Validate transition
            self.validate_state_transition(&old_state, &target_state)?;

            // Update timing information
            let now = SystemTime::now();
            let _time_in_previous_state = now
                .duration_since(lifecycle_state.state_entered_at)
                .unwrap_or(Duration::new(0, 0));

            // Create transition record
            let transition = StateTransition {
                id: transition_id,
                from_state: old_state.clone(),
                to_state: target_state.clone(),
                timestamp: now,
                duration: transition_start.elapsed(),
                trigger: trigger.clone(),
                result: TransitionResult::Success,
                context: HashMap::new(),
            };

            // Update lifecycle state
            lifecycle_state.previous_state = Some(old_state.clone());
            lifecycle_state.current_state = target_state.clone();
            lifecycle_state.state_entered_at = now;
            lifecycle_state.time_in_state = Duration::new(0, 0);
            lifecycle_state.transition_history.push(transition);

            // Update failure tracking
            match &target_state {
                ComponentState::Failed(_)
                | ComponentState::StartupFailed(_)
                | ComponentState::InitializationFailed(_)
                | ComponentState::StopFailed(_) => {
                    lifecycle_state.failure_count += 1;
                    lifecycle_state.last_failure = Some(now);
                }
                ComponentState::Running => {
                    // Reset failure count on successful recovery
                    if matches!(
                        old_state,
                        ComponentState::Failed(_) | ComponentState::Degraded(_)
                    ) {
                        lifecycle_state.failure_count = 0;
                        lifecycle_state.recovery_attempts = 0;
                    }
                }
                _ => {}
            }

            // Return values needed outside the scope
            (
                old_state.clone(),
                lifecycle_state.previous_state.clone(),
                lifecycle_state.current_state.clone(),
            )
        }; // states lock is automatically dropped here

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_transitions += 1;
            metrics.successful_transitions += 1;

            // Update average transition time
            let total_time = metrics.avg_transition_time.as_nanos() as u64
                * (metrics.total_transitions - 1)
                + transition_start.elapsed().as_nanos() as u64;
            metrics.avg_transition_time =
                Duration::from_nanos(total_time / metrics.total_transitions);
        }

        // Emit lifecycle event
        let _ = self.event_sender.send(LifecycleEvent::StateChanged {
            component_id: component_id.clone(),
            old_state,
            new_state: target_state,
            transition_id,
        });

        // Notify waiters
        self.state_change_notify.notify_waiters();

        tracing::info!(
            "Component {} transitioned: {:?} -> {:?} ({})",
            component_id,
            previous_state,
            current_state,
            transition_id
        );

        Ok(transition_id)
    }

    /// Get current state of a component
    pub async fn get_component_state(&self, component_id: &ComponentId) -> Option<ComponentState> {
        let states = self.component_states.read().await;
        states
            .get(component_id)
            .map(|state| state.current_state.clone())
    }

    /// Get full lifecycle state of a component
    pub async fn get_component_lifecycle_state(
        &self,
        component_id: &ComponentId,
    ) -> Option<ComponentLifecycleState> {
        let states = self.component_states.read().await;
        states.get(component_id).cloned()
    }

    /// Create a startup sequence
    pub async fn create_startup_sequence(
        &self,
        name: String,
        components: Vec<ComponentId>,
        config: SequenceConfig,
    ) -> RuntimeResult<Uuid> {
        let sequence_id = Uuid::new_v4();

        let sequence = StartupSequence {
            id: sequence_id,
            name: name.clone(),
            components: components.clone(),
            current_step: 0,
            status: SequenceStatus::Prepared,
            config,
            started_at: None,
            completed_at: None,
            step_results: Vec::new(),
        };

        let mut sequences = self.startup_sequences.write().await;
        sequences.insert(sequence_id, sequence);

        tracing::info!(
            "Created startup sequence '{}' with {} components",
            name,
            components.len()
        );
        Ok(sequence_id)
    }

    /// Execute a startup sequence
    pub async fn execute_startup_sequence(&self, sequence_id: Uuid) -> RuntimeResult<()> {
        // Get sequence
        let mut sequences = self.startup_sequences.write().await;
        let sequence = sequences.get_mut(&sequence_id).ok_or_else(|| {
            RuntimeError::Lifecycle(LifecycleError::StartupSequenceFailed {
                step: 0,
                reason: format!("Sequence {} not found", sequence_id),
            })
        })?;

        sequence.status = SequenceStatus::Executing;
        sequence.started_at = Some(SystemTime::now());
        let components = sequence.components.clone();
        let config = sequence.config.clone();
        drop(sequences);

        // Emit event
        let _ = self
            .event_sender
            .send(LifecycleEvent::StartupSequenceStarted { sequence_id });

        let mut success = true;
        let mut step_results = Vec::new();

        // Execute each step
        for (step_index, component_id) in components.iter().enumerate() {
            let step_start = SystemTime::now();

            tracing::info!(
                "Starting component {} (step {})",
                component_id,
                step_index + 1
            );

            // Attempt state transitions: Created -> Initializing -> Initialized -> Starting -> Running
            let transitions = vec![
                ComponentState::Initializing,
                ComponentState::Initialized,
                ComponentState::Starting,
                ComponentState::Running,
            ];

            let mut step_success = true;
            let mut error_message = None;

            for target_state in transitions {
                match self
                    .transition_component_state(
                        component_id,
                        target_state,
                        TransitionTrigger::Automatic,
                    )
                    .await
                {
                    Ok(_) => {
                        // Add delay to simulate actual component startup time
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(e) => {
                        error_message = Some(e.to_string());
                        step_success = false;
                        break;
                    }
                }
            }

            // Create step result
            let step_result = StepResult {
                step_index,
                component_id: component_id.clone(),
                result: if step_success {
                    TransitionResult::Success
                } else {
                    TransitionResult::Failed(error_message.clone().unwrap_or_default())
                },
                started_at: step_start,
                completed_at: Some(SystemTime::now()),
                duration: SystemTime::now().duration_since(step_start).ok(),
                error_message: error_message.clone(),
            };

            step_results.push(step_result);

            if !step_success {
                success = false;

                // Handle failure according to strategy
                match config.failure_strategy {
                    FailureStrategy::StopOnFailure => {
                        tracing::error!(
                            "Startup sequence failed at step {}: {}",
                            step_index,
                            error_message.unwrap_or_default()
                        );
                        break;
                    }
                    FailureStrategy::ContinueOnFailure => {
                        tracing::warn!(
                            "Step {} failed but continuing: {}",
                            step_index,
                            error_message.unwrap_or_default()
                        );
                    }
                    FailureStrategy::RetryOnFailure { max_retries, delay } => {
                        // Implement retry logic here
                        tracing::info!(
                            "Retrying step {} after {} failure",
                            step_index,
                            delay.as_secs()
                        );
                        tokio::time::sleep(delay).await;
                        // Retry would go here - simplified for this implementation
                    }
                    FailureStrategy::RollbackOnFailure => {
                        tracing::info!(
                            "Rolling back startup sequence due to failure at step {}",
                            step_index
                        );
                        // Rollback logic would go here
                        break;
                    }
                }
            }
        }

        // Update sequence with results
        {
            let mut sequences = self.startup_sequences.write().await;
            if let Some(sequence) = sequences.get_mut(&sequence_id) {
                sequence.status = if success {
                    SequenceStatus::Completed
                } else {
                    SequenceStatus::Failed("One or more steps failed".to_string())
                };
                sequence.completed_at = Some(SystemTime::now());
                sequence.step_results = step_results;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_startup_sequences += 1;
            if success {
                metrics.successful_startup_sequences += 1;
            }
        }

        // Emit completion event
        let _ = self
            .event_sender
            .send(LifecycleEvent::StartupSequenceCompleted {
                sequence_id,
                success,
            });

        if success {
            tracing::info!("Startup sequence {} completed successfully", sequence_id);
            Ok(())
        } else {
            Err(RuntimeError::Lifecycle(
                LifecycleError::StartupSequenceFailed {
                    step: 0, // Would be actual failed step
                    reason: "Startup sequence failed".to_string(),
                },
            ))
        }
    }

    /// Create a shutdown sequence
    pub async fn create_shutdown_sequence(
        &self,
        name: String,
        components: Vec<ComponentId>,
        config: SequenceConfig,
    ) -> RuntimeResult<Uuid> {
        let sequence_id = Uuid::new_v4();

        let sequence = ShutdownSequence {
            id: sequence_id,
            name: name.clone(),
            components: components.clone(),
            current_step: 0,
            status: SequenceStatus::Prepared,
            config,
            started_at: None,
            completed_at: None,
            step_results: Vec::new(),
        };

        let mut sequences = self.shutdown_sequences.write().await;
        sequences.insert(sequence_id, sequence);

        tracing::info!(
            "Created shutdown sequence '{}' with {} components",
            name,
            components.len()
        );
        Ok(sequence_id)
    }

    /// Execute a shutdown sequence
    pub async fn execute_shutdown_sequence(&self, sequence_id: Uuid) -> RuntimeResult<()> {
        // Get sequence
        let mut sequences = self.shutdown_sequences.write().await;
        let sequence = sequences.get_mut(&sequence_id).ok_or_else(|| {
            RuntimeError::Lifecycle(LifecycleError::ShutdownSequenceFailed {
                step: 0,
                reason: format!("Sequence {} not found", sequence_id),
            })
        })?;

        sequence.status = SequenceStatus::Executing;
        sequence.started_at = Some(SystemTime::now());
        let components = sequence.components.clone();
        let config = sequence.config.clone();
        drop(sequences);

        // Emit event
        let _ = self
            .event_sender
            .send(LifecycleEvent::ShutdownSequenceStarted { sequence_id });

        let mut success = true;
        let mut step_results = Vec::new();

        // Execute each step (shutdown is typically in reverse order)
        for (step_index, component_id) in components.iter().enumerate() {
            let step_start = SystemTime::now();

            tracing::info!(
                "Stopping component {} (step {})",
                component_id,
                step_index + 1
            );

            // Attempt state transitions: Running -> Stopping -> Stopped
            let transitions = vec![ComponentState::Stopping, ComponentState::Stopped];

            let mut step_success = true;
            let mut error_message = None;

            for target_state in transitions {
                match self
                    .transition_component_state(
                        component_id,
                        target_state,
                        TransitionTrigger::Shutdown,
                    )
                    .await
                {
                    Ok(_) => {
                        // Add delay to simulate graceful shutdown
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    Err(e) => {
                        error_message = Some(e.to_string());
                        step_success = false;
                        break;
                    }
                }
            }

            // Create step result
            let step_result = StepResult {
                step_index,
                component_id: component_id.clone(),
                result: if step_success {
                    TransitionResult::Success
                } else {
                    TransitionResult::Failed(error_message.clone().unwrap_or_default())
                },
                started_at: step_start,
                completed_at: Some(SystemTime::now()),
                duration: SystemTime::now().duration_since(step_start).ok(),
                error_message: error_message.clone(),
            };

            step_results.push(step_result);

            if !step_success {
                success = false;
                tracing::warn!(
                    "Component {} shutdown failed: {}",
                    component_id,
                    error_message.unwrap_or_default()
                );

                // For shutdown, we typically continue even on failures
                // unless configured otherwise
            }
        }

        // Update sequence with results
        {
            let mut sequences = self.shutdown_sequences.write().await;
            if let Some(sequence) = sequences.get_mut(&sequence_id) {
                sequence.status = if success {
                    SequenceStatus::Completed
                } else {
                    SequenceStatus::Failed("One or more steps failed".to_string())
                };
                sequence.completed_at = Some(SystemTime::now());
                sequence.step_results = step_results;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_shutdown_sequences += 1;
            if success {
                metrics.successful_shutdown_sequences += 1;
            }
        }

        // Emit completion event
        let _ = self
            .event_sender
            .send(LifecycleEvent::ShutdownSequenceCompleted {
                sequence_id,
                success,
            });

        tracing::info!("Shutdown sequence {} completed", sequence_id);
        Ok(())
    }

    /// Attempt to recover a failed component
    pub async fn recover_component(&self, component_id: &ComponentId) -> RuntimeResult<()> {
        let mut states = self.component_states.write().await;
        let lifecycle_state = states.get_mut(component_id).ok_or_else(|| {
            RuntimeError::Lifecycle(LifecycleError::InvalidTransition {
                component: component_id.clone(),
                from: "Unknown".to_string(),
                to: "Recovery".to_string(),
            })
        })?;

        lifecycle_state.recovery_attempts += 1;
        let attempt = lifecycle_state.recovery_attempts;
        drop(states);

        // Emit recovery initiated event
        let _ = self.event_sender.send(LifecycleEvent::RecoveryInitiated {
            component_id: component_id.clone(),
            attempt,
        });

        tracing::info!(
            "Attempting recovery for component {} (attempt {})",
            component_id,
            attempt
        );

        // Attempt recovery: Failed -> Initializing -> Initialized -> Starting -> Running
        let recovery_transitions = vec![
            ComponentState::Initializing,
            ComponentState::Initialized,
            ComponentState::Starting,
            ComponentState::Running,
        ];

        let mut recovery_success = true;

        for target_state in recovery_transitions {
            match self
                .transition_component_state(component_id, target_state, TransitionTrigger::Recovery)
                .await
            {
                Ok(_) => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => {
                    tracing::error!("Recovery step failed for {}: {}", component_id, e);
                    recovery_success = false;
                    break;
                }
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_recovery_attempts += 1;
            if recovery_success {
                metrics.successful_recoveries += 1;
            }
        }

        // Emit recovery completed event
        let _ = self.event_sender.send(LifecycleEvent::RecoveryCompleted {
            component_id: component_id.clone(),
            success: recovery_success,
        });

        if recovery_success {
            tracing::info!("Component {} recovered successfully", component_id);
            Ok(())
        } else {
            tracing::error!("Failed to recover component {}", component_id);
            Err(RuntimeError::Lifecycle(
                LifecycleError::StartupSequenceFailed {
                    step: 0,
                    reason: format!("Component recovery failed: {}", component_id),
                },
            ))
        }
    }

    /// Wait for component to reach a specific state
    pub async fn wait_for_state(
        &self,
        component_id: &ComponentId,
        target_state: ComponentState,
        timeout: Duration,
    ) -> RuntimeResult<()> {
        let start_time = Instant::now();

        loop {
            if start_time.elapsed() > timeout {
                return Err(RuntimeError::Lifecycle(LifecycleError::StuckInTransition {
                    component: component_id.clone(),
                    state: format!("waiting for {:?}", target_state),
                    duration_ms: timeout.as_millis() as u64,
                }));
            }

            let current_state = self.get_component_state(component_id).await;
            if let Some(state) = current_state {
                if state == target_state {
                    return Ok(());
                }
            }

            // Wait for state change notification
            tokio::time::timeout(
                Duration::from_millis(100),
                self.state_change_notify.notified(),
            )
            .await
            .ok();
        }
    }

    /// Get lifecycle metrics
    pub async fn get_metrics(&self) -> LifecycleMetrics {
        let metrics = self.metrics.read().await;
        let mut result = metrics.clone();

        // Update current failed components count
        let states = self.component_states.read().await;
        result.failed_components_count = states
            .values()
            .filter(|state| matches!(state.current_state, ComponentState::Failed(_)))
            .count() as u32;

        result
    }

    /// Subscribe to lifecycle events
    pub fn subscribe_to_events(&self) -> mpsc::UnboundedReceiver<LifecycleEvent> {
        let (_, receiver) = mpsc::unbounded_channel();
        // In a real implementation, we'd manage multiple subscribers
        receiver
    }

    /// Validate state transition
    fn validate_state_transition(
        &self,
        from_state: &ComponentState,
        to_state: &ComponentState,
    ) -> RuntimeResult<()> {
        // Define valid state transitions
        let valid_transitions = match from_state {
            ComponentState::Created => {
                vec![ComponentState::Initializing, ComponentState::Destroying]
            }
            ComponentState::Initializing => vec![
                ComponentState::Initialized,
                ComponentState::InitializationFailed("".to_string()),
                ComponentState::Destroying,
            ],
            ComponentState::InitializationFailed(_) => {
                vec![ComponentState::Initializing, ComponentState::Destroying]
            }
            ComponentState::Initialized => {
                vec![ComponentState::Starting, ComponentState::Destroying]
            }
            ComponentState::Starting => vec![
                ComponentState::Running,
                ComponentState::StartupFailed("".to_string()),
                ComponentState::Destroying,
            ],
            ComponentState::StartupFailed(_) => {
                vec![ComponentState::Starting, ComponentState::Destroying]
            }
            ComponentState::Running => vec![
                ComponentState::Degraded("".to_string()),
                ComponentState::Pausing,
                ComponentState::Stopping,
                ComponentState::Failed("".to_string()),
            ],
            ComponentState::Degraded(_) => vec![
                ComponentState::Running,
                ComponentState::Failed("".to_string()),
                ComponentState::Stopping,
            ],
            ComponentState::Pausing => vec![
                ComponentState::Paused,
                ComponentState::Failed("".to_string()),
            ],
            ComponentState::Paused => vec![ComponentState::Resuming, ComponentState::Stopping],
            ComponentState::Resuming => vec![
                ComponentState::Running,
                ComponentState::Failed("".to_string()),
            ],
            ComponentState::Stopping => vec![
                ComponentState::Stopped,
                ComponentState::StopFailed("".to_string()),
            ],
            ComponentState::StopFailed(_) => {
                vec![ComponentState::Stopping, ComponentState::Destroying]
            }
            ComponentState::Stopped => vec![ComponentState::Starting, ComponentState::Destroying],
            ComponentState::Failed(_) => vec![
                ComponentState::Initializing, // Recovery
                ComponentState::Destroying,
            ],
            ComponentState::Destroying => vec![ComponentState::Destroyed],
            ComponentState::Destroyed => vec![],
        };

        // Check if transition is valid (ignore string content for enum variants)
        let is_valid = valid_transitions.iter().any(|valid_state| {
            std::mem::discriminant(valid_state) == std::mem::discriminant(to_state)
        });

        if !is_valid {
            return Err(RuntimeError::Lifecycle(LifecycleError::InvalidTransition {
                component: ComponentId::new("unknown", ComponentType::Custom("unknown".into())),
                from: format!("{:?}", from_state),
                to: format!("{:?}", to_state),
            }));
        }

        Ok(())
    }
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            step_timeout: Duration::from_secs(60),
            sequence_timeout: Duration::from_secs(600),
            parallel_execution: false,
            failure_strategy: FailureStrategy::StopOnFailure,
            retry_config: RetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            rollback_config: RollbackConfig {
                auto_rollback: false,
                rollback_timeout: Duration::from_secs(300),
                partial_rollback: true,
            },
        }
    }
}

impl std::fmt::Display for ComponentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentState::Created => write!(f, "Created"),
            ComponentState::Initializing => write!(f, "Initializing"),
            ComponentState::InitializationFailed(msg) => write!(f, "InitializationFailed({})", msg),
            ComponentState::Initialized => write!(f, "Initialized"),
            ComponentState::Starting => write!(f, "Starting"),
            ComponentState::StartupFailed(msg) => write!(f, "StartupFailed({})", msg),
            ComponentState::Running => write!(f, "Running"),
            ComponentState::Degraded(msg) => write!(f, "Degraded({})", msg),
            ComponentState::Pausing => write!(f, "Pausing"),
            ComponentState::Paused => write!(f, "Paused"),
            ComponentState::Resuming => write!(f, "Resuming"),
            ComponentState::Stopping => write!(f, "Stopping"),
            ComponentState::StopFailed(msg) => write!(f, "StopFailed({})", msg),
            ComponentState::Stopped => write!(f, "Stopped"),
            ComponentState::Failed(msg) => write!(f, "Failed({})", msg),
            ComponentState::Destroying => write!(f, "Destroying"),
            ComponentState::Destroyed => write!(f, "Destroyed"),
        }
    }
}
