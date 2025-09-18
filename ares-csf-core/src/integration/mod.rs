//! Integration framework for CSF components.

pub mod monitoring;
pub mod runtime;
pub mod testing;

// Re-export key types
pub use monitoring::{MonitorConfig, SystemHealthStatus, HealthLevel};
pub use runtime::{DrppRuntime, DrppConfig, RuntimeEvent, RuntimeStats};
pub use testing::{DrppTestScenario, TestResults};

use crate::error::{Error, Result};
use crate::types::{ComponentId, Timestamp, Priority};
use serde::{Deserialize, Serialize};

/// Component state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentState {
    /// Component identifier
    pub id: ComponentId,
    /// Current status
    pub status: ComponentStatus,
    /// Health level
    pub health: HealthLevel,
    /// Last update timestamp
    pub last_update: Timestamp,
    /// Performance metrics
    pub metrics: ComponentMetrics,
}

/// Component status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentStatus {
    /// Component is initializing
    Initializing,
    /// Component is running normally
    Running,
    /// Component is degraded but functional
    Degraded,
    /// Component has failed
    Failed,
    /// Component is shutting down
    Shutdown,
}

/// Component performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network I/O bytes per second
    pub network_io_bps: u64,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Response time in milliseconds
    pub response_time_ms: f64,
}

impl Default for ComponentMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            network_io_bps: 0,
            error_rate: 0.0,
            response_time_ms: 0.0,
        }
    }
}

/// Dashboard state for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Overall system health
    pub system_health: HealthLevel,
    /// Active components
    pub components: Vec<ComponentState>,
    /// Current alerts
    pub alerts: Vec<Alert>,
    /// System-wide metrics
    pub system_metrics: SystemMetrics,
    /// Last update timestamp
    pub last_update: Timestamp,
}

/// System alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Component that triggered the alert
    pub component_id: Option<ComponentId>,
    /// Timestamp when alert was created
    pub timestamp: Timestamp,
    /// Whether alert has been acknowledged
    pub acknowledged: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert requiring immediate attention
    Critical,
}

/// System-wide metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Total CPU usage across all components
    pub total_cpu_usage: f64,
    /// Total memory usage in bytes
    pub total_memory_usage: u64,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Total number of active connections
    pub active_connections: u32,
    /// Message throughput (messages per second)
    pub message_throughput: f64,
}

/// Emergent pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern type
    pub pattern_type: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern parameters
    pub parameters: std::collections::HashMap<String, f64>,
    /// Detection timestamp
    pub detected_at: Timestamp,
    /// Components involved in the pattern
    pub involved_components: Vec<ComponentId>,
}

impl EmergentPattern {
    /// Create a new emergent pattern
    pub fn new(id: String, pattern_type: String, confidence: f64) -> Self {
        Self {
            id,
            pattern_type,
            confidence,
            parameters: std::collections::HashMap::new(),
            detected_at: Timestamp::now(),
            involved_components: Vec::new(),
        }
    }

    /// Add a parameter to the pattern
    pub fn add_parameter(&mut self, key: String, value: f64) {
        self.parameters.insert(key, value);
    }

    /// Add a component to the pattern
    pub fn add_component(&mut self, component_id: ComponentId) {
        self.involved_components.push(component_id);
    }
}

/// Phase transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionEvent {
    /// Event ID
    pub id: String,
    /// Source phase
    pub from_phase: crate::types::Phase,
    /// Target phase
    pub to_phase: crate::types::Phase,
    /// Transition probability
    pub probability: f64,
    /// Energy change
    pub energy_delta: f64,
    /// Event timestamp
    pub timestamp: Timestamp,
    /// Component that triggered the transition
    pub component_id: ComponentId,
}

impl PhaseTransitionEvent {
    /// Create a new phase transition event
    pub fn new(
        from_phase: crate::types::Phase,
        to_phase: crate::types::Phase,
        component_id: ComponentId,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from_phase,
            to_phase,
            probability: 1.0,
            energy_delta: 0.0,
            timestamp: Timestamp::now(),
            component_id,
        }
    }

    /// Calculate phase difference
    pub fn phase_difference(&self) -> f64 {
        self.from_phase.difference(&self.to_phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_state() {
        let state = ComponentState {
            id: ComponentId::new("test-component"),
            status: ComponentStatus::Running,
            health: HealthLevel::Healthy,
            last_update: Timestamp::now(),
            metrics: ComponentMetrics::default(),
        };

        assert_eq!(state.status, ComponentStatus::Running);
        assert_eq!(state.health, HealthLevel::Healthy);
    }

    #[test]
    fn test_emergent_pattern() {
        let mut pattern = EmergentPattern::new(
            "pattern-1".to_string(),
            "synchronization".to_string(),
            0.95,
        );

        pattern.add_parameter("frequency".to_string(), 2.5);
        pattern.add_component(ComponentId::new("comp-1"));

        assert_eq!(pattern.confidence, 0.95);
        assert_eq!(pattern.parameters.len(), 1);
        assert_eq!(pattern.involved_components.len(), 1);
    }

    #[test]
    fn test_phase_transition_event() {
        let from_phase = crate::types::Phase::new(0.0);
        let to_phase = crate::types::Phase::new(std::f64::consts::PI);
        let component_id = ComponentId::new("test-component");

        let event = PhaseTransitionEvent::new(from_phase, to_phase, component_id);

        assert!((event.phase_difference() - std::f64::consts::PI).abs() < 1e-10);
    }
}