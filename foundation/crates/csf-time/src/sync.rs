//! Temporal coherence and synchronization utilities

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;

use crate::{LogicalTime, TimeError, TimeResult};

/// Event in the temporal system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Unique event identifier
    pub id: String,
    /// Logical timestamp when event occurred
    pub timestamp: LogicalTime,
    /// Event payload data
    pub data: Vec<u8>,
}

impl Event {
    /// Create a new event
    pub fn new(id: String, timestamp: LogicalTime, data: Vec<u8>) -> Self {
        Self {
            id,
            timestamp,
            data,
        }
    }

    /// Create event with empty data
    pub fn new_empty(id: String, timestamp: LogicalTime) -> Self {
        Self::new(id, timestamp, Vec::new())
    }
}

/// Temporal coherence validator for distributed systems
pub trait TemporalCoherence: Send + Sync {
    /// Validate temporal coherence of a set of events
    fn validate_coherence(&self, events: &[Event]) -> TimeResult<()>;

    /// Check if two events are causally ordered
    fn are_causally_ordered(&self, event1: &Event, event2: &Event) -> bool;

    /// Find causal dependencies in event set
    fn find_dependencies(&self, events: &[Event]) -> Vec<(String, String)>;

    /// Detect temporal anomalies
    fn detect_anomalies(&self, events: &[Event]) -> Vec<TemporalAnomaly>;
}

/// Types of temporal anomalies that can be detected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalAnomaly {
    /// Causality violation between two events
    CausalityViolation {
        /// The event that should have caused the effect
        cause: String,
        /// The event that happened but violated causality
        effect: String,
    },
    /// Clock skew beyond acceptable bounds
    ClockSkew {
        /// ID of the node with clock skew
        node_id: u64,
        /// Amount of skew in nanoseconds
        skew_ns: i64,
    },
    /// Event ordering inconsistency
    OrderingInconsistency {
        /// First event in the inconsistent ordering
        event1: String,
        /// Second event in the inconsistent ordering
        event2: String,
    },
    /// Missing causal dependency
    MissingDependency {
        /// The event that is missing its dependency
        event: String,
        /// The expected causal predecessor
        expected_cause: String,
    },
}

/// Causality validator implementation  
#[derive(Debug)]
pub struct CausalityValidator {
    /// Maximum allowed clock skew in nanoseconds
    max_clock_skew: i64,
    /// Event history for dependency tracking
    event_history: Arc<RwLock<HashMap<String, Event>>>,
    /// Known causal relationships
    causal_graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl CausalityValidator {
    /// Create a new causality validator
    pub fn new(max_clock_skew: i64) -> Self {
        Self {
            max_clock_skew,
            event_history: Arc::new(RwLock::new(HashMap::new())),
            causal_graph: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for CausalityValidator {
    fn default() -> Self {
        Self::new(1_000_000) // 1ms in nanoseconds
    }
}

impl CausalityValidator {
    /// Add an event to the history
    pub fn add_event(&self, event: Event) {
        self.event_history.write().insert(event.id.clone(), event);
    }

    /// Add a causal dependency
    pub fn add_dependency(&self, cause: String, effect: String) {
        self.causal_graph
            .write()
            .entry(effect)
            .or_default()
            .push(cause);
    }

    /// Check for clock skew between nodes
    fn check_clock_skew(&self, events: &[Event]) -> Vec<TemporalAnomaly> {
        let mut anomalies = Vec::new();
        let mut node_times: HashMap<u64, (u64, u64)> = HashMap::new(); // (min_time, max_time)

        // Collect time ranges for each node
        for event in events {
            let physical_time = event.timestamp.physical;
            let node_id = event.timestamp.node_id;

            node_times
                .entry(node_id)
                .and_modify(|(min, max)| {
                    *min = (*min).min(physical_time);
                    *max = (*max).max(physical_time);
                })
                .or_insert((physical_time, physical_time));
        }

        // Check for excessive skew between nodes
        if node_times.len() > 1 {
            let times: Vec<_> = node_times.values().collect();
            for i in 0..times.len() {
                for j in (i + 1)..times.len() {
                    let skew = (times[i].0 as i64) - (times[j].0 as i64);
                    if skew.abs() > self.max_clock_skew {
                        let node_ids: Vec<_> = node_times.keys().collect();
                        anomalies.push(TemporalAnomaly::ClockSkew {
                            node_id: *node_ids[i],
                            skew_ns: skew,
                        });
                    }
                }
            }
        }

        anomalies
    }

    /// Check for causality violations
    fn check_causality_violations(&self, events: &[Event]) -> Vec<TemporalAnomaly> {
        let mut anomalies = Vec::new();
        let causal_graph = self.causal_graph.read();

        for event in events {
            if let Some(dependencies) = causal_graph.get(&event.id) {
                for dep_id in dependencies {
                    if let Some(dep_event) = events.iter().find(|e| &e.id == dep_id) {
                        // Check if dependency happened before this event
                        if !dep_event.timestamp.happens_before(event.timestamp) {
                            anomalies.push(TemporalAnomaly::CausalityViolation {
                                cause: dep_id.clone(),
                                effect: event.id.clone(),
                            });
                        }
                    } else {
                        // Missing dependency
                        anomalies.push(TemporalAnomaly::MissingDependency {
                            event: event.id.clone(),
                            expected_cause: dep_id.clone(),
                        });
                    }
                }
            }
        }

        anomalies
    }

    /// Check for ordering inconsistencies
    fn check_ordering_consistency(&self, events: &[Event]) -> Vec<TemporalAnomaly> {
        let mut anomalies = Vec::new();

        // Sort events by timestamp
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Check if original order matches sorted order
        for (i, event) in events.iter().enumerate() {
            if event.id != sorted_events[i].id {
                // Find where this event should be
                if let Some(correct_pos) = sorted_events.iter().position(|e| e.id == event.id) {
                    if correct_pos != i {
                        anomalies.push(TemporalAnomaly::OrderingInconsistency {
                            event1: event.id.clone(),
                            event2: sorted_events[i].id.clone(),
                        });
                    }
                }
            }
        }

        anomalies
    }
}

impl TemporalCoherence for CausalityValidator {
    #[instrument(level = "debug")]
    fn validate_coherence(&self, events: &[Event]) -> TimeResult<()> {
        let anomalies = self.detect_anomalies(events);

        if !anomalies.is_empty() {
            let violations: Vec<_> = anomalies
                .iter()
                .filter(|a| matches!(a, TemporalAnomaly::CausalityViolation { .. }))
                .collect();

            if !violations.is_empty() {
                // Use the first violation to create a meaningful error
                return Err(TimeError::CausalityViolation {
                    expected: crate::LogicalTime::new(0, 0, 0), // Placeholder - need proper causality detection
                    actual: crate::LogicalTime::new(0, 0, 0), // Placeholder - need proper causality detection
                });
            }
        }

        Ok(())
    }

    fn are_causally_ordered(&self, event1: &Event, event2: &Event) -> bool {
        event1.timestamp.happens_before(event2.timestamp)
            || event2.timestamp.happens_before(event1.timestamp)
    }

    fn find_dependencies(&self, events: &[Event]) -> Vec<(String, String)> {
        let mut dependencies = Vec::new();
        let causal_graph = self.causal_graph.read();

        for event in events {
            if let Some(deps) = causal_graph.get(&event.id) {
                for dep in deps {
                    dependencies.push((dep.clone(), event.id.clone()));
                }
            }
        }

        dependencies
    }

    fn detect_anomalies(&self, events: &[Event]) -> Vec<TemporalAnomaly> {
        let mut anomalies = Vec::new();

        anomalies.extend(self.check_clock_skew(events));
        anomalies.extend(self.check_causality_violations(events));
        anomalies.extend(self.check_ordering_consistency(events));

        anomalies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LogicalTime;

    #[test]
    fn test_event_creation() {
        let timestamp = LogicalTime::new(1000, 0, 1);
        let event = Event::new("test".to_string(), timestamp, vec![1, 2, 3]);

        assert_eq!(event.id, "test");
        assert_eq!(event.timestamp, timestamp);
        assert_eq!(event.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_causality_validator_creation() {
        let validator = CausalityValidator::new(1_000_000);
        assert_eq!(validator.max_clock_skew, 1_000_000);
    }

    #[test]
    fn test_causal_ordering() {
        let validator = CausalityValidator::default();

        let event1 = Event::new_empty("e1".to_string(), LogicalTime::new(100, 0, 1));
        let event2 = Event::new_empty("e2".to_string(), LogicalTime::new(200, 0, 1));

        assert!(validator.are_causally_ordered(&event1, &event2));
    }

    #[test]
    fn test_coherence_validation_success() {
        let validator = CausalityValidator::default();

        let events = vec![
            Event::new_empty("e1".to_string(), LogicalTime::new(100, 0, 1)),
            Event::new_empty("e2".to_string(), LogicalTime::new(200, 0, 1)),
        ];

        assert!(validator.validate_coherence(&events).is_ok());
    }

    #[test]
    fn test_causality_violation_detection() {
        let validator = CausalityValidator::default();

        // Add a causal dependency: e1 -> e2
        validator.add_dependency("e1".to_string(), "e2".to_string());

        // Create events where e2 happens before e1 (violation)
        let events = vec![
            Event::new_empty("e1".to_string(), LogicalTime::new(200, 0, 1)),
            Event::new_empty("e2".to_string(), LogicalTime::new(100, 0, 1)),
        ];

        let anomalies = validator.detect_anomalies(&events);
        assert!(!anomalies.is_empty());

        assert!(matches!(
            anomalies[0],
            TemporalAnomaly::CausalityViolation { .. }
        ));
    }

    #[test]
    fn test_clock_skew_detection() {
        let validator = CausalityValidator::new(100); // Very small allowed skew

        let events = vec![
            Event::new_empty("e1".to_string(), LogicalTime::new(1000, 0, 1)),
            Event::new_empty("e2".to_string(), LogicalTime::new(2000, 0, 2)), // 1ms skew
        ];

        let anomalies = validator.detect_anomalies(&events);

        // Should detect clock skew
        assert!(anomalies
            .iter()
            .any(|a| matches!(a, TemporalAnomaly::ClockSkew { .. })));
    }

    #[test]
    fn test_dependency_tracking() {
        let validator = CausalityValidator::default();

        validator.add_dependency("e1".to_string(), "e2".to_string());
        validator.add_dependency("e2".to_string(), "e3".to_string());

        let events = vec![
            Event::new_empty("e1".to_string(), LogicalTime::new(100, 0, 1)),
            Event::new_empty("e2".to_string(), LogicalTime::new(200, 0, 1)),
            Event::new_empty("e3".to_string(), LogicalTime::new(300, 0, 1)),
        ];

        let dependencies = validator.find_dependencies(&events);
        assert_eq!(dependencies.len(), 2);
        assert!(dependencies.contains(&("e1".to_string(), "e2".to_string())));
        assert!(dependencies.contains(&("e2".to_string(), "e3".to_string())));
    }
}
