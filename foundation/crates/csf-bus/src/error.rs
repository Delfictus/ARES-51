//! Error types for the Phase Coherence Bus (PCB)
//!
//! Provides structured error handling for all bus operations following
//! the Goal 2 specification for production-grade error management.

use thiserror::Error;
use uuid::Uuid;

/// Primary error type for all Phase Coherence Bus operations
#[derive(Error, Debug)]
pub enum BusError {
    /// Subscription failed for a specific type
    #[error("Subscription failed for type {type_name}: {reason}")]
    SubscriptionFailed {
        /// The name of the type that failed to subscribe.
        type_name: String,
        /// The reason for the failure.
        reason: String,
    },

    /// Message publish timeout exceeded
    #[error("Message publish timeout after {timeout_ms}ms")]
    PublishTimeout { 
        /// The timeout in milliseconds.
        timeout_ms: u64 
    },

    /// Hardware acceleration unavailable or failed
    #[error("Hardware acceleration unavailable: {details}")]
    HardwareUnavailable { 
        /// The details of the hardware error.
        details: String 
    },

    /// Backpressure limit reached - channel full
    #[error("Backpressure limit reached for type {type_name} - channel full")]
    BackpressureLimitReached { 
        /// The name of the type that failed to publish.
        type_name: String 
    },

    /// Channel was unexpectedly closed
    #[error("Channel closed for subscription {subscription_id}")]
    ChannelClosed { 
        /// The ID of the subscription that was closed.
        subscription_id: Uuid 
    },

    /// Subscription not found
    #[error("Subscription {subscription_id} not found")]
    SubscriptionNotFound { 
        /// The ID of the subscription that was not found.
        subscription_id: Uuid 
    },

    /// Message serialization failed
    #[error("Message serialization failed: {reason}")]
    SerializationFailed { 
        /// The reason for the failure.
        reason: String 
    },

    /// Message deserialization failed
    #[error("Message deserialization failed: {reason}")]
    DeserializationFailed { 
        /// The reason for the failure.
        reason: String 
    },

    /// Temporal coherence violation detected
    #[error("Temporal coherence violation: {details}")]
    TemporalViolation { 
        /// The details of the temporal violation.
        details: String 
    },

    /// Performance target missed
    #[error("Performance target missed - {metric}: {actual} > {target}")]
    PerformanceViolation {
        /// The name of the metric that was violated.
        metric: String,
        /// The actual value of the metric.
        actual: u64,
        /// The target value of the metric.
        target: u64,
    },

    /// Resource exhaustion (memory, CPU, etc.)
    #[error("Resource exhausted: {resource} - {details}")]
    ResourceExhausted { 
        /// The resource that was exhausted.
        resource: String, 
        /// The details of the resource exhaustion.
        details: String 
    },

    /// Invalid packet or malformed data
    #[error("Invalid packet: {reason}")]
    InvalidPacket { 
        /// The reason for the invalid packet.
        reason: String 
    },

    /// TSC calibration or hardware timing error
    #[error("TSC timing error: {details}")]
    TimingError { 
        /// The details of the timing error.
        details: String 
    },

    /// SIMD optimization failure
    #[error("SIMD optimization failed: {details}")]
    SimdError { 
        /// The details of the SIMD error.
        details: String 
    },

    /// Internal bus error that shouldn't happen in normal operation
    #[error("Internal bus error: {details}")]
    Internal { 
        /// The details of the internal error.
        details: String 
    },

    /// Component initialization failed
    #[error("Failed to initialize {component}: {reason}")]
    InitializationFailed { 
        /// The component that failed to initialize.
        component: String, 
        /// The reason for the failure.
        reason: String 
    },
}

impl BusError {
    /// Create a subscription failed error
    pub fn subscription_failed(type_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::SubscriptionFailed {
            type_name: type_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a publish timeout error
    pub fn publish_timeout(timeout_ms: u64) -> Self {
        Self::PublishTimeout { timeout_ms }
    }

    /// Create a hardware unavailable error
    pub fn hardware_unavailable(details: impl Into<String>) -> Self {
        Self::HardwareUnavailable {
            details: details.into(),
        }
    }

    /// Create a backpressure limit reached error
    pub fn backpressure_limit(type_name: impl Into<String>) -> Self {
        Self::BackpressureLimitReached {
            type_name: type_name.into(),
        }
    }

    /// Create a channel closed error
    pub fn channel_closed(subscription_id: Uuid) -> Self {
        Self::ChannelClosed { subscription_id }
    }

    /// Create a subscription not found error
    pub fn subscription_not_found(subscription_id: Uuid) -> Self {
        Self::SubscriptionNotFound { subscription_id }
    }

    /// Create a serialization failed error
    pub fn serialization_failed(reason: impl Into<String>) -> Self {
        Self::SerializationFailed {
            reason: reason.into(),
        }
    }

    /// Create a deserialization failed error
    pub fn deserialization_failed(reason: impl Into<String>) -> Self {
        Self::DeserializationFailed {
            reason: reason.into(),
        }
    }

    /// Create a temporal violation error
    pub fn temporal_violation(details: impl Into<String>) -> Self {
        Self::TemporalViolation {
            details: details.into(),
        }
    }

    /// Create a performance violation error
    pub fn performance_violation(metric: impl Into<String>, actual: u64, target: u64) -> Self {
        Self::PerformanceViolation {
            metric: metric.into(),
            actual,
            target,
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>, details: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            details: details.into(),
        }
    }

    /// Create an invalid packet error
    pub fn invalid_packet(reason: impl Into<String>) -> Self {
        Self::InvalidPacket {
            reason: reason.into(),
        }
    }

    /// Create a TSC timing error
    pub fn timing_error(details: impl Into<String>) -> Self {
        Self::TimingError {
            details: details.into(),
        }
    }

    /// Create a SIMD error
    pub fn simd_error(details: impl Into<String>) -> Self {
        Self::SimdError {
            details: details.into(),
        }
    }

    /// Create an internal error
    pub fn internal(details: impl Into<String>) -> Self {
        Self::Internal {
            details: details.into(),
        }
    }

    // === TTW Integration Error Constructors ===

    /// Create a causality violation error
    pub fn causality_violation(message_id: impl Into<String>) -> Self {
        Self::TemporalViolation {
            details: format!(
                "Causality violation detected for message: {}",
                message_id.into()
            ),
        }
    }

    /// Create a deadline missed error
    pub fn deadline_missed(_task_id: impl Into<String>) -> Self {
        Self::PerformanceViolation {
            metric: "deadline".to_string(),
            actual: 0, // Would be actual completion time
            target: 0, // Would be deadline
        }
    }

    /// Create a dependencies not satisfied error
    pub fn dependencies_not_satisfied(missing_deps: Vec<String>) -> Self {
        Self::TemporalViolation {
            details: format!("Dependencies not satisfied: {:?}", missing_deps),
        }
    }

    /// Create a queue full error
    pub fn queue_full() -> Self {
        Self::ResourceExhausted {
            resource: "message_queue".to_string(),
            details: "Temporal message queue is full".to_string(),
        }
    }

    /// Create a time source error
    pub fn time_source_error(details: impl Into<String>) -> Self {
        Self::TimingError {
            details: details.into(),
        }
    }
}

impl From<csf_time::TimeError> for BusError {
    fn from(error: csf_time::TimeError) -> Self {
        BusError::InitializationFailed {
            component: "TimeSource".to_string(),
            reason: format!("TimeError: {}", error),
        }
    }
}

/// Result type for bus operations
pub type BusResult<T> = Result<T, BusError>;

/// Legacy error type for backwards compatibility
#[derive(Error, Debug)]
pub enum Error {
    /// Wrapper around the new BusError
    #[error(transparent)]
    Bus(#[from] BusError),

    /// Direct serialization error
    #[error("Serialization error")]
    Serialization(#[from] bincode::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_error_creation() {
        let error = BusError::subscription_failed("TestType", "Channel full");
        match error {
            BusError::SubscriptionFailed { type_name, reason } => {
                assert_eq!(type_name, "TestType");
                assert_eq!(reason, "Channel full");
            }
            _ => {
                assert!(false, "Unexpected error type: {:?}", error);
            }
        }
    }

    #[test]
    fn test_bus_error_timeout() {
        let error = BusError::publish_timeout(1000);
        match error {
            BusError::PublishTimeout { timeout_ms } => {
                assert_eq!(timeout_ms, 1000);
            }
            _ => {
                assert!(false, "Unexpected error type: {:?}", error);
            }
        }
    }

    #[test]
    fn test_performance_violation_error() {
        let error = BusError::performance_violation("latency", 2000, 1000);
        match error {
            BusError::PerformanceViolation {
                metric,
                actual,
                target,
            } => {
                assert_eq!(metric, "latency");
                assert_eq!(actual, 2000);
                assert_eq!(target, 1000);
            }
            _ => {
                assert!(false, "Unexpected error type: {:?}", error);
            }
        }
    }
}