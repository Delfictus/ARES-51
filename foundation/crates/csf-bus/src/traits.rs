//! Core traits for the Phase Coherence Bus (PCB) implementation
//!
//! Defines the EventBusTx and EventBusRx traits that provide the foundational
//! interfaces for zero-copy, lock-free message passing with hardware-accelerated routing.

use crate::error::BusError;
use crate::packet::PhasePacket;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for a message on the bus
pub type MessageId = Uuid;

/// Unique identifier for a subscription
pub type SubscriptionId = Uuid;

/// Result type for bus operations
pub type BusResult<T> = Result<T, BusError>;

/// Statistics for bus operation monitoring
#[derive(Debug, Clone, Default)]
pub struct BusStats {
    /// Total number of packets published
    pub packets_published: u64,
    /// Total number of packets delivered to subscribers
    pub packets_delivered: u64,
    /// Total number of packets dropped due to backpressure
    pub packets_dropped: u64,
    /// Current number of active subscriptions
    pub active_subscriptions: u64,
    /// Peak message latency in nanoseconds
    pub peak_latency_ns: u64,
    /// Average message latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Current throughput in messages per second
    pub throughput_mps: u64,
}

/// Event Bus transmitter interface for publishing messages
#[async_trait]
pub trait EventBusTx: Send + Sync {
    /// Publish a single packet asynchronously with guaranteed delivery
    ///
    /// Returns the message ID for tracking and correlation.
    /// Will block if backpressure limits are reached.
    async fn publish<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
    ) -> BusResult<MessageId>;

    /// Try to publish a packet without blocking
    ///
    /// Returns immediately with an error if backpressure limits would be exceeded.
    /// Optimized for hot paths requiring guaranteed low latency.
    fn try_publish<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packet: PhasePacket<T>,
    ) -> BusResult<MessageId>;

    /// Publish multiple packets as a batch for improved throughput
    ///
    /// Optimizes for throughput by processing multiple messages together.
    /// Returns message IDs in the same order as input packets.
    async fn publish_batch<T: Any + Send + Sync + Clone + 'static>(
        &self,
        packets: Vec<PhasePacket<T>>,
    ) -> BusResult<Vec<MessageId>>;

    /// Get current bus statistics for monitoring
    fn get_stats(&self) -> BusStats;

    /// Get the number of active subscribers for a given type
    fn subscriber_count<T: Any + Send + Sync + Clone + 'static>(&self) -> usize;

    /// Check if the bus is healthy and operating within performance targets
    fn is_healthy(&self) -> bool;
}

/// Receiver handle for subscription management
pub struct Receiver<T> {
    /// The actual receiver channel for type-erased packets
    rx: tokio::sync::mpsc::Receiver<Arc<PhasePacket<dyn std::any::Any + Send + Sync>>>,
    /// Subscription ID for management
    subscription_id: SubscriptionId,
    /// Phantom data to maintain type safety
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Receiver<T>
where
    T: std::any::Any + Send + Sync + Clone + 'static,
{
    /// Create a new receiver for type-erased packets
    pub fn new(
        rx: tokio::sync::mpsc::Receiver<Arc<PhasePacket<dyn std::any::Any + Send + Sync>>>,
    ) -> Self {
        Self {
            rx,
            subscription_id: Uuid::new_v4(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the subscription ID
    pub fn subscription_id(&self) -> SubscriptionId {
        self.subscription_id
    }

    /// Receive a message (async) with type casting
    pub async fn recv(&mut self) -> Option<T> {
        if let Some(erased_packet_arc) = self.rx.recv().await {
            // We need to clone the Arc to get ownership, then try downcasting
            // This approach works around the Arc ownership issue
            match Arc::try_unwrap(erased_packet_arc) {
                Ok(erased_packet) => {
                    // Successfully unwrapped - we have ownership
                    if let Ok(typed_payload) = erased_packet.payload.downcast::<T>() {
                        Some(*typed_payload)
                    } else {
                        // Type mismatch - this shouldn't happen in normal operation
                        None
                    }
                }
                Err(arc_packet) => {
                    // Arc is still shared - need to clone the content
                    // This is a fallback path for when Arc has multiple references
                    arc_packet.payload.downcast_ref::<T>().map(|typed_payload| (*typed_payload).clone())
                }
            }
        } else {
            None
        }
    }

    /// Try to receive a message without blocking
    pub fn try_recv(&mut self) -> Result<T, tokio::sync::mpsc::error::TryRecvError> {
        match self.rx.try_recv() {
            Ok(erased_packet_arc) => {
                // Use the same Arc unwrapping approach as in recv()
                match Arc::try_unwrap(erased_packet_arc) {
                    Ok(erased_packet) => {
                        // Successfully unwrapped - we have ownership
                        if let Ok(typed_payload) = erased_packet.payload.downcast::<T>() {
                            Ok(*typed_payload)
                        } else {
                            // Type mismatch - return Empty to indicate no valid message
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
                        }
                    }
                    Err(arc_packet) => {
                        // Arc is still shared - need to clone the content
                        if let Some(typed_payload) = arc_packet.payload.downcast_ref::<T>() {
                            Ok((*typed_payload).clone())
                        } else {
                            // Type mismatch - return Empty to indicate no valid message
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
                        }
                    }
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Get the number of queued messages
    pub fn len(&self) -> usize {
        // This is approximate due to concurrent access
        self.rx.len()
    }

    /// Check if the receiver is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Event Bus receiver interface for subscribing to messages
#[async_trait]
pub trait EventBusRx: Send + Sync {
    /// Subscribe to all packets with payload type T
    ///
    /// Returns a receiver that will get all published messages of type T.
    /// The subscription is active until the receiver is dropped.
    async fn subscribe<T: Any + Send + Sync + Clone + 'static>(&self) -> BusResult<Receiver<T>>;

    /// Subscribe with a filter predicate
    ///
    /// Only messages that pass the filter function will be delivered.
    /// The filter is applied at the subscriber level to avoid unnecessary copying.
    fn subscribe_filtered<T, F>(&self, filter: F) -> BusResult<Receiver<T>>
    where
        T: Any + Send + Sync + Clone + 'static,
        F: Fn(&PhasePacket<T>) -> bool + Send + Sync + 'static;

    /// Unsubscribe from a specific subscription
    ///
    /// Removes the subscription identified by the given ID.
    /// Returns an error if the subscription ID is not found.
    fn unsubscribe<T: Any + Send + Sync + Clone + 'static>(
        &self,
        subscription_id: SubscriptionId,
    ) -> BusResult<()>;

    /// Get all active subscription IDs for monitoring
    fn active_subscriptions(&self) -> Vec<SubscriptionId>;

    /// Get the total number of active subscriptions
    fn subscription_count(&self) -> usize;
}

/// Combined trait for buses that support both publishing and subscribing
pub trait EventBus: EventBusTx + EventBusRx {
    /// Get comprehensive bus health information
    fn health_check(&self) -> BusHealthCheck;
}

/// Detailed health information for the bus
#[derive(Debug, Clone)]
pub struct BusHealthCheck {
    /// Whether the bus is operating normally
    pub is_healthy: bool,
    /// Current performance metrics
    pub stats: BusStats,
    /// Any health warnings or issues
    pub warnings: Vec<String>,
    /// Last health check timestamp (from TimeSource)
    pub last_check_ns: u64,
}

impl BusHealthCheck {
    /// Create a new healthy status
    pub fn healthy(stats: BusStats, timestamp_ns: u64) -> Self {
        Self {
            is_healthy: true,
            stats,
            warnings: Vec::new(),
            last_check_ns: timestamp_ns,
        }
    }

    /// Create an unhealthy status with warnings
    pub fn unhealthy(stats: BusStats, warnings: Vec<String>, timestamp_ns: u64) -> Self {
        Self {
            is_healthy: false,
            stats,
            warnings,
            last_check_ns: timestamp_ns,
        }
    }

    /// Add a warning to the health check
    pub fn add_warning(&mut self, warning: String) {
        let is_empty = warning.is_empty();
        self.warnings.push(warning);
        if !is_empty {
            self.is_healthy = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_stats_default() {
        let stats = BusStats::default();
        assert_eq!(stats.packets_published, 0);
        assert_eq!(stats.packets_delivered, 0);
        assert_eq!(stats.packets_dropped, 0);
    }

    #[test]
    fn test_bus_health_check_healthy() {
        let stats = BusStats::default();
        let health = BusHealthCheck::healthy(stats, 1000);

        assert!(health.is_healthy);
        assert!(health.warnings.is_empty());
        assert_eq!(health.last_check_ns, 1000);
    }

    #[test]
    fn test_bus_health_check_unhealthy() {
        let stats = BusStats::default();
        let warnings = vec!["High latency".to_string()];
        let health = BusHealthCheck::unhealthy(stats, warnings.clone(), 2000);

        assert!(!health.is_healthy);
        assert_eq!(health.warnings, warnings);
        assert_eq!(health.last_check_ns, 2000);
    }

    #[test]
    fn test_receiver_creation() {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        drop(tx); // Close sender

        let receiver: Receiver<Arc<PhasePacket<dyn std::any::Any + Send + Sync>>> =
            Receiver::new(rx);
        assert!(!receiver.subscription_id().is_nil());
    }
}
