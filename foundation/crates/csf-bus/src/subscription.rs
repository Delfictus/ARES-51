//! Subscription management for the Phase Coherence Bus.

use crate::{error::BusError, packet::PhasePacket};
use std::{
    any::Any,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Handle for managing subscription lifecycle and sending messages.
#[derive(Debug, Clone)]
pub struct SubscriptionHandle {
    /// Unique identifier for the subscription.
    pub id: Uuid,
    /// The sender part of the channel.
    pub sender: mpsc::Sender<Arc<PhasePacket<dyn Any + Send + Sync>>>,
    /// Statistics for this subscription.
    pub stats: Arc<SubscriptionStats>,
    /// Timestamp of creation.
    pub created_at_ns: u64,
    /// Whether this subscription is still active.
    active: Arc<AtomicBool>,
}

/// Statistics for a subscription.
#[derive(Debug, Default)]
pub struct SubscriptionStats {
    /// The number of messages received by the subscription.
    pub messages_received: AtomicU64,
    /// The number of messages dropped by the subscription.
    pub messages_dropped: AtomicU64,
    /// The timestamp of the last received message.
    pub last_received_ns: AtomicU64,
}

impl SubscriptionHandle {
    /// Create a new subscription handle.
    pub fn new(
        id: Uuid,
        sender: mpsc::Sender<Arc<PhasePacket<dyn Any + Send + Sync>>>,
        created_at_ns: u64,
    ) -> Self {
        Self {
            id,
            sender,
            stats: Arc::new(SubscriptionStats::default()),
            created_at_ns,
            active: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Send a packet to the subscriber.
    pub async fn send(
        &self,
        packet: Arc<PhasePacket<dyn Any + Send + Sync>>,
    ) -> Result<(), BusError> {
        self.sender
            .send(packet)
            .await
            .map_err(|_| BusError::channel_closed(self.id))
    }

    /// Check if the subscription is still active.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Cancel the subscription.
    pub fn cancel(&self) {
        self.active.store(false, Ordering::Relaxed);
    }
}