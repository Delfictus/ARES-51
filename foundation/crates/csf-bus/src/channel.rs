//! Channel implementation for typed message passing

use crate::error::BusError;
use crate::packet::PhasePacket;
use csf_time::{TimeSource, TimeSourceImpl}; // Import TimeSource trait
use parking_lot::RwLock;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::mpsc;

/// A type alias for a vector of senders.
type Senders<T> = Vec<mpsc::Sender<Arc<PhasePacket<T>>>>;

/// A typed channel for phase packets
pub struct Channel<T> {
    /// Active subscriber senders
    receivers: Arc<RwLock<Senders<T>>>,

    /// Channel statistics
    stats: ChannelStats,

    /// Per-subscriber channel capacity
    capacity: usize,
}

struct ChannelStats {
    messages_sent: std::sync::atomic::AtomicU64,
    messages_dropped: std::sync::atomic::AtomicU64,
    subscriber_count: std::sync::atomic::AtomicU32,
}

impl<T: Send + Sync + 'static> Channel<T> {
    /// Create a new channel with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            receivers: Arc::new(RwLock::new(Vec::new())),
            stats: ChannelStats {
                messages_sent: std::sync::atomic::AtomicU64::new(0),
                messages_dropped: std::sync::atomic::AtomicU64::new(0),
                subscriber_count: std::sync::atomic::AtomicU32::new(0),
            },
            capacity,
        }
    }

    /// Send a packet to all subscribers
    pub async fn send(&self, packet: Arc<PhasePacket<T>>) -> Result<(), BusError> {
        use std::sync::atomic::Ordering;

        let receivers = self.receivers.read();

        for sender in receivers.iter() {
            if sender.try_send(packet.clone()).is_err() {
                self.stats.messages_dropped.fetch_add(1, Ordering::Relaxed);
                return Err(BusError::channel_closed(uuid::Uuid::new_v4()));
            }
        }

        self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Subscribe to this channel
    pub async fn subscribe<F>(
        &self,
        handler: F,
    ) -> Result<crate::subscription::SubscriptionHandle, BusError>
    where
        F: Fn(Arc<PhasePacket<dyn Any + Send + Sync>>) -> Result<(), BusError> + Send + 'static,
    {
        use std::sync::atomic::Ordering;

        // Create dedicated channel for this subscriber
        let (tx, mut rx) = mpsc::channel(self.capacity);
        let (erased_tx, mut erased_rx) = mpsc::channel(self.capacity);

        // Add sender to subscriber list
        self.receivers.write().push(tx.clone());
        self.stats.subscriber_count.fetch_add(1, Ordering::Relaxed);

        // Spawn handler task
        let subscription_id = uuid::Uuid::new_v4();
        let created_at_ns: u64 = TimeSourceImpl::new()?.now_ns()?.into(); // Convert NanoTime to u64
        let erased_tx_clone = erased_tx.clone();
        let handle = crate::subscription::SubscriptionHandle::new(
            subscription_id,
            erased_tx, // Use the transformed sender
            created_at_ns,
        );
        let handle_clone1 = handle.clone();
        let handle_clone2 = handle.clone();

        tokio::spawn(async move {
            while handle_clone1.is_active() {
                match rx.recv().await {
                    Some(packet) => {
                        // Since we need to convert Arc<PhasePacket<T>> to Arc<PhasePacket<dyn Any + Send + Sync>>
                        // and we can't clone trait objects, we require T: Clone for this operation
                        match Arc::try_unwrap(packet) {
                            Ok(owned_packet) => {
                                let erased = Arc::new(owned_packet.into_erased());
                                let _ = erased_tx_clone.send(erased).await;
                            }
                            Err(_) => {
                                // Multiple references exist, skip this message to avoid cloning
                                // In production, consider logging this case
                                continue;
                            }
                        }
                    }
                    None => break,
                }
            }
        });

        tokio::spawn(async move {
            while handle_clone2.is_active() {
                match erased_rx.recv().await {
                    Some(packet) => {
                        let _ = handler(packet);
                    }
                    None => break,
                }
            }
        });

        Ok(handle)
    }
}