//! Phase packet definitions for quantum-temporal communication.

use crate::types::{Phase, PhaseState, Timestamp};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Phase packet for quantum-temporal communication
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhasePacket {
    /// Unique packet identifier
    pub id: Uuid,
    /// Source component identifier
    pub source: Option<crate::types::ComponentId>,
    /// Destination component identifier
    pub destination: Option<crate::types::ComponentId>,
    /// Phase state information
    pub phase_state: PhaseState,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Packet priority
    pub priority: crate::types::Priority,
    /// Creation timestamp
    pub timestamp: Timestamp,
    /// Optional correlation ID
    pub correlation_id: Option<Uuid>,
    /// Packet payload
    pub payload: Vec<u8>,
}

impl PhasePacket {
    /// Create a new phase packet
    pub fn new(phase_state: PhaseState, payload: Vec<u8>) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: None,
            destination: None,
            phase_state,
            sequence: 0,
            priority: crate::types::Priority::Normal,
            timestamp: Timestamp::now(),
            correlation_id: None,
            payload,
        }
    }

    /// Get the age of the packet
    pub fn age(&self) -> std::time::Duration {
        let now = Timestamp::now();
        now.duration_since(self.timestamp)
    }

    /// Check if the packet has expired
    pub fn is_expired(&self, ttl: std::time::Duration) -> bool {
        self.age() > ttl
    }

    /// Get packet size in bytes
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.payload.len()
    }
}

impl Default for PhasePacket {
    fn default() -> Self {
        Self::new(
            PhaseState::now(Phase::zero(), 1.0),
            Vec::new(),
        )
    }
}