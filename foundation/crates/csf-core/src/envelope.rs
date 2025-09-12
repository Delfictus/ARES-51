//! A container for transporting data through the system.

use bytes::Bytes;
use uuid::Uuid;

#[cfg(feature = "net")]
use serde::{Deserialize, Serialize};

/// A wrapper for messages, containing the payload and metadata.
///
/// Envelopes are used for all inter-task and inter-node communication,
/// providing a consistent structure for data in motion.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "net", derive(Serialize, Deserialize))]
pub struct Envelope {
    /// A unique identifier for this specific message envelope.
    id: Uuid,
    /// The actual data payload.
    payload: Bytes,
}

impl Envelope {
    /// Creates a new `Envelope` with the given payload.
    ///
    /// A unique ID is automatically generated.
    pub fn new(payload: Bytes) -> Self {
        Self {
            id: Uuid::new_v4(),
            payload,
        }
    }

    /// Creates a new `Envelope` with a specific ID and payload.
    /// This is intended for testing purposes where predictable IDs are required.
    #[cfg(any(test, feature = "proptest"))]
    pub fn new_with_id(id: Uuid, payload: Bytes) -> Self {
        Self { id, payload }
    }

    /// Returns the unique identifier of the envelope.
    pub fn id(&self) -> &Uuid {
        &self.id
    }

    /// Returns a reference to the payload.
    pub fn payload(&self) -> &Bytes {
        &self.payload
    }

    /// Consumes the `Envelope` and returns the payload.
    pub fn into_payload(self) -> Bytes {
        self.payload
    }
}
