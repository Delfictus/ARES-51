//! A container for transporting data through the system.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A wrapper for messages, containing the payload and metadata.
///
/// Envelopes are used for all inter-task and inter-node communication,
/// providing a consistent structure for data in motion with enhanced
/// metadata for CSF temporal coherence and routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Envelope {
    /// A unique identifier for this specific message envelope.
    id: Uuid,
    /// The actual data payload.
    payload: Bytes,
    /// Source component identifier.
    source: Option<crate::types::ComponentId>,
    /// Destination component identifier.
    destination: Option<crate::types::ComponentId>,
    /// Timestamp when the envelope was created.
    timestamp: crate::types::Timestamp,
    /// Priority level for routing and processing.
    priority: crate::types::Priority,
    /// Optional correlation ID for request-response patterns.
    correlation_id: Option<Uuid>,
    /// Message type identifier.
    message_type: String,
    /// Additional metadata key-value pairs.
    metadata: std::collections::HashMap<String, String>,
}

impl Envelope {
    /// Creates a new `Envelope` with the given payload.
    ///
    /// A unique ID and timestamp are automatically generated.
    pub fn new(payload: Bytes) -> Self {
        Self {
            id: Uuid::new_v4(),
            payload,
            source: None,
            destination: None,
            timestamp: crate::types::Timestamp::now(),
            priority: crate::types::Priority::Normal,
            correlation_id: None,
            message_type: "unknown".to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Creates a new `Envelope` with a specific message type.
    pub fn new_with_type<T: Into<String>>(payload: Bytes, message_type: T) -> Self {
        let mut envelope = Self::new(payload);
        envelope.message_type = message_type.into();
        envelope
    }

    /// Creates a new `Envelope` with a specific ID and payload.
    /// This is intended for testing purposes where predictable IDs are required.
    #[cfg(any(test, feature = "testing"))]
    pub fn new_with_id(id: Uuid, payload: Bytes) -> Self {
        Self {
            id,
            payload,
            source: None,
            destination: None,
            timestamp: crate::types::Timestamp::now(),
            priority: crate::types::Priority::Normal,
            correlation_id: None,
            message_type: "test".to_string(),
            metadata: std::collections::HashMap::new(),
        }
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

    /// Returns the source component ID.
    pub fn source(&self) -> Option<&crate::types::ComponentId> {
        self.source.as_ref()
    }

    /// Sets the source component ID.
    pub fn set_source(&mut self, source: crate::types::ComponentId) {
        self.source = Some(source);
    }

    /// Returns the destination component ID.
    pub fn destination(&self) -> Option<&crate::types::ComponentId> {
        self.destination.as_ref()
    }

    /// Sets the destination component ID.
    pub fn set_destination(&mut self, destination: crate::types::ComponentId) {
        self.destination = Some(destination);
    }

    /// Returns the envelope timestamp.
    pub fn timestamp(&self) -> crate::types::Timestamp {
        self.timestamp
    }

    /// Returns the envelope priority.
    pub fn priority(&self) -> crate::types::Priority {
        self.priority
    }

    /// Sets the envelope priority.
    pub fn set_priority(&mut self, priority: crate::types::Priority) {
        self.priority = priority;
    }

    /// Returns the correlation ID.
    pub fn correlation_id(&self) -> Option<&Uuid> {
        self.correlation_id.as_ref()
    }

    /// Sets the correlation ID for request-response patterns.
    pub fn set_correlation_id(&mut self, correlation_id: Uuid) {
        self.correlation_id = Some(correlation_id);
    }

    /// Returns the message type.
    pub fn message_type(&self) -> &str {
        &self.message_type
    }

    /// Sets the message type.
    pub fn set_message_type<T: Into<String>>(&mut self, message_type: T) {
        self.message_type = message_type.into();
    }

    /// Returns a reference to the metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, String> {
        &self.metadata
    }

    /// Returns a mutable reference to the metadata.
    pub fn metadata_mut(&mut self) -> &mut std::collections::HashMap<String, String> {
        &mut self.metadata
    }

    /// Adds a metadata key-value pair.
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Gets a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Creates a reply envelope with the same correlation ID.
    pub fn create_reply(&self, payload: Bytes) -> Self {
        let mut reply = Self::new(payload);
        reply.correlation_id = Some(self.id);
        reply.destination = self.source.clone();
        reply.source = self.destination.clone();
        reply.priority = self.priority;
        reply
    }

    /// Creates a forwarded envelope preserving routing information.
    pub fn create_forward(&self, new_destination: crate::types::ComponentId) -> Self {
        let mut forward = self.clone();
        forward.id = Uuid::new_v4();
        forward.destination = Some(new_destination);
        forward.timestamp = crate::types::Timestamp::now();
        forward.add_metadata("forwarded_from".to_string(), self.id.to_string());
        forward
    }

    /// Gets the age of the envelope since creation.
    pub fn age(&self) -> std::time::Duration {
        let now = crate::types::Timestamp::now();
        now.duration_since(self.timestamp)
    }

    /// Checks if the envelope has expired based on a TTL.
    pub fn is_expired(&self, ttl: std::time::Duration) -> bool {
        self.age() > ttl
    }

    /// Gets the size of the envelope in bytes.
    pub fn size_bytes(&self) -> usize {
        self.payload.len() 
            + std::mem::size_of::<Uuid>() * 2 // id + correlation_id
            + self.message_type.len()
            + self.metadata.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>()
            + std::mem::size_of::<crate::types::Timestamp>()
            + std::mem::size_of::<crate::types::Priority>()
    }

    /// Serializes the envelope to bytes.
    pub fn to_bytes(&self) -> Result<Bytes, crate::error::Error> {
        bincode::serialize(self)
            .map(|data| Bytes::from(data))
            .map_err(|e| crate::error::Error::serialization(e.to_string()))
    }

    /// Deserializes an envelope from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, crate::error::Error> {
        bincode::deserialize(data)
            .map_err(|e| crate::error::Error::serialization(e.to_string()))
    }

    /// Builder pattern for creating envelopes.
    pub fn builder(payload: Bytes) -> EnvelopeBuilder {
        EnvelopeBuilder::new(payload)
    }
}

/// Builder for creating envelopes with fluent API.
pub struct EnvelopeBuilder {
    envelope: Envelope,
}

impl EnvelopeBuilder {
    /// Creates a new envelope builder.
    pub fn new(payload: Bytes) -> Self {
        Self {
            envelope: Envelope::new(payload),
        }
    }

    /// Sets the source component.
    pub fn source(mut self, source: crate::types::ComponentId) -> Self {
        self.envelope.set_source(source);
        self
    }

    /// Sets the destination component.
    pub fn destination(mut self, destination: crate::types::ComponentId) -> Self {
        self.envelope.set_destination(destination);
        self
    }

    /// Sets the priority.
    pub fn priority(mut self, priority: crate::types::Priority) -> Self {
        self.envelope.set_priority(priority);
        self
    }

    /// Sets the correlation ID.
    pub fn correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.envelope.set_correlation_id(correlation_id);
        self
    }

    /// Sets the message type.
    pub fn message_type<T: Into<String>>(mut self, message_type: T) -> Self {
        self.envelope.set_message_type(message_type);
        self
    }

    /// Adds metadata.
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.envelope.add_metadata(key, value);
        self
    }

    /// Builds the envelope.
    pub fn build(self) -> Envelope {
        self.envelope
    }
}

impl From<Bytes> for Envelope {
    fn from(payload: Bytes) -> Self {
        Self::new(payload)
    }
}

impl From<Vec<u8>> for Envelope {
    fn from(payload: Vec<u8>) -> Self {
        Self::new(Bytes::from(payload))
    }
}

impl From<String> for Envelope {
    fn from(payload: String) -> Self {
        Self::new(Bytes::from(payload.into_bytes()))
    }
}

impl From<&str> for Envelope {
    fn from(payload: &str) -> Self {
        Self::new(Bytes::from(payload.as_bytes().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_creation() {
        let payload = Bytes::from("test payload");
        let envelope = Envelope::new(payload.clone());
        
        assert_eq!(envelope.payload(), &payload);
        assert_eq!(envelope.priority(), crate::types::Priority::Normal);
        assert!(envelope.source().is_none());
        assert!(envelope.destination().is_none());
    }

    #[test]
    fn test_envelope_builder() {
        let payload = Bytes::from("test payload");
        let source = crate::types::ComponentId::new("source");
        let dest = crate::types::ComponentId::new("dest");
        
        let envelope = Envelope::builder(payload.clone())
            .source(source.clone())
            .destination(dest.clone())
            .priority(crate::types::Priority::High)
            .message_type("test")
            .metadata("key", "value")
            .build();
            
        assert_eq!(envelope.payload(), &payload);
        assert_eq!(envelope.source(), Some(&source));
        assert_eq!(envelope.destination(), Some(&dest));
        assert_eq!(envelope.priority(), crate::types::Priority::High);
        assert_eq!(envelope.message_type(), "test");
        assert_eq!(envelope.get_metadata("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_envelope_reply() {
        let payload = Bytes::from("original");
        let reply_payload = Bytes::from("reply");
        let source = crate::types::ComponentId::new("source");
        let dest = crate::types::ComponentId::new("dest");
        
        let mut original = Envelope::new(payload);
        original.set_source(source.clone());
        original.set_destination(dest.clone());
        
        let reply = original.create_reply(reply_payload.clone());
        
        assert_eq!(reply.payload(), &reply_payload);
        assert_eq!(reply.correlation_id(), Some(original.id()));
        assert_eq!(reply.destination(), Some(&source));
        assert_eq!(reply.source(), Some(&dest));
    }

    #[test]
    fn test_envelope_serialization() {
        let payload = Bytes::from("test payload");
        let envelope = Envelope::new(payload);
        
        let bytes = envelope.to_bytes().unwrap();
        let deserialized = Envelope::from_bytes(&bytes).unwrap();
        
        assert_eq!(envelope.id(), deserialized.id());
        assert_eq!(envelope.payload(), deserialized.payload());
    }

    #[test]
    fn test_envelope_age() {
        let envelope = Envelope::new(Bytes::from("test"));
        let age = envelope.age();
        assert!(age.as_nanos() > 0);
        
        let ttl = std::time::Duration::from_secs(1);
        assert!(!envelope.is_expired(ttl));
    }

    #[test]
    fn test_envelope_size() {
        let payload = Bytes::from("test payload");
        let envelope = Envelope::new(payload.clone());
        let size = envelope.size_bytes();
        assert!(size >= payload.len());
    }

    #[test]
    fn test_envelope_from_conversions() {
        let string_envelope = Envelope::from("test string");
        assert_eq!(string_envelope.payload(), &Bytes::from("test string".as_bytes()));
        
        let vec_envelope = Envelope::from(vec![1, 2, 3, 4]);
        assert_eq!(vec_envelope.payload(), &Bytes::from(vec![1, 2, 3, 4]));
    }
}