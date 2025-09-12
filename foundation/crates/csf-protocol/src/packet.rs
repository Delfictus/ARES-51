//! Core packet definitions and structures

use crate::flags::PacketFlags;
use csf_shared_types::{NanoTime, PacketType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique packet identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PacketId(Uuid);

impl PacketId {
    /// Create a new random packet ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a packet ID from a UUID
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the inner UUID
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Get UUID as u128 for hashing
    #[must_use]
    pub fn as_u128(&self) -> u128 {
        self.0.as_u128()
    }
}

impl Default for PacketId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PacketId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Packet header containing routing and metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketHeader {
    /// Protocol version for forward compatibility
    pub version: u8,

    /// Unique packet identifier
    pub packet_id: PacketId,

    /// Type of packet (Control, Data, Event, Stream)
    pub packet_type: PacketType,

    /// Packet processing and state flags
    pub flags: PacketFlags,

    /// Priority level (0-255, higher = more urgent)
    pub priority: u8,

    /// Timestamp when packet was created
    pub timestamp: NanoTime,

    /// Source component/node identifier
    pub source_node: u16,

    /// Destination component/node identifier (`u16::MAX` = broadcast)
    pub destination_node: u16,

    /// Hash for causality tracking and ordering
    pub causality_hash: u64,

    /// Sequence number for fragmented packets
    pub sequence_number: Option<u32>,

    /// Legacy sequence field for compatibility
    pub sequence: u32,

    /// Total number of fragments (for fragmented packets)
    pub fragment_count: Option<u32>,

    /// Packet size in bytes (for validation)
    pub payload_size: u32,

    /// Checksum for integrity verification
    pub checksum: u32,
}

impl PacketHeader {
    /// Create a new packet header with sensible defaults
    #[must_use]
    pub fn new(packet_type: PacketType, source_node: u16, destination_node: u16) -> Self {
        Self {
            version: crate::PROTOCOL_VERSION,
            packet_id: PacketId::new(),
            packet_type,
            flags: PacketFlags::empty(),
            priority: 128, // Normal priority
            timestamp: NanoTime::now(),
            source_node,
            destination_node,
            causality_hash: 0,
            sequence_number: None,
            sequence: 0,
            fragment_count: None,
            payload_size: 0,
            checksum: 0,
        }
    }

    /// Create a new packet header with enterprise `TimeSource` (temporal violation compliant)
    /// 
    /// For higher-level protocols that need deterministic timing, this method allows
    /// injection of a timestamp while maintaining backward compatibility.
    #[must_use]
    pub fn new_with_timestamp(packet_type: PacketType, source_node: u16, destination_node: u16, timestamp: NanoTime) -> Self {
        Self {
            version: crate::PROTOCOL_VERSION,
            packet_id: PacketId::new(),
            packet_type,
            flags: PacketFlags::empty(),
            priority: 128, // Normal priority
            timestamp,
            source_node,
            destination_node,
            causality_hash: 0,
            sequence_number: None,
            sequence: 0,
            fragment_count: None,
            payload_size: 0,
            checksum: 0,
        }
    }

    /// Check if this is a broadcast packet
    #[must_use]
    pub fn is_broadcast(&self) -> bool {
        self.destination_node == u16::MAX
    }

    /// Check if this is a fragmented packet
    #[must_use]
    pub fn is_fragmented(&self) -> bool {
        self.flags.contains(PacketFlags::FRAGMENTED)
    }

    /// Check if this is the last fragment
    #[must_use]
    pub fn is_last_fragment(&self) -> bool {
        self.flags.contains(PacketFlags::LAST_FRAGMENT)
    }

    /// Get effective priority (combines priority field and flags)
    #[must_use]
    pub fn effective_priority(&self) -> u8 {
        let flag_priority = self.flags.priority_level() * 64; // Scale flag priority
        self.priority.saturating_add(flag_priority)
    }

    /// Calculate checksum for the header
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn calculate_checksum(&self) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.version.hash(&mut hasher);
        self.packet_id.hash(&mut hasher);
        self.packet_type.hash(&mut hasher);
        self.flags.hash(&mut hasher);
        self.priority.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        self.source_node.hash(&mut hasher);
        self.destination_node.hash(&mut hasher);
        self.causality_hash.hash(&mut hasher);
        self.sequence_number.hash(&mut hasher);
        self.sequence.hash(&mut hasher);
        self.fragment_count.hash(&mut hasher);
        self.payload_size.hash(&mut hasher);

        hasher.finish() as u32
    }

    /// Validate header consistency
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the header is inconsistent.
    pub fn validate(&self) -> Result<(), crate::ValidationError> {
        if self.version > crate::PROTOCOL_VERSION {
            return Err(crate::ValidationError::UnsupportedVersion(self.version));
        }

        if self.is_fragmented() {
            if self.sequence_number.is_none() {
                return Err(crate::ValidationError::MissingSequenceNumber);
            }
            if self.fragment_count.is_none() {
                return Err(crate::ValidationError::MissingFragmentCount);
            }
        }

        if self.payload_size as usize > crate::MAX_PACKET_SIZE {
            return Err(crate::ValidationError::PayloadTooLarge {
                size: self.payload_size as usize,
                max: crate::MAX_PACKET_SIZE,
            });
        }

        Ok(())
    }
}

/// Packet payload containing data and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketPayload {
    /// The actual data payload
    pub data: Vec<u8>,

    /// Metadata key-value pairs
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PacketPayload {
    /// Create an empty payload
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create payload with data
    #[must_use]
    pub fn with_data(data: Vec<u8>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata entry
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get total payload size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len() + self.metadata_size()
    }

    /// Get estimated metadata size
    fn metadata_size(&self) -> usize {
        self.metadata
            .iter()
            .map(|(k, v)| k.len() + v.to_string().len())
            .sum()
    }

    /// Validate payload constraints
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the payload is invalid.
    pub fn validate(&self) -> Result<(), crate::ValidationError> {
        if self.metadata.len() > crate::MAX_METADATA_ENTRIES {
            return Err(crate::ValidationError::TooManyMetadataEntries {
                size: self.metadata.len(),
                max: crate::MAX_METADATA_ENTRIES,
            });
        }

        if self.size() > crate::MAX_PACKET_SIZE {
            return Err(crate::ValidationError::PayloadTooLarge {
                size: self.size(),
                max: crate::MAX_PACKET_SIZE,
            });
        }

        Ok(())
    }
}

/// Generic phase packet with type-safe payloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePacket<T = PacketPayload> {
    /// Packet header with routing and metadata
    pub header: PacketHeader,

    /// Type-safe payload
    pub payload: T,
}

impl<T> PhasePacket<T> {
    /// Create a new phase packet
    #[must_use]
    pub fn new(
        packet_type: PacketType,
        source_node: u16,
        destination_node: u16,
        payload: T,
    ) -> Self {
        Self {
            header: PacketHeader::new(packet_type, source_node, destination_node),
            payload,
        }
    }

    /// Set packet flags
    #[must_use]
    pub fn with_flags(mut self, flags: PacketFlags) -> Self {
        self.header.flags = flags;
        self
    }

    /// Set packet priority
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.header.priority = priority;
        self
    }

    /// Set causality hash
    #[must_use]
    pub fn with_causality_hash(mut self, hash: u64) -> Self {
        self.header.causality_hash = hash;
        self
    }

    /// Add a flag to the packet
    #[must_use]
    pub fn add_flag(mut self, flag: PacketFlags) -> Self {
        self.header.flags |= flag;
        self
    }

    /// Remove a flag from the packet
    #[must_use]
    pub fn remove_flag(mut self, flag: PacketFlags) -> Self {
        self.header.flags &= !flag;
        self
    }

    /// Check if packet has a specific flag
    #[must_use]
    pub fn has_flag(&self, flag: PacketFlags) -> bool {
        self.header.flags.contains(flag)
    }

    /// Transform the payload type
    #[must_use]
    pub fn map_payload<U, F>(self, f: F) -> PhasePacket<U>
    where
        F: FnOnce(T) -> U,
    {
        PhasePacket {
            header: self.header,
            payload: f(self.payload),
        }
    }
}

impl<T> PhasePacket<T>
where
    T: serde::Serialize,
{
    /// Calculate and update payload size in header
    ///
    /// # Errors
    ///
    /// Returns `ProtocolError` if serialization fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn update_payload_size(&mut self) -> Result<(), crate::ProtocolError> {
        let serialized = bincode::serialize(&self.payload)
            .map_err(|e| crate::PacketEncodeError::SerializationFailed(e.to_string()))?;
        self.header.payload_size = serialized.len() as u32;
        Ok(())
    }

    /// Calculate and update header checksum
    pub fn update_checksum(&mut self) {
        self.header.checksum = self.header.calculate_checksum();
    }

    /// Finalize packet (update size and checksum)
    ///
    /// # Errors
    ///
    /// Returns `ProtocolError` if serialization fails.
    pub fn finalize(mut self) -> Result<Self, crate::ProtocolError> {
        self.update_payload_size()?;
        self.update_checksum();
        Ok(self)
    }
}

// Convenience type aliases
/// Standard phase packet with binary payload
pub type BinaryPacket = PhasePacket<PacketPayload>;

/// Text packet for string payloads
pub type TextPacket = PhasePacket<String>;

/// JSON packet for structured data
pub type JsonPacket = PhasePacket<serde_json::Value>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_id_generation() {
        let id1 = PacketId::new();
        let id2 = PacketId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_header_creation() {
        let header = PacketHeader::new(PacketType::Data, 1, 2);
        assert_eq!(header.packet_type, PacketType::Data);
        assert_eq!(header.source_node, 1);
        assert_eq!(header.destination_node, 2);
        assert_eq!(header.version, crate::PROTOCOL_VERSION);
    }

    #[test]
    fn test_broadcast_detection() {
        let header = PacketHeader::new(PacketType::Control, 1, u16::MAX);
        assert!(header.is_broadcast());
    }

    #[test]
    fn test_payload_creation() {
        let payload = PacketPayload::with_data(b"test".to_vec())
            .with_metadata("key", serde_json::json!("value"));

        assert_eq!(payload.data, b"test");
        assert!(payload.metadata.contains_key("key"));
    }

    #[test]
    fn test_phase_packet_creation() {
        let packet = PhasePacket::new(
            PacketType::Event,
            10,
            20,
            PacketPayload::with_data(b"event data".to_vec()),
        )
        .with_priority(200)
        .add_flag(PacketFlags::HIGH_PRIORITY);

        assert_eq!(packet.header.packet_type, PacketType::Event);
        assert_eq!(packet.header.priority, 200);
        assert!(packet.has_flag(PacketFlags::HIGH_PRIORITY));
    }

    #[test]
    fn test_payload_transformation() {
        let binary_packet = PhasePacket::new(
            PacketType::Data,
            1,
            2,
            PacketPayload::with_data(b"hello".to_vec()),
        );

        let text_packet =
            binary_packet.map_payload(|p| String::from_utf8_lossy(&p.data).into_owned());
        assert_eq!(text_packet.payload, "hello");
    }
}