//! Encoding and decoding traits for packet serialization

use crate::{PacketHeader, PhasePacket};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use lz4::block::{compress, decompress, CompressionMode};
use serde::{de::DeserializeOwned, Serialize};

/// Errors that can occur during packet encoding
#[derive(Debug, thiserror::Error)]
pub enum PacketEncodeError {
    /// Serialization failed
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    /// Buffer too small
    #[error("Buffer too small: need {needed}, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// Compression failed
    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    /// Invalid packet state
    #[error("Invalid packet state: {0}")]
    InvalidState(String),
}

/// Errors that can occur during packet decoding
#[derive(Debug, thiserror::Error)]
pub enum PacketDecodeError {
    /// Deserialization failed
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    /// Buffer underrun
    #[error("Buffer underrun: need {needed}, have {available}")]
    BufferUnderrun { needed: usize, available: usize },

    /// Decompression failed
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    /// Invalid packet format
    #[error("Invalid packet format: {0}")]
    InvalidFormat(String),

    /// Checksum mismatch
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Unsupported version
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),
}

/// Trait for encoding packets to binary format
pub trait PacketEncoder {
    /// Encode a packet to bytes
    ///
    /// # Errors
    ///
    /// Returns `PacketEncodeError` if serialization fails.
    fn encode<T>(&self, packet: &PhasePacket<T>) -> Result<Bytes, PacketEncodeError>
    where
        T: Serialize;

    /// Encode a packet header only
    ///
    /// # Errors
    ///
    /// Returns `PacketEncodeError` if serialization fails.
    fn encode_header(&self, header: &PacketHeader) -> Result<Bytes, PacketEncodeError>;

    /// Estimate encoded size
    ///
    /// # Errors
    ///
    /// Returns `PacketEncodeError` if serialization fails.
    fn encoded_size<T>(&self, packet: &PhasePacket<T>) -> Result<usize, PacketEncodeError>
    where
        T: Serialize;
}

/// Trait for decoding packets from binary format
pub trait PacketDecoder {
    /// Decode a packet from bytes
    ///
    /// # Errors
    ///
    /// Returns `PacketDecodeError` if deserialization fails.
    fn decode<T>(&self, data: &[u8]) -> Result<PhasePacket<T>, PacketDecodeError>
    where
        T: DeserializeOwned;

    /// Decode header only
    ///
    /// # Errors
    ///
    /// Returns `PacketDecodeError` if deserialization fails.
    fn decode_header(&self, data: &[u8]) -> Result<PacketHeader, PacketDecodeError>;

    /// Check if buffer contains a complete packet
    ///
    /// # Errors
    ///
    /// Returns `PacketDecodeError` if the buffer is malformed.
    fn is_complete(&self, data: &[u8]) -> Result<bool, PacketDecodeError>;
}

/// Binary codec for efficient packet encoding/decoding
#[derive(Debug, Clone, Default)]
pub struct BinaryCodec {
    /// Enable compression for large payloads
    pub compression_enabled: bool,

    /// Compression threshold in bytes
    pub compression_threshold: usize,

    /// Enable checksum validation
    pub checksum_enabled: bool,
}

impl BinaryCodec {
    /// Create new binary codec with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            compression_enabled: true,
            compression_threshold: 1024, // 1KB
            checksum_enabled: true,
        }
    }

    /// Create codec with custom settings
    #[must_use]
    pub fn with_settings(
        compression_enabled: bool,
        compression_threshold: usize,
        checksum_enabled: bool,
    ) -> Self {
        Self {
            compression_enabled,
            compression_threshold,
            checksum_enabled,
        }
    }

    /// Calculate packet checksum
    #[allow(clippy::cast_possible_truncation)]
    fn calculate_checksum(data: &[u8]) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Compress data if enabled and above threshold
    fn maybe_compress(&self, data: &[u8]) -> (Vec<u8>, bool) {
        if self.compression_enabled && data.len() > self.compression_threshold {
            if let Ok(compressed_data) = compress(data, Some(CompressionMode::HIGHCOMPRESSION(12)), true) {
                // Only use compression if it actually saves space
                if compressed_data.len() < data.len() {
                    (compressed_data, true)
                } else {
                    (data.to_vec(), false)
                }
            } else {
                // Fall back to uncompressed if compression fails
                (data.to_vec(), false)
            }
        } else {
            (data.to_vec(), false)
        }
    }

    /// Decompress data if compressed
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn maybe_decompress(
        data: &[u8],
        compressed: bool,
    ) -> Result<Vec<u8>, PacketDecodeError> {
        if compressed {
            // We need to know the original size for decompression
            // For now, use a reasonable maximum size estimate
            let max_decompressed_size = data.len() * 10; // Assume max 10x compression ratio
            match decompress(data, Some(max_decompressed_size as i32)) {
                Ok(decompressed_data) => Ok(decompressed_data),
                Err(_) => Err(PacketDecodeError::DecompressionFailed(
                    "Failed to decompress data".to_string(),
                )),
            }
        } else {
            Ok(data.to_vec())
        }
    }
}

impl PacketEncoder for BinaryCodec {
    #[allow(clippy::cast_possible_truncation)]
    fn encode<T>(&self, packet: &PhasePacket<T>) -> Result<Bytes, PacketEncodeError>
    where
        T: Serialize,
    {
        let mut buffer = BytesMut::new();

        // Serialize header
        let header_bytes = bincode::serialize(&packet.header)
            .map_err(|e| PacketEncodeError::SerializationFailed(e.to_string()))?;

        // Serialize payload
        let payload_bytes = bincode::serialize(&packet.payload)
            .map_err(|e| PacketEncodeError::SerializationFailed(e.to_string()))?;

        // Maybe compress payload
        let (compressed_payload, is_compressed) = self.maybe_compress(&payload_bytes);

        // Write header length (4 bytes)
        buffer.put_u32(header_bytes.len() as u32);

        // Write header
        buffer.put_slice(&header_bytes);

        // Write payload length (4 bytes)
        buffer.put_u32(compressed_payload.len() as u32);

        // Write compression flag (1 byte)
        buffer.put_u8(u8::from(is_compressed));

        // Write payload
        buffer.put_slice(&compressed_payload);

        // Write checksum if enabled (4 bytes)
        if self.checksum_enabled {
            let checksum = BinaryCodec::calculate_checksum(&buffer);
            buffer.put_u32(checksum);
        }

        Ok(buffer.freeze())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_header(&self, header: &PacketHeader) -> Result<Bytes, PacketEncodeError> {
        let header_bytes = bincode::serialize(header)
            .map_err(|e| PacketEncodeError::SerializationFailed(e.to_string()))?;

        let mut buffer = BytesMut::new();
        buffer.put_u32(header_bytes.len() as u32);
        buffer.put_slice(&header_bytes);

        Ok(buffer.freeze())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encoded_size<T>(&self, packet: &PhasePacket<T>) -> Result<usize, PacketEncodeError>
    where
        T: Serialize,
    {
        let header_size = bincode::serialized_size(&packet.header)
            .map_err(|e| PacketEncodeError::SerializationFailed(e.to_string()))?;

        let payload_size = bincode::serialized_size(&packet.payload)
            .map_err(|e| PacketEncodeError::SerializationFailed(e.to_string()))?;

        // Header length (4) + header + payload length (4) + compression flag (1) + payload + checksum (4)
        Ok(4 + header_size as usize
            + 4
            + 1
            + payload_size as usize
            + if self.checksum_enabled { 4 } else { 0 })
    }
}

impl PacketDecoder for BinaryCodec {
    #[allow(clippy::cast_possible_truncation)]
    fn decode<T>(&self, data: &[u8]) -> Result<PhasePacket<T>, PacketDecodeError>
    where
        T: DeserializeOwned,
    {
        let mut cursor = std::io::Cursor::new(data);

        // Read header length
        let header_len = cursor.get_u32() as usize;
        if cursor.remaining() < header_len {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: header_len,
                available: cursor.remaining(),
            });
        }

        // Read header
        let header_bytes =
            &data[cursor.position() as usize..cursor.position() as usize + header_len];
        cursor.advance(header_len);

        let header: PacketHeader = bincode::deserialize(header_bytes)
            .map_err(|e| PacketDecodeError::DeserializationFailed(e.to_string()))?;

        // Read payload length
        if cursor.remaining() < 4 {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: 4,
                available: cursor.remaining(),
            });
        }
        let payload_len = cursor.get_u32() as usize;

        // Read compression flag
        if cursor.remaining() < 1 {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: 1,
                available: cursor.remaining(),
            });
        }
        let is_compressed = cursor.get_u8() == 1;

        // Read payload
        if cursor.remaining() < payload_len {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: payload_len,
                available: cursor.remaining(),
            });
        }

        let payload_bytes =
            &data[cursor.position() as usize..cursor.position() as usize + payload_len];
        cursor.advance(payload_len);

        // Decompress if needed
        let decompressed_payload = BinaryCodec::maybe_decompress(payload_bytes, is_compressed)?;

        // Verify checksum if enabled
        if self.checksum_enabled {
            if cursor.remaining() < 4 {
                return Err(PacketDecodeError::BufferUnderrun {
                    needed: 4,
                    available: cursor.remaining(),
                });
            }

            let expected_checksum = cursor.get_u32();
            let actual_checksum = BinaryCodec::calculate_checksum(&data[..cursor.position() as usize - 4]);

            if expected_checksum != actual_checksum {
                return Err(PacketDecodeError::ChecksumMismatch {
                    expected: expected_checksum,
                    actual: actual_checksum,
                });
            }
        }

        // Deserialize payload
        let payload: T = bincode::deserialize(&decompressed_payload)
            .map_err(|e| PacketDecodeError::DeserializationFailed(e.to_string()))?;

        Ok(PhasePacket { header, payload })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn decode_header(&self, data: &[u8]) -> Result<PacketHeader, PacketDecodeError> {
        let mut cursor = std::io::Cursor::new(data);

        if cursor.remaining() < 4 {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: 4,
                available: cursor.remaining(),
            });
        }

        let header_len = cursor.get_u32() as usize;
        if cursor.remaining() < header_len {
            return Err(PacketDecodeError::BufferUnderrun {
                needed: header_len,
                available: cursor.remaining(),
            });
        }

        let header_bytes = &data[4..4 + header_len];
        let header: PacketHeader = bincode::deserialize(header_bytes)
            .map_err(|e| PacketDecodeError::DeserializationFailed(e.to_string()))?;

        Ok(header)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn is_complete(&self, data: &[u8]) -> Result<bool, PacketDecodeError> {
        if data.len() < 4 {
            return Ok(false);
        }

        let mut cursor = std::io::Cursor::new(data);
        let header_len = cursor.get_u32() as usize;

        if cursor.remaining() < header_len + 4 + 1 {
            return Ok(false);
        }

        cursor.advance(header_len);
        let payload_len = cursor.get_u32() as usize;
        cursor.advance(1); // compression flag

        let total_needed =
            4 + header_len + 4 + 1 + payload_len + if self.checksum_enabled { 4 } else { 0 };

        Ok(data.len() >= total_needed)
    }
}

/// Convenience struct combining encoder and decoder
#[derive(Debug, Clone)]
pub struct PacketCodec {
    encoder: BinaryCodec,
    decoder: BinaryCodec,
}

impl PacketCodec {
    /// Create new packet codec
    #[must_use]
    pub fn new() -> Self {
        let codec = BinaryCodec::new();
        Self {
            encoder: codec.clone(),
            decoder: codec,
        }
    }

    /// Encode a packet
    ///
    /// # Errors
    ///
    /// Returns `PacketEncodeError` if serialization fails.
    pub fn encode<T>(&self, packet: &PhasePacket<T>) -> Result<Bytes, PacketEncodeError>
    where
        T: Serialize,
    {
        self.encoder.encode(packet)
    }

    /// Decode a packet
    ///
    /// # Errors
    ///
    /// Returns `PacketDecodeError` if deserialization fails.
    pub fn decode<T>(&self, data: &[u8]) -> Result<PhasePacket<T>, PacketDecodeError>
    where
        T: DeserializeOwned,
    {
        self.decoder.decode(data)
    }
}

impl Default for PacketCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PacketPayload, PacketType, PhasePacket};

    #[test]
    fn test_binary_codec_roundtrip() {
        let codec = BinaryCodec::new();

        let packet = PhasePacket::new(
            PacketType::Data,
            1,
            2,
            PacketPayload::with_data(b"test data".to_vec()),
        )
        .finalize()
        .unwrap();

        let encoded = codec.encode(&packet).unwrap();
        let decoded: PhasePacket<PacketPayload> = codec.decode(&encoded).unwrap();

        assert_eq!(packet.header.packet_id, decoded.header.packet_id);
        assert_eq!(packet.payload.data, decoded.payload.data);
    }

    #[test]
    fn test_header_only_encoding() {
        let codec = BinaryCodec::new();
        let header = PacketHeader::new(PacketType::Control, 5, 10);

        let encoded = codec.encode_header(&header).unwrap();
        let decoded = codec.decode_header(&encoded).unwrap();

        assert_eq!(header.packet_type, decoded.packet_type);
        assert_eq!(header.source_node, decoded.source_node);
        assert_eq!(header.destination_node, decoded.destination_node);
    }

    #[test]
    fn test_incomplete_packet_detection() {
        let codec = BinaryCodec::new();

        // Empty buffer
        assert!(!codec.is_complete(&[]).unwrap());

        // Partial header
        assert!(!codec.is_complete(&[0, 0, 0, 10]).unwrap());
    }

    #[test]
    fn test_size_estimation() {
        let codec = BinaryCodec::new();
        let packet = PhasePacket::new(
            PacketType::Data,
            1,
            2,
            PacketPayload::with_data(b"test".to_vec()),
        );

        let estimated_size = codec.encoded_size(&packet).unwrap();
        let actual_encoded = codec.encode(&packet).unwrap();

        // Estimated size should be close to actual (within reasonable margin)
        let size_diff = (estimated_size as i32 - actual_encoded.len() as i32).abs();
        assert!(
            size_diff < 100,
            "Size estimation too far off: estimated={}, actual={}",
            estimated_size,
            actual_encoded.len()
        );
    }
}
