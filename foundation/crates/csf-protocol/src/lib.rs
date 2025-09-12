//! Canonical protocol definitions for ARES `ChronoFabric` System
//!
//! This crate provides the single source of truth for all packet definitions,
//! flags, headers, and encoding/decoding traits used across the CSF system.
//!
//! # Architecture
//!
//! - `PacketHeader`: Core packet metadata and routing information
//! - `PacketFlags`: Bitflags for packet state and processing hints
//! - `PhasePacket<T>`: Generic packet container with type-safe payloads
//! - `PacketCodec`: Encoding/decoding traits for network and FFI
//!
//! # Design Principles
//!
//! - **Single source of truth**: No duplicate packet definitions
//! - **Forward compatibility**: Non-exhaustive enums and versioned headers
//! - **Security first**: Invariant enforcement at construction time
//! - **Zero-copy**: Efficient serialization and deserialization
//! - **Type safety**: Generic payloads with compile-time guarantees

#![deny(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(missing_docs)]

pub mod codec;
pub mod flags;
pub mod packet;
pub mod validation;

// Re-export core types for convenience
pub use codec::{PacketCodec, PacketDecodeError, PacketEncodeError};
pub use flags::PacketFlags;
pub use packet::{BinaryPacket, PacketHeader, PacketId, PacketPayload, PhasePacket};
pub use validation::{PacketValidator, ValidationError};

// Default PhasePacket type for backwards compatibility
pub type DefaultPhasePacket = PhasePacket<PacketPayload>;

// Re-export shared types
pub use csf_shared_types::{ComponentId, NanoTime, PacketType, PrecisionLevel, TaskId};

/// Protocol version for forward compatibility
pub const PROTOCOL_VERSION: u8 = 1;

/// Maximum packet size in bytes (16MB)
pub const MAX_PACKET_SIZE: usize = 16 * 1024 * 1024;

/// Maximum metadata entries per packet
pub const MAX_METADATA_ENTRIES: usize = 256;

/// Result type for protocol operations
pub type ProtocolResult<T> = Result<T, ProtocolError>;

/// Protocol-level errors
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// Packet validation failed
    #[error("Packet validation failed: {0}")]
    ValidationFailed(#[from] ValidationError),

    /// Encoding error
    #[error("Packet encoding failed: {0}")]
    EncodeFailed(#[from] PacketEncodeError),

    /// Decoding error  
    #[error("Packet decoding failed: {0}")]
    DecodeFailed(#[from] PacketDecodeError),

    /// Unsupported protocol version
    #[error("Unsupported protocol version: {version}")]
    UnsupportedVersion { version: u8 },

    /// Packet too large
    #[error("Packet size {size} exceeds maximum {max}")]
    PacketTooLarge { size: usize, max: usize },
}
