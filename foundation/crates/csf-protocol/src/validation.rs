//! Packet validation and invariant enforcement

use crate::{PacketFlags, PacketHeader, PhasePacket};
use std::collections::HashMap;

/// Packet validation errors
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ValidationError {
    /// Unsupported protocol version
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),

    /// Missing sequence number for fragmented packet
    #[error("Fragmented packet missing sequence number")]
    MissingSequenceNumber,

    /// Missing fragment count for fragmented packet
    #[error("Fragmented packet missing fragment count")]
    MissingFragmentCount,

    /// Payload too large
    #[error("Payload size {size} exceeds maximum {max}")]
    PayloadTooLarge { size: usize, max: usize },

    /// Too many metadata entries
    #[error("Metadata entries {size} exceeds maximum {max}")]
    TooManyMetadataEntries { size: usize, max: usize },

    /// Invalid flag combination
    #[error("Invalid flag combination: {details}")]
    InvalidFlags { details: String },

    /// Checksum mismatch
    #[error("Header checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Invalid sequence number
    #[error("Invalid sequence number {seq} for fragment count {total}")]
    InvalidSequenceNumber { seq: u32, total: u32 },

    /// Priority conflict between field and flags
    #[error("Priority conflict: field={field}, flags={flags}")]
    PriorityConflict { field: u8, flags: u8 },

    /// Security policy violation
    #[error("Security policy violation: {policy}")]
    SecurityViolation { policy: String },
}

/// Security policies for packet validation
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Require encryption for high priority packets
    pub encrypt_high_priority: bool,

    /// Maximum priority level allowed
    pub max_priority: u8,

    /// Require audit logging for certain flags
    pub audit_flags: PacketFlags,

    /// Forbidden flag combinations
    pub forbidden_combinations: Vec<(PacketFlags, PacketFlags)>,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            encrypt_high_priority: true,
            max_priority: 255,
            audit_flags: PacketFlags::ERROR | PacketFlags::DROPPED,
            forbidden_combinations: vec![
                // Can't be both high and low priority
                (PacketFlags::HIGH_PRIORITY, PacketFlags::LOW_PRIORITY),
                // Error packets shouldn't be processed
                (PacketFlags::ERROR, PacketFlags::PROCESSED),
                // Dropped packets shouldn't be processed
                (PacketFlags::DROPPED, PacketFlags::PROCESSED),
            ],
        }
    }
}

/// Packet validator with configurable policies
#[derive(Debug, Clone)]
pub struct PacketValidator {
    /// Security policy to enforce
    policy: SecurityPolicy,

    /// Cache of validation results for performance
    validation_cache: HashMap<u64, bool>,
}

impl PacketValidator {
    /// Create a new validator with default policy
    #[must_use]
    pub fn new() -> Self {
        Self {
            policy: SecurityPolicy::default(),
            validation_cache: HashMap::new(),
        }
    }

    /// Create validator with custom policy
    #[must_use]
    pub fn with_policy(policy: SecurityPolicy) -> Self {
        Self {
            policy,
            validation_cache: HashMap::new(),
        }
    }

    /// Validate a packet header
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the header is invalid.
    pub fn validate_header(&self, header: &PacketHeader) -> Result<(), ValidationError> {
        // Basic header validation
        header.validate()?;

        // Security policy validation
        self.validate_security_policy(header)?;

        // Flag combination validation
        self.validate_flag_combinations(header.flags)?;

        // Checksum validation
        let expected_checksum = header.calculate_checksum();
        if header.checksum != 0 && header.checksum != expected_checksum {
            return Err(ValidationError::ChecksumMismatch {
                expected: expected_checksum,
                actual: header.checksum,
            });
        }

        // Fragmentation validation
        if header.is_fragmented() {
            Self::validate_fragmentation(header)?;
        }

        Ok(())
    }

    /// Validate a complete packet
    ///
    /// # Errors
    ///
    /// Returns `ValidationError` if the packet is invalid.
    #[allow(clippy::cast_possible_truncation)]
    pub fn validate_packet<T>(&self, packet: &PhasePacket<T>) -> Result<(), ValidationError>
    where
        T: serde::Serialize,
    {
        // Validate header
        self.validate_header(&packet.header)?;

        // Validate payload size consistency
        let serialized =
            bincode::serialize(&packet.payload).map_err(|_| ValidationError::PayloadTooLarge {
                size: 0,
                max: crate::MAX_PACKET_SIZE,
            })?;

        if serialized.len() as u32 != packet.header.payload_size {
            return Err(ValidationError::PayloadTooLarge {
                size: serialized.len(),
                max: packet.header.payload_size as usize,
            });
        }

        Ok(())
    }

    /// Validate security policy compliance
    fn validate_security_policy(&self, header: &PacketHeader) -> Result<(), ValidationError> {
        // Check maximum priority
        if header.priority > self.policy.max_priority {
            return Err(ValidationError::SecurityViolation {
                policy: format!(
                    "Priority {} exceeds maximum {}",
                    header.priority, self.policy.max_priority
                ),
            });
        }

        // Check encryption requirements
        if self.policy.encrypt_high_priority
            && header.flags.contains(PacketFlags::HIGH_PRIORITY)
            && !header.flags.contains(PacketFlags::ENCRYPTED)
        {
            return Err(ValidationError::SecurityViolation {
                policy: "High priority packets must be encrypted".to_string(),
            });
        }

        // Check audit requirements
        if header.flags.intersects(self.policy.audit_flags)
            && !header.flags.contains(PacketFlags::AUDIT)
        {
            return Err(ValidationError::SecurityViolation {
                policy: "Packets with sensitive flags must be audited".to_string(),
            });
        }

        Ok(())
    }

    /// Validate flag combinations
    fn validate_flag_combinations(&self, flags: PacketFlags) -> Result<(), ValidationError> {
        for (flag1, flag2) in &self.policy.forbidden_combinations {
            if flags.contains(*flag1) && flags.contains(*flag2) {
                return Err(ValidationError::InvalidFlags {
                    details: format!("Cannot combine {flag1} and {flag2}"),
                });
            }
        }

        Ok(())
    }

    /// Validate fragmentation settings
    fn validate_fragmentation(header: &PacketHeader) -> Result<(), ValidationError> {
        if let (Some(seq), Some(total)) = (header.sequence_number, header.fragment_count) {
            if seq >= total {
                return Err(ValidationError::InvalidSequenceNumber { seq, total });
            }
        }

        Ok(())
    }

    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }

    /// Get cache size for monitoring
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.validation_cache.len()
    }
}

impl Default for PacketValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to validate a packet with default policy
///
/// # Errors
///
/// Returns `ValidationError` if the packet is invalid.
pub fn validate_packet<T>(packet: &PhasePacket<T>) -> Result<(), ValidationError>
where
    T: serde::Serialize,
{
    PacketValidator::new().validate_packet(packet)
}

/// Convenience function to validate a header with default policy
///
/// # Errors
///
/// Returns `ValidationError` if the header is invalid.
pub fn validate_header(header: &PacketHeader) -> Result<(), ValidationError> {
    PacketValidator::new().validate_header(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PacketPayload, PacketType, PhasePacket};

    #[test]
    fn test_valid_header() {
        let header = PacketHeader::new(PacketType::Data, 1, 2);
        assert!(validate_header(&header).is_ok());
    }

    #[test]
    fn test_invalid_flag_combination() {
        let validator = PacketValidator::new();
        let flags = PacketFlags::HIGH_PRIORITY | PacketFlags::LOW_PRIORITY;

        let result = validator.validate_flag_combinations(flags);
        assert!(matches!(result, Err(ValidationError::InvalidFlags { .. })));
    }

    #[test]
    fn test_security_policy_encryption() {
        let mut policy = SecurityPolicy::default();
        policy.encrypt_high_priority = true;

        let validator = PacketValidator::with_policy(policy);

        let mut header = PacketHeader::new(PacketType::Data, 1, 2);
        header.flags = PacketFlags::HIGH_PRIORITY; // High priority but not encrypted

        let result = validator.validate_security_policy(&header);
        assert!(matches!(
            result,
            Err(ValidationError::SecurityViolation { .. })
        ));
    }

    #[test]
    fn test_fragmentation_validation() {
        let validator = PacketValidator::new();

        let mut header = PacketHeader::new(PacketType::Data, 1, 2);
        header.flags = PacketFlags::FRAGMENTED;
        header.sequence_number = Some(5);
        header.fragment_count = Some(3); // Invalid: seq >= total

        let result = PacketValidator::validate_fragmentation(&header);
        assert!(matches!(
            result,
            Err(ValidationError::InvalidSequenceNumber { .. })
        ));
    }

    #[test]
    fn test_complete_packet_validation() {
        let mut packet = PhasePacket::new(
            PacketType::Data,
            1,
            2,
            PacketPayload::with_data(b"test".to_vec()),
        );

        packet = packet.finalize().unwrap();
        assert!(validate_packet(&packet).is_ok());
    }
}