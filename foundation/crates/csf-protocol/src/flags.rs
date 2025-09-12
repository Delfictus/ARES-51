//! Packet flags for state and processing hints

use serde::{Deserialize, Serialize};

bitflags::bitflags! {
    /// Bitflags representing packet state and processing hints
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct PacketFlags: u32 {
        /// Packet has been processed by a module
        const PROCESSED = 1 << 0;

        /// Packet contains or represents an error condition
        const ERROR = 1 << 1;

        /// Urgent processing required
        const URGENT = 1 << 2;

        /// Payload is compressed
        const COMPRESSED = 1 << 3;

        /// Packet was dropped during processing
        const DROPPED = 1 << 4;

        /// Packet is temporarily buffered
        const BUFFERED = 1 << 5;

        /// Packet payload is encrypted
        const ENCRYPTED = 1 << 6;

        /// Packet requires acknowledgment
        const ACK_REQUIRED = 1 << 7;

        /// This packet is an acknowledgment
        const ACKNOWLEDGMENT = 1 << 8;

        /// Packet is part of a fragmented message
        const FRAGMENTED = 1 << 9;

        /// Last fragment in a sequence
        const LAST_FRAGMENT = 1 << 10;

        /// Packet contains quantum correlation data
        const QUANTUM_CORRELATED = 1 << 11;

        /// High priority processing
        const HIGH_PRIORITY = 1 << 12;

        /// Low priority processing
        const LOW_PRIORITY = 1 << 13;

        /// Packet is for diagnostic/monitoring purposes
        const DIAGNOSTIC = 1 << 14;

        /// Packet should be logged for audit trail
        const AUDIT = 1 << 15;

        // Reserve high bits for future use
        const _RESERVED_16 = 1 << 16;
        const _RESERVED_17 = 1 << 17;
        const _RESERVED_18 = 1 << 18;
        const _RESERVED_19 = 1 << 19;
        const _RESERVED_20 = 1 << 20;
        const _RESERVED_21 = 1 << 21;
        const _RESERVED_22 = 1 << 22;
        const _RESERVED_23 = 1 << 23;
        const _RESERVED_24 = 1 << 24;
        const _RESERVED_25 = 1 << 25;
        const _RESERVED_26 = 1 << 26;
        const _RESERVED_27 = 1 << 27;
        const _RESERVED_28 = 1 << 28;
        const _RESERVED_29 = 1 << 29;
        const _RESERVED_30 = 1 << 30;
        const _RESERVED_31 = 1 << 31;
    }
}

impl PacketFlags {
    /// Check if packet has any priority flags set
    #[must_use]
    pub fn has_priority(&self) -> bool {
        self.intersects(Self::HIGH_PRIORITY | Self::LOW_PRIORITY | Self::URGENT)
    }

    /// Get effective priority level (0 = low, 1 = normal, 2 = high, 3 = urgent)
    #[must_use]
    pub fn priority_level(&self) -> u8 {
        if self.contains(Self::URGENT) {
            3
        } else if self.contains(Self::HIGH_PRIORITY) {
            2
        } else {
            u8::from(!self.contains(Self::LOW_PRIORITY))
        }
    }

    /// Check if packet requires secure handling
    #[must_use]
    pub fn requires_security(&self) -> bool {
        self.contains(Self::ENCRYPTED)
    }

    /// Check if packet is in an error state
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.contains(Self::ERROR) || self.contains(Self::DROPPED)
    }

    /// Check if packet is complete (not fragmented or last fragment)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.contains(Self::FRAGMENTED) || self.contains(Self::LAST_FRAGMENT)
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        Self::empty()
    }
}

impl std::fmt::Display for PacketFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut flags = Vec::new();

        if self.contains(Self::PROCESSED) {
            flags.push("PROCESSED");
        }
        if self.contains(Self::ERROR) {
            flags.push("ERROR");
        }
        if self.contains(Self::URGENT) {
            flags.push("URGENT");
        }
        if self.contains(Self::COMPRESSED) {
            flags.push("COMPRESSED");
        }
        if self.contains(Self::DROPPED) {
            flags.push("DROPPED");
        }
        if self.contains(Self::BUFFERED) {
            flags.push("BUFFERED");
        }
        if self.contains(Self::ENCRYPTED) {
            flags.push("ENCRYPTED");
        }
        if self.contains(Self::ACK_REQUIRED) {
            flags.push("ACK_REQUIRED");
        }
        if self.contains(Self::ACKNOWLEDGMENT) {
            flags.push("ACKNOWLEDGMENT");
        }
        if self.contains(Self::FRAGMENTED) {
            flags.push("FRAGMENTED");
        }
        if self.contains(Self::LAST_FRAGMENT) {
            flags.push("LAST_FRAGMENT");
        }
        if self.contains(Self::QUANTUM_CORRELATED) {
            flags.push("QUANTUM_CORRELATED");
        }
        if self.contains(Self::HIGH_PRIORITY) {
            flags.push("HIGH_PRIORITY");
        }
        if self.contains(Self::LOW_PRIORITY) {
            flags.push("LOW_PRIORITY");
        }
        if self.contains(Self::DIAGNOSTIC) {
            flags.push("DIAGNOSTIC");
        }
        if self.contains(Self::AUDIT) {
            flags.push("AUDIT");
        }

        if flags.is_empty() {
            write!(f, "NONE")
        } else {
            write!(f, "{}", flags.join("|"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_levels() {
        assert_eq!(PacketFlags::URGENT.priority_level(), 3);
        assert_eq!(PacketFlags::HIGH_PRIORITY.priority_level(), 2);
        assert_eq!(PacketFlags::empty().priority_level(), 1);
        assert_eq!(PacketFlags::LOW_PRIORITY.priority_level(), 0);
    }

    #[test]
    fn test_security_requirements() {
        assert!(PacketFlags::ENCRYPTED.requires_security());
        assert!(!PacketFlags::empty().requires_security());
    }

    #[test]
    fn test_error_detection() {
        assert!(PacketFlags::ERROR.is_error());
        assert!(PacketFlags::DROPPED.is_error());
        assert!(!PacketFlags::PROCESSED.is_error());
    }

    #[test]
    fn test_fragmentation() {
        assert!(PacketFlags::empty().is_complete());
        assert!(!PacketFlags::FRAGMENTED.is_complete());
        assert!((PacketFlags::FRAGMENTED | PacketFlags::LAST_FRAGMENT).is_complete());
    }
}