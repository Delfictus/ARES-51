//! Packet routing logic

use crate::packet::PhasePacket;
use csf_core::{ComponentId, Priority};


/// Intelligent packet router
#[derive(Default)]
pub struct PacketRouter {
    /// Routing rules
    rules: Vec<RoutingRule>,
}

/// A routing rule
pub struct RoutingRule {
    /// Source component filter
    pub source: Option<ComponentId>,

    /// Packet type filter
    pub packet_type: Option<std::any::TypeId>,

    /// Target components
    pub targets: Vec<ComponentId>,

    /// Priority threshold
    pub min_priority: Priority,
}

impl PacketRouter {
    /// Create a new router
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
        }
    }

    /// Add a routing rule
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
    }

    /// Compute target components for a packet
    pub fn compute_targets<T>(&self, packet: &PhasePacket<T>) -> u64 {
        let mut target_mask = packet.routing_metadata.target_component_mask;

        // Apply routing rules
        for rule in &self.rules {
            if self.rule_matches(rule, packet) {
                // Add targets from rule
                for target in &rule.targets {
                    target_mask |= self.component_to_mask(target);
                }
            }
        }

        target_mask
    }

    fn rule_matches<T>(&self, rule: &RoutingRule, packet: &PhasePacket<T>) -> bool {
        // Check source
        if let Some(source) = rule.source {
            if packet.routing_metadata.source_id != source {
                return false;
            }
        }

        // Check priority
        if packet.routing_metadata.priority > rule.min_priority {
            return false;
        }

        true
    }

    fn component_to_mask(&self, component: &ComponentId) -> u64 {
        match *component {
            ComponentId::DRPP => 1 << 0,
            ComponentId::ADP => 1 << 1,
            ComponentId::EGC => 1 << 2,
            ComponentId::EMS => 1 << 3,
            // For other components, use a hash-based approach
            _ => {
                // Simple approach: use the first 6 bits of the hash
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                component.hash(&mut hasher);
                let hash = hasher.finish();

                1 << (hash % 64)
            }
        }
    }
}