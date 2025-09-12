//! PhasePacket<T> serialization system for ARES ChronoFabric Phase Coherence Bus
//!
//! This module provides a sophisticated serialization system for quantum-aware message passing
//! with temporal correlation preservation and sub-microsecond performance targets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::{ComponentId, NanoTime};
use csf_time::TimeSource;

/// Type alias for phase coherence factors
pub type CoherenceFactor = f64;

/// Type alias for quantum phase angles in radians
pub type PhaseAngle = f64;

/// Phase coherence states for quantum message correlation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhaseState {
    /// Coherent phase - messages maintain quantum correlation
    Coherent(PhaseAngle),
    /// Decoherent phase - quantum correlation lost
    Decoherent,
    /// Superposition phase - quantum superposition of multiple states
    Superposition(Vec<(PhaseAngle, CoherenceFactor)>),
    /// Entangled phase - quantum entanglement with other packets
    Entangled(ComponentId, PhaseAngle),
}

impl Default for PhaseState {
    fn default() -> Self {
        PhaseState::Coherent(0.0)
    }
}

/// High-performance PhasePacket for quantum-aware message serialization
///
/// PhasePacket<T> provides a serialization container that preserves quantum
/// coherence properties while achieving sub-microsecond performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePacket<T>
where
    T: Clone,
{
    /// Unique packet identifier
    pub packet_id: ComponentId,

    /// Phase state for quantum correlation
    pub phase_state: PhaseState,

    /// The actual payload data
    pub payload: T,

    /// Coherence factor for the entire packet
    pub coherence_factor: CoherenceFactor,

    /// Creation timestamp
    pub timestamp: NanoTime,

    /// Phase-locked routing information
    pub routing_info: HashMap<ComponentId, PhaseAngle>,

    /// Quantum entanglement map
    pub entanglement_map: HashMap<ComponentId, CoherenceFactor>,

    /// Type marker for generic safety
    _marker: PhantomData<T>,
}

impl<T> PhasePacket<T>
where
    T: Clone,
{
    /// Create a new PhasePacket with coherent phase
    pub fn new(payload: T) -> Self {
        let packet_id = ComponentId::new(rand::random());
        let timestamp = NanoTime::from_nanos(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        );

        Self {
            packet_id,
            phase_state: PhaseState::Coherent(0.0),
            payload,
            coherence_factor: 1.0,
            timestamp,
            routing_info: HashMap::new(),
            entanglement_map: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Create a new PhasePacket with specific phase state
    pub fn with_phase_state(payload: T, phase_state: PhaseState) -> Self {
        let mut packet = Self::new(payload);
        packet.phase_state = phase_state;
        packet
    }

    /// Add routing information for phase-aware delivery
    pub fn add_route(&mut self, component: ComponentId, phase: PhaseAngle) {
        self.routing_info.insert(component, phase);
    }

    /// Create quantum entanglement with another component
    pub fn entangle_with(&mut self, component: ComponentId, strength: CoherenceFactor) {
        self.entanglement_map
            .insert(component, strength.clamp(0.0, 1.0));

        // Update phase state to entangled
        if let PhaseState::Coherent(phase) = self.phase_state {
            self.phase_state = PhaseState::Entangled(component, phase);
        }

        // Enhance coherence due to entanglement (allow > 1.0 for quantum enhancement)
        self.coherence_factor = self.coherence_factor * (1.0 + strength * 0.1);
    }

    /// Apply quantum phase shift to the packet
    pub fn apply_phase_shift(&mut self, phase_shift: PhaseAngle) {
        match &mut self.phase_state {
            PhaseState::Coherent(phase) => {
                *phase = (*phase + phase_shift) % (2.0 * std::f64::consts::PI);
            }
            PhaseState::Entangled(_, phase) => {
                *phase = (*phase + phase_shift) % (2.0 * std::f64::consts::PI);
            }
            PhaseState::Superposition(states) => {
                for (phase, _) in states.iter_mut() {
                    *phase = (*phase + phase_shift) % (2.0 * std::f64::consts::PI);
                }
            }
            PhaseState::Decoherent => {
                // Cannot apply phase shift to decoherent state
            }
        }

        // Update routing phases
        for phase in self.routing_info.values_mut() {
            *phase = (*phase + phase_shift) % (2.0 * std::f64::consts::PI);
        }
    }

    /// Check if packet is phase-coherent
    pub fn is_coherent(&self) -> bool {
        matches!(
            self.phase_state,
            PhaseState::Coherent(_) | PhaseState::Entangled(_, _) | PhaseState::Superposition(_)
        ) && self.coherence_factor > 0.5
    }

    /// Calculate phase correlation with another packet
    pub fn phase_correlation(&self, other: &Self) -> CoherenceFactor {
        let self_phase = match self.phase_state {
            PhaseState::Coherent(phase) | PhaseState::Entangled(_, phase) => phase,
            PhaseState::Superposition(ref states) => {
                if states.is_empty() {
                    return 0.0;
                }
                states
                    .iter()
                    .map(|(phase, weight)| phase * weight)
                    .sum::<f64>()
                    / states.iter().map(|(_, weight)| weight).sum::<f64>()
            }
            PhaseState::Decoherent => return 0.0,
        };

        let other_phase = match other.phase_state {
            PhaseState::Coherent(phase) | PhaseState::Entangled(_, phase) => phase,
            PhaseState::Superposition(ref states) => {
                if states.is_empty() {
                    return 0.0;
                }
                states
                    .iter()
                    .map(|(phase, weight)| phase * weight)
                    .sum::<f64>()
                    / states.iter().map(|(_, weight)| weight).sum::<f64>()
            }
            PhaseState::Decoherent => return 0.0,
        };

        let phase_diff = (self_phase - other_phase).abs();
        let normalized_diff = phase_diff.min(2.0 * std::f64::consts::PI - phase_diff);

        (1.0 - normalized_diff / std::f64::consts::PI)
            * self.coherence_factor
            * other.coherence_factor
    }

    /// Get packet age in nanoseconds
    pub fn age_ns(&self) -> u64 {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Simplified age calculation
        current_time.saturating_sub(self.timestamp.as_nanos())
    }

    /// Check if packet has expired based on coherence window
    pub fn is_expired(&self, coherence_window_ns: u64) -> bool {
        self.age_ns() > coherence_window_ns || !self.is_coherent()
    }

    /// Quantum serialize to bytes
    pub fn quantum_serialize(&self) -> Result<Vec<u8>, String>
    where
        T: serde::Serialize,
    {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    /// Quantum deserialize from bytes
    pub fn quantum_deserialize(data: &[u8]) -> Result<Self, String>
    where
        T: serde::de::DeserializeOwned,
    {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }

    /// Create a new PhasePacket with coherent phase using enterprise TimeSource
    pub fn new_enterprise(payload: T, time_source: &dyn TimeSource) -> Self {
        let packet_id = ComponentId::new(rand::random());
        let timestamp = time_source.now_ns().unwrap_or_else(|_| NanoTime::from_nanos(0));

        Self {
            packet_id,
            phase_state: PhaseState::Coherent(0.0),
            payload,
            coherence_factor: 1.0,
            timestamp,
            routing_info: HashMap::new(),
            entanglement_map: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Create a new PhasePacket with specific phase state using enterprise TimeSource
    pub fn with_phase_state_enterprise(payload: T, phase_state: PhaseState, time_source: &dyn TimeSource) -> Self {
        let mut packet = Self::new_enterprise(payload, time_source);
        packet.phase_state = phase_state;
        packet
    }

    /// Get packet age using enterprise TimeSource
    pub fn age_ns_enterprise(&self, time_source: &dyn TimeSource) -> u64 {
        let current_time = time_source.now_ns().unwrap_or_else(|_| NanoTime::from_nanos(0));
        current_time.saturating_sub(self.timestamp).as_nanos()
    }

    /// Check if packet has expired using enterprise TimeSource
    pub fn is_expired_enterprise(&self, coherence_window_ns: u64, time_source: &dyn TimeSource) -> bool {
        self.age_ns_enterprise(time_source) > coherence_window_ns || !self.is_coherent()
    }
}

// PhasePacket uses existing NanoTime::as_nanos() method

impl<T> fmt::Display for PhasePacket<T>
where
    T: Clone + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PhasePacket {{")?;
        writeln!(f, "  id: {:?}", self.packet_id)?;
        writeln!(f, "  phase_state: {:?}", self.phase_state)?;
        writeln!(f, "  coherence: {:.3}", self.coherence_factor)?;
        writeln!(f, "  age_ns: {}", self.age_ns())?;
        writeln!(f, "  routes: {}", self.routing_info.len())?;
        writeln!(f, "  entanglements: {}", self.entanglement_map.len())?;
        writeln!(f, "  payload: {}", self.payload)?;
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_packet_creation() {
        let payload = "test data".to_string();
        let packet = PhasePacket::new(payload.clone());

        assert_eq!(packet.payload, payload);
        assert!(packet.is_coherent());
        assert_eq!(packet.coherence_factor, 1.0);
        assert!(matches!(packet.phase_state, PhaseState::Coherent(0.0)));
    }

    #[test]
    fn test_phase_state_operations() {
        let payload = 42i32;
        let mut packet = PhasePacket::with_phase_state(
            payload,
            PhaseState::Coherent(std::f64::consts::PI / 4.0),
        );

        // Apply phase shift
        packet.apply_phase_shift(std::f64::consts::PI / 4.0);

        match packet.phase_state {
            PhaseState::Coherent(phase) => {
                assert!((phase - std::f64::consts::PI / 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected coherent phase state"),
        }
    }

    #[test]
    fn test_quantum_entanglement() {
        let mut packet = PhasePacket::new(vec![1, 2, 3, 4]);
        let component = ComponentId::new(123);

        packet.entangle_with(component, 0.9);

        assert!(packet.entanglement_map.contains_key(&component));
        assert_eq!(packet.entanglement_map[&component], 0.9);
        assert!(packet.coherence_factor > 1.0); // Enhanced by entanglement

        match packet.phase_state {
            PhaseState::Entangled(entangled_component, _) => {
                assert_eq!(entangled_component, component);
            }
            _ => panic!("Expected entangled phase state"),
        }
    }

    #[test]
    fn test_phase_correlation() {
        let packet1 = PhasePacket::with_phase_state("data1".to_string(), PhaseState::Coherent(0.0));

        let packet2 = PhasePacket::with_phase_state(
            "data2".to_string(),
            PhaseState::Coherent(std::f64::consts::PI),
        );

        let correlation = packet1.phase_correlation(&packet2);
        assert!(correlation < 0.1); // Anti-correlated phases

        let packet3 = PhasePacket::with_phase_state("data3".to_string(), PhaseState::Coherent(0.1));

        let correlation2 = packet1.phase_correlation(&packet3);
        assert!(correlation2 > 0.9); // Highly correlated phases
    }

    #[test]
    fn test_packet_routing() {
        let mut packet = PhasePacket::new(42u64);

        let route1 = ComponentId::new(1);
        let route2 = ComponentId::new(2);

        packet.add_route(route1, std::f64::consts::PI / 3.0);
        packet.add_route(route2, std::f64::consts::PI * 2.0 / 3.0);

        assert_eq!(packet.routing_info.len(), 2);
        assert!(packet.routing_info.contains_key(&route1));
        assert!(packet.routing_info.contains_key(&route2));
    }

    #[test]
    fn test_packet_serialization() {
        let packet = PhasePacket::new("test serialization".to_string());

        let serialized = packet.quantum_serialize().unwrap();
        let deserialized: PhasePacket<String> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();

        assert_eq!(packet.payload, deserialized.payload);
        assert_eq!(packet.coherence_factor, deserialized.coherence_factor);
        assert_eq!(packet.phase_state, deserialized.phase_state);
    }

    #[test]
    fn test_superposition_state() {
        let states = vec![(0.0, 0.6), (std::f64::consts::PI / 2.0, 0.4)];

        let packet =
            PhasePacket::with_phase_state(100u32, PhaseState::Superposition(states.clone()));

        assert!(packet.is_coherent());

        if let PhaseState::Superposition(packet_states) = &packet.phase_state {
            assert_eq!(packet_states.len(), 2);
        } else {
            panic!("Expected superposition state");
        }
    }

    #[test]
    fn test_serialization_performance() {
        let packet = PhasePacket::new(vec![1u8; 1024]); // 1KB payload

        let start = std::time::Instant::now();
        let serialized = packet.quantum_serialize().unwrap();
        let serialize_time = start.elapsed();

        let start = std::time::Instant::now();
        let _deserialized: PhasePacket<Vec<u8>> =
            PhasePacket::quantum_deserialize(&serialized).unwrap();
        let deserialize_time = start.elapsed();

        // Performance validation - these should be very fast
        println!("Serialize time: {:?}", serialize_time);
        println!("Deserialize time: {:?}", deserialize_time);

        // Basic sanity check - should complete in reasonable time
        assert!(serialize_time.as_micros() < 1000); // Less than 1ms
        assert!(deserialize_time.as_micros() < 1000); // Less than 1ms
    }
}
