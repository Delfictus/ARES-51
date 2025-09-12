"""//! Tests for the distributed resonance computation

use hephaestus_forge::{
    distributed::{DistributedPhaseLattice, NetworkTopology},
    resonance::{ComputationTensor, ResonantSolution},
};
use nalgebra::DMatrix;
use std::sync::Arc;

#[tokio::test]
async fn test_distributed_lattice_initialization() {
    let lattice = DistributedPhaseLattice::new("node_0".to_string(), (8, 8, 4)).await;
    // This is a basic check to ensure the lattice is created.
    // More detailed checks would require accessing internal state, which is not exposed.
}

#[tokio::test]
async fn test_distributed_protocol_initialization() {
    let mut protocol = DistributedResonanceProtocol::new(NetworkTopology::Ring);
    protocol.initialize(4).await;
    // This is a basic check to ensure the protocol is created.
    // More detailed checks would require accessing internal state, which is not exposed.
}
""