//! 3D Structure Generation Module
//! Converts PRCT energy landscape to atomic coordinates

pub mod atom;
pub mod backbone;
pub mod sidechains;
pub mod pdb_writer;
pub mod folder;
pub mod sequences;

pub use atom::{Atom, AtomType, Element, BackboneAtom};
pub use backbone::generate_backbone_coordinates;
pub use sidechains::{place_side_chains, AminoAcid};
pub use pdb_writer::{write_pdb_file, write_pdb_structure};
pub use folder::PRCTFolder;
pub use sequences::CASP_SEQUENCES;

use crate::PRCTResult;
use std::f64::consts::PI;

/// 3D coordinate structure containing backbone and side chains
#[derive(Debug, Clone)]
pub struct Structure3D {
    pub atoms: Vec<Atom>,
    pub target_id: String,
    pub confidence: f64,
    pub energy: f64,
}

impl Structure3D {
    pub fn new(atoms: Vec<Atom>, target_id: String, confidence: f64, energy: f64) -> Self {
        Self {
            atoms,
            target_id,
            confidence,
            energy,
        }
    }

    pub fn backbone_atoms(&self) -> Vec<&Atom> {
        self.atoms.iter()
            .filter(|atom| matches!(atom.atom_type, AtomType::Backbone))
            .collect()
    }

    pub fn sidechain_atoms(&self) -> Vec<&Atom> {
        self.atoms.iter()
            .filter(|atom| matches!(atom.atom_type, AtomType::SideChain))
            .collect()
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    pub fn residue_count(&self) -> usize {
        self.atoms.iter()
            .map(|atom| atom.residue_id)
            .max()
            .unwrap_or(0) as usize
    }
}

/// Standard bond lengths in Angstroms (from crystallographic data)
pub mod bond_lengths {
    pub const N_CA: f64 = 1.458;   // N-Cα bond
    pub const CA_C: f64 = 1.525;   // Cα-C bond
    pub const C_N: f64 = 1.329;    // C-N peptide bond
    pub const C_O: f64 = 1.231;    // C=O double bond
    pub const CA_CB: f64 = 1.530;  // Cα-Cβ bond
}

/// Standard bond angles in radians
pub mod bond_angles {
    use std::f64::consts::PI;

    pub const N_CA_C: f64 = 111.2 * PI / 180.0;     // N-Cα-C angle
    pub const CA_C_N: f64 = 116.2 * PI / 180.0;     // Cα-C-N angle
    pub const C_N_CA: f64 = 121.7 * PI / 180.0;     // C-N-Cα angle
    pub const CA_C_O: f64 = 120.8 * PI / 180.0;     // Cα-C=O angle
    pub const N_CA_CB: f64 = 110.1 * PI / 180.0;    // N-Cα-Cβ angle
}

/// Utility functions for coordinate calculations
pub fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
}

pub fn normalize_vector(v: &mut [f64; 3]) {
    let len = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
    if len > 1e-10 {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

pub fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Convert degrees to radians
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Convert radians to degrees
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * 180.0 / PI
}