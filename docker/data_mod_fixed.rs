//! Data handling and structure management for PRCT engine

pub mod force_field;
pub mod atomic_data;
pub mod pdb_parser;
pub mod ramachandran;
pub mod contact_map;
pub mod casp16_loader;
pub mod lga_scoring;
pub mod blind_test_protocol;
pub mod casp16_comparison;

pub use force_field::*;
pub use atomic_data::*;
pub use pdb_parser::*;
pub use ramachandran::*;
pub use contact_map::*;
pub use casp16_loader::*;
pub use lga_scoring::*;
pub use blind_test_protocol::*;
pub use casp16_comparison::*;

use ndarray::{Array1, Array2};
use crate::PRCTError;

#[derive(Debug, Clone)]
pub struct ForceFieldParams {
    pub bond_k: f64,
    pub angle_k: f64,
    pub dihedral_k: f64,
    pub lj_epsilon: f64,
    pub lj_sigma: f64,
    pub coulomb_k: f64,
}

impl ForceFieldParams {
    pub fn new() -> Self {
        Self {
            bond_k: 300.0,
            angle_k: 50.0,
            dihedral_k: 1.0,
            lj_epsilon: 0.1,
            lj_sigma: 3.5,
            coulomb_k: 332.0,
        }
    }
}