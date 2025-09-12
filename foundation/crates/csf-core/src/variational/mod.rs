//! Variational Action Principle Implementation
//!
//! This module implements the core theoretical framework for the DRPP/ADP system,
//! translating variational calculus and action principles into computational structures.

pub mod action_principle;
pub mod energy_functional;
pub mod hamiltonian;
pub mod lagrangian;
pub mod phase_space;
pub mod phd_energy_functional;
pub mod topological_data_analysis;

pub use action_principle::*;
pub use energy_functional::*;
pub use hamiltonian::*;
pub use lagrangian::*;
pub use phase_space::*;
pub use phd_energy_functional::*;
pub use topological_data_analysis::*;
