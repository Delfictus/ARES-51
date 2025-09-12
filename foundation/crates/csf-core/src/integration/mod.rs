//! ARES ChronoFabric Integration Framework
//!
//! This module provides the unified runtime for the DRPP system, integrating
//! variational energy functionals, phase space dynamics, and bus communication
//! into a cohesive framework for emergent relational behavior.

pub mod monitoring;
pub mod runtime;
pub mod testing;

pub use monitoring::*;
pub use runtime::*;
pub use testing::*;
