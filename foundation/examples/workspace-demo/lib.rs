//! ARES ChronoFabric Library
//!
//! Core library for the ChronoFabric temporal computing system.

pub mod config;
pub mod runtime;

pub use config::ChronoFabricConfig;
pub use runtime::ChronoFabricRuntime;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{ChronoFabricConfig, ChronoFabricRuntime};
    pub use anyhow::{Error, Result};
    pub use tracing::{debug, error, info, trace, warn};
}
