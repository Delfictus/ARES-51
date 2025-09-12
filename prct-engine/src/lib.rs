//! PRCT Engine - Phase Resonance Chromatic-TSP for Protein Folding
//! 
//! This crate implements the revolutionary PRCT algorithm for protein structure
//! prediction using phase resonance dynamics and chromatic graph optimization.
//! 
//! # Zero Drift Implementation
//! This implementation follows strict anti-drift methodology:
//! - NO hardcoded return values
//! - NO random placeholders 
//! - NO simulated data
//! - ALL values computed from real scientific data
//! 
//! Author: Ididia Serfaty
//! Classification: TOP SECRET

pub mod dataset_downloader;
pub mod phase_resonance;
pub mod protein_optimizer;
pub mod validation;
pub mod benchmarks;

pub use dataset_downloader::*;
pub use phase_resonance::*;
pub use protein_optimizer::*;
pub use validation::*;

use std::fmt;
use thiserror::Error;

/// Core error types for PRCT engine operations
#[derive(Error, Debug)]
pub enum PRCTError {
    #[error("Dataset download failed: {0}")]
    DatasetDownload(String),
    
    #[error("Data validation failed: {0}")]
    DataValidation(String),
    
    #[error("Phase resonance computation failed: {0}")]
    PhaseResonance(String),
    
    #[error("Protein optimization failed: {0}")]
    ProteinOptimization(String),
    
    #[error("Performance benchmark failed: {0}")]
    BenchmarkFailure(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type PRCTResult<T> = Result<T, PRCTError>;

/// Configuration for PRCT engine operations
#[derive(Debug, Clone)]
pub struct PRCTConfig {
    /// Base directory for dataset storage
    pub data_directory: std::path::PathBuf,
    
    /// Number of parallel download threads
    pub download_threads: usize,
    
    /// Phase resonance integration tolerance
    pub phase_tolerance: f64,
    
    /// Maximum optimization iterations
    pub max_iterations: usize,
    
    /// Performance benchmarking enabled
    pub enable_benchmarks: bool,
    
    /// Validation level (0=basic, 1=standard, 2=comprehensive)
    pub validation_level: u8,
}

impl Default for PRCTConfig {
    fn default() -> Self {
        Self {
            data_directory: std::path::PathBuf::from("datasets-vault"),
            download_threads: 8,
            phase_tolerance: 1e-8,
            max_iterations: 100_000,
            enable_benchmarks: true,
            validation_level: 2,
        }
    }
}

/// Statistics tracking for PRCT operations
#[derive(Debug, Clone, Default)]
pub struct PRCTStats {
    pub datasets_downloaded: usize,
    pub total_download_size_gb: f64,
    pub proteins_processed: usize,
    pub average_rmsd: f64,
    pub average_folding_time_seconds: f64,
    pub validation_tests_passed: usize,
    pub validation_tests_failed: usize,
}

impl fmt::Display for PRCTStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PRCT Statistics:\n")?;
        write!(f, "  Datasets downloaded: {}\n", self.datasets_downloaded)?;
        write!(f, "  Total data size: {:.1}GB\n", self.total_download_size_gb)?;
        write!(f, "  Proteins processed: {}\n", self.proteins_processed)?;
        write!(f, "  Average RMSD: {:.3}Å\n", self.average_rmsd)?;
        write!(f, "  Average folding time: {:.1}s\n", self.average_folding_time_seconds)?;
        write!(f, "  Validation success rate: {:.1}%\n", 
               100.0 * self.validation_tests_passed as f64 / 
               (self.validation_tests_passed + self.validation_tests_failed).max(1) as f64);
        Ok(())
    }
}