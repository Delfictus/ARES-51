//! Core error types for the ARES CSF platform.

use thiserror::Error;

/// The primary error type for all operations in the CSF core crate.
///
/// This error type encompasses all possible failure modes in the CSF system,
/// from low-level I/O operations to high-level quantum computation failures.
#[derive(Debug, Error)]
pub enum Error {
    /// Error originating from I/O operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error during serialization or deserialization.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// The system clock is not functioning correctly.
    #[error("System time error: {0}")]
    SystemTime(#[from] std::time::SystemTimeError),

    /// A required configuration value is missing or invalid.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// A queue is full and cannot accept new items (backpressure).
    #[error("Queue is full")]
    QueueFull,

    /// A queue or channel is empty and a receive operation failed.
    #[error("Queue is empty")]
    QueueEmpty,

    /// The operation could not be completed because the component is shutting down.
    #[error("Component is shutting down")]
    Shutdown,

    /// Invalid data was provided to a function.
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// A task failed to execute.
    #[error("Task execution failed: {0}")]
    TaskFailed(String),

    /// A consensus operation failed.
    #[error("Consensus failure: {0}")]
    Consensus(String),

    /// A ledger operation failed.
    #[error("Ledger operation failed: {0}")]
    Ledger(String),

    /// Tensor operation error.
    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    /// Quantum computation error.
    #[error("Quantum computation failed: {0}")]
    QuantumError(String),

    /// HPC runtime error.
    #[error("HPC runtime error: {0}")]
    HpcError(String),

    /// GPU acceleration error.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Memory allocation error.
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// Temporal coherence violation.
    #[error("Temporal coherence violated: {0}")]
    TemporalError(String),

    /// Phase synchronization error.
    #[error("Phase synchronization failed: {0}")]
    PhaseError(String),

    /// Variational optimization error.
    #[error("Variational optimization failed: {0}")]
    VariationalError(String),

    /// Enterprise integration error.
    #[error("Enterprise integration error: {0}")]
    EnterpriseError(String),

    /// Trading engine error.
    #[error("Trading engine error: {0}")]
    TradingError(String),

    /// Data processing error.
    #[error("Data processing error: {0}")]
    DataError(String),

    /// Network communication error.
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Authentication or authorization error.
    #[error("Authentication error: {0}")]
    AuthError(String),

    /// Resource exhaustion error.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Timeout error.
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// A generic error with a descriptive message.
    #[error("{0}")]
    Generic(String),
}

impl Error {
    /// Create a new serialization error.
    pub fn serialization<S: Into<String>>(msg: S) -> Self {
        Self::Serialization(msg.into())
    }

    /// Create a new configuration error.
    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        Self::Configuration(msg.into())
    }

    /// Create a new invalid data error.
    pub fn invalid_data<S: Into<String>>(msg: S) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Create a new task failed error.
    pub fn task_failed<S: Into<String>>(msg: S) -> Self {
        Self::TaskFailed(msg.into())
    }

    /// Create a new consensus error.
    pub fn consensus<S: Into<String>>(msg: S) -> Self {
        Self::Consensus(msg.into())
    }

    /// Create a new ledger error.
    pub fn ledger<S: Into<String>>(msg: S) -> Self {
        Self::Ledger(msg.into())
    }

    /// Create a new tensor error.
    pub fn tensor<S: Into<String>>(msg: S) -> Self {
        Self::TensorError(msg.into())
    }

    /// Create a new quantum error.
    pub fn quantum<S: Into<String>>(msg: S) -> Self {
        Self::QuantumError(msg.into())
    }

    /// Create a new HPC error.
    pub fn hpc<S: Into<String>>(msg: S) -> Self {
        Self::HpcError(msg.into())
    }

    /// Create a new GPU error.
    pub fn gpu<S: Into<String>>(msg: S) -> Self {
        Self::GpuError(msg.into())
    }

    /// Create a new memory error.
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        Self::MemoryError(msg.into())
    }

    /// Create a new temporal error.
    pub fn temporal<S: Into<String>>(msg: S) -> Self {
        Self::TemporalError(msg.into())
    }

    /// Create a new phase error.
    pub fn phase<S: Into<String>>(msg: S) -> Self {
        Self::PhaseError(msg.into())
    }

    /// Create a new variational error.
    pub fn variational<S: Into<String>>(msg: S) -> Self {
        Self::VariationalError(msg.into())
    }

    /// Create a new enterprise error.
    pub fn enterprise<S: Into<String>>(msg: S) -> Self {
        Self::EnterpriseError(msg.into())
    }

    /// Create a new trading error.
    pub fn trading<S: Into<String>>(msg: S) -> Self {
        Self::TradingError(msg.into())
    }

    /// Create a new data error.
    pub fn data<S: Into<String>>(msg: S) -> Self {
        Self::DataError(msg.into())
    }

    /// Create a new network error.
    pub fn network<S: Into<String>>(msg: S) -> Self {
        Self::NetworkError(msg.into())
    }

    /// Create a new authentication error.
    pub fn auth<S: Into<String>>(msg: S) -> Self {
        Self::AuthError(msg.into())
    }

    /// Create a new resource exhausted error.
    pub fn resource_exhausted<S: Into<String>>(msg: S) -> Self {
        Self::ResourceExhausted(msg.into())
    }

    /// Create a new timeout error.
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Self::Timeout(msg.into())
    }

    /// Create a new generic error.
    pub fn generic<S: Into<String>>(msg: S) -> Self {
        Self::Generic(msg.into())
    }
}

// Additional conversions for common error types
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Self::NetworkError(e.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Self::Generic(e.to_string())
    }
}

/// Result type alias for CSF operations
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::configuration("Missing parameter");
        assert!(err.to_string().contains("Configuration error"));
        assert!(err.to_string().contains("Missing parameter"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let csf_err: Error = io_err.into();
        assert!(matches!(csf_err, Error::Io(_)));
    }

    #[test]
    fn test_tensor_error() {
        let err = Error::tensor("Matrix not invertible");
        assert!(err.to_string().contains("Tensor operation failed"));
    }

    #[test]
    fn test_quantum_error() {
        let err = Error::quantum("Decoherence detected");
        assert!(err.to_string().contains("Quantum computation failed"));
    }
}