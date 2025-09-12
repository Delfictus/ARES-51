//! Core error types for the CSF platform.
use std::error::Error as StdError;
use std::fmt;
use std::time::SystemTimeError;

/// The primary error type for all operations in the `csf-core` crate and
/// its dependent crates.
#[derive(Debug)]
pub enum Error {
    /// Error originating from I/O operations.
    Io(std::io::Error),

    /// Error during serialization or deserialization.
    Serialization(String),

    /// The system clock is not functioning correctly.
    SystemTime(SystemTimeError),

    /// A required configuration value is missing or invalid.
    Configuration(String),

    /// A queue is full and cannot accept new items (backpressure).
    QueueFull,

    /// A queue or channel is empty and a receive operation failed.
    QueueEmpty,

    /// The operation could not be completed because the component is shutting down.
    Shutdown,

    /// Invalid data was provided to a function.
    InvalidData(String),

    /// A task failed to execute.
    TaskFailed(String),

    /// A consensus operation failed.
    Consensus(String),

    /// A ledger operation failed.
    Ledger(String),

    /// A generic error with a descriptive message.
    Generic(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Serialization(s) => write!(f, "Serialization error: {s}"),
            Error::SystemTime(e) => write!(f, "System time error: {e}"),
            Error::Configuration(s) => write!(f, "Configuration error: {s}"),
            Error::QueueFull => write!(f, "Queue is full"),
            Error::QueueEmpty => write!(f, "Queue is empty"),
            Error::Shutdown => write!(f, "Component is shutting down"),
            Error::InvalidData(s) => write!(f, "Invalid data: {s}"),
            Error::TaskFailed(s) => write!(f, "Task execution failed: {s}"),
            Error::Consensus(s) => write!(f, "Consensus failure: {s}"),
            Error::Ledger(s) => write!(f, "Ledger operation failed: {s}"),
            Error::Generic(s) => write!(f, "{s}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::SystemTime(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<SystemTimeError> for Error {
    fn from(e: SystemTimeError) -> Self {
        Error::SystemTime(e)
    }
}
