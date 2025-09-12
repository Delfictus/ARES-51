//! Error types for the neuromorphic CLI

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Neuromorphic system error: {0}")]
    Neuromorphic(String),
    
    #[error("Backend initialization failed: {0}")]
    BackendInit(String),
    
    #[error("Natural language processing error: {0}")]
    NLP(String),
    
    #[error("Learning system error: {0}")]
    Learning(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Hardware detection failed: {0}")]
    Hardware(String),
    
    #[error("Python integration error: {0}")]
    Python(#[from] pyo3::PyErr),
    
    #[error("IO operation failed: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    
    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),
    
    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },
    
    #[error("Command execution failed: {message}")]
    CommandFailed { message: String },
    
    #[error("Timeout: operation took longer than {seconds}s")]
    Timeout { seconds: u64 },
    
    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: String },
}

pub type Result<T> = std::result::Result<T, CliError>;

impl CliError {
    /// Create a neuromorphic system error
    pub fn neuromorphic<S: Into<String>>(msg: S) -> Self {
        Self::Neuromorphic(msg.into())
    }
    
    /// Create a backend initialization error
    pub fn backend_init<S: Into<String>>(msg: S) -> Self {
        Self::BackendInit(msg.into())
    }
    
    /// Create an NLP processing error
    pub fn nlp<S: Into<String>>(msg: S) -> Self {
        Self::NLP(msg.into())
    }
    
    /// Create a learning system error
    pub fn learning<S: Into<String>>(msg: S) -> Self {
        Self::Learning(msg.into())
    }
    
    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a hardware detection error
    pub fn hardware<S: Into<String>>(msg: S) -> Self {
        Self::Hardware(msg.into())
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Neuromorphic(_) => true,
            Self::NLP(_) => true,
            Self::Learning(_) => true,
            Self::Python(_) => true,
            Self::Timeout { .. } => true,
            Self::CommandFailed { .. } => true,
            _ => false,
        }
    }
}