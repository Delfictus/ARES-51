//! Simplified error types for backend implementation

use crate::Backend;

/// Result type for MLIR operations
pub type MlirResult<T> = Result<T, MlirError>;

/// MLIR error types
#[derive(Debug, thiserror::Error)]
pub enum MlirError {
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
    
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Backend-specific errors
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Backend {backend} is not supported")]
    UnsupportedBackend { backend: Backend },
    
    #[error("No healthy backends available")]
    NoHealthyBackends,
    
    #[error("Executor not found for backend {backend}")]
    ExecutorNotFound { backend: Backend },
    
    #[error("Execution failed on backend {backend}: {source}")]
    ExecutionError {
        backend: Backend,
        source: Box<dyn std::error::Error + Send + Sync>,
        fallback_error: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    
    #[error("Execution timeout on backend {backend} after {timeout_seconds}s")]
    ExecutionTimeout {
        backend: Backend,
        timeout_seconds: u64,
    },
    
    #[error("No suitable backend found: {reason}")]
    NoSuitableBackend { reason: String },
    
    #[error("Backend initialization failed for {backend}: {source}")]
    InitializationError {
        backend: Backend,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Backend cleanup failed for {backend}: {source}")]
    CleanupError {
        backend: Backend,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Feature {feature} not enabled")]
    FeatureNotEnabled { feature: String },
    
    #[error("CUDA not available")]
    CudaNotAvailable,
    
    #[error("HIP not available")]
    HipNotAvailable,
    
    #[error("Vulkan not available")]
    VulkanNotAvailable,
    
    #[error("TPU not available")]
    TpuNotAvailable,
    
    #[error("Resource acquisition failed for {resource}: {source}")]
    ResourceAcquisitionError {
        resource: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
}

/// Memory-specific errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Transfer engine not found for backend {backend}")]
    TransferEngineNotFound { backend: Backend },
    
    #[error("Unsupported transfer type")]
    UnsupportedTransferType,
    
    #[error("Memory allocation failed")]
    AllocationFailed,
}