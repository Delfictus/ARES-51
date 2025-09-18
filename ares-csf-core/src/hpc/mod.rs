//! High-performance computing module for CSF.

pub mod distributed_computing;
pub mod gpu_acceleration;
pub mod memory_optimization;
pub mod performance_profiler;
pub mod simd_operations;
pub mod streaming_processor;

// Re-export key types
pub use distributed_computing::{DistributedCompute, HardwareCapabilities};
pub use gpu_acceleration::GPULinearAlgebra;
pub use memory_optimization::{MemoryPool, OptimizedMatrix};
pub use performance_profiler::PerformanceProfiler;
pub use simd_operations::SIMDLinearAlgebra;
pub use streaming_processor::{StreamingBuffer, StreamingProcessor};

/// HPC configuration
#[derive(Debug, Clone)]
pub struct HPCConfiguration {
    /// Number of CPU cores to use
    pub cpu_cores: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Streaming buffer size
    pub stream_buffer_size: usize,
}

impl Default for HPCConfiguration {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            enable_gpu: false,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_simd: true,
            stream_buffer_size: 1024 * 1024, // 1MB
        }
    }
}

/// HPC runtime for managing high-performance operations
pub struct HPCRuntime {
    config: HPCConfiguration,
    #[cfg(feature = "gpu")]
    gpu_context: Option<gpu_acceleration::GpuContext>,
    memory_pool: memory_optimization::MemoryPool,
}

impl HPCRuntime {
    /// Create a new HPC runtime
    pub async fn new(config: &crate::CSFConfig) -> crate::Result<Self> {
        let hpc_config = HPCConfiguration {
            cpu_cores: config.worker_threads,
            enable_gpu: config.enable_gpu,
            memory_pool_size: config.memory_pool_size * 1024 * 1024, // Convert MB to bytes
            enable_simd: true,
            stream_buffer_size: 1024 * 1024,
        };

        let memory_pool = memory_optimization::MemoryPool::new(hpc_config.memory_pool_size);

        Ok(Self {
            config: hpc_config,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            memory_pool,
        })
    }

    /// Get the HPC configuration
    pub fn config(&self) -> &HPCConfiguration {
        &self.config
    }

    /// Get memory pool reference
    pub fn memory_pool(&self) -> &memory_optimization::MemoryPool {
        &self.memory_pool
    }
}