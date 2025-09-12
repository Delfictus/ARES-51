//! High-Performance Computing Module
//!
//! This module provides GPU acceleration, SIMD vectorization, distributed computing,
//! and real-time streaming capabilities for production-scale ARES ChronoFabric deployments.

pub mod distributed_computing;
pub mod gpu_acceleration;
pub mod memory_optimization;
pub mod performance_profiler;
pub mod simd_operations;
pub mod streaming_processor;

pub use distributed_computing::*;
pub use gpu_acceleration::*;
pub use memory_optimization::*;
pub use performance_profiler::*;
pub use simd_operations::*;
pub use streaming_processor::*;

/// High-performance computing configuration
#[derive(Debug, Clone)]
pub struct HPCConfiguration {
    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// GPU device selection (None for auto-select)
    pub gpu_device_id: Option<usize>,

    /// Enable SIMD vectorization
    pub enable_simd: bool,

    /// Number of CPU cores to utilize
    pub cpu_cores: Option<usize>,

    /// Memory pool size for large computations (in GB)
    pub memory_pool_gb: f64,

    /// Enable distributed computing
    pub enable_distributed: bool,

    /// Cluster node addresses for distributed computing
    pub cluster_nodes: Vec<String>,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Streaming buffer size
    pub stream_buffer_size: usize,

    /// Enable data compression
    pub enable_compression: bool,
}

impl Default for HPCConfiguration {
    fn default() -> Self {
        Self {
            enable_gpu: false, // Default disabled for compatibility
            gpu_device_id: None,
            enable_simd: true,
            cpu_cores: None, // Use all available
            memory_pool_gb: 2.0,
            enable_distributed: false,
            cluster_nodes: Vec::new(),
            enable_profiling: false,
            stream_buffer_size: 8192,
            enable_compression: true,
        }
    }
}

/// Hardware capability detection
pub struct HardwareCapabilities {
    pub cpu_cores: usize,
    pub gpu_available: bool,
    pub gpu_memory_gb: Option<f64>,
    pub system_memory_gb: f64,
    pub simd_features: SIMDFeatures,
}

#[derive(Debug, Clone)]
pub struct SIMDFeatures {
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub neon: bool, // ARM NEON
}

impl HardwareCapabilities {
    /// Detect system hardware capabilities
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get();
        let system_memory_gb = Self::detect_system_memory_gb();
        let simd_features = Self::detect_simd_features();

        #[cfg(feature = "gpu")]
        let (gpu_available, gpu_memory_gb) = Self::detect_gpu_capabilities();

        #[cfg(not(feature = "gpu"))]
        let (gpu_available, gpu_memory_gb) = (false, None);

        Self {
            cpu_cores,
            gpu_available,
            gpu_memory_gb,
            system_memory_gb,
            simd_features,
        }
    }

    fn detect_system_memory_gb() -> f64 {
        let sys = sysinfo::System::new_all();
        sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0) // Convert from bytes to GB
    }

    fn detect_simd_features() -> SIMDFeatures {
        SIMDFeatures {
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512: is_x86_feature_detected!("avx512f"),
            sse41: is_x86_feature_detected!("sse4.1"),
            sse42: is_x86_feature_detected!("sse4.2"),
            neon: cfg!(target_arch = "aarch64"),
        }
    }

    #[cfg(feature = "gpu")]
    fn detect_gpu_capabilities() -> (bool, Option<f64>) {
        // GPU detection implementation would go here
        (false, None) // Placeholder
    }

    /// Get optimal configuration based on hardware
    pub fn optimal_config(&self) -> HPCConfiguration {
        HPCConfiguration {
            enable_gpu: self.gpu_available,
            gpu_device_id: None,
            enable_simd: self.simd_features.avx2 || self.simd_features.avx,
            cpu_cores: Some(self.cpu_cores),
            memory_pool_gb: (self.system_memory_gb * 0.25).min(4.0), // 25% of system memory, max 4GB
            enable_distributed: false,
            cluster_nodes: Vec::new(),
            enable_profiling: false,
            stream_buffer_size: if self.system_memory_gb > 16.0 {
                16384
            } else {
                8192
            },
            enable_compression: true,
        }
    }
}

/// Compute device abstraction
pub enum ComputeDevice {
    CPU { cores: usize },
    GPU { device_id: usize, memory_gb: f64 },
    Distributed { nodes: Vec<String> },
}

/// Performance metrics for HPC operations
#[derive(Debug, Clone)]
pub struct HPCMetrics {
    pub computation_time_ms: u64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: Option<f32>,
    pub cpu_utilization_percent: f32,
    pub cache_hit_rate: f32,
}

impl HPCMetrics {
    pub fn new() -> Self {
        Self {
            computation_time_ms: 0,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization_percent: None,
            cpu_utilization_percent: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}
