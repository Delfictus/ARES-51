//! Hardware abstraction layer for ARES CSF
//!
//! Provides direct hardware access for real-time performance including:
//! - Hardware timestamps (TSC, HPET)
//! - CPU affinity and frequency control
//! - Cache control and prefetching
//! - NUMA awareness
//! - Hardware performance counters

#![warn(missing_docs)]
#![allow(unsafe_code)] // Required for hardware access

pub mod cache;
pub mod cpu;
pub mod memory;
pub mod timestamp;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "neuromorphic")]
pub mod neuromorphic;

pub use cache::{flush_cache_line, prefetch_data};
pub use cpu::{disable_cpu_scaling, set_cpu_affinity, CpuTopology};
pub use memory::{allocate_huge_pages, pin_memory, MemoryRegion};
pub use timestamp::{hardware_timestamp, HardwareTimer};

/// Initialize hardware abstraction layer
pub fn init() -> anyhow::Result<()> {
    // Check CPU features
    cpu::check_required_features()?;

    // Initialize high-precision timer
    timestamp::init_timer()?;

    // Set up memory allocators
    memory::init_allocators()?;

    Ok(())
}

/// Hardware capabilities
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// Available accelerators
    pub accelerators: Vec<Accelerator>,
}

/// CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Number of physical cores
    pub physical_cores: u32,
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: u32,
    /// CPU frequency in MHz
    pub frequency_mhz: u32,
    /// Cache sizes in bytes
    pub cache_sizes: CacheSizes,
    /// NUMA nodes
    pub numa_nodes: Vec<NumaNode>,
    /// CPU features
    pub features: CpuFeatures,
}

/// Cache sizes
#[derive(Debug, Clone, Copy)]
pub struct CacheSizes {
    /// L1 data cache size
    pub l1d: usize,
    /// L1 instruction cache size
    pub l1i: usize,
    /// L2 cache size
    pub l2: usize,
    /// L3 cache size
    pub l3: usize,
    /// Cache line size
    pub line_size: usize,
}

/// CPU features
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
    /// TSX (Transactional Synchronization Extensions)
    pub tsx: bool,
    /// RDRAND instruction
    pub rdrand: bool,
    /// RDSEED instruction
    pub rdseed: bool,
    /// AES-NI support
    pub aes_ni: bool,
    /// SHA extensions
    pub sha: bool,
}

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: u32,
    /// CPUs in this node
    pub cpus: Vec<u32>,
    /// Memory size in bytes
    pub memory_bytes: usize,
    /// Distance to other nodes
    pub distances: Vec<u32>,
}

/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total physical memory
    pub total_bytes: usize,
    /// Available memory
    pub available_bytes: usize,
    /// Huge page size (2MB or 1GB)
    pub huge_page_size: usize,
    /// Number of NUMA nodes
    pub numa_nodes: u32,
}

/// Hardware accelerator
#[derive(Debug, Clone)]
pub enum Accelerator {
    /// NVIDIA GPU
    Gpu(GpuInfo),
    /// Intel Loihi neuromorphic chip
    Neuromorphic(NeuromorphicInfo),
    /// FPGA
    Fpga(FpgaInfo),
    /// TPU
    Tpu(TpuInfo),
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device ID
    pub id: u32,
    /// Device name
    pub name: String,
    /// Memory in bytes
    pub memory_bytes: usize,
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Number of SMs
    pub multiprocessors: u32,
    /// CUDA cores
    pub cuda_cores: u32,
}

/// Neuromorphic chip information
#[derive(Debug, Clone)]
pub struct NeuromorphicInfo {
    /// Chip ID
    pub id: u32,
    /// Chip type
    pub chip_type: String,
    /// Number of neuromorphic cores
    pub cores: u32,
    /// Synapses per core
    pub synapses_per_core: u32,
}

/// FPGA information
#[derive(Debug, Clone)]
pub struct FpgaInfo {
    /// Device ID
    pub id: u32,
    /// Device model
    pub model: String,
    /// Logic elements
    pub logic_elements: u32,
    /// Block RAM (KB)
    pub block_ram_kb: u32,
}

/// TPU information
#[derive(Debug, Clone)]
pub struct TpuInfo {
    /// Device ID
    pub id: u32,
    /// TPU version
    pub version: u32,
    /// Number of cores
    pub cores: u32,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f32,
}

/// Get system hardware information
pub fn get_hardware_info() -> anyhow::Result<HardwareInfo> {
    Ok(HardwareInfo {
        cpu: cpu::get_cpu_info()?,
        memory: memory::get_memory_info()?,
        accelerators: detect_accelerators()?,
    })
}

fn detect_accelerators() -> anyhow::Result<Vec<Accelerator>> {
    let mut accelerators = Vec::new();

    // Detect GPUs
    #[cfg(feature = "cuda")]
    {
        if let Ok(gpus) = crate::gpu::detect_nvidia_gpus() {
            accelerators.extend(gpus.into_iter().map(Accelerator::Gpu));
        }
    }

    // Detect neuromorphic chips
    #[cfg(feature = "neuromorphic")]
    {
        if let Ok(chips) = crate::neuromorphic::detect_loihi_chips() {
            accelerators.extend(chips.into_iter().map(Accelerator::Neuromorphic));
        }
    }

    Ok(accelerators)
}
