//! Hardware detection and capability assessment for neuromorphic computing

use anyhow::Result;
use std::process::Command;
use tracing::{debug, info, warn};

use csf_hardware::neuromorphic::detect_loihi_chips;

#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// CPU information
    pub cpu_cores: usize,
    pub cpu_model: String,
    
    /// GPU information
    pub has_cuda_gpu: bool,
    pub gpu_memory_gb: f32,
    pub gpu_model: String,
    
    /// Neuromorphic chip information
    pub has_neuromorphic_chip: bool,
    pub neuromorphic_chips: Vec<NeuromorphicChipInfo>,
    
    /// System memory
    pub system_memory_gb: f32,
    
    /// Platform information
    pub platform: Platform,
}

#[derive(Debug, Clone)]
pub struct NeuromorphicChipInfo {
    pub chip_type: String,
    pub cores: u32,
    pub synapses_per_core: u32,
    pub total_neurons: u32,
}

#[derive(Debug, Clone)]
pub enum Platform {
    Linux,
    MacOS,
    Windows,
    Unknown,
}

impl HardwareCapabilities {
    pub fn summary(&self) -> String {
        format!(
            "CPU: {} cores ({}), GPU: {} ({}GB), Neuromorphic: {}",
            self.cpu_cores,
            self.cpu_model,
            if self.has_cuda_gpu { "CUDA" } else { "None" },
            self.gpu_memory_gb,
            if self.has_neuromorphic_chip { "Available" } else { "None" }
        )
    }
    
    pub fn optimal_threads(&self) -> usize {
        // Reserve 1 core for system, use rest for neuromorphic simulation
        (self.cpu_cores - 1).max(1)
    }
    
    pub fn can_use_gpu_acceleration(&self) -> bool {
        self.has_cuda_gpu && self.gpu_memory_gb >= 4.0
    }
}

pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect all available hardware capabilities
    pub async fn detect() -> Result<HardwareCapabilities> {
        info!("Detecting hardware capabilities for neuromorphic computing");
        
        let cpu_info = Self::detect_cpu().await?;
        let gpu_info = Self::detect_gpu().await?;
        let neuromorphic_info = Self::detect_neuromorphic().await?;
        let memory_info = Self::detect_memory().await?;
        let platform = Self::detect_platform();
        
        let capabilities = HardwareCapabilities {
            cpu_cores: cpu_info.0,
            cpu_model: cpu_info.1,
            has_cuda_gpu: gpu_info.0,
            gpu_memory_gb: gpu_info.1,
            gpu_model: gpu_info.2,
            has_neuromorphic_chip: !neuromorphic_info.is_empty(),
            neuromorphic_chips: neuromorphic_info,
            system_memory_gb: memory_info,
            platform,
        };
        
        info!("Hardware detection complete: {}", capabilities.summary());
        Ok(capabilities)
    }
    
    async fn detect_cpu() -> Result<(usize, String)> {
        let cores = num_cpus::get();
        let model = Self::get_cpu_model().unwrap_or_else(|| "Unknown".to_string());
        
        debug!("Detected CPU: {} cores, model: {}", cores, model);
        Ok((cores, model))
    }
    
    async fn detect_gpu() -> Result<(bool, f32, String)> {
        // Check for NVIDIA GPU using nvidia-smi
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().next() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        let name = parts[0].trim().to_string();
                        let memory_mb: f32 = parts[1].trim().parse().unwrap_or(0.0);
                        let memory_gb = memory_mb / 1024.0;
                        
                        debug!("Detected NVIDIA GPU: {} with {}GB memory", name, memory_gb);
                        return Ok((true, memory_gb, name));
                    }
                }
            },
            _ => {
                debug!("nvidia-smi not available or failed");
            }
        }
        
        // Check for AMD GPU using rocm-smi (if available)
        let output = Command::new("rocm-smi").arg("--showproductname").output();
        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("GPU") {
                    debug!("Detected AMD GPU (ROCm)");
                    return Ok((true, 8.0, "AMD GPU (ROCm)".to_string())); // Estimate
                }
            }
        }
        
        debug!("No GPU detected");
        Ok((false, 0.0, "None".to_string()))
    }
    
    async fn detect_neuromorphic() -> Result<Vec<NeuromorphicChipInfo>> {
        debug!("Detecting neuromorphic chips");
        
        // Use existing CSF hardware detection
        let loihi_chips = detect_loihi_chips().unwrap_or_default();
        
        let mut chips = Vec::new();
        for chip in loihi_chips {
            chips.push(NeuromorphicChipInfo {
                chip_type: chip.chip_type,
                cores: chip.cores,
                synapses_per_core: chip.synapses_per_core,
                total_neurons: chip.cores * chip.synapses_per_core,
            });
        }
        
        if !chips.is_empty() {
            debug!("Detected {} neuromorphic chips", chips.len());
        } else {
            debug!("No neuromorphic chips detected");
        }
        
        Ok(chips)
    }
    
    async fn detect_memory() -> Result<f32> {
        // Get system memory using sysinfo
        use sysinfo::System;
        
        let mut sys = System::new_all();
        sys.refresh_memory();
        
        let total_memory_kb = sys.total_memory();
        let memory_gb = total_memory_kb as f32 / (1024.0 * 1024.0);
        
        debug!("Detected system memory: {:.1}GB", memory_gb);
        Ok(memory_gb)
    }
    
    fn detect_platform() -> Platform {
        match std::env::consts::OS {
            "linux" => Platform::Linux,
            "macos" => Platform::MacOS,
            "windows" => Platform::Windows,
            _ => Platform::Unknown,
        }
    }
    
    fn get_cpu_model() -> Option<String> {
        // Try to get CPU model from /proc/cpuinfo on Linux
        if cfg!(target_os = "linux") {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(model) = line.split(':').nth(1) {
                            return Some(model.trim().to_string());
                        }
                    }
                }
            }
        }
        
        // Fallback: try using system command
        let output = Command::new("lscpu").output();
        if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.starts_with("Model name:") {
                    if let Some(model) = line.split(':').nth(1) {
                        return Some(model.trim().to_string());
                    }
                }
            }
        }
        
        None
    }
}