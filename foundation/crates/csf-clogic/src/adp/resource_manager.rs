//! Resource management for ADP

use std::sync::atomic::{AtomicU64, Ordering};
use sysinfo::{System, SystemExt, ProcessorExt};
use parking_lot::RwLock;

/// Resource manager for monitoring and managing system resources
pub struct ResourceManager {
    /// System information
    system: RwLock<System>,
    
    /// Resource limits
    limits: ResourceLimits,
    
    /// Current resource usage
    cpu_usage: AtomicU64,
    memory_usage: AtomicU64,
    
    /// Resource allocation tracking
    allocated_cpu: AtomicU64,
    allocated_memory: AtomicU64,
}

#[derive(Debug, Clone)]
struct ResourceLimits {
    max_cpu_percent: f64,
    max_memory_bytes: u64,
    reserved_cpu_percent: f64,
    reserved_memory_bytes: u64,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new(config: &super::AdpConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let total_memory = system.total_memory() * 1024; // Convert to bytes
        let cpu_count = system.processors().len();
        
        let limits = ResourceLimits {
            max_cpu_percent: 80.0, // Use up to 80% of CPU
            max_memory_bytes: (total_memory as f64 * 0.8) as u64,
            reserved_cpu_percent: 20.0, // Keep 20% reserved
            reserved_memory_bytes: (total_memory as f64 * 0.2) as u64,
        };
        
        Self {
            system: RwLock::new(system),
            limits,
            cpu_usage: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            allocated_cpu: AtomicU64::new(0),
            allocated_memory: AtomicU64::new(0),
        }
    }
    
    /// Update resource usage statistics
    pub fn update_usage(&self) {
        let mut system = self.system.write();
        system.refresh_cpu();
        system.refresh_memory();
        
        // Calculate CPU usage
        let cpu_usage = system.global_processor_info().cpu_usage() as u64;
        self.cpu_usage.store(cpu_usage, Ordering::Relaxed);
        
        // Calculate memory usage
        let used_memory = system.used_memory() * 1024; // Convert to bytes
        self.memory_usage.store(used_memory, Ordering::Relaxed);
    }
    
    /// Check if resources are available for a new node
    pub fn can_allocate_node(&self, estimated_cpu: f64, estimated_memory: u64) -> bool {
        let current_cpu = self.cpu_usage.load(Ordering::Relaxed) as f64;
        let current_memory = self.memory_usage.load(Ordering::Relaxed);
        
        let projected_cpu = current_cpu + estimated_cpu;
        let projected_memory = current_memory + estimated_memory;
        
        projected_cpu <= self.limits.max_cpu_percent
            && projected_memory <= self.limits.max_memory_bytes
    }
    
    /// Allocate resources for a node
    pub fn allocate_resources(&self, cpu_percent: f64, memory_bytes: u64) -> Result<ResourceAllocation> {
        if !self.can_allocate_node(cpu_percent, memory_bytes) {
            return Err(anyhow::anyhow!("Insufficient resources"));
        }
        
        let cpu_units = (cpu_percent * 100.0) as u64;
        self.allocated_cpu.fetch_add(cpu_units, Ordering::Relaxed);
        self.allocated_memory.fetch_add(memory_bytes, Ordering::Relaxed);
        
        Ok(ResourceAllocation {
            cpu_units,
            memory_bytes,
        })
    }
    
    /// Release allocated resources
    pub fn release_resources(&self, allocation: ResourceAllocation) {
        self.allocated_cpu.fetch_sub(allocation.cpu_units, Ordering::Relaxed);
        self.allocated_memory.fetch_sub(allocation.memory_bytes, Ordering::Relaxed);
    }
    
    /// Get current resource status
    pub fn get_status(&self) -> ResourceStatus {
        self.update_usage();
        
        let cpu_usage = self.cpu_usage.load(Ordering::Relaxed) as f64;
        let memory_usage = self.memory_usage.load(Ordering::Relaxed);
        let allocated_cpu = self.allocated_cpu.load(Ordering::Relaxed) as f64 / 100.0;
        let allocated_memory = self.allocated_memory.load(Ordering::Relaxed);
        
        ResourceStatus {
            cpu_usage_percent: cpu_usage,
            memory_usage_bytes: memory_usage,
            allocated_cpu_percent: allocated_cpu,
            allocated_memory_bytes: allocated_memory,
            available_cpu_percent: self.limits.max_cpu_percent - cpu_usage,
            available_memory_bytes: self.limits.max_memory_bytes - memory_usage,
        }
    }
    
    /// Predict future resource needs
    pub fn predict_resource_needs(&self, time_horizon_ms: u64) -> ResourcePrediction {
        // Simple linear prediction based on current trends
        // In a real implementation, this would use more sophisticated models
        
        let status = self.get_status();
        let growth_rate = 0.1; // 10% growth per time unit
        
        let predicted_cpu = status.cpu_usage_percent * (1.0 + growth_rate);
        let predicted_memory = (status.memory_usage_bytes as f64 * (1.0 + growth_rate)) as u64;
        
        ResourcePrediction {
            time_horizon_ms,
            predicted_cpu_percent: predicted_cpu,
            predicted_memory_bytes: predicted_memory,
            confidence: 0.7, // Medium confidence
        }
    }
}

/// Resource allocation handle
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    cpu_units: u64, // CPU percentage * 100
    memory_bytes: u64,
}

/// Current resource status
#[derive(Debug, Clone)]
pub struct ResourceStatus {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub allocated_cpu_percent: f64,
    pub allocated_memory_bytes: u64,
    pub available_cpu_percent: f64,
    pub available_memory_bytes: u64,
}

/// Resource usage prediction
#[derive(Debug, Clone)]
pub struct ResourcePrediction {
    pub time_horizon_ms: u64,
    pub predicted_cpu_percent: f64,
    pub predicted_memory_bytes: u64,
    pub confidence: f64,
}