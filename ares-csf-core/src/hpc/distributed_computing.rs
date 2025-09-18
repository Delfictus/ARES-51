//! Distributed computing capabilities for HPC workloads.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Hardware capabilities information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total RAM in bytes
    pub total_memory: u64,
    /// Available GPU devices
    pub gpu_devices: Vec<GpuInfo>,
    /// Network bandwidth in bytes/sec
    pub network_bandwidth: u64,
    /// Storage type and speed
    pub storage_info: StorageInfo,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// Device ID
    pub device_id: u32,
    /// Device name
    pub name: String,
    /// Memory in bytes
    pub memory: u64,
    /// Compute capability
    pub compute_capability: String,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Storage type (SSD, HDD, NVMe)
    pub storage_type: String,
    /// Read speed in bytes/sec
    pub read_speed: u64,
    /// Write speed in bytes/sec
    pub write_speed: u64,
}

impl HardwareCapabilities {
    /// Detect hardware capabilities automatically
    pub fn detect() -> Result<Self> {
        let cpu_cores = num_cpus::get();
        
        let sys = sysinfo::System::new_all();
        let total_memory = sys.total_memory() * 1024; // Convert KB to bytes
        
        // Placeholder for GPU detection
        let gpu_devices = Vec::new();
        
        // Estimate network bandwidth (placeholder)
        let network_bandwidth = 1_000_000_000; // 1 Gbps
        
        let storage_info = StorageInfo {
            storage_type: "Unknown".to_string(),
            read_speed: 500_000_000,   // 500 MB/s estimate
            write_speed: 400_000_000,  // 400 MB/s estimate
        };
        
        Ok(Self {
            cpu_cores,
            total_memory,
            gpu_devices,
            network_bandwidth,
            storage_info,
        })
    }

    /// Calculate compute score
    pub fn compute_score(&self) -> f64 {
        let cpu_score = self.cpu_cores as f64 * 1.0;
        let memory_score = (self.total_memory as f64 / 1_000_000_000.0) * 0.5; // GB * 0.5
        let gpu_score = self.gpu_devices.len() as f64 * 10.0;
        
        cpu_score + memory_score + gpu_score
    }
}

/// Distributed compute interface
#[async_trait::async_trait]
pub trait DistributedCompute: Send + Sync {
    /// Submit a computation task
    async fn submit_task(&self, task: ComputeTask) -> Result<TaskHandle>;
    
    /// Get task result
    async fn get_result(&self, handle: TaskHandle) -> Result<ComputeResult>;
    
    /// Cancel a running task
    async fn cancel_task(&self, handle: TaskHandle) -> Result<()>;
    
    /// Get cluster status
    async fn cluster_status(&self) -> Result<ClusterStatus>;
}

/// A computation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeTask {
    /// Unique task identifier
    pub id: String,
    /// Task type
    pub task_type: String,
    /// Input data
    pub input_data: Vec<u8>,
    /// Resource requirements
    pub requirements: ResourceRequirements,
    /// Priority level
    pub priority: crate::types::Priority,
}

/// Resource requirements for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Required CPU cores
    pub cpu_cores: usize,
    /// Required memory in bytes
    pub memory_bytes: u64,
    /// Requires GPU
    pub requires_gpu: bool,
    /// Estimated runtime in seconds
    pub estimated_runtime: u64,
}

/// Handle for tracking task execution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskHandle {
    /// Task ID
    pub task_id: String,
    /// Submission timestamp
    pub submitted_at: crate::types::Timestamp,
}

/// Result of a computation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResult {
    /// Task handle
    pub handle: TaskHandle,
    /// Execution status
    pub status: TaskStatus,
    /// Output data
    pub output_data: Vec<u8>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Task execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is queued
    Queued,
    /// Task is running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed(String),
    /// Task was cancelled
    Cancelled,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Start time
    pub start_time: crate::types::Timestamp,
    /// End time
    pub end_time: Option<crate::types::Timestamp>,
    /// CPU time used
    pub cpu_time_ms: u64,
    /// Memory peak usage
    pub peak_memory_bytes: u64,
    /// Network I/O
    pub network_io_bytes: u64,
}

impl ExecutionStats {
    /// Calculate total execution time
    pub fn execution_time(&self) -> Option<std::time::Duration> {
        self.end_time.map(|end| end.duration_since(self.start_time))
    }
}

/// Cluster status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    /// Total nodes in cluster
    pub total_nodes: usize,
    /// Active nodes
    pub active_nodes: usize,
    /// Total CPU cores
    pub total_cpu_cores: usize,
    /// Available CPU cores
    pub available_cpu_cores: usize,
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Queued tasks
    pub queued_tasks: usize,
    /// Running tasks
    pub running_tasks: usize,
}

impl ClusterStatus {
    /// Calculate CPU utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        if self.total_cpu_cores > 0 {
            let used_cores = self.total_cpu_cores - self.available_cpu_cores;
            (used_cores as f64 / self.total_cpu_cores as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory_bytes > 0 {
            let used_memory = self.total_memory_bytes - self.available_memory_bytes;
            (used_memory as f64 / self.total_memory_bytes as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Local distributed compute implementation
pub struct LocalDistributedCompute {
    capabilities: HardwareCapabilities,
    task_queue: std::sync::Arc<tokio::sync::Mutex<Vec<ComputeTask>>>,
    results: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, ComputeResult>>>,
}

impl LocalDistributedCompute {
    /// Create a new local distributed compute instance
    pub fn new() -> Result<Self> {
        let capabilities = HardwareCapabilities::detect()?;
        
        Ok(Self {
            capabilities,
            task_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            results: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        })
    }

    /// Get hardware capabilities
    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }
}

#[async_trait::async_trait]
impl DistributedCompute for LocalDistributedCompute {
    async fn submit_task(&self, task: ComputeTask) -> Result<TaskHandle> {
        let handle = TaskHandle {
            task_id: task.id.clone(),
            submitted_at: crate::types::Timestamp::now(),
        };
        
        {
            let mut queue = self.task_queue.lock().await;
            queue.push(task);
        }
        
        Ok(handle)
    }

    async fn get_result(&self, handle: TaskHandle) -> Result<ComputeResult> {
        let results = self.results.lock().await;
        results
            .get(&handle.task_id)
            .cloned()
            .ok_or_else(|| Error::task_failed("Task result not found".to_string()))
    }

    async fn cancel_task(&self, handle: TaskHandle) -> Result<()> {
        // Simple implementation: mark as cancelled in results
        let mut results = self.results.lock().await;
        let result = ComputeResult {
            handle,
            status: TaskStatus::Cancelled,
            output_data: Vec::new(),
            stats: ExecutionStats {
                start_time: crate::types::Timestamp::now(),
                end_time: Some(crate::types::Timestamp::now()),
                cpu_time_ms: 0,
                peak_memory_bytes: 0,
                network_io_bytes: 0,
            },
        };
        results.insert(result.handle.task_id.clone(), result);
        Ok(())
    }

    async fn cluster_status(&self) -> Result<ClusterStatus> {
        let queue = self.task_queue.lock().await;
        let results = self.results.lock().await;
        
        let running_tasks = results
            .values()
            .filter(|r| matches!(r.status, TaskStatus::Running))
            .count();
        
        Ok(ClusterStatus {
            total_nodes: 1,
            active_nodes: 1,
            total_cpu_cores: self.capabilities.cpu_cores,
            available_cpu_cores: self.capabilities.cpu_cores.saturating_sub(running_tasks),
            total_memory_bytes: self.capabilities.total_memory,
            available_memory_bytes: self.capabilities.total_memory / 2, // Estimate
            queued_tasks: queue.len(),
            running_tasks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_capabilities() {
        let caps = HardwareCapabilities::detect().unwrap();
        assert!(caps.cpu_cores > 0);
        assert!(caps.total_memory > 0);
        assert!(caps.compute_score() > 0.0);
    }

    #[test]
    fn test_cluster_status() {
        let status = ClusterStatus {
            total_nodes: 4,
            active_nodes: 4,
            total_cpu_cores: 32,
            available_cpu_cores: 16,
            total_memory_bytes: 64 * 1024 * 1024 * 1024, // 64 GB
            available_memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            queued_tasks: 5,
            running_tasks: 10,
        };

        assert_eq!(status.cpu_utilization(), 50.0);
        assert_eq!(status.memory_utilization(), 50.0);
    }

    #[tokio::test]
    async fn test_local_distributed_compute() {
        let compute = LocalDistributedCompute::new().unwrap();
        
        let task = ComputeTask {
            id: "test-task".to_string(),
            task_type: "matrix-multiply".to_string(),
            input_data: vec![1, 2, 3, 4],
            requirements: ResourceRequirements {
                cpu_cores: 1,
                memory_bytes: 1024,
                requires_gpu: false,
                estimated_runtime: 10,
            },
            priority: crate::types::Priority::Normal,
        };

        let handle = compute.submit_task(task).await.unwrap();
        assert_eq!(handle.task_id, "test-task");

        let status = compute.cluster_status().await.unwrap();
        assert_eq!(status.queued_tasks, 1);
    }
}