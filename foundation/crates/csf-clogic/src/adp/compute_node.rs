//! Compute node implementation for ADP

use csf_core::prelude::*;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use tokio::sync::mpsc;

/// A single compute node in the ADP cluster
pub struct ComputeNode {
    /// Node ID
    id: super::NodeId,
    
    /// Task queue
    task_queue: mpsc::Sender<ProcessingTask>,
    task_receiver: RwLock<Option<mpsc::Receiver<ProcessingTask>>>,
    
    /// Processing metrics
    processed_count: AtomicU64,
    error_count: AtomicU64,
    total_processing_time: AtomicU64,
    
    /// Node state
    is_running: AtomicBool,
    current_load: AtomicU64,
    
    /// Configuration
    queue_size: usize,
}

struct ProcessingTask {
    packet: PhasePacket,
    result_sender: tokio::sync::oneshot::Sender<Result<PhasePacket>>,
}

impl ComputeNode {
    /// Create a new compute node
    pub fn new(id: super::NodeId, config: &super::AdpConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.task_queue_size);
        
        Self {
            id,
            task_queue: tx,
            task_receiver: RwLock::new(Some(rx)),
            processed_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
            current_load: AtomicU64::new(0),
            queue_size: config.task_queue_size,
        }
    }
    
    /// Get node ID
    pub fn id(&self) -> super::NodeId {
        self.id
    }
    
    /// Start the compute node
    pub async fn start(&self) -> Result<()> {
        if self.is_running.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already running
        }
        
        let mut receiver = self.task_receiver.write().take()
            .ok_or_else(|| anyhow::anyhow!("Node already started"))?;
        
        let processed_count = self.processed_count.clone();
        let error_count = self.error_count.clone();
        let total_processing_time = self.total_processing_time.clone();
        let current_load = self.current_load.clone();
        
        tokio::spawn(async move {
            while let Some(task) = receiver.recv().await {
                let start_time = hardware_timestamp();
                
                // Process the packet
                let result = Self::process_packet_internal(task.packet).await;
                
                // Update metrics
                let processing_time = hardware_timestamp() - start_time;
                total_processing_time.fetch_add(processing_time, Ordering::Relaxed);
                
                match &result {
                    Ok(_) => {
                        processed_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
                
                // Update load estimate
                let queue_len = receiver.len();
                let load = (queue_len as f64 / receiver.max_capacity() as f64 * 100.0) as u64;
                current_load.store(load, Ordering::Relaxed);
                
                // Send result back
                let _ = task.result_sender.send(result);
            }
        });
        
        Ok(())
    }
    
    /// Stop the compute node
    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// Process a packet on this node
    pub async fn process(&self, packet: BinaryPacket) -> Result<BinaryPacket> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Node is not running"));
        }
        
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        let task = ProcessingTask {
            packet,
            result_sender: tx,
        };
        
        self.task_queue.send(task).await
            .map_err(|_| anyhow::anyhow!("Failed to queue task"))?;
        
        rx.await
            .map_err(|_| anyhow::anyhow!("Task processing cancelled"))?
    }
    
    /// Get current utilization (0.0 - 1.0)
    pub fn get_utilization(&self) -> f64 {
        self.current_load.load(Ordering::Relaxed) as f64 / 100.0
    }
    
    /// Get node metrics
    pub fn get_metrics(&self) -> NodeMetrics {
        let processed = self.processed_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        
        let avg_processing_time = if processed > 0 {
            total_time / processed
        } else {
            0
        };
        
        NodeMetrics {
            node_id: self.id,
            processed_count: processed,
            error_count: errors,
            avg_processing_time_ns: avg_processing_time,
            utilization: self.get_utilization(),
        }
    }
    
    /// Internal packet processing logic
    async fn process_packet_internal(mut packet: BinaryPacket) -> Result<BinaryPacket> {
        // Simulate complex processing
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        
        // Add processing metadata
        packet.header.flags |= PacketFlags::PROCESSED;
        packet.payload.metadata.insert(
            "adp_node".to_string(),
            serde_json::json!({
                "processed": true,
                "timestamp": hardware_timestamp(),
            })
        );
        
        Ok(packet)
    }
}

/// Node metrics
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    pub node_id: super::NodeId,
    pub processed_count: u64,
    pub error_count: u64,
    pub avg_processing_time_ns: u64,
    pub utilization: f64,
}