//! Load balancing strategies for ADP

use super::{NodeId, ComputeNode};
use csf_core::prelude::*;
use csf_shared_types::PacketType;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Load balancer for distributing work across compute nodes
pub struct LoadBalancer {
    /// Available nodes
    nodes: DashMap<NodeId, Arc<ComputeNode>>,
    
    /// Round-robin counter
    round_robin_counter: AtomicUsize,
    
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    HashBased,
    Adaptive,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(config: &super::AdpConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            round_robin_counter: AtomicUsize::new(0),
            strategy: if config.predictive_scaling {
                LoadBalancingStrategy::Adaptive
            } else {
                LoadBalancingStrategy::LeastLoaded
            },
        }
    }
    
    /// Register a compute node
    pub fn register_node(&self, id: NodeId, node: Arc<ComputeNode>) {
        self.nodes.insert(id, node);
    }
    
    /// Unregister a compute node
    pub fn unregister_node(&self, id: NodeId) {
        self.nodes.remove(&id);
    }
    
    /// Select a node for processing a packet
    pub fn select_node(&self, packet: &BinaryPacket) -> Result<Arc<ComputeNode>> {
        if self.nodes.is_empty() {
            return Err(anyhow::anyhow!("No compute nodes available"));
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(),
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded(),
            LoadBalancingStrategy::HashBased => self.select_hash_based(packet),
            LoadBalancingStrategy::Adaptive => self.select_adaptive(packet),
        }
    }
    
    /// Round-robin selection
    fn select_round_robin(&self) -> Result<Arc<ComputeNode>> {
        let count = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        let index = count % self.nodes.len();
        
        self.nodes.iter()
            .nth(index)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to select node"))
    }
    
    /// Select least loaded node
    fn select_least_loaded(&self) -> Result<Arc<ComputeNode>> {
        self.nodes
            .iter()
            .min_by(|a, b| {
                let a_util = a.value().get_utilization();
                let b_util = b.value().get_utilization();
                a_util
                    .partial_cmp(&b_util)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.value().clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to select node"))
    }
    
    /// Hash-based selection for affinity
    fn select_hash_based(&self, packet: &BinaryPacket) -> Result<Arc<ComputeNode>> {
        let hash = packet.header.packet_id.as_u128() as usize;
        let index = hash % self.nodes.len();
        
        self.nodes.iter()
            .nth(index)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to select node"))
    }
    
    /// Adaptive selection based on packet characteristics
    fn select_adaptive(&self, packet: &BinaryPacket) -> Result<Arc<ComputeNode>> {
        // Check for affinity hint in metadata
        if let Some(affinity) = packet.payload.metadata.get("node_affinity") {
            if let Some(node_id) = affinity.as_u64() {
                if let Some(entry) = self.nodes.iter()
                    .find(|e| e.key().0 == node_id)
                {
                    return Ok(entry.value().clone());
                }
            }
        }
        
        // Check packet priority
        if packet.header.priority >= 128 {
            // High priority - select least loaded
            self.select_least_loaded()
        } else if packet.header.packet_type == PacketType::Data {
            // Data packets - use hash for consistency
            self.select_hash_based(packet)
        } else {
            // Default to round-robin
            self.select_round_robin()
        }
    }
    
    /// Get current load statistics
    pub fn get_load_stats(&self) -> LoadStats {
        let mut total_utilization = 0.0;
        let mut min_utilization = 1.0;
        let mut max_utilization = 0.0;
        let node_count = self.nodes.len();
        
        for entry in self.nodes.iter() {
            let util = entry.value().get_utilization();
            total_utilization += util;
            min_utilization = min_utilization.min(util);
            max_utilization = max_utilization.max(util);
        }
        
        LoadStats {
            node_count,
            avg_utilization: if node_count > 0 { total_utilization / node_count as f64 } else { 0.0 },
            min_utilization,
            max_utilization,
            load_variance: self.calculate_load_variance(total_utilization / node_count as f64),
        }
    }
    
    /// Calculate load variance across nodes
    fn calculate_load_variance(&self, mean: f64) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        
        let variance_sum: f64 = self.nodes.iter()
            .map(|entry| {
                let util = entry.value().get_utilization();
                (util - mean).powi(2)
            })
            .sum();
        
        variance_sum / self.nodes.len() as f64
    }
}

/// Load statistics
#[derive(Debug, Clone)]
pub struct LoadStats {
    pub node_count: usize,
    pub avg_utilization: f64,
    pub min_utilization: f64,
    pub max_utilization: f64,
    pub load_variance: f64,
}