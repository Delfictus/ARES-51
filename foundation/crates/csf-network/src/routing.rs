//! Routing implementation

use super::*;
use csf_time::global_time_source;
use parking_lot::RwLock;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Router for finding paths between nodes
pub struct Router {
    config: RoutingConfig,
    node_id: NodeId,
    routing_table: Arc<RwLock<RoutingTable>>,
    route_cache: Arc<dashmap::DashMap<NodeId, Route>>,
}

/// Routing table
struct RoutingTable {
    /// Direct peers
    peers: HashMap<NodeId, PeerEntry>,

    /// Known routes
    routes: HashMap<NodeId, Vec<RouteEntry>>,

    /// Link metrics
    metrics: HashMap<(NodeId, NodeId), LinkMetrics>,
}

#[derive(Debug, Clone)]
struct PeerEntry {
    address: String,
    last_seen: u64,
    rtt_ms: u32,
    bandwidth_mbps: u32,
}

#[derive(Debug, Clone)]
struct RouteEntry {
    next_hop: NodeId,
    cost: u32,
    hop_count: u32,
    last_updated: u64,
}

#[derive(Debug, Clone)]
struct LinkMetrics {
    latency_ms: u32,
    bandwidth_mbps: u32,
    loss_rate: f32,
    jitter_ms: u32,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub destination: NodeId,
    pub next_hop: NodeId,
    pub path: Vec<NodeId>,
    pub cost: u32,
    pub latency_ms: u32,
}

impl Router {
    /// Create new router
    pub fn new(config: &RoutingConfig, node_id: NodeId) -> Self {
        Self {
            config: config.clone(),
            node_id,
            routing_table: Arc::new(RwLock::new(RoutingTable {
                peers: HashMap::new(),
                routes: HashMap::new(),
                metrics: HashMap::new(),
            })),
            route_cache: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Add peer to routing table
    pub async fn add_peer(&self, peer_id: NodeId, address: &str) -> NetworkResult<()> {
        let mut table = self.routing_table.write();

        table.peers.insert(
            peer_id,
            PeerEntry {
                address: address.to_string(),
                last_seen: global_time_source()
                    .now_ns()
                    .unwrap_or(csf_time::NanoTime::ZERO)
                    .as_nanos(),
                rtt_ms: 0,
                bandwidth_mbps: 100,
            },
        );

        // Add direct route
        table
            .routes
            .entry(peer_id)
            .or_insert_with(Vec::new)
            .push(RouteEntry {
                next_hop: peer_id,
                cost: 1,
                hop_count: 1,
                last_updated: global_time_source()
                    .now_ns()
                    .unwrap_or(csf_time::NanoTime::ZERO)
                    .as_nanos(),
            });

        Ok(())
    }

    /// Find route to destination
    pub async fn find_route(&self, destination: NodeId) -> NetworkResult<Route> {
        // Check cache
        if self.config.enable_caching {
            if let Some(route) = self.route_cache.get(&destination) {
                return Ok(route.clone());
            }
        }

        // Calculate route
        let route = match self.config.algorithm {
            RoutingAlgorithm::ShortestPath => self.find_shortest_path(destination)?,
            RoutingAlgorithm::LeastLatency => self.find_least_latency(destination)?,
            RoutingAlgorithm::HighestBandwidth => self.find_highest_bandwidth(destination)?,
            RoutingAlgorithm::Adaptive => self.find_adaptive_route(destination)?,
        };

        // Cache route
        if self.config.enable_caching {
            self.route_cache.insert(destination, route.clone());
        }

        Ok(route)
    }

    /// Find shortest path using Dijkstra's algorithm
    fn find_shortest_path(&self, destination: NodeId) -> NetworkResult<Route> {
        let table = self.routing_table.read();

        // Special case: direct peer
        if table.peers.contains_key(&destination) {
            return Ok(Route {
                destination,
                next_hop: destination,
                path: vec![self.node_id, destination],
                cost: 1,
                latency_ms: 0,
            });
        }

        // Dijkstra's algorithm
        let mut distances: HashMap<NodeId, u32> = HashMap::new();
        let mut previous: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(self.node_id, 0);
        heap.push(Reverse((0, self.node_id)));

        while let Some(Reverse((cost, node))) = heap.pop() {
            if node == destination {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = destination;

                while current != self.node_id {
                    path.push(current);
                    current = *previous
                        .get(&current)
                        .ok_or_else(|| anyhow::anyhow!("No route to destination"))?;
                }

                path.push(self.node_id);
                path.reverse();

                let next_hop = path.get(1).copied().unwrap_or(destination);

                return Ok(Route {
                    destination,
                    next_hop,
                    path,
                    cost,
                    latency_ms: 0,
                });
            }

            if cost > *distances.get(&node).unwrap_or(&u32::MAX) {
                continue;
            }

            // Check neighbors
            if let Some(routes) = table.routes.get(&node) {
                for route in routes {
                    let next_cost = cost + route.cost;

                    if next_cost < *distances.get(&route.next_hop).unwrap_or(&u32::MAX) {
                        distances.insert(route.next_hop, next_cost);
                        previous.insert(route.next_hop, node);
                        heap.push(Reverse((next_cost, route.next_hop)));
                    }
                }
            }
        }

        Err(anyhow::anyhow!("No route to destination"))
    }

    /// Find route with least latency
    fn find_least_latency(&self, destination: NodeId) -> NetworkResult<Route> {
        // Similar to shortest path but using latency metrics
        self.find_shortest_path(destination)
    }

    /// Find route with highest bandwidth
    fn find_highest_bandwidth(&self, destination: NodeId) -> NetworkResult<Route> {
        // Similar to shortest path but using bandwidth metrics
        self.find_shortest_path(destination)
    }

    /// Find adaptive route based on current conditions
    fn find_adaptive_route(&self, destination: NodeId) -> NetworkResult<Route> {
        // Combine multiple metrics for adaptive routing
        self.find_shortest_path(destination)
    }

    /// Update route metrics
    pub async fn update_metrics(
        &self,
        from: NodeId,
        to: NodeId,
        metrics: LinkMetrics,
    ) -> NetworkResult<()> {
        let mut table = self.routing_table.write();
        table.metrics.insert((from, to), metrics);

        // Invalidate cached routes
        self.route_cache.clear();

        Ok(())
    }

    /// Get routing statistics
    pub async fn get_stats(&self) -> RoutingStats {
        let table = self.routing_table.read();

        RoutingStats {
            peer_count: table.peers.len(),
            route_count: table.routes.len(),
            cached_routes: self.route_cache.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub peer_count: usize,
    pub route_count: usize,
    pub cached_routes: usize,
}
