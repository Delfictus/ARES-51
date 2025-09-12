//! Network protocol implementation for ARES CSF
//!
//! Provides high-performance, secure network communication with
//! support for multiple transport protocols and peer discovery.

use csf_bus::packet::PhasePacket;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod compression;
pub mod discovery;
pub mod protocol;
pub mod quic;
pub mod routing;
pub mod security;
pub mod transport;

pub use discovery::Discovery;
pub use protocol::PeerInfo;
pub use protocol::{Protocol, ProtocolMessage};
pub use routing::{Route, Router};
pub use transport::{Connection, Transport, TransportConfig};

/// Result type for network operations
pub type NetworkResult<T> = std::result::Result<T, anyhow::Error>;

/// Network node configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkConfig {
    /// Unique node identifier
    pub node_id: NodeId,

    /// Addresses to listen on (e.g., ip:port)
    pub listen_addrs: Vec<String>,

    /// Transport configuration
    pub transport: TransportConfig,

    /// Discovery configuration
    pub discovery: DiscoveryConfig,

    /// Security configuration
    pub security: SecurityConfig,

    /// Routing configuration
    pub routing: RoutingConfig,

    /// Compression configuration
    pub compression: CompressionConfig,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct NodeId(pub u64);

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        if bytes.len() >= 8 {
            array.copy_from_slice(&bytes[..8]);
        }
        Self(u64::from_be_bytes(array))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryConfig {
    /// Enable mDNS discovery
    pub enable_mdns: bool,

    /// Enable DHT discovery
    pub enable_dht: bool,

    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,

    /// Discovery interval in seconds
    pub discovery_interval: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecurityConfig {
    /// Enable TLS for transport
    pub enable_tls: bool,

    /// Path to TLS certificate (PEM)
    pub cert_path: Option<String>,

    /// Path to TLS private key (PEM)
    pub key_path: Option<String>,

    /// Enable payload encryption
    pub enable_encryption: bool,

    /// Enable authentication/signing
    pub enable_auth: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingConfig {
    /// Routing algorithm selection
    pub algorithm: RoutingAlgorithm,

    /// Max hop count allowed
    pub max_hops: u32,

    /// Route timeout (ms)
    pub route_timeout_ms: u64,

    /// Enable route caching
    pub enable_caching: bool,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum RoutingAlgorithm {
    ShortestPath,
    LeastLatency,
    HighestBandwidth,
    Adaptive,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level (1-9)
    pub level: u32,

    /// Min size to compress (bytes)
    pub min_size: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum CompressionAlgorithm {
    Lz4,
    Zstd,
    None,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId::new(),
            listen_addrs: vec!["0.0.0.0:7878".to_string()],
            transport: TransportConfig::default(),
            discovery: DiscoveryConfig {
                enable_mdns: true,
                enable_dht: true,
                bootstrap_nodes: Vec::new(),
                discovery_interval: 30,
            },
            security: SecurityConfig {
                enable_tls: true,
                cert_path: None,
                key_path: None,
                enable_encryption: true,
                enable_auth: true,
            },
            routing: RoutingConfig {
                algorithm: RoutingAlgorithm::Adaptive,
                max_hops: 10,
                route_timeout_ms: 5000,
                enable_caching: true,
            },
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Lz4,
                level: 3,
                min_size: 1024,
            },
        }
    }
}

/// Network node implementation
pub struct NetworkNode {
    /// Configuration
    config: NetworkConfig,

    /// Transport layer
    transport: Arc<Transport>,

    /// Discovery service
    discovery: Arc<Discovery>,

    /// Router
    router: Arc<Router>,

    /// Protocol handler
    protocol: Arc<Protocol>,

    /// Active connections
    connections: Arc<RwLock<std::collections::HashMap<NodeId, Arc<Connection>>>>,

    /// Node state
    state: Arc<RwLock<NodeState>>,

    /// Metrics
    metrics: Arc<NetworkMetrics>,
}

#[derive(Debug, Default)]
struct NodeState {
    /// Is node running
    running: bool,

    /// Connected peers
    peers: Vec<NodeId>,

    /// Network statistics
    stats: NetworkStats,
}

#[derive(Debug, Default, Clone)]
pub struct NetworkStats {
    /// Packets sent
    pub packets_sent: u64,

    /// Packets received
    pub packets_received: u64,

    /// Bytes sent
    pub bytes_sent: u64,

    /// Bytes received
    pub bytes_received: u64,

    /// Active connections
    pub active_connections: u32,

    /// Failed connections
    pub failed_connections: u32,
}

impl NetworkNode {
    /// Create a new network node
    pub async fn new(
        config: NetworkConfig,
        bus: Arc<csf_bus::PhaseCoherenceBus>,
    ) -> NetworkResult<Self> {
        // Initialize transport
        let transport = Arc::new(Transport::new(&config.transport).await?);

        // Initialize discovery
        let discovery = Arc::new(Discovery::new(&config.discovery, config.node_id).await?);

        // Initialize router
        let router = Arc::new(Router::new(&config.routing, config.node_id));

        // Initialize protocol
        let protocol = Arc::new(Protocol::new(config.node_id, bus));

        // Initialize metrics
        let metrics = Arc::new(NetworkMetrics::new()?);

        Ok(Self {
            config,
            transport,
            discovery,
            router,
            protocol,
            connections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            state: Arc::new(RwLock::new(NodeState::default())),
            metrics,
        })
    }

    /// Start the network node
    pub async fn start(&self) -> NetworkResult<()> {
        let mut state = self.state.write().await;
        if state.running {
            return Ok(());
        }

        // Start transport listeners
        for addr in &self.config.listen_addrs {
            self.transport.listen(addr).await?;
        }

        // Start discovery
        self.discovery.start().await?;

        // Start accepting connections
        let self_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            self_clone.accept_loop().await;
        });

        state.running = true;
        Ok(())
    }

    /// Stop the network node
    pub async fn stop(&self) -> NetworkResult<()> {
        let mut state = self.state.write().await;
        if !state.running {
            return Ok(());
        }

        // Stop discovery
        self.discovery.stop().await?;

        // Close all connections
        let connections = self.connections.read().await;
        for (_, conn) in connections.iter() {
            conn.close().await?;
        }

        // Stop transport
        self.transport.stop().await?;

        state.running = false;
        Ok(())
    }

    /// Connect to a peer
    pub async fn connect(&self, peer_addr: &str) -> NetworkResult<NodeId> {
        // Establish transport connection
        let conn = self.transport.connect(peer_addr).await?;

        // Perform handshake
        let peer_info = self.protocol.handshake(&conn).await?;

        // Store connection
        self.connections
            .write()
            .await
            .insert(peer_info.node_id, Arc::new(conn));

        // Update routing table
        self.router.add_peer(peer_info.node_id, peer_addr).await?;

        // Update state
        let mut state = self.state.write().await;
        state.peers.push(peer_info.node_id);
        state.stats.active_connections += 1;

        Ok(peer_info.node_id)
    }

    /// Send a packet to a peer
    pub async fn send(&self, peer_id: NodeId, packet: PhasePacket<Vec<u8>>) -> NetworkResult<()> {
        // Get route to peer
        let route = self.router.find_route(peer_id).await?;

        // Get connection
        let connections = self.connections.read().await;
        let conn = connections
            .get(&route.next_hop)
            .ok_or_else(|| anyhow::anyhow!("No connection to next hop"))?;

        // Encode the packet using proper serialization
        let data = self.protocol.encode_packet(&packet)?;
        let compressed = self.compress_data(&data)?;

        // Send data
        conn.send(&compressed).await?;

        // Update metrics
        self.metrics.record_packet_sent(compressed.len());

        Ok(())
    }

    /// Broadcast a packet to all peers
    pub async fn broadcast(&self, packet: PhasePacket<Vec<u8>>) -> NetworkResult<()> {
        let connections = self.connections.read().await;

        // Clone packet for each peer (simple but functional implementation)
        for (peer_id, conn) in connections.iter() {
            let packet_clone = packet.clone();
            let data = self.protocol.encode_packet(&packet_clone)?;
            let compressed = self.compress_data(&data)?;

            // Send to each peer - log errors but don't fail entire broadcast
            if let Err(e) = conn.send(&compressed).await {
                tracing::warn!("Failed to broadcast to peer {}: {}", peer_id.0, e);
            } else {
                self.metrics.record_packet_sent(compressed.len());
            }
        }

        Ok(())
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        self.state.read().await.stats.clone()
    }

    /// Accept incoming connections
    async fn accept_loop(self: Arc<Self>) {
        loop {
            match self.transport.accept().await {
                Ok(conn) => {
                    // Spawn concurrent connection handler
                    let self_clone = Arc::clone(&self);
                    tokio::spawn(async move {
                        if let Err(e) = self_clone.handle_connection(conn).await {
                            tracing::error!("Failed to handle connection: {}", e);
                        }
                    });
                }
                Err(e) => {
                    tracing::error!("Accept error: {}", e);
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Handle incoming connection
    async fn handle_connection(&self, conn: Connection) -> NetworkResult<()> {
        // Perform handshake
        let peer_info = self.protocol.accept_handshake(&conn).await?;

        // Store connection
        self.connections
            .write()
            .await
            .insert(peer_info.node_id, Arc::new(conn.clone()));

        // Update routing table
        self.router
            .add_peer(peer_info.node_id, &peer_info.address)
            .await?;

        // Handle incoming messages
        loop {
            match conn.recv().await {
                Ok(data) => {
                    // Decompress and decode
                    let decompressed = self.decompress_data(&data)?;
                    let packet = self.protocol.decode_packet(&decompressed)?;

                    // Process packet asynchronously to ensure Send safety
                    let protocol_clone = Arc::clone(&self.protocol);
                    let metrics_clone = Arc::clone(&self.metrics);
                    let data_len = data.len();
                    
                    tokio::spawn(async move {
                        if let Err(e) = protocol_clone.handle_packet(packet).await {
                            tracing::error!("Failed to handle packet: {}", e);
                        }
                        metrics_clone.record_packet_received(data_len);
                    });
                }
                Err(e) => {
                    tracing::debug!("Connection closed: {}", e);
                    break;
                }
            }
        }

        // Clean up connection
        self.connections.write().await.remove(&peer_info.node_id);

        Ok(())
    }

    /// Compress data
    fn compress_data(&self, data: &[u8]) -> NetworkResult<Vec<u8>> {
        if !self.config.compression.enabled || data.len() < self.config.compression.min_size {
            return Ok(data.to_vec());
        }

        compression::compress(
            data,
            self.config.compression.algorithm,
            self.config.compression.level,
        )
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> NetworkResult<Vec<u8>> {
        if !self.config.compression.enabled {
            return Ok(data.to_vec());
        }

        compression::decompress(data, self.config.compression.algorithm)
    }
}

impl Clone for NetworkNode {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            transport: self.transport.clone(),
            discovery: self.discovery.clone(),
            router: self.router.clone(),
            protocol: self.protocol.clone(),
            connections: self.connections.clone(),
            state: self.state.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

/// Network metrics
pub struct NetworkMetrics {
    packets_sent: prometheus::Counter,
    packets_received: prometheus::Counter,
    bytes_sent: prometheus::Counter,
    bytes_received: prometheus::Counter,
    _connection_errors: prometheus::Counter,
    _latency_histogram: prometheus::Histogram,
}

impl NetworkMetrics {
    fn new() -> NetworkResult<Self> {
        let packets_sent =
            prometheus::Counter::new("csf_net_packets_sent", "Total packets sent")
                .map_err(|e| anyhow::anyhow!("Failed to create packets_sent counter: {}", e))?;
        let packets_received =
            prometheus::Counter::new("csf_net_packets_received", "Total packets received")
                .map_err(|e| anyhow::anyhow!("Failed to create packets_received counter: {}", e))?;
        let bytes_sent = prometheus::Counter::new("csf_net_bytes_sent", "Total bytes sent")
            .map_err(|e| anyhow::anyhow!("Failed to create bytes_sent counter: {}", e))?;
        let bytes_received =
            prometheus::Counter::new("csf_net_bytes_received", "Total bytes received")
                .map_err(|e| anyhow::anyhow!("Failed to create bytes_received counter: {}", e))?;
        let connection_errors =
            prometheus::Counter::new("csf_net_connection_errors", "Total connection errors")
                .map_err(|e| {
                    anyhow::anyhow!("Failed to create connection_errors counter: {}", e)
                })?;
        let latency_histogram = prometheus::Histogram::with_opts(prometheus::HistogramOpts::new(
            "csf_net_latency",
            "Network latency in milliseconds",
        ))
        .map_err(|e| anyhow::anyhow!("Failed to create latency_histogram: {}", e))?;

        Ok(Self {
            packets_sent,
            packets_received,
            bytes_sent,
            bytes_received,
            _connection_errors: connection_errors,
            _latency_histogram: latency_histogram,
        })
    }

    fn record_packet_sent(&self, size: usize) {
        self.packets_sent.inc();
        self.bytes_sent.inc_by(size as f64);
    }

    fn record_packet_received(&self, size: usize) {
        self.packets_received.inc();
        self.bytes_received.inc_by(size as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_node_creation() {
        let config = NetworkConfig::default();
        let bus = Arc::new(
            csf_bus::PhaseCoherenceBus::new(Default::default())
                .expect("Bus creation should not fail with default config"),
        );

        let node = NetworkNode::new(config, bus)
            .await
            .expect("NetworkNode creation should not fail with valid config");
        assert!(!node.state.read().await.running);
    }
}