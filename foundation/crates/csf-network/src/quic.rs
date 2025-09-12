//! QUIC transport implementation for ARES CSF

use crate::NetworkResult;
use csf_time::{global_time_source, NanoTime};
use quinn::{ClientConfig, Connection as QuicConnection, Endpoint, ServerConfig};
use rustls::{Certificate, PrivateKey};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Connection statistics structure
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub path_stats: quinn::PathStats,
    pub frame_stats: quinn::FrameStats,
    pub rtt: Duration,
    pub cwnd: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packet_loss: f64,
    pub throughput_bps: f64,
}

/// QUIC transport configuration
#[derive(Debug, Clone)]
pub struct QuicConfig {
    /// Server name for TLS
    pub server_name: String,

    /// Certificate chain
    pub cert_chain: Vec<Certificate>,

    /// Private key
    pub private_key: PrivateKey,

    /// Max idle timeout (ms)
    pub max_idle_timeout_ms: u64,

    /// Keep alive interval (ms)
    pub keep_alive_interval_ms: u64,

    /// Max concurrent streams
    pub max_concurrent_streams: u64,

    /// Stream receive window
    pub stream_receive_window: u64,

    /// Connection receive window
    pub connection_receive_window: u64,

    /// Enable 0-RTT
    pub enable_0rtt: bool,

    /// Congestion control algorithm
    pub congestion_control: CongestionControl,
}

#[derive(Debug, Clone, Copy)]
pub enum CongestionControl {
    Cubic,
    Bbr,
    NewReno,
}

impl Default for QuicConfig {
    fn default() -> Self {
        // Attempt to generate self-signed cert for testing. If it fails, fall back to empty certs.
        let (cert_chain, private_key) =
            match rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).and_then(
                |cert| {
                    let cert_der = cert.cert.der().to_vec();
                    let key_der = cert.key_pair.serialize_der();
                    Ok((vec![Certificate(cert_der)], PrivateKey(key_der)))
                },
            ) {
                Ok((chain, key)) => (chain, key),
                Err(_) => (Vec::new(), PrivateKey(Vec::new())),
            };

        Self {
            server_name: "localhost".to_string(),
            cert_chain,
            private_key,
            max_idle_timeout_ms: 30_000,
            keep_alive_interval_ms: 10_000,
            max_concurrent_streams: 100,
            stream_receive_window: 10 * 1024 * 1024, // 10MB
            connection_receive_window: 100 * 1024 * 1024, // 100MB
            enable_0rtt: true,
            congestion_control: CongestionControl::Bbr,
        }
    }
}

/// QUIC transport implementation
pub struct QuicTransport {
    config: QuicConfig,
    endpoint: Arc<RwLock<Option<Endpoint>>>,
    connections: Arc<RwLock<std::collections::HashMap<SocketAddr, Arc<QuicConnectionWrapper>>>>,
    incoming_rx: Arc<RwLock<mpsc::Receiver<QuicConnection>>>,
    incoming_tx: mpsc::Sender<QuicConnection>,
}

impl QuicTransport {
    /// Create new QUIC transport
    pub fn new(config: QuicConfig) -> NetworkResult<Self> {
        let (incoming_tx, incoming_rx) = mpsc::channel(100);

        Ok(Self {
            config,
            endpoint: Arc::new(RwLock::new(None)),
            connections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            incoming_rx: Arc::new(RwLock::new(incoming_rx)),
            incoming_tx,
        })
    }

    /// Start listening on address
    pub async fn listen(&self, addr: SocketAddr) -> NetworkResult<()> {
        let mut endpoint_guard = self.endpoint.write().await;
        if endpoint_guard.is_some() {
            return Err(anyhow::anyhow!("Already listening"));
        }

        // Create server config
        let server_config = self.create_server_config()?;

        // Create endpoint
        let endpoint = Endpoint::server(server_config, addr)?;
        *endpoint_guard = Some(endpoint.clone());
        drop(endpoint_guard);

        tracing::info!("QUIC endpoint listening on {}", addr);

        Ok(())
    }

    /// Connect to remote address
    pub async fn connect(&self, addr: SocketAddr) -> NetworkResult<Arc<QuicConnectionWrapper>> {
        let mut endpoint = self.get_or_create_client_endpoint().await?;

        // Create client config
        let client_config = self.create_client_config()?;
        endpoint.set_default_client_config(client_config);

        // Connect
        let connecting = endpoint.connect(addr, &self.config.server_name)?;
        let connection = connecting.await?;

        // Wrap connection
        let wrapper = Arc::new(QuicConnectionWrapper::new(connection));

        // Store connection
        self.connections.write().await.insert(addr, wrapper.clone());

        Ok(wrapper)
    }

    /// Accept incoming connection
    pub async fn accept(&self) -> NetworkResult<Arc<QuicConnectionWrapper>> {
        let endpoint = self.endpoint.read().await;
        if let Some(endpoint) = endpoint.as_ref() {
            if let Some(incoming) = endpoint.accept().await {
                let connection = incoming.await?;
                let wrapper = Arc::new(QuicConnectionWrapper::new(connection));
                return Ok(wrapper);
            }
        }
        Err(anyhow::anyhow!("Endpoint not listening"))
    }

    /// Close transport
    pub async fn close(&self) -> NetworkResult<()> {
        // Close all connections
        let connections = self.connections.write().await;
        for (_, conn) in connections.iter() {
            conn.close().await?;
        }
        drop(connections);

        // Close endpoint
        if let Some(endpoint) = self.endpoint.write().await.take() {
            endpoint.close(0u32.into(), b"shutdown");
        }

        Ok(())
    }

    /// Create server configuration
    fn create_server_config(&self) -> NetworkResult<ServerConfig> {
        // Convert rustls types to quinn types
        let cert_chain: Vec<quinn::rustls::pki_types::CertificateDer> = self
            .config
            .cert_chain
            .iter()
            .map(|cert| quinn::rustls::pki_types::CertificateDer::from(cert.0.clone()))
            .collect();

        let private_key = quinn::rustls::pki_types::PrivateKeyDer::from(
            quinn::rustls::pki_types::PrivatePkcs8KeyDer::from(self.config.private_key.0.clone()),
        );

        let mut server_config = ServerConfig::with_single_cert(cert_chain, private_key)?;
        let transport_config = Arc::get_mut(&mut server_config.transport)
            .ok_or_else(|| anyhow::anyhow!("Failed to get mutable transport config"))?;
        transport_config.max_concurrent_uni_streams(0_u8.into());

        Ok(server_config)
    }

    /// Create client configuration
    fn create_client_config(&self) -> NetworkResult<ClientConfig> {
        // Use platform verifier for simplified high-performance setup
        let mut client_config = ClientConfig::with_platform_verifier();
        
        // Configure transport parameters optimized for 1M+ msgs/sec
        let mut transport_config = quinn::TransportConfig::default();
        transport_config.max_concurrent_uni_streams(0_u8.into());
        transport_config.max_concurrent_bidi_streams(quinn::VarInt::from_u32(
            self.config.max_concurrent_streams.min(u32::MAX as u64) as u32
        ));
        transport_config.stream_receive_window(quinn::VarInt::from_u32(
            self.config.stream_receive_window.min(u32::MAX as u64) as u32
        ));
        transport_config.receive_window(quinn::VarInt::from_u32(
            self.config.connection_receive_window.min(u32::MAX as u64) as u32
        ));
        
        // Set aggressive keep-alive for high throughput
        transport_config.keep_alive_interval(Some(
            Duration::from_millis(self.config.keep_alive_interval_ms)
        ));
        transport_config.max_idle_timeout(Some(
            quinn::VarInt::from_u32(self.config.max_idle_timeout_ms as u32).try_into()
                .unwrap_or_else(|_| quinn::IdleTimeout::try_from(quinn::VarInt::from_u32(30000)).unwrap())
        ));

        client_config.transport_config(Arc::new(transport_config));
        
        Ok(client_config)
    }

    /// Get or create client endpoint
    async fn get_or_create_client_endpoint(&self) -> NetworkResult<Endpoint> {
        let mut endpoint_guard = self.endpoint.write().await;
        if let Some(endpoint) = endpoint_guard.as_ref() {
            return Ok(endpoint.clone());
        }

        let endpoint = Endpoint::client("[::]:0".parse()?)?;
        *endpoint_guard = Some(endpoint.clone());

        Ok(endpoint)
    }
}

/// QUIC connection wrapper
pub struct QuicConnectionWrapper {
    connection: QuicConnection,
    streams: Arc<RwLock<StreamManager>>,
    start_time: Instant,
}

struct StreamManager {
    next_stream_id: u64,
    active_streams: std::collections::HashMap<u64, quinn::SendStream>,
}

impl QuicConnectionWrapper {
    fn new(connection: QuicConnection) -> Self {
        Self {
            connection,
            streams: Arc::new(RwLock::new(StreamManager {
                next_stream_id: 0,
                active_streams: std::collections::HashMap::new(),
            })),
            start_time: Instant::now(),
        }
    }

    /// Send data on the connection
    pub async fn send(&self, data: &[u8]) -> NetworkResult<()> {
        // Open new stream for each message
        let (mut send_stream, _) = self.connection.open_bi().await?;

        // Write length prefix
        let len_bytes = (data.len() as u32).to_be_bytes();
        send_stream.write_all(&len_bytes).await?;

        // Write data
        send_stream.write_all(data).await?;

        // Finish stream
        send_stream.finish()?;

        Ok(())
    }

    /// Send data with priority
    pub async fn send_priority(&self, data: &[u8], priority: u8) -> NetworkResult<()> {
        let (mut send_stream, _) = self.connection.open_bi().await?;

        // Set stream priority
        send_stream.set_priority(priority.into())?;

        // Write length prefix
        let len_bytes = (data.len() as u32).to_be_bytes();
        send_stream.write_all(&len_bytes).await?;

        // Write data
        send_stream.write_all(data).await?;

        // Finish stream
        send_stream.finish()?;

        Ok(())
    }

    /// Receive data from the connection
    pub async fn recv(&self) -> NetworkResult<Vec<u8>> {
        // Accept incoming stream
        let (_, mut recv_stream) = self.connection.accept_bi().await?;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        recv_stream.read_exact(&mut len_bytes).await?;
        let len = u32::from_be_bytes(len_bytes) as usize;

        // Read data
        let mut data = vec![0u8; len];
        recv_stream.read_exact(&mut data).await?;

        Ok(data)
    }

    /// Open a unidirectional stream
    pub async fn open_uni(&self) -> NetworkResult<quinn::SendStream> {
        Ok(self.connection.open_uni().await?)
    }

    /// Open a bidirectional stream
    pub async fn open_bi(&self) -> NetworkResult<(quinn::SendStream, quinn::RecvStream)> {
        Ok(self.connection.open_bi().await?)
    }

    /// Get connection statistics with Quinn 0.11+ compatibility
    pub fn stats(&self) -> ConnectionStats {
        let stats = self.connection.stats();
        // Extract actual byte counts from frame statistics
        let bytes_sent = stats.frame_tx.acks + stats.frame_tx.crypto + stats.frame_tx.stream;
        let bytes_received = stats.frame_rx.acks + stats.frame_rx.crypto + stats.frame_rx.stream;
        
        ConnectionStats {
            path_stats: stats.path,
            frame_stats: stats.frame_tx,
            rtt: stats.path.rtt,
            cwnd: stats.path.cwnd,
            bytes_sent,
            bytes_received,
            packet_loss: if stats.path.sent_packets > 0 {
                stats.path.lost_packets as f64 / stats.path.sent_packets as f64
            } else {
                0.0
            },
            throughput_bps: self.calculate_throughput(bytes_sent + bytes_received),
        }
    }

    /// Get RTT estimate
    pub fn rtt(&self) -> std::time::Duration {
        self.connection.rtt()
    }

    /// Calculate throughput in bits per second
    fn calculate_throughput(&self, total_bytes: u64) -> f64 {
        let elapsed = self.start_time.elapsed();
        if elapsed.as_secs_f64() > 0.0 {
            (total_bytes as f64 * 8.0) / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Close the connection
    pub async fn close(&self) -> NetworkResult<()> {
        self.connection.close(0u32.into(), b"close");
        Ok(())
    }
}

/// Skip server verification for development
#[derive(Debug)]
struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl rustls::client::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

/// QUIC-based routing protocol
pub struct QuicRouter {
    node_id: crate::NodeId,
    routing_table: Arc<RwLock<RoutingTable>>,
    transport: Arc<QuicTransport>,
}

#[derive(Default)]
struct RoutingTable {
    routes: std::collections::HashMap<crate::NodeId, RouteInfo>,
    next_hop: std::collections::HashMap<crate::NodeId, crate::NodeId>,
}

struct RouteInfo {
    next_hop: SocketAddr,
    metric: u32,
    last_update: NanoTime,
}

impl QuicRouter {
    pub fn new(node_id: crate::NodeId, transport: Arc<QuicTransport>) -> Self {
        Self {
            node_id,
            routing_table: Arc::new(RwLock::new(RoutingTable::default())),
            transport,
        }
    }

    /// Update routing table
    pub async fn update_route(&self, dest: crate::NodeId, next_hop: SocketAddr, metric: u32) {
        let mut table = self.routing_table.write().await;
        table.routes.insert(
            dest,
            RouteInfo {
                next_hop,
                metric,
                last_update: global_time_source().now_ns().unwrap_or(NanoTime::ZERO),
            },
        );
    }

    /// Find route to destination
    pub async fn find_route(&self, dest: crate::NodeId) -> Option<SocketAddr> {
        let table = self.routing_table.read().await;
        table.routes.get(&dest).map(|info| info.next_hop)
    }

    /// Broadcast routing update
    pub async fn broadcast_routes(
        &self,
        connections: &[Arc<QuicConnectionWrapper>],
    ) -> NetworkResult<()> {
        let table = self.routing_table.read().await;
        let routes: Vec<_> = table
            .routes
            .iter()
            .map(|(dest, info)| (*dest, info.metric))
            .collect();
        drop(table);

        // Create routing update message
        let update = RoutingUpdate {
            node_id: self.node_id,
            routes,
            timestamp: global_time_source()
                .now_ns()
                .unwrap_or(NanoTime::ZERO)
                .as_secs(),
        };

        let data = bincode::serialize(&update)?;

        // Broadcast to all connections
        for conn in connections {
            if let Err(e) = conn.send(&data).await {
                tracing::warn!("Failed to send routing update: {}", e);
            }
        }

        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RoutingUpdate {
    node_id: crate::NodeId,
    routes: Vec<(crate::NodeId, u32)>,
    timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quic_transport() {
        let config = QuicConfig::default();
        let transport = QuicTransport::new(config)
            .expect("QuicTransport should initialize with default config");

        // Start listening
        let addr = "127.0.0.1:0"
            .parse()
            .expect("Localhost address should parse correctly");
        transport
            .listen(addr)
            .await
            .expect("QuicTransport should be able to listen on localhost");

        // Close transport
        transport
            .close()
            .await
            .expect("QuicTransport should close cleanly");
    }
}
