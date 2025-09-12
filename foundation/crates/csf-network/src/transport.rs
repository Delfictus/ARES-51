//! Transport layer implementation

use super::*;
use crate::quic::{QuicConfig, QuicConnectionWrapper, QuicTransport};
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

/// Transport configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransportConfig {
    /// Transport protocol
    pub protocol: TransportProtocol,

    /// Buffer size
    pub buffer_size: usize,

    /// Connection timeout (ms)
    pub connection_timeout_ms: u64,

    /// Keepalive interval (ms)
    pub keepalive_interval_ms: u64,

    /// Max frame size
    pub max_frame_size: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum TransportProtocol {
    Quic,
    Tcp,
    WebSocket,
    UnixSocket,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            protocol: TransportProtocol::Quic,
            buffer_size: 65536,
            connection_timeout_ms: 5000,
            keepalive_interval_ms: 30000,
            max_frame_size: 1048576, // 1MB
        }
    }
}

/// Transport layer
pub struct Transport {
    config: TransportConfig,
    listeners: Arc<RwLock<Vec<TransportListener>>>,
    quic_transport: Option<Arc<QuicTransport>>,
}

enum TransportListener {
    Tcp(Arc<TcpListener>),
    Quic(Arc<QuicTransport>),
    WebSocket(Arc<TcpListener>),
}

impl Transport {
    /// Create new transport
    pub async fn new(config: &TransportConfig) -> NetworkResult<Self> {
        let quic_transport = match config.protocol {
            TransportProtocol::Quic => {
                let quic_config = QuicConfig::default();
                Some(Arc::new(QuicTransport::new(quic_config)?))
            }
            _ => None,
        };

        Ok(Self {
            config: config.clone(),
            listeners: Arc::new(RwLock::new(Vec::new())),
            quic_transport,
        })
    }

    /// Listen on address
    pub async fn listen(&self, addr: &str) -> NetworkResult<()> {
        let socket_addr: SocketAddr = addr.parse()?;

        match self.config.protocol {
            TransportProtocol::Quic => {
                if let Some(transport) = &self.quic_transport {
                    transport.listen(socket_addr).await?;
                    self.listeners
                        .write()
                        .push(TransportListener::Quic(transport.clone()));
                }
            }
            TransportProtocol::Tcp => {
                let listener = TcpListener::bind(socket_addr).await?;
                self.listeners
                    .write()
                    .push(TransportListener::Tcp(Arc::new(listener)));
            }
            TransportProtocol::WebSocket => {
                let listener = TcpListener::bind(socket_addr).await?;
                self.listeners
                    .write()
                    .push(TransportListener::WebSocket(Arc::new(listener)));
            }
            _ => return Err(anyhow::anyhow!("Unsupported transport protocol")),
        }

        Ok(())
    }

    /// Connect to address
    pub async fn connect(&self, addr: &str) -> NetworkResult<Connection> {
        let socket_addr: SocketAddr = addr.parse()?;

        match self.config.protocol {
            TransportProtocol::Quic => {
                let transport = self
                    .quic_transport
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("QUIC not initialized"))?;

                let connection = transport.connect(socket_addr).await?;
                Ok(Connection::Quic(connection))
            }
            TransportProtocol::Tcp => {
                let stream = TcpStream::connect(socket_addr).await?;
                Ok(Connection::Tcp(Arc::new(tokio::sync::Mutex::new(stream))))
            }
            TransportProtocol::WebSocket => {
                let stream = TcpStream::connect(socket_addr).await?;
                let stream = MaybeTlsStream::Plain(stream);
                let (ws_stream, _) =
                    tokio_tungstenite::client_async(format!("ws://{}", addr), stream).await?;
                Ok(Connection::WebSocket(Arc::new(tokio::sync::Mutex::new(
                    ws_stream,
                ))))
            }
            _ => Err(anyhow::anyhow!("Unsupported transport protocol")),
        }
    }

    /// Accept incoming connection
    pub async fn accept(&self) -> NetworkResult<Connection> {
        // In a real implementation, this would use select! to accept from multiple listeners
        match self.config.protocol {
            TransportProtocol::Quic => {
                let transport = self
                    .quic_transport
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("QUIC not initialized"))?;

                let connection = transport.accept().await?;
                Ok(Connection::Quic(connection))
            }
            _ => Err(anyhow::anyhow!("Accept not implemented for this transport")),
        }
    }

    /// Stop transport
    pub async fn stop(&self) -> NetworkResult<()> {
        if let Some(transport) = &self.quic_transport {
            transport.close().await?;
        }

        self.listeners.write().clear();
        Ok(())
    }
}

/// Connection abstraction
#[derive(Clone)]
pub enum Connection {
    Quic(Arc<QuicConnectionWrapper>),
    Tcp(Arc<tokio::sync::Mutex<TcpStream>>),
    WebSocket(Arc<tokio::sync::Mutex<WebSocketStream<MaybeTlsStream<TcpStream>>>>),
}

impl Connection {
    /// Send data
    pub async fn send(&self, data: &[u8]) -> NetworkResult<()> {
        match self {
            Connection::Quic(conn) => conn.send(data).await?,
            Connection::Tcp(stream) => {
                use tokio::io::AsyncWriteExt;
                let mut stream = stream.lock().await;
                stream.write_all(data).await?;
            }
            Connection::WebSocket(ws) => {
                use tokio_tungstenite::tungstenite::Message;
                let mut ws = ws.lock().await;
                ws.send(Message::Binary(data.to_vec())).await?;
            }
        }
        Ok(())
    }

    /// Receive data
    pub async fn recv(&self) -> NetworkResult<Vec<u8>> {
        match self {
            Connection::Quic(conn) => conn.recv().await,
            Connection::Tcp(stream) => {
                use tokio::io::AsyncReadExt;
                let mut stream = stream.lock().await;
                let mut buffer = vec![0; 65536];
                let n = stream.read(&mut buffer).await?;
                buffer.truncate(n);
                Ok(buffer)
            }
            Connection::WebSocket(ws) => {
                use tokio_tungstenite::tungstenite::Message;
                let mut ws = ws.lock().await;
                match ws.next().await {
                    Some(Ok(Message::Binary(data))) => Ok(data),
                    Some(Ok(Message::Text(text))) => Ok(text.into_bytes()),
                    _ => Err(anyhow::anyhow!("Invalid message type")),
                }
            }
        }
    }

    /// Close connection
    pub async fn close(&self) -> NetworkResult<()> {
        match self {
            Connection::Quic(conn) => {
                conn.close().await?;
            }
            Connection::Tcp(stream) => {
                let stream = stream.lock().await;
                // No explicit shutdown needed for TcpStream - it drops automatically
                drop(stream);
            }
            Connection::WebSocket(ws) => {
                let mut ws = ws.lock().await;
                ws.close(None).await?;
            }
        }
        Ok(())
    }
}
