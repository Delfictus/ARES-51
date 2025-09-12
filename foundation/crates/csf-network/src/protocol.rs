//! Network protocol implementation

use super::*;
use csf_bus::{packet::PhasePacket, PhaseCoherenceBus};
use csf_time::global_time_source;
use serde::{Deserialize, Serialize};

/// Protocol handler
pub struct Protocol {
    node_id: NodeId,
    bus: Arc<PhaseCoherenceBus>,
    handlers: dashmap::DashMap<MessageType, Box<dyn MessageHandler>>,
}

/// Protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMessage {
    /// Message type
    pub msg_type: MessageType,

    /// Source node
    pub source: NodeId,

    /// Destination node
    pub destination: NodeId,

    /// Message ID
    pub msg_id: u64,

    /// Timestamp
    pub timestamp: u64,

    /// Payload
    pub payload: MessagePayload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    Handshake,
    PhasePacket,
    Ping,
    Pong,
    RouteUpdate,
    PeerDiscovery,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    Handshake(HandshakePayload),
    PhasePacket(Vec<u8>), // Serialized packet data
    Ping(u64),
    Pong(u64),
    RouteUpdate(Vec<RouteInfo>),
    PeerDiscovery(Vec<PeerInfo>),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakePayload {
    pub version: u32,
    pub node_info: PeerInfo,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    pub destination: NodeId,
    pub next_hop: NodeId,
    pub cost: u32,
    pub latency_ms: u32,
}

/// Message handler trait
trait MessageHandler: Send + Sync {
    fn handle(&self, msg: ProtocolMessage) -> NetworkResult<Option<ProtocolMessage>>;
}

impl Protocol {
    /// Create new protocol handler
    pub fn new(node_id: NodeId, bus: Arc<PhaseCoherenceBus>) -> Self {
        Self {
            node_id,
            bus,
            handlers: dashmap::DashMap::new(),
        }
    }

    /// Perform handshake with peer
    pub async fn handshake(&self, conn: &Connection) -> NetworkResult<PeerInfo> {
        // Send handshake
        let handshake_msg = ProtocolMessage {
            msg_type: MessageType::Handshake,
            source: self.node_id,
            destination: NodeId(0), // Unknown yet
            msg_id: rand::random(),
            timestamp: global_time_source()
                .now_ns()
                .unwrap_or(csf_time::NanoTime::ZERO)
                .as_nanos(),
            payload: MessagePayload::Handshake(HandshakePayload {
                version: 1,
                node_info: PeerInfo {
                    node_id: self.node_id,
                    address: "".to_string(), // Will be filled by peer
                    public_key: vec![],
                    capabilities: vec!["csf-1.0".to_string()],
                },
                capabilities: vec!["phase-packets".to_string()],
            }),
        };

        let data = self.encode_message(&handshake_msg)?;
        conn.send(&data).await?;

        // Receive response
        let response_data = conn.recv().await?;
        let response = self.decode_message(&response_data)?;

        match response.payload {
            MessagePayload::Handshake(payload) => Ok(payload.node_info),
            _ => Err(anyhow::anyhow!("Invalid handshake response")),
        }
    }

    /// Accept handshake from peer
    pub async fn accept_handshake(&self, conn: &Connection) -> NetworkResult<PeerInfo> {
        // Receive handshake
        let data = conn.recv().await?;
        let msg = self.decode_message(&data)?;

        let peer_info = match msg.payload {
            MessagePayload::Handshake(payload) => payload.node_info,
            _ => return Err(anyhow::anyhow!("Expected handshake")),
        };

        // Send response
        let response = ProtocolMessage {
            msg_type: MessageType::Handshake,
            source: self.node_id,
            destination: peer_info.node_id,
            msg_id: rand::random(),
            timestamp: global_time_source()
                .now_ns()
                .unwrap_or(csf_time::NanoTime::ZERO)
                .as_nanos(),
            payload: MessagePayload::Handshake(HandshakePayload {
                version: 1,
                node_info: PeerInfo {
                    node_id: self.node_id,
                    address: "".to_string(),
                    public_key: vec![],
                    capabilities: vec!["csf-1.0".to_string()],
                },
                capabilities: vec!["phase-packets".to_string()],
            }),
        };

        let response_data = self.encode_message(&response)?;
        conn.send(&response_data).await?;

        Ok(peer_info)
    }

    /// Handle incoming packet
    pub async fn handle_packet(&self, packet: PhasePacket<Vec<u8>>) -> NetworkResult<()> {
        // Production-grade packet handling with proper type safety
        let processed_packet =
            PhasePacket::new(packet.payload.clone(), packet.routing_metadata.source_id);
        let now = csf_time::global_time_source()
            .now_ns()
            .unwrap_or(csf_time::NanoTime::ZERO);
        let duration = csf_time::Duration::from_millis(1000); // 1 second deadline
        let deadline =
            csf_time::NanoTime::from_nanos(now.as_nanos().saturating_add(duration.as_nanos()));
        self.bus
            .publish_with_deadline(processed_packet, deadline)
            .await?;
        Ok(())
    }

    /// Encode packet for transmission with production-grade serialization
    pub fn encode_packet(&self, packet: &PhasePacket<Vec<u8>>) -> NetworkResult<Vec<u8>> {
        // Extract destination from packet metadata or use broadcast
        // Since RoutingMetadata doesn't have a 'destination' field, we'll use the source_id for routing
        // or default to broadcast if no specific routing is needed
        let destination = NodeId(packet.routing_metadata.source_id.inner());

        // Serialize packet payload using bincode for efficient binary encoding
        let serialized_payload = bincode::serialize(&packet.payload)
            .map_err(|e| anyhow::anyhow!("Failed to serialize packet payload: {}", e))?;

        let msg = ProtocolMessage {
            msg_type: MessageType::PhasePacket,
            source: self.node_id,
            destination,
            msg_id: rand::random(),
            timestamp: global_time_source()
                .now_ns()
                .unwrap_or(csf_time::NanoTime::ZERO)
                .as_nanos(),
            payload: MessagePayload::PhasePacket(serialized_payload),
        };

        self.encode_message(&msg)
    }

    /// Decode received packet
    pub fn decode_packet(&self, data: &[u8]) -> NetworkResult<PhasePacket<Vec<u8>>> {
        let msg = self.decode_message(data)?;

        match msg.payload {
            MessagePayload::PhasePacket(packet_data) => {
                // Create a new PhasePacket with the received data using bus packet format
                let packet =
                    PhasePacket::new(packet_data, csf_core::ComponentId::new(msg.source.0));
                Ok(packet)
            }
            _ => Err(anyhow::anyhow!("Not a phase packet")),
        }
    }

    /// Encode protocol message
    fn encode_message(&self, msg: &ProtocolMessage) -> NetworkResult<Vec<u8>> {
        Ok(bincode::serialize(msg)?)
    }

    /// Decode protocol message
    fn decode_message(&self, data: &[u8]) -> NetworkResult<ProtocolMessage> {
        Ok(bincode::deserialize(data)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub address: String,
    pub public_key: Vec<u8>,
    pub capabilities: Vec<String>,
}
