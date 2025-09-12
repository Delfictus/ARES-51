//! Peer discovery implementation using libp2p (mDNS + Kademlia)

use super::{DiscoveryConfig, NetworkResult, NodeId, PeerInfo};
use futures::StreamExt;
use libp2p::identity::Keypair;
use libp2p::kad::{store::MemoryStore, Behaviour as KademliaBehaviour};
use libp2p::mdns::tokio::Behaviour as MdnsBehaviour;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::swarm::{NetworkBehaviour, Swarm, SwarmEvent};
use libp2p::{noise, tcp, yamux, Multiaddr, PeerId, SwarmBuilder};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Discovery service
pub struct Discovery {
    config: DiscoveryConfig,
    node_id: NodeId,
    discovered_peers: Arc<RwLock<HashSet<PeerInfo>>>,
    swarm: Arc<Mutex<Swarm<DiscoveryBehaviour>>>,
    loop_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// Combined discovery behaviour
#[derive(NetworkBehaviour)]
pub struct DiscoveryBehaviour {
    pub kademlia: KademliaBehaviour<MemoryStore>,
    pub mdns: Toggle<MdnsBehaviour>,
}

impl Discovery {
    /// Create new discovery service
    pub async fn new(config: &DiscoveryConfig, node_id: NodeId) -> NetworkResult<Self> {
        let local_key = Keypair::generate_ed25519();
        let cfg = config.clone();

        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(move |key| {
                let local_peer_id = PeerId::from(key.public());
                let store = MemoryStore::new(local_peer_id);
                let kademlia = KademliaBehaviour::new(local_peer_id, store);
                let mdns_opt = if cfg.enable_mdns {
                    MdnsBehaviour::new(Default::default(), local_peer_id).ok()
                } else {
                    None
                };
                DiscoveryBehaviour {
                    kademlia,
                    mdns: Toggle::from(mdns_opt),
                }
            })?
            .build();

        // Listen on an ephemeral TCP port
        let listen_addr: Multiaddr = "/ip4/0.0.0.0/tcp/0".parse()?;
        swarm.listen_on(listen_addr)?;

        // Dial bootstrap nodes if any
        for addr in &config.bootstrap_nodes {
            if let Ok(ma) = addr.parse::<Multiaddr>() {
                let _ = swarm.dial(ma);
            }
        }

        Ok(Self {
            config: config.clone(),
            node_id,
            discovered_peers: Arc::new(RwLock::new(HashSet::new())),
            swarm: Arc::new(Mutex::new(swarm)),
            loop_handle: tokio::sync::Mutex::new(None),
        })
    }

    /// Start discovery
    pub async fn start(&self) -> NetworkResult<()> {
        let swarm = self.swarm.clone();
        let discovered = self.discovered_peers.clone();

        let handle = tokio::spawn(async move {
            loop {
                let event_opt = { swarm.lock().await.select_next_some().await };

                match Some(event_opt) {
                    Some(SwarmEvent::NewListenAddr { address, .. }) => {
                        tracing::info!("discovery_listening" = %address);
                    }
                    Some(SwarmEvent::Behaviour(DiscoveryBehaviourEvent::Mdns(event))) => {
                        use libp2p::mdns::Event;
                        match event {
                            Event::Discovered(list) => {
                                let mut guard = discovered.write().await;
                                for (peer_id, multiaddr) in list {
                                    guard.insert(PeerInfo {
                                        node_id: NodeId::from_bytes(&peer_id.to_bytes()),
                                        address: multiaddr.to_string(),
                                        public_key: peer_id.to_bytes(),
                                        capabilities: vec![],
                                    });
                                }
                            }
                            Event::Expired(list) => {
                                let ids: Vec<NodeId> = list
                                    .into_iter()
                                    .map(|(peer_id, _)| NodeId::from_bytes(&peer_id.to_bytes()))
                                    .collect();
                                let mut guard = discovered.write().await;
                                guard.retain(|p| !ids.iter().any(|id| id.0 == p.node_id.0));
                            }
                        }
                    }
                    Some(SwarmEvent::Behaviour(DiscoveryBehaviourEvent::Kademlia(_ev))) => {
                        // Optionally process Kademlia events for routing, peers, etc.
                    }
                    Some(_) => {}
                    None => break,
                }
            }
        });

        *self.loop_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Stop discovery
    pub async fn stop(&self) -> NetworkResult<()> {
        if let Some(handle) = self.loop_handle.lock().await.take() {
            handle.abort();
        }
        Ok(())
    }

    /// Get discovered peers
    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.discovered_peers.read().await.iter().cloned().collect()
    }

    /// Bootstrap with known peers
    pub async fn bootstrap(&self, peers: Vec<String>) -> NetworkResult<()> {
        let mut swarm = self.swarm.lock().await;
        for addr in peers {
            match addr.parse::<Multiaddr>() {
                Ok(ma) => {
                    let _ = swarm.dial(ma);
                }
                Err(e) => {
                    tracing::warn!("invalid_bootstrap_addr" = %addr, error = %e);
                }
            }
        }
        Ok(())
    }
}
