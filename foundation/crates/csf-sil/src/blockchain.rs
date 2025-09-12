//! Blockchain implementation with Practical Byzantine Fault Tolerance (PBFT)

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use futures::stream::{FuturesUnordered, StreamExt};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use blake3::Hasher;
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};

/// Node ID type
pub type NodeId = [u8; 32];

/// Block ID type
pub type BlockId = [u8; 32];

/// View number for PBFT
pub type ViewNumber = u64;

/// Sequence number
pub type SequenceNumber = u64;

/// Transaction in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Packet ID
    pub packet_id: crate::PacketId,
    /// Data hash
    pub hash: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
    /// Signature
    pub signature: Signature,
}

/// Block in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block ID (hash)
    pub id: BlockId,
    /// Sequence number
    pub sequence: SequenceNumber,
    /// Previous block hash
    pub prev_hash: BlockId,
    /// Merkle root of transactions
    pub merkle_root: [u8; 32],
    /// Transactions
    pub transactions: Vec<Transaction>,
    /// Timestamp
    pub timestamp: u64,
    /// Block producer
    pub producer: NodeId,
    /// Producer signature
    pub signature: Signature,
}

impl Block {
    /// Compute block hash
    pub fn compute_hash(&self) -> BlockId {
        let mut hasher = Hasher::new();
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(&self.prev_hash);
        hasher.update(&self.merkle_root);
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&self.producer);
        
        let hash = hasher.finalize();
        *hash.as_bytes()
    }
    
    /// Compute merkle root of transactions
    pub fn compute_merkle_root(transactions: &[Transaction]) -> [u8; 32] {
        if transactions.is_empty() {
            return [0; 32];
        }
        
        // Simple merkle tree implementation
        let mut hashes: Vec<[u8; 32]> = transactions.iter()
            .map(|tx| {
                let mut hasher = Hasher::new();
                hasher.update(&tx.packet_id.to_bytes());
                hasher.update(&tx.hash);
                hasher.update(&tx.timestamp.to_le_bytes());
                *hasher.finalize().as_bytes()
            })
            .collect();
        
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in hashes.chunks(2) {
                let mut hasher = Hasher::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]); // Duplicate for odd number
                }
                next_level.push(*hasher.finalize().as_bytes());
            }
            
            hashes = next_level;
        }
        
        hashes[0]
    }
}

/// PBFT consensus state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusPhase {
    /// Normal operation
    Normal,
    /// View change in progress
    ViewChange,
    /// Pre-prepare phase
    PrePrepare,
    /// Prepare phase
    Prepare,
    /// Commit phase
    Commit,
}

/// PBFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftMessage {
    /// Request from client
    Request(Request),
    /// Pre-prepare message (from primary)
    PrePrepare(PrePrepare),
    /// Prepare message
    Prepare(Prepare),
    /// Commit message
    Commit(Commit),
    /// Reply to client
    Reply(Reply),
    /// View change message
    ViewChange(ViewChange),
    /// New view message
    NewView(NewView),
    /// Checkpoint message
    Checkpoint(Checkpoint),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub operation: Operation,
    pub timestamp: u64,
    pub client: NodeId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Add transaction to blockchain
    AddTransaction(Transaction),
    /// Query blockchain state
    Query(Query),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrePrepare {
    pub view: ViewNumber,
    pub sequence: SequenceNumber,
    pub digest: [u8; 32],
    pub request: Request,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prepare {
    pub view: ViewNumber,
    pub sequence: SequenceNumber,
    pub digest: [u8; 32],
    pub node: NodeId,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub view: ViewNumber,
    pub sequence: SequenceNumber,
    pub digest: [u8; 32],
    pub node: NodeId,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    pub view: ViewNumber,
    pub timestamp: u64,
    pub client: NodeId,
    pub node: NodeId,
    pub result: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChange {
    pub new_view: ViewNumber,
    pub last_sequence: SequenceNumber,
    pub checkpoints: Vec<Checkpoint>,
    pub node: NodeId,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewView {
    pub view: ViewNumber,
    pub view_changes: Vec<ViewChange>,
    pub pre_prepares: Vec<PrePrepare>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub sequence: SequenceNumber,
    pub digest: [u8; 32],
    pub node: NodeId,
    pub signature: Signature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub query_type: QueryType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    /// Get block by ID
    GetBlock(BlockId),
    /// Get latest block
    GetLatestBlock,
    /// Get transaction
    GetTransaction([u8; 32]),
}

/// PBFT consensus implementation
pub struct PbftConsensus {
    /// Node ID
    node_id: NodeId,
    /// Node keypair
    keypair: Arc<Keypair>,
    /// Current view number
    view_number: Arc<RwLock<ViewNumber>>,
    /// Current sequence number
    sequence_number: Arc<RwLock<SequenceNumber>>,
    /// Node public keys
    node_keys: Arc<RwLock<HashMap<NodeId, PublicKey>>>,
    /// Active nodes
    active_nodes: Arc<RwLock<HashSet<NodeId>>>,
    /// Current phase
    phase: Arc<RwLock<ConsensusPhase>>,
    /// Pending requests
    pending_requests: Arc<RwLock<VecDeque<Request>>>,
    /// Pre-prepare log
    pre_prepare_log: Arc<RwLock<HashMap<(ViewNumber, SequenceNumber), PrePrepare>>>,
    /// Prepare messages
    prepare_messages: Arc<RwLock<HashMap<(ViewNumber, SequenceNumber), HashSet<Prepare>>>>,
    /// Commit messages
    commit_messages: Arc<RwLock<HashMap<(ViewNumber, SequenceNumber), HashSet<Commit>>>>,
    /// Message sender
    message_tx: mpsc::Sender<(NodeId, PbftMessage)>,
    /// Message receiver
    message_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<(NodeId, PbftMessage)>>>,
    /// Blockchain state
    blockchain: Arc<RwLock<Vec<Block>>>,
    /// Transaction pool
    tx_pool: Arc<RwLock<Vec<Transaction>>>,
}

impl PbftConsensus {
    /// Create new PBFT consensus instance
    pub fn new(node_id: NodeId, keypair: Keypair, peers: Vec<(NodeId, PublicKey)>) -> Self {
        let (message_tx, message_rx) = mpsc::channel(1000);
        
        let mut node_keys = HashMap::new();
        let mut active_nodes = HashSet::new();
        
        // Add self
        node_keys.insert(node_id, keypair.public);
        active_nodes.insert(node_id);
        
        // Add peers
        for (peer_id, peer_key) in peers {
            node_keys.insert(peer_id, peer_key);
            active_nodes.insert(peer_id);
        }
        
        Self {
            node_id,
            keypair: Arc::new(keypair),
            view_number: Arc::new(RwLock::new(0)),
            sequence_number: Arc::new(RwLock::new(0)),
            node_keys: Arc::new(RwLock::new(node_keys)),
            active_nodes: Arc::new(RwLock::new(active_nodes)),
            phase: Arc::new(RwLock::new(ConsensusPhase::Normal)),
            pending_requests: Arc::new(RwLock::new(VecDeque::new())),
            pre_prepare_log: Arc::new(RwLock::new(HashMap::new())),
            prepare_messages: Arc::new(RwLock::new(HashMap::new())),
            commit_messages: Arc::new(RwLock::new(HashMap::new())),
            message_tx,
            message_rx: Arc::new(tokio::sync::Mutex::new(message_rx)),
            blockchain: Arc::new(RwLock::new(Vec::new())),
            tx_pool: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Check if this node is the primary
    pub fn is_primary(&self) -> bool {
        let view = self.view_number.read();
        let nodes = self.active_nodes.read();
        let node_list: Vec<_> = nodes.iter().cloned().collect();
        
        if node_list.is_empty() {
            return false;
        }
        
        let primary_index = (*view as usize) % node_list.len();
        node_list[primary_index] == self.node_id
    }
    
    /// Get required votes for consensus (2f + 1 where f is max faulty nodes)
    pub fn required_votes(&self) -> usize {
        let n = self.active_nodes.read().len();
        let f = (n - 1) / 3;
        2 * f + 1
    }
    
    /// Process incoming message
    pub async fn handle_message(&self, from: NodeId, message: PbftMessage) -> Result<()> {
        match message {
            PbftMessage::Request(req) => self.handle_request(req).await?,
            PbftMessage::PrePrepare(pp) => self.handle_pre_prepare(from, pp).await?,
            PbftMessage::Prepare(p) => self.handle_prepare(p).await?,
            PbftMessage::Commit(c) => self.handle_commit(c).await?,
            PbftMessage::ViewChange(vc) => self.handle_view_change(vc).await?,
            PbftMessage::NewView(nv) => self.handle_new_view(nv).await?,
            _ => {}
        }
        Ok(())
    }
    
    /// Handle client request
    async fn handle_request(&self, request: Request) -> Result<()> {
        if self.is_primary() {
            // Primary: start consensus
            let view = *self.view_number.read();
            let sequence = {
                let mut seq = self.sequence_number.write();
                *seq += 1;
                *seq
            };
            
            // Create pre-prepare message
            let digest = self.compute_request_digest(&request)?;
            let pre_prepare = PrePrepare {
                view,
                sequence,
                digest,
                request: request.clone(),
            };
            
            // Store in log
            self.pre_prepare_log.write().insert((view, sequence), pre_prepare.clone());
            
            // Broadcast pre-prepare
            self.broadcast(PbftMessage::PrePrepare(pre_prepare)).await?;
            
            // Send prepare as well
            let prepare = self.create_prepare(view, sequence, digest)?;
            self.broadcast(PbftMessage::Prepare(prepare)).await?;
        } else {
            // Backup: forward to primary
            self.pending_requests.write().push_back(request);
        }
        
        Ok(())
    }
    
    /// Handle pre-prepare message
    async fn handle_pre_prepare(&self, from: NodeId, pre_prepare: PrePrepare) -> Result<()> {
        // Verify it's from primary
        if !self.is_node_primary(from, pre_prepare.view) {
            return Ok(());
        }
        
        // Verify digest matches request
    let computed_digest = self.compute_request_digest(&pre_prepare.request)?;
        if computed_digest != pre_prepare.digest {
            return Ok(());
        }
        
        // Store pre-prepare
        self.pre_prepare_log.write().insert(
            (pre_prepare.view, pre_prepare.sequence),
            pre_prepare.clone()
        );
        
        // Send prepare message
        let prepare = self.create_prepare(
            pre_prepare.view,
            pre_prepare.sequence,
            pre_prepare.digest
        )?;
        
        self.broadcast(PbftMessage::Prepare(prepare)).await?;
        
        Ok(())
    }
    
    /// Handle prepare message
    async fn handle_prepare(&self, prepare: Prepare) -> Result<()> {
        // Verify signature
        self.verify_prepare(&prepare)?;
        
        // Store prepare message
        let key = (prepare.view, prepare.sequence);
        self.prepare_messages.write()
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(prepare);
        
        // Check if we have enough prepares
        let prepare_count = self.prepare_messages.read()
            .get(&key)
            .map(|set| set.len())
            .unwrap_or(0);
        
        if prepare_count >= self.required_votes() {
            // Move to commit phase
            let commit = self.create_commit(prepare.view, prepare.sequence, prepare.digest)?;
            self.broadcast(PbftMessage::Commit(commit)).await?;
        }
        
        Ok(())
    }
    
    /// Handle commit message
    async fn handle_commit(&self, commit: Commit) -> Result<()> {
        // Verify signature
        self.verify_commit(&commit)?;
        
        // Store commit message
        let key = (commit.view, commit.sequence);
        self.commit_messages.write()
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(commit);
        
        // Check if we have enough commits
        let commit_count = self.commit_messages.read()
            .get(&key)
            .map(|set| set.len())
            .unwrap_or(0);
        
        if commit_count >= self.required_votes() {
            // Execute the request
            if let Some(pre_prepare) = self.pre_prepare_log.read().get(&key) {
                self.execute_request(&pre_prepare.request).await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle view change
    async fn handle_view_change(&self, view_change: ViewChange) -> Result<()> {
        let current_view = *self.view_number.read();
        
        // Only accept view change if it's for a higher view number
        if view_change.new_view <= current_view {
            return Ok(()); // Ignore old view change
        }
        
        // Verify signature on view change message
        let view_change_bytes = bincode::serialize(&(
            view_change.new_view,
            view_change.last_sequence,
            &view_change.checkpoints,
            view_change.node,
        ))?;
        
        if let Some(public_key) = self.node_keys.read().get(&view_change.node) {
            if public_key.verify(&view_change_bytes, &view_change.signature).is_err() {
                return Ok(()); // Invalid signature, ignore
            }
        } else {
            return Ok(()); // Unknown node, ignore
        }
        
        // Store view change message
        let mut view_changes = HashMap::new();
        view_changes.insert(view_change.node, view_change.clone());
        
        // Check if we have enough view changes to trigger new view
        let active_nodes = self.active_nodes.read().clone();
        let required_view_changes = (active_nodes.len() * 2 / 3) + 1; // f+1 view changes needed
        
        if view_changes.len() >= required_view_changes {
            self.trigger_new_view(view_change.new_view, view_changes).await?;
        }
        
        Ok(())
    }
    
    /// Trigger new view after collecting enough view change messages
    async fn trigger_new_view(&self, new_view: ViewNumber, view_changes: HashMap<NodeId, ViewChange>) -> Result<()> {
        // Check if we are the new primary
        let active_nodes: Vec<NodeId> = self.active_nodes.read().iter().cloned().collect();
        if active_nodes.is_empty() {
            return Ok(());
        }
        
        // Simple primary selection: primary = view % number_of_nodes
        let primary_index = (new_view as usize) % active_nodes.len();
        let new_primary = active_nodes[primary_index];
        
        if new_primary == self.node_id {
            // We are the new primary, send NEW-VIEW message
            self.send_new_view(new_view, view_changes).await?;
        }
        
        // Update our view number
        *self.view_number.write() = new_view;
        *self.phase.write() = ConsensusPhase::PrePrepare;
        
        Ok(())
    }
    
    /// Send new view message as the new primary
    async fn send_new_view(&self, view: ViewNumber, view_changes: HashMap<NodeId, ViewChange>) -> Result<()> {
        // Collect valid pre-prepare messages from view changes
        let mut pre_prepares = Vec::new();
        let mut max_sequence = 0;
        
        for view_change in view_changes.values() {
            if view_change.last_sequence > max_sequence {
                max_sequence = view_change.last_sequence;
            }
        }
        
        // Create pre-prepare messages for any missing sequences
        // For simplicity, we'll start fresh from the last stable checkpoint
        let stable_sequence = max_sequence;
        *self.sequence_number.write() = stable_sequence + 1;
        
        // Create NEW-VIEW message
        let new_view_msg = NewView {
            view,
            view_changes: view_changes.into_values().collect(),
            pre_prepares,
        };
        
        // Broadcast NEW-VIEW message to all nodes
        for &node_id in self.active_nodes.read().iter() {
            if node_id != self.node_id {
                let _ = self.message_tx.send((node_id, PbftMessage::NewView(new_view_msg.clone()))).await;
            }
        }
        
        Ok(())
    }
    
    /// Handle new view message
    async fn handle_new_view(&self, new_view: NewView) -> Result<()> {
        let current_view = *self.view_number.read();
        
        // Only accept new view if it's for a higher view number
        if new_view.view <= current_view {
            return Ok(()); // Ignore old new view
        }
        
        // Validate that we have enough view changes
        let active_nodes = self.active_nodes.read().clone();
        let required_view_changes = (active_nodes.len() * 2 / 3) + 1;
        
        if new_view.view_changes.len() < required_view_changes {
            return Ok(()); // Not enough view changes
        }
        
        // Verify all view change signatures
        for view_change in &new_view.view_changes {
            let view_change_bytes = bincode::serialize(&(
                view_change.new_view,
                view_change.last_sequence,
                &view_change.checkpoints,
                view_change.node,
            ))?;
            
            if let Some(public_key) = self.node_keys.read().get(&view_change.node) {
                if public_key.verify(&view_change_bytes, &view_change.signature).is_err() {
                    return Ok(()); // Invalid signature, reject new view
                }
            } else {
                return Ok(()); // Unknown node, reject
            }
        }
        
        // Update to new view
        *self.view_number.write() = new_view.view;
        *self.phase.write() = ConsensusPhase::PrePrepare;
        
        // Process any pre-prepare messages in the new view
        for pre_prepare in new_view.pre_prepares {
            self.handle_pre_prepare(pre_prepare).await?;
        }
        
        Ok(())
    }
    
    /// Execute request
    async fn execute_request(&self, request: &Request) -> Result<()> {
        match &request.operation {
            Operation::AddTransaction(tx) => {
                // Add to transaction pool
                self.tx_pool.write().push(tx.clone());
                
                // Create block if we have enough transactions
                if self.tx_pool.read().len() >= 10 {
                    self.create_block().await?;
                }
            }
            Operation::Query(_query) => {
                // Handle query
            }
        }
        
        Ok(())
    }
    
    /// Create new block
    async fn create_block(&self) -> Result<()> {
        let mut tx_pool = self.tx_pool.write();
        let transactions: Vec<_> = tx_pool.drain(..).collect();
        
        let blockchain = self.blockchain.read();
        let prev_hash = blockchain.last()
            .map(|b| b.id)
            .unwrap_or([0; 32]);
        
        let sequence = blockchain.len() as u64;
        
        let merkle_root = Block::compute_merkle_root(&transactions);
        
        let mut block = Block {
            id: [0; 32],
            sequence,
            prev_hash,
            merkle_root,
            transactions,
            timestamp: crate::hardware_timestamp(),
            producer: self.node_id,
            signature: Signature::from_bytes(&[0; 64])?,
        };
        
        // Compute block hash
        block.id = block.compute_hash();
        
        // Sign block
        let block_bytes = bincode::serialize(&block)?;
        block.signature = self.keypair.sign(&block_bytes);
        
        // Add to blockchain
        drop(blockchain);
        self.blockchain.write().push(block);
        
        Ok(())
    }
    
    /// Broadcast message to all nodes
    async fn broadcast(&self, message: PbftMessage) -> Result<()> {
        let nodes = self.active_nodes.read().clone();
        
        for node in nodes {
            if node != self.node_id {
                self.message_tx.send((node, message.clone())).await?;
            }
        }
        
        Ok(())
    }
    
    /// Check if node is primary for given view
    fn is_node_primary(&self, node: NodeId, view: ViewNumber) -> bool {
        let nodes = self.active_nodes.read();
        let node_list: Vec<_> = nodes.iter().cloned().collect();
        
        if node_list.is_empty() {
            return false;
        }
        
        let primary_index = (view as usize) % node_list.len();
        node_list[primary_index] == node
    }
    
    /// Compute request digest
    fn compute_request_digest(&self, request: &Request) -> Result<[u8; 32]> {
        let data = bincode::serialize(request)?;
        let mut hasher = Hasher::new();
        hasher.update(&data);
        Ok(*hasher.finalize().as_bytes())
    }
    
    /// Create prepare message
    fn create_prepare(&self, view: ViewNumber, sequence: SequenceNumber, digest: [u8; 32]) -> Result<Prepare> {
        let prepare = Prepare {
            view,
            sequence,
            digest,
            node: self.node_id,
            signature: Signature::from_bytes(&[0; 64])?,
        };
        
        // Sign prepare
        let data = bincode::serialize(&prepare)?;
        let signature = self.keypair.sign(&data);
        
        Ok(Prepare { signature, ..prepare })
    }
    
    /// Create commit message
    fn create_commit(&self, view: ViewNumber, sequence: SequenceNumber, digest: [u8; 32]) -> Result<Commit> {
        let commit = Commit {
            view,
            sequence,
            digest,
            node: self.node_id,
            signature: Signature::from_bytes(&[0; 64])?,
        };
        
        // Sign commit
        let data = bincode::serialize(&commit)?;
        let signature = self.keypair.sign(&data);
        
        Ok(Commit { signature, ..commit })
    }
    
    /// Verify prepare message
    fn verify_prepare(&self, prepare: &Prepare) -> Result<()> {
        let node_keys = self.node_keys.read();
        let public_key = node_keys.get(&prepare.node)
            .ok_or(anyhow::anyhow!("Unknown node"))?;
        
        let mut prepare_copy = prepare.clone();
        prepare_copy.signature = Signature::from_bytes(&[0; 64])?;
        let data = bincode::serialize(&prepare_copy)?;
        
        public_key.verify(&data, &prepare.signature)
            .map_err(|_| anyhow::anyhow!("Invalid signature"))?;
        
        Ok(())
    }
    
    /// Verify commit message
    fn verify_commit(&self, commit: &Commit) -> Result<()> {
        let node_keys = self.node_keys.read();
        let public_key = node_keys.get(&commit.node)
            .ok_or(anyhow::anyhow!("Unknown node"))?;
        
        let mut commit_copy = commit.clone();
        commit_copy.signature = Signature::from_bytes(&[0; 64])?;
        let data = bincode::serialize(&commit_copy)?;
        
        public_key.verify(&data, &commit.signature)
            .map_err(|_| anyhow::anyhow!("Invalid signature"))?;
        
        Ok(())
    }
    
    /// Run consensus protocol
    pub async fn run(&self) -> Result<()> {
        let mut message_rx = self.message_rx.lock().await;
        
        while let Some((from, message)) = message_rx.recv().await {
            if let Err(e) = self.handle_message(from, message).await {
                tracing::error!("Error handling message: {}", e);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_merkle_root() {
        let transactions = vec![
            Transaction {
                packet_id: crate::PacketId::new(),
                hash: [1; 32],
                timestamp: 123,
                signature: Signature::from_bytes(&[0; 64]).unwrap(),
            },
            Transaction {
                packet_id: crate::PacketId::new(),
                hash: [2; 32],
                timestamp: 124,
                signature: Signature::from_bytes(&[0; 64]).unwrap(),
            },
        ];
        
        let root = Block::compute_merkle_root(&transactions);
        assert_ne!(root, [0; 32]);
    }
    
    #[tokio::test]
    async fn test_pbft_basic() {
        // Create nodes
        let mut nodes = Vec::new();
        let mut peers = Vec::new();
        
        for i in 0..4 {
            let mut csprng = rand::thread_rng();
            let keypair = Keypair::generate(&mut csprng);
            let node_id = [i; 32];
            peers.push((node_id, keypair.public));
            nodes.push((node_id, keypair));
        }
        
        // Create consensus instances
        let consensuses: Vec<_> = nodes.into_iter()
            .map(|(id, kp)| {
                let peers_copy = peers.iter()
                    .filter(|(pid, _)| *pid != id)
                    .cloned()
                    .collect();
                Arc::new(PbftConsensus::new(id, kp, peers_copy))
            })
            .collect();
        
        // Test primary detection
        assert!(consensuses[0].is_primary());
        assert_eq!(consensuses[0].required_votes(), 3);
    }
}