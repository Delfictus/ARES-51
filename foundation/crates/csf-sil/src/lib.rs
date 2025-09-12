//! Secure Immutable Ledger (SIL) Core

use blake3::Hasher;
use csf_core::prelude::*;
use csf_time::{global_time_source, NanoTime};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use parking_lot::RwLock;
use std::sync::Arc;

pub mod audit;
pub mod crypto;
pub mod storage;

use audit::AuditLog;
use storage::Storage;

/// SIL Core configuration
#[derive(Debug, Clone)]
pub struct SilConfig {
    /// Enable blockchain integration
    pub blockchain_enabled: bool,

    /// Enable encryption for sensitive data
    pub encryption_enabled: bool,

    /// Audit log retention in days
    pub audit_retention_days: u32,

    /// Storage backend
    pub storage_backend: StorageBackend,
}

impl Default for SilConfig {
    /// Creates a default `SilConfig` with in-memory storage.
    fn default() -> Self {
        Self {
            blockchain_enabled: false,
            encryption_enabled: false,
            audit_retention_days: 30,
            storage_backend: StorageBackend::Memory,
        }
    }
}

impl SilConfig {
    /// Creates a new builder for `SilConfig`.
    pub fn builder() -> SilConfigBuilder {
        SilConfigBuilder::default()
    }
}

/// Builder for [`SilConfig`].
#[derive(Default)]
pub struct SilConfigBuilder {
    config: SilConfig,
}

impl SilConfigBuilder {
    /// Creates a new `SilConfigBuilder` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether blockchain integration is enabled.
    pub fn blockchain(mut self, enabled: bool) -> Self {
        self.config.blockchain_enabled = enabled;
        self
    }

    /// Sets the storage backend to use.
    pub fn storage(mut self, backend: StorageBackend) -> Self {
        self.config.storage_backend = backend;
        self
    }

    /// Builds the `SilConfig`.
    pub fn build(self) -> SilConfig {
        self.config
    }
}

#[derive(Debug, Clone)]
pub enum StorageBackend {
    Sled(String),
    RocksDB(String),
    Memory,
}

/// The Secure Immutable Ledger core
pub struct SilCore {
    /// Signing keypair
    signing_key: Arc<SigningKey>,

    /// Storage backend
    storage: Arc<dyn Storage>,

    /// Audit log
    audit_log: Arc<AuditLog>,

    

    /// Hash chain state
    chain_state: Arc<RwLock<ChainState>>,
}

/// Hash chain state
struct ChainState {
    /// Current chain head hash
    head_hash: [u8; 32],

    /// Chain length
    chain_length: u64,

    /// Last update timestamp
    last_update_ns: NanoTime,
}

impl SilCore {
    /// Create a new SIL instance
    pub fn new(config: SilConfig) -> Result<Self, Error> {
        // Initialize storage
        let storage = storage::create_storage(&config.storage_backend)?;

        // Load or generate signing key
        let signing_key = Self::load_or_generate_signing_key(&*storage)?;

        // Initialize audit log
        let audit_log = Arc::new(AuditLog::new(storage.clone(), config.audit_retention_days)?);

        // Initialize chain state
        let chain_state = Arc::new(RwLock::new(ChainState {
            head_hash: [0; 32],
            chain_length: 0,
            last_update_ns: global_time_source().now_ns().unwrap_or(NanoTime::ZERO),
        }));

        Ok(Self {
            signing_key: Arc::new(signing_key),
            storage,
            audit_log,
            chain_state,
        })
    }

    /// Compute hash for data with chain linking
    pub fn compute_hash(&self, data: &[u8]) -> [u8; 32] {
        let state = self.chain_state.read();

        let mut hasher = Hasher::new();
        hasher.update(&state.head_hash);
        hasher.update(&state.chain_length.to_le_bytes());
        hasher.update(data);

        let hash = hasher.finalize();
        *hash.as_bytes()
    }

    /// Commit data to the ledger
    pub async fn commit(&self, packet_id: PacketId, data: &[u8]) -> Result<CommitProof, Error> {
        // Compute hash
        let hash = self.compute_hash(data);

        // Create ledger entry
        let entry = LedgerEntry {
            packet_id,
            hash,
            timestamp_ns: global_time_source().now_ns().unwrap_or(NanoTime::ZERO),
            signature: None,
        };

        // Sign entry
        let signature = self.sign_entry(&entry)?;
        let mut signed_entry = entry;
        signed_entry.signature = Some(signature);

        // Store entry
        self.storage.store_entry(&signed_entry).await?;

        // Update chain state
        {
            let mut state = self.chain_state.write();
            state.head_hash = hash;
            state.chain_length += 1;
            state.last_update_ns = signed_entry.timestamp_ns;
        }

        // Log to audit trail
        self.audit_log.log_commit(&signed_entry).await?;

        Ok(CommitProof {
            entry_hash: hash,
            chain_length: self.chain_state.read().chain_length,
            signature,
        })
    }

    /// Verify a commitment proof
    ///
    /// Returns `Ok(())` if the proof is valid for an entry in the ledger.
    ///
    /// # Errors
    ///
    /// Returns an error if the entry is not found, the signature is invalid,
    /// or the proof is otherwise malformed.
    pub async fn verify_proof(&self, proof: &CommitProof) -> Result<(), Error> {
        // Retrieve entry from storage
        let entry = self
            .storage
            .get_entry_by_hash(&proof.entry_hash)
            .await?
            .ok_or(Error::EntryNotFound)?;

        // The signature in the proof must match the one in the stored entry.
        if entry.signature.as_ref() != Some(&proof.signature) {
            return Err(Error::InvalidSignature);
        }

        // Verify the signature against the entry's content.
        let signature = entry.signature.as_ref().ok_or(Error::MissingSignature)?;
        self.verify_signature(&entry, signature)?;

        Ok(())
    }

    /// Retrieve a ledger entry by its content hash.
    pub async fn get_entry(&self, hash: &[u8; 32]) -> Result<Option<LedgerEntry>, Error> {
        self.storage
            .get_entry_by_hash(hash)
            .await
            .map_err(Into::into)
    }

    /// Get the public key of this SIL instance.
    pub fn public_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Get the current chain state
    pub fn chain_state(&self) -> (u64, [u8; 32]) {
        let state = self.chain_state.read();
        (state.chain_length, state.head_hash)
    }

    fn load_or_generate_signing_key(storage: &dyn Storage) -> Result<SigningKey, Error> {
        // Try to load existing key
        if let Some(key_bytes) = storage.get_signing_key()? {
            let bytes: [u8; 32] = key_bytes.try_into().map_err(|_| Error::InvalidKey)?;
            Ok(SigningKey::from_bytes(&bytes))
        } else {
            // Generate new key
            let signing_key = SigningKey::from_bytes(&rand::random::<[u8; 32]>());

            // Store for future use
            storage.store_signing_key(&signing_key.to_bytes())?;

            Ok(signing_key)
        }
    }

    fn sign_entry(&self, entry: &LedgerEntry) -> Result<Signature, Error> {
        let data = bincode::serialize(entry)?;
        Ok(self.signing_key.sign(&data))
    }

    fn verify_signature(&self, entry: &LedgerEntry, signature: &Signature) -> Result<(), Error> {
        let mut entry_copy = entry.clone();
        entry_copy.signature = None;
        let data = bincode::serialize(&entry_copy)?;

        self.signing_key
            .verifying_key()
            .verify(&data, signature)
            .map_err(|_| Error::InvalidSignature)?;

        Ok(())
    }
}

/// Ledger entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    /// Associated packet ID
    pub packet_id: PacketId,

    /// Content hash
    pub hash: [u8; 32],

    /// Timestamp
    pub timestamp_ns: NanoTime,

    /// Digital signature
    pub signature: Option<Signature>,
}

/// Commitment proof
#[derive(Debug, Clone)]
pub struct CommitProof {
    /// Entry hash
    pub entry_hash: [u8; 32],

    /// Chain position
    pub chain_length: u64,

    /// Signature
    pub signature: Signature,
}

/// SIL errors
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Storage error
    #[error("Storage error: {0}")]
    Storage(#[from] storage::StorageError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    Crypto(#[from] ed25519_dalek::ed25519::Error),

    /// Invalid key error
    #[error("Invalid key format")]
    InvalidKey,

    /// Invalid signature
    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Signature is missing from the ledger entry")]
    MissingSignature,

    /// Entry not found
    #[error("Entry not found")]
    EntryNotFound,

    /// Other error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use csf_time::{initialize_simulated_time_source, NanoTime};
    use std::sync::Once;

    /// Initialize test environment with proper time source
    fn setup_test_environment() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            // Initialize simulated time source for deterministic testing
            initialize_simulated_time_source(NanoTime::from_secs(1_700_000_000));

            // Initialize tracing for test debugging
            let _ = tracing_subscriber::fmt()
                .with_test_writer()
                .with_max_level(tracing::Level::DEBUG)
                .try_init();
        });
    }

    #[tokio::test]
    async fn test_sil_commit_verify() {
        setup_test_environment();
        let config = SilConfig::builder().storage(StorageBackend::Memory).build();

        let sil = SilCore::new(config).unwrap();

        // Commit some data
        let packet_id = PacketId::new();
        let data = b"test data";
        let proof = sil.commit(packet_id, data).await.unwrap();

        // Verify the proof
        assert!(sil.verify_proof(&proof).await.is_ok());

        // Check chain state
        let (length, _) = sil.chain_state();
        assert_eq!(length, 1);
    }

    #[tokio::test]
    async fn test_sil_verification_failures() {
        setup_test_environment();
        let config = SilConfig::builder().storage(StorageBackend::Memory).build();
        let sil = SilCore::new(config).unwrap();

        // 1. Commit some data to have a valid proof
        let packet_id = PacketId::new();
        let data = b"test data";
        let proof = sil.commit(packet_id, data).await.unwrap();

        // 2. Test with a non-existent entry hash
        let mut bad_proof_hash = proof.clone();
        bad_proof_hash.entry_hash = [1; 32];
        let result = sil.verify_proof(&bad_proof_hash).await;
        assert!(matches!(result, Err(Error::EntryNotFound)));

        // 3. Test with a tampered signature
        let mut bad_proof_sig = proof.clone();
        // Create a corrupted signature
        let mut sig_bytes = bad_proof_sig.signature.to_bytes();
        sig_bytes[0] ^= 0xff;
        bad_proof_sig.signature = Signature::from_bytes(&sig_bytes);
        let result = sil.verify_proof(&bad_proof_sig).await;
        assert!(matches!(result, Err(Error::InvalidSignature)));
    }

    #[tokio::test]
    async fn test_get_entry_and_public_key() {
        setup_test_environment();
        let config = SilConfig::builder().storage(StorageBackend::Memory).build();
        let sil = SilCore::new(config).unwrap();
        let public_key = sil.public_key();

        let packet_id = PacketId::new();
        let data = b"some important data";
        let proof = sil.commit(packet_id, data).await.unwrap();

        let entry = sil.get_entry(&proof.entry_hash).await.unwrap().unwrap();

        assert_eq!(entry.hash, proof.entry_hash);
        assert_eq!(entry.packet_id, packet_id);
        let signature = entry.signature.unwrap();
        let mut entry_to_verify = entry;
        entry_to_verify.signature = None;
        let entry_bytes = bincode::serialize(&entry_to_verify).unwrap();
        assert!(public_key.verify(&entry_bytes, &signature).is_ok());
    }
}
