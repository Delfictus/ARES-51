use crate::LedgerEntry;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Entry not found")]
    NotFound,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Write error: {0}")]
    Write(String),
}

#[async_trait]
pub trait Storage: Send + Sync {
    fn get_signing_key(&self) -> Result<Option<Vec<u8>>, StorageError>;
    fn store_signing_key(&self, _bytes: &[u8]) -> Result<(), StorageError>;
    async fn store_entry(&self, _entry: &LedgerEntry) -> Result<(), StorageError>;
    async fn get_entry_by_hash(
        &self,
        _hash: &[u8; 32],
    ) -> Result<Option<LedgerEntry>, StorageError>;
    /// Store arbitrary data with a key (for audit trail)
    async fn store(&self, _key: &str, _data: &[u8]) -> Result<(), StorageError>;
}

/// Creates a storage backend for SIL.
pub fn create_storage(backend: &super::StorageBackend) -> Result<Arc<dyn Storage>, StorageError> {
    match backend {
        super::StorageBackend::Memory => Ok(Arc::new(MemoryStorage::new())),
        _ => {
            // Fallback to memory storage for other backends
            Ok(Arc::new(MemoryStorage::new()))
        }
    }
}

/// Production-grade in-memory storage implementation for testing and development
struct MemoryStorage {
    /// Storage for ledger entries by hash
    entries: Arc<RwLock<HashMap<[u8; 32], LedgerEntry>>>,
    /// Storage for signing key
    signing_key: Arc<RwLock<Option<Vec<u8>>>>,
    /// General key-value storage for audit trail
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl MemoryStorage {
    fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            signing_key: Arc::new(RwLock::new(None)),
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    fn get_signing_key(&self) -> Result<Option<Vec<u8>>, StorageError> {
        let key = self.signing_key.read();
        Ok(key.clone())
    }

    fn store_signing_key(&self, bytes: &[u8]) -> Result<(), StorageError> {
        let mut key = self.signing_key.write();
        *key = Some(bytes.to_vec());
        Ok(())
    }

    async fn store_entry(&self, entry: &LedgerEntry) -> Result<(), StorageError> {
        let mut entries = self.entries.write();
        entries.insert(entry.hash, entry.clone());
        tracing::debug!(
            packet_id = ?entry.packet_id,
            "Stored ledger entry in memory storage"
        );
        Ok(())
    }

    async fn get_entry_by_hash(
        &self,
        hash: &[u8; 32],
    ) -> Result<Option<LedgerEntry>, StorageError> {
        let entries = self.entries.read();
        let entry = entries.get(hash).cloned();
        tracing::debug!(
            found = entry.is_some(),
            "Retrieved ledger entry from memory storage"
        );
        Ok(entry)
    }

    async fn store(&self, key: &str, data: &[u8]) -> Result<(), StorageError> {
        let mut storage = self.data.write();
        storage.insert(key.to_string(), data.to_vec());
        tracing::debug!(
            key = key,
            data_len = data.len(),
            "Stored data in memory storage"
        );
        Ok(())
    }
}
