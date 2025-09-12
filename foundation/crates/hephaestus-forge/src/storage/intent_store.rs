//! Core intent storage implementation using RocksDB for ACID compliance and high performance

use crate::{intent::*, types::ForgeResult, ForgeError};
use super::{StorageConfig, CompressionAlgorithm, SnapshotInfo};
use rocksdb::{DB, Options, ColumnFamily, WriteBatch, Transaction, TransactionDB, TransactionDBOptions, TransactionOptions};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::task;

/// Column family names for different data types
const CF_INTENTS: &str = "intents";
const CF_VERSIONS: &str = "versions";
const CF_METADATA: &str = "metadata";
const CF_INDEX_PRIORITY: &str = "index_priority";
const CF_INDEX_TARGET: &str = "index_target";
const CF_INDEX_TIMESTAMP: &str = "index_timestamp";

/// Core intent store implementation
pub struct IntentStore {
    db: Arc<TransactionDB>,
    config: StorageConfig,
}

impl IntentStore {
    /// Create a new intent store
    pub async fn new(config: &StorageConfig) -> ForgeResult<Self> {
        let db = task::spawn_blocking({
            let config = config.clone();
            move || -> ForgeResult<TransactionDB> {
                let mut opts = Options::default();
                opts.create_if_missing(true);
                opts.create_missing_column_families(true);
                
                // Performance optimizations
                opts.set_max_open_files(10000);
                opts.set_use_fsync(false);
                opts.set_bytes_per_sync(1048576);
                opts.set_max_background_jobs(6);
                opts.set_max_subcompactions(2);
                
                // Memory settings
                opts.set_write_buffer_size(config.indexing.write_buffer_mb * 1024 * 1024);
                opts.set_target_file_size_base(64 * 1024 * 1024);
                opts.set_level_zero_file_num_compaction_trigger(8);
                opts.set_level_zero_slowdown_writes_trigger(17);
                opts.set_level_zero_stop_writes_trigger(24);
                opts.set_num_levels(config.indexing.max_levels);
                
                // Compression settings
                if config.compression.enabled {
                    let compression_type = match config.compression.algorithm {
                        CompressionAlgorithm::Snappy => rocksdb::DBCompressionType::Snappy,
                        CompressionAlgorithm::Lz4 => rocksdb::DBCompressionType::Lz4,
                        CompressionAlgorithm::Zstd => rocksdb::DBCompressionType::Zstd,
                    };
                    opts.set_compression_type(compression_type);
                }
                
                // Block-based table options
                let mut block_opts = rocksdb::BlockBasedOptions::default();
                block_opts.set_block_size(16 * 1024);
                block_opts.set_cache_index_and_filter_blocks(true);
                block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
                
                if config.indexing.bloom_filters {
                    block_opts.set_bloom_filter(10.0, false);
                }
                
                // Block cache
                let cache = rocksdb::Cache::new_lru_cache(config.indexing.block_cache_mb * 1024 * 1024);
                block_opts.set_block_cache(&cache);
                opts.set_block_based_table_factory(&block_opts);
                
                // Transaction DB options
                let txn_db_opts = TransactionDBOptions::default();
                
                // Column families
                let column_families = vec![
                    CF_INTENTS,
                    CF_VERSIONS, 
                    CF_METADATA,
                    CF_INDEX_PRIORITY,
                    CF_INDEX_TARGET,
                    CF_INDEX_TIMESTAMP,
                ];
                
                TransactionDB::open(&opts, &txn_db_opts, &config.db_path, column_families)
                    .map_err(|e| ForgeError::StorageError(format!("Failed to open database: {}", e)))
            }
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Failed to initialize database: {}", e)))??;
        
        Ok(Self {
            db: Arc::new(db),
            config: config.clone(),
        })
    }
    
    /// Store an optimization intent with version
    pub async fn put_intent(&self, intent: &OptimizationIntent, version: u32) -> ForgeResult<()> {
        let db = self.db.clone();
        let intent_data = bincode::serialize(intent)
            .map_err(|e| ForgeError::StorageError(format!("Serialization failed: {}", e)))?;
        let intent_key = Self::intent_key(&intent.id);
        let version_key = Self::version_key(&intent.id, version);
        
        task::spawn_blocking(move || -> ForgeResult<()> {
            let txn_opts = TransactionOptions::default();
            let txn = db.transaction_opt(&rocksdb::WriteOptions::default(), &txn_opts);
            
            // Store intent in main CF
            let cf_intents = db.cf_handle(CF_INTENTS)
                .ok_or_else(|| ForgeError::StorageError("CF_INTENTS not found".to_string()))?;
            txn.put_cf(&cf_intents, &intent_key, &intent_data)
                .map_err(|e| ForgeError::StorageError(format!("Failed to store intent: {}", e)))?;
            
            // Store versioned copy
            let cf_versions = db.cf_handle(CF_VERSIONS)
                .ok_or_else(|| ForgeError::StorageError("CF_VERSIONS not found".to_string()))?;
            txn.put_cf(&cf_versions, &version_key, &intent_data)
                .map_err(|e| ForgeError::StorageError(format!("Failed to store version: {}", e)))?;
            
            // Update indexes
            Self::update_indexes(&txn, &db, intent, version)?;
            
            // Store metadata
            Self::store_metadata(&txn, &db, intent, version)?;
            
            txn.commit()
                .map_err(|e| ForgeError::StorageError(format!("Transaction commit failed: {}", e)))?;
                
            Ok(())
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Database operation failed: {}", e)))?
    }
    
    /// Retrieve an optimization intent by ID
    pub async fn get_intent(&self, intent_id: &IntentId) -> ForgeResult<Option<OptimizationIntent>> {
        let db = self.db.clone();
        let intent_key = Self::intent_key(intent_id);
        
        task::spawn_blocking(move || -> ForgeResult<Option<OptimizationIntent>> {
            let cf_intents = db.cf_handle(CF_INTENTS)
                .ok_or_else(|| ForgeError::StorageError("CF_INTENTS not found".to_string()))?;
                
            match db.get_cf(&cf_intents, &intent_key) {
                Ok(Some(data)) => {
                    let intent: OptimizationIntent = bincode::deserialize(&data)
                        .map_err(|e| ForgeError::StorageError(format!("Deserialization failed: {}", e)))?;
                    Ok(Some(intent))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(ForgeError::StorageError(format!("Failed to retrieve intent: {}", e))),
            }
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Database operation failed: {}", e)))?
    }
    
    /// Retrieve a specific version of an intent
    pub async fn get_intent_version(&self, intent_id: &IntentId, version: u32) -> ForgeResult<Option<OptimizationIntent>> {
        let db = self.db.clone();
        let version_key = Self::version_key(intent_id, version);
        
        task::spawn_blocking(move || -> ForgeResult<Option<OptimizationIntent>> {
            let cf_versions = db.cf_handle(CF_VERSIONS)
                .ok_or_else(|| ForgeError::StorageError("CF_VERSIONS not found".to_string()))?;
                
            match db.get_cf(&cf_versions, &version_key) {
                Ok(Some(data)) => {
                    let intent: OptimizationIntent = bincode::deserialize(&data)
                        .map_err(|e| ForgeError::StorageError(format!("Deserialization failed: {}", e)))?;
                    Ok(Some(intent))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(ForgeError::StorageError(format!("Failed to retrieve version: {}", e))),
            }
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Database operation failed: {}", e)))?
    }
    
    /// Delete an intent and all its versions
    pub async fn delete_intent(&self, intent_id: &IntentId) -> ForgeResult<()> {
        let db = self.db.clone();
        let intent_key = Self::intent_key(intent_id);
        let intent_id_prefix = format!("{}:", intent_id);
        
        task::spawn_blocking(move || -> ForgeResult<()> {
            let txn_opts = TransactionOptions::default();
            let txn = db.transaction_opt(&rocksdb::WriteOptions::default(), &txn_opts);
            
            // Delete from main CF
            let cf_intents = db.cf_handle(CF_INTENTS)
                .ok_or_else(|| ForgeError::StorageError("CF_INTENTS not found".to_string()))?;
            txn.delete_cf(&cf_intents, &intent_key)
                .map_err(|e| ForgeError::StorageError(format!("Failed to delete intent: {}", e)))?;
            
            // Delete all versions
            let cf_versions = db.cf_handle(CF_VERSIONS)
                .ok_or_else(|| ForgeError::StorageError("CF_VERSIONS not found".to_string()))?;
                
            let iter = db.iterator_cf(&cf_versions, rocksdb::IteratorMode::From(intent_id_prefix.as_bytes(), rocksdb::Direction::Forward));
            for item in iter {
                let (key, _) = item.map_err(|e| ForgeError::StorageError(format!("Iterator error: {}", e)))?;
                let key_str = String::from_utf8_lossy(&key);
                
                if !key_str.starts_with(&intent_id_prefix) {
                    break; // No more versions for this intent
                }
                
                txn.delete_cf(&cf_versions, &key)
                    .map_err(|e| ForgeError::StorageError(format!("Failed to delete version: {}", e)))?;
            }
            
            // Clean up indexes
            Self::cleanup_indexes(&txn, &db, intent_id)?;
            
            txn.commit()
                .map_err(|e| ForgeError::StorageError(format!("Transaction commit failed: {}", e)))?;
                
            Ok(())
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Database operation failed: {}", e)))?
    }
    
    /// Count total intents
    pub async fn count_intents(&self) -> ForgeResult<u64> {
        let db = self.db.clone();
        
        task::spawn_blocking(move || -> ForgeResult<u64> {
            let cf_intents = db.cf_handle(CF_INTENTS)
                .ok_or_else(|| ForgeError::StorageError("CF_INTENTS not found".to_string()))?;
                
            let iter = db.iterator_cf(&cf_intents, rocksdb::IteratorMode::Start);
            let count = iter.count() as u64;
            Ok(count)
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Database operation failed: {}", e)))?
    }
    
    /// Get storage size in bytes
    pub async fn storage_size(&self) -> ForgeResult<u64> {
        let db_path = self.config.db_path.clone();
        
        task::spawn_blocking(move || -> ForgeResult<u64> {
            let mut total_size = 0u64;
            
            fn visit_dir(dir: &Path, total: &mut u64) -> std::io::Result<()> {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dir(&path, total)?;
                    } else {
                        *total += entry.metadata()?.len();
                    }
                }
                Ok(())
            }
            
            visit_dir(Path::new(&db_path), &mut total_size)
                .map_err(|e| ForgeError::StorageError(format!("Failed to calculate size: {}", e)))?;
            
            Ok(total_size)
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Size calculation failed: {}", e)))?
    }
    
    /// Compact database to optimize storage
    pub async fn compact(&self) -> ForgeResult<()> {
        let db = self.db.clone();
        
        task::spawn_blocking(move || -> ForgeResult<()> {
            // Compact each column family
            let column_families = vec![
                CF_INTENTS, CF_VERSIONS, CF_METADATA,
                CF_INDEX_PRIORITY, CF_INDEX_TARGET, CF_INDEX_TIMESTAMP,
            ];
            
            for cf_name in column_families {
                if let Some(cf) = db.cf_handle(cf_name) {
                    db.compact_range_cf(&cf, None::<&[u8]>, None::<&[u8]>);
                }
            }
            
            Ok(())
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Compaction failed: {}", e)))?
    }
    
    /// Create a database snapshot
    pub async fn create_snapshot(&self, path: &Path) -> ForgeResult<SnapshotInfo> {
        let db = self.db.clone();
        let path = path.to_path_buf();
        let created_at = chrono::Utc::now();
        
        task::spawn_blocking(move || -> ForgeResult<SnapshotInfo> {
            // Create checkpoint (RocksDB snapshot mechanism)
            let checkpoint = rocksdb::checkpoint::Checkpoint::new(&db)
                .map_err(|e| ForgeError::StorageError(format!("Failed to create checkpoint: {}", e)))?;
            
            std::fs::create_dir_all(&path)
                .map_err(|e| ForgeError::StorageError(format!("Failed to create snapshot directory: {}", e)))?;
                
            checkpoint.create_checkpoint(&path)
                .map_err(|e| ForgeError::StorageError(format!("Failed to create snapshot: {}", e)))?;
            
            // Calculate snapshot size
            let mut size_bytes = 0u64;
            fn visit_dir(dir: &Path, total: &mut u64) -> std::io::Result<()> {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dir(&path, total)?;
                    } else {
                        *total += entry.metadata()?.len();
                    }
                }
                Ok(())
            }
            
            visit_dir(&path, &mut size_bytes)
                .map_err(|e| ForgeError::StorageError(format!("Failed to calculate snapshot size: {}", e)))?;
            
            // Count intents in snapshot
            let temp_opts = Options::default();
            let temp_db = DB::open_for_read_only(&temp_opts, &path, false)
                .map_err(|e| ForgeError::StorageError(format!("Failed to open snapshot for counting: {}", e)))?;
            
            let intent_count = if let Some(cf) = temp_db.cf_handle(CF_INTENTS) {
                temp_db.iterator_cf(&cf, rocksdb::IteratorMode::Start).count() as u64
            } else {
                0
            };
            
            Ok(SnapshotInfo {
                path: path.to_string_lossy().to_string(),
                created_at,
                size_bytes,
                intent_count,
            })
        }).await
        .map_err(|e| ForgeError::StorageError(format!("Snapshot operation failed: {}", e)))?
    }
    
    /// Restore database from snapshot
    pub async fn restore_snapshot(&self, _path: &Path) -> ForgeResult<()> {
        // Note: In a real implementation, this would involve stopping the current DB,
        // replacing files, and reopening. For now, we'll return an error indicating
        // that this requires a restart.
        Err(ForgeError::StorageError(
            "Snapshot restoration requires system restart - use backup/restore tools".to_string()
        ))
    }
    
    // Helper methods
    
    fn intent_key(intent_id: &IntentId) -> Vec<u8> {
        format!("intent:{}", intent_id).into_bytes()
    }
    
    fn version_key(intent_id: &IntentId, version: u32) -> Vec<u8> {
        format!("{}:{:08}", intent_id, version).into_bytes()
    }
    
    fn update_indexes(
        txn: &Transaction<TransactionDB>, 
        db: &TransactionDB, 
        intent: &OptimizationIntent, 
        version: u32
    ) -> ForgeResult<()> {
        // Priority index
        let cf_priority = db.cf_handle(CF_INDEX_PRIORITY)
            .ok_or_else(|| ForgeError::StorageError("CF_INDEX_PRIORITY not found".to_string()))?;
        let priority_key = format!("{}:{}", intent.priority as u8, intent.id);
        txn.put_cf(&cf_priority, priority_key.as_bytes(), &version.to_le_bytes())
            .map_err(|e| ForgeError::StorageError(format!("Failed to update priority index: {}", e)))?;
        
        // Target index
        let cf_target = db.cf_handle(CF_INDEX_TARGET)
            .ok_or_else(|| ForgeError::StorageError("CF_INDEX_TARGET not found".to_string()))?;
        let target_key = match &intent.target {
            crate::intent::OptimizationTarget::Module(id) => format!("module:{}:{}", id.0, intent.id),
            crate::intent::OptimizationTarget::ModuleName(name) => format!("name:{}:{}", name, intent.id),
            crate::intent::OptimizationTarget::ComponentGroup(group) => format!("group:{}:{}", group, intent.id),
            crate::intent::OptimizationTarget::System => format!("system:{}", intent.id),
        };
        txn.put_cf(&cf_target, target_key.as_bytes(), &version.to_le_bytes())
            .map_err(|e| ForgeError::StorageError(format!("Failed to update target index: {}", e)))?;
        
        // Timestamp index
        let cf_timestamp = db.cf_handle(CF_INDEX_TIMESTAMP)
            .ok_or_else(|| ForgeError::StorageError("CF_INDEX_TIMESTAMP not found".to_string()))?;
        let timestamp_key = format!("{}:{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0), intent.id);
        txn.put_cf(&cf_timestamp, timestamp_key.as_bytes(), &version.to_le_bytes())
            .map_err(|e| ForgeError::StorageError(format!("Failed to update timestamp index: {}", e)))?;
        
        Ok(())
    }
    
    fn store_metadata(
        txn: &Transaction<TransactionDB>, 
        db: &TransactionDB, 
        intent: &OptimizationIntent, 
        version: u32
    ) -> ForgeResult<()> {
        let cf_metadata = db.cf_handle(CF_METADATA)
            .ok_or_else(|| ForgeError::StorageError("CF_METADATA not found".to_string()))?;
        
        let metadata = IntentMetadata {
            intent_id: intent.id.clone(),
            version,
            created_at: chrono::Utc::now(),
            objectives_count: intent.objectives.len(),
            constraints_count: intent.constraints.len(),
            priority: intent.priority,
        };
        
        let metadata_key = format!("meta:{}:{:08}", intent.id, version);
        let metadata_data = bincode::serialize(&metadata)
            .map_err(|e| ForgeError::StorageError(format!("Failed to serialize metadata: {}", e)))?;
            
        txn.put_cf(&cf_metadata, metadata_key.as_bytes(), &metadata_data)
            .map_err(|e| ForgeError::StorageError(format!("Failed to store metadata: {}", e)))?;
        
        Ok(())
    }
    
    fn cleanup_indexes(
        txn: &Transaction<TransactionDB>, 
        db: &TransactionDB, 
        intent_id: &IntentId
    ) -> ForgeResult<()> {
        let intent_id_str = intent_id.to_string();
        
        // Clean priority index
        if let Some(cf_priority) = db.cf_handle(CF_INDEX_PRIORITY) {
            let iter = db.iterator_cf(&cf_priority, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, _) = item.map_err(|e| ForgeError::StorageError(format!("Iterator error: {}", e)))?;
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(&intent_id_str) {
                    txn.delete_cf(&cf_priority, &key)
                        .map_err(|e| ForgeError::StorageError(format!("Failed to delete priority index: {}", e)))?;
                }
            }
        }
        
        // Clean target index
        if let Some(cf_target) = db.cf_handle(CF_INDEX_TARGET) {
            let iter = db.iterator_cf(&cf_target, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, _) = item.map_err(|e| ForgeError::StorageError(format!("Iterator error: {}", e)))?;
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(&intent_id_str) {
                    txn.delete_cf(&cf_target, &key)
                        .map_err(|e| ForgeError::StorageError(format!("Failed to delete target index: {}", e)))?;
                }
            }
        }
        
        // Clean timestamp index
        if let Some(cf_timestamp) = db.cf_handle(CF_INDEX_TIMESTAMP) {
            let iter = db.iterator_cf(&cf_timestamp, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, _) = item.map_err(|e| ForgeError::StorageError(format!("Iterator error: {}", e)))?;
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(&intent_id_str) {
                    txn.delete_cf(&cf_timestamp, &key)
                        .map_err(|e| ForgeError::StorageError(format!("Failed to delete timestamp index: {}", e)))?;
                }
            }
        }
        
        // Clean metadata
        if let Some(cf_metadata) = db.cf_handle(CF_METADATA) {
            let prefix = format!("meta:{}:", intent_id);
            let iter = db.iterator_cf(&cf_metadata, rocksdb::IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
            for item in iter {
                let (key, _) = item.map_err(|e| ForgeError::StorageError(format!("Iterator error: {}", e)))?;
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                txn.delete_cf(&cf_metadata, &key)
                    .map_err(|e| ForgeError::StorageError(format!("Failed to delete metadata: {}", e)))?;
            }
        }
        
        Ok(())
    }
}

/// Metadata for stored intents
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IntentMetadata {
    intent_id: IntentId,
    version: u32,
    created_at: chrono::DateTime<chrono::Utc>,
    objectives_count: usize,
    constraints_count: usize,
    priority: Priority,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_store() -> (IntentStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();
        
        let store = IntentStore::new(&config).await.unwrap();
        (store, temp_dir)
    }
    
    #[tokio::test]
    async fn test_store_and_retrieve() {
        let (store, _temp_dir) = create_test_store().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Store intent
        store.put_intent(&intent, 1).await.unwrap();
        
        // Retrieve intent
        let retrieved = store.get_intent(&intent_id).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, intent_id);
        assert_eq!(retrieved.priority, Priority::High);
    }
    
    #[tokio::test]
    async fn test_versioning() {
        let (store, _temp_dir) = create_test_store().await;
        
        let mut intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::Medium)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Store version 1
        store.put_intent(&intent, 1).await.unwrap();
        
        // Store version 2 with different priority
        intent.priority = Priority::High;
        store.put_intent(&intent, 2).await.unwrap();
        
        // Retrieve latest version
        let latest = store.get_intent(&intent_id).await.unwrap().unwrap();
        assert_eq!(latest.priority, Priority::High);
        
        // Retrieve specific version
        let v1 = store.get_intent_version(&intent_id, 1).await.unwrap().unwrap();
        assert_eq!(v1.priority, Priority::Medium);
        
        let v2 = store.get_intent_version(&intent_id, 2).await.unwrap().unwrap();
        assert_eq!(v2.priority, Priority::High);
    }
    
    #[tokio::test]
    async fn test_delete() {
        let (store, _temp_dir) = create_test_store().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Store intent with multiple versions
        store.put_intent(&intent, 1).await.unwrap();
        store.put_intent(&intent, 2).await.unwrap();
        
        // Verify it exists
        assert!(store.get_intent(&intent_id).await.unwrap().is_some());
        
        // Delete intent
        store.delete_intent(&intent_id).await.unwrap();
        
        // Verify it's deleted
        assert!(store.get_intent(&intent_id).await.unwrap().is_none());
        assert!(store.get_intent_version(&intent_id, 1).await.unwrap().is_none());
        assert!(store.get_intent_version(&intent_id, 2).await.unwrap().is_none());
    }
}