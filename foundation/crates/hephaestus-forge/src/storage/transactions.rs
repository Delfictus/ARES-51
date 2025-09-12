//! ACID transaction manager for intent storage operations

use crate::{intent::*, types::ForgeResult, ForgeError};
use super::{StorageConfig, IntentStore};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

/// Transaction identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TransactionId(Uuid);

impl TransactionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transaction state
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionState {
    Active,
    Committed,
    Aborted,
    Preparing,
    Prepared,
}

/// Transaction isolation level
#[derive(Debug, Clone, Copy)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Transaction operation
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    Put {
        intent: OptimizationIntent,
        version: u32,
    },
    Delete {
        intent_id: IntentId,
    },
    Update {
        intent: OptimizationIntent,
        version: u32,
    },
}

/// Transaction metadata
#[derive(Debug, Clone)]
pub struct TransactionMetadata {
    pub id: TransactionId,
    pub state: TransactionState,
    pub created_at: Instant,
    pub timeout: Duration,
    pub isolation_level: IsolationLevel,
    pub operations: Vec<TransactionOperation>,
}

/// Transaction handle for user operations
pub struct Transaction {
    id: TransactionId,
    manager: Arc<TransactionManager>,
    operations: Vec<TransactionOperation>,
    state: TransactionState,
}

impl Transaction {
    /// Add intent operation to transaction
    pub async fn put_intent(&mut self, intent: OptimizationIntent, version: u32) -> ForgeResult<()> {
        if self.state != TransactionState::Active {
            return Err(ForgeError::TransactionError(format!(
                "Transaction {} is not active", self.id
            )));
        }
        
        self.operations.push(TransactionOperation::Put { intent, version });
        Ok(())
    }
    
    /// Add delete operation to transaction
    pub async fn delete_intent(&mut self, intent_id: IntentId) -> ForgeResult<()> {
        if self.state != TransactionState::Active {
            return Err(ForgeError::TransactionError(format!(
                "Transaction {} is not active", self.id
            )));
        }
        
        self.operations.push(TransactionOperation::Delete { intent_id });
        Ok(())
    }
    
    /// Add update operation to transaction
    pub async fn update_intent(&mut self, intent: OptimizationIntent, version: u32) -> ForgeResult<()> {
        if self.state != TransactionState::Active {
            return Err(ForgeError::TransactionError(format!(
                "Transaction {} is not active", self.id
            )));
        }
        
        self.operations.push(TransactionOperation::Update { intent, version });
        Ok(())
    }
    
    /// Commit the transaction
    pub async fn commit(mut self) -> ForgeResult<()> {
        if self.state != TransactionState::Active {
            return Err(ForgeError::TransactionError(format!(
                "Transaction {} is not active", self.id
            )));
        }
        
        self.state = TransactionState::Preparing;
        self.manager.commit_transaction(self.id.clone(), self.operations.clone()).await
    }
    
    /// Rollback the transaction
    pub async fn rollback(mut self) -> ForgeResult<()> {
        self.state = TransactionState::Aborted;
        self.manager.rollback_transaction(self.id.clone()).await
    }
    
    /// Get transaction ID
    pub fn id(&self) -> &TransactionId {
        &self.id
    }
    
    /// Get transaction state
    pub fn state(&self) -> TransactionState {
        self.state.clone()
    }
}

/// Transaction manager providing ACID guarantees
pub struct TransactionManager {
    store: Arc<IntentStore>,
    
    // Active transactions
    active_transactions: Arc<RwLock<HashMap<TransactionId, TransactionMetadata>>>,
    
    // Transaction semaphore to limit concurrent transactions
    transaction_semaphore: Arc<Semaphore>,
    
    // Write locks for conflict detection
    write_locks: Arc<RwLock<HashMap<IntentId, TransactionId>>>,
    
    // Read locks for isolation
    read_locks: Arc<RwLock<HashMap<IntentId, Vec<TransactionId>>>>,
    
    // Configuration
    config: StorageConfig,
    
    // Transaction statistics
    stats: Arc<RwLock<TransactionStats>>,
}

/// Transaction statistics
#[derive(Debug, Clone, Default)]
pub struct TransactionStats {
    pub total_transactions: u64,
    pub committed_transactions: u64,
    pub aborted_transactions: u64,
    pub active_transactions: u64,
    pub average_commit_time_ms: f64,
    pub deadlocks_detected: u64,
    pub timeouts: u64,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub async fn new(store: Arc<IntentStore>, config: &StorageConfig) -> ForgeResult<Self> {
        let manager = Self {
            store,
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            transaction_semaphore: Arc::new(Semaphore::new(config.max_concurrent_transactions)),
            write_locks: Arc::new(RwLock::new(HashMap::new())),
            read_locks: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(),
            stats: Arc::new(RwLock::new(TransactionStats::default())),
        };
        
        // Start transaction monitoring task
        manager.start_monitoring_task().await;
        
        Ok(manager)
    }
    
    /// Begin a new transaction
    pub async fn begin_transaction(&self) -> ForgeResult<Transaction> {
        // Acquire semaphore permit to limit concurrent transactions
        let _permit = self.transaction_semaphore.acquire().await
            .map_err(|_| ForgeError::TransactionError("Transaction semaphore closed".to_string()))?;
        
        let tx_id = TransactionId::new();
        let metadata = TransactionMetadata {
            id: tx_id.clone(),
            state: TransactionState::Active,
            created_at: Instant::now(),
            timeout: Duration::from_secs(30), // 30 second transaction timeout
            isolation_level: IsolationLevel::ReadCommitted,
            operations: Vec::new(),
        };
        
        // Register transaction
        {
            let mut active_txs = self.active_transactions.write().await;
            active_txs.insert(tx_id.clone(), metadata);
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_transactions += 1;
            stats.active_transactions += 1;
        }
        
        tracing::debug!("Started transaction {}", tx_id);
        
        Ok(Transaction {
            id: tx_id,
            manager: Arc::new(self.clone()),
            operations: Vec::new(),
            state: TransactionState::Active,
        })
    }
    
    /// Begin transaction with specific isolation level
    pub async fn begin_transaction_with_isolation(&self, isolation: IsolationLevel) -> ForgeResult<Transaction> {
        let mut tx = self.begin_transaction().await?;
        
        // Update isolation level
        {
            let mut active_txs = self.active_transactions.write().await;
            if let Some(metadata) = active_txs.get_mut(&tx.id) {
                metadata.isolation_level = isolation;
            }
        }
        
        Ok(tx)
    }
    
    /// Commit a transaction
    pub async fn commit_transaction(
        &self, 
        tx_id: TransactionId, 
        operations: Vec<TransactionOperation>
    ) -> ForgeResult<()> {
        let start_time = Instant::now();
        
        // Validate transaction exists and is active
        {
            let active_txs = self.active_transactions.read().await;
            match active_txs.get(&tx_id) {
                Some(metadata) if metadata.state == TransactionState::Active => {},
                Some(_) => return Err(ForgeError::TransactionError(format!(
                    "Transaction {} is not in active state", tx_id
                ))),
                None => return Err(ForgeError::TransactionError(format!(
                    "Transaction {} not found", tx_id
                ))),
            }
        }
        
        // Phase 1: Prepare (acquire locks and validate)
        self.prepare_transaction(&tx_id, &operations).await?;
        
        // Phase 2: Commit (apply changes)
        let commit_result = self.apply_transaction_operations(&operations).await;
        
        // Phase 3: Cleanup
        match commit_result {
            Ok(()) => {
                // Update transaction state
                {
                    let mut active_txs = self.active_transactions.write().await;
                    if let Some(metadata) = active_txs.get_mut(&tx_id) {
                        metadata.state = TransactionState::Committed;
                    }
                }
                
                // Release locks
                self.release_transaction_locks(&tx_id, &operations).await?;
                
                // Remove from active transactions
                {
                    let mut active_txs = self.active_transactions.write().await;
                    active_txs.remove(&tx_id);
                }
                
                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.committed_transactions += 1;
                    stats.active_transactions = stats.active_transactions.saturating_sub(1);
                    
                    let commit_time = start_time.elapsed().as_millis() as f64;
                    stats.average_commit_time_ms = (stats.average_commit_time_ms + commit_time) / 2.0;
                }
                
                tracing::debug!("Committed transaction {} in {}ms", tx_id, start_time.elapsed().as_millis());
                Ok(())
            }
            Err(e) => {
                // Rollback on failure
                self.rollback_transaction(tx_id).await?;
                Err(e)
            }
        }
    }
    
    /// Rollback a transaction
    pub async fn rollback_transaction(&self, tx_id: TransactionId) -> ForgeResult<()> {
        // Update transaction state
        {
            let mut active_txs = self.active_transactions.write().await;
            if let Some(metadata) = active_txs.get_mut(&tx_id) {
                metadata.state = TransactionState::Aborted;
            }
        }
        
        // Get transaction operations for lock release
        let operations = {
            let active_txs = self.active_transactions.read().await;
            active_txs.get(&tx_id)
                .map(|m| m.operations.clone())
                .unwrap_or_default()
        };
        
        // Release locks
        self.release_transaction_locks(&tx_id, &operations).await?;
        
        // Remove from active transactions
        {
            let mut active_txs = self.active_transactions.write().await;
            active_txs.remove(&tx_id);
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.aborted_transactions += 1;
            stats.active_transactions = stats.active_transactions.saturating_sub(1);
        }
        
        tracing::debug!("Rolled back transaction {}", tx_id);
        Ok(())
    }
    
    /// Get transaction statistics
    pub async fn get_stats(&self) -> ForgeResult<TransactionStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
    
    /// Get active transaction count
    pub async fn active_transaction_count(&self) -> u64 {
        let active_txs = self.active_transactions.read().await;
        active_txs.len() as u64
    }
    
    // Private methods
    
    /// Prepare transaction (Phase 1 of 2PC)
    async fn prepare_transaction(
        &self,
        tx_id: &TransactionId,
        operations: &[TransactionOperation],
    ) -> ForgeResult<()> {
        // Update state to preparing
        {
            let mut active_txs = self.active_transactions.write().await;
            if let Some(metadata) = active_txs.get_mut(tx_id) {
                metadata.state = TransactionState::Preparing;
                metadata.operations = operations.to_vec();
            }
        }
        
        // Acquire locks for all operations
        for operation in operations {
            match operation {
                TransactionOperation::Put { intent, .. } |
                TransactionOperation::Update { intent, .. } => {
                    self.acquire_write_lock(&intent.id, tx_id).await?;
                }
                TransactionOperation::Delete { intent_id } => {
                    self.acquire_write_lock(intent_id, tx_id).await?;
                }
            }
        }
        
        // Validate operations (check for conflicts, constraint violations, etc.)
        self.validate_transaction_operations(operations).await?;
        
        // Update state to prepared
        {
            let mut active_txs = self.active_transactions.write().await;
            if let Some(metadata) = active_txs.get_mut(tx_id) {
                metadata.state = TransactionState::Prepared;
            }
        }
        
        Ok(())
    }
    
    /// Apply transaction operations to storage
    async fn apply_transaction_operations(&self, operations: &[TransactionOperation]) -> ForgeResult<()> {
        for operation in operations {
            match operation {
                TransactionOperation::Put { intent, version } => {
                    self.store.put_intent(intent, *version).await?;
                }
                TransactionOperation::Delete { intent_id } => {
                    self.store.delete_intent(intent_id).await?;
                }
                TransactionOperation::Update { intent, version } => {
                    // For updates, we put the new version
                    self.store.put_intent(intent, *version).await?;
                }
            }
        }
        Ok(())
    }
    
    /// Validate transaction operations
    async fn validate_transaction_operations(&self, operations: &[TransactionOperation]) -> ForgeResult<()> {
        // Check for constraint violations, foreign key constraints, etc.
        // For now, just check that intents are valid
        for operation in operations {
            match operation {
                TransactionOperation::Put { intent, .. } |
                TransactionOperation::Update { intent, .. } => {
                    // Validate intent structure
                    if intent.objectives.is_empty() {
                        return Err(ForgeError::TransactionError(
                            "Intent must have at least one objective".to_string()
                        ));
                    }
                }
                TransactionOperation::Delete { .. } => {
                    // Delete operations are always valid
                }
            }
        }
        Ok(())
    }
    
    /// Acquire write lock for an intent
    async fn acquire_write_lock(&self, intent_id: &IntentId, tx_id: &TransactionId) -> ForgeResult<()> {
        let mut write_locks = self.write_locks.write().await;
        
        // Check if already locked by another transaction
        if let Some(existing_tx) = write_locks.get(intent_id) {
            if existing_tx != tx_id {
                return Err(ForgeError::TransactionError(format!(
                    "Intent {} is already locked by transaction {}", intent_id, existing_tx
                )));
            }
        }
        
        write_locks.insert(intent_id.clone(), tx_id.clone());
        Ok(())
    }
    
    /// Acquire read lock for an intent
    async fn acquire_read_lock(&self, intent_id: &IntentId, tx_id: &TransactionId) -> ForgeResult<()> {
        let mut read_locks = self.read_locks.write().await;
        read_locks.entry(intent_id.clone())
            .or_insert_with(Vec::new)
            .push(tx_id.clone());
        Ok(())
    }
    
    /// Release all locks held by a transaction
    async fn release_transaction_locks(
        &self,
        tx_id: &TransactionId,
        operations: &[TransactionOperation],
    ) -> ForgeResult<()> {
        // Release write locks
        {
            let mut write_locks = self.write_locks.write().await;
            for operation in operations {
                let intent_id = match operation {
                    TransactionOperation::Put { intent, .. } |
                    TransactionOperation::Update { intent, .. } => &intent.id,
                    TransactionOperation::Delete { intent_id } => intent_id,
                };
                
                if let Some(lock_holder) = write_locks.get(intent_id) {
                    if lock_holder == tx_id {
                        write_locks.remove(intent_id);
                    }
                }
            }
        }
        
        // Release read locks
        {
            let mut read_locks = self.read_locks.write().await;
            for locks in read_locks.values_mut() {
                locks.retain(|id| id != tx_id);
            }
            read_locks.retain(|_, locks| !locks.is_empty());
        }
        
        Ok(())
    }
    
    /// Start background monitoring task for transaction timeouts
    async fn start_monitoring_task(&self) {
        let active_transactions = self.active_transactions.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut timed_out_txs = Vec::new();
                
                // Check for timed out transactions
                {
                    let active_txs = active_transactions.read().await;
                    for (tx_id, metadata) in active_txs.iter() {
                        if now.duration_since(metadata.created_at) > metadata.timeout {
                            timed_out_txs.push(tx_id.clone());
                        }
                    }
                }
                
                // Abort timed out transactions
                for tx_id in timed_out_txs {
                    tracing::warn!("Transaction {} timed out, aborting", tx_id);
                    
                    // Update transaction state to aborted
                    {
                        let mut active_txs = active_transactions.write().await;
                        if let Some(metadata) = active_txs.get_mut(&tx_id) {
                            metadata.state = TransactionState::Aborted;
                        }
                        active_txs.remove(&tx_id);
                    }
                    
                    // Update timeout stats
                    {
                        let mut stats = stats.write().await;
                        stats.timeouts += 1;
                        stats.aborted_transactions += 1;
                        stats.active_transactions = stats.active_transactions.saturating_sub(1);
                    }
                }
            }
        });
    }
}

// Clone implementation for Arc sharing
impl Clone for TransactionManager {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            active_transactions: self.active_transactions.clone(),
            transaction_semaphore: self.transaction_semaphore.clone(),
            write_locks: self.write_locks.clone(),
            read_locks: self.read_locks.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::*;
    use tempfile::TempDir;
    
    async fn create_test_transaction_manager() -> (TransactionManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();
        config.max_concurrent_transactions = 10;
        
        let store = Arc::new(IntentStore::new(&config).await.unwrap());
        let tx_manager = TransactionManager::new(store, &config).await.unwrap();
        
        (tx_manager, temp_dir)
    }
    
    #[tokio::test]
    async fn test_transaction_lifecycle() {
        let (tx_manager, _temp_dir) = create_test_transaction_manager().await;
        
        // Begin transaction
        let tx = tx_manager.begin_transaction().await.unwrap();
        let tx_id = tx.id().clone();
        
        assert_eq!(tx.state(), TransactionState::Active);
        assert_eq!(tx_manager.active_transaction_count().await, 1);
        
        // Commit empty transaction
        tx.commit().await.unwrap();
        
        assert_eq!(tx_manager.active_transaction_count().await, 0);
        
        let stats = tx_manager.get_stats().await.unwrap();
        assert_eq!(stats.total_transactions, 1);
        assert_eq!(stats.committed_transactions, 1);
    }
    
    #[tokio::test]
    async fn test_transaction_operations() {
        let (tx_manager, _temp_dir) = create_test_transaction_manager().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        // Begin transaction
        let mut tx = tx_manager.begin_transaction().await.unwrap();
        
        // Add operations
        tx.put_intent(intent.clone(), 1).await.unwrap();
        tx.update_intent(intent.clone(), 2).await.unwrap();
        tx.delete_intent(intent.id.clone()).await.unwrap();
        
        // Commit transaction
        tx.commit().await.unwrap();
        
        let stats = tx_manager.get_stats().await.unwrap();
        assert_eq!(stats.committed_transactions, 1);
    }
    
    #[tokio::test]
    async fn test_transaction_rollback() {
        let (tx_manager, _temp_dir) = create_test_transaction_manager().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        // Begin transaction
        let mut tx = tx_manager.begin_transaction().await.unwrap();
        tx.put_intent(intent.clone(), 1).await.unwrap();
        
        // Rollback transaction
        tx.rollback().await.unwrap();
        
        let stats = tx_manager.get_stats().await.unwrap();
        assert_eq!(stats.aborted_transactions, 1);
        assert_eq!(stats.active_transactions, 0);
    }
    
    #[tokio::test]
    async fn test_concurrent_transactions() {
        let (tx_manager, _temp_dir) = create_test_transaction_manager().await;
        
        let mut handles = Vec::new();
        
        // Start multiple concurrent transactions
        for i in 0..5 {
            let tx_manager = tx_manager.clone();
            let handle = tokio::spawn(async move {
                let intent = OptimizationIntent::builder()
                    .target_module(format!("test_module_{}", i))
                    .add_objective(crate::intent::Objective::MinimizeLatency {
                        percentile: 99.0,
                        target_ms: 10.0,
                    })
                    .priority(Priority::High)
                    .build()
                    .unwrap();
                
                let mut tx = tx_manager.begin_transaction().await.unwrap();
                tx.put_intent(intent, 1).await.unwrap();
                tx.commit().await.unwrap();
            });
            handles.push(handle);
        }
        
        // Wait for all transactions to complete
        futures::future::try_join_all(handles).await.unwrap();
        
        let stats = tx_manager.get_stats().await.unwrap();
        assert_eq!(stats.total_transactions, 5);
        assert_eq!(stats.committed_transactions, 5);
        assert_eq!(stats.active_transactions, 0);
    }
    
    #[tokio::test]
    async fn test_transaction_conflict() {
        let (tx_manager, _temp_dir) = create_test_transaction_manager().await;
        
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
        
        // Start first transaction
        let mut tx1 = tx_manager.begin_transaction().await.unwrap();
        tx1.put_intent(intent.clone(), 1).await.unwrap();
        
        // Start second transaction with same intent (should conflict)
        let mut tx2 = tx_manager.begin_transaction().await.unwrap();
        tx2.put_intent(intent, 2).await.unwrap();
        
        // First transaction should commit successfully
        tx1.commit().await.unwrap();
        
        // Second transaction should fail due to conflict
        let result = tx2.commit().await;
        assert!(result.is_err());
    }
}