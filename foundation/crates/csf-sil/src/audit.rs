use crate::LedgerEntry;
use crate::{storage::StorageError, Storage};
use blake3::Hasher;
use csf_time::{global_time_source, Duration, NanoTime};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Audit trail entry for immutable logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrailEntry {
    /// Timestamp when audit entry was created
    pub timestamp_ns: NanoTime,

    /// Original ledger entry hash for reference
    pub ledger_entry_hash: [u8; 32],

    /// Packet ID being audited
    pub packet_id: crate::PacketId,

    /// Audit entry hash for integrity verification
    pub audit_hash: [u8; 32],

    /// Operation type that triggered the audit
    pub operation: AuditOperation,

    /// Additional context information
    pub context: Option<String>,
}

/// Types of operations that trigger audit entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOperation {
    /// Ledger commit operation
    Commit,
    /// Entry verification
    Verification,
    /// Entry retrieval
    Retrieval,
    /// System operation
    System(String),
}

/// Audit log implementation with persistence and retention
pub struct AuditLog {
    storage: Arc<dyn Storage>,
    retention_days: u32,
    next_audit_id: std::sync::atomic::AtomicU64,
}

impl AuditLog {
    /// Create a new audit log with proper storage backend
    pub fn new(storage: Arc<dyn Storage>, retention_days: u32) -> Result<Self, StorageError> {
        Ok(AuditLog {
            storage,
            retention_days,
            next_audit_id: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Record a commit operation in the audit trail
    pub async fn log_commit(&self, entry: &LedgerEntry) -> Result<(), StorageError> {
        self.log_operation(entry, AuditOperation::Commit, None)
            .await
    }

    /// Record a verification operation in the audit trail
    pub async fn log_verification(
        &self,
        entry: &LedgerEntry,
        success: bool,
    ) -> Result<(), StorageError> {
        let context = Some(format!("verification_result: {}", success));
        self.log_operation(entry, AuditOperation::Verification, context)
            .await
    }

    /// Record an entry retrieval in the audit trail
    pub async fn log_retrieval(&self, entry: &LedgerEntry) -> Result<(), StorageError> {
        self.log_operation(entry, AuditOperation::Retrieval, None)
            .await
    }

    /// Record a system operation in the audit trail
    pub async fn log_system_operation(
        &self,
        operation: &str,
        context: Option<String>,
    ) -> Result<(), StorageError> {
        // Create a synthetic ledger entry for system operations
        let system_entry = LedgerEntry {
            packet_id: crate::PacketId::new(),
            hash: [0; 32], // System operations have no content hash
            timestamp_ns: global_time_source().now_ns().unwrap_or(NanoTime::ZERO),
            signature: None,
        };

        self.log_operation(
            &system_entry,
            AuditOperation::System(operation.to_string()),
            context,
        )
        .await
    }

    /// Internal method to log any audit operation
    async fn log_operation(
        &self,
        entry: &LedgerEntry,
        operation: AuditOperation,
        context: Option<String>,
    ) -> Result<(), StorageError> {
        let timestamp = global_time_source().now_ns().unwrap_or(NanoTime::ZERO);

        // Create audit entry
        let audit_entry = AuditTrailEntry {
            timestamp_ns: timestamp,
            ledger_entry_hash: entry.hash,
            packet_id: entry.packet_id,
            audit_hash: [0; 32], // Will be computed below
            operation,
            context,
        };

        // Compute audit hash for integrity
        let mut audit_entry_with_hash = audit_entry;
        audit_entry_with_hash.audit_hash = self.compute_audit_hash(&audit_entry_with_hash);

        // Serialize and store
        let serialized = bincode::serialize(&audit_entry_with_hash)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let audit_id = self
            .next_audit_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let audit_key = format!("audit_{:016x}", audit_id);

        // Store audit entry
        self.storage
            .store(&audit_key, &serialized)
            .await
            .map_err(|e| StorageError::Write(format!("Failed to store audit entry: {}", e)))?;

        // Log audit entry creation for observability
        tracing::info!(
            audit_id = audit_id,
            operation = ?audit_entry_with_hash.operation,
            packet_id = ?entry.packet_id,
            "Audit trail entry recorded"
        );

        Ok(())
    }

    /// Compute cryptographic hash of audit entry for integrity verification
    fn compute_audit_hash(&self, entry: &AuditTrailEntry) -> [u8; 32] {
        let mut hasher = Hasher::new();

        // Hash key components for audit integrity
        hasher.update(&entry.timestamp_ns.as_nanos().to_le_bytes());
        hasher.update(&entry.ledger_entry_hash);

        // Serialize PacketId for hashing (using bincode)
        let packet_id_bytes = bincode::serialize(&entry.packet_id).unwrap_or_default();
        hasher.update(&packet_id_bytes);

        // Hash operation type
        let operation_bytes = bincode::serialize(&entry.operation).unwrap_or_default();
        hasher.update(&operation_bytes);

        // Hash context if present
        if let Some(context) = &entry.context {
            hasher.update(context.as_bytes());
        }

        *hasher.finalize().as_bytes()
    }

    /// Retrieve audit entries for a specific packet ID
    pub async fn get_audit_trail(
        &self,
        packet_id: crate::PacketId,
    ) -> Result<Vec<AuditTrailEntry>, StorageError> {
        // This would implement retrieval logic based on storage backend
        // For now, return empty vector as this requires indexed storage
        tracing::warn!(
            "Audit trail retrieval not yet implemented for packet {:?}",
            packet_id
        );
        Ok(Vec::new())
    }

    /// Clean up old audit entries based on retention policy
    pub async fn cleanup_old_entries(&self) -> Result<u64, StorageError> {
        let retention_duration = Duration::from_secs((self.retention_days as u64) * 24 * 60 * 60);
        let now = global_time_source().now_ns().unwrap_or(NanoTime::ZERO);
        let cutoff_time =
            NanoTime::from_nanos(now.as_nanos().saturating_sub(retention_duration.as_nanos()));

        tracing::info!(
            retention_days = self.retention_days,
            cutoff_time = cutoff_time.as_nanos(),
            "Starting audit log cleanup"
        );

        // Return 0 for now - actual cleanup would iterate through stored audit entries
        Ok(0)
    }
}
