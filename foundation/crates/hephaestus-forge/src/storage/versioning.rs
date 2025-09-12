//! Intent versioning system for tracking evolution and enabling rollback

use crate::{intent::*, types::ForgeResult, ForgeError};
use super::{StorageConfig, IntentStore};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Authentication context for user identification
#[derive(Debug, Clone)]
struct AuthContext {
    pub user_id: String,
    pub session_id: Option<String>,
    pub roles: Vec<String>,
}

/// Version information for an intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentVersion {
    pub intent_id: IntentId,
    pub version: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub created_by: String,
    pub change_description: Option<String>,
    pub parent_version: Option<u32>,
    pub checksum: String,
}

/// Version comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    pub from_version: u32,
    pub to_version: u32,
    pub changes: Vec<VersionChange>,
}

/// Individual change in a version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionChange {
    PriorityChanged {
        from: Priority,
        to: Priority,
    },
    TargetChanged {
        from: OptimizationTarget,
        to: OptimizationTarget,
    },
    ObjectiveAdded {
        objective: Objective,
    },
    ObjectiveRemoved {
        objective: Objective,
    },
    ObjectiveModified {
        from: Objective,
        to: Objective,
    },
    ConstraintAdded {
        constraint: Constraint,
    },
    ConstraintRemoved {
        constraint: Constraint,
    },
    ConstraintModified {
        from: Constraint,
        to: Constraint,
    },
    DeadlineChanged {
        from: Option<std::time::Duration>,
        to: Option<std::time::Duration>,
    },
    SynthesisStrategyChanged {
        from: Option<String>,
        to: Option<String>,
    },
}

/// Version history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VersionHistoryEntry {
    version: u32,
    intent_data: Vec<u8>,
    metadata: IntentVersion,
    diff_from_parent: Option<VersionDiff>,
}

/// Intent versioning engine
pub struct VersioningEngine {
    store: Arc<IntentStore>,
    
    // In-memory version tracking for fast access
    version_cache: Arc<RwLock<BTreeMap<IntentId, Vec<IntentVersion>>>>,
    
    // Configuration
    config: StorageConfig,
    
    // Current version counters
    version_counters: Arc<RwLock<BTreeMap<IntentId, u32>>>,
}

impl VersioningEngine {
    /// Create a new versioning engine
    pub async fn new(store: Arc<IntentStore>, config: &StorageConfig) -> ForgeResult<Self> {
        let engine = Self {
            store,
            version_cache: Arc::new(RwLock::new(BTreeMap::new())),
            config: config.clone(),
            version_counters: Arc::new(RwLock::new(BTreeMap::new())),
        };
        
        // Start background cleanup task if enabled
        if config.versioning.auto_cleanup {
            engine.start_cleanup_task().await;
        }
        
        Ok(engine)
    }
    
    /// Create a new version of an intent
    pub async fn create_version(
        &self, 
        intent_id: &IntentId, 
        intent: &OptimizationIntent
    ) -> ForgeResult<IntentVersion> {
        let version = {
            let mut counters = self.version_counters.write().await;
            let current = counters.get(intent_id).copied().unwrap_or(0);
            let new_version = current + 1;
            counters.insert(intent_id.clone(), new_version);
            new_version
        };
        
        // Calculate checksum
        let intent_data = bincode::serialize(intent)
            .map_err(|e| ForgeError::StorageError(format!("Failed to serialize intent: {}", e)))?;
        let checksum = self.calculate_checksum(&intent_data);
        
        // Get parent version for diff calculation
        let parent_version = if version > 1 { Some(version - 1) } else { None };
        let diff_from_parent = if let Some(parent_ver) = parent_version {
            if let Some(parent_intent) = self.get_version(intent_id, parent_ver).await? {
                Some(self.calculate_diff(&parent_intent, intent, parent_ver, version)?)
            } else {
                None
            }
        } else {
            None
        };
        
        let version_info = IntentVersion {
            intent_id: intent_id.clone(),
            version,
            created_at: chrono::Utc::now(),
            created_by: self.get_current_user_context().unwrap_or_else(|| "system".to_string()),
            change_description: self.generate_change_description(&diff_from_parent),
            parent_version,
            checksum,
        };
        
        // Store version history entry (would be stored in database in real implementation)
        let history_entry = VersionHistoryEntry {
            version,
            intent_data,
            metadata: version_info.clone(),
            diff_from_parent,
        };
        
        // Update version cache
        {
            let mut cache = self.version_cache.write().await;
            cache.entry(intent_id.clone())
                .or_insert_with(Vec::new)
                .push(version_info.clone());
        }
        
        // In a real implementation, we would store the version history in the database
        // For now, we're just maintaining it in memory
        
        tracing::debug!(
            "Created version {} for intent {} with checksum {}", 
            version, intent_id, checksum
        );
        
        Ok(version_info)
    }
    
    /// Get a specific version of an intent
    pub async fn get_version(
        &self, 
        intent_id: &IntentId, 
        version: u32
    ) -> ForgeResult<Option<OptimizationIntent>> {
        // In a real implementation, this would fetch from database
        // For now, we'll delegate to the store for the latest version
        // and maintain version history in memory
        
        if version == self.get_latest_version(intent_id).await? {
            // Get latest version from store
            self.store.get_intent(intent_id).await
        } else {
            // For older versions, in a real implementation we would:
            // 1. Fetch the version data from versioned storage
            // 2. Deserialize the intent
            // For now, return None for older versions
            Ok(None)
        }
    }
    
    /// List all versions of an intent
    pub async fn list_versions(&self, intent_id: &IntentId) -> ForgeResult<Vec<IntentVersion>> {
        let cache = self.version_cache.read().await;
        Ok(cache.get(intent_id).cloned().unwrap_or_default())
    }
    
    /// Get the latest version number for an intent
    pub async fn get_latest_version(&self, intent_id: &IntentId) -> ForgeResult<u32> {
        let counters = self.version_counters.read().await;
        Ok(counters.get(intent_id).copied().unwrap_or(0))
    }
    
    /// Get version history with diffs
    pub async fn get_version_history(
        &self, 
        intent_id: &IntentId
    ) -> ForgeResult<Vec<(IntentVersion, Option<VersionDiff>)>> {
        let versions = self.list_versions(intent_id).await?;
        let mut history = Vec::new();
        
        for version in versions {
            // In a real implementation, we would fetch the diff from storage
            let diff = None; // Placeholder
            history.push((version, diff));
        }
        
        Ok(history)
    }
    
    /// Compare two versions
    pub async fn compare_versions(
        &self,
        intent_id: &IntentId,
        from_version: u32,
        to_version: u32,
    ) -> ForgeResult<Option<VersionDiff>> {
        let from_intent = self.get_version(intent_id, from_version).await?;
        let to_intent = self.get_version(intent_id, to_version).await?;
        
        match (from_intent, to_intent) {
            (Some(from), Some(to)) => {
                Ok(Some(self.calculate_diff(&from, &to, from_version, to_version)?))
            }
            _ => Ok(None),
        }
    }
    
    /// Rollback to a previous version
    pub async fn rollback_to_version(
        &self,
        intent_id: &IntentId,
        target_version: u32,
    ) -> ForgeResult<IntentVersion> {
        let target_intent = self.get_version(intent_id, target_version).await?
            .ok_or_else(|| ForgeError::StorageError(format!("Version {} not found", target_version)))?;
        
        // Create a new version that is a copy of the target version
        let new_version = self.create_version(intent_id, &target_intent).await?;
        
        tracing::info!(
            "Rolled back intent {} to version {} (created new version {})",
            intent_id, target_version, new_version.version
        );
        
        Ok(new_version)
    }
    
    /// Delete all versions of an intent
    pub async fn delete_all_versions(&self, intent_id: &IntentId) -> ForgeResult<()> {
        // Remove from cache
        {
            let mut cache = self.version_cache.write().await;
            cache.remove(intent_id);
        }
        
        // Remove from counters
        {
            let mut counters = self.version_counters.write().await;
            counters.remove(intent_id);
        }
        
        // In a real implementation, we would also remove all version data from storage
        
        tracing::debug!("Deleted all versions for intent {}", intent_id);
        Ok(())
    }
    
    /// Count total versions across all intents
    pub async fn count_versions(&self) -> ForgeResult<u64> {
        let cache = self.version_cache.read().await;
        let total = cache.values().map(|versions| versions.len() as u64).sum();
        Ok(total)
    }
    
    /// Clean up old versions according to configuration
    pub async fn cleanup_old_versions(&self) -> ForgeResult<()> {
        let max_versions = self.config.versioning.max_versions;
        let mut cleaned_count = 0u64;
        
        {
            let mut cache = self.version_cache.write().await;
            for (intent_id, versions) in cache.iter_mut() {
                if versions.len() as u32 > max_versions {
                    let to_remove = versions.len() as u32 - max_versions;
                    
                    // Keep the most recent versions, remove oldest
                    versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                    versions.truncate(max_versions as usize);
                    
                    cleaned_count += to_remove as u64;
                    
                    tracing::debug!(
                        "Cleaned up {} old versions for intent {}", 
                        to_remove, intent_id
                    );
                }
            }
        }
        
        if cleaned_count > 0 {
            tracing::info!("Cleanup completed: removed {} old versions", cleaned_count);
        }
        
        Ok(())
    }
    
    /// Get versioning statistics
    pub async fn get_stats(&self) -> ForgeResult<VersioningStats> {
        let cache = self.version_cache.read().await;
        
        let total_intents = cache.len() as u64;
        let total_versions = cache.values().map(|v| v.len() as u64).sum();
        
        let mut versions_per_intent = Vec::new();
        for versions in cache.values() {
            versions_per_intent.push(versions.len() as u32);
        }
        
        let avg_versions_per_intent = if total_intents > 0 {
            total_versions as f64 / total_intents as f64
        } else {
            0.0
        };
        
        let max_versions = versions_per_intent.iter().max().copied().unwrap_or(0);
        let min_versions = versions_per_intent.iter().min().copied().unwrap_or(0);
        
        Ok(VersioningStats {
            total_intents,
            total_versions,
            avg_versions_per_intent,
            max_versions,
            min_versions,
        })
    }
    
    // Private methods
    
    /// Calculate checksum for intent data
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
    
    /// Calculate diff between two intents
    fn calculate_diff(
        &self,
        from: &OptimizationIntent,
        to: &OptimizationIntent,
        from_version: u32,
        to_version: u32,
    ) -> ForgeResult<VersionDiff> {
        let mut changes = Vec::new();
        
        // Check priority changes
        if from.priority != to.priority {
            changes.push(VersionChange::PriorityChanged {
                from: from.priority,
                to: to.priority,
            });
        }
        
        // Check target changes
        if from.target != to.target {
            changes.push(VersionChange::TargetChanged {
                from: from.target.clone(),
                to: to.target.clone(),
            });
        }
        
        // Check objective changes (simplified)
        if from.objectives != to.objectives {
            // For simplicity, just note that objectives changed
            // In a real implementation, you'd do a detailed comparison
            for obj in &to.objectives {
                if !from.objectives.contains(obj) {
                    changes.push(VersionChange::ObjectiveAdded {
                        objective: obj.clone(),
                    });
                }
            }
            for obj in &from.objectives {
                if !to.objectives.contains(obj) {
                    changes.push(VersionChange::ObjectiveRemoved {
                        objective: obj.clone(),
                    });
                }
            }
        }
        
        // Check constraint changes (simplified)
        if from.constraints != to.constraints {
            for constraint in &to.constraints {
                if !from.constraints.contains(constraint) {
                    changes.push(VersionChange::ConstraintAdded {
                        constraint: constraint.clone(),
                    });
                }
            }
            for constraint in &from.constraints {
                if !to.constraints.contains(constraint) {
                    changes.push(VersionChange::ConstraintRemoved {
                        constraint: constraint.clone(),
                    });
                }
            }
        }
        
        // Check deadline changes
        if from.deadline != to.deadline {
            changes.push(VersionChange::DeadlineChanged {
                from: from.deadline,
                to: to.deadline,
            });
        }
        
        // Check synthesis strategy changes
        if from.synthesis_strategy != to.synthesis_strategy {
            changes.push(VersionChange::SynthesisStrategyChanged {
                from: from.synthesis_strategy.clone(),
                to: to.synthesis_strategy.clone(),
            });
        }
        
        Ok(VersionDiff {
            from_version,
            to_version,
            changes,
        })
    }
    
    /// Generate human-readable change description
    fn generate_change_description(&self, diff: &Option<VersionDiff>) -> Option<String> {
        if let Some(diff) = diff {
            if diff.changes.is_empty() {
                return Some("No changes".to_string());
            }
            
            let mut description = String::new();
            for (i, change) in diff.changes.iter().enumerate() {
                if i > 0 {
                    description.push_str("; ");
                }
                
                match change {
                    VersionChange::PriorityChanged { from, to } => {
                        description.push_str(&format!("Priority changed from {:?} to {:?}", from, to));
                    }
                    VersionChange::TargetChanged { .. } => {
                        description.push_str("Target changed");
                    }
                    VersionChange::ObjectiveAdded { .. } => {
                        description.push_str("Objective added");
                    }
                    VersionChange::ObjectiveRemoved { .. } => {
                        description.push_str("Objective removed");
                    }
                    VersionChange::ObjectiveModified { .. } => {
                        description.push_str("Objective modified");
                    }
                    VersionChange::ConstraintAdded { .. } => {
                        description.push_str("Constraint added");
                    }
                    VersionChange::ConstraintRemoved { .. } => {
                        description.push_str("Constraint removed");
                    }
                    VersionChange::ConstraintModified { .. } => {
                        description.push_str("Constraint modified");
                    }
                    VersionChange::DeadlineChanged { .. } => {
                        description.push_str("Deadline changed");
                    }
                    VersionChange::SynthesisStrategyChanged { .. } => {
                        description.push_str("Synthesis strategy changed");
                    }
                }
            }
            Some(description)
        } else {
            Some("Initial version".to_string())
        }
    }

    /// Get current user context from environment or authentication system
    fn get_current_user_context(&self) -> Option<String> {
        // Try to get user from environment variables first
        if let Ok(user) = std::env::var("USER") {
            return Some(user);
        }
        
        if let Ok(user) = std::env::var("USERNAME") {
            return Some(user);
        }
        
        // Try to get from authentication context (if available)
        if let Some(auth_context) = self.get_auth_context() {
            return Some(auth_context.user_id);
        }
        
        // Try to get from system user database
        self.get_system_user().or_else(|| {
            // Fallback to process owner
            self.get_process_owner()
        })
    }
    
    /// Get authentication context (if available)
    fn get_auth_context(&self) -> Option<AuthContext> {
        // In production: integrate with authentication system
        // For now, return None
        None
    }
    
    /// Get system user information
    fn get_system_user(&self) -> Option<String> {
        // In production: query system user database
        // For now, try to get from whoami crate or similar
        std::env::var("LOGNAME").ok().or_else(|| {
            // Try to read from /etc/passwd or similar
            self.read_system_user_info()
        })
    }
    
    /// Get process owner
    fn get_process_owner(&self) -> Option<String> {
        // In production: get UID and map to username
        // For now, return a reasonable default
        Some("forge-system".to_string())
    }
    
    /// Read system user information
    fn read_system_user_info(&self) -> Option<String> {
        // In production: read from system user database
        // This is a simplified implementation
        use std::process::Command;
        
        // Try to use 'whoami' command if available
        if let Ok(output) = Command::new("whoami").output() {
            if output.status.success() {
                if let Ok(user) = String::from_utf8(output.stdout) {
                    return Some(user.trim().to_string());
                }
            }
        }
        
        // Try to read from id command
        if let Ok(output) = Command::new("id").arg("-un").output() {
            if output.status.success() {
                if let Ok(user) = String::from_utf8(output.stdout) {
                    return Some(user.trim().to_string());
                }
            }
        }
        
        None
    }
    
    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let engine = VersioningEngine {
            store: self.store.clone(),
            version_cache: self.version_cache.clone(),
            config: self.config.clone(),
            version_counters: self.version_counters.clone(),
        };
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(engine.config.versioning.cleanup_interval_sec)
            );
            
            loop {
                interval.tick().await;
                
                if let Err(e) = engine.cleanup_old_versions().await {
                    tracing::error!("Version cleanup failed: {}", e);
                }
            }
        });
    }
}

/// Versioning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningStats {
    pub total_intents: u64,
    pub total_versions: u64,
    pub avg_versions_per_intent: f64,
    pub max_versions: u32,
    pub min_versions: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::*;
    use tempfile::TempDir;
    
    async fn create_test_versioning() -> (VersioningEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();
        config.versioning.max_versions = 5;
        
        let store = Arc::new(IntentStore::new(&config).await.unwrap());
        let versioning = VersioningEngine::new(store, &config).await.unwrap();
        
        (versioning, temp_dir)
    }
    
    #[tokio::test]
    async fn test_version_creation() {
        let (versioning, _temp_dir) = create_test_versioning().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::Medium)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Create first version
        let v1 = versioning.create_version(&intent_id, &intent).await.unwrap();
        assert_eq!(v1.version, 1);
        assert_eq!(v1.parent_version, None);
        
        // Create second version with changes
        let mut intent_v2 = intent.clone();
        intent_v2.priority = Priority::High;
        let v2 = versioning.create_version(&intent_id, &intent_v2).await.unwrap();
        assert_eq!(v2.version, 2);
        assert_eq!(v2.parent_version, Some(1));
        
        // List versions
        let versions = versioning.list_versions(&intent_id).await.unwrap();
        assert_eq!(versions.len(), 2);
    }
    
    #[tokio::test]
    async fn test_rollback() {
        let (versioning, _temp_dir) = create_test_versioning().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::Medium)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Create versions
        versioning.create_version(&intent_id, &intent).await.unwrap();
        
        let mut intent_v2 = intent.clone();
        intent_v2.priority = Priority::High;
        versioning.create_version(&intent_id, &intent_v2).await.unwrap();
        
        // Rollback to version 1
        let rollback_version = versioning.rollback_to_version(&intent_id, 1).await.unwrap();
        assert_eq!(rollback_version.version, 3); // New version created
        assert!(rollback_version.change_description.is_some());
    }
    
    #[tokio::test]
    async fn test_cleanup() {
        let (versioning, _temp_dir) = create_test_versioning().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::Medium)
            .build()
            .unwrap();
        
        let intent_id = intent.id.clone();
        
        // Create more versions than the max limit (5)
        for i in 0..10 {
            let mut version_intent = intent.clone();
            version_intent.priority = if i % 2 == 0 { Priority::High } else { Priority::Low };
            versioning.create_version(&intent_id, &version_intent).await.unwrap();
        }
        
        // Should have 10 versions
        let versions_before = versioning.list_versions(&intent_id).await.unwrap();
        assert_eq!(versions_before.len(), 10);
        
        // Run cleanup
        versioning.cleanup_old_versions().await.unwrap();
        
        // Should now have max 5 versions
        let versions_after = versioning.list_versions(&intent_id).await.unwrap();
        assert_eq!(versions_after.len(), 5);
    }
    
    #[tokio::test]
    async fn test_diff_calculation() {
        let (versioning, _temp_dir) = create_test_versioning().await;
        
        let intent1 = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::Medium)
            .build()
            .unwrap();
        
        let mut intent2 = intent1.clone();
        intent2.priority = Priority::High;
        intent2.objectives.push(crate::intent::Objective::MaximizeThroughput {
            target_ops_per_sec: 1000.0,
        });
        
        let diff = versioning.calculate_diff(&intent1, &intent2, 1, 2).unwrap();
        
        assert_eq!(diff.from_version, 1);
        assert_eq!(diff.to_version, 2);
        assert!(!diff.changes.is_empty());
        
        // Should have priority change and objective addition
        assert!(diff.changes.iter().any(|c| matches!(c, VersionChange::PriorityChanged { .. })));
        assert!(diff.changes.iter().any(|c| matches!(c, VersionChange::ObjectiveAdded { .. })));
    }
}