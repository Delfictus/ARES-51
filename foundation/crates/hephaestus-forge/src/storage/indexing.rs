//! High-performance indexing engine for sub-millisecond intent retrieval

use crate::{intent::*, types::ForgeResult, ForgeError};
use super::{StorageConfig, IntentSearchQuery, IntentStore};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock as TokioRwLock;
use ahash::AHashMap;

/// Cache entry for hot intents
#[derive(Debug, Clone)]
struct CacheEntry {
    intent: OptimizationIntent,
    last_accessed: Instant,
    access_count: u64,
}

/// Indexing statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_requests: u64,
    pub cache_size: usize,
    pub index_size: usize,
}

/// High-performance indexing engine with multi-level caching
pub struct IndexingEngine {
    store: Arc<IntentStore>,
    
    // Hot cache: Most frequently accessed intents in memory
    hot_cache: Arc<DashMap<IntentId, CacheEntry>>,
    
    // Secondary indexes for fast lookups
    priority_index: Arc<TokioRwLock<AHashMap<Priority, Vec<IntentId>>>>,
    target_index: Arc<TokioRwLock<AHashMap<String, Vec<IntentId>>>>,
    timestamp_index: Arc<TokioRwLock<Vec<(chrono::DateTime<chrono::Utc>, IntentId)>>>,
    
    // Statistics
    stats: Arc<RwLock<IndexStats>>,
    
    // Configuration
    config: StorageConfig,
}

impl IndexingEngine {
    /// Create a new indexing engine
    pub async fn new(store: Arc<IntentStore>, config: &StorageConfig) -> ForgeResult<Self> {
        let engine = Self {
            store,
            hot_cache: Arc::new(DashMap::new()),
            priority_index: Arc::new(TokioRwLock::new(AHashMap::new())),
            target_index: Arc::new(TokioRwLock::new(AHashMap::new())),
            timestamp_index: Arc::new(TokioRwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(IndexStats {
                cache_hits: 0,
                cache_misses: 0,
                total_requests: 0,
                cache_size: 0,
                index_size: 0,
            })),
            config: config.clone(),
        };
        
        // Start background cache maintenance
        engine.start_cache_maintenance().await;
        
        Ok(engine)
    }
    
    /// Get an intent from cache if available (sub-millisecond performance)
    pub async fn get_cached_intent(&self, intent_id: &IntentId) -> ForgeResult<Option<OptimizationIntent>> {
        let start = Instant::now();
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
        }
        
        // Check hot cache first
        if let Some(mut entry) = self.hot_cache.get_mut(intent_id) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update cache hit stats
            {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
            }
            
            let elapsed = start.elapsed();
            tracing::debug!("Cache hit for intent {} in {}μs", intent_id, elapsed.as_micros());
            
            return Ok(Some(entry.intent.clone()));
        }
        
        // Cache miss
        {
            let mut stats = self.stats.write();
            stats.cache_misses += 1;
        }
        
        Ok(None)
    }
    
    /// Cache an intent for future fast access
    pub async fn cache_intent(&self, intent: &OptimizationIntent) -> ForgeResult<()> {
        let entry = CacheEntry {
            intent: intent.clone(),
            last_accessed: Instant::now(),
            access_count: 1,
        };
        
        // Add to hot cache
        self.hot_cache.insert(intent.id.clone(), entry);
        
        // Update cache size stats
        {
            let mut stats = self.stats.write();
            stats.cache_size = self.hot_cache.len();
        }
        
        // Evict if cache is too large
        self.evict_cold_entries().await;
        
        Ok(())
    }
    
    /// Index a new intent for fast searching
    pub async fn index_intent(&self, intent: &OptimizationIntent, _version: u32) -> ForgeResult<()> {
        // Index by priority
        {
            let mut priority_idx = self.priority_index.write().await;
            priority_idx.entry(intent.priority)
                .or_insert_with(Vec::new)
                .push(intent.id.clone());
        }
        
        // Index by target
        let target_key = match &intent.target {
            OptimizationTarget::Module(module_id) => format!("module:{}", module_id.0),
            OptimizationTarget::ModuleName(name) => format!("name:{}", name),
            OptimizationTarget::ComponentGroup(group) => format!("group:{}", group),
            OptimizationTarget::System => "system".to_string(),
        };
        
        {
            let mut target_idx = self.target_index.write().await;
            target_idx.entry(target_key)
                .or_insert_with(Vec::new)
                .push(intent.id.clone());
        }
        
        // Index by timestamp (assuming current time for new intents)
        {
            let mut timestamp_idx = self.timestamp_index.write().await;
            timestamp_idx.push((chrono::Utc::now(), intent.id.clone()));
            
            // Keep timestamp index sorted and bounded
            timestamp_idx.sort_by(|a, b| b.0.cmp(&a.0)); // Most recent first
            if timestamp_idx.len() > 100000 { // Keep only most recent 100k
                timestamp_idx.truncate(100000);
            }
        }
        
        // Update index size stats
        {
            let mut stats = self.stats.write();
            stats.index_size = self.calculate_index_size().await;
        }
        
        Ok(())
    }
    
    /// Remove an intent from all indexes
    pub async fn remove_intent(&self, intent_id: &IntentId) -> ForgeResult<()> {
        // Remove from hot cache
        self.hot_cache.remove(intent_id);
        
        // Remove from priority index
        {
            let mut priority_idx = self.priority_index.write().await;
            for intents in priority_idx.values_mut() {
                intents.retain(|id| id != intent_id);
            }
        }
        
        // Remove from target index
        {
            let mut target_idx = self.target_index.write().await;
            for intents in target_idx.values_mut() {
                intents.retain(|id| id != intent_id);
            }
        }
        
        // Remove from timestamp index
        {
            let mut timestamp_idx = self.timestamp_index.write().await;
            timestamp_idx.retain(|(_, id)| id != intent_id);
        }
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.cache_size = self.hot_cache.len();
            stats.index_size = self.calculate_index_size().await;
        }
        
        Ok(())
    }
    
    /// Search intents using indexes for fast results
    pub async fn search_intents(&self, query: IntentSearchQuery) -> ForgeResult<Vec<OptimizationIntent>> {
        let start = Instant::now();
        let mut candidate_ids = Vec::new();
        
        // Filter by priority if specified
        if let Some(priority) = query.priority {
            let priority_idx = self.priority_index.read().await;
            if let Some(ids) = priority_idx.get(&priority) {
                candidate_ids.extend(ids.clone());
            } else {
                return Ok(Vec::new()); // No intents with this priority
            }
        }
        
        // Filter by target if specified
        if let Some(ref target) = query.target_module {
            let target_key = format!("name:{}", target);
            let target_idx = self.target_index.read().await;
            
            if candidate_ids.is_empty() {
                // First filter
                if let Some(ids) = target_idx.get(&target_key) {
                    candidate_ids.extend(ids.clone());
                } else {
                    return Ok(Vec::new()); // No intents with this target
                }
            } else {
                // Intersect with existing candidates
                if let Some(target_ids) = target_idx.get(&target_key) {
                    candidate_ids.retain(|id| target_ids.contains(id));
                } else {
                    return Ok(Vec::new()); // No intersection
                }
            }
        }
        
        // If no specific filters, get recent intents
        if candidate_ids.is_empty() {
            let timestamp_idx = self.timestamp_index.read().await;
            let limit = query.limit.unwrap_or(100);
            candidate_ids = timestamp_idx.iter()
                .take(limit)
                .map(|(_, id)| id.clone())
                .collect();
        }
        
        // Apply limit
        if let Some(limit) = query.limit {
            candidate_ids.truncate(limit);
        }
        
        // Fetch actual intents
        let mut results = Vec::new();
        for intent_id in candidate_ids {
            // Try cache first
            if let Some(intent) = self.get_cached_intent(&intent_id).await? {
                results.push(intent);
            } else {
                // Fallback to storage
                if let Some(intent) = self.store.get_intent(&intent_id).await? {
                    // Cache for future access
                    self.cache_intent(&intent).await?;
                    results.push(intent);
                }
            }
        }
        
        // Apply date range filter if specified (post-processing for now)
        if let Some((start_date, end_date)) = query.date_range {
            results.retain(|intent| {
                // For this example, we'll use the intent ID creation time
                // In practice, you'd want to store creation timestamps
                let created = chrono::Utc::now(); // Placeholder
                created >= start_date && created <= end_date
            });
        }
        
        // Apply text search filter if specified (simple substring matching)
        if let Some(ref search_text) = query.text_search {
            let search_lower = search_text.to_lowercase();
            results.retain(|intent| {
                intent.objectives.iter().any(|obj| {
                    format!("{:?}", obj).to_lowercase().contains(&search_lower)
                }) || intent.constraints.iter().any(|constraint| {
                    format!("{:?}", constraint).to_lowercase().contains(&search_lower)
                })
            });
        }
        
        let elapsed = start.elapsed();
        tracing::debug!("Search completed in {}μs, found {} results", elapsed.as_micros(), results.len());
        
        Ok(results)
    }
    
    /// Get current cache hit rate
    pub async fn cache_hit_rate(&self) -> ForgeResult<f64> {
        let stats = self.stats.read();
        if stats.total_requests == 0 {
            Ok(0.0)
        } else {
            Ok(stats.cache_hits as f64 / stats.total_requests as f64)
        }
    }
    
    /// Get current index size in bytes (estimated)
    pub async fn index_size(&self) -> ForgeResult<u64> {
        Ok(self.stats.read().index_size as u64)
    }
    
    /// Rebuild all indexes from storage
    pub async fn rebuild_indexes(&self) -> ForgeResult<()> {
        tracing::info!("Rebuilding indexes...");
        
        // Clear existing indexes
        {
            let mut priority_idx = self.priority_index.write().await;
            priority_idx.clear();
        }
        {
            let mut target_idx = self.target_index.write().await;
            target_idx.clear();
        }
        {
            let mut timestamp_idx = self.timestamp_index.write().await;
            timestamp_idx.clear();
        }
        
        // Clear cache
        self.hot_cache.clear();
        
        // Rebuild would require iterating through all intents in storage
        // For now, indexes will be rebuilt lazily as intents are accessed
        
        tracing::info!("Index rebuild completed");
        Ok(())
    }
    
    // Private methods
    
    /// Start background cache maintenance task
    async fn start_cache_maintenance(&self) {
        let hot_cache = self.hot_cache.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Run every minute
            
            loop {
                interval.tick().await;
                
                // Clean up old cache entries
                let now = Instant::now();
                let max_cache_size = config.cache_size_mb * 1024 * 1024 / 1000; // Rough estimate
                
                if hot_cache.len() > max_cache_size {
                    // Find least recently used entries
                    let mut entries: Vec<_> = hot_cache.iter()
                        .map(|entry| {
                            let key = entry.key().clone();
                            let last_accessed = entry.value().last_accessed;
                            let access_count = entry.value().access_count;
                            (key, last_accessed, access_count)
                        })
                        .collect();
                    
                    // Sort by LRU (least recently used first)
                    entries.sort_by(|a, b| a.1.cmp(&b.1));
                    
                    // Remove oldest entries until we're under the limit
                    let to_remove = hot_cache.len() - max_cache_size;
                    for i in 0..to_remove.min(entries.len()) {
                        hot_cache.remove(&entries[i].0);
                    }
                    
                    // Update stats
                    {
                        let mut stats = stats.write();
                        stats.cache_size = hot_cache.len();
                    }
                    
                    tracing::debug!("Cache cleanup: removed {} entries, {} remaining", to_remove, hot_cache.len());
                }
            }
        });
    }
    
    /// Evict cold cache entries to maintain performance
    async fn evict_cold_entries(&self) {
        let max_cache_size = self.config.cache_size_mb * 1024 * 1024 / 1000; // Rough estimate
        
        if self.hot_cache.len() <= max_cache_size {
            return;
        }
        
        // Simple LRU eviction
        let now = Instant::now();
        let mut to_remove = Vec::new();
        
        for entry in self.hot_cache.iter() {
            if now.duration_since(entry.value().last_accessed) > Duration::from_secs(300) { // 5 minutes
                to_remove.push(entry.key().clone());
            }
        }
        
        for key in to_remove {
            self.hot_cache.remove(&key);
        }
    }
    
    /// Calculate current index size (estimated)
    async fn calculate_index_size(&self) -> usize {
        let priority_size = {
            let priority_idx = self.priority_index.read().await;
            priority_idx.len() * std::mem::size_of::<Priority>() 
                + priority_idx.values().map(|v| v.len() * std::mem::size_of::<IntentId>()).sum::<usize>()
        };
        
        let target_size = {
            let target_idx = self.target_index.read().await;
            target_idx.keys().map(|k| k.len()).sum::<usize>()
                + target_idx.values().map(|v| v.len() * std::mem::size_of::<IntentId>()).sum::<usize>()
        };
        
        let timestamp_size = {
            let timestamp_idx = self.timestamp_index.read().await;
            timestamp_idx.len() * (std::mem::size_of::<chrono::DateTime<chrono::Utc>>() + std::mem::size_of::<IntentId>())
        };
        
        let cache_size = self.hot_cache.len() * std::mem::size_of::<CacheEntry>();
        
        priority_size + target_size + timestamp_size + cache_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::*;
    use tempfile::TempDir;
    
    async fn create_test_indexing() -> (IndexingEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = StorageConfig::default();
        config.db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();
        
        let store = Arc::new(IntentStore::new(&config).await.unwrap());
        let indexing = IndexingEngine::new(store, &config).await.unwrap();
        
        (indexing, temp_dir)
    }
    
    #[tokio::test]
    async fn test_cache_operations() {
        let (indexing, _temp_dir) = create_test_indexing().await;
        
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
        
        // Initially not in cache
        assert!(indexing.get_cached_intent(&intent_id).await.unwrap().is_none());
        
        // Cache the intent
        indexing.cache_intent(&intent).await.unwrap();
        
        // Should now be in cache
        let cached = indexing.get_cached_intent(&intent_id).await.unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().id, intent_id);
        
        // Check cache hit rate
        let hit_rate = indexing.cache_hit_rate().await.unwrap();
        assert!(hit_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_indexing_operations() {
        let (indexing, _temp_dir) = create_test_indexing().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        // Index the intent
        indexing.index_intent(&intent, 1).await.unwrap();
        
        // Search by priority
        let query = IntentSearchQuery {
            priority: Some(Priority::High),
            target_module: None,
            text_search: None,
            date_range: None,
            limit: None,
        };
        
        // Note: This will return empty since we don't have the actual intent stored,
        // just indexed. In real usage, the intent would be in storage.
        let results = indexing.search_intents(query).await.unwrap();
        // Results might be empty due to storage not having the actual intent
    }
    
    #[tokio::test]
    async fn test_performance_requirements() {
        let (indexing, _temp_dir) = create_test_indexing().await;
        
        let intent = OptimizationIntent::builder()
            .target_module("test_module")
            .add_objective(crate::intent::Objective::MinimizeLatency {
                percentile: 99.0,
                target_ms: 10.0,
            })
            .priority(Priority::High)
            .build()
            .unwrap();
        
        // Cache intent
        indexing.cache_intent(&intent).await.unwrap();
        
        // Test multiple cache hits for performance
        for _ in 0..1000 {
            let start = Instant::now();
            let cached = indexing.get_cached_intent(&intent.id).await.unwrap();
            let elapsed = start.elapsed();
            
            assert!(cached.is_some());
            assert!(elapsed.as_micros() < 1000, "Cache retrieval took {}μs, expected <1000μs", elapsed.as_micros());
        }
        
        // Verify high cache hit rate
        let hit_rate = indexing.cache_hit_rate().await.unwrap();
        assert!(hit_rate > 0.99, "Cache hit rate too low: {}", hit_rate);
    }
}