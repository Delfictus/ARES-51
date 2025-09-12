//! Lock-free memory pool for zero-allocation trading
//! 
//! Pre-allocates memory pools for common sizes with thread-local caching

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use anyhow::Result;

/// Pool configuration
pub struct PoolConfig {
    pub size_classes: Vec<usize>,
    pub blocks_per_size: usize,
    pub enable_numa: bool,
    pub warm_pages: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            size_classes: vec![64, 256, 1024, 4096, 16384],
            blocks_per_size: 1024,
            enable_numa: false,
            warm_pages: true,
        }
    }
}

/// Size-specific pool
struct SizePool {
    size: usize,
    free_list: SegQueue<NonNull<u8>>,
    allocated_count: AtomicUsize,
    total_blocks: usize,
    memory: Vec<u8>,
}

impl SizePool {
    fn new(size: usize, blocks: usize, warm: bool) -> Result<Self> {
        let total_size = size * blocks;
        let layout = Layout::from_size_align(total_size, 64)?; // Cache-line aligned
        
        let memory = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(anyhow::anyhow!("Failed to allocate memory"));
            }
            
            // Warm pages if requested
            if warm {
                for i in (0..total_size).step_by(4096) {
                    ptr.add(i).write(0);
                }
            }
            
            Vec::from_raw_parts(ptr, total_size, total_size)
        };
        
        let free_list = SegQueue::new();
        
        // Add all blocks to free list
        for i in 0..blocks {
            let offset = i * size;
            let ptr = unsafe { NonNull::new_unchecked(memory.as_ptr().add(offset) as *mut u8) };
            free_list.push(ptr);
        }
        
        Ok(Self {
            size,
            free_list,
            allocated_count: AtomicUsize::new(0),
            total_blocks: blocks,
            memory,
        })
    }
    
    fn allocate(&self) -> Option<NonNull<u8>> {
        self.free_list.pop().map(|ptr| {
            self.allocated_count.fetch_add(1, Ordering::Relaxed);
            ptr
        })
    }
    
    fn deallocate(&self, ptr: NonNull<u8>) {
        self.free_list.push(ptr);
        self.allocated_count.fetch_sub(1, Ordering::Relaxed);
    }
    
    fn contains(&self, ptr: NonNull<u8>) -> bool {
        let addr = ptr.as_ptr() as usize;
        let start = self.memory.as_ptr() as usize;
        let end = start + self.memory.len();
        addr >= start && addr < end
    }
}

/// Thread-local cache
thread_local! {
    static LOCAL_CACHE: std::cell::RefCell<LocalCache> = std::cell::RefCell::new(LocalCache::new());
}

struct LocalCache {
    caches: Vec<Vec<NonNull<u8>>>,
    cache_size: usize,
    miss_count: usize,
}

impl LocalCache {
    fn new() -> Self {
        Self {
            caches: vec![Vec::new(); 5],
            cache_size: 16,
            miss_count: 0,
        }
    }
    
    fn get(&mut self, size_class: usize) -> Option<NonNull<u8>> {
        if size_class < self.caches.len() {
            self.caches[size_class].pop()
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    fn put(&mut self, size_class: usize, ptr: NonNull<u8>) -> bool {
        if size_class < self.caches.len() && self.caches[size_class].len() < self.cache_size {
            self.caches[size_class].push(ptr);
            true
        } else {
            false
        }
    }
}

/// Pool statistics
#[derive(Debug)]
pub struct PoolStatistics {
    pub allocations: usize,
    pub deallocations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub bytes_allocated: usize,
}

/// Main memory pool
pub struct MemoryPool {
    pools: Vec<Arc<SizePool>>,
    size_to_pool: Vec<usize>,
    stats: Arc<Statistics>,
}

struct Statistics {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(config: PoolConfig) -> Result<Self> {
        let mut pools = Vec::new();
        let mut size_to_pool = vec![0; 65536];
        
        // Create pools for each size class
        for (i, &size) in config.size_classes.iter().enumerate() {
            let pool = Arc::new(SizePool::new(size, config.blocks_per_size, config.warm_pages)?);
            pools.push(pool);
            
            // Map sizes to pools
            let start = if i == 0 { 0 } else { config.size_classes[i - 1] + 1 };
            let end = size;
            for s in start..=end {
                if s < size_to_pool.len() {
                    size_to_pool[s] = i;
                }
            }
        }
        
        Ok(Self {
            pools,
            size_to_pool,
            stats: Arc::new(Statistics {
                allocations: AtomicUsize::new(0),
                deallocations: AtomicUsize::new(0),
                cache_hits: AtomicUsize::new(0),
                cache_misses: AtomicUsize::new(0),
            }),
        })
    }
    
    /// Allocate memory
    #[inline(always)]
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        // Try thread-local cache first
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            
            let pool_idx = if size < self.size_to_pool.len() {
                self.size_to_pool[size]
            } else {
                self.pools.len() - 1
            };
            
            // Check cache
            if let Some(ptr) = cache.get(pool_idx) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(ptr);
            }
            
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            
            // Get from global pool
            if pool_idx < self.pools.len() {
                self.pools[pool_idx].allocate()
            } else {
                None
            }
        })
    }
    
    /// Deallocate memory
    #[inline(always)]
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let pool_idx = if size < self.size_to_pool.len() {
            self.size_to_pool[size]
        } else {
            self.pools.len() - 1
        };
        
        // Try to return to thread-local cache
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            
            if !cache.put(pool_idx, ptr) {
                // Cache full, return to global pool
                if pool_idx < self.pools.len() {
                    self.pools[pool_idx].deallocate(ptr);
                }
            }
        });
        
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStatistics {
        let mut total_allocated = 0;
        for pool in &self.pools {
            total_allocated += pool.allocated_count.load(Ordering::Relaxed) * pool.size;
        }
        
        PoolStatistics {
            allocations: self.stats.allocations.load(Ordering::Relaxed),
            deallocations: self.stats.deallocations.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            bytes_allocated: total_allocated,
        }
    }
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(PoolConfig::default()).unwrap();
        
        // Allocate
        let ptr1 = pool.allocate(64).unwrap();
        let ptr2 = pool.allocate(256).unwrap();
        
        // Different addresses
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
        
        // Deallocate
        pool.deallocate(ptr1, 64);
        pool.deallocate(ptr2, 256);
        
        // Check stats
        let stats = pool.get_stats();
        assert_eq!(stats.deallocations, 2);
    }
    
    #[test]
    fn test_thread_local_cache() {
        let pool = MemoryPool::new(PoolConfig::default()).unwrap();
        
        // Allocate and deallocate same size
        let ptr = pool.allocate(64).unwrap();
        pool.deallocate(ptr, 64);
        
        // Should hit cache on next allocation
        let ptr2 = pool.allocate(64).unwrap();
        
        let stats = pool.get_stats();
        assert!(stats.cache_hits > 0);
    }
}