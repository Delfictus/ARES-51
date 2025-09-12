//! Memory management for real-time operation

use parking_lot::Mutex;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod pool;

/// Real-time memory allocator
#[derive(Default)]
pub struct RealTimeAllocator {
    /// Memory pools for different size classes
    pools: [Mutex<pool::MemoryPool>; 8],

    /// Statistics
    stats: AllocatorStats,

    /// Fallback to system allocator
    fallback: System,
}

#[derive(Default)]
struct AllocatorStats {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    bytes_allocated: AtomicUsize,
    pool_hits: AtomicUsize,
    fallback_allocations: AtomicUsize,
}

/// Size classes for memory pools (bytes)
const SIZE_CLASSES: [usize; 8] = [
    64,      // Small objects
    256,     // Medium objects
    1024,    // 1KB
    4096,    // 4KB (page size)
    16384,   // 16KB
    65536,   // 64KB
    262144,  // 256KB
    1048576, // 1MB
];

impl RealTimeAllocator {
    /// Create a new real-time allocator
    pub fn new() -> Self {
        let pools = [
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[0], 1000)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[1], 500)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[2], 200)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[3], 100)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[4], 50)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[5], 20)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[6], 10)),
            Mutex::new(pool::MemoryPool::new(SIZE_CLASSES[7], 5)),
        ];

        Self {
            pools,
            stats: AllocatorStats {
                allocations: AtomicUsize::new(0),
                deallocations: AtomicUsize::new(0),
                bytes_allocated: AtomicUsize::new(0),
                pool_hits: AtomicUsize::new(0),
                fallback_allocations: AtomicUsize::new(0),
            },
            fallback: System,
        }
    }

    fn find_pool(&self, size: usize) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&s| s >= size)
    }
}

unsafe impl GlobalAlloc for RealTimeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();

        // Try to allocate from a pool
        if let Some(pool_idx) = self.find_pool(size) {
            if let Some(ptr) = self.pools[pool_idx].lock().allocate() {
                self.stats.allocations.fetch_add(1, Ordering::Relaxed);
                self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_allocated
                    .fetch_add(SIZE_CLASSES[pool_idx], Ordering::Relaxed);
                return ptr;
            }
        }

        // Fall back to system allocator
        self.stats
            .fallback_allocations
            .fetch_add(1, Ordering::Relaxed);
        self.fallback.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();

        // Try to return to pool
        if let Some(pool_idx) = self.find_pool(size) {
            if self.pools[pool_idx].lock().deallocate(ptr) {
                self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_allocated
                    .fetch_sub(SIZE_CLASSES[pool_idx], Ordering::Relaxed);
                return;
            }
        }

        // Fall back to system allocator
        self.fallback.dealloc(ptr, layout)
    }
}

/// Initialize the global allocator
pub fn init_allocator() {
    // In a real implementation, we would set this as the global allocator
    // #[global_allocator]
    // static ALLOCATOR: RealTimeAllocator = RealTimeAllocator::new();
}
