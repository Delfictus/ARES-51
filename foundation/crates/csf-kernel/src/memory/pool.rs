//! Memory pool implementation for predictable allocation

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// A lock-free memory pool for fixed-size allocations
#[derive(Default)]
pub struct MemoryPool {
    /// Size of each block
    block_size: usize,

    /// Total number of blocks
    num_blocks: usize,

    /// Head of the free list
    free_list: AtomicPtr<FreeBlock>,

    /// Memory buffer
    buffer: Vec<u8>,

    /// Statistics
    allocated_count: AtomicUsize,
}

#[repr(C)]
struct FreeBlock {
    next: *mut FreeBlock,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        // Ensure alignment
        let block_size = block_size.max(std::mem::size_of::<FreeBlock>());
        let block_size = block_size.next_power_of_two();

        // Allocate buffer
        let total_size = block_size * num_blocks;
        let buffer = vec![0; total_size];

        // Initialize free list
        let mut pool = Self {
            block_size,
            num_blocks,
            free_list: AtomicPtr::new(ptr::null_mut()),
            buffer,
            allocated_count: AtomicUsize::new(0),
        };

        pool.init_free_list();
        pool
    }

    fn init_free_list(&mut self) {
        let base_ptr = self.buffer.as_mut_ptr();

        // Build linked list of free blocks
        let mut prev_block: *mut FreeBlock = ptr::null_mut();

        for i in (0..self.num_blocks).rev() {
            let block_ptr = unsafe { base_ptr.add(i * self.block_size) as *mut FreeBlock };

            unsafe {
                (*block_ptr).next = prev_block;
            }

            prev_block = block_ptr;
        }

        self.free_list.store(prev_block, Ordering::Release);
    }

    /// Allocate a block from the pool
    pub fn allocate(&self) -> Option<*mut u8> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                return None; // Pool exhausted
            }

            let next = unsafe { (*head).next };

            // Try to update the free list
            match self.free_list.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocated_count.fetch_add(1, Ordering::Relaxed);
                    return Some(head as *mut u8);
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Deallocate a block back to the pool
    pub fn deallocate(&self, ptr: *mut u8) -> bool {
        // Verify the pointer is from this pool
        let base = self.buffer.as_ptr() as usize;
        let ptr_addr = ptr as usize;
        let offset = ptr_addr.wrapping_sub(base);

        if offset >= self.buffer.len() || !offset.is_multiple_of(self.block_size) {
            return false; // Not from this pool
        }

        let block = ptr as *mut FreeBlock;

        loop {
            let head = self.free_list.load(Ordering::Acquire);

            unsafe {
                (*block).next = head;
            }

            // Try to update the free list
            match self.free_list.compare_exchange_weak(
                head,
                block,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocated_count.fetch_sub(1, Ordering::Relaxed);
                    return true;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Get the number of allocated blocks
    pub fn allocated_count(&self) -> usize {
        self.allocated_count.load(Ordering::Relaxed)
    }

    /// Get the number of free blocks
    pub fn free_count(&self) -> usize {
        self.num_blocks - self.allocated_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(64, 10);

        // Allocate all blocks
        let mut blocks = Vec::new();
        for _ in 0..10 {
            let block = pool
                .allocate()
                .expect("Should allocate from pool with available blocks");
            blocks.push(block);
        }

        // Pool should be exhausted
        assert!(pool.allocate().is_none());
        assert_eq!(pool.allocated_count(), 10);

        // Deallocate one block
        let block = blocks
            .pop()
            .expect("blocks vector should not be empty after 10 allocations");
        pool.deallocate(block);
        assert_eq!(pool.allocated_count(), 9);

        // Should be able to allocate again
        assert!(pool.allocate().is_some());
        assert_eq!(pool.allocated_count(), 10);
    }
}