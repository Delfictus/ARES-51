//! Memory management with huge pages and NUMA awareness

use anyhow::Result;
use libc::{
    c_void, madvise, mlock, mmap, munlock, munmap, MADV_DONTFORK, MADV_HUGEPAGE, MAP_ANONYMOUS,
    MAP_HUGETLB, MAP_POPULATE, MAP_PRIVATE, PROT_READ, PROT_WRITE,
};
use std::alloc::{GlobalAlloc, Layout};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Memory region with huge page support
pub struct MemoryRegion {
    ptr: NonNull<u8>,
    size: usize,
    #[allow(dead_code)]
    huge_pages: bool,
}

impl MemoryRegion {
    /// Allocate memory region with optional huge pages
    pub fn new(size: usize, huge_pages: bool) -> Result<Self> {
        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS;

        if huge_pages {
            flags |= MAP_HUGETLB;
        }

        // Pre-fault pages
        flags |= MAP_POPULATE;

        let ptr = unsafe { mmap(ptr::null_mut(), size, PROT_READ | PROT_WRITE, flags, -1, 0) };

        if ptr == libc::MAP_FAILED {
            return Err(anyhow::anyhow!(
                "mmap failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let ptr =
            NonNull::new(ptr as *mut u8).ok_or_else(|| anyhow::anyhow!("mmap returned null"))?;

        // Advise kernel about usage
        unsafe {
            if huge_pages {
                madvise(ptr.as_ptr() as *mut c_void, size, MADV_HUGEPAGE);
            }
            madvise(ptr.as_ptr() as *mut c_void, size, MADV_DONTFORK);
        }

        Ok(Self {
            ptr,
            size,
            huge_pages,
        })
    }

    /// Lock memory to prevent swapping
    pub fn lock(&self) -> Result<()> {
        let ret = unsafe { mlock(self.ptr.as_ptr() as *const c_void, self.size) };

        if ret != 0 {
            return Err(anyhow::anyhow!(
                "mlock failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        Ok(())
    }

    /// Unlock memory
    pub fn unlock(&self) -> Result<()> {
        let ret = unsafe { munlock(self.ptr.as_ptr() as *const c_void, self.size) };

        if ret != 0 {
            return Err(anyhow::anyhow!(
                "munlock failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        Ok(())
    }

    /// Get pointer to memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr() as *mut c_void, self.size);
        }
    }
}

/// Lock-free memory pool for real-time allocation
pub struct MemoryPool {
    /// Size class for this pool
    size_class: usize,
    /// Free list head
    free_list: AtomicPtr<FreeBlock>,
    /// Total allocated bytes
    allocated_bytes: AtomicUsize,
    /// Number of allocations
    allocation_count: AtomicUsize,
    /// Backing memory regions
    regions: parking_lot::RwLock<Vec<MemoryRegion>>,
}

#[repr(C)]
struct FreeBlock {
    next: *mut FreeBlock,
}

impl MemoryPool {
    /// Create new memory pool for given size class
    pub fn new(size_class: usize) -> Self {
        Self {
            size_class,
            free_list: AtomicPtr::new(ptr::null_mut()),
            allocated_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            regions: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Allocate from pool
    pub fn allocate(&self) -> *mut u8 {
        // Try to pop from free list (lock-free)
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                // Need to allocate new chunk
                return self.allocate_new_chunk();
            }

            let next = unsafe { (*head).next };

            // CAS to pop from free list
            match self.free_list.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocation_count.fetch_add(1, Ordering::Relaxed);
                    return head as *mut u8;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Deallocate to pool
    ///
    /// # Safety
    ///
    /// The `ptr` must have been allocated from this pool and must not be used after deallocation.
    pub unsafe fn deallocate(&self, ptr: *mut u8) {
        let block = ptr as *mut FreeBlock;

        // Push to free list (lock-free)
        loop {
            let old_head = self.free_list.load(Ordering::Acquire);
            (*block).next = old_head;

            match self.free_list.compare_exchange_weak(
                old_head,
                block,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocation_count.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    fn allocate_new_chunk(&self) -> *mut u8 {
        const CHUNK_SIZE: usize = 2 * 1024 * 1024; // 2MB huge page

        // Allocate new region
        let region = match MemoryRegion::new(CHUNK_SIZE, true) {
            Ok(r) => r,
            Err(_) => {
                // Fallback to regular pages
                match MemoryRegion::new(CHUNK_SIZE, false) {
                    Ok(r) => r,
                    Err(_) => return ptr::null_mut(),
                }
            }
        };

        // Lock memory to prevent swapping
        let _ = region.lock();

        let data_start = region.as_ptr();
        let num_blocks = CHUNK_SIZE / self.size_class;

        // Initialize free list for this chunk
        unsafe {
            for i in 0..num_blocks - 1 {
                let block = data_start.add(i * self.size_class) as *mut FreeBlock;
                let next_block = data_start.add((i + 1) * self.size_class) as *mut FreeBlock;
                (*block).next = next_block;
            }

            // Last block points to null
            let last_block = data_start.add((num_blocks - 1) * self.size_class) as *mut FreeBlock;
            (*last_block).next = ptr::null_mut();

            // Link all blocks except first to free list
            if num_blocks > 1 {
                let second_block = data_start.add(self.size_class) as *mut FreeBlock;

                loop {
                    let old_head = self.free_list.load(Ordering::Acquire);
                    (*last_block).next = old_head;

                    match self.free_list.compare_exchange_weak(
                        old_head,
                        second_block,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => break,
                        Err(_) => continue,
                    }
                }
            }
        }

        // Update statistics
        self.allocated_bytes
            .fetch_add(CHUNK_SIZE, Ordering::Relaxed);

        // Store region
        self.regions.write().push(region);

        // Return first block
        data_start
    }
}

/// Real-time allocator with multiple size classes
pub struct RealTimeAllocator {
    pools: [MemoryPool; NUM_SIZE_CLASSES],
    large_allocator: LargeAllocator,
}

const SIZE_CLASSES: [usize; 16] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
];

const NUM_SIZE_CLASSES: usize = SIZE_CLASSES.len();

impl Default for RealTimeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl RealTimeAllocator {
    /// Create new real-time allocator
    pub fn new() -> Self {
        let pools = [
            MemoryPool::new(SIZE_CLASSES[0]),
            MemoryPool::new(SIZE_CLASSES[1]),
            MemoryPool::new(SIZE_CLASSES[2]),
            MemoryPool::new(SIZE_CLASSES[3]),
            MemoryPool::new(SIZE_CLASSES[4]),
            MemoryPool::new(SIZE_CLASSES[5]),
            MemoryPool::new(SIZE_CLASSES[6]),
            MemoryPool::new(SIZE_CLASSES[7]),
            MemoryPool::new(SIZE_CLASSES[8]),
            MemoryPool::new(SIZE_CLASSES[9]),
            MemoryPool::new(SIZE_CLASSES[10]),
            MemoryPool::new(SIZE_CLASSES[11]),
            MemoryPool::new(SIZE_CLASSES[12]),
            MemoryPool::new(SIZE_CLASSES[13]),
            MemoryPool::new(SIZE_CLASSES[14]),
            MemoryPool::new(SIZE_CLASSES[15]),
        ];

        Self {
            pools,
            large_allocator: LargeAllocator::new(),
        }
    }

    fn find_size_class(size: usize) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&s| s >= size)
    }
}

unsafe impl GlobalAlloc for RealTimeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();

        // Find appropriate size class
        if let Some(pool_index) = Self::find_size_class(size) {
            self.pools[pool_index].allocate()
        } else {
            // Large allocation
            self.large_allocator.allocate(layout)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();

        if let Some(pool_index) = Self::find_size_class(size) {
            self.pools[pool_index].deallocate(ptr)
        } else {
            self.large_allocator.deallocate(ptr, layout)
        }
    }
}

/// Large allocator for sizes > 256KB
struct LargeAllocator {
    allocations: parking_lot::RwLock<std::collections::HashMap<*mut u8, (Layout, MemoryRegion)>>,
}

impl LargeAllocator {
    fn new() -> Self {
        Self {
            allocations: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        // Allocate with alignment padding
        let total_size = size + align;

        let region = match MemoryRegion::new(total_size, true) {
            Ok(r) => r,
            Err(_) => return ptr::null_mut(),
        };

        // Lock memory
        let _ = region.lock();

        // Align the pointer
        let ptr = region.as_ptr() as usize;
        let aligned_ptr = ((ptr + align - 1) & !(align - 1)) as *mut u8;

        // Record allocation
        self.allocations
            .write()
            .insert(aligned_ptr, (layout, region));

        aligned_ptr
    }

    unsafe fn deallocate(&self, ptr: *mut u8, _layout: Layout) {
        self.allocations.write().remove(&ptr);
    }
}

/// Allocate huge pages
pub fn allocate_huge_pages(size: usize) -> Result<MemoryRegion> {
    MemoryRegion::new(size, true)
}

/// Pin memory to prevent swapping
pub fn pin_memory(ptr: *mut u8, size: usize) -> Result<()> {
    let ret = unsafe { mlock(ptr as *const c_void, size) };

    if ret != 0 {
        return Err(anyhow::anyhow!(
            "mlock failed: {}",
            std::io::Error::last_os_error()
        ));
    }

    Ok(())
}

/// Get memory information
pub fn get_memory_info() -> Result<crate::MemoryInfo> {
    let mut total_bytes = 0;
    let mut available_bytes = 0;

    // Parse /proc/meminfo
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    total_bytes = kb * 1024;
                }
            }
        } else if line.starts_with("MemAvailable:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    available_bytes = kb * 1024;
                }
            }
        }
    }

    // Get huge page size
    let huge_page_size = get_huge_page_size()?;

    // Count NUMA nodes
    let numa_nodes = count_numa_nodes()?;

    Ok(crate::MemoryInfo {
        total_bytes,
        available_bytes,
        huge_page_size,
        numa_nodes,
    })
}

fn get_huge_page_size() -> Result<usize> {
    let path = "/proc/meminfo";
    let contents = std::fs::read_to_string(path)?;

    for line in contents.lines() {
        if line.starts_with("Hugepagesize:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    return Ok(kb * 1024);
                }
            }
        }
    }

    // Default to 2MB
    Ok(2 * 1024 * 1024)
}

fn count_numa_nodes() -> Result<u32> {
    let mut count = 0;
    let node_path = "/sys/devices/system/node";

    if let Ok(entries) = std::fs::read_dir(node_path) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("node") {
                count += 1;
            }
        }
    }

    Ok(count.max(1))
}

/// Initialize memory allocators
pub fn init_allocators() -> Result<()> {
    // Ensure huge pages are available
    enable_transparent_huge_pages()?;

    Ok(())
}

fn enable_transparent_huge_pages() -> Result<()> {
    let path = "/sys/kernel/mm/transparent_hugepage/enabled";
    if std::path::Path::new(path).exists() {
        std::fs::write(path, b"always").ok();
    }

    Ok(())
}
