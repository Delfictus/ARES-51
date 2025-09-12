//! NUMA-aware memory pool with CPU affinity
//! 
//! Features:
//! - NUMA node detection and allocation
//! - CPU pinning for cache locality
//! - Huge pages support
//! - Memory prefetching

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, AtomicBool, Ordering};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::alloc::{alloc, dealloc, Layout};
use std::thread;
use std::fs;
use parking_lot::{RwLock, Mutex};
use anyhow::{Result, Context};

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub id: usize,
    pub cpus: Vec<usize>,
    pub memory_kb: usize,
    pub distance_map: HashMap<usize, u32>,
}

impl NumaNode {
    fn detect_nodes() -> Result<Vec<NumaNode>> {
        let mut nodes = Vec::new();
        
        // Read NUMA topology from sysfs
        let node_path = "/sys/devices/system/node/";
        
        if let Ok(entries) = fs::read_dir(node_path) {
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                
                if name_str.starts_with("node") {
                    if let Some(id_str) = name_str.strip_prefix("node") {
                        if let Ok(id) = id_str.parse::<usize>() {
                            // Read CPUs for this node
                            let cpu_list_path = format!("{}/node{}/cpulist", node_path, id);
                            let cpu_list = fs::read_to_string(&cpu_list_path)
                                .unwrap_or_else(|_| String::from("0"));
                            
                            let cpus = Self::parse_cpu_list(&cpu_list);
                            
                            // Read memory info
                            let meminfo_path = format!("{}/node{}/meminfo", node_path, id);
                            let memory_kb = Self::parse_memory_info(&meminfo_path)
                                .unwrap_or(0);
                            
                            // Read distance map
                            let distance_path = format!("{}/node{}/distance", node_path, id);
                            let distance_map = Self::parse_distance_map(&distance_path, id)
                                .unwrap_or_default();
                            
                            nodes.push(NumaNode {
                                id,
                                cpus,
                                memory_kb,
                                distance_map,
                            });
                        }
                    }
                }
            }
        }
        
        // Fallback to single node if NUMA not available
        if nodes.is_empty() {
            let num_cpus = num_cpus::get();
            nodes.push(NumaNode {
                id: 0,
                cpus: (0..num_cpus).collect(),
                memory_kb: Self::get_total_memory_kb(),
                distance_map: HashMap::from([(0, 10)]),
            });
        }
        
        Ok(nodes)
    }
    
    fn parse_cpu_list(cpu_list: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        
        for part in cpu_list.trim().split(',') {
            if let Some((start, end)) = part.split_once('-') {
                if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    for cpu in s..=e {
                        cpus.push(cpu);
                    }
                }
            } else if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        
        cpus
    }
    
    fn parse_memory_info(path: &str) -> Result<usize> {
        let content = fs::read_to_string(path)?;
        
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].parse().ok().context("Failed to parse memory");
                }
            }
        }
        
        Ok(0)
    }
    
    fn parse_distance_map(path: &str, node_id: usize) -> Result<HashMap<usize, u32>> {
        let content = fs::read_to_string(path)?;
        let mut map = HashMap::new();
        
        for (i, distance_str) in content.split_whitespace().enumerate() {
            if let Ok(distance) = distance_str.parse::<u32>() {
                map.insert(i, distance);
            }
        }
        
        if map.is_empty() {
            map.insert(node_id, 10); // Local distance
        }
        
        Ok(map)
    }
    
    fn get_total_memory_kb() -> usize {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse().unwrap_or(0);
                    }
                }
            }
        }
        
        8_000_000 // Default 8GB
    }
}

/// Memory segment allocated from specific NUMA node
struct NumaSegment {
    ptr: NonNull<u8>,
    layout: Layout,
    node_id: usize,
    huge_page: bool,
}

unsafe impl Send for NumaSegment {}
unsafe impl Sync for NumaSegment {}

impl Drop for NumaSegment {
    fn drop(&mut self) {
        unsafe {
            if self.huge_page {
                // Unmap huge pages
                #[cfg(target_os = "linux")]
                {
                    libc::munmap(
                        self.ptr.as_ptr() as *mut libc::c_void,
                        self.layout.size(),
                    );
                }
            } else {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

/// NUMA-aware memory arena
pub struct NumaArena {
    node_id: usize,
    segments: RwLock<Vec<NumaSegment>>,
    free_lists: Vec<Mutex<Vec<NonNull<u8>>>>,
    allocated: AtomicUsize,
    freed: AtomicUsize,
    huge_pages_enabled: bool,
}

impl NumaArena {
    fn new(node_id: usize, huge_pages: bool) -> Self {
        // Create free lists for different size classes
        let mut free_lists = Vec::new();
        for _ in 0..16 {
            free_lists.push(Mutex::new(Vec::new()));
        }
        
        Self {
            node_id,
            segments: RwLock::new(Vec::new()),
            free_lists,
            allocated: AtomicUsize::new(0),
            freed: AtomicUsize::new(0),
            huge_pages_enabled: huge_pages,
        }
    }
    
    fn allocate(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        // Round up to size class
        let size_class = Self::size_to_class(size);
        let actual_size = Self::class_to_size(size_class);
        
        // Try to get from free list
        if let Some(ptr) = self.free_lists[size_class].lock().pop() {
            self.allocated.fetch_add(actual_size, Ordering::Relaxed);
            return Ok(ptr);
        }
        
        // Allocate new segment
        self.allocate_new_segment(actual_size, align)
    }
    
    fn allocate_new_segment(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(size, align)
            .context("Invalid layout")?;
        
        let ptr = if self.huge_pages_enabled && size >= 2 * 1024 * 1024 {
            // Try to allocate huge page
            self.allocate_huge_page(size)?
        } else {
            // Regular allocation with NUMA binding
            self.allocate_numa_bound(layout)?
        };
        
        let segment = NumaSegment {
            ptr,
            layout,
            node_id: self.node_id,
            huge_page: self.huge_pages_enabled && size >= 2 * 1024 * 1024,
        };
        
        self.segments.write().push(segment);
        self.allocated.fetch_add(size, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    #[cfg(target_os = "linux")]
    fn allocate_huge_page(&self, size: usize) -> Result<NonNull<u8>> {
        use libc::{mmap, MAP_ANONYMOUS, MAP_PRIVATE, MAP_HUGETLB, PROT_READ, PROT_WRITE};
        
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            )
        };
        
        if ptr == libc::MAP_FAILED {
            return Err(anyhow::anyhow!("Failed to allocate huge page"));
        }
        
        // Bind to NUMA node
        self.bind_memory_to_node(ptr as *mut u8, size)?;
        
        NonNull::new(ptr as *mut u8)
            .context("Null pointer from mmap")
    }
    
    #[cfg(not(target_os = "linux"))]
    fn allocate_huge_page(&self, size: usize) -> Result<NonNull<u8>> {
        // Fallback to regular allocation
        let layout = Layout::from_size_align(size, 4096)?;
        self.allocate_numa_bound(layout)
    }
    
    fn allocate_numa_bound(&self, layout: Layout) -> Result<NonNull<u8>> {
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(anyhow::anyhow!("Allocation failed"));
        }
        
        // Bind to NUMA node
        self.bind_memory_to_node(ptr, layout.size())?;
        
        NonNull::new(ptr)
            .context("Null pointer from alloc")
    }
    
    #[cfg(target_os = "linux")]
    fn bind_memory_to_node(&self, _ptr: *mut u8, _size: usize) -> Result<()> {
        // Note: mbind requires libc with NUMA support which may not be available
        // in all environments. For production use, consider using the numa crate
        // or ensuring libc is compiled with NUMA support.
        
        // Simplified implementation - just log that we would bind to NUMA
        if self.node_id > 0 {
            eprintln!("Info: Would bind memory to NUMA node {} (NUMA binding disabled)", self.node_id);
        }
        
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn bind_memory_to_node(&self, _ptr: *mut u8, _size: usize) -> Result<()> {
        // No-op on non-Linux systems
        Ok(())
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let size_class = Self::size_to_class(size);
        
        // Return to free list
        self.free_lists[size_class].lock().push(ptr);
        self.freed.fetch_add(size, Ordering::Relaxed);
    }
    
    fn size_to_class(size: usize) -> usize {
        // Size classes: 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 2M+
        match size {
            0..=64 => 0,
            65..=128 => 1,
            129..=256 => 2,
            257..=512 => 3,
            513..=1024 => 4,
            1025..=2048 => 5,
            2049..=4096 => 6,
            4097..=8192 => 7,
            8193..=16384 => 8,
            16385..=32768 => 9,
            32769..=65536 => 10,
            65537..=131072 => 11,
            131073..=262144 => 12,
            262145..=524288 => 13,
            524289..=1048576 => 14,
            _ => 15,
        }
    }
    
    fn class_to_size(class: usize) -> usize {
        match class {
            0 => 64,
            1 => 128,
            2 => 256,
            3 => 512,
            4 => 1024,
            5 => 2048,
            6 => 4096,
            7 => 8192,
            8 => 16384,
            9 => 32768,
            10 => 65536,
            11 => 131072,
            12 => 262144,
            13 => 524288,
            14 => 1048576,
            _ => 2097152,
        }
    }
}

/// NUMA-aware memory pool
pub struct NumaMemoryPool {
    nodes: Vec<NumaNode>,
    arenas: Vec<Arc<NumaArena>>,
    thread_node_map: Arc<RwLock<HashMap<thread::ThreadId, usize>>>,
    prefetch_enabled: AtomicBool,
    stats: Arc<NumaStats>,
}

/// NUMA statistics
pub struct NumaStats {
    pub local_allocations: AtomicU64,
    pub remote_allocations: AtomicU64,
    pub migrations: AtomicU64,
    pub huge_pages_used: AtomicU64,
}

impl Default for NumaStats {
    fn default() -> Self {
        Self {
            local_allocations: AtomicU64::new(0),
            remote_allocations: AtomicU64::new(0),
            migrations: AtomicU64::new(0),
            huge_pages_used: AtomicU64::new(0),
        }
    }
}

impl NumaMemoryPool {
    pub fn new(enable_huge_pages: bool) -> Result<Self> {
        let nodes = NumaNode::detect_nodes()?;
        
        let mut arenas = Vec::new();
        for node in &nodes {
            arenas.push(Arc::new(NumaArena::new(node.id, enable_huge_pages)));
        }
        
        Ok(Self {
            nodes,
            arenas,
            thread_node_map: Arc::new(RwLock::new(HashMap::new())),
            prefetch_enabled: AtomicBool::new(true),
            stats: Arc::new(NumaStats::default()),
        })
    }
    
    /// Pin current thread to specific NUMA node
    pub fn pin_thread_to_node(&self, node_id: usize) -> Result<()> {
        if node_id >= self.nodes.len() {
            return Err(anyhow::anyhow!("Invalid NUMA node ID"));
        }
        
        let node = &self.nodes[node_id];
        if node.cpus.is_empty() {
            return Err(anyhow::anyhow!("No CPUs available on node"));
        }
        
        // Pin to first CPU of the node
        self.set_cpu_affinity(node.cpus[0])?;
        
        // Record mapping
        self.thread_node_map.write()
            .insert(thread::current().id(), node_id);
        
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn set_cpu_affinity(&self, cpu: usize) -> Result<()> {
        use libc::{cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};
        
        unsafe {
            let mut cpuset: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut cpuset);
            CPU_SET(cpu, &mut cpuset);
            
            let result = sched_setaffinity(
                0,
                std::mem::size_of::<cpu_set_t>(),
                &cpuset,
            );
            
            if result != 0 {
                return Err(anyhow::anyhow!("Failed to set CPU affinity"));
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn set_cpu_affinity(&self, _cpu: usize) -> Result<()> {
        // No-op on non-Linux systems
        Ok(())
    }
    
    /// Allocate memory on local NUMA node
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        let node_id = self.get_current_node();
        let arena = &self.arenas[node_id];
        
        let ptr = arena.allocate(size, 64)?;
        
        // Prefetch if enabled
        if self.prefetch_enabled.load(Ordering::Relaxed) {
            self.prefetch_memory(ptr.as_ptr(), size);
        }
        
        // Update stats
        self.stats.local_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Allocate memory on specific NUMA node
    pub fn allocate_on_node(&self, size: usize, node_id: usize) -> Result<NonNull<u8>> {
        if node_id >= self.arenas.len() {
            return Err(anyhow::anyhow!("Invalid NUMA node ID"));
        }
        
        let arena = &self.arenas[node_id];
        let ptr = arena.allocate(size, 64)?;
        
        // Update stats
        let current_node = self.get_current_node();
        if current_node != node_id {
            self.stats.remote_allocations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.local_allocations.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(ptr)
    }
    
    /// Deallocate memory
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        // Find which arena owns this pointer
        let node_id = self.get_current_node();
        self.arenas[node_id].deallocate(ptr, size);
    }
    
    /// Migrate memory between NUMA nodes
    pub fn migrate(&self, ptr: NonNull<u8>, size: usize, to_node: usize) -> Result<NonNull<u8>> {
        if to_node >= self.arenas.len() {
            return Err(anyhow::anyhow!("Invalid NUMA node ID"));
        }
        
        // Allocate on target node
        let new_ptr = self.allocate_on_node(size, to_node)?;
        
        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_ptr.as_ptr(),
                size,
            );
        }
        
        // Free old allocation
        self.deallocate(ptr, size);
        
        // Update stats
        self.stats.migrations.fetch_add(1, Ordering::Relaxed);
        
        Ok(new_ptr)
    }
    
    fn get_current_node(&self) -> usize {
        let thread_id = thread::current().id();
        
        if let Some(&node_id) = self.thread_node_map.read().get(&thread_id) {
            return node_id;
        }
        
        // Default to node 0
        0
    }
    
    #[cfg(target_arch = "x86_64")]
    fn prefetch_memory(&self, ptr: *const u8, size: usize) {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        
        let cache_line_size = 64;
        let num_lines = (size + cache_line_size - 1) / cache_line_size;
        
        unsafe {
            for i in 0..num_lines {
                let addr = ptr.add(i * cache_line_size);
                _mm_prefetch(addr as *const i8, _MM_HINT_T0);
            }
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn prefetch_memory(&self, _ptr: *const u8, _size: usize) {
        // No-op on non-x86_64
    }
    
    /// Get NUMA topology information
    pub fn get_topology(&self) -> &[NumaNode] {
        &self.nodes
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> NumaPoolStats {
        let mut total_allocated = 0;
        let mut total_freed = 0;
        
        for arena in &self.arenas {
            total_allocated += arena.allocated.load(Ordering::Relaxed);
            total_freed += arena.freed.load(Ordering::Relaxed);
        }
        
        NumaPoolStats {
            nodes: self.nodes.len(),
            total_allocated,
            total_freed,
            local_allocations: self.stats.local_allocations.load(Ordering::Relaxed),
            remote_allocations: self.stats.remote_allocations.load(Ordering::Relaxed),
            migrations: self.stats.migrations.load(Ordering::Relaxed),
            huge_pages_used: self.stats.huge_pages_used.load(Ordering::Relaxed),
        }
    }
}

/// NUMA pool statistics
#[derive(Debug)]
pub struct NumaPoolStats {
    pub nodes: usize,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub local_allocations: u64,
    pub remote_allocations: u64,
    pub migrations: u64,
    pub huge_pages_used: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numa_detection() {
        let nodes = NumaNode::detect_nodes().unwrap();
        assert!(!nodes.is_empty());
        println!("Detected {} NUMA nodes", nodes.len());
        
        for node in &nodes {
            println!("Node {}: {} CPUs, {} KB memory",
                     node.id, node.cpus.len(), node.memory_kb);
        }
    }
    
    #[test]
    fn test_numa_allocation() {
        let pool = NumaMemoryPool::new(false).unwrap();
        
        // Allocate on local node
        let ptr = pool.allocate(1024).unwrap();
        assert!(!ptr.as_ptr().is_null());
        
        // Deallocate
        pool.deallocate(ptr, 1024);
        
        // Check stats
        let stats = pool.get_stats();
        assert_eq!(stats.local_allocations, 1);
    }
    
    #[test]
    fn test_size_classes() {
        assert_eq!(NumaArena::size_to_class(32), 0);
        assert_eq!(NumaArena::size_to_class(100), 1);
        assert_eq!(NumaArena::size_to_class(200), 2);
        assert_eq!(NumaArena::size_to_class(1000), 4);
        assert_eq!(NumaArena::size_to_class(1_000_000), 14);
        assert_eq!(NumaArena::size_to_class(10_000_000), 15);
    }
    
    #[test]
    fn test_thread_pinning() {
        let pool = NumaMemoryPool::new(false).unwrap();
        
        // Try to pin to node 0
        let result = pool.pin_thread_to_node(0);
        
        // Should succeed if we have at least one node
        if pool.nodes.len() > 0 {
            assert!(result.is_ok());
        }
    }
}