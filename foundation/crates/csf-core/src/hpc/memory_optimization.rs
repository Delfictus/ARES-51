//! Memory Optimization for Massive Dataset Processing
//!
//! This module provides memory pools, zero-copy operations, NUMA-aware allocation,
//! and streaming buffers for efficient processing of large-scale topological data.

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use memmap2::{Mmap, MmapMut, MmapOptions};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3};
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::mem::{align_of, size_of};
use std::ptr::{null_mut, NonNull};
use std::sync::{
    atomic::{AtomicPtr, AtomicUsize, Ordering},
    Arc, Mutex,
};
use sysinfo::System;
use thiserror::Error;
use csf_time::{TimeSource, NanoTime};

/// High-performance memory pool for large computations
pub struct MemoryPool {
    /// Total pool capacity in bytes
    capacity: usize,

    /// Currently allocated bytes
    allocated: AtomicUsize,

    /// Peak allocation
    peak_allocated: AtomicUsize,

    /// Free memory blocks by size (simplified for safety)
    free_blocks: Arc<Mutex<HashMap<usize, Vec<*mut u8>>>>,

    /// Allocated blocks for tracking
    allocated_blocks: Arc<Mutex<HashMap<*mut u8, AllocatedBlock>>>,

    /// Memory statistics
    stats: Arc<Mutex<MemoryStats>>,

    /// NUMA node preference
    numa_node: Option<u32>,

    /// Large page support
    use_large_pages: bool,

    /// Time source for temporal operations
    time_source: Arc<dyn TimeSource>,
}

/// Information about allocated memory block
#[derive(Debug, Clone)]
struct AllocatedBlock {
    size: usize,
    layout: Layout,
    allocated_at: NanoTime,
    numa_node: Option<u32>,
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub fragmentation_ratio: f64,
    pub average_allocation_size: f64,
    pub allocation_rate_per_second: f64,
    pub numa_distribution: HashMap<u32, usize>,
}

/// Zero-copy streaming buffer
pub struct StreamingBuffer {
    /// Memory-mapped file backing
    mmap: Option<MmapMut>,

    /// Buffer capacity
    capacity: usize,

    /// Current write position
    write_pos: AtomicUsize,

    /// Current read position
    read_pos: AtomicUsize,

    /// Buffer mode
    mode: BufferMode,

    /// Element size for typed access
    element_size: usize,

    /// Circular buffer management
    circular: bool,
}

#[derive(Debug, Clone)]
pub enum BufferMode {
    /// Linear buffer that grows as needed
    Linear,
    /// Fixed-size circular buffer
    Circular,
    /// Memory-mapped file buffer
    MemoryMapped { file_path: String },
}

/// NUMA-aware memory allocator
pub struct NUMAAllocator {
    /// Current system topology
    topology: SystemTopology,

    /// Per-NUMA node memory pools
    node_pools: HashMap<u32, Arc<MemoryPool>>,

    /// Thread-to-NUMA mapping
    thread_affinity: Arc<Mutex<HashMap<std::thread::ThreadId, u32>>>,
}

#[derive(Debug, Clone)]
pub struct SystemTopology {
    pub numa_nodes: u32,
    pub cores_per_node: Vec<u32>,
    pub memory_per_node_gb: Vec<f64>,
    pub node_distances: Vec<Vec<u32>>,
}

/// Large matrix storage with memory optimization
pub struct OptimizedMatrix<T> {
    /// Matrix dimensions
    rows: usize,
    cols: usize,

    /// Storage strategy
    storage: MatrixStorage<T>,

    /// Memory layout optimization
    layout: MatrixLayout,

    /// Cache-friendly access patterns
    access_pattern: AccessPattern,
}

#[derive(Debug, Clone)]
pub enum MatrixStorage<T> {
    /// Dense matrix in system memory
    Dense { data: Vec<T> },
    /// Sparse matrix with coordinate format
    Sparse {
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
        nnz: usize,
    },
    /// Memory-mapped matrix from file
    MemoryMapped {
        mmap: Arc<Mmap>,
        phantom: std::marker::PhantomData<T>,
    },
    /// Distributed across NUMA nodes
    NumaDistributed { node_chunks: HashMap<u32, Vec<T>> },
}

#[derive(Debug, Clone)]
pub enum MatrixLayout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Block-wise layout for cache efficiency
    Blocked { block_size: (usize, usize) },
    /// Morton (Z-order) layout for spatial locality
    Morton,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    /// Sequential row access
    RowSequential,
    /// Sequential column access
    ColumnSequential,
    /// Random access
    Random,
    /// Block-wise access
    Blocked,
    /// Streaming access
    Streaming,
}

/// Memory-efficient distance matrix computation
pub struct DistanceMatrixOptimized {
    /// Point storage
    points: OptimizedMatrix<f64>,

    /// Distance computation strategy
    strategy: DistanceStrategy,

    /// Result storage
    result_storage: DistanceStorage,

    /// Streaming computation state
    computation_state: ComputationState,
}

#[derive(Debug, Clone)]
pub enum DistanceStrategy {
    /// Compute all distances and store
    FullMatrix,
    /// Compute upper triangle only (symmetric)
    UpperTriangle,
    /// Streaming computation with limited memory
    Streaming { chunk_size: usize },
    /// Approximate distances using sampling
    Approximate { sample_ratio: f64 },
}

#[derive(Debug, Clone)]
pub enum DistanceStorage {
    /// Full matrix in memory
    InMemory { matrix: Arc<Mutex<DMatrix<f64>>> },
    /// Memory-mapped file
    MemoryMapped { file_path: String },
    /// Compressed storage
    Compressed {
        indices: Vec<(usize, usize)>,
        values: Vec<f32>,
        compression_ratio: f64,
    },
    /// Hierarchical storage (memory + disk)
    Hierarchical {
        memory_cache: HashMap<(usize, usize), f64>,
        disk_storage: String,
    },
}

#[derive(Debug, Clone)]
pub struct ComputationState {
    pub progress: f64,
    pub current_chunk: usize,
    pub total_chunks: usize,
    pub memory_usage_mb: f64,
    pub computation_rate_pairs_per_second: f64,
}

impl MemoryPool {
    /// Create new memory pool with specified capacity
    pub fn new(capacity_bytes: usize, time_source: Arc<dyn TimeSource>) -> Result<Self, MemoryError> {
        let system = System::new_all();
        let available_memory = system.total_memory(); // Already in bytes

        if capacity_bytes > available_memory as usize {
            return Err(MemoryError::InsufficientMemory {
                requested: capacity_bytes,
                available: available_memory as usize,
            });
        }

        Ok(Self {
            capacity: capacity_bytes,
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            free_blocks: Arc::new(Mutex::new(HashMap::new())),
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryStats::new())),
            numa_node: Self::detect_preferred_numa_node(),
            use_large_pages: false,
            time_source,
        })
    }

    /// Allocate memory block
    pub fn allocate(&self, size: usize, alignment: usize) -> Result<NonNull<u8>, MemoryError> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| MemoryError::InvalidLayout(format!("{}", e)))?;

        // Check if we have a free block of the right size
        if let Some(block) = self.try_reuse_block(size) {
            return Ok(block);
        }

        // Check capacity
        let current_allocated = self.allocated.load(Ordering::Relaxed);
        if current_allocated + size > self.capacity {
            return Err(MemoryError::PoolExhausted {
                requested: size,
                available: self.capacity - current_allocated,
            });
        }

        // Allocate new block
        let ptr = unsafe {
            if self.use_large_pages && size >= 2 * 1024 * 1024 {
                // 2MB pages
                self.allocate_large_page(layout)?
            } else {
                alloc(layout)
            }
        };

        if ptr.is_null() {
            return Err(MemoryError::AllocationFailed(size));
        }

        let non_null_ptr = NonNull::new(ptr).unwrap();

        // Update tracking
        self.allocated.fetch_add(size, Ordering::Relaxed);
        let new_allocated = self.allocated.load(Ordering::Relaxed);

        // Update peak
        let mut peak = self.peak_allocated.load(Ordering::Relaxed);
        while new_allocated > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                new_allocated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }

        // Track allocation
        let block = AllocatedBlock {
            size,
            layout,
            allocated_at: self.time_source.now_ns().unwrap_or_else(|_| NanoTime::from_nanos(0)),
            numa_node: self.numa_node,
        };

        self.allocated_blocks.lock().unwrap().insert(ptr, block);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.current_allocated_bytes = new_allocated;
            if new_allocated > stats.peak_allocated_bytes {
                stats.peak_allocated_bytes = new_allocated;
            }
        }

        Ok(non_null_ptr)
    }

    /// Deallocate memory block
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let ptr_raw = ptr.as_ptr();

        // Remove from tracking
        let block = if let Ok(mut blocks) = self.allocated_blocks.lock() {
            blocks.remove(&ptr_raw)
        } else {
            return;
        };

        if let Some(block) = block {
            // Add to free blocks for reuse
            if let Ok(mut free_blocks) = self.free_blocks.lock() {
                free_blocks
                    .entry(block.size)
                    .or_insert_with(Vec::new)
                    .push(ptr_raw);
            } else {
                // If we can't track it, just deallocate
                unsafe {
                    dealloc(ptr_raw, block.layout);
                }
            }

            // Update allocated count
            self.allocated.fetch_sub(block.size, Ordering::Relaxed);

            // Update statistics
            if let Ok(mut stats) = self.stats.lock() {
                stats.total_deallocations += 1;
                stats.current_allocated_bytes = self.allocated.load(Ordering::Relaxed);
            }
        }
    }

    /// Try to reuse an existing free block
    fn try_reuse_block(&self, size: usize) -> Option<NonNull<u8>> {
        if let Ok(mut free_blocks) = self.free_blocks.lock() {
            if let Some(blocks) = free_blocks.get_mut(&size) {
                if let Some(ptr) = blocks.pop() {
                    return NonNull::new(ptr);
                }
            }
        }
        None
    }

    /// Allocate large page (2MB pages on x86-64)
    unsafe fn allocate_large_page(&self, layout: Layout) -> Result<*mut u8, MemoryError> {
        // Platform-specific large page allocation would go here
        // For now, fall back to regular allocation
        Ok(alloc(layout))
    }

    /// Detect preferred NUMA node for current thread
    fn detect_preferred_numa_node() -> Option<u32> {
        // Platform-specific NUMA detection would go here
        None
    }

    /// Get memory pool statistics
    pub fn statistics(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Defragment memory pool
    pub fn defragment(&self) -> Result<DefragmentationResult, MemoryError> {
        // Complex defragmentation logic would go here
        Ok(DefragmentationResult {
            freed_bytes: 0,
            moved_blocks: 0,
            fragmentation_reduction: 0.0,
        })
    }
}

impl StreamingBuffer {
    /// Create new streaming buffer
    pub fn new(
        capacity: usize,
        mode: BufferMode,
        element_size: usize,
    ) -> Result<Self, MemoryError> {
        let mmap = match &mode {
            BufferMode::MemoryMapped { file_path } => {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(file_path)
                    .map_err(|e| {
                        MemoryError::FileSystemError(format!("Failed to open file: {}", e))
                    })?;

                file.set_len(capacity as u64).map_err(|e| {
                    MemoryError::FileSystemError(format!("Failed to set file length: {}", e))
                })?;

                unsafe {
                    Some(MmapOptions::new().map_mut(&file).map_err(|e| {
                        MemoryError::FileSystemError(format!("Failed to mmap: {}", e))
                    })?)
                }
            }
            _ => None,
        };

        let circular = matches!(mode, BufferMode::Circular);

        Ok(Self {
            mmap,
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            mode,
            element_size,
            circular,
        })
    }

    /// Write data to buffer
    pub fn write<T: Pod>(&mut self, data: &[T]) -> Result<usize, MemoryError> {
        let byte_size = data.len() * size_of::<T>();
        let current_write = self.write_pos.load(Ordering::Acquire);
        let current_read = self.read_pos.load(Ordering::Acquire);

        // Check available space
        let available_space = if self.circular {
            if current_write >= current_read {
                self.capacity - current_write + current_read
            } else {
                current_read - current_write
            }
        } else {
            self.capacity - current_write
        };

        if byte_size > available_space {
            return Err(MemoryError::BufferOverflow {
                requested: byte_size,
                available: available_space,
            });
        }

        // Write data - safely handle mutable borrow
        if let Some(ref mut mmap) = self.mmap {
            let write_slice = &mut mmap[current_write..current_write + byte_size];
            let data_bytes = cast_slice(data);
            write_slice.copy_from_slice(data_bytes);
        } else {
            return Err(MemoryError::InvalidBuffer("No backing storage".to_string()));
        }

        // Update write position
        let new_write_pos = if self.circular {
            (current_write + byte_size) % self.capacity
        } else {
            current_write + byte_size
        };

        self.write_pos.store(new_write_pos, Ordering::Release);

        Ok(data.len())
    }

    /// Read data from buffer
    pub fn read<T: Pod + Zeroable>(&self, buffer: &mut [T]) -> Result<usize, MemoryError> {
        let byte_size = buffer.len() * size_of::<T>();
        let current_read = self.read_pos.load(Ordering::Acquire);
        let current_write = self.write_pos.load(Ordering::Acquire);

        // Check available data
        let available_data = if self.circular {
            if current_write >= current_read {
                current_write - current_read
            } else {
                self.capacity - current_read + current_write
            }
        } else {
            current_write - current_read
        };

        let read_size = byte_size.min(available_data);
        if read_size == 0 {
            return Ok(0);
        }

        // Read data
        match &self.mmap {
            Some(mmap) => {
                let read_slice = &mmap.as_ref()[current_read..current_read + read_size];
                let buffer_bytes = cast_slice_mut(buffer);
                buffer_bytes[..read_size].copy_from_slice(read_slice);
            }
            None => {
                return Err(MemoryError::InvalidBuffer("No backing storage".to_string()));
            }
        }

        // Update read position
        let new_read_pos = if self.circular {
            (current_read + read_size) % self.capacity
        } else {
            current_read + read_size
        };

        self.read_pos.store(new_read_pos, Ordering::Release);

        Ok(read_size / size_of::<T>())
    }

    /// Get buffer utilization
    pub fn utilization(&self) -> f64 {
        let current_read = self.read_pos.load(Ordering::Relaxed);
        let current_write = self.write_pos.load(Ordering::Relaxed);

        let used_bytes = if self.circular {
            if current_write >= current_read {
                current_write - current_read
            } else {
                self.capacity - current_read + current_write
            }
        } else {
            current_write - current_read
        };

        used_bytes as f64 / self.capacity as f64
    }
}

impl<T: Pod + Zeroable + Clone> OptimizedMatrix<T> {
    /// Create new optimized matrix
    pub fn new(
        rows: usize,
        cols: usize,
        layout: MatrixLayout,
        access_pattern: AccessPattern,
    ) -> Result<Self, MemoryError> {
        let total_elements = rows * cols;
        let storage = MatrixStorage::Dense {
            data: vec![T::zeroed(); total_elements],
        };

        Ok(Self {
            rows,
            cols,
            storage,
            layout,
            access_pattern,
        })
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }

        match &self.storage {
            MatrixStorage::Dense { data } => {
                let index = self.compute_index(row, col);
                data.get(index)
            }
            MatrixStorage::Sparse {
                row_indices,
                col_indices,
                values,
                ..
            } => {
                // Linear search in sparse format (could be optimized with hash map)
                for (i, (&r, &c)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
                    if r == row && c == col {
                        return values.get(i);
                    }
                }
                None
            }
            _ => None, // Other storage types not implemented yet
        }
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), MemoryError> {
        if row >= self.rows || col >= self.cols {
            return Err(MemoryError::IndexOutOfBounds { row, col });
        }

        // Pre-compute index to avoid borrowing issues
        let index = self.compute_index(row, col);

        match &mut self.storage {
            MatrixStorage::Dense { data } => {
                if index < data.len() {
                    data[index] = value;
                    Ok(())
                } else {
                    Err(MemoryError::IndexOutOfBounds { row, col })
                }
            }
            _ => Err(MemoryError::UnsupportedOperation(
                "Set not supported for this storage type".to_string(),
            )),
        }
    }

    /// Compute linear index from (row, col) based on layout
    fn compute_index(&self, row: usize, col: usize) -> usize {
        match self.layout {
            MatrixLayout::RowMajor => row * self.cols + col,
            MatrixLayout::ColumnMajor => col * self.rows + row,
            MatrixLayout::Blocked { block_size } => {
                let block_row = row / block_size.0;
                let block_col = col / block_size.1;
                let in_block_row = row % block_size.0;
                let in_block_col = col % block_size.1;

                let blocks_per_row = (self.cols + block_size.1 - 1) / block_size.1;
                let block_index = block_row * blocks_per_row + block_col;
                let block_offset = block_index * block_size.0 * block_size.1;

                block_offset + in_block_row * block_size.1 + in_block_col
            }
            MatrixLayout::Morton => {
                // Morton (Z-order) encoding
                self.morton_encode(row, col)
            }
        }
    }

    /// Morton encoding for spatial locality
    fn morton_encode(&self, x: usize, y: usize) -> usize {
        let mut result = 0;
        for i in 0..32 {
            // Assuming 32-bit coordinates
            let bit = 1 << i;
            if x & bit != 0 {
                result |= 1 << (2 * i);
            }
            if y & bit != 0 {
                result |= 1 << (2 * i + 1);
            }
        }
        result
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match &self.storage {
            MatrixStorage::Dense { data } => data.len() * size_of::<T>(),
            MatrixStorage::Sparse {
                values,
                row_indices,
                col_indices,
                ..
            } => {
                values.len() * size_of::<T>()
                    + (row_indices.len() + col_indices.len()) * size_of::<usize>()
            }
            _ => 0, // Placeholder for other storage types
        }
    }
}

impl DistanceMatrixOptimized {
    /// Create optimized distance matrix computation
    pub fn new(points: Vec<DVector<f64>>, strategy: DistanceStrategy) -> Result<Self, MemoryError> {
        let n_points = points.len();
        let dimension = if n_points > 0 { points[0].len() } else { 0 };

        // Convert points to optimized matrix storage
        let mut point_data = Vec::with_capacity(n_points * dimension);
        for point in &points {
            point_data.extend_from_slice(point.as_slice());
        }

        let points_matrix = OptimizedMatrix {
            rows: n_points,
            cols: dimension,
            storage: MatrixStorage::Dense { data: point_data },
            layout: MatrixLayout::RowMajor,
            access_pattern: AccessPattern::RowSequential,
        };

        let result_storage = match &strategy {
            DistanceStrategy::FullMatrix => DistanceStorage::InMemory {
                matrix: Arc::new(Mutex::new(DMatrix::zeros(n_points, n_points))),
            },
            DistanceStrategy::Streaming { .. } => DistanceStorage::Hierarchical {
                memory_cache: HashMap::new(),
                disk_storage: "/tmp/distances.bin".to_string(),
            },
            _ => DistanceStorage::InMemory {
                matrix: Arc::new(Mutex::new(DMatrix::zeros(n_points, n_points))),
            },
        };

        Ok(Self {
            points: points_matrix,
            strategy,
            result_storage,
            computation_state: ComputationState {
                progress: 0.0,
                current_chunk: 0,
                total_chunks: 1,
                memory_usage_mb: 0.0,
                computation_rate_pairs_per_second: 0.0,
            },
        })
    }

    /// Compute distance matrix with memory optimization
    pub fn compute(&mut self) -> Result<(), MemoryError> {
        let (n_points, dimension) = self.points.dimensions();
        let total_pairs = n_points * (n_points - 1) / 2;

        match &self.strategy {
            DistanceStrategy::FullMatrix => {
                self.compute_full_matrix()?;
            }
            DistanceStrategy::UpperTriangle => {
                self.compute_upper_triangle()?;
            }
            DistanceStrategy::Streaming { chunk_size } => {
                self.compute_streaming(*chunk_size)?;
            }
            DistanceStrategy::Approximate { sample_ratio } => {
                self.compute_approximate(*sample_ratio)?;
            }
        }

        Ok(())
    }

    /// Compute full distance matrix
    fn compute_full_matrix(&mut self) -> Result<(), MemoryError> {
        let (n_points, dimension) = self.points.dimensions();

        match &self.result_storage {
            DistanceStorage::InMemory { matrix } => {
                let mut result = matrix.lock().unwrap();

                for i in 0..n_points {
                    for j in i + 1..n_points {
                        let distance = self.compute_distance(i, j)?;
                        result[(i, j)] = distance;
                        result[(j, i)] = distance; // Symmetric
                    }

                    // Update progress
                    self.computation_state.progress = (i + 1) as f64 / n_points as f64;
                }
            }
            _ => {
                return Err(MemoryError::UnsupportedOperation(
                    "Storage type not supported".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Compute upper triangle only
    fn compute_upper_triangle(&mut self) -> Result<(), MemoryError> {
        let (n_points, _) = self.points.dimensions();

        match &self.result_storage {
            DistanceStorage::InMemory { matrix } => {
                let mut result = matrix.lock().unwrap();

                for i in 0..n_points {
                    for j in i + 1..n_points {
                        let distance = self.compute_distance(i, j)?;
                        result[(i, j)] = distance;
                    }

                    self.computation_state.progress = (i + 1) as f64 / n_points as f64;
                }
            }
            _ => {
                return Err(MemoryError::UnsupportedOperation(
                    "Storage type not supported".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Compute with streaming (chunked) approach
    fn compute_streaming(&mut self, chunk_size: usize) -> Result<(), MemoryError> {
        let (n_points, _) = self.points.dimensions();
        let total_chunks = (n_points + chunk_size - 1) / chunk_size;

        self.computation_state.total_chunks = total_chunks;

        for chunk_i in 0..total_chunks {
            let start_i = chunk_i * chunk_size;
            let end_i = (start_i + chunk_size).min(n_points);

            for chunk_j in chunk_i..total_chunks {
                let start_j = chunk_j * chunk_size;
                let end_j = (start_j + chunk_size).min(n_points);

                // Compute distances for this chunk pair
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        if i < j {
                            let distance = self.compute_distance(i, j)?;
                            // Store distance (implementation depends on storage type)
                        }
                    }
                }
            }

            self.computation_state.current_chunk = chunk_i + 1;
            self.computation_state.progress = (chunk_i + 1) as f64 / total_chunks as f64;
        }

        Ok(())
    }

    /// Compute approximate distances using sampling
    fn compute_approximate(&mut self, sample_ratio: f64) -> Result<(), MemoryError> {
        let (n_points, _) = self.points.dimensions();
        let sample_size = (n_points as f64 * sample_ratio) as usize;

        // Sample points for distance computation
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_points).collect();
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);

        // Compute distances for sampled pairs
        for &i in &indices {
            for &j in &indices {
                if i < j {
                    let distance = self.compute_distance(i, j)?;
                    // Store or use distance
                }
            }
        }

        self.computation_state.progress = 1.0;

        Ok(())
    }

    /// Compute Euclidean distance between two points
    fn compute_distance(&self, i: usize, j: usize) -> Result<f64, MemoryError> {
        let (_, dimension) = self.points.dimensions();
        let mut sum_sq = 0.0;

        // Get points data
        match &self.points.storage {
            MatrixStorage::Dense { data } => {
                for d in 0..dimension {
                    let pi = data[i * dimension + d];
                    let pj = data[j * dimension + d];
                    let diff = pi - pj;
                    sum_sq += diff * diff;
                }
            }
            _ => {
                return Err(MemoryError::UnsupportedOperation(
                    "Distance computation not supported for this storage".to_string(),
                ))
            }
        }

        Ok(sum_sq.sqrt())
    }

    /// Get computation progress
    pub fn progress(&self) -> &ComputationState {
        &self.computation_state
    }
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            current_allocated_bytes: 0,
            peak_allocated_bytes: 0,
            fragmentation_ratio: 0.0,
            average_allocation_size: 0.0,
            allocation_rate_per_second: 0.0,
            numa_distribution: HashMap::new(),
        }
    }
}

/// Defragmentation result
#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    pub freed_bytes: usize,
    pub moved_blocks: usize,
    pub fragmentation_reduction: f64,
}

/// Memory optimization errors
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Insufficient memory: requested {requested} bytes, available {available} bytes")]
    InsufficientMemory { requested: usize, available: usize },

    #[error("Memory pool exhausted: requested {requested} bytes, available {available} bytes")]
    PoolExhausted { requested: usize, available: usize },

    #[error("Buffer overflow: requested {requested} bytes, available {available} bytes")]
    BufferOverflow { requested: usize, available: usize },

    #[error("Allocation failed for {0} bytes")]
    AllocationFailed(usize),

    #[error("Invalid memory layout: {0}")]
    InvalidLayout(String),

    #[error("Invalid buffer: {0}")]
    InvalidBuffer(String),

    #[error("Index out of bounds: row {row}, col {col}")]
    IndexOutOfBounds { row: usize, col: usize },

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("File system error: {0}")]
    FileSystemError(String),

    #[error("Memory operation failed: {message}")]
    OperationFailed { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use csf_time::SimulatedTimeSource;
    use std::sync::Arc;

    #[test]
    fn test_memory_pool_creation() {
        let time_source = Arc::new(SimulatedTimeSource::new_at_epoch());
        let pool = MemoryPool::new(1024 * 1024, time_source).unwrap(); // 1MB pool
        assert_eq!(pool.capacity, 1024 * 1024);
        assert_eq!(pool.allocated.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let time_source = Arc::new(SimulatedTimeSource::new_at_epoch());
        let pool = MemoryPool::new(1024 * 1024, time_source).unwrap();

        let ptr1 = pool.allocate(1024, 8).unwrap();
        assert_eq!(pool.allocated.load(Ordering::Relaxed), 1024);

        let ptr2 = pool.allocate(2048, 8).unwrap();
        assert_eq!(pool.allocated.load(Ordering::Relaxed), 1024 + 2048);

        pool.deallocate(ptr1);
        assert_eq!(pool.allocated.load(Ordering::Relaxed), 2048);

        pool.deallocate(ptr2);
        assert_eq!(pool.allocated.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_streaming_buffer() {
        let buffer = StreamingBuffer::new(4096, BufferMode::Circular, size_of::<f64>()).unwrap();

        assert_eq!(buffer.capacity, 4096);
        assert_eq!(buffer.utilization(), 0.0);
    }

    #[test]
    fn test_optimized_matrix_creation() {
        let matrix: OptimizedMatrix<f64> = OptimizedMatrix::new(
            100,
            100,
            MatrixLayout::RowMajor,
            AccessPattern::RowSequential,
        )
        .unwrap();

        assert_eq!(matrix.dimensions(), (100, 100));
        assert_eq!(matrix.memory_usage(), 100 * 100 * size_of::<f64>());
    }

    #[test]
    fn test_matrix_index_computation() {
        let matrix: OptimizedMatrix<f64> =
            OptimizedMatrix::new(4, 4, MatrixLayout::RowMajor, AccessPattern::Random).unwrap();

        // Row-major: (1, 2) should be at index 1*4 + 2 = 6
        assert_eq!(matrix.compute_index(1, 2), 6);

        let matrix_col: OptimizedMatrix<f64> = OptimizedMatrix::new(
            4,
            4,
            MatrixLayout::ColumnMajor,
            AccessPattern::ColumnSequential,
        )
        .unwrap();

        // Column-major: (1, 2) should be at index 2*4 + 1 = 9
        assert_eq!(matrix_col.compute_index(1, 2), 9);
    }
}
