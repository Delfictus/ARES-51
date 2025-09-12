//! CPU cache control and optimization

use std::sync::atomic::{compiler_fence, Ordering};

/// Cache line size (typically 64 bytes on modern x86_64)
pub const CACHE_LINE_SIZE: usize = 64;

/// Prefetch data into cache
#[inline(always)]
pub fn prefetch_data<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

/// Prefetch data for write
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

/// Prefetch data into L2 cache
#[inline(always)]
pub fn prefetch_l2<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
}

/// Prefetch data non-temporally (bypass cache)
#[inline(always)]
pub fn prefetch_nta<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};
        _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
    }
}

/// Flush cache line
#[inline(always)]
pub fn flush_cache_line<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_mm_clflush;
        _mm_clflush(ptr as *const u8);
    }
}

/// Memory fence for ordering
#[inline(always)]
pub fn memory_fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_mm_mfence;
        _mm_mfence();
    }

    compiler_fence(Ordering::SeqCst);
}

/// Store fence
#[inline(always)]
pub fn store_fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_mm_sfence;
        _mm_sfence();
    }

    compiler_fence(Ordering::Release);
}

/// Load fence
#[inline(always)]
pub fn load_fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_mm_lfence;
        _mm_lfence();
    }

    compiler_fence(Ordering::Acquire);
}

/// Cache-aligned allocation marker
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> std::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Ensure data is cache-aligned
pub fn is_cache_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(CACHE_LINE_SIZE)
}

/// Round up to cache line boundary
pub const fn cache_align_size(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// Cache line padding to avoid false sharing
#[repr(align(64))]
struct CacheLinePad(#[allow(dead_code)] [u8; 64]);

/// Pad struct to avoid false sharing
pub struct CachePadded<T> {
    value: T,
    _pad: std::mem::MaybeUninit<CacheLinePad>,
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            _pad: std::mem::MaybeUninit::uninit(),
        }
    }
}

impl<T> CachePadded<T> {
    /// Create a new cache-padded value
    pub const fn new(value: T) -> Self {
        Self {
            value,
            _pad: std::mem::MaybeUninit::uninit(),
        }
    }
}

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// Non-temporal (streaming) store
///
/// # Safety
///
/// Caller must ensure that `src` and `dst` are valid pointers for `len` bytes,
/// and that the memory regions do not overlap.
#[inline(always)]
pub unsafe fn stream_store(dst: *mut u8, src: *const u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::_mm_stream_si64;

        let mut offset = 0;

        // Stream 64-bit values
        while offset + 8 <= len {
            let value = *(src.add(offset) as *const i64);
            _mm_stream_si64(dst.add(offset) as *mut i64, value);
            offset += 8;
        }

        // Handle remainder
        while offset < len {
            *dst.add(offset) = *src.add(offset);
            offset += 1;
        }

        // Ensure stores are visible
        store_fence();
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

/// Optimize memory access pattern for cache
pub struct CacheOptimizer {
    l1_size: usize,
    #[allow(dead_code)]
    l2_size: usize,
    #[allow(dead_code)]
    l3_size: usize,
    line_size: usize,
}

impl CacheOptimizer {
    /// Create a new cache optimizer with given cache sizes
    pub fn new(cache_sizes: &crate::CacheSizes) -> Self {
        Self {
            l1_size: cache_sizes.l1d,
            l2_size: cache_sizes.l2,
            l3_size: cache_sizes.l3,
            line_size: cache_sizes.line_size,
        }
    }

    /// Calculate optimal block size for cache blocking
    pub fn optimal_block_size(&self, element_size: usize) -> usize {
        // Use 1/2 of L1 cache for working set
        let working_set_size = self.l1_size / 2;
        let elements_per_line = self.line_size / element_size;
        let lines_in_working_set = working_set_size / self.line_size;

        // Return number of elements that fit in working set
        lines_in_working_set * elements_per_line
    }

    /// Calculate optimal tile size for 2D blocking
    pub fn optimal_tile_size(&self, element_size: usize) -> (usize, usize) {
        let block_size = self.optimal_block_size(element_size);
        let tile_size = (block_size as f64).sqrt() as usize;

        // Ensure tile dimensions are multiples of cache line
        let elements_per_line = self.line_size / element_size;
        let aligned_tile = tile_size.div_ceil(elements_per_line) * elements_per_line;

        (aligned_tile, aligned_tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_alignment() {
        let aligned = CacheAligned::new(42u64);
        let ptr = &aligned as *const _;
        assert!(is_cache_aligned(ptr));
    }

    #[test]
    fn test_cache_padding() {
        // Ensure CachePadded is exactly one cache line
        assert_eq!(std::mem::size_of::<CachePadded<u8>>(), CACHE_LINE_SIZE);
        assert_eq!(std::mem::size_of::<CachePadded<u64>>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_cache_align_size() {
        assert_eq!(cache_align_size(1), 64);
        assert_eq!(cache_align_size(64), 64);
        assert_eq!(cache_align_size(65), 128);
    }
}
