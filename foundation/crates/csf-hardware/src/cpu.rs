//! CPU control and topology management

use anyhow::{Context, Result};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use raw_cpuid::CpuId;
use std::fs;

/// Set CPU affinity for current thread
pub fn set_cpu_affinity(cpu_id: u32) -> Result<()> {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(cpu_id as usize)?;

    sched_setaffinity(Pid::from_raw(0), &cpu_set).context("Failed to set CPU affinity")?;

    Ok(())
}

/// Set thread priority to real-time
pub fn set_realtime_priority(priority: i32) -> Result<()> {
    use libc::{sched_param, sched_setscheduler, SCHED_FIFO};

    let params = sched_param {
        sched_priority: priority,
    };

    let ret = unsafe { sched_setscheduler(0, SCHED_FIFO, &params) };

    if ret != 0 {
        return Err(anyhow::anyhow!(
            "Failed to set realtime priority: {}",
            std::io::Error::last_os_error()
        ));
    }

    Ok(())
}

/// Disable CPU frequency scaling for deterministic performance
pub fn disable_cpu_scaling(cpu_id: u32) -> Result<()> {
    // Set governor to performance mode
    let governor_path = format!(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor",
        cpu_id
    );

    if std::path::Path::new(&governor_path).exists() {
        fs::write(&governor_path, b"performance")
            .context("Failed to set CPU governor to performance")?;
    }

    // Disable Intel turbo boost for consistency
    let turbo_path = "/sys/devices/system/cpu/intel_pstate/no_turbo";
    if std::path::Path::new(turbo_path).exists() {
        fs::write(turbo_path, b"1").context("Failed to disable turbo boost")?;
    }

    // Disable C-states for low latency
    disable_cpu_idle_states(cpu_id)?;

    Ok(())
}

/// Disable CPU idle states (C-states) for low latency
fn disable_cpu_idle_states(cpu_id: u32) -> Result<()> {
    let idle_path = format!("/sys/devices/system/cpu/cpu{}/cpuidle", cpu_id);

    if std::path::Path::new(&idle_path).exists() {
        for entry in fs::read_dir(&idle_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let disable_path = path.join("disable");
                if disable_path.exists() {
                    fs::write(&disable_path, b"1").ok();
                }
            }
        }
    }

    Ok(())
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Physical package (socket) ID
    pub package_id: u32,
    /// Physical core ID within package
    pub core_id: u32,
    /// Logical CPU ID
    pub cpu_id: u32,
    /// NUMA node
    pub numa_node: u32,
    /// Sibling CPUs (hyperthreading)
    pub siblings: Vec<u32>,
}

/// Get CPU topology for all CPUs
pub fn get_cpu_topology() -> Result<Vec<CpuTopology>> {
    let mut topology = Vec::new();

    let cpu_count = num_cpus::get();

    for cpu_id in 0..cpu_count {
        let package_id = read_cpu_file(cpu_id, "topology/physical_package_id")?;
        let core_id = read_cpu_file(cpu_id, "topology/core_id")?;
        let numa_node = read_numa_node(cpu_id)?;
        let siblings = read_cpu_siblings(cpu_id)?;

        topology.push(CpuTopology {
            package_id: package_id as u32,
            core_id: core_id as u32,
            cpu_id: cpu_id as u32,
            numa_node: numa_node as u32,
            siblings,
        });
    }

    Ok(topology)
}

fn read_cpu_file(cpu_id: usize, file: &str) -> Result<usize> {
    let path = format!("/sys/devices/system/cpu/cpu{}/{}", cpu_id, file);
    let contents = fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path))?;

    contents
        .trim()
        .parse()
        .with_context(|| format!("Failed to parse {}", path))
}

fn read_numa_node(cpu_id: usize) -> Result<usize> {
    // Try to read NUMA node from sysfs
    for node_id in 0..8 {
        let path = format!("/sys/devices/system/node/node{}/cpu{}", node_id, cpu_id);
        if std::path::Path::new(&path).exists() {
            return Ok(node_id);
        }
    }

    // Fallback: assume single NUMA node
    Ok(0)
}

fn read_cpu_siblings(cpu_id: usize) -> Result<Vec<u32>> {
    let path = format!(
        "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
        cpu_id
    );

    if let Ok(contents) = fs::read_to_string(&path) {
        parse_cpu_list(&contents)
    } else {
        // Fallback: no siblings
        Ok(vec![cpu_id as u32])
    }
}

fn parse_cpu_list(list: &str) -> Result<Vec<u32>> {
    let mut cpus = Vec::new();

    for part in list.trim().split(',') {
        if part.contains('-') {
            let range: Vec<&str> = part.split('-').collect();
            if range.len() == 2 {
                let start: u32 = range[0].parse()?;
                let end: u32 = range[1].parse()?;
                for cpu in start..=end {
                    cpus.push(cpu);
                }
            }
        } else {
            cpus.push(part.parse()?);
        }
    }

    Ok(cpus)
}

/// Get CPU information using CPUID
pub fn get_cpu_info() -> Result<crate::CpuInfo> {
    let cpuid = CpuId::new();

    let mut features = crate::CpuFeatures::default();

    // Check CPU features
    if let Some(feature_info) = cpuid.get_feature_info() {
        features.aes_ni = feature_info.has_aesni();
        features.rdrand = feature_info.has_rdrand();
    }

    if let Some(extended_features) = cpuid.get_extended_feature_info() {
        features.avx2 = extended_features.has_avx2();
        features.sha = extended_features.has_sha();
        features.rdseed = extended_features.has_rdseed();

        // Check for AVX-512
        features.avx512 = extended_features.has_avx512f();

        // Check for TSX
        features.tsx = extended_features.has_hle() || extended_features.has_rtm();
    }

    // Get cache information
    let cache_sizes = get_cache_sizes(&cpuid)?;

    // Get CPU frequency
    let frequency_mhz = get_cpu_frequency()?;

    // Get topology
    let topology = get_cpu_topology()?;
    let physical_cores = topology
        .iter()
        .map(|t| (t.package_id, t.core_id))
        .collect::<std::collections::HashSet<_>>()
        .len() as u32;

    let logical_cores = topology.len() as u32;

    // Get NUMA information
    let numa_nodes = get_numa_nodes()?;

    Ok(crate::CpuInfo {
        physical_cores,
        logical_cores,
        frequency_mhz,
        cache_sizes,
        numa_nodes,
        features,
    })
}

fn get_cache_sizes<R: raw_cpuid::CpuIdReader>(cpuid: &CpuId<R>) -> Result<crate::CacheSizes> {
    let mut l1d = 0;
    let mut l1i = 0;
    let mut l2 = 0;
    let mut l3 = 0;
    let line_size = 64; // Default

    // Intel cache info
    if let Some(cache_info) = cpuid.get_cache_info() {
        for cache in cache_info {
            // Use the raw cache descriptor to infer cache sizes
            // This is a simplified interpretation - actual cache detection
            // would need more sophisticated parsing of cache descriptors
            match cache.num {
                0x2C => l1d = 32 * 1024,  // 32KB L1 data cache
                0x30 => l1i = 32 * 1024,  // 32KB L1 instruction cache
                0x7C => l2 = 1024 * 1024, // 1MB L2 cache
                0x7D => l2 = 2048 * 1024, // 2MB L2 cache
                0x7F => l2 = 512 * 1024,  // 512KB L2 cache
                0x85 => l2 = 2048 * 1024, // 2MB L2 cache
                0x86 => l2 = 512 * 1024,  // 512KB L2 cache
                0x87 => l2 = 1024 * 1024, // 1MB L2 cache
                _ => {}                   // Unknown descriptor
            }
        }
    }

    // Fallback to typical values if detection failed
    if l1d == 0 {
        l1d = 32 * 1024; // 32KB
        l1i = 32 * 1024;
        l2 = 256 * 1024; // 256KB
        l3 = 8 * 1024 * 1024; // 8MB
    }

    Ok(crate::CacheSizes {
        l1d,
        l1i,
        l2,
        l3,
        line_size,
    })
}

fn get_cpu_frequency() -> Result<u32> {
    // Try to read from cpuinfo_max_freq
    let path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq";
    if let Ok(contents) = fs::read_to_string(path) {
        if let Ok(khz) = contents.trim().parse::<u32>() {
            return Ok(khz / 1000); // Convert to MHz
        }
    }

    // Fallback: parse /proc/cpuinfo
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    for line in cpuinfo.lines() {
        if line.starts_with("cpu MHz") {
            if let Some(mhz_str) = line.split(':').nth(1) {
                if let Ok(mhz) = mhz_str.trim().parse::<f32>() {
                    return Ok(mhz as u32);
                }
            }
        }
    }

    // Default fallback
    Ok(2000) // 2 GHz
}

fn get_numa_nodes() -> Result<Vec<crate::NumaNode>> {
    let mut nodes = Vec::new();

    // Read NUMA topology from sysfs
    let node_path = "/sys/devices/system/node";
    if let Ok(entries) = fs::read_dir(node_path) {
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if let Some(stripped) = name_str.strip_prefix("node") {
                if let Ok(node_id) = stripped.parse::<u32>() {
                    let cpus = read_node_cpus(node_id)?;
                    let memory_bytes = read_node_memory(node_id)?;
                    let distances = read_node_distances(node_id)?;

                    nodes.push(crate::NumaNode {
                        id: node_id,
                        cpus,
                        memory_bytes,
                        distances,
                    });
                }
            }
        }
    }

    // Fallback: single NUMA node
    if nodes.is_empty() {
        let cpus: Vec<u32> = (0..num_cpus::get() as u32).collect();
        let memory_bytes = get_total_memory()?;

        nodes.push(crate::NumaNode {
            id: 0,
            cpus,
            memory_bytes,
            distances: vec![10], // Local distance
        });
    }

    Ok(nodes)
}

fn read_node_cpus(node_id: u32) -> Result<Vec<u32>> {
    let path = format!("/sys/devices/system/node/node{}/cpulist", node_id);
    if let Ok(contents) = fs::read_to_string(&path) {
        parse_cpu_list(&contents)
    } else {
        Ok(Vec::new())
    }
}

fn read_node_memory(node_id: u32) -> Result<usize> {
    let path = format!("/sys/devices/system/node/node{}/meminfo", node_id);
    if let Ok(contents) = fs::read_to_string(&path) {
        for line in contents.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(kb) = parts[3].parse::<usize>() {
                        return Ok(kb * 1024); // Convert to bytes
                    }
                }
            }
        }
    }
    Ok(0)
}

fn read_node_distances(node_id: u32) -> Result<Vec<u32>> {
    let path = format!("/sys/devices/system/node/node{}/distance", node_id);
    if let Ok(contents) = fs::read_to_string(&path) {
        Ok(contents
            .split_whitespace()
            .filter_map(|s| s.parse::<u32>().ok())
            .collect::<Vec<_>>())
    } else {
        Ok(vec![10]) // Default local distance
    }
}

fn get_total_memory() -> Result<usize> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<usize>() {
                    return Ok(kb * 1024); // Convert to bytes
                }
            }
        }
    }
    Ok(0)
}

/// Check required CPU features
pub fn check_required_features() -> Result<()> {
    let cpuid = CpuId::new();

    // Check for x86_64
    #[cfg(not(target_arch = "x86_64"))]
    return Err(anyhow::anyhow!("ARES CSF requires x86_64 architecture"));

    // Check for required features
    if let Some(feature_info) = cpuid.get_feature_info() {
        if !feature_info.has_sse2() {
            return Err(anyhow::anyhow!("CPU must support SSE2"));
        }
    }

    if let Some(extended_features) = cpuid.get_extended_feature_info() {
        if !extended_features.has_avx2() {
            return Err(anyhow::anyhow!("CPU must support AVX2"));
        }
    }

    Ok(())
}
