# ARES ChronoSynclastic Fabric (CSF) Core

High-performance distributed computation fabric for quantum-temporal processing, variational optimization, and enterprise-scale tensor operations.

[![Crates.io](https://img.shields.io/crates/v/ares-csf-core.svg)](https://crates.io/crates/ares-csf-core)
[![Documentation](https://docs.rs/ares-csf-core/badge.svg)](https://docs.rs/ares-csf-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The CSF Core provides the foundational infrastructure for building distributed, quantum-aware computational systems. It combines traditional high-performance computing with quantum-inspired algorithms and temporal synchronization for next-generation applications.

## Features

- 🌌 **Quantum-Temporal Processing**: Hybrid quantum-classical computation with temporal coherence
- ⚡ **High-Performance Tensors**: Enterprise-grade tensor operations with GPU acceleration
- 🔗 **Distributed Computing**: Fault-tolerant distributed computation with consensus protocols
- 🧮 **Variational Optimization**: Advanced optimization algorithms for complex search spaces
- 📊 **Streaming Processing**: Real-time data processing with zero-copy operations
- 🏢 **Enterprise Integration**: Production-ready systems with monitoring and observability
- 🧠 **Computational Intelligence**: Advanced algorithms for scientific computing
- 🔬 **Research Tools**: Cutting-edge algorithms for scientific computing

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ares-csf-core = "0.1.0"
```

### Basic Usage

```rust
use ares_csf_core::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Create a basic CSF computation context
    let config = CSFConfig::default();
    let mut context = CSFContext::new(config)?;

    // Initialize HPC runtime for high-performance computing
    context.initialize_hpc().await?;

    // Create a simple tensor and perform operations
    let tensor = RealTensor::new(vec![3, 3]);
    println!("Created {}x{} tensor", tensor.shape()[0], tensor.shape()[1]);

    // Run proof of power demonstration
    let results = run_proof_of_power_demo().await?;
    println!("Achieved {:?} certification level", results.certification_level);

    Ok(())
}
```

### Quantum-Temporal Processing

```rust
use ares_csf_core::prelude::*;

// Create phase states for quantum processing
let phase = Phase::new(std::f64::consts::PI / 4.0);
let coherence = 0.95;
let state = PhaseState::now(phase, coherence);

// Create quantum state
let amplitudes = vec![
    num_complex::Complex64::new(0.707, 0.0),
    num_complex::Complex64::new(0.0, 0.707),
];
let quantum_state = QuantumState::new(amplitudes, phase, coherence);

println!("Quantum state fidelity: {:.3}", quantum_state.amplitudes[0].norm());
```

### High-Performance Tensor Operations

```rust
use ares_csf_core::prelude::*;

// Create matrices
let mut a = RealTensor::new(vec![2, 2]);
a.set(&[0, 0], 1.0)?;
a.set(&[0, 1], 2.0)?;
a.set(&[1, 0], 3.0)?;
a.set(&[1, 1], 4.0)?;

let mut b = RealTensor::new(vec![2, 2]);
b.set(&[0, 0], 5.0)?;
b.set(&[0, 1], 6.0)?;
b.set(&[1, 0], 7.0)?;
b.set(&[1, 1], 8.0)?;

// Perform matrix multiplication
let result = tensor_multiply(&a, &b)?;
println!("Matrix multiplication result: {:?}", result.get(&[0, 0]));

// Calculate eigenvalues
let (eigenvalues, eigenvectors) = eigenvalue_decomposition(&a)?;
println!("Dominant eigenvalue: {:.3}", eigenvalues[0]);
```

### Distributed Computing

```rust
use ares_csf_core::prelude::*;

// Create distributed compute engine
let compute = LocalDistributedCompute::new()?;

// Submit a computation task
let task = ComputeTask {
    id: "matrix-multiply".to_string(),
    task_type: "linear-algebra".to_string(),
    input_data: vec![1, 2, 3, 4],
    requirements: ResourceRequirements {
        cpu_cores: 2,
        memory_bytes: 1024 * 1024, // 1MB
        requires_gpu: false,
        estimated_runtime: 10, // 10 seconds
    },
    priority: Priority::High,
};

let handle = compute.submit_task(task).await?;
println!("Task submitted: {}", handle.task_id);

// Check cluster status
let status = compute.cluster_status().await?;
println!("CPU utilization: {:.1}%", status.cpu_utilization());
```

### Computational Data Processing

```rust
use ares_csf_core::prelude::*;

// Create computational data source
let metadata = DataSourceMetadata {
    id: "tensor-stream".to_string(),
    name: "Tensor Data Stream".to_string(),
    description: "Real-time tensor computations".to_string(),
    update_frequency_ms: 100,
    retention_period: std::time::Duration::from_secs(3600),
    real_time: true,
};

// Process computational data
println!("Data source: {} - Real-time: {}", metadata.name, metadata.real_time);
```

### Variational Optimization

```rust
use ares_csf_core::prelude::*;

// Create advanced optimizer
let optimizer = AdvancedOptimizer::new(OptimizationAlgorithm::QuantumInspired)
    .with_learning_rate(0.01)
    .with_max_iterations(1000);

// Optimize parameters
let initial_params = vec![1.0, 2.0, 3.0];
let optimized = optimizer.optimize(initial_params)?;
println!("Optimized parameters: {:?}", optimized);

// Create energy functional
let coefficients = vec![1.0, -0.5, 0.25];
let energy_func = EnergyFunctional::new(coefficients);

let phase_state = PhaseState::now(Phase::new(0.5), 0.9);
let energy = energy_func.evaluate(&phase_state)?;
println!("System energy: {:.3}", energy);
```

### Monitoring and Health Checks

```rust
use ares_csf_core::prelude::*;

// Create health monitor
let config = MonitorConfig::default();
let monitor = HealthMonitor::new(config);

// Check system health
let health = monitor.check_health().await?;
println!("System health: {:?}", health.overall_health);
println!("CPU usage: {:.1}%", health.report.cpu_stats.current_usage);
println!("Memory usage: {:.1}%", health.report.memory_stats.usage_percentage);
```

## Feature Flags

Configure the crate with feature flags to enable specific functionality:

```toml
[dependencies]
ares-csf-core = { version = "0.1.0", features = ["tensor-ops", "hpc", "enterprise"] }
```

### Available Features

- **`std`** (default) - Standard library support
- **`tensor-ops`** (default) - Enable tensor operations with nalgebra/ndarray
- **`streaming`** (default) - Enable streaming data processing
- **`hpc`** - High-performance computing with GPU support
- **`gpu`** - GPU acceleration with CUDA/OpenCL
- **`cuda`** - NVIDIA CUDA support
- **`simd`** - SIMD optimizations
- **`quantum`** - Quantum computing backend
- **`enterprise`** - Enterprise features (HTTP clients, monitoring)
- **`profiling`** - Performance profiling tools
- **`testing`** - Property-based testing utilities

## Architecture

### Core Components

- **Ports & Adapters**: Hexagonal architecture with clean interfaces
- **Tensor Operations**: High-performance linear algebra with GPU acceleration
- **Quantum Processing**: Quantum state management and coherence tracking
- **Temporal Synchronization**: Hybrid logical clocks and causality tracking
- **Distributed Computing**: Fault-tolerant task distribution and consensus
- **Variational Optimization**: Advanced optimization algorithms
- **Computational Intelligence**: Advanced algorithms for scientific computing
- **Monitoring**: Real-time health monitoring and observability

### Performance Targets

- **Tensor Operations**: 10+ TFLOPS on consumer hardware
- **Network Latency**: < 10μs for local operations
- **Memory Efficiency**: Zero-copy streaming for TB+ datasets
- **Quantum Fidelity**: > 99.9% gate fidelity at scale
- **Computational Latency**: < 100ns operation execution
- **Temporal Precision**: Nanosecond synchronization accuracy

## Examples

See the [examples](examples/) directory for comprehensive usage examples:

- [Basic Operations](examples/basic_operations.rs) - Core functionality demonstration
- [Quantum Computing](examples/quantum_demo.rs) - Quantum state manipulation
- [HPC Workloads](examples/hpc_demo.rs) - Distributed computing examples
- [Scientific Computing](examples/scientific_demo.rs) - Research applications
- [Monitoring](examples/monitoring_demo.rs) - Health checking and observability

## Integration

### With ARES Neuromorphic Core

```rust
use ares_csf_core::prelude::*;
use ares_neuromorphic_core::NeuromorphicEngine;

// CSF provides the computation fabric
let csf_config = CSFConfig::default();
let csf_context = CSFContext::new(csf_config)?;

// Neuromorphic engine provides spike processing
let neuro_config = ares_neuromorphic_core::EngineConfig::default();
let neuro_engine = NeuromorphicEngine::new(neuro_config).await?;

// Process market data through both systems
let market_data = /* your market data */;
let spike_pattern = /* convert to spikes */;
let prediction = neuro_engine.process(&spike_pattern).await?;
```

### With ARES Spike Encoding

```rust
use ares_csf_core::prelude::*;
use ares_spike_encoding::{SpikeEncoder, EncodingMethod};

// Create spike encoder
let mut encoder = SpikeEncoder::new(1000, 1000.0)?
    .with_method(EncodingMethod::Rate)?;

// Encode market data to spikes
let market_data = MarketData::new("BTC-USD", 50000.0, 1.5);
let spike_pattern = encoder.encode(&market_data)?;

// Process with CSF quantum backend
let quantum_state = /* convert spikes to quantum state */;
```

## Benchmarking

Run the included benchmarks to validate performance:

```bash
# Tensor operation benchmarks
cargo bench tensor_benchmarks --features tensor-ops

# Quantum computation benchmarks  
cargo bench quantum_benchmarks --features quantum

# HPC distributed computing benchmarks
cargo bench hpc_benchmarks --features hpc

# Full proof of power demonstration
cargo run --example proof_of_power --release
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this crate in academic research, please cite:

```bibtex
@software{ares_csf_core,
  title={ARES ChronoSynclastic Fabric Core: High-Performance Quantum-Temporal Computing},
  author={Serfaty, Ididia},
  year={2025},
  url={https://github.com/ares-systems/ares-csf-core}
}
```

## Related Projects

- [ARES Neuromorphic Core](https://github.com/ares-systems/ares-neuromorphic-core) - Neuromorphic spike processing
- [ARES Spike Encoding](https://github.com/ares-systems/ares-spike-encoding) - Neural spike encoding algorithms
- [ARES-51](https://github.com/ares-systems/ares-51) - Complete neuromorphic computational system

---

**Built for the future of computation** 🚀