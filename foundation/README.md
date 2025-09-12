# NovaCore - ARES Chronosynclastic Fabric (CSF)

[![Build Status](https://github.com/Delfictus/NovaCore/workflows/CI/badge.svg)](https://github.com/Delfictus/NovaCore/actions)
[![Documentation](https://docs.rs/nova-core/badge.svg)](https://docs.rs/nova-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)

**NovaCore** is a next-generation real-time computing platform implementing the ARES Chronosynclastic Fabric (CSF) - a revolutionary architecture that unifies temporal task management, distributed processing, and neuromorphic computing paradigms. Built for extreme performance and designed for the future of edge computing, AI/ML workloads, and real-time systems.

## 🚀 Key Features

### Core Technologies

- **⚡ Temporal Task Weaver (TTW)**: Causality-aware scheduling with predictive temporal analysis and quantum-inspired optimization
- **📡 Phase Coherence Bus (PCB)**: Zero-copy, lock-free message passing with hardware-accelerated routing
- **🧠 C-LOGIC Integration**: Advanced cognitive computing modules:
  - **DRPP** (Dynamic Resonance Pattern Processor): Neural oscillator networks for pattern detection
  - **ADP** (Adaptive Distributed Processing): Self-balancing compute fabric
  - **EGC** (Emergent Governance Controller): Autonomous system governance
  - **EMS** (Emotional Modeling System): Affective computing and decision weighting
- **📊 Historical Market Data System**: Multi-source data fetching and streaming replay for model validation
- **🔥 MLIR Runtime**: Multi-backend hardware acceleration (CPU, CUDA, Vulkan, WebGPU, TPU)
- **🔒 Secure Immutable Ledger (SIL)**: Cryptographic audit trail with quantum-resistant signatures
- **🌐 Distributed Network**: QUIC/TCP/WebSocket support with automatic peer discovery

### Performance & Scalability

- Sub-microsecond latency for critical paths
- Horizontal scaling to thousands of nodes
- GPU/TPU acceleration for ML workloads
- Real-time telemetry and distributed tracing
- Memory-safe Rust implementation

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│                    (Your Real-Time Systems)                      │
├─────────────────────────────────────────────────────────────────┤
│                          FFI Bindings                            │
│              C API │ Python API │ WebAssembly API               │
├─────────────────────────────────────────────────────────────────┤
│                       C-LOGIC Subsystem                          │
│  ┌─────────┬──────────┬──────────┬──────────┬──────────────┐   │
│  │  DRPP   │   ADP    │   EGC    │   EMS    │     SIL      │   │
│  │ Pattern │ Adaptive │Governance│ Emotion  │   Secure     │   │
│  │Detection│Computing │ Control  │ Modeling │   Ledger     │   │
│  └─────────┴──────────┴──────────┴──────────┴──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      MLIR Runtime Engine                         │
│         ┌──────┬──────┬────────┬─────────┬──────────┐          │
│         │ CPU  │ CUDA │ Vulkan │ WebGPU  │   TPU    │          │
│         └──────┴──────┴────────┴─────────┴──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                   Phase Coherence Bus (PCB)                      │
│                  Zero-Copy Message Transport                     │
├─────────────────────────────────────────────────────────────────┤
│                  Temporal Task Weaver (TTW)                      │
│               Causality-Aware Task Scheduling                    │
├─────────────────────────────────────────────────────────────────┤
│                       CSF Core Runtime                           │
│                   Memory Pool │ Time Sync                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Dependencies

For a complete list of all dependencies, see [DEPENDENCIES.md](DEPENDENCIES.md). Here's what you need to get started:

### Quick Install (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config \
    libssl-dev libclang-dev protobuf-compiler

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build NovaCore
git clone https://github.com/Delfictus/NovaCore.git
cd NovaCore/ChronoFabric
cargo build --release
```

### Core Requirements

- **Rust**: 1.75+ (via [rustup](https://rustup.rs/))
- **Build Tools**: cmake 3.20+, gcc 9+/clang 11+
- **Libraries**: OpenSSL, protobuf
- **Optional**: CUDA 11.0+ (GPU), Docker (containers)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Delfictus/NovaCore.git
cd NovaCore/ChronoFabric

# Set up development environment
./scripts/setup-dev.sh

# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_sensor_fusion
cargo run --example historical_data_validation
```

### Docker Deployment

```bash
# Build Docker image
docker build -t novacore .

# Run container
docker run -d -p 8080:8080 -p 9090:9090 novacore

# Or use docker-compose for full stack
docker-compose up -d
```

## 📚 Documentation

- [Development Guide](DEVELOPMENT.md) - Detailed development setup and guidelines
- [API Documentation](https://docs.rs/nova-core) - Full API reference
- [Architecture Deep Dive](docs/architecture.md) - In-depth technical details
- [Performance Tuning](docs/performance.md) - Optimization guidelines
- **Historical Market Data System**:
  - [Quick Start Guide](docs/QUICK_START.md) - Get started in 5 minutes
  - [Complete Documentation](docs/HISTORICAL_DATA_SYSTEM.md) - Comprehensive guide
  - [API Reference](docs/API_REFERENCE.md) - Detailed API documentation

## 💻 Usage Examples

### Basic Rust Example

```rust
use csf_core::{CSFRuntime, PhasePacket};
use csf_kernel::Task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize runtime
    let runtime = CSFRuntime::new(Default::default()).await?;
    
    // Create a task
    let task = Task::new("sensor_processing", |packet: PhasePacket| {
        // Process sensor data
        println!("Processing: {:?}", packet);
    });
    
    // Schedule task
    runtime.schedule(task).await?;
    
    // Start runtime
    runtime.start().await?;
    
    Ok(())
}
```

### Python Example

```python
import ares_csf

# Initialize runtime
runtime = ares_csf.CSFRuntime()

# Define callback
def process_packet(packet):
    print(f"Processing: {packet.amplitude} at {packet.frequency}Hz")

# Create and schedule task
task = runtime.create_task("sensor_task", process_packet)
runtime.schedule(task)

# Run
runtime.start()
```

### Historical Market Data Example

```rust
use csf_core::prelude::*;
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure historical data fetching
    let config = HistoricalDataConfig {
        data_source: DataSource::YahooFinance,
        symbols: vec!["AAPL".to_string(), "GOOGL".to_string()],
        start_date: Utc::now() - Duration::days(30),
        end_date: Utc::now(),
        interval: TimeInterval::Daily,
        playback_speed: 10.0, // 10x speed for validation
        max_retries: 3,
        rate_limit_ms: 1000,
    };

    // Fetch historical data
    let mut fetcher = HistoricalDataFetcher::new(config);
    let all_data = fetcher.fetch_all_data().await?;

    // Process data for model validation
    for (symbol, data) in all_data {
        println!("Validating model with {} points for {}", data.len(), symbol);
        
        // Stream data through quantum temporal correlation system
        fetcher.replay_data_with_timing(data, |stream_data| {
            // Process each data point as if it's arriving live
            let prediction = process_quantum_data(stream_data);
            println!("Prediction: ${:.2}", prediction.predicted_price);
        }).await?;
    }

    Ok(())
}
```

## 🛠️ Development

### Building from Source

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Build with specific features
cargo build --features "cuda python-bindings"

# Build documentation
cargo doc --open
```

### Running Tests

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run benchmarks
cargo bench

# Run stress tests
./scripts/stress-test.sh
```

### Project Structure

```
NovaCore/ChronoFabric/
├── crates/
│   ├── csf-core/       # Core types and runtime
│   ├── csf-kernel/     # Task scheduler and memory management
│   ├── csf-bus/        # Phase Coherence Bus implementation
│   ├── csf-clogic/     # C-LOGIC modules (DRPP, ADP, EGC, EMS)
│   ├── csf-mlir/       # MLIR runtime and backends
│   ├── csf-sil/        # Secure Immutable Ledger
│   ├── csf-ffi/        # Foreign Function Interface bindings
│   ├── csf-network/    # Network protocols and discovery
│   └── csf-telemetry/  # Metrics and tracing
├── examples/           # Usage examples
├── benches/           # Performance benchmarks
├── scripts/           # Build and deployment scripts
└── deployments/       # Docker and Kubernetes configs
```

## 🔌 Integration

NovaCore provides multiple integration options:

- **Rust Native**: Direct integration for Rust applications
- **C API**: For C/C++ applications via FFI
- **Python Bindings**: High-level Python API via PyO3
- **WebAssembly**: Run in browsers and edge environments
- **gRPC/REST**: Network API for distributed systems

## 📊 Performance

NovaCore is designed for extreme performance:

- **Throughput**: 1M+ messages/second on single node
- **Latency**: < 1μs for local message passing
- **Scalability**: Linear scaling to 1000+ nodes
- **Memory**: Zero-copy architecture minimizes allocations

See [benchmarks/](benches/) for detailed performance tests.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code of Conduct
- Development workflow
- Coding standards
- Pull request process

## 🔒 Security

NovaCore implements defense-in-depth security:

- Quantum-resistant cryptography
- Memory-safe Rust implementation
- Secure communication channels
- Audit logging and compliance

Report security issues to: security@novacore.io

## 📄 License

This project is dual-licensed:

- **MIT License** - See [LICENSE-MIT](LICENSE-MIT)
- **Apache License 2.0** - See [LICENSE-APACHE](LICENSE-APACHE)

You may choose either license for your use case.

## 🙏 Acknowledgments

NovaCore builds upon cutting-edge research in:

- Temporal computing and causality
- Neuromorphic architectures
- Distributed systems
- Quantum computing paradigms

Special thanks to the Rust community and all contributors.

---

<p align="center">
  Built with ❤️ for the future of computing
</p>

<p align="center">
  <a href="https://novacore.io">Website</a> •
  <a href="https://docs.novacore.io">Documentation</a> •
  <a href="https://github.com/Delfictus/NovaCore/issues">Issues</a> •
  <a href="https://discord.gg/novacore">Discord</a>
</p>