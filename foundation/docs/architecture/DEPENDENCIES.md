# NovaCore Dependencies

This document provides a comprehensive list of all dependencies required to build, run, and develop NovaCore.

## Table of Contents

- [System Requirements](#system-requirements)
- [Build Dependencies](#build-dependencies)
- [Runtime Dependencies](#runtime-dependencies)
- [Development Dependencies](#development-dependencies)
- [Optional Dependencies](#optional-dependencies)
- [Rust Crate Dependencies](#rust-crate-dependencies)
- [Python Dependencies](#python-dependencies)
- [Container Dependencies](#container-dependencies)

## System Requirements

### Minimum Requirements

- **OS**: Linux (kernel 5.4+), macOS 11+, Windows 10+ with WSL2
- **CPU**: x86_64 or ARM64 with AVX2 support recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB free space for build artifacts
- **Network**: Internet connection for dependency downloads

### Supported Platforms

- Ubuntu 20.04+ (primary development platform)
- Debian 11+
- RHEL/CentOS 8+
- Fedora 35+
- macOS 11+ (Big Sur and later)
- Windows 10/11 with WSL2

## Build Dependencies

### Required

```bash
# Core build tools
rustc 1.75.0+      # Rust compiler (install via rustup)
cargo 1.75.0+      # Rust package manager
cmake 3.20+        # Build system for native dependencies
gcc 9.0+ / clang 11+  # C/C++ compiler
pkg-config         # Library configuration tool
git 2.25+          # Version control

# System libraries
libssl-dev         # OpenSSL development files
libclang-dev       # Clang development files
protobuf-compiler  # Protocol buffer compiler
```

### Installation Commands

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    protobuf-compiler \
    git \
    curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake pkg-config openssl protobuf

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Fedora/RHEL
```bash
sudo dnf install -y \
    gcc \
    gcc-c++ \
    cmake \
    openssl-devel \
    clang-devel \
    protobuf-compiler \
    pkg-config

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Runtime Dependencies

### Core Runtime

```bash
# System libraries
libc6 2.31+       # GNU C Library
libgcc1           # GCC support library
libstdc++6        # GNU Standard C++ Library
libssl1.1         # OpenSSL libraries
zlib1g            # Compression library
```

### Network Stack
```bash
# For QUIC support
libquiche         # Cloudflare QUIC implementation

# For mDNS discovery
avahi-daemon      # Service discovery
```

## Development Dependencies

### Recommended Tools

```bash
# Code analysis
rust-analyzer     # LSP server for IDE support
clippy           # Rust linter
rustfmt          # Rust code formatter
cargo-audit      # Security vulnerability scanner
cargo-tarpaulin  # Code coverage tool

# Performance analysis
perf             # Linux performance tools
valgrind         # Memory profiler
heaptrack        # Heap memory profiler
flamegraph       # Profiling visualization

# Documentation
mdbook           # Documentation generator
cargo-doc        # API documentation
```

### Installation
```bash
# Rust development tools
rustup component add rust-analyzer clippy rustfmt
cargo install cargo-audit cargo-tarpaulin flamegraph

# System tools (Ubuntu/Debian)
sudo apt install -y valgrind heaptrack linux-tools-generic
```

## Optional Dependencies

### GPU Acceleration

#### CUDA (NVIDIA GPUs)
```bash
# CUDA Toolkit 11.0+
cuda-toolkit-11-8     # CUDA compiler and libraries
cudnn8               # Deep learning primitives
libnccl2             # Multi-GPU communication

# Installation
# Follow NVIDIA's official guide: https://developer.nvidia.com/cuda-downloads
```

#### Vulkan (Cross-platform GPU)
```bash
vulkan-sdk          # Vulkan development files
mesa-vulkan-drivers # Open source Vulkan drivers
```

#### WebGPU (Web/Native GPU)
```bash
# Included via wgpu Rust crate, no system dependencies
```

### Machine Learning Acceleration

#### Intel MKL
```bash
intel-mkl          # Intel Math Kernel Library
# Or install via Intel oneAPI
```

#### OpenBLAS
```bash
libopenblas-dev    # Open source BLAS implementation
```

### Hardware-Specific

#### TPU Support
```bash
# Google Cloud TPU
libtpu              # TPU runtime library
# Requires Google Cloud SDK
```

#### Neuromorphic Hardware
```bash
# Intel Loihi
nxsdk               # Intel neuromorphic SDK
# Requires Intel DevCloud access

# BrainChip Akida
akida-sdk           # Akida development kit
```

## Rust Crate Dependencies

### Core Dependencies

```toml
# From workspace Cargo.toml
[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
prost = "0.12"

# Networking
quinn = "0.10"          # QUIC implementation
tonic = "0.10"          # gRPC
tower = "0.4"
hyper = "1.1"

# Cryptography
ring = "0.17"           # Crypto primitives
ed25519-dalek = "2.1"   # Digital signatures
blake3 = "1.5"          # Hashing

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging and metrics
tracing = "0.1"
tracing-subscriber = "0.3"
prometheus = "0.13"
opentelemetry = "0.21"

# Data structures
dashmap = "5.5"         # Concurrent hashmap
crossbeam = "0.8"       # Lock-free structures
parking_lot = "0.12"    # Efficient mutexes

# FFI
pyo3 = { version = "0.20", features = ["extension-module"] }
wasm-bindgen = "0.2"
cbindgen = "0.26"
```

### MLIR Dependencies

```toml
mlir-sys = "0.2"        # MLIR bindings
llvm-sys = "170"        # LLVM bindings
inkwell = "0.2"         # LLVM wrapper
```

### GPU Dependencies

```toml
# CUDA
cust = "0.3"            # CUDA Rust bindings
cuda-sys = "0.2"        # CUDA FFI

# Vulkan
vulkano = "0.34"        # Vulkan wrapper
ash = "0.37"            # Raw Vulkan bindings

# WebGPU
wgpu = "0.19"           # WebGPU implementation
```

## Python Dependencies

### Core Python Bindings

```txt
# requirements.txt for Python bindings
numpy>=1.24.0          # Numerical arrays
maturin>=1.4.0         # Build tool for PyO3
pytest>=7.0.0          # Testing framework
black>=23.0.0          # Code formatter
mypy>=1.0.0           # Type checker
```

### Installation
```bash
pip install -r crates/csf-ffi/requirements.txt
```

## Container Dependencies

### Docker Base Images

```dockerfile
# Production runtime
FROM rust:1.75-slim

# Development environment
FROM rust:1.75

# GPU-enabled
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
```

### Docker Dependencies

```yaml
# docker-compose.yml services
services:
  # Metrics storage
  prometheus:
    image: prom/prometheus:v2.45.0
  
  # Visualization
  grafana:
    image: grafana/grafana:10.0.0
  
  # Tracing
  jaeger:
    image: jaegertracing/all-in-one:1.47
  
  # Message queue (optional)
  nats:
    image: nats:2.10-alpine
```

## Version Matrix

| Component | Minimum Version | Recommended | Notes |
|-----------|----------------|-------------|-------|
| Rust | 1.75.0 | Latest stable | Required for async traits |
| LLVM | 15.0 | 17.0 | For MLIR support |
| CUDA | 11.0 | 11.8+ | For GPU acceleration |
| Python | 3.8 | 3.11 | For Python bindings |
| Node.js | 16.0 | 20.0 | For WASM builds |
| Docker | 20.10 | 24.0 | For containers |
| Kubernetes | 1.24 | 1.28 | For orchestration |

## Dependency Management

### Updating Dependencies

```bash
# Update Rust dependencies
cargo update

# Check for outdated dependencies
cargo outdated

# Security audit
cargo audit

# Update system dependencies (Ubuntu)
sudo apt update && sudo apt upgrade
```

### Lock Files

The project uses lock files to ensure reproducible builds:

- `Cargo.lock` - Rust dependencies
- `package-lock.json` - Node.js dependencies (for WASM)
- `requirements.lock` - Python dependencies

Always commit lock files to version control.

## Troubleshooting

### Common Issues

1. **OpenSSL not found**
   ```bash
   # Ubuntu/Debian
   sudo apt install libssl-dev pkg-config
   
   # macOS
   brew install openssl
   export OPENSSL_DIR=$(brew --prefix openssl)
   ```

2. **CUDA not detected**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. **Protobuf version mismatch**
   ```bash
   # Use specific version
   cargo clean
   PROTOC_VERSION=3.21.0 cargo build
   ```

## Support

For dependency-related issues:

1. Check the [Troubleshooting Guide](docs/troubleshooting.md)
2. Search [existing issues](https://github.com/Delfictus/NovaCore/issues)
3. Join our [Discord community](https://discord.gg/novacore)
4. Open a new issue with dependency details

---

Last updated: 2024-01-15