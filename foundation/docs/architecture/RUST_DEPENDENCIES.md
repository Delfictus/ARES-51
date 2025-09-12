# Rust Dependencies Overview

This document details all Rust crate dependencies used in the NovaCore project.

## Core Dependencies

### Async Runtime & Concurrency

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `tokio` | 1.35 | Async runtime | `full` - all features enabled |
| `async-trait` | 0.1 | Async traits support | - |
| `futures` | 0.3 | Async utilities | - |
| `crossbeam` | 0.8 | Lock-free data structures | - |
| `parking_lot` | 0.12 | Efficient synchronization | - |
| `dashmap` | 5.5 | Concurrent HashMap | - |
| `rayon` | 1.8 | Data parallelism | - |

### Serialization & Data

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `serde` | 1.0 | Serialization framework | `derive` |
| `serde_json` | 1.0 | JSON support | - |
| `bincode` | 1.3 | Binary serialization | - |
| `prost` | 0.12 | Protocol Buffers | - |
| `bytes` | 1.5 | Byte buffer utilities | - |
| `bytemuck` | 1.14 | Safe transmutation | `derive` |

### Networking

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `quinn` | 0.10 | QUIC protocol | - |
| `tonic` | 0.10 | gRPC framework | - |
| `tower` | 0.4 | Service abstractions | `full` |
| `hyper` | 1.1 | HTTP implementation | `full` |
| `axum` | 0.7 | Web framework | - |
| `tokio-tungstenite` | 0.21 | WebSocket support | - |

### Cryptography & Security

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `ring` | 0.17 | Crypto primitives | - |
| `ed25519-dalek` | 2.1 | Ed25519 signatures | - |
| `x25519-dalek` | 2.0 | X25519 key exchange | - |
| `blake3` | 1.5 | BLAKE3 hashing | - |
| `argon2` | 0.5 | Password hashing | - |
| `zeroize` | 1.7 | Secure memory clearing | `derive` |

### Error Handling & Logging

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `anyhow` | 1.0 | Error handling | - |
| `thiserror` | 1.0 | Error derive macros | - |
| `tracing` | 0.1 | Structured logging | - |
| `tracing-subscriber` | 0.3 | Log subscriber | `env-filter`, `fmt` |
| `log` | 0.4 | Logging facade | - |

### Metrics & Observability

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `prometheus` | 0.13 | Prometheus metrics | - |
| `opentelemetry` | 0.21 | Distributed tracing | `rt-tokio` |
| `opentelemetry-otlp` | 0.14 | OTLP exporter | - |
| `opentelemetry-prometheus` | 0.14 | Prometheus exporter | - |
| `tracing-opentelemetry` | 0.22 | Tracing integration | - |

## Hardware Acceleration

### MLIR & LLVM

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `mlir-sys` | 0.2 | MLIR C bindings | - |
| `llvm-sys` | 170 | LLVM C bindings | - |
| `inkwell` | 0.2 | Safe LLVM wrapper | `llvm17-0` |

### GPU Support

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `cust` | 0.3 | CUDA Rust bindings | - |
| `cuda-sys` | 0.2 | CUDA FFI | - |
| `vulkano` | 0.34 | Vulkan wrapper | - |
| `ash` | 0.37 | Raw Vulkan bindings | - |
| `wgpu` | 0.19 | WebGPU implementation | - |

## FFI & Language Bindings

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `pyo3` | 0.20 | Python bindings | `extension-module`, `abi3-py38` |
| `wasm-bindgen` | 0.2 | WASM bindings | - |
| `cbindgen` | 0.26 | C header generation | - |
| `cxx` | 1.0 | C++ interop | - |

## Development Dependencies

| Crate | Version | Purpose | Features |
|-------|---------|---------|----------|
| `criterion` | 0.5 | Benchmarking | `html_reports` |
| `proptest` | 1.4 | Property testing | - |
| `quickcheck` | 1.0 | Property testing | - |
| `insta` | 1.34 | Snapshot testing | - |
| `pretty_assertions` | 1.4 | Better test assertions | - |

## Platform-Specific Dependencies

### Linux

```toml
[target.'cfg(target_os = "linux")'.dependencies]
libc = "0.2"
nix = { version = "0.27", features = ["fs", "process", "signal"] }
inotify = "0.10"
```

### macOS

```toml
[target.'cfg(target_os = "macos")'.dependencies]
core-foundation = "0.9"
security-framework = "2.9"
```

### Windows

```toml
[target.'cfg(windows)'.dependencies]
windows = { version = "0.52", features = ["Win32_Foundation", "Win32_System"] }
winapi = { version = "0.3", features = ["winbase"] }
```

## Feature Flags

### Default Features

```toml
[features]
default = ["tokio-runtime", "metrics", "telemetry"]
```

### Optional Features

```toml
# Runtime options
tokio-runtime = ["tokio/full", "opentelemetry/rt-tokio"]
async-std-runtime = ["async-std", "opentelemetry/rt-async-std"]

# GPU backends
cuda = ["cust", "cuda-sys"]
vulkan = ["vulkano", "ash"]
webgpu = ["wgpu"]

# Language bindings
python-bindings = ["pyo3", "numpy"]
wasm-bindings = ["wasm-bindgen", "web-sys"]
c-bindings = ["cbindgen"]

# Hardware features
simd = ["packed_simd"]
neuromorphic = ["spiking-neural-networks"]

# Debugging
debug-tools = ["flame", "pprof", "heaptrack"]
```

## Version Policy

- **Major versions**: Update only for breaking changes
- **Minor versions**: Update for new features
- **Patch versions**: Update automatically for bug fixes

## Security Considerations

1. **Audit regularly**: Run `cargo audit` weekly
2. **Pin versions**: Use exact versions for production
3. **Review updates**: Check changelogs before updating
4. **Test thoroughly**: Run full test suite after updates

## License Compatibility

All dependencies are compatible with MIT/Apache-2.0 dual licensing:

- MIT License compatible: ✓
- Apache-2.0 compatible: ✓
- GPL dependencies: ✗ (None)

## Build Dependencies

These are only needed for building, not runtime:

```toml
[build-dependencies]
prost-build = "0.12"
tonic-build = "0.10"
cc = "1.0"
cmake = "0.1"
bindgen = "0.69"
```

## Benchmarking Dependencies

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
divan = "0.1"
iai = "0.1"
```

## Testing Dependencies

```toml
[dev-dependencies]
mockall = "0.12"
fake = "2.9"
arbitrary = "1.3"
test-case = "3.3"
serial_test = "3.0"
```