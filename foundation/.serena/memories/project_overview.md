# ARES ChronoFabric Project Overview

## Purpose
ARES ChronoFabric is a sophisticated quantum temporal correlation system built in Rust. It's a next-generation real-time computing platform implementing the ARES Chronosynclastic Fabric (CSF) architecture that unifies temporal task management, distributed processing, and neuromorphic computing paradigms.

## Key Technologies
- **Temporal Task Weaver (TTW)**: Causality-aware scheduling with predictive temporal analysis
- **Phase Coherence Bus (PCB)**: Zero-copy, lock-free message passing
- **C-LOGIC Integration**: Advanced cognitive computing modules (DRPP, ADP, EGC, EMS)
- **MLIR Runtime**: Multi-backend hardware acceleration (CPU, CUDA, Vulkan, WebGPU, TPU)
- **Secure Immutable Ledger (SIL)**: Cryptographic audit trail
- **Historical Market Data System**: Multi-source data fetching and streaming replay

## Tech Stack
- **Language**: Rust 1.75+
- **Async Runtime**: Tokio with full features
- **Serialization**: Serde, bincode, zerocopy
- **Cryptography**: blake3, ed25519-dalek, x25519-dalek, ring
- **Storage**: sled, rocksdb
- **ML/Math**: nalgebra, ndarray, candle (0.9.x)
- **Time**: chrono with clock features
- **Metrics**: prometheus, sysinfo
- **Testing**: criterion, proptest, quickcheck, approx

## Workspace Structure
- **csf-core**: Core tensor operations and quantum calculations
- **csf-time**: High-precision temporal operations  
- **csf-kernel**: Task scheduler and memory management
- **csf-bus**: Phase Coherence Bus implementation
- **csf-clogic**: C-LOGIC modules (DRPP, ADP, EGC, EMS)
- **csf-mlir**: MLIR runtime and backends
- **csf-sil**: Secure Immutable Ledger
- **csf-ffi**: Foreign Function Interface bindings
- **csf-network**: Network protocols and discovery
- **csf-telemetry**: Metrics and tracing
- **csf-shared-types**: Common types to break circular dependencies
- **ares-neuromorphic-cli**: Command-line interface
- **hephaestus-forge**: Integration system
- **ares-trading**: Trading system components