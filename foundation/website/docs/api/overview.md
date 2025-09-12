---
sidebar_position: 0
title: "API Overview"
description: "Complete API reference for ARES ChronoFabric system components"
---

# API Reference Overview

This section provides comprehensive API documentation for all ARES ChronoFabric system components.

## Core Components

### [csf-bus](./csf-bus.md)
Phase Coherence Bus API for high-performance message routing and event handling with sub-microsecond latency.

### [csf-time](./csf-time.md) 
Temporal Task Weaver API providing precise timing, Hybrid Logical Clocks, and deadline scheduling.

### [csf-network](./csf-network.md)
Distributed networking API with QUIC transport, mTLS security, and optional libp2p discovery.

### [csf-consensus](./csf-consensus.md)
PBFT consensus mechanism API for distributed agreement and fault tolerance.

### [csf-sil](./csf-sil.md)
Secure Immutable Ledger API for cryptographically verifiable audit trails and data integrity.

## Architecture Principles

All APIs follow these design principles:

- **Type Safety**: Strong typing with compile-time guarantees
- **Zero-Copy**: Efficient memory usage with `bytes::Bytes` and `Arc<T>`
- **Async-First**: Built on `tokio` for high-performance async I/O
- **Observability**: Integrated tracing spans for monitoring and debugging
- **Error Handling**: Comprehensive error types using `thiserror`
- **Backpressure**: Flow control to prevent resource exhaustion