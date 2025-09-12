---
sidebar_position: 2
title: "Performance"
description: "Performance optimization guidelines and benchmarking for ARES ChronoFabric"
---

# Performance

- Use `bytes::Bytes` for zero-copy payloads.
- Preallocate buffers. Avoid per-message allocs.
- Backpressure throughout bus and network.
- QUIC pooling, retry budgets, tuned idle timeouts.
- Criterion benchmarks under `benches/`; store baselines.
- Profile: `cargo bench` + `perf`/`dhat-heap` as needed.