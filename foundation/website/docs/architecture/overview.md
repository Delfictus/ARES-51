---
sidebar_position: 2
title: "Architecture Overview"
description: "High-level system architecture overview for ARES ChronoFabric"
---

# Architecture

## Crate graph
- `csf-core`: domain types, error enums, config, feature flags.
- `csf-bus`: `Envelope`, `EventBusTx/Rx` over a single bus abstraction.
- `csf-time`: `TimeSource`, `HlcClock`, `DeadlineScheduler`.
- `csf-network`: QUIC (`quinn`) transport, `rustls` mTLS, optional libp2p discovery. Retry budgets, pooling, backpressure.
- `csf-consensus`: PBFT baseline with pluggable engines; persistence hooks to SIL.
- `csf-sil`: Secure Immutable Ledger with Merkle accumulator, checkpoints, audit export.
- `ares-missions/*`: CEW, Swarm, Neuromorphic, Digital Twin, Optical Stealth, Cyber‑EM, Backscatter, Countermeasures, Orchestrator, Federated Learning.
- `ares-testkit`: deterministic integration sims across bus/net/consensus/ledger.

## Boundaries and flows
- All inter-module messages go through the **one** `csf-bus`.
- Time comes only from `csf-time::TimeSource`; no direct `Instant::now()` in logic.
- Consensus writes to SIL; SIL emits audit bundles.
- Network uses connection pooling and backpressure; no unbounded channels.
- Observability via `tracing` on every boundary; Prometheus + OTEL exporters.

## Invariants
- No panics on hot paths. No `unwrap`/`expect` in library crates.
- Unsafe code isolated in `unsafe/` with docs and tests.
- Deterministic mode for tests and sims.