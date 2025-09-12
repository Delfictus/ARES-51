---
sidebar_position: 3
title: "Testing"
description: "Comprehensive testing strategy for ARES ChronoFabric system reliability"
---

# Testing

## Unit
- `cargo test --workspace` with `#[forbid(unsafe_code)]` where possible.

## Property tests
- `proptest` for bus ordering, scheduler deadlines, PBFT transitions.

## Fuzz
- `cargo-fuzz`. Harnesses for envelope parsing, consensus messages, ledger proofs.

## Concurrency
- Loom tests for schedulers, channels, PBFT state transitions.

## UB checks
- Miri on core crates.

## Deterministic sims
- `ares-testkit` runs seeded, time-sourced scenarios. No real time or randomness.

## Coverage
- Include coverage job in CI; gate on floor value.