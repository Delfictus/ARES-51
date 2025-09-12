# ARES + ChronoFabric — Documentation

Production-grade Rust rewrite with determinism, auditability, and safety. One bus. One scheduler. One config. One tracing/metrics stack.

## 🎯 START HERE: Strategic Roadmap

**[STRATEGIC_ROADMAP.md](STRATEGIC_ROADMAP.md)** - The definitive project roadmap with:
- 22-week implementation timeline with 7 development phases  
- 22 specific milestones with acceptance criteria
- Resource requirements and risk analysis
- Success metrics and performance targets

This roadmap is the **SINGLE SOURCE OF TRUTH** for project direction and should be referenced for all development decisions.

## Quick map
- `ARCHITECTURE.md`: crates and boundaries
- `CONFIGURATION.md`: single config and features
- `OBSERVABILITY.md`: tracing, metrics, OTEL
- `SECURITY.md`: crypto, mTLS, PQC, secrets
- `THREAT_MODEL.md`: STRIDE model and mitigations
- `TESTING.md`: unit, property, fuzz, Loom, Miri, sims
- `DETERMINISM.md`: time, scheduler, seeds, bounded async
- `PERFORMANCE.md`: memory, backpressure, benchmarks
- `CI.md`: pipeline gates and coverage
- `CONTRIBUTING.md` + `CODING_STANDARDS.md`
- `API/`: trait surfaces per crate
- `ADR/`: decisions; 0001 enforces one-bus/one-scheduler/one-config
- `continue_agent.md`: agent rules and slash-commands

## Build
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -D warnings -W clippy::pedantic
cargo test --workspace
```

## Run deterministic sims
See `TESTING.md` and `DETERMINISM.md`.
