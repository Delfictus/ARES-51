---Continue agent — ARES/ChronoFabric commands

/walk [path|crate] [--workspace] [--deps] [--public-only] [--todo]
Scan tree, map crates, list public APIs, tech debt, and build graph.

/fix [path|crate] [--limit=N] [--include=GLOB] [--exclude=GLOB] [--keep-green]
Make the code compile green with minimal diffs. Update all call sites.

/wire [boundary] [--impl=TRAIT] [--crate=NAME] [--feature=NAME]
Wire traits across bus/time/consensus/ledger. Enforce one bus/scheduler/config.

/harden [target] [--panic=deny] [--unwrap=forbid] [--errors]
Remove unwrap/expect, add thiserror types, anyhow only at edges.

/observe [target] [--spans] [--metrics] [--otel] [--prom] [--console]
Add tracing spans, Prometheus histograms/counters, OTEL exporter.

/determinize [target] [--seed=U64] [--time=TimeSource] [--bound=N] [--rng=StdRng]
Replace realtime with TimeSource, bound tasks/channels, make tests deterministic.

/scheduler [crate=csf-time] [--deadline] [--rm] [--deterministic]
Implement TimeSource, HLC, DeadlineScheduler. Add tests and docs.

/net [crate=csf-network] [--quic] [--mtls] [--retry] [--pool] [--backpressure]
Implement QUIC (quinn), rustls mTLS, retry budgets, connection pooling.

/consensus [crate=csf-consensus] [--pbft] [--pluggable]
PBFT baseline with pluggable engines. Define messages, state, persistence hooks.

/sil [crate=csf-sil] [--merkle] [--checkpoint] [--audit-export]
SecureImmutableLedger with Merkle accumulator, checkpoints, audit export.

/testkit [scenario] [--deterministic] [--seed=U64] [--nodes=N]
Deterministic integration sims over bus/net/consensus/ledger.

/bench [crate] [--crit] [--profile=dev|release] [--baseline=ID]
Add/run Criterion benches and store baselines.

/fuzz [crate] [--target=PATH] [--add]
Set up cargo-fuzz, add harness, seed corpus. Wire CI smoke job.

/loom [crate] [--target=MOD]
Add Loom concurrency tests for schedulers, queues, and PBFT transitions.

/miri [crate]
Run Miri for UB checks; gate in CI as optional job.

/deny [workspace]
Add cargo-deny config and audit exceptions with justification.

/sbom [workspace] [--provenance]
Generate CycloneDX SBOM and SLSA provenance; add CI steps.

/ci [--github] [--coverage] [--cache]
CI with fmt, clippy pedantic, tests, fuzz smoke, coverage, SBOM, audit.

/docs [path|crate] [--public] [--readmes] [--invariants]
Write rustdoc and crate READMEs. Document invariants, safety, feature matrix.

/lint [path|crate]
description: A description of your rule
---

Your rule content
