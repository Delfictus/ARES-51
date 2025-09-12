# Continue Agent: ARES/ChronoFabric Rules

1) Mission
- Rewrite and harden the ARES system in a Rust monorepo with ChronoFabric.
- Enforce one bus, one scheduler, one config, one tracing/metrics stack.
- Improve determinism, auditability, and safety.

2) Scope of edits
- Prefer minimal diffs. Preserve compile green at each step.
- Touch one crate or boundary per change unless a single diff is safer.
- Never edit generated files or lockfiles by hand.

3) Code standards
- Rust 2021, MSRV pinned. `cargo fmt`, `clippy -W clippy::pedantic` clean.
- No `unwrap`/`expect` in library code. Use `thiserror` + `anyhow` at edges.
- No panics on hot paths. Preallocate. Use `bytes::Bytes` for payloads.
- Unsafe isolated behind `unsafe/` modules with docs and tests.

4) Architecture
- Core traits: implement `csf-time::{TimeSource,HlcClock,DeadlineScheduler}`, `csf-consensus::Consensus`, `csf-sil::SecureImmutableLedger`, `csf-bus::{EventBusTx,EventBusRx,Envelope}` first.
- Networking: QUIC via `quinn`, mTLS via `rustls`; optional libp2p discovery.
- Consensus: PBFT baseline. Pluggable backends.
- Ledger: Merkle accumulator, checkpoints, audit export.
- Scheduler: deadline + rate-monotonic; deterministic test mode; `no_std` leaves if possible.

5) Observability
- `tracing` spans at every boundary. No `println!`.
- Prometheus counters/histograms. OTEL exporter. `tokio-console` optional.

6) Security
- mTLS, key storage abstraction, PQC hybrid option.
- Sign audit bundles. Deny insecure deps: `cargo-deny`, `cargo audit`.

7) Testing and CI
- Unit + property tests (`proptest`). Fuzz (`cargo-fuzz`).
- Concurrency checks (Loom). UB checks (Miri).
- Deterministic integration sims in `ares-testkit`.
- CI: fmt, clippy pedantic, tests, fuzz smoke, coverage, SBOM, provenance.

8) Repository hygiene
- One config crate. One scheduler. One bus. No duplicate infra.
- Features gate adapters. No runtime DI on hot paths.
- Pin versions. Document feature matrices.

9) Work method
- Always: search repo → open files → reason → propose patch diff.
- Run `cargo check -q` in the edited crate and at workspace root after each diff.
- If a change breaks build, self-revert or include the fix in the same diff.
- When APIs change, update all call sites in the same commit.

10) Determinism
- Provide deterministic modes and seeds. Avoid time.now in logic; use `TimeSource`.
- For async, bound tasks. No unbounded channels. Backpressure everywhere.

11) Docs
- Public items have rustdoc. Module‐level README per crate.
- Record invariants and safety notes where `unsafe` appears.

12) Non-goals and refusals
- Do not invent protocols, keys, or legal text.
- Do not remove tests to make builds pass.
- Do not introduce global singletons beyond the one config.

13) Commit style
- Conventional commits. Scope = crate. Body lists invariants touched and tests added.

14) When unsure
- Prefer reading code and proposing a minimal diff over asking questions.
- If ambiguity would risk data loss or security, stop and request a clear decision point.

# Commands

/walk [path|crate] [--workspace] [--deps] [--public-only] [--todo]
/fix [path|crate] [--limit=N] [--include=GLOB] [--exclude=GLOB] [--keep-green]
/wire [boundary] [--impl=TRAIT] [--crate=NAME] [--feature=NAME]
/harden [target] [--panic=deny] [--unwrap=forbid] [--errors]
/observe [target] [--spans] [--metrics] [--otel] [--prom] [--console]
/determinize [target] [--seed=U64] [--time=TimeSource] [--bound=N] [--rng=StdRng]
/scheduler [crate=csf-time] [--deadline] [--rm] [--deterministic]
/net [crate=csf-network] [--quic] [--mtls] [--retry] [--pool] [--backpressure]
/consensus [crate=csf-consensus] [--pbft] [--pluggable]
/sil [crate=csf-sil] [--merkle] [--checkpoint] [--audit-export]
/testkit [scenario] [--deterministic] [--seed=U64] [--nodes=N]
/bench [crate] [--crit] [--profile=dev|release] [--baseline=ID]
/fuzz [crate] [--target=PATH] [--add]
/loom [crate] [--target=MOD]
/miri [crate]
/deny [workspace]
/sbom [workspace] [--provenance]
/ci [--github] [--coverage] [--cache]
/docs [path|crate] [--public] [--readmes] [--invariants]
