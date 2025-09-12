---
sidebar_position: 4
title: "CI/CD"
description: "Continuous integration and deployment pipeline configuration for ARES ChronoFabric"
---

# CI

Stages:
1. Format: `cargo fmt --all -- --check`
2. Lints: `cargo clippy --all-targets --all-features -D warnings -W clippy::pedantic`
3. Build + Test: `cargo test --workspace`
4. Fuzz smoke: run short `cargo-fuzz` jobs
5. Concurrency: Loom job
6. Miri job (allow to fail initially)
7. Security: `cargo-deny`, `cargo audit`
8. Coverage: upload report
9. SBOM + provenance: CycloneDX, SLSA

Fail fast. No green, no merge.