---
sidebar_position: 1
title: "Security Overview"
description: "Security architecture and implementation for ARES ChronoFabric systems"
---

# Security

- mTLS via `rustls`. Key storage behind an abstraction; PIV/HSM ready.
- PQC hybrid mode optional (documented feature flag).
- All audit bundles from SIL are signed and versioned.
- `cargo-deny` and `cargo audit` run in CI; exceptions require justification.
- Secrets never in repo. Support environment and OS keychain backends.
- Threat model and mitigations in `THREAT_MODEL.md`.