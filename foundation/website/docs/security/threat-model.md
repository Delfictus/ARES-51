---
sidebar_position: 2
title: "Threat Model"
description: "STRIDE-based threat analysis and security mitigations for ARES ChronoFabric"
---

# Threat Model (STRIDE)

## Scope
Bus, network, consensus, ledger, scheduler, missions, and telemetry.

## Assets
- Message integrity and ordering
- Ledger immutability and audit bundles
- Keys and identities
- Deterministic runs and seeds
- Telemetry endpoints

## Trust boundaries
- Node ↔ Node over QUIC
- App ↔ Bus
- Consensus ↔ SIL
- Telemetry exporters

## STRIDE table (abridged)
- **Spoofing**: mTLS, node identity pins, optional PQC hybrid.
- **Tampering**: Merkle ledger, signed audit bundles, PBFT.
- **Repudiation**: SIL checkpoints, append-only WAL, trace IDs.
- **Information Disclosure**: mTLS, least-privilege keys, role-scoped metrics.
- **DoS**: Backpressure, bounded queues, retry budgets, rate limits.
- **Elevation**: No runtime DI for hot paths, feature-gated adapters, code reviews.

## Residual risks
- Side channels via timing/telemetry. Mitigate by rate limiting and aggregation.
- Misconfiguration. Single config and validation on boot.