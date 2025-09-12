---
sidebar_position: 2
title: "ADR 0001: Single Infrastructure Components"
description: "Decision to use single bus, scheduler, and configuration system"
---

# ADR 0001: One bus, one scheduler, one config

- Status: Accepted
- Date: 2025-08-17

## Context
Duplicate infrastructure causes nondeterminism and audit gaps.

## Decision
- Single bus abstraction in `csf-bus` for all message flows.
- Single scheduler in `csf-time` with deadline + rate-monotonic policies.
- Single config crate in `csf-core::config` consumed by all crates.

## Consequences
- Simpler reasoning and testing.
- Fewer integration failures.
- Requires coordinated migrations and feature gating.