# Architecture Review

This document assesses the repository's compliance with the specified Hexagonal Architecture guardrails.

## Overall Assessment

The repository shows a clear *intent* to follow a hexagonal architecture. The separation of concerns into different crates (`csf-core`, `csf-clogic`, and various adapter crates like `csf-network`, `csf-bus`, etc.) is a strong foundation. However, the implementation is incomplete, and critical components for enforcing the architecture are missing.

--- 

### 1. Ports Definition (`csf-core`)

- **Status:** ✅ **Compliant**
- **Evidence:** The `csf-core` crate correctly defines the primary port traits.
- **Details:** The file `crates/csf-core/src/ports.rs` contains the following required ports:
  - `TimeSource`
  - `HlcClock`
  - `DeadlineScheduler`
  - `Consensus`
  - `SecureImmutableLedger`
  - `EventBusTx`
  - `EventBusRx`

--- 

### 2. Domain Logic Isolation (`csf-clogic`)

- **Status:** ✅ **Compliant**
- **Evidence:** The domain logic crate `csf-clogic` does not appear to have direct dependencies on adapter crates.
- **Details:** A `grep` scan for imports of `csf_bus`, `csf_network`, `csf_sil`, `csf_telemetry`, or `csf_hardware` within `crates/csf-clogic/src` returned no results. This indicates that the domain logic is correctly decoupled from specific adapter implementations.

--- 

### 3. Adapter Crate Structure

- **Status:** 🟡 **Partially Compliant**
- **Evidence:** Crates for adapters exist (`csf-bus`, `csf-network`, etc.), but their implementation and enforcement are incomplete.
- **Details:** The project structure includes crates that are clearly intended to be adapters. However, without the orchestrator (`csf-runtime`), there is no mechanism to ensure that these crates *only* contain adapters and that they are used correctly.

--- 

### 4. Orchestration & One-of Selection (`csf-runtime`)

- **Status:** ❌ **Non-Compliant (CRITICAL)**
- **Evidence:** The `csf-runtime` crate, which is designated as the orchestrator, is missing from the workspace.
- **Details:** The core component responsible for selecting and assembling the production adapters for each port is not implemented. The `crates/` directory does not contain `csf-runtime`. As a result, there are no `compile_error!` checks to enforce that exactly one adapter per port is selected at build time. This is a major architectural gap that undermines the entire hexagonal model.

--- 

## Summary of Architectural Gaps

1.  **Missing Orchestrator (P0):** The lack of a `csf-runtime` crate means there is no central point for application assembly and no enforcement of the one-adapter-per-port rule.
2.  **Build Failures (P0):** The project does not currently build, which prevents any further architectural validation. The build failures seem to stem from a recent refactoring of `csf-core` that was not propagated to dependent crates.
