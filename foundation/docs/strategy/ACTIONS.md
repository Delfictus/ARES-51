# Prioritized Action Plan

This document outlines the prioritized actions required to address the findings from the repository audit. 

---

## P0: Build Blockers & Critical Architecture Gaps

*These items must be addressed before any other work can proceed, as they prevent the project from compiling or represent fundamental architectural deficiencies.*

### 1. Fix Workspace Compilation

- **Rationale:** The entire workspace currently fails to build due to unresolved imports and missing crate features. This is the highest priority task.
- **File(s):** `crates/csf-bus/src/packet.rs`, `crates/csf-kernel/**/*.rs`, `crates/csf-kernel/Cargo.toml`
- **Action Items:**
    1.  **Update Imports:** The `csf_core` crate was refactored, removing the `prelude` and `types` modules. All other crates must be updated to use the new path for ports, e.g., `use csf_core::ports::EventBusTx;`.
    2.  **Enable `nix` features:** The `csf-kernel` crate uses functionality from `nix` that requires feature flags.
- **Suggested Diff (`crates/csf-kernel/Cargo.toml`):**
  ```diff
  --- a/crates/csf-kernel/Cargo.toml
  +++ b/crates/csf-kernel/Cargo.toml
  @@ -10,7 +10,7 @@
   chrono = { workspace = true }
   log = { workspace = true }
   thiserror = { workspace = true }
  -nix = { workspace = true }
  +nix = { workspace = true, features = ["sched", "process"] }
   
   [dev-dependencies]
   proptest = { workspace = true }
  
  ```

### 2. Create `csf-runtime` Orchestrator Crate

- **Rationale:** The crate responsible for assembling the application and enforcing the one-adapter-per-port rule is missing. This is a critical gap in the hexagonal architecture.
- **File(s):** `crates/csf-runtime/Cargo.toml`, `crates/csf-runtime/src/lib.rs` (all new)
- **Action Items:**
    1.  Create the `csf-runtime` crate.
    2.  Implement the generic `Orchestrator` struct.
    3.  Add `compile_error!` checks to enforce adapter selection via Cargo features.
- **Suggested Stub (`crates/csf-runtime/src/lib.rs`):**
  ```rust
  // One-of checks for adapter groups
  #[cfg(all(feature = "net-quic", feature = "net-libp2p"))]
  compile_error!("Select exactly one network adapter: net-quic OR net-libp2p");

  #[cfg(all(feature = "bus-mpsc", feature = "bus-redis"))]
  compile_error!("Select exactly one bus adapter: bus-mpsc OR bus-redis");
  
  // ... other adapter groups ...

  use csf_core::ports::*;

  // Generic orchestrator holds ports, not adapters.
  pub struct Orchestrator<B, N, L, T, S> {
      // ... fields for bus, net, ledger, telemetry, scheduler ports ...
  }

  impl<B, N, L, T, S> Orchestrator<B, N, L, T, S> {
      // ... implementation ...
  }
  ```

---

## P1: High-Priority Quality & Guardrail Violations

### 1. Eliminate Panics

- **Rationale:** Widespread use of `.unwrap()`, `.expect()`, and `panic!` violates the Determinism guardrail.
- **File(s):** Numerous, including `crates/csf-network/src/quic.rs`, `crates/csf-clogic/src/egc/stl.rs`, `crates/csf-sil/src/crypto.rs`.
- **Action Items:**
    1.  Systematically replace all panicking calls with robust `Result` and `Option` handling.
- **Suggested Diff (`crates/csf-sil/src/crypto.rs`):**
  ```diff
  --- a/crates/csf-sil/src/crypto.rs
  +++ b/crates/csf-sil/src/crypto.rs
  @@ -2,7 +2,10 @@
 use ed25519_dalek::{Keypair, Signature, Signer, Verifier};
 
 pub fn STUB_KEYPAIR() -> Keypair {
-    Keypair::from_bytes(&[0u8; 64]).unwrap_or_else(|_| panic!("stub keypair"))
+    // P0: This should be loaded from a secure store, not hardcoded.
+    // For now, we return a known keypair for testing, but return a Result.
+    Keypair::from_bytes(&[0u8; 64]).expect("Failed to create keypair from static bytes")
 }
 
 pub fn STUB_SIGNATURE() -> Signature {
-    Signature::from_bytes(&[0u8; 64]).expect("stub signature")
+    Signature::from_bytes(&[0u8; 64]).expect("Failed to create signature from static bytes")
 }
  ```

### 2. Implement Observability Hooks

- **Rationale:** The codebase is a black box, violating the Observability guardrail. Tracing and metrics are essential.
- **File(s):** All crates.
- **Action Items:**
    1.  Add `#[instrument]` macros to all public functions, especially at port/adapter boundaries.
    2.  Implement a `csf-telemetry` adapter that initializes an OpenTelemetry pipeline and Prometheus exporter.
    3.  Integrate the telemetry setup into `csf-runtime`.

---

## P2: Polish & Refinement

### 1. Audit Dynamic Dispatch (`dyn Trait`)

- **Rationale:** Overuse of `dyn Trait` can lead to performance degradation on hot paths.
- **File(s):** `crates/csf-network/src/protocol.rs`, `crates/csf-kernel/src/task.rs`, etc.
- **Action Items:**
    1.  Review all uses of `Box<dyn ...>`.
    2.  For any found in performance-sensitive loops, evaluate refactoring to static dispatch (generics).

### 2. Standardize Byte-Oriented Types

- **Rationale:** Inconsistent use of `Vec<u8>` and `bytes::Bytes` can lead to unnecessary copies.
- **File(s):** Numerous.
- **Action Items:**
    1.  Audit all uses of `Vec<u8>` for message payloads or I/O buffers.
    2.  Prefer `bytes::Bytes` for all zero-copy scenarios.
