# Copilot Instructions for ARES CSF (ChronoSynclastic Fabric)

## Project Overview

ARES CSF is a real-time computing platform implementing **hexagonal architecture** with the following key components:

- **csf-core**: Defines ports (traits) and types for hexagonal architecture boundaries
- **csf-clogic**: Domain logic for cognitive computing modules (DRPP, ADP, EGC, EMS)  
- **csf-bus**: Phase Coherence Bus for zero-copy message passing
- **csf-kernel**: Temporal Task Weaver with causality-aware scheduling
- **csf-telemetry**: OpenTelemetry-based metrics and distributed tracing
- **csf-network**: Multi-protocol network layer (QUIC/TCP/WebSocket)
- **csf-sil**: Secure Immutable Ledger with cryptographic audit trails

## Critical Architecture Patterns

### Hexagonal Architecture Enforcement
```rust
// Always use ports from csf-core, never direct adapter dependencies
use csf_core::ports::{EventBusTx, DeadlineScheduler, TimeSource};

// Domain logic (csf-clogic) MUST NOT import adapters
// ❌ Never: use csf_bus::MpscBus; 
// ✅ Always: use csf_core::ports::EventBusTx;
```

### Phase Packet Communication
```rust
// The central data flow primitive
use csf_core::prelude::*;

let packet = PhasePacket::new(payload, ComponentId::DRPP)
    .with_priority(Priority::High)
    .with_deadline(deadline_ns);
```

### Cross-Module Communication (C-LOGIC)
```rust
// Modules communicate through named bus channels
self.bus.create_channel("drpp.patterns", "adp.input").await?;
self.bus.create_channel("ems.modulation", "drpp.modulation").await?;
```

## Essential Build Commands

```bash
# Core development workflow
cargo check -p csf-core      # Always start here - validates ports
cargo check --workspace      # Full workspace compilation check
just check                   # Format + lint + test pipeline

# Feature-based building (critical for adapters)
cargo build --features "bus-mpsc,net-quic"    # Adapter selection
cargo test -p csf-telemetry --features nvidia-gpu

# Development tools
cargo watch -x 'check -p csf-core'
just flame chronofabric      # Performance profiling
cargo tarpaulin --out Html   # Coverage analysis
```

## Project-Specific Conventions

### Error Handling Pattern
```rust
// Unified error types in each crate
pub type Result<T> = std::result::Result<T, Error>;

// csf-core::Error for ports, crate-specific for implementations
use csf_core::Error;           // In domain logic
use csf_telemetry::TelemetryError;  // In adapters
```

### API Stability Levels
- **csf-core**: Stable API - minimal changes, full documentation required
- **csf-clogic**: Domain logic - stable interfaces, evolving implementations  
- **Adapter crates**: Implementation details - can change with port compliance

### Workspace Dependencies
```toml
# Always use workspace = true for consistency
[dependencies]
tokio = { workspace = true, features = ["rt", "sync"] }
csf-core = { path = "../csf-core", features = ["net"] }
```

## Critical Integration Points

### Telemetry Integration
```rust
// Every module must instrument with OpenTelemetry
use opentelemetry::trace::Span as _;
use csf_telemetry::{telemetry, start_span};

let mut span = start_span("operation_name")?;
span.set_attribute("component".to_string(), "drpp".into());
```

### Bus Registration Pattern
```rust
// Standard subscription pattern for all modules
let handle = bus.subscribe::<DataType>(move |packet| {
    let _span = start_span("process_packet")?;
    // Process packet...
    Ok(())
}).await?;
```

### Testing Conventions
```rust
// Use proptest for property-based testing in csf-core
#[cfg(test)]
use proptest::prelude::*;

// Async test pattern with tokio
#[tokio::test]
async fn test_feature() -> Result<()> {
    // Setup...
}
```

## Common Debugging Workflows

```bash
# Debug telemetry issues
RUST_LOG=csf_telemetry=debug,opentelemetry=debug cargo run

# Profile async runtime
tokio-console &
TOKIO_CONSOLE_BIND=127.0.0.1:6669 cargo run --features tokio-console

# Network debugging  
RUST_LOG=csf_network=trace,quinn=debug cargo run

# Memory debugging
RUST_BACKTRACE=full cargo run
```

## Integration Dependencies

### sysinfo API Migration
Current version 0.31.4 removed `SystemExt` trait:
```rust
// ❌ Old: use sysinfo::{System, SystemExt};
// ✅ New: use sysinfo::System;
// Note: disk/network methods need API updates
```

### Feature Flag Management
- Use `#[cfg(feature = "...")]` for optional hardware support
- Adapter selection enforced at compile-time (csf-runtime missing)
- GPU features: `nvidia-gpu`, `cuda`, neuromorphic hardware

## Development Environment

```bash
# Complete setup
./scripts/setup-dev.sh        # Installs tools, git hooks
just setup                    # Alternative with just
docker-compose run csf-dev    # Containerized development

# Pre-commit validation
just pre-commit               # Full check pipeline
cargo deny check              # Security audit
cargo machete                 # Unused dependency detection
```

When working on this codebase, always validate changes against the hexagonal architecture constraints and ensure cross-module communication follows the established Phase Packet patterns.
