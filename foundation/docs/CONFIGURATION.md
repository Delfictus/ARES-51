# Configuration

Single config crate (`csf-core::config`) consumed by all others. No duplicate configs.

## Example (TOML)
```toml
[bus]
queue_capacity = 1024

[network]
quic_idle_timeout_ms = 30000
retry_budget = "3x exponential"
pool_max = 64
mtls = true
pqc_hybrid = true

[consensus]
engine = "pbft"
view_timeout_ms = 2000

[scheduler]
policy = "deadline"
deterministic = true
tick_ms = 5

[telemetry]
prometheus_bind = "0.0.0.0:9464"
otel_endpoint = "http://localhost:4317"
tracing_level = "info"
```

## Features
- `net-libp2p`: enable peer discovery.
- `pqc`: enable PQC hybrid handshakes.
- `deterministic`: force seeded RNGs and fixed time source.
