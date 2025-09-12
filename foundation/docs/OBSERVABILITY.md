# Observability

## Tracing
- Use `tracing` spans at crate boundaries and critical paths.
- Include correlation IDs from `Envelope`.

## Metrics
Prometheus counters/histograms. Naming:
- `ares_bus_envelopes_total{direction="tx|rx", crate=""}`
- `ares_net_conn_attempts_total{result="ok|err"}`
- `ares_consensus_view_changes_total`
- `ares_sil_append_latency_seconds`
- `ares_scheduler_deadline_misses_total`

## OTEL
- Export traces and metrics via OTLP. Config in `CONFIGURATION.md`.

## tokio-console
- Optional per-feature flag. Disabled in release builds by default.
