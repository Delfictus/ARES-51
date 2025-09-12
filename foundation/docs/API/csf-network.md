# API: csf-network

- QUIC via `quinn`
- mTLS via `rustls`
- Optional libp2p discovery

## Concepts
- Connection pooling
- Retry budgets
- Backpressure-aware send/recv

## Hooks
- Integrates with `csf-bus` for message ingress/egress.
