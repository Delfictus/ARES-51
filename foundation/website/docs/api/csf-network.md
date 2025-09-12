---
sidebar_position: 3
title: "csf-network API"
description: "Distributed networking API with QUIC transport and mTLS security"
---

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