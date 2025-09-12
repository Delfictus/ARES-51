---
sidebar_position: 1
title: "csf-bus API"
description: "Phase Coherence Bus API reference for message routing and event handling"
---

# API: csf-bus

## Types
```rust
use bytes::Bytes;

pub struct Envelope {
    pub id: u128,
    pub topic: String,
    pub payload: Bytes,
    pub span: tracing::Span,
}

pub trait EventBusTx {
    fn publish(&self, env: Envelope) -> Result<(), BusError>;
}

pub trait EventBusRx {
    fn subscribe(&self, topic: &str) -> Result<Subscriber, BusError>;
}
```
## Notes
- Bounded queues. Backpressure on `publish`.