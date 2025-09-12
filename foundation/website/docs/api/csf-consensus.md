---
sidebar_position: 2
title: "csf-consensus API"
description: "PBFT consensus mechanism API for distributed agreement"
---

# API: csf-consensus (PBFT baseline)

## Traits
```rust
pub trait Consensus {
    fn propose(&self, payload: bytes::Bytes) -> Result<(), ConsensusError>;
    fn on_message(&self, msg: PbftMessage) -> Result<(), ConsensusError>;
}
```
## Persistence hooks
- Append decisions to SIL with Merkle root updates.