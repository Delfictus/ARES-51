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
