# API: csf-sil (Secure Immutable Ledger)

## Traits
```rust
pub trait SecureImmutableLedger {
    fn append(&self, bytes: bytes::Bytes) -> Result<AppendReceipt, SilError>;
    fn checkpoint(&self) -> Result<Checkpoint, SilError>;
    fn export_audit(&self, to: &mut dyn std::io::Write) -> Result<(), SilError>;
}
```
## Data structures
- Merkle accumulator over content hashes.
- Periodic checkpoints with signatures.
