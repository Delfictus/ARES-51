# API: csf-time

## Traits
```rust
pub trait TimeSource: Send + Sync {
    fn now(&self) -> std::time::Duration; // monotonic epoch
}

pub trait HlcClock: Send + Sync {
    fn now_hlc(&self) -> (u64, u64); // (physical, logical)
    fn merge(&self, remote: (u64, u64));
}

pub trait DeadlineScheduler {
    fn schedule<F>(&self, deadline: std::time::Instant, f: F) where F: FnOnce() + Send + 'static;
}
```
## Notes
- Deterministic mode replaces `Instant::now` with fixed tick.
