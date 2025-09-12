# Determinism

- Replace wall-clock with `csf-time::TimeSource` and `HlcClock`.
- Provide `deterministic` feature: fixed seeds, fixed tick, reproducible scheduling.
- Bound all queues and tasks. No unbounded channels.
- For randomness use `StdRng` seeded from config/test kit.
- Record seeds in test logs for replay.
