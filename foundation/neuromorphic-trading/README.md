# Neuromorphic-Deterministic Trading System Prototype

## Overview
Isolated prototype environment for developing the Hybrid Neuromorphic-Deterministic Trading System.

## Working Directory
```
/home/diddy/dev/ares-neuromorphic-prototype/
```

## Branch
`neuromorphic-trading-prototype`

## Structure
```
neuromorphic-trading/
├── src/
│   ├── lib.rs           # Main library entry
│   ├── neuromorphic/    # Neuromorphic components
│   ├── deterministic/   # Deterministic trading logic
│   ├── hybrid/          # Integration layer
│   └── metrics/         # Performance monitoring
├── tests/               # Integration tests
├── benches/             # Performance benchmarks
└── docs/                # Architecture documentation
```

## Development Commands
```bash
# Navigate to prototype
cd /home/diddy/dev/ares-neuromorphic-prototype

# Build prototype
cargo build -p neuromorphic-trading

# Run tests
cargo test -p neuromorphic-trading

# Run benchmarks
cargo bench -p neuromorphic-trading

# Switch back to main repo
cd /home/diddy/dev/ares-monorepo
```

## Git Workflow
```bash
# Make changes in prototype
cd /home/diddy/dev/ares-neuromorphic-prototype
# ... make changes ...
git add .
git commit -m "feat: implement X"

# When ready to merge
cd /home/diddy/dev/ares-monorepo
git merge neuromorphic-trading-prototype

# Clean up worktree when done
git worktree remove ../ares-neuromorphic-prototype
```