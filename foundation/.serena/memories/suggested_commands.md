# ARES ChronoFabric Development Commands

## Core Development Commands

### Building
```bash
# Debug build
cargo build --workspace

# Release build
cargo build --release --workspace

# Build specific crate
cargo build -p csf-core
```

### Testing
```bash
# Run all tests
cargo test --workspace

# Run tests with single thread (for integration tests)
cargo test --workspace -- --test-threads=1

# Run specific crate tests
cargo test -p csf-core
cargo test -p csf-time

# Run integration tests
cargo test --test '*' -- --test-threads=1

# Test with coverage
cargo tarpaulin --out Html --output-dir coverage
```

### Code Quality
```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Lint code
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit
cargo deny check

# Combined quality check
make check  # or just check (runs fmt-check, lint, test)
```

### Running
```bash
# Development run
cargo run -- --config config/dev.toml
RUST_LOG=debug cargo run

# Release run  
cargo run --release -- --config config/production.toml

# Run examples
cargo run --example basic_sensor_fusion
cargo run --example historical_data_validation
```

### Benchmarking
```bash
# Run benchmarks
cargo bench --all
cargo criterion --all

# Performance profiling
just flame
```

### Using Just (Recommended)
```bash
just            # List available commands
just build      # Build project
just test       # Run tests
just check      # Run all quality checks
just run        # Development run
just lint       # Run clippy
just fmt        # Format code
just bench      # Run benchmarks
```

### Docker
```bash
# Build Docker image
docker build -t ares-csf:latest .

# Run with docker-compose
docker-compose up -d
docker-compose down
```