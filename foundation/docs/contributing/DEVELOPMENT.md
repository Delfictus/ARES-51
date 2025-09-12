# ARES CSF Development Guide

## Quick Start

1. **Initialize the project:**
   ```bash
   ./init-project.sh
   ```

2. **Set up development environment:**
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Build and test:**
   ```bash
   make check  # Format, lint, and test
   make build  # Build debug binary
   ```

4. **Run development server:**
   ```bash
   make run
   # or with just:
   just watch
   ```

## Project Structure

```
ares-csf/
├── crates/              # Workspace crates
│   ├── csf-core/       # Core types and traits
│   ├── csf-kernel/     # Chronos kernel & TTW scheduler
│   ├── csf-bus/        # Phase Coherence Bus
│   ├── csf-clogic/     # C-LOGIC modules (DRPP, ADP, EGC, EMS)
│   ├── csf-mlir/       # MLIR runtime integration
│   ├── csf-sil/        # Secure Immutable Ledger
│   ├── csf-ffi/        # Foreign function interface
│   ├── csf-network/    # Network protocol
│   └── csf-telemetry/  # Metrics and tracing
├── examples/            # Example applications
├── tests/              # Integration tests
├── benches/            # Performance benchmarks
├── scripts/            # Build and deployment scripts
├── config/             # Configuration files
├── deployments/        # Kubernetes and Docker configs
├── monitoring/         # Prometheus and Grafana configs
└── docs/               # Documentation

```

## Development Workflow

### 1. Local Development

**VS Code:**
```bash
code .  # Opens with all configured settings
```

**GitHub Codespaces:**
- Click "Code" → "Codespaces" → "Create codespace on main"
- Full development environment in the cloud

### 2. Testing

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# Specific crate
cargo test -p csf-core

# With coverage
just test-coverage
```

### 3. Benchmarking

```bash
# Run all benchmarks
cargo bench

# Compare with baseline
just bench-compare baseline-name

# Stress testing
./scripts/stress-test.sh all
```

### 4. Code Quality

```bash
# Format code
cargo fmt --all

# Lint with clippy
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Check everything
make check
```

## Docker Development

### Build and Run

```bash
# Build image
docker build -t ares-csf:dev .

# Run single node
docker run -p 8080:8080 -p 9090:9090 ares-csf:dev

# Run cluster
docker-compose up -d

# View logs
docker-compose logs -f csf-node-1
```

### Development Container

```bash
# Start dev container
docker-compose run --rm csf-dev bash

# Inside container
cargo build
cargo test
```

## Kubernetes Development

### Local Testing (kind/minikube)

```bash
# Create cluster
kind create cluster --name csf-test

# Deploy
kubectl apply -k deployments/kubernetes/overlays/development

# Check status
kubectl get pods -n ares-csf-dev

# View logs
kubectl logs -f deployment/ares-csf -n ares-csf-dev
```

### Production Deployment

```bash
# Deploy to production
kubectl apply -k deployments/kubernetes/overlays/production

# Or with Helm
helm install ares-csf deployments/helm/ares-csf \
  --namespace ares-csf \
  --create-namespace
```

## Monitoring

### Local Monitoring Stack

```bash
# Start monitoring
docker-compose --profile monitoring up -d

# Access:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9093
# - Jaeger: http://localhost:16686
```

### Key Metrics

- `csf_packet_latency_seconds`: Packet processing latency
- `csf_tasks_scheduled_total`: Number of tasks scheduled
- `csf_deadline_misses_total`: Real-time deadline violations
- `csf_drpp_coherence`: DRPP pattern recognition coherence

## CI/CD

### GitHub Actions

The project includes comprehensive CI/CD:
- **CI**: Format, lint, test, security audit
- **Performance**: Automated benchmarking
- **Release**: Multi-platform builds
- **Security**: Dependency scanning, SBOM generation

### Local CI Testing

```bash
# Install act
brew install act  # or your package manager

# Run CI locally
act -j test
```

## Common Tasks

### Adding a New Crate

1. Create crate structure:
   ```bash
   cargo new --lib crates/csf-newfeature
   ```

2. Add to workspace:
   ```toml
   # Cargo.toml
   members = [
     # ...
     "crates/csf-newfeature",
   ]
   ```

3. Add dependencies:
   ```toml
   # crates/csf-newfeature/Cargo.toml
   [dependencies]
   chronofabric = { path = "../.." }
   ```

### Running Specific Components

```bash
# Just DRPP
cargo run --bin drpp-standalone

# Just the bus
cargo run --bin bus-server

# With specific config
cargo run -- --config config/custom.toml
```

### Performance Profiling

```bash
# CPU flame graph
cargo flamegraph --bin ares-csf

# Memory profiling
valgrind --tool=massif target/release/ares-csf
ms_print massif.out.*

# GPU profiling (NVIDIA)
nsys profile target/release/ares-csf
```

## Troubleshooting

### Build Issues

```bash
# Clean build
cargo clean
rm -rf target/

# Update dependencies
cargo update

# Check for conflicts
cargo tree --duplicates
```

### Runtime Issues

```bash
# Enable debug logging
RUST_LOG=debug,ares_csf=trace cargo run

# Enable backtrace
RUST_BACKTRACE=full cargo run

# Check system limits
ulimit -a
```

### Performance Issues

```bash
# Profile with perf
perf record -g target/release/ares-csf
perf report

# Check allocations
DHAT_HEAP=1 cargo run --features dhat-heap
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Run checks: `make check`
5. Commit: `git commit -m 'feat(component): add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Create Pull Request

## Resources

- [Architecture Documentation](docs/architecture/README.md)
- [API Reference](https://docs.ares-csf.io)
- [Discord Community](https://discord.gg/ares-csf)
- [Performance Tuning Guide](docs/performance/README.md)

## Getting Help

- **Issues**: https://github.com/ares-systems/chronofabric/issues
- **Discussions**: https://github.com/ares-systems/chronofabric/discussions
- **Email**: dev@ares-systems.io
- **Security**: security@ares-systems.io