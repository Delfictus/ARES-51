# Task Completion Workflow for ARES ChronoFabric

## Essential Commands to Run After Code Changes

### 1. Formatting Check
```bash
cargo fmt --all -- --check
```
If this fails, run:
```bash
cargo fmt --all
```

### 2. Linting
```bash
cargo clippy --all-targets --all-features -- -D warnings
```
Must pass with zero warnings.

### 3. Testing
```bash
# Basic tests
cargo test --workspace

# Integration tests (single-threaded)
cargo test --workspace -- --test-threads=1
```

### 4. Security Validation
```bash
# Security audit
cargo audit

# Dependency compliance
cargo deny check
```

### 5. Combined Quality Check
```bash
# Using Make
make check

# Using Just (recommended)
just check

# Manual sequence
cargo fmt --all -- --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo test --workspace
```

## Build Verification
```bash
# Debug build
cargo build --workspace

# Release build (for performance-critical changes)
cargo build --release --workspace
```

## Specialized Validation Scripts
For security-critical changes, also run:
```bash
./scripts/safety_validation.sh
./scripts/comprehensive_security_test.sh
./scripts/validate_mlir_security.sh
```

## Git Workflow
- **Never commit** without passing all quality checks
- **Author attribution**: Ensure commits are attributed to Ididia Serfaty
- **Commit messages**: Follow conventional commit format
- **Hooks**: Automated git hooks remove AI references and ensure proper attribution

## Performance Validation
For performance-sensitive changes:
```bash
just bench              # Run benchmarks
cargo criterion --all   # Detailed criterion benchmarks
```

## Documentation
After significant changes:
```bash
cargo doc --no-deps --all-features --open
```