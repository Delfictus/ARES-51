# Contributing to ARES CSF

Thank you for your interest in contributing to ARES Chronosynclastic Fabric! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chronofabric.git
   cd chronofabric
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ares-systems/chronofabric.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

### Setting Up Development Environment

1. Install prerequisites (see [README.md](README.md))
2. Run the setup script:
   ```bash
   ./scripts/setup-dev.sh
   ```
3. Build the project:
   ```bash
   cargo build --all
   ```
4. Run tests:
   ```bash
   cargo test --all
   ```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test additions or fixes
- `perf/` - Performance improvements

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `build`: Build system changes
- `ci`: CI configuration changes
- `chore`: Other changes

Example:
```
feat(drpp): implement adaptive coupling matrix updates

Add dynamic adjustment of coupling strengths based on
transfer entropy measurements. This improves pattern
recognition accuracy by 15%.

Closes #123
```

## Coding Standards

### Rust Style Guide

1. **Format code** with rustfmt:
   ```bash
   cargo fmt --all
   ```

2. **Lint code** with clippy:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```

3. **Documentation**:
   - All public APIs must have doc comments
   - Include examples in doc comments where appropriate
   - Run `cargo doc --no-deps --open` to preview

4. **Error Handling**:
   - Use `Result<T, Error>` for fallible operations
   - Implement custom error types with `thiserror`
   - Avoid `unwrap()` except in tests

5. **Performance**:
   - Profile before optimizing
   - Document performance-critical sections
   - Prefer zero-copy operations
   - Use const generics where appropriate

### Architecture Guidelines

1. **Separation of Concerns**:
   - Keep modules focused and cohesive
   - Use traits for abstraction
   - Minimize dependencies between crates

2. **Concurrency**:
   - Prefer message passing over shared state
   - Use appropriate synchronization primitives
   - Document thread safety requirements

3. **Real-time Constraints**:
   - Avoid allocations in hot paths
   - Use bounded data structures
   - Profile and benchmark critical paths

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions and modules
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_phase_packet_creation() {
           // Test implementation
       }
   }
   ```

2. **Integration Tests**: Test component interactions
   ```rust
   // tests/integration_test.rs
   #[test]
   fn test_bus_communication() {
       // Test implementation
   }
   ```

3. **Property Tests**: Use proptest for invariant testing
   ```rust
   proptest! {
       #[test]
       fn test_causality_ordering(tasks: Vec<Task>) {
           // Property test implementation
       }
   }
   ```

4. **Benchmarks**: Performance regression tests
   ```rust
   use criterion::{black_box, criterion_group, criterion_main, Criterion};
   
   fn bench_packet_processing(c: &mut Criterion) {
       c.bench_function("packet_processing", |b| {
           b.iter(|| process_packet(black_box(&packet)))
       });
   }
   ```

### Test Coverage

- Aim for >80% code coverage
- 100% coverage for safety-critical components (EGC, TTW)
- Run coverage with:
  ```bash
  cargo tarpaulin --out Html
  ```

## Submitting Changes

1. **Update your feature branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**:
   ```bash
   cargo test --all
   cargo test --all --release
   cargo bench --no-run
   ```

3. **Update documentation** if needed

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Use the PR template
   - Reference related issues
   - Include benchmark results for performance changes
   - Add screenshots for UI changes

## Review Process

### PR Requirements

- [ ] All tests pass
- [ ] Code is formatted and linted
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] Performance impact assessed

### Review Timeline

- Initial review: Within 48 hours
- Follow-up reviews: Within 24 hours
- Merge decision: After 2 approvals

### Review Checklist

Reviewers will check:

1. **Correctness**: Does the code do what it claims?
2. **Design**: Is the approach sound?
3. **Performance**: Are there any regressions?
4. **Security**: Are there any vulnerabilities?
5. **Documentation**: Is it adequate?
6. **Tests**: Are they comprehensive?

## Additional Resources

- [Architecture Guide](docs/architecture/README.md)
- [API Documentation](https://docs.ares-csf.io)
- [Discord Community](https://discord.gg/ares-csf)
- [Developer Blog](https://blog.ares-systems.io)

## Questions?

- Technical questions: dev@ares-systems.io
- Security concerns: security@ares-systems.io
- General inquiries: info@ares-systems.io

Thank you for contributing to ARES CSF!