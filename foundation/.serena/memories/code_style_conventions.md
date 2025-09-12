# ARES ChronoFabric Code Style and Conventions

## General Formatting (.editorconfig)
- **Charset**: UTF-8
- **Line endings**: LF (Unix-style)
- **Final newline**: Always insert
- **Trim trailing whitespace**: Yes
- **Indent style**: Spaces (4 spaces default)

## Rust-Specific Style
- **Indent size**: 4 spaces
- **Max line length**: 100 characters
- **Edition**: 2021
- **Formatting**: Standard rustfmt (cargo fmt)
- **Linting**: Clippy with `-D warnings` (treat warnings as errors)

## File-Specific Conventions
- **TOML files**: 2 spaces indent
- **YAML files**: 2 spaces indent  
- **JSON files**: 2 spaces indent
- **Markdown**: 80 character line limit, preserve trailing whitespace
- **Shell scripts**: 2 spaces indent
- **MLIR files**: 2 spaces indent

## Code Quality Standards
- **Zero warnings**: All clippy warnings must be addressed
- **Format compliance**: Code must pass `cargo fmt --check`
- **Test coverage**: Use cargo tarpaulin for coverage reports
- **Security**: Regular cargo audit and cargo deny checks

## Naming Conventions
- **Crates**: kebab-case with `csf-` prefix (e.g., csf-core, csf-time)
- **Modules**: snake_case
- **Types**: PascalCase (e.g., PhaseCoherenceBus, TimeError)
- **Functions**: snake_case
- **Constants**: SCREAMING_SNAKE_CASE

## Dependencies Management
- **Workspace dependencies**: Centralized in root Cargo.toml
- **Version strategy**: Conservative, security-focused
- **Banned crates**: openssl (use rustls), failure (use thiserror/anyhow), lazy_static (use once_cell)
- **License compliance**: MIT, Apache-2.0, BSD variants allowed

## Testing Patterns
- **Integration tests**: Single-threaded (`--test-threads=1`)
- **Unit tests**: Per-crate in lib.rs and module files
- **Benchmarks**: Using criterion framework
- **Property testing**: Using proptest and quickcheck