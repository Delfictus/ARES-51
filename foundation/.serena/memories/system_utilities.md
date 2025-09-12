# System Utilities and Environment (Linux)

## Available Build Tools
- **just**: Modern command runner (preferred) - `just --list` to see available commands
- **make**: Traditional make system - `make help` for targets
- **cargo**: Rust package manager and build tool

## Development Scripts
Located in `scripts/` directory:
- `setup-dev.sh`: Development environment setup
- `stress-test.sh`: Performance stress testing
- `safety_validation.sh`: Safety and security validation
- `comprehensive_security_test.sh`: Comprehensive security testing
- `validate_mlir_security.sh`: MLIR-specific security validation
- `enterprise_security_validation.sh`: Enterprise security checks
- `check-protocol-compliance.sh`: Protocol compliance validation

## Docker and Deployment
- **Docker**: `docker-compose.yml` for local development
- **Kubernetes**: Deployments in `deployments/kubernetes/`
- **Monitoring**: Grafana, Prometheus, Jaeger stack

## Development Tools Available
- **Code Quality**: cargo fmt, cargo clippy, cargo audit, cargo deny
- **Testing**: cargo test, cargo tarpaulin (coverage), criterion (benchmarks)
- **Documentation**: cargo doc, mdbook
- **Performance**: cargo flamegraph, perf integration
- **WebAssembly**: wasm-pack for browser bindings

## Git Configuration
- **Author**: Ididia Serfaty (ididiaserfaty@protonmail.com)
- **Hooks**: Automated removal of AI references from commits
- **Workflow**: Standard git flow with quality gate enforcement

## System Requirements
- **Rust**: 1.75+ via rustup
- **Build tools**: cmake 3.20+, gcc 9+/clang 11+
- **Libraries**: OpenSSL, protobuf, libclang
- **Optional**: CUDA 11.0+, Docker, kubectl