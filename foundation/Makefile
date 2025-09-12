.PHONY: help build test clean run docker lint fmt check release

# Default target
help:
	@echo "ARES CSF Development Makefile"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  make build      - Build debug binary"
	@echo "  make release    - Build release binary"
	@echo "  make test       - Run all tests"
	@echo "  make bench      - Run benchmarks"
	@echo "  make lint       - Run clippy linter"
	@echo "  make fmt        - Format code"
	@echo "  make check      - Run all checks (fmt, lint, test)"
	@echo "  make run        - Run development server"
	@echo "  make docker     - Build Docker image"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make docs       - Generate documentation"
	@echo "  make setup      - Setup development environment"

# Build targets
build:
	cargo build --all

release:
	cargo build --release --all

# Test targets
test:
	cargo test --all -- --test-threads=1

test-verbose:
	cargo test --all -- --nocapture --test-threads=1

integration-test:
	cargo test --test '*' -- --test-threads=1

# Benchmark
bench:
	cargo bench --all

# Code quality
lint:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

check: fmt-check lint test

# Run targets
run:
	RUST_LOG=debug cargo run -- --config config/dev.toml

run-release:
	cargo run --release -- --config config/production.toml

# Docker targets
docker:
	docker build -t ares-csf:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Documentation
docs:
	cargo doc --no-deps --all-features --open

# Clean
clean:
	cargo clean
	rm -rf target/
	rm -rf dist/

# Setup
setup:
	./scripts/setup-dev.sh

# Release management
dist: release
	./scripts/build-release.sh

# Kubernetes
k8s-deploy:
	kubectl apply -k deployments/kubernetes/base/

k8s-delete:
	kubectl delete -k deployments/kubernetes/base/

# Monitoring
monitoring-up:
	docker-compose --profile monitoring up -d

monitoring-down:
	docker-compose --profile monitoring down

# Security
security-audit:
	cargo audit
	cargo deny check

# Performance
perf-record:
	cargo build --release
	perf record --call-graph=dwarf target/release/ares-csf --config config/bench.toml

perf-report:
	perf report

# CI simulation
ci-local:
	act -j test