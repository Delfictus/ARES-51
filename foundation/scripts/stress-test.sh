#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_ROOT/target/release/chronofabric"

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    log_error "Binary not found at $BINARY"
    log_info "Please build with: cargo build --release"
    exit 1
fi

log_info "Starting ChronoFabric stress test..."
log_info "Binary: $BINARY"

# Run stress test
"$BINARY" --config config/stress-test.toml --log-level debug