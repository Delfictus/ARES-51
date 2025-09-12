# Multi-stage build for ARES CSF

# Stage 1: Build environment
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    clang \
    llvm-dev \
    libclang-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/ares-csf

# Copy workspace files
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ ./crates/

# Build dependencies first (for better caching)
RUN mkdir -p crates/csf-core/src && \
    echo "fn main() {}" > crates/csf-core/src/main.rs && \
    cargo build --release && \
    rm -rf crates/*/src

# Copy actual source code
COPY . .

# Build release binary
RUN cargo build --release --features default

# Stage 2: Runtime environment
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libprotobuf32 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -u 1000 -U ares

# Create necessary directories
RUN mkdir -p /opt/ares-csf /etc/ares-csf /var/log/ares-csf /var/lib/ares-csf && \
    chown -R ares:ares /opt/ares-csf /etc/ares-csf /var/log/ares-csf /var/lib/ares-csf

# Copy binary from builder
COPY --from=builder /usr/src/ares-csf/target/release/chronofabric /opt/ares-csf/

# Copy configuration
COPY --from=builder /usr/src/ares-csf/config/default.toml /etc/ares-csf/config.toml

# Set up environment
ENV RUST_LOG=info
ENV CSF_CONFIG_PATH=/etc/ares-csf/config.toml

# Switch to non-root user
USER ares

# Expose ports
EXPOSE 8080 9000 9090 4317

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/opt/ares-csf/chronofabric", "--health-check"]

# Run the binary
ENTRYPOINT ["/opt/ares-csf/chronofabric"]
CMD ["--config", "/etc/ares-csf/config.toml"]

# Stage 3: Development environment (optional)
FROM builder AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust tools
RUN rustup component add rustfmt clippy rust-analyzer

WORKDIR /workspace

# Keep container running for development
CMD ["/bin/bash"]

# Stage 4: CUDA-enabled runtime (optional)
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04 AS cuda-runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libprotobuf23 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -u 1000 -U ares

# Create necessary directories
RUN mkdir -p /opt/ares-csf /etc/ares-csf /var/log/ares-csf /var/lib/ares-csf && \
    chown -R ares:ares /opt/ares-csf /etc/ares-csf /var/log/ares-csf /var/lib/ares-csf

# Copy binary from builder (must be built with cuda feature)
COPY --from=builder /usr/src/ares-csf/target/release/chronofabric /opt/ares-csf/

# Copy configuration
COPY --from=builder /usr/src/ares-csf/config/default.toml /etc/ares-csf/config.toml

# Set up environment
ENV RUST_LOG=info
ENV CSF_CONFIG_PATH=/etc/ares-csf/config.toml
ENV CUDA_VISIBLE_DEVICES=all

# Switch to non-root user
USER ares

# Expose ports
EXPOSE 8080 9000 9090 4317

# Run the binary
ENTRYPOINT ["/opt/ares-csf/chronofabric"]
CMD ["--config", "/etc/ares-csf/config.toml", "--features", "cuda"]