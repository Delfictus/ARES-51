# Production Dockerfile for Hephaestus Forge
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build release binary with optimizations
RUN cargo build --release -p hephaestus-forge

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 forge

# Copy binary from builder
COPY --from=builder /app/target/release/hephaestus-forge /usr/local/bin/

# Create data directories
RUN mkdir -p /var/log/hephaestus-forge \
    && mkdir -p /var/lib/hephaestus-forge \
    && chown -R forge:forge /var/log/hephaestus-forge \
    && chown -R forge:forge /var/lib/hephaestus-forge

# Switch to non-root user
USER forge

# Expose metrics and API ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the binary
ENTRYPOINT ["hephaestus-forge"]
CMD ["--config", "/etc/forge/config.toml"]