---
sidebar_position: 1
title: "Installation"
description: "Installation guide for ARES ChronoFabric system components"
---

# Installation

## System Requirements

### Hardware Requirements
- **CPU**: x86_64 with TSC support, minimum 4 cores
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: SSD with > 1000 IOPS
- **Network**: 1Gbps+ for distributed deployments

### Software Requirements
- **Rust**: 1.70+ (latest stable recommended)
- **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+, or equivalent)
- **Dependencies**: See [Dependencies Reference](./reference/dependencies.md)

## Installation Methods

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-org/ares-chronofabric.git
cd ares-chronofabric

# Build all components
cargo build --release --workspace

# Run tests to verify installation
cargo test --workspace

# Install system-wide (optional)
cargo install --path .
```

### Binary Release

```bash
# Download latest release
wget https://github.com/your-org/ares-chronofabric/releases/latest/download/ares-chronofabric-linux-x64.tar.gz

# Extract and install
tar -xzf ares-chronofabric-linux-x64.tar.gz
sudo mv ares-chronofabric /usr/local/bin/
```

### Docker Container

```bash
# Run with Docker
docker run -it areschronofabric/chronofabric:latest

# Or with docker-compose
wget https://raw.githubusercontent.com/your-org/ares-chronofabric/main/docker-compose.yml
docker-compose up -d
```

## Verification

After installation, verify the system is working correctly:

```bash
# Check version
ares-chronofabric --version

# Run basic system test
ares-chronofabric test --quick

# Start minimal system
ares-chronofabric start --config examples/minimal.toml
```

## Next Steps

- [Quick Start Guide](./quick-start.md)
- [Configuration](./configuration.md)
- [System Architecture Overview](./architecture/overview.md)