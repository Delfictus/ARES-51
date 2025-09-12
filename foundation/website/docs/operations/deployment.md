---
sidebar_position: 1
title: "Deployment"
description: "Production deployment guide for ARES ChronoFabric systems"
---

# Deployment

This guide covers deploying ARES ChronoFabric in production environments.

## Deployment Architectures

### Single Node Deployment
For development and testing:

```bash
# Start with basic configuration
ares-chronofabric start --config production.toml
```

### Multi-Node Cluster
For production high-availability:

```yaml
# docker-compose.yml
version: '3.8'
services:
  chronofabric-node1:
    image: ares/chronofabric:latest
    environment:
      - NODE_ID=1
      - CLUSTER_PEERS=node2:8080,node3:8080
  chronofabric-node2:
    image: ares/chronofabric:latest
    environment:
      - NODE_ID=2
      - CLUSTER_PEERS=node1:8080,node3:8080
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chronofabric
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chronofabric
  template:
    spec:
      containers:
      - name: chronofabric
        image: ares/chronofabric:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Configuration Management

### Production Configuration
```toml
[bus]
queue_capacity = 4096
enable_quantum_optimization = true

[network]
bind_address = "0.0.0.0:8080"
quic_idle_timeout_ms = 30000
mtls = true

[telemetry]
prometheus_bind = "0.0.0.0:9464"
otel_endpoint = "http://jaeger:4317"
tracing_level = "warn"
```

## Health Checks

```bash
# HTTP health endpoint
curl http://localhost:8080/health

# CLI health check
ares-chronofabric health --timeout 5s
```

## Scaling Considerations

- **CPU**: 2+ cores per node, preferably with high frequency
- **Memory**: 4GB+ for production workloads
- **Network**: 10Gbps+ for high-throughput scenarios
- **Storage**: SSD with high IOPS for consensus and ledger

## Security Hardening

1. Enable mTLS for all inter-node communication
2. Use proper certificate management
3. Implement network segmentation
4. Regular security audits with `cargo audit`

See the [Security Guide](../security/overview.md) for complete security setup.