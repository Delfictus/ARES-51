---
sidebar_position: 2
title: "Quick Start"
description: "Get up and running with ARES ChronoFabric in minutes"
---

# Quick Start Guide

This guide will help you get ARES ChronoFabric up and running in just a few minutes.

## Prerequisites

- ARES ChronoFabric installed (see [Installation Guide](./installation.md))
- Basic familiarity with Rust (optional, for development)

## 1. Create Your First System

```bash
# Create a new project directory
mkdir my-chronofabric-app
cd my-chronofabric-app

# Initialize configuration
ares-chronofabric init --template basic
```

This creates a basic configuration file `chronofabric.toml`:

```toml
[bus]
queue_capacity = 1024

[network]
bind_address = "127.0.0.1:8080"
quic_idle_timeout_ms = 30000

[time]
deterministic = false
tick_ms = 1

[telemetry]
prometheus_bind = "0.0.0.0:9464"
tracing_level = "info"
```

## 2. Start the System

```bash
# Start with the default configuration
ares-chronofabric start

# Or with custom config
ares-chronofabric start --config chronofabric.toml
```

You should see output similar to:

```
[INFO] ChronoFabric v0.1.0 starting...
[INFO] Phase Coherence Bus initialized (capacity: 1024)
[INFO] Temporal Task Weaver active (quantum optimization: enabled)
[INFO] System ready - listening on 127.0.0.1:8080
```

## 3. Send Your First Message

Open a new terminal and use the CLI to send a test message:

```bash
# Send a simple message
ares-chronofabric send --topic "hello" --data "Hello ChronoFabric!"

# Send with priority
ares-chronofabric send --topic "urgent" --data "Critical message" --priority high

# Send with deadline
ares-chronofabric send --topic "timed" --data "Time-sensitive data" --deadline 1s
```

## 4. Monitor System Performance

```bash
# Check system status
ares-chronofabric status

# View performance metrics
ares-chronofabric metrics

# Watch real-time performance
ares-chronofabric monitor --follow
```

Expected output:
```
System Status: ✓ Healthy
Latency: 347ns avg, 1.2μs peak
Throughput: 1.3M msg/sec
Memory: 45MB used
Temporal Coherence: 99.97%
```

## 5. Basic Rust Integration

Create a simple Rust application that uses ChronoFabric:

```rust
use ares_chronofabric::{PhaseCoherenceBus, PhasePacket, ComponentId};
use tokio;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MyMessage {
    content: String,
    timestamp: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the system
    let bus = PhaseCoherenceBus::new(Default::default())?;
    
    // Create a message
    let message = MyMessage {
        content: "Hello from Rust!".to_string(),
        timestamp: ares_chronofabric::time::now()?.as_nanos(),
    };
    
    // Send the message
    let packet = PhasePacket::new(message, ComponentId::custom(1));
    let message_id = bus.publish(packet).await?;
    
    println!("Message sent with ID: {}", message_id);
    Ok(())
}
```

Add to your `Cargo.toml`:
```toml
[dependencies]
ares-chronofabric = "0.1.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## 6. Performance Testing

Test the system's performance capabilities:

```bash
# Run built-in benchmarks
ares-chronofabric bench --duration 30s

# Test latency specifically
ares-chronofabric bench --test latency --target 1us

# Test throughput
ares-chronofabric bench --test throughput --target 1000000
```

## Next Steps

Now that you have ChronoFabric running:

1. **Learn the Architecture**: Read the [System Architecture](./architecture/overview.md)
2. **Configure for Production**: See [Configuration Guide](./configuration.md)
3. **Integrate with Your App**: Check out [Integration Examples](./guides/integration-examples.md)
4. **Monitor Performance**: Set up [Observability](./operations/observability.md)
5. **Optimize Performance**: Follow the [Performance Guide](./operations/performance.md)

## Common Issues

### Port Already in Use
```bash
# Change the port in configuration
ares-chronofabric start --bind 127.0.0.1:8081
```

### Permission Denied
```bash
# Run with appropriate permissions for hardware timing
sudo ares-chronofabric start --enable-hardware-timing
```

### High Latency
```bash
# Enable quantum optimization
ares-chronofabric start --quantum-optimization aggressive
```

For more troubleshooting help, see the [Performance Troubleshooting Guide](./guides/performance-troubleshooting.md).