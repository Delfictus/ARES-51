# ARES CSF Examples

This directory contains example applications demonstrating various features and use cases of the ARES Chronosynclastic Fabric.

## Examples

### 1. Basic Sensor Fusion (`basic_sensor_fusion/`)
A simple example showing how to:
- Create multiple sensor components
- Fuse sensor data for improved accuracy
- Apply basic safety monitoring

**Run with:**
```bash
cd basic_sensor_fusion
cargo run --bin sensor-fusion
```

### 2. Real-Time Control (`real_time_control/`)
Demonstrates real-time control loops with:
- Deadline-aware task scheduling
- Causality tracking
- EGC safety verification

**Run with:**
```bash
cd real_time_control
cargo run --bin rt-control
```

### 3. Distributed Consensus (`distributed_consensus/`)
Shows distributed CSF deployment with:
- Multi-node communication
- Consensus protocols
- Fault tolerance

**Run with:**
```bash
cd distributed_consensus
cargo run --bin consensus-node -- --node-id node1
```

## Building Examples

All examples can be built from the workspace root:

```bash
# Build all examples
cargo build --examples

# Build specific example
cargo build --example sensor-fusion

# Build with optimizations
cargo build --release --examples
```

## Running with Features

Some examples support additional features:

```bash
# With CUDA support
cargo run --example sensor-fusion --features cuda

# With neuromorphic support
cargo run --example rt-control --features neuromorphic
```

## Creating Your Own Example

To create a new example:

1. Create a new directory: `mkdir my_example`
2. Add a `Cargo.toml` file:
   ```toml
   [package]
   name = "my-example"
   version = "0.1.0"
   edition = "2021"
   
   [[bin]]
   name = "my-example"
   path = "main.rs"
   
   [dependencies]
   chronofabric = { path = "../.." }
   # Add other dependencies
   ```
3. Create your `main.rs` file
4. Add to workspace members in root `Cargo.toml`

## Common Patterns

### Creating Phase Packets
```rust
use csf_core::prelude::*;

let packet = PhasePacket::new(payload, ComponentId::DRPP)
    .with_priority(Priority::High)
    .with_deadline(deadline_ns);
```

### Subscribing to Events
```rust
// In real implementation
bus.subscribe::<SensorData>(|packet| {
    println!("Received: {:?}", packet);
    Ok(())
}).await?;
```

### Implementing C-LOGIC Components
```rust
struct MyComponent {
    bus: Arc<PhaseCoherenceBus>,
}

impl MyComponent {
    async fn process(&self, input: PhasePacket<Input>) -> Result<()> {
        // Process input
        let output = self.compute(input.payload)?;
        
        // Publish result
        self.bus.publish(output, metadata).await?;
        Ok(())
    }
}
```

## Debugging

Enable debug logging:
```bash
RUST_LOG=debug cargo run --example sensor-fusion
```

Enable backtrace:
```bash
RUST_BACKTRACE=1 cargo run --example sensor-fusion
```

## Performance Testing

Run examples with performance monitoring:
```bash
# With built-in metrics
cargo run --release --example sensor-fusion -- --enable-metrics

# With external profiler
perf record cargo run --release --example sensor-fusion
perf report
```