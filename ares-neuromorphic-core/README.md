# ARES Neuromorphic Core

High-performance neuromorphic prediction engine for trading systems.

[![Crates.io](https://img.shields.io/crates/v/ares-neuromorphic-core.svg)](https://crates.io/crates/ares-neuromorphic-core)
[![Documentation](https://docs.rs/ares-neuromorphic-core/badge.svg)](https://docs.rs/ares-neuromorphic-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ARES Neuromorphic Core provides the core neuromorphic computing capabilities for the ARES trading system, including spiking neural networks, reservoir computing, and signal processing algorithms optimized for financial market prediction.

## Features

- 🧠 **Spike Encoding**: Convert market data to spike trains using various encoding schemes
- 🌊 **Reservoir Computing**: Liquid State Machines for temporal pattern recognition  
- 📊 **Signal Processing**: Advanced neuromorphic signal processing algorithms
- 🔍 **Pattern Detection**: Real-time pattern recognition in market data
- ⚡ **High Performance**: Optimized for low-latency financial applications
- 🔧 **Configurable**: Flexible configuration for different use cases

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ares-neuromorphic-core = "0.1.0"
```

### Basic Usage

```rust
use ares_neuromorphic_core::{NeuromorphicEngine, EngineConfig, MarketData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine configuration
    let config = EngineConfig {
        spike_neurons: 10000,
        reservoir_size: 5000,
        encoding_window_ms: 1000.0,
        ..Default::default()
    };

    // Initialize the neuromorphic engine
    let mut engine = NeuromorphicEngine::new(config)?;

    // Process market data
    let market_data = MarketData {
        price: 50000.0,
        volume: 1.5,
        timestamp: chrono::Utc::now(),
        symbol: "BTC-USD".to_string(),
        metrics: None,
    };

    let prediction = engine.process(market_data)?;
    println!("Prediction: {:?}", prediction.direction);
    println!("Confidence: {:.2}", prediction.confidence);

    Ok(())
}
```

### Async Usage

```rust
use ares_neuromorphic_core::{AsyncNeuromorphicEngine, EngineConfig, MarketData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let engine = AsyncNeuromorphicEngine::new(config)?;

    let market_data = MarketData::new("ETH-USD".to_string(), 3000.0, 2.1);
    let prediction = engine.process(market_data).await?;
    
    println!("Async prediction: {:?}", prediction);
    Ok(())
}
```

## Architecture

The neuromorphic engine follows a multi-stage processing pipeline:

```
Market Data → Spike Encoding → Reservoir Computing → Pattern Detection → Prediction
```

### Components

1. **Spike Encoding** (`spike` module)
   - Rate coding
   - Temporal coding  
   - Population coding
   - Phase coding

2. **Reservoir Computing** (`reservoir` module)
   - Liquid State Machines
   - Echo State Networks
   - Temporal dynamics
   - Memory capacity

3. **Pattern Detection** (`signals` module)
   - Template matching
   - Temporal analysis
   - Signal generation
   - Confidence estimation

4. **Core Engine** (`core` module)
   - Main processing pipeline
   - Configuration management
   - Statistics tracking
   - Error handling

## Configuration

### Engine Configuration

```rust
use ares_neuromorphic_core::EngineConfig;

let config = EngineConfig {
    spike_neurons: 10000,        // Number of encoding neurons
    reservoir_size: 5000,        // Reservoir network size
    encoding_window_ms: 1000.0,  // Time window for encoding
    spectral_radius: 0.95,       // Reservoir spectral radius
    connection_prob: 0.1,        // Connection probability
    leak_rate: 0.3,              // Neuron leak rate
    confidence_threshold: 0.6,   // Minimum prediction confidence
    enable_caching: true,        // Enable pattern caching
    max_cache_size: 10000,       // Maximum cache entries
};
```

### Encoding Methods

```rust
use ares_neuromorphic_core::spike::{SpikeEncoder, EncodingMethod};

let mut encoder = SpikeEncoder::new(1000, 1000.0)?
    .with_encoding_method(EncodingMethod::Population);
```

## Examples

### Custom Market Data

```rust
use ares_neuromorphic_core::MarketData;
use std::collections::HashMap;

let mut custom_metrics = HashMap::new();
custom_metrics.insert("volatility".to_string(), 0.025);
custom_metrics.insert("rsi".to_string(), 65.0);

let market_data = MarketData {
    price: 45000.0,
    volume: 2.3,
    timestamp: chrono::Utc::now(),
    symbol: "BTC-USD".to_string(),
    metrics: Some(custom_metrics),
};
```

### Pattern Detection

```rust
use ares_neuromorphic_core::signals::{PatternDetector, DetectorConfig};

let config = DetectorConfig {
    confidence_threshold: 0.7,
    max_patterns: 3,
    enable_temporal: true,
    sensitivity: 0.8,
    ..Default::default()
};

let mut detector = PatternDetector::with_config(config);
```

### Signal Generation

```rust
use ares_neuromorphic_core::signals::TradingSignal;

// From detected patterns
let signal = detector.generate_signal(&detected_patterns, &reservoir_state);

match signal.direction {
    SignalDirection::Buy => println!("Buy signal with strength {:?}", signal.strength),
    SignalDirection::Sell => println!("Sell signal with strength {:?}", signal.strength),
    SignalDirection::Hold => println!("Hold signal"),
}
```

## Performance

The engine is optimized for high-frequency trading applications:

- **Latency**: < 10 microseconds typical processing time
- **Throughput**: > 1M samples/second on modern hardware
- **Memory**: ~100MB baseline memory usage
- **Scalability**: Linear scaling with number of neurons

### Benchmarks

```bash
cargo bench --features="async"
```

## Features Flags

- `default = ["async", "std"]` - Standard library and async support
- `async` - Enable async processing capabilities
- `brian2` - Enable Brian2 neuromorphic simulator integration
- `cuda` - Enable CUDA GPU acceleration
- `std` - Standard library support (disable for no_std)

## Integration Examples

### With Paper Trading

```rust
use ares_neuromorphic_core::NeuromorphicEngine;
// Assuming you have the paper trading crate
// use neuromorphic_paper_trader::NeuromorphicPaperTrader;

let mut engine = NeuromorphicEngine::new(config)?;
// let mut trader = NeuromorphicPaperTrader::new(trading_config);

// In your processing loop:
let prediction = engine.process(market_data)?;
if prediction.confidence > 0.7 {
    // Convert prediction to trading signal
    // trader.process_prediction_signal(trading_signal).await?;
}
```

### With Real-time Data

```rust
use tokio_stream::StreamExt;

let mut data_stream = market_data_stream(); // Your data source
let mut engine = NeuromorphicEngine::new(config)?;

while let Some(market_data) = data_stream.next().await {
    match engine.process(market_data) {
        Ok(prediction) => {
            if prediction.confidence > threshold {
                // Act on high-confidence predictions
                process_prediction(prediction).await;
            }
        }
        Err(e) => eprintln!("Processing error: {}", e),
    }
}
```

## Error Handling

The crate provides comprehensive error types:

```rust
use ares_neuromorphic_core::ProcessingError;

match engine.process(market_data) {
    Ok(prediction) => handle_prediction(prediction),
    Err(ProcessingError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(ProcessingError::SpikeEncodingError(msg)) => {
        eprintln!("Encoding error: {}", msg);
    }
    Err(ProcessingError::ReservoirError(msg)) => {
        eprintln!("Reservoir error: {}", msg);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Documentation

- [API Documentation](https://docs.rs/ares-neuromorphic-core)
- [Architecture Guide](docs/architecture.md)
- [Performance Tuning](docs/performance.md)
- [Integration Examples](examples/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [ARES-51](https://github.com/ares-systems/ares-51) - Full neuromorphic trading system
- [Neuromorphic Paper Trader](https://github.com/ares-systems/neuromorphic-paper-trader) - Paper trading implementation

## Citation

If you use this crate in academic research, please cite:

```bibtex
@software{ares_neuromorphic_core,
  title={ARES Neuromorphic Core: High-Performance Neuromorphic Computing for Financial Markets},
  author={Serfaty, Ididia},
  year={2025},
  url={https://github.com/ares-systems/ares-neuromorphic-core}
}
```