# ARES Spike Encoding

High-performance neuromorphic spike encoding algorithms for converting market data and other time-series data into neural spike trains.

[![Crates.io](https://img.shields.io/crates/v/ares-spike-encoding.svg)](https://crates.io/crates/ares-spike-encoding)
[![Documentation](https://docs.rs/ares-spike-encoding/badge.svg)](https://docs.rs/ares-spike-encoding)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ARES Spike Encoding provides various spike encoding methods used in neuromorphic computing and spiking neural networks, optimized for financial market data processing. Convert continuous time-series data into discrete spike trains that can be processed by neuromorphic algorithms.

## Features

- 🧠 **Multiple Encoding Methods**: Rate, temporal, population, phase, latency, and burst coding
- ⚡ **High Performance**: Optimized algorithms with SIMD support and caching
- 🔧 **Flexible Configuration**: Extensive parameter customization for different use cases
- 📊 **Financial Market Optimized**: Specialized presets for crypto, forex, stocks, and more
- 🎯 **Quality Control**: Built-in validation and quality metrics
- 📈 **Real-time Capable**: Designed for low-latency streaming applications

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ares-spike-encoding = "0.1.0"
```

### Basic Usage

```rust
use ares_spike_encoding::{SpikeEncoder, EncodingMethod, MarketData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a rate-based spike encoder
    let mut encoder = SpikeEncoder::new(1000, 1000.0)?
        .with_method(EncodingMethod::Rate)?;

    // Create market data
    let market_data = MarketData::new("BTC-USD", 50000.0, 1.5)
        .with_metric("volatility".to_string(), 0.025);

    // Encode to spike pattern
    let spike_pattern = encoder.encode(&market_data)?;

    println!("Generated {} spikes over {:.1}ms", 
             spike_pattern.spike_count(), 
             spike_pattern.duration_ms);
    println!("Spike rate: {:.1} Hz", spike_pattern.spike_rate());

    Ok(())
}
```

### Using Presets

```rust
use ares_spike_encoding::{SpikeEncoder, FinancialPresets, MarketData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use cryptocurrency-optimized preset
    let config = FinancialPresets::crypto_config();
    let mut encoder = SpikeEncoder::with_config(config)?;

    let market_data = MarketData::new("ETH-USD", 3000.0, 2.1);
    let pattern = encoder.encode(&market_data)?;

    println!("Crypto-optimized encoding: {} spikes", pattern.spike_count());
    Ok(())
}
```

## Encoding Methods

### Rate Coding
Encodes values as spike frequencies - higher values produce more spikes per unit time.

```rust
use ares_spike_encoding::{rate_encoder, MarketData};

let mut encoder = rate_encoder(1000, 1000.0)?;
let data = MarketData::new("BTC-USD", 50000.0, 1.5);
let pattern = encoder.encode(&data)?;
```

### Temporal Coding
Encodes values as spike timing - higher values produce earlier spikes.

```rust
use ares_spike_encoding::{temporal_encoder, MarketData};

let mut encoder = temporal_encoder(1000, 1000.0)?;
let data = MarketData::new("ETH-USD", 3000.0, 2.0);
let pattern = encoder.encode(&data)?;
```

### Population Coding
Uses multiple neurons with overlapping tuning curves to represent values.

```rust
use ares_spike_encoding::{population_encoder, MarketData};

let mut encoder = population_encoder(1000, 1000.0)?;
let data = MarketData::new("SOL-USD", 100.0, 5.0);
let pattern = encoder.encode(&data)?;
```

### Phase Coding
Encodes values as the phase relationship between spike trains.

```rust
use ares_spike_encoding::{phase_encoder, MarketData};

let mut encoder = phase_encoder(1000, 1000.0)?;
let data = MarketData::new("ADA-USD", 1.5, 1000.0);
let pattern = encoder.encode(&data)?;
```

## Configuration

### Custom Encoding Configuration

```rust
use ares_spike_encoding::{
    SpikeEncoder, EncodingConfig, EncodingMethod, 
    RateParams, RateScaling, NoiseParams, NoiseType
};

let config = EncodingConfig {
    neuron_count: 5000,
    window_ms: 500.0,
    method: EncodingMethod::Rate,
    rate_params: RateParams {
        max_rate: 200.0,
        min_rate: 5.0,
        scaling: RateScaling::Logarithmic,
    },
    noise_params: NoiseParams {
        level: 0.01,
        noise_type: NoiseType::Gaussian,
        seed: Some(42),
    },
    ..Default::default()
};

let mut encoder = SpikeEncoder::with_config(config)?;
```

### Using Preset Builder

```rust
use ares_spike_encoding::{PresetBuilder, EncodingMethod};

let config = PresetBuilder::from_preset("balanced")
    .unwrap()
    .neuron_count(10000)
    .window_ms(500.0)
    .method(EncodingMethod::Population)
    .max_rate(300.0)
    .noise_level(0.005)
    .build();

let mut encoder = SpikeEncoder::with_config(config)?;
```

## Presets

### General Purpose Presets

- **`balanced`** - Good default for most applications
- **`high_frequency`** - Optimized for HFT applications
- **`low_latency`** - Minimal latency configuration
- **`high_precision`** - Maximum accuracy configuration
- **`memory_efficient`** - Minimal memory usage
- **`research`** - Extensive parameters for research

### Financial Market Presets

- **`crypto`** - Cryptocurrency markets (high volatility)
- **`forex`** - Foreign exchange markets (lower volatility)
- **`stocks`** - Stock markets (moderate volatility)
- **`commodities`** - Commodity markets (slow movements)
- **`derivatives`** - Options/derivatives (complex patterns)
- **`hft`** - High-frequency trading (ultra-low latency)

```rust
use ares_spike_encoding::{SpikeEncoder, FinancialPresets};

// Different market configurations
let crypto_config = FinancialPresets::crypto_config();
let forex_config = FinancialPresets::forex_config();
let hft_config = FinancialPresets::hft_config();
```

## Advanced Features

### Batch Processing

```rust
use ares_spike_encoding::{batch_encode, rate_encoder, MarketData};

let mut encoder = rate_encoder(1000, 1000.0)?;
let data_points = vec![
    MarketData::new("BTC-USD", 50000.0, 1.0),
    MarketData::new("BTC-USD", 51000.0, 1.2),
    MarketData::new("BTC-USD", 49000.0, 0.8),
];

let patterns = batch_encode(&mut encoder, &data_points)?;
println!("Processed {} data points", patterns.len());
```

### Adaptive Encoding

```rust
use ares_spike_encoding::{adaptive_encode, MarketData};

// Automatically choose best encoding method
let data = MarketData::new("BTC-USD", 50000.0, 1.5)
    .with_metric("rsi".to_string(), 65.0);

let pattern = adaptive_encode(&data, 1000, 1000.0)?;
```

### Pattern Analysis

```rust
let pattern = encoder.encode(&market_data)?;

// Analyze the generated pattern
println!("Spike count: {}", pattern.spike_count());
println!("Spike rate: {:.1} Hz", pattern.spike_rate());
println!("Active neurons: {:?}", pattern.active_neurons());

if let Some(isi) = pattern.average_isi() {
    println!("Average inter-spike interval: {:.2}ms", isi);
}

if let Some(cv) = pattern.coefficient_of_variation() {
    println!("Coefficient of variation: {:.3}", cv);
}

// Get firing rates by neuron
let rates = pattern.neuron_firing_rates();
println!("Neuron firing rates: {:?}", rates);
```

### Quality Metrics

```rust
// Access pattern metadata and quality metrics
let metadata = &pattern.metadata;
println!("Pattern strength: {:.2}", metadata.strength);
println!("Encoding method: {:?}", pattern.encoding_method);

if let Some(metrics) = &metadata.quality_metrics {
    if let Some(snr) = metrics.snr {
        println!("Signal-to-noise ratio: {:.2}", snr);
    }
}
```

## Performance

The library is optimized for high-throughput applications:

- **Encoding Rate**: > 10M samples/second on modern hardware
- **Latency**: < 1 microsecond typical encoding time
- **Memory**: Configurable memory usage with caching
- **Scalability**: Linear scaling with neuron count

### Benchmarking

```bash
cargo bench --features="ndarray-support"
```

## Feature Flags

- `default = ["std", "ndarray-support"]` - Standard features
- `std` - Standard library support (disable for no_std)
- `ndarray-support` - Enable ndarray integration
- `simd` - Enable SIMD acceleration
- `brian2` - Enable Brian2 simulator integration

### N-dimensional Array Support

```toml
[dependencies]
ares-spike-encoding = { version = "0.1.0", features = ["ndarray-support"] }
```

```rust
use ares_spike_encoding::NDArrayEncoder;
use ndarray::Array1;

let mut encoder = NDArrayEncoder::new(1000, 1000.0)?;
let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let pattern = encoder.encode_array(&data)?;
```

## Integration Examples

### With ARES Neuromorphic Core

```rust
use ares_spike_encoding::{SpikeEncoder, FinancialPresets};
// use ares_neuromorphic_core::NeuromorphicEngine;

let encoder_config = FinancialPresets::crypto_config();
let mut encoder = SpikeEncoder::with_config(encoder_config)?;

// let mut engine = NeuromorphicEngine::new(engine_config)?;

// In processing loop:
let market_data = get_market_data();
let spike_pattern = encoder.encode(&market_data)?;
// let prediction = engine.process_spikes(&spike_pattern)?;
```

### Real-time Data Processing

```rust
use tokio_stream::StreamExt;
use ares_spike_encoding::{SpikeEncoder, FinancialPresets};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = FinancialPresets::hft_config(); // Ultra-low latency
    let mut encoder = SpikeEncoder::with_config(config)?;
    
    let mut data_stream = market_data_stream(); // Your data source
    
    while let Some(market_data) = data_stream.next().await {
        let start = std::time::Instant::now();
        
        match encoder.encode(&market_data) {
            Ok(pattern) => {
                let latency = start.elapsed();
                println!("Encoded {} spikes in {:?}", 
                        pattern.spike_count(), latency);
                
                // Process pattern...
            }
            Err(e) => eprintln!("Encoding error: {}", e),
        }
    }
    
    Ok(())
}
```

## Error Handling

```rust
use ares_spike_encoding::{SpikeEncoder, EncodingError};

match encoder.encode(&market_data) {
    Ok(pattern) => {
        // Process successful encoding
        process_pattern(pattern);
    }
    Err(EncodingError::InvalidInput(msg)) => {
        eprintln!("Invalid input data: {}", msg);
    }
    Err(EncodingError::EncodingFailed(msg)) => {
        eprintln!("Encoding algorithm failed: {}", msg);
    }
    Err(EncodingError::QualityFailed(msg)) => {
        eprintln!("Quality control failed: {}", msg);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Documentation

- [API Documentation](https://docs.rs/ares-spike-encoding)
- [Encoding Methods Guide](docs/encoding-methods.md)
- [Performance Tuning](docs/performance.md)
- [Financial Applications](docs/financial-usage.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [ARES Neuromorphic Core](https://github.com/ares-systems/ares-neuromorphic-core) - Core neuromorphic engine
- [ARES-51](https://github.com/ares-systems/ares-51) - Full neuromorphic trading system

## Citation

If you use this crate in academic research, please cite:

```bibtex
@software{ares_spike_encoding,
  title={ARES Spike Encoding: High-Performance Neuromorphic Spike Encoding for Financial Markets},
  author={Serfaty, Ididia},
  year={2025},
  url={https://github.com/ares-systems/ares-spike-encoding}
}
```