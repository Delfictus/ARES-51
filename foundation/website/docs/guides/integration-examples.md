---
sidebar_position: 1
title: "Integration Examples"
description: "Real-world integration examples for ARES ChronoFabric systems"
---

# Integration Examples

This guide provides practical examples of integrating ARES ChronoFabric into various application architectures.

## Microservices Integration

### Service-to-Service Communication

```rust
use ares_chronofabric::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OrderCreated {
    order_id: String,
    customer_id: String,
    amount: f64,
    timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InventoryReservation {
    order_id: String,
    items: Vec<String>,
    reserved_at: u64,
}

// Order Service
#[tokio::main]
async fn order_service() -> Result<(), Box<dyn std::error::Error>> {
    let bus = PhaseCoherenceBus::new(BusConfig::default())?;
    
    // Create order
    let order = OrderCreated {
        order_id: "order-123".to_string(),
        customer_id: "customer-456".to_string(),
        amount: 99.99,
        timestamp: csf_time::now()?.as_nanos(),
    };
    
    // Publish order creation event
    let packet = PhasePacket::new(order, ComponentId::OrderService)
        .with_priority(Priority::High)
        .with_guaranteed_delivery(3);
        
    bus.publish(packet).await?;
    Ok(())
}

// Inventory Service
#[tokio::main] 
async fn inventory_service() -> Result<(), Box<dyn std::error::Error>> {
    let bus = PhaseCoherenceBus::new(BusConfig::default())?;
    
    // Subscribe to order events
    let mut rx = bus.subscribe::<OrderCreated>().await?;
    
    while let Some(order_packet) = rx.recv().await {
        let order = &order_packet.payload;
        
        // Reserve inventory
        let reservation = InventoryReservation {
            order_id: order.order_id.clone(),
            items: vec!["item-1".to_string(), "item-2".to_string()],
            reserved_at: csf_time::now()?.as_nanos(),
        };
        
        // Publish reservation event
        let packet = PhasePacket::new(reservation, ComponentId::InventoryService)
            .with_correlation_id(order_packet.id); // Link to original order
            
        bus.publish(packet).await?;
    }
    
    Ok(())
}
```

## Event Sourcing Pattern

```rust
use ares_chronofabric::prelude::*;
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DomainEvent {
    aggregate_id: String,
    event_type: String,
    event_data: Value,
    version: u64,
    timestamp: u64,
}

struct EventStore {
    bus: Arc<PhaseCoherenceBus>,
    storage: Arc<dyn EventStorage>,
}

impl EventStore {
    async fn append_events(
        &self,
        stream_id: &str,
        expected_version: u64,
        events: Vec<DomainEvent>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Store events atomically
        let stored_events = self.storage.append(stream_id, expected_version, events).await?;
        
        // Publish events to bus for projections
        for event in stored_events {
            let packet = PhasePacket::new(event, ComponentId::EventStore)
                .with_priority(Priority::High)
                .with_deadline(csf_time::now()? + Duration::from_millis(100));
                
            self.bus.publish(packet).await?;
        }
        
        Ok(())
    }
}
```

## Real-time Analytics

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MetricPoint {
    metric_name: String,
    value: f64,
    labels: HashMap<String, String>,
    timestamp: u64,
}

struct MetricsCollector {
    bus: Arc<PhaseCoherenceBus>,
    aggregator: Arc<MetricsAggregator>,
}

impl MetricsCollector {
    async fn start_collection(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut rx = self.bus.subscribe::<MetricPoint>().await?;
        
        // Real-time metrics aggregation
        tokio::spawn(async move {
            let mut window = FixedWindow::new(Duration::from_secs(60));
            
            while let Some(metric_packet) = rx.recv().await {
                let metric = &metric_packet.payload;
                
                // Add to time window
                window.add_point(metric.clone());
                
                // Publish aggregated metrics every minute
                if window.is_full() {
                    let aggregated = window.aggregate();
                    let packet = PhasePacket::new(aggregated, ComponentId::MetricsAggregator)
                        .with_priority(Priority::Medium);
                        
                    self.bus.publish(packet).await?;
                    window.reset();
                }
            }
            
            Ok::<(), Box<dyn std::error::Error>>(())
        });
        
        Ok(())
    }
}
```

## Stream Processing Pipeline

```rust
use ares_chronofabric::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawData {
    sensor_id: String,
    value: f64,
    timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProcessedData {
    sensor_id: String,
    normalized_value: f64,
    anomaly_score: f64,
    timestamp: u64,
}

struct StreamProcessor {
    bus: Arc<PhaseCoherenceBus>,
    model: Arc<AnomalyDetectionModel>,
}

impl StreamProcessor {
    async fn start_processing(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut raw_rx = self.bus.subscribe::<RawData>().await?;
        
        while let Some(raw_packet) = raw_rx.recv().await {
            let raw = &raw_packet.payload;
            
            // Process data with deadline constraint
            let deadline = raw_packet.timestamp + Duration::from_millis(10);
            
            if csf_time::now()? < deadline {
                // Normalize and detect anomalies
                let normalized = self.normalize_value(raw.value);
                let anomaly_score = self.model.predict_anomaly(normalized).await?;
                
                let processed = ProcessedData {
                    sensor_id: raw.sensor_id.clone(),
                    normalized_value: normalized,
                    anomaly_score,
                    timestamp: csf_time::now()?.as_nanos(),
                };
                
                // Publish processed data
                let packet = PhasePacket::new(processed, ComponentId::StreamProcessor)
                    .with_deadline(deadline)
                    .add_temporal_correlation(vec![raw_packet.id]);
                    
                self.bus.publish(packet).await?;
            } else {
                // Handle deadline miss
                tracing::warn!("Processing deadline missed for sensor {}", raw.sensor_id);
            }
        }
        
        Ok(())
    }
}
```

## Load Testing Integration

```rust
use ares_chronofabric::prelude::*;
use tokio::time::{interval, Duration};

/// High-throughput load testing
async fn load_test_publisher(
    bus: Arc<PhaseCoherenceBus>,
    target_mps: u64, // Messages per second
    duration: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let interval_ns = 1_000_000_000 / target_mps;
    let mut ticker = interval(Duration::from_nanos(interval_ns));
    
    let start_time = std::time::Instant::now();
    let mut message_count = 0u64;
    
    while start_time.elapsed() < duration {
        ticker.tick().await;
        
        let test_data = TestMessage {
            id: message_count,
            payload: format!("Test message {}", message_count),
            timestamp: csf_time::now()?.as_nanos(),
        };
        
        let packet = PhasePacket::new(test_data, ComponentId::LoadTester)
            .with_priority(Priority::Medium);
            
        // Non-blocking publish to handle backpressure
        match bus.try_publish(packet) {
            Ok(_) => message_count += 1,
            Err(BusError::ResourceExhausted { .. }) => {
                // Handle backpressure gracefully
                tokio::time::sleep(Duration::from_micros(1)).await;
            }
            Err(e) => return Err(e.into()),
        }
    }
    
    println!("Load test completed: {} messages sent", message_count);
    Ok(())
}
```

## Foreign Function Interface (FFI)

```rust
// C API bindings
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

#[no_mangle]
pub extern "C" fn chronofabric_init() -> c_int {
    match PhaseCoherenceBus::new(BusConfig::default()) {
        Ok(_) => 0,  // Success
        Err(_) => -1, // Error
    }
}

#[no_mangle]
pub extern "C" fn chronofabric_send_message(
    topic: *const c_char,
    data: *const c_char,
    data_len: usize,
) -> c_int {
    unsafe {
        let topic_str = CStr::from_ptr(topic).to_str().unwrap();
        let data_slice = std::slice::from_raw_parts(data as *const u8, data_len);
        
        // Create and send message
        // Implementation details...
        
        0 // Success
    }
}
```

## Python Integration (via FFI)

```python
import ctypes
import json
from ctypes import c_char_p, c_int, c_size_t

# Load the ChronoFabric library
lib = ctypes.CDLL('./libchronofabric.so')

# Define function signatures
lib.chronofabric_init.restype = c_int
lib.chronofabric_send_message.argtypes = [c_char_p, c_char_p, c_size_t]
lib.chronofabric_send_message.restype = c_int

class ChronoFabricClient:
    def __init__(self):
        result = lib.chronofabric_init()
        if result != 0:
            raise RuntimeError("Failed to initialize ChronoFabric")
    
    def send_message(self, topic: str, data: dict):
        json_data = json.dumps(data).encode('utf-8')
        topic_bytes = topic.encode('utf-8')
        
        result = lib.chronofabric_send_message(
            topic_bytes, json_data, len(json_data)
        )
        
        if result != 0:
            raise RuntimeError("Failed to send message")

# Usage
client = ChronoFabricClient()
client.send_message("sensor_data", {
    "sensor_id": "temp_01",
    "value": 23.5,
    "timestamp": time.time_ns()
})
```

These examples demonstrate how to integrate ARES ChronoFabric into various application patterns while leveraging its high-performance messaging and temporal coherence capabilities.