# ARES ChronoFabric Integration Examples Guide

**Version**: 1.0  
**Last Updated**: August 26, 2025  
**Document Type**: Integration Guide

> **🔧 Practical Integration**: This guide provides real-world examples and patterns for integrating ARES ChronoFabric into production systems.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Integration Patterns](#basic-integration-patterns)
3. [Advanced Use Cases](#advanced-use-cases)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### System Requirements

Before integrating ARES ChronoFabric, ensure your system meets the requirements:

```bash
# Hardware Requirements
- CPU: x86_64 with TSC support (Intel/AMD)
- RAM: Minimum 4GB, recommended 16GB+  
- Storage: SSD recommended for low-latency I/O
- Network: 1Gbps+ for distributed deployments

# Software Requirements  
- Rust: 1.75+ with cargo
- OS: Linux (Ubuntu 20.04+), macOS, Windows 10+
- Optional: CUDA 11.0+ (GPU acceleration)
```

### Quick Setup

```bash
# Clone repository
git clone https://github.com/ares-systems/chronofabric
cd chronofabric

# Install dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config \
    libssl-dev libclang-dev protobuf-compiler

# Build with optimizations
cargo build --release --features "hardware-acceleration"

# Run basic test
cargo test --lib csf_bus::tests::test_hardware_router
```

### Basic Project Setup

Add ARES ChronoFabric to your `Cargo.toml`:

```toml
[dependencies]
# Core components
csf-core = { path = "path/to/chronofabric/crates/csf-core" }
csf-bus = { path = "path/to/chronofabric/crates/csf-bus" }
csf-time = { path = "path/to/chronofabric/crates/csf-time" }

# Optional advanced features
csf-clogic = { path = "path/to/chronofabric/crates/csf-clogic", optional = true }
csf-mlir = { path = "path/to/chronofabric/crates/csf-mlir", optional = true }

# Required async runtime
tokio = { version = "1.35", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[features]
default = ["hardware-acceleration"]
hardware-acceleration = []
cognitive-modules = ["csf-clogic"]
ml-acceleration = ["csf-mlir"]
```

---

## Basic Integration Patterns

### 1. Simple Pub/Sub System

**Use Case**: Event-driven microservices communication.

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket, BusConfig};
use csf_core::prelude::*;
use csf_time::initialize_global_time_source;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio;
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserRegistered {
    user_id: u64,
    email: String,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmailNotification {
    to: String,
    subject: String,
    body: String,
    priority: String,
}

/// Simple event-driven user registration system
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize ChronoFabric time system
    initialize_global_time_source()?;
    
    // Create high-performance bus
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig {
        channel_buffer_size: 1024,
    })?);
    
    info!("ARES ChronoFabric system initialized");
    
    // Start email notification service
    let email_service = start_email_service(bus.clone()).await?;
    
    // Start user analytics service  
    let analytics_service = start_analytics_service(bus.clone()).await?;
    
    // Simulate user registrations
    for i in 1..=10 {
        let event = UserRegistered {
            user_id: i,
            email: format!("user{}@example.com", i),
            timestamp: csf_time::now()?.as_nanos(),
        };
        
        let packet = PhasePacket::new(event, ComponentId::custom(100))
            .with_priority(Priority::Normal)
            .with_targets(0xFF); // Broadcast to all services
        
        bus.publish(packet).await?;
        
        info!("Published user registration: user_id={}", i);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    // Wait for processing to complete
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    // Check system performance
    let stats = bus.get_stats();
    info!(
        "System performance: {} messages/sec, avg latency {}ns",
        stats.throughput_mps, stats.avg_latency_ns
    );
    
    Ok(())
}

async fn start_email_service(
    bus: Arc<PhaseCoherenceBus>
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    let mut rx = bus.subscribe::<UserRegistered>().await?;
    
    Ok(tokio::spawn(async move {
        info!("Email service started");
        
        while let Some(user_packet) = rx.recv().await {
            let user = &user_packet.payload;
            
            // Create welcome email
            let notification = EmailNotification {
                to: user.email.clone(),
                subject: "Welcome to our platform!".to_string(),
                body: format!("Hello! Your account {} has been created.", user.user_id),
                priority: "normal".to_string(),
            };
            
            // Send via bus (could be consumed by actual email sender)
            let email_packet = PhasePacket::new(notification, ComponentId::custom(200))
                .with_priority(Priority::Normal);
                
            if let Err(e) = bus.publish(email_packet).await {
                error!("Failed to publish email notification: {}", e);
            } else {
                info!("Email notification queued for user {}", user.user_id);
            }
        }
        
        info!("Email service stopped");
    }))
}

async fn start_analytics_service(
    bus: Arc<PhaseCoherenceBus>
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    let mut rx = bus.subscribe::<UserRegistered>().await?;
    
    Ok(tokio::spawn(async move {
        info!("Analytics service started");
        let mut user_count = 0u64;
        
        while let Some(user_packet) = rx.recv().await {
            user_count += 1;
            
            // Simulate analytics processing
            info!(
                "Analytics: Total users = {}, Latest registration = {}",
                user_count, user_packet.payload.user_id
            );
            
            // Could publish analytics events back to bus
            if user_count % 5 == 0 {
                info!("Milestone: {} users registered!", user_count);
            }
        }
        
        info!("Analytics service stopped");
    }))
}
```

### 2. Request-Response Pattern

**Use Case**: Synchronous-style communication over async message bus.

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket};
use csf_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeRequest {
    request_id: Uuid,
    operation: String,
    operands: Vec<f64>,
    timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeResponse {
    request_id: Uuid,
    result: Result<f64, String>,
    processing_time_ns: u64,
}

/// Request-Response service implementation
pub struct ComputeService {
    bus: Arc<PhaseCoherenceBus>,
    pending_requests: Arc<RwLock<HashMap<Uuid, oneshot::Sender<ComputeResponse>>>>,
}

impl ComputeService {
    pub async fn new(bus: Arc<PhaseCoherenceBus>) -> Result<Self, Box<dyn std::error::Error>> {
        let service = Self {
            bus: bus.clone(),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Start request processor
        service.start_request_processor().await?;
        
        // Start response handler
        service.start_response_handler().await?;
        
        Ok(service)
    }
    
    /// Send request and wait for response
    pub async fn compute(
        &self,
        operation: &str,
        operands: Vec<f64>,
        timeout: tokio::time::Duration,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let request_id = Uuid::new_v4();
        let (response_tx, response_rx) = oneshot::channel();
        
        // Register pending request
        self.pending_requests.write().await.insert(request_id, response_tx);
        
        // Send request
        let request = ComputeRequest {
            request_id,
            operation: operation.to_string(),
            operands,
            timeout_ms: timeout.as_millis() as u64,
        };
        
        let packet = PhasePacket::new(request, ComponentId::custom(300))
            .with_priority(Priority::High)
            .with_timeout(timeout.as_nanos() as u64);
        
        self.bus.publish(packet).await?;
        
        // Wait for response with timeout
        match tokio::time::timeout(timeout, response_rx).await {
            Ok(Ok(response)) => match response.result {
                Ok(value) => Ok(value),
                Err(error) => Err(error.into()),
            },
            Ok(Err(_)) => Err("Response channel closed".into()),
            Err(_) => Err("Request timeout".into()),
        }
    }
    
    async fn start_request_processor(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let mut rx = bus.subscribe::<ComputeRequest>().await?;
        
        tokio::spawn(async move {
            while let Some(request_packet) = rx.recv().await {
                let start_time = csf_time::now().unwrap();
                let request = &request_packet.payload;
                
                // Process computation
                let result = match request.operation.as_str() {
                    "add" => Ok(request.operands.iter().sum()),
                    "multiply" => Ok(request.operands.iter().product()),
                    "mean" => {
                        if request.operands.is_empty() {
                            Err("Cannot compute mean of empty list".to_string())
                        } else {
                            Ok(request.operands.iter().sum::<f64>() / request.operands.len() as f64)
                        }
                    },
                    _ => Err(format!("Unknown operation: {}", request.operation)),
                };
                
                let end_time = csf_time::now().unwrap();
                let processing_time = end_time.as_nanos() - start_time.as_nanos();
                
                // Send response
                let response = ComputeResponse {
                    request_id: request.request_id,
                    result,
                    processing_time_ns: processing_time,
                };
                
                let response_packet = PhasePacket::new(response, ComponentId::custom(301));
                
                if let Err(e) = bus.publish(response_packet).await {
                    tracing::error!("Failed to publish compute response: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_response_handler(&self) -> Result<(), Box<dyn std::error::Error>> {
        let pending_requests = self.pending_requests.clone();
        let mut rx = self.bus.subscribe::<ComputeResponse>().await?;
        
        tokio::spawn(async move {
            while let Some(response_packet) = rx.recv().await {
                let response = response_packet.payload;
                
                // Find and notify waiting request
                if let Some(tx) = pending_requests.write().await.remove(&response.request_id) {
                    if let Err(_) = tx.send(response) {
                        tracing::warn!("Failed to send response - receiver dropped");
                    }
                }
            }
        });
        
        Ok(())
    }
}

// Usage example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    csf_time::initialize_global_time_source()?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig::default())?);
    let compute_service = ComputeService::new(bus).await?;
    
    // Test various operations
    let add_result = compute_service.compute(
        "add", 
        vec![1.0, 2.0, 3.0], 
        tokio::time::Duration::from_secs(1)
    ).await?;
    println!("Add result: {}", add_result);
    
    let mean_result = compute_service.compute(
        "mean",
        vec![10.0, 20.0, 30.0],
        tokio::time::Duration::from_secs(1)
    ).await?;
    println!("Mean result: {}", mean_result);
    
    Ok(())
}
```

### 3. Stream Processing Pipeline

**Use Case**: Real-time data processing with multiple stages.

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket};
use csf_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawSensorData {
    sensor_id: u32,
    raw_value: f64,
    timestamp: u64,
    quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FilteredData {
    sensor_id: u32,
    filtered_value: f64,
    timestamp: u64,
    filter_applied: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AggregatedData {
    window_start: u64,
    window_end: u64,
    sensor_count: u32,
    avg_value: f64,
    min_value: f64,
    max_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Alert {
    alert_type: String,
    message: String,
    severity: String,
    timestamp: u64,
}

/// Stream processing pipeline with multiple stages
pub struct SensorProcessingPipeline {
    bus: Arc<PhaseCoherenceBus>,
}

impl SensorProcessingPipeline {
    pub async fn new(bus: Arc<PhaseCoherenceBus>) -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline = Self { bus };
        
        // Start processing stages
        pipeline.start_filter_stage().await?;
        pipeline.start_aggregation_stage().await?;
        pipeline.start_alert_stage().await?;
        pipeline.start_output_stage().await?;
        
        Ok(pipeline)
    }
    
    /// Inject raw sensor data into pipeline
    pub async fn inject_data(&self, data: RawSensorData) -> Result<(), Box<dyn std::error::Error>> {
        let packet = PhasePacket::new(data, ComponentId::custom(400))
            .with_priority(Priority::High)
            .with_targets(0xFF);
        
        self.bus.publish(packet).await?;
        Ok(())
    }
    
    async fn start_filter_stage(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let mut rx = bus.subscribe::<RawSensorData>().await?;
        
        tokio::spawn(async move {
            tracing::info!("Filter stage started");
            
            while let Some(data_packet) = rx.recv().await {
                let raw_data = &data_packet.payload;
                
                // Quality check and filtering
                if raw_data.quality < 0.5 {
                    tracing::warn!("Low quality data from sensor {}, skipping", raw_data.sensor_id);
                    continue;
                }
                
                // Apply simple moving average filter
                let filtered_value = raw_data.raw_value * 0.7 + 
                                   (raw_data.raw_value * 0.3); // Simplified filter
                
                let filtered_data = FilteredData {
                    sensor_id: raw_data.sensor_id,
                    filtered_value,
                    timestamp: raw_data.timestamp,
                    filter_applied: "moving_average".to_string(),
                };
                
                let packet = PhasePacket::new(filtered_data, ComponentId::custom(401))
                    .with_priority(Priority::Normal);
                
                if let Err(e) = bus.publish(packet).await {
                    tracing::error!("Failed to publish filtered data: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_aggregation_stage(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let mut rx = bus.subscribe::<FilteredData>().await?;
        
        tokio::spawn(async move {
            tracing::info!("Aggregation stage started");
            
            let mut window_data: Vec<FilteredData> = Vec::new();
            let window_size = std::time::Duration::from_secs(5); // 5-second windows
            let mut last_aggregation = csf_time::now().unwrap();
            
            while let Some(data_packet) = rx.recv().await {
                let filtered_data = data_packet.payload;
                window_data.push(filtered_data);
                
                let current_time = csf_time::now().unwrap();
                if current_time.as_duration().saturating_sub(last_aggregation.as_duration()) >= window_size {
                    // Perform aggregation
                    if !window_data.is_empty() {
                        let values: Vec<f64> = window_data.iter().map(|d| d.filtered_value).collect();
                        
                        let aggregated = AggregatedData {
                            window_start: last_aggregation.as_nanos(),
                            window_end: current_time.as_nanos(),
                            sensor_count: window_data.len() as u32,
                            avg_value: values.iter().sum::<f64>() / values.len() as f64,
                            min_value: values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            max_value: values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                        };
                        
                        let packet = PhasePacket::new(aggregated, ComponentId::custom(402));
                        
                        if let Err(e) = bus.publish(packet).await {
                            tracing::error!("Failed to publish aggregated data: {}", e);
                        }
                        
                        window_data.clear();
                        last_aggregation = current_time;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_alert_stage(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let mut rx = bus.subscribe::<AggregatedData>().await?;
        
        tokio::spawn(async move {
            tracing::info!("Alert stage started");
            
            while let Some(agg_packet) = rx.recv().await {
                let agg_data = &agg_packet.payload;
                
                // Check for alert conditions
                if agg_data.avg_value > 100.0 {
                    let alert = Alert {
                        alert_type: "HIGH_VALUE".to_string(),
                        message: format!("High average value detected: {:.2}", agg_data.avg_value),
                        severity: "WARNING".to_string(),
                        timestamp: csf_time::now().unwrap().as_nanos(),
                    };
                    
                    let packet = PhasePacket::new(alert, ComponentId::custom(403))
                        .with_priority(Priority::High);
                    
                    if let Err(e) = bus.publish(packet).await {
                        tracing::error!("Failed to publish alert: {}", e);
                    }
                }
                
                if agg_data.sensor_count < 3 {
                    let alert = Alert {
                        alert_type: "LOW_DATA".to_string(),
                        message: format!("Low sensor data count: {}", agg_data.sensor_count),
                        severity: "INFO".to_string(),
                        timestamp: csf_time::now().unwrap().as_nanos(),
                    };
                    
                    let packet = PhasePacket::new(alert, ComponentId::custom(403))
                        .with_priority(Priority::Normal);
                    
                    if let Err(e) = bus.publish(packet).await {
                        tracing::error!("Failed to publish alert: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_output_stage(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let mut agg_rx = bus.subscribe::<AggregatedData>().await?;
        let mut alert_rx = bus.subscribe::<Alert>().await?;
        
        // Handle aggregated data output
        tokio::spawn(async move {
            while let Some(agg_packet) = agg_rx.recv().await {
                let agg = &agg_packet.payload;
                tracing::info!(
                    "AGGREGATED: {} sensors, avg={:.2}, min={:.2}, max={:.2}",
                    agg.sensor_count, agg.avg_value, agg.min_value, agg.max_value
                );
            }
        });
        
        // Handle alert output
        tokio::spawn(async move {
            while let Some(alert_packet) = alert_rx.recv().await {
                let alert = &alert_packet.payload;
                tracing::warn!(
                    "ALERT [{}]: {} - {}",
                    alert.severity, alert.alert_type, alert.message
                );
            }
        });
        
        Ok(())
    }
}

// Usage example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    csf_time::initialize_global_time_source()?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig {
        channel_buffer_size: 2048, // Large buffer for stream processing
    })?);
    
    let pipeline = SensorProcessingPipeline::new(bus.clone()).await?;
    
    // Simulate sensor data stream
    for i in 0..50 {
        let sensor_data = RawSensorData {
            sensor_id: (i % 5) + 1, // 5 different sensors
            raw_value: 50.0 + (i as f64 * 2.0) + (i as f64).sin() * 10.0, // Trending data
            timestamp: csf_time::now()?.as_nanos(),
            quality: if i % 10 == 0 { 0.3 } else { 0.9 }, // Occasional low quality
        };
        
        pipeline.inject_data(sensor_data).await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    // Let the pipeline process remaining data
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
    
    // Check system performance
    let stats = bus.get_stats();
    tracing::info!(
        "Pipeline performance: {} messages processed, avg latency {}ns",
        stats.packets_delivered, stats.avg_latency_ns
    );
    
    Ok(())
}
```

---

## Advanced Use Cases

### 1. Distributed State Machine

**Use Case**: Coordinated state management across multiple nodes.

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket};
use csf_core::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum WorkflowState {
    Pending,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateTransition {
    workflow_id: String,
    from_state: WorkflowState,
    to_state: WorkflowState,
    node_id: String,
    timestamp: u64,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateQuery {
    workflow_id: String,
    requester_node: String,
    request_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateResponse {
    workflow_id: String,
    current_state: WorkflowState,
    last_updated: u64,
    request_id: Uuid,
}

/// Distributed state machine with consensus
pub struct DistributedStateMachine {
    bus: Arc<PhaseCoherenceBus>,
    node_id: String,
    state_store: Arc<RwLock<HashMap<String, (WorkflowState, u64)>>>,
}

impl DistributedStateMachine {
    pub async fn new(
        bus: Arc<PhaseCoherenceBus>,
        node_id: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let state_machine = Self {
            bus: bus.clone(),
            node_id,
            state_store: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Start state transition handler
        state_machine.start_transition_handler().await?;
        
        // Start query handler
        state_machine.start_query_handler().await?;
        
        Ok(state_machine)
    }
    
    /// Transition workflow to new state
    pub async fn transition_state(
        &self,
        workflow_id: &str,
        to_state: WorkflowState,
        metadata: HashMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get current state
        let from_state = {
            let store = self.state_store.read().await;
            store.get(workflow_id).map(|(state, _)| *state).unwrap_or(WorkflowState::Pending)
        };
        
        // Validate transition
        if !self.is_valid_transition(from_state, to_state) {
            return Err(format!(
                "Invalid state transition: {:?} -> {:?}",
                from_state, to_state
            ).into());
        }
        
        // Broadcast state transition
        let transition = StateTransition {
            workflow_id: workflow_id.to_string(),
            from_state,
            to_state,
            node_id: self.node_id.clone(),
            timestamp: csf_time::now()?.as_nanos(),
            metadata,
        };
        
        let packet = PhasePacket::new(transition, ComponentId::custom(500))
            .with_priority(Priority::High)
            .with_targets(0xFF); // Broadcast to all nodes
        
        self.bus.publish(packet).await?;
        
        Ok(())
    }
    
    /// Query current state of workflow
    pub async fn query_state(&self, workflow_id: &str) -> Result<WorkflowState, Box<dyn std::error::Error>> {
        // First check local store
        {
            let store = self.state_store.read().await;
            if let Some((state, _)) = store.get(workflow_id) {
                return Ok(*state);
            }
        }
        
        // Query other nodes
        let query = StateQuery {
            workflow_id: workflow_id.to_string(),
            requester_node: self.node_id.clone(),
            request_id: Uuid::new_v4(),
        };
        
        let packet = PhasePacket::new(query, ComponentId::custom(501))
            .with_priority(Priority::Normal)
            .with_timeout(5_000_000_000); // 5 second timeout
        
        self.bus.publish(packet).await?;
        
        // Wait for response (simplified - in production, use proper async coordination)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Check store again
        let store = self.state_store.read().await;
        Ok(store.get(workflow_id).map(|(state, _)| *state).unwrap_or(WorkflowState::Pending))
    }
    
    fn is_valid_transition(&self, from: WorkflowState, to: WorkflowState) -> bool {
        use WorkflowState::*;
        matches!(
            (from, to),
            (Pending, Processing) |
            (Processing, Completed) |
            (Processing, Failed) |
            (Failed, Pending) // Allow retry
        )
    }
    
    async fn start_transition_handler(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let state_store = self.state_store.clone();
        let mut rx = bus.subscribe::<StateTransition>().await?;
        
        tokio::spawn(async move {
            while let Some(transition_packet) = rx.recv().await {
                let transition = &transition_packet.payload;
                
                // Update local state store
                let mut store = state_store.write().await;
                store.insert(
                    transition.workflow_id.clone(),
                    (transition.to_state, transition.timestamp),
                );
                
                tracing::info!(
                    "State transition: {} {} -> {:?} (from node {})",
                    transition.workflow_id,
                    transition.from_state as u8,
                    transition.to_state,
                    transition.node_id
                );
            }
        });
        
        Ok(())
    }
    
    async fn start_query_handler(&self) -> Result<(), Box<dyn std::error::Error>> {
        let bus = self.bus.clone();
        let state_store = self.state_store.clone();
        let node_id = self.node_id.clone();
        let mut rx = bus.subscribe::<StateQuery>().await?;
        
        tokio::spawn(async move {
            while let Some(query_packet) = rx.recv().await {
                let query = &query_packet.payload;
                
                // Don't respond to our own queries
                if query.requester_node == node_id {
                    continue;
                }
                
                // Check if we have the state
                let store = state_store.read().await;
                if let Some((state, last_updated)) = store.get(&query.workflow_id) {
                    let response = StateResponse {
                        workflow_id: query.workflow_id.clone(),
                        current_state: *state,
                        last_updated: *last_updated,
                        request_id: query.request_id,
                    };
                    
                    let packet = PhasePacket::new(response, ComponentId::custom(502));
                    
                    if let Err(e) = bus.publish(packet).await {
                        tracing::error!("Failed to publish state response: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
}

// Usage example with multiple nodes
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    csf_time::initialize_global_time_source()?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig::default())?);
    
    // Create multiple state machine nodes
    let node1 = DistributedStateMachine::new(bus.clone(), "node-1".to_string()).await?;
    let node2 = DistributedStateMachine::new(bus.clone(), "node-2".to_string()).await?;
    let node3 = DistributedStateMachine::new(bus.clone(), "node-3".to_string()).await?;
    
    // Simulate distributed workflow
    let workflow_id = "workflow-123";
    
    // Node 1 starts the workflow
    node1.transition_state(
        workflow_id,
        WorkflowState::Processing,
        [("initiator".to_string(), "node-1".to_string())].into(),
    ).await?;
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Node 2 queries the state (should see Processing)
    let state = node2.query_state(workflow_id).await?;
    tracing::info!("Node 2 sees workflow state: {:?}", state);
    
    // Node 3 completes the workflow
    node3.transition_state(
        workflow_id,
        WorkflowState::Completed,
        [("completer".to_string(), "node-3".to_string())].into(),
    ).await?;
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // All nodes should see the final state
    let final_state1 = node1.query_state(workflow_id).await?;
    let final_state2 = node2.query_state(workflow_id).await?;
    let final_state3 = node3.query_state(workflow_id).await?;
    
    tracing::info!(
        "Final states - Node1: {:?}, Node2: {:?}, Node3: {:?}",
        final_state1, final_state2, final_state3
    );
    
    Ok(())
}
```

### 2. Event Sourcing with ChronoFabric

**Use Case**: Event-driven architecture with temporal ordering and replay.

```rust
use csf_bus::{PhaseCoherenceBus, PhasePacket};
use csf_core::prelude::*;
use csf_time::{LogicalTime, global_hlc_now};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DomainEvent {
    aggregate_id: String,
    event_type: String,
    event_data: serde_json::Value,
    sequence_number: u64,
    logical_time: LogicalTime,
    correlation_id: Option<Uuid>,
    causation_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventQuery {
    aggregate_id: String,
    from_sequence: u64,
    max_count: usize,
    query_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]  
struct EventsResult {
    query_id: Uuid,
    events: Vec<DomainEvent>,
    has_more: bool,
}

/// Event store with temporal ordering and causality tracking
pub struct EventStore {
    bus: Arc<PhaseCoherenceBus>,
    event_streams: Arc<RwLock<HashMap<String, VecDeque<DomainEvent>>>>,
    projections: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl EventStore {
    pub async fn new(bus: Arc<PhaseCoherenceBus>) -> Result<Self, Box<dyn std::error::Error>> {
        let store = Self {
            bus: bus.clone(),
            event_streams: Arc::new(RwLock::new(HashMap::new())),
            projections: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Start event handler
        store.start_event_handler().await?;
        
        // Start projection updater
        store.start_projection_updater().await?;
        
        // Start query handler
        store.start_query_handler().await?;
        
        Ok(store)
    }
    
    /// Append event to stream with causality tracking
    pub async fn append_event(
        &self,
        aggregate_id: &str,
        event_type: &str,
        event_data: serde_json::Value,
        correlation_id: Option<Uuid>,
        causation_id: Option<Uuid>,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        // Get next sequence number
        let sequence_number = {
            let streams = self.event_streams.read().await;
            streams.get(aggregate_id).map(|s| s.len() as u64).unwrap_or(0)
        };
        
        // Create event with logical time for causality
        let event = DomainEvent {
            aggregate_id: aggregate_id.to_string(),
            event_type: event_type.to_string(),
            event_data,
            sequence_number,
            logical_time: global_hlc_now()?,
            correlation_id,
            causation_id,
        };
        
        // Publish event with causal dependencies
        let mut packet = PhasePacket::new(event, ComponentId::custom(600))
            .with_priority(Priority::High);
        
        if let Some(causation_id) = causation_id {
            packet.add_temporal_correlation(vec![causation_id]);
        }
        
        self.bus.publish(packet).await?;
        
        Ok(sequence_number)
    }
    
    /// Query events from stream
    pub async fn get_events(
        &self,
        aggregate_id: &str,
        from_sequence: u64,
        max_count: usize,
    ) -> Result<Vec<DomainEvent>, Box<dyn std::error::Error>> {
        let query = EventQuery {
            aggregate_id: aggregate_id.to_string(),
            from_sequence,
            max_count,
            query_id: Uuid::new_v4(),
        };
        
        let packet = PhasePacket::new(query, ComponentId::custom(601));
        self.bus.publish(packet).await?;
        
        // In a real implementation, this would wait for the query response
        // For demo purposes, return from local storage
        let streams = self.event_streams.read().await;
        if let Some(stream) = streams.get(aggregate_id) {
            let events: Vec<DomainEvent> = stream
                .iter()
                .skip(from_sequence as usize)
                .take(max_count)
                .cloned()
                .collect();
            Ok(events)
        } else {
            Ok(vec![])
        }
    }
    
    /// Get current projection state
    pub async fn get_projection(&self, projection_name: &str) -> Option<serde_json::Value> {
        let projections = self.projections.read().await;
        projections.get(projection_name).cloned()
    }
    
    async fn start_event_handler(&self) -> Result<(), Box<dyn std::error::Error>> {
        let event_streams = self.event_streams.clone();
        let mut rx = self.bus.subscribe::<DomainEvent>().await?;
        
        tokio::spawn(async move {
            while let Some(event_packet) = rx.recv().await {
                let event = event_packet.payload;
                
                // Store event in stream
                {
                    let mut streams = event_streams.write().await;
                    let stream = streams.entry(event.aggregate_id.clone()).or_insert_with(VecDeque::new);
                    stream.push_back(event.clone());
                }
                
                tracing::info!(
                    "Event stored: {} {} seq={} time={:?}",
                    event.aggregate_id,
                    event.event_type,
                    event.sequence_number,
                    event.logical_time
                );
            }
        });
        
        Ok(())
    }
    
    async fn start_projection_updater(&self) -> Result<(), Box<dyn std::error::Error>> {
        let projections = self.projections.clone();
        let mut rx = self.bus.subscribe::<DomainEvent>().await?;
        
        tokio::spawn(async move {
            while let Some(event_packet) = rx.recv().await {
                let event = &event_packet.payload;
                
                // Update projections based on event type
                match event.event_type.as_str() {
                    "UserRegistered" => {
                        let mut proj = projections.write().await;
                        let user_count = proj.get("user_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        proj.insert("user_count".to_string(), serde_json::json!(user_count + 1));
                    },
                    "OrderPlaced" => {
                        let mut proj = projections.write().await;
                        let order_count = proj.get("order_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        proj.insert("order_count".to_string(), serde_json::json!(order_count + 1));
                        
                        // Update total revenue if amount is provided
                        if let Some(amount) = event.event_data.get("amount").and_then(|v| v.as_f64()) {
                            let total_revenue = proj.get("total_revenue")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            proj.insert("total_revenue".to_string(), serde_json::json!(total_revenue + amount));
                        }
                    },
                    _ => {
                        // Generic event count
                        let mut proj = projections.write().await;
                        let event_count = proj.get("total_events")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        proj.insert("total_events".to_string(), serde_json::json!(event_count + 1));
                    }
                }
                
                tracing::debug!("Projections updated for event: {}", event.event_type);
            }
        });
        
        Ok(())
    }
    
    async fn start_query_handler(&self) -> Result<(), Box<dyn std::error::Error>> {
        let event_streams = self.event_streams.clone();
        let bus = self.bus.clone();
        let mut rx = bus.subscribe::<EventQuery>().await?;
        
        tokio::spawn(async move {
            while let Some(query_packet) = rx.recv().await {
                let query = &query_packet.payload;
                
                // Get events from stream
                let events = {
                    let streams = event_streams.read().await;
                    if let Some(stream) = streams.get(&query.aggregate_id) {
                        stream
                            .iter()
                            .skip(query.from_sequence as usize)
                            .take(query.max_count)
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    }
                };
                
                let has_more = {
                    let streams = event_streams.read().await;
                    streams.get(&query.aggregate_id)
                        .map(|s| s.len() > query.from_sequence as usize + query.max_count)
                        .unwrap_or(false)
                };
                
                let result = EventsResult {
                    query_id: query.query_id,
                    events,
                    has_more,
                };
                
                let packet = PhasePacket::new(result, ComponentId::custom(602));
                if let Err(e) = bus.publish(packet).await {
                    tracing::error!("Failed to publish query result: {}", e);
                }
            }
        });
        
        Ok(())
    }
}

// Usage example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    csf_time::initialize_global_time_source()?;
    csf_time::initialize_global_hlc(
        Arc::new(csf_time::TimeSourceImpl::new()?),
        1
    )?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig::default())?);
    let event_store = EventStore::new(bus.clone()).await?;
    
    // Simulate event-driven business logic
    let correlation_id = Uuid::new_v4();
    
    // User registration event
    let user_event = serde_json::json!({
        "user_id": "user-123",
        "email": "user@example.com",
        "name": "John Doe"
    });
    
    let user_seq = event_store.append_event(
        "user-123",
        "UserRegistered",
        user_event,
        Some(correlation_id),
        None,
    ).await?;
    
    // Order placement event (caused by user registration)
    let order_event = serde_json::json!({
        "order_id": "order-456",
        "user_id": "user-123",
        "amount": 99.99,
        "items": ["item1", "item2"]
    });
    
    let order_seq = event_store.append_event(
        "order-456",
        "OrderPlaced",
        order_event,
        Some(correlation_id),
        Some(Uuid::new_v4()), // Caused by user registration
    ).await?;
    
    // Wait for events to be processed
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Query events back
    let user_events = event_store.get_events("user-123", 0, 10).await?;
    let order_events = event_store.get_events("order-456", 0, 10).await?;
    
    tracing::info!("User events: {} found", user_events.len());
    tracing::info!("Order events: {} found", order_events.len());
    
    // Check projections
    if let Some(user_count) = event_store.get_projection("user_count").await {
        tracing::info!("Total users: {}", user_count);
    }
    
    if let Some(order_count) = event_store.get_projection("order_count").await {
        tracing::info!("Total orders: {}", order_count);
    }
    
    if let Some(revenue) = event_store.get_projection("total_revenue").await {
        tracing::info!("Total revenue: ${}", revenue);
    }
    
    Ok(())
}
```

---

## Performance Optimization

### 1. High-Throughput Configuration

```rust
use csf_bus::{PhaseCoherenceBus, BusConfig, PhasePacket};
use csf_core::prelude::*;
use std::sync::Arc;

/// Optimized configuration for high-throughput scenarios
async fn create_high_throughput_bus() -> Result<Arc<PhaseCoherenceBus>, Box<dyn std::error::Error>> {
    let config = BusConfig {
        channel_buffer_size: 8192, // Large buffers to reduce backpressure
    };
    
    let bus = Arc::new(PhaseCoherenceBus::new(config)?);
    
    // Enable quantum optimization for better routing
    bus.set_quantum_optimization(true);
    
    Ok(bus)
}

/// Batch publishing for maximum throughput
async fn batch_publish_example(
    bus: &PhaseCoherenceBus,
) -> Result<(), Box<dyn std::error::Error>> {
    // Collect messages into batches
    let mut batch = Vec::new();
    
    for i in 0..1000 {
        let message = HighThroughputMessage {
            id: i,
            payload: format!("message-{}", i),
            timestamp: csf_time::now()?.as_nanos(),
        };
        
        let packet = PhasePacket::new(message, ComponentId::custom(700))
            .with_priority(Priority::Normal)
            .with_delivery_options(csf_bus::DeliveryOptions {
                guaranteed_delivery: false, // Allow drops under extreme load
                use_hardware_acceleration: true,
                simd_flags: 0xFF,
                ..Default::default()
            });
        
        batch.push(packet);
        
        // Publish in batches of 100
        if batch.len() >= 100 {
            let message_ids = bus.publish_batch(batch.drain(..).collect()).await?;
            tracing::debug!("Published batch of {} messages", message_ids.len());
        }
    }
    
    // Publish remaining messages
    if !batch.is_empty() {
        bus.publish_batch(batch).await?;
    }
    
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HighThroughputMessage {
    id: u32,
    payload: String,
    timestamp: u64,
}
```

### 2. Low-Latency Optimization

```rust
use csf_time::QuantumOffset;

/// Ultra-low latency message processing
async fn low_latency_processing(
    bus: &PhaseCoherenceBus,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use quantum optimization for minimal latency
    let quantum_offset = QuantumOffset::new(
        0.2,    // Low amplitude for speed
        1.5,    // Higher frequency for responsiveness
        1800.0  // Optimal phase for minimal latency
    );
    
    let critical_message = CriticalMessage {
        alert_type: "SYSTEM_FAILURE".to_string(),
        severity: 10,
        details: "Critical system component failure detected".to_string(),
    };
    
    let packet = PhasePacket::with_quantum_optimization(critical_message, quantum_offset)
        .with_priority(Priority::Critical)
        .with_guaranteed_delivery(0) // No retries for minimum latency
        .with_timeout(100_000); // 100μs timeout
    
    let start_time = csf_time::now()?;
    bus.publish(packet).await?;
    let end_time = csf_time::now()?;
    
    let latency_ns = end_time.as_nanos() - start_time.as_nanos();
    
    if latency_ns > 1_000 { // 1μs target
        tracing::warn!("Latency target exceeded: {}ns", latency_ns);
    } else {
        tracing::info!("Low-latency publish successful: {}ns", latency_ns);
    }
    
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CriticalMessage {
    alert_type: String,
    severity: u8,
    details: String,
}
```

### 3. Memory-Efficient Processing

```rust
/// Memory-efficient large message handling
async fn memory_efficient_processing(
    bus: &PhaseCoherenceBus,
) -> Result<(), Box<dyn std::error::Error>> {
    // For large payloads, use reference counting to avoid copies
    let large_data = Arc::new(vec![0u8; 1_000_000]); // 1MB payload
    
    // Create message with Arc payload to avoid copying
    let large_message = LargeDataMessage {
        data_ref: large_data.clone(),
        metadata: "Processing batch data".to_string(),
    };
    
    let packet = PhasePacket::new(large_message, ComponentId::custom(800))
        .with_delivery_options(csf_bus::DeliveryOptions {
            use_hardware_acceleration: true,
            guaranteed_delivery: false, // Reduce memory pressure
            ..Default::default()
        });
    
    // The Arc ensures zero-copy delivery to multiple subscribers
    bus.publish(packet).await?;
    
    tracing::info!("Large message published with zero-copy semantics");
    
    Ok(())
}

#[derive(Debug, Clone)]
struct LargeDataMessage {
    data_ref: Arc<Vec<u8>>,
    metadata: String,
}

// Manual Serialize/Deserialize implementation for Arc<Vec<u8>>
impl serde::Serialize for LargeDataMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("LargeDataMessage", 2)?;
        state.serialize_field("data_ref", &**self.data_ref)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for LargeDataMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
        
        struct LargeDataMessageVisitor;
        
        impl<'de> Visitor<'de> for LargeDataMessageVisitor {
            type Value = LargeDataMessage;
            
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct LargeDataMessage")
            }
            
            fn visit_map<V>(self, mut map: V) -> Result<LargeDataMessage, V::Error>
            where V: MapAccess<'de> {
                let mut data_ref = None;
                let mut metadata = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        "data_ref" => {
                            if data_ref.is_some() {
                                return Err(de::Error::duplicate_field("data_ref"));
                            }
                            data_ref = Some(Arc::new(map.next_value()?));
                        }
                        "metadata" => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                
                let data_ref = data_ref.ok_or_else(|| de::Error::missing_field("data_ref"))?;
                let metadata = metadata.ok_or_else(|| de::Error::missing_field("metadata"))?;
                
                Ok(LargeDataMessage { data_ref, metadata })
            }
        }
        
        const FIELDS: &'static [&'static str] = &["data_ref", "metadata"];
        deserializer.deserialize_struct("LargeDataMessage", FIELDS, LargeDataMessageVisitor)
    }
}
```

---

## Production Deployment

### 1. Docker Deployment

Create a `Dockerfile` for your ChronoFabric application:

```dockerfile
# Multi-stage build for optimal image size
FROM rust:1.75-slim-bullseye AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy source code
COPY . .

# Build with optimizations
RUN cargo build --release --features "hardware-acceleration"

# Runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false -m -d /app chronofabric

# Copy binary
COPY --from=builder /app/target/release/your-app /app/your-app

# Set permissions
RUN chown chronofabric:chronofabric /app/your-app

# Switch to app user
USER chronofabric
WORKDIR /app

# Expose ports (adjust as needed)
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["./your-app"]
```

### 2. Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chronofabric-app
  labels:
    app: chronofabric
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chronofabric
  template:
    metadata:
      labels:
        app: chronofabric
    spec:
      containers:
      - name: chronofabric
        image: your-registry/chronofabric-app:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: CHRONOFABRIC_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: chronofabric-config

---
apiVersion: v1
kind: Service
metadata:
  name: chronofabric-service
spec:
  selector:
    app: chronofabric
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: chronofabric-config
data:
  config.toml: |
    [bus]
    channel_buffer_size = 2048
    
    [time]
    enable_hardware_timing = true
    quantum_optimization = true
    
    [performance]
    target_latency_ns = 1000
    target_throughput_mps = 1000000
```

### 3. Production Configuration

```toml
# config/production.toml
[bus]
channel_buffer_size = 8192
enable_hardware_acceleration = true

[time]
enable_hardware_timing = true
quantum_optimization = true
tsc_calibration_interval_ms = 60000

[performance]
target_latency_ns = 1000
target_throughput_mps = 1000000
health_check_interval_ms = 5000

[monitoring]
enable_metrics = true
prometheus_endpoint = "0.0.0.0:9090"
tracing_endpoint = "http://jaeger:14268"
log_level = "info"

[clustering]
node_id = "${CHRONOFABRIC_NODE_ID}"
discovery_method = "kubernetes"
cluster_name = "production"

[security]
enable_tls = true
cert_file = "/etc/ssl/certs/chronofabric.crt"
key_file = "/etc/ssl/private/chronofabric.key"
ca_file = "/etc/ssl/certs/ca.crt"
```

### 4. Production Application Template

```rust
use csf_bus::{PhaseCoherenceBus, BusConfig};
use csf_time::initialize_global_time_source;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into())
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    info!("Starting ChronoFabric application");
    
    // Load configuration
    let config = load_config().await?;
    
    // Initialize ChronoFabric systems
    initialize_global_time_source()?;
    
    let bus = Arc::new(PhaseCoherenceBus::new(BusConfig {
        channel_buffer_size: config.bus.channel_buffer_size,
    })?);
    
    info!("ChronoFabric bus initialized");
    
    // Start application services
    let services = start_services(bus.clone(), &config).await?;
    
    // Start health check and metrics endpoints
    let health_server = start_health_server(&config).await?;
    let metrics_server = start_metrics_server(bus.clone(), &config).await?;
    
    info!("Application started successfully");
    
    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received, gracefully shutting down...");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
    
    // Graceful shutdown
    shutdown_services(services).await?;
    
    // Final health check
    let final_stats = bus.get_stats();
    info!(
        "Final statistics: {} messages processed, avg latency {}ns",
        final_stats.packets_delivered,
        final_stats.avg_latency_ns
    );
    
    info!("Application shutdown complete");
    Ok(())
}

async fn load_config() -> Result<AppConfig, Box<dyn std::error::Error>> {
    // Load from environment variables and config files
    Ok(AppConfig::default())
}

async fn start_services(
    bus: Arc<PhaseCoherenceBus>,
    config: &AppConfig,
) -> Result<Vec<tokio::task::JoinHandle<()>>, Box<dyn std::error::Error>> {
    let mut services = Vec::new();
    
    // Start your application-specific services here
    // services.push(start_user_service(bus.clone()).await?);
    // services.push(start_order_service(bus.clone()).await?);
    // services.push(start_notification_service(bus.clone()).await?);
    
    Ok(services)
}

async fn start_health_server(
    config: &AppConfig,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    use warp::Filter;
    
    let health = warp::path("health")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})));
    
    let ready = warp::path("ready")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({"status": "ready"})));
    
    let routes = health.or(ready);
    
    let handle = tokio::spawn(async move {
        warp::serve(routes)
            .run(([0, 0, 0, 0], 8080))
            .await;
    });
    
    Ok(handle)
}

async fn start_metrics_server(
    bus: Arc<PhaseCoherenceBus>,
    config: &AppConfig,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    use warp::Filter;
    
    let metrics = warp::path("metrics")
        .and(warp::get())
        .map(move || {
            let stats = bus.get_stats();
            let metrics_text = format!(
                "# HELP chronofabric_packets_published_total Total packets published\n\
                 # TYPE chronofabric_packets_published_total counter\n\
                 chronofabric_packets_published_total {}\n\
                 # HELP chronofabric_avg_latency_ns Average latency in nanoseconds\n\
                 # TYPE chronofabric_avg_latency_ns gauge\n\
                 chronofabric_avg_latency_ns {}\n\
                 # HELP chronofabric_throughput_mps Current throughput in messages per second\n\
                 # TYPE chronofabric_throughput_mps gauge\n\
                 chronofabric_throughput_mps {}\n",
                stats.packets_published,
                stats.avg_latency_ns,
                stats.throughput_mps
            );
            
            warp::reply::with_header(
                metrics_text,
                "content-type",
                "text/plain; version=0.0.4"
            )
        });
    
    let handle = tokio::spawn(async move {
        warp::serve(metrics)
            .run(([0, 0, 0, 0], 9090))
            .await;
    });
    
    Ok(handle)
}

async fn shutdown_services(
    services: Vec<tokio::task::JoinHandle<()>>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Shutting down {} services", services.len());
    
    // Gracefully shutdown all services
    for (i, service) in services.into_iter().enumerate() {
        service.abort();
        info!("Service {} shutdown complete", i);
    }
    
    Ok(())
}

#[derive(Debug, Clone)]
struct AppConfig {
    pub bus: BusConfiguration,
    pub monitoring: MonitoringConfig,
    pub clustering: ClusterConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bus: BusConfiguration {
                channel_buffer_size: 2048,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                prometheus_port: 9090,
            },
            clustering: ClusterConfig {
                node_id: std::env::var("CHRONOFABRIC_NODE_ID")
                    .unwrap_or_else(|_| "node-1".to_string()),
            },
        }
    }
}

#[derive(Debug, Clone)]
struct BusConfiguration {
    channel_buffer_size: usize,
}

#[derive(Debug, Clone)]
struct MonitoringConfig {
    enable_metrics: bool,
    prometheus_port: u16,
}

#[derive(Debug, Clone)]
struct ClusterConfig {
    node_id: String,
}
```

---

This comprehensive integration guide provides practical examples and production-ready patterns for implementing ARES ChronoFabric in real-world applications. The examples demonstrate key concepts like event-driven architecture, distributed state management, stream processing, and production deployment strategies.

**Document Information**:
- **Generated**: August 26, 2025
- **Generator**: Claude Code (Sonnet 4) 
- **Project**: ARES ChronoFabric v0.1.0
- **License**: Proprietary - ARES Systems