//! Financial Market Data Integration Example for ARES ChronoFabric
//! 
//! Demonstrates how to feed market data into the system and extract
//! visualization-ready predictions.

use csf_core::prelude::*;
use csf_shared_types::{ComponentId, NanoTime, PacketType};
use std::collections::HashMap;

/// Market data point structure
#[derive(Debug, Clone)]
pub struct MarketDataPoint {
    pub timestamp: u64,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

/// Market prediction output format for visualization
#[derive(Debug, Clone)]
pub struct MarketPrediction {
    pub symbol: String,
    pub timestamp: u64,
    pub predicted_price: f64,
    pub confidence: f64,
    pub trend_direction: TrendDirection,
    pub support_levels: Vec<f64>,
    pub resistance_levels: Vec<f64>,
    pub correlation_map: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Convert market data into RelationalTensor format
pub fn market_data_to_tensor(
    data_points: Vec<MarketDataPoint>,
    window_size: usize,
) -> Result<RelationalTensor<f64>, Box<dyn std::error::Error>> {
    // Create tensor dimensions: [time_steps, features, symbols]
    let num_points = data_points.len();
    let features_per_point = 7; // OHLC + volume + bid/ask
    
    // Flatten data into vector
    let mut tensor_data = Vec::with_capacity(num_points * features_per_point);
    let mut metadata = RelationalMetadata::new();
    
    for (idx, point) in data_points.iter().enumerate() {
        tensor_data.push(point.open);
        tensor_data.push(point.high);
        tensor_data.push(point.low);
        tensor_data.push(point.close);
        tensor_data.push(point.volume);
        tensor_data.push(point.bid);
        tensor_data.push(point.ask);
        
        // Add temporal correlation
        let component_id = ComponentId::custom(idx as u64);
        metadata.set_temporal_correlation(
            component_id,
            NanoTime::from_millis(point.timestamp),
        );
    }
    
    // Set correlation mappings for price relationships
    metadata.add_correlation(0, 3); // Open to Close
    metadata.add_correlation(1, 2); // High to Low
    metadata.add_correlation(5, 6); // Bid to Ask
    
    // Create tensor with shape [time_steps, features]
    let shape = vec![num_points, features_per_point];
    let mut tensor = RelationalTensor::new(tensor_data, shape)?;
    tensor.metadata = metadata;
    tensor.name = Some("MarketData".to_string());
    
    Ok(tensor)
}

/// Create a phase packet for market data transmission
pub fn create_market_packet(
    tensor: RelationalTensor<f64>,
    priority: u8,
) -> PhasePacket {
    // Serialize tensor data
    let tensor_bytes = bincode::serialize(&tensor).unwrap_or_default();
    
    let mut metadata = serde_json::Map::new();
    metadata.insert("data_type".to_string(), serde_json::Value::String("market_tensor".to_string()));
    metadata.insert("coherence".to_string(), serde_json::Value::Number(
        serde_json::Number::from_f64(tensor.metadata.coherence_factor).unwrap_or_default()
    ));
    
    PhasePacket {
        header: PacketHeader {
            version: 1,
            packet_id: PacketId::new(),
            packet_type: PacketType::Data,
            priority,
            flags: PacketFlags::RELIABLE | PacketFlags::COMPRESSED,
            timestamp: hardware_timestamp(),
            source_node: ComponentId::custom(100).inner() as u16,
            destination_node: ComponentId::ADP.inner() as u16, // Send to Adaptive Processing
            causality_hash: 0,
            sequence_number: None,
            sequence: 0,
            fragment_count: None,
            payload_size: tensor_bytes.len() as u32,
            checksum: 0,
        },
        payload: PacketPayload {
            data: tensor_bytes,
            metadata,
        },
    }
}

/// Parse prediction results from system output
pub fn parse_prediction_output(
    output_tensor: RelationalTensor<f64>,
    symbol: String,
) -> MarketPrediction {
    // Extract prediction values from tensor
    let data = output_tensor.data.as_slice().unwrap_or(&[]);
    
    // Parse correlation data from metadata
    let mut correlation_map = HashMap::new();
    for (component_id, entanglement) in &output_tensor.metadata.entanglement_map {
        correlation_map.insert(
            format!("component_{}", component_id.inner()),
            *entanglement,
        );
    }
    
    MarketPrediction {
        symbol,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        predicted_price: data.first().copied().unwrap_or(0.0),
        confidence: output_tensor.metadata.coherence_factor,
        trend_direction: if data.len() > 1 && data[1] > 0.5 {
            TrendDirection::Bullish
        } else if data.len() > 1 && data[1] < -0.5 {
            TrendDirection::Bearish
        } else {
            TrendDirection::Neutral
        },
        support_levels: data.iter()
            .skip(2)
            .take(3)
            .copied()
            .collect(),
        resistance_levels: data.iter()
            .skip(5)
            .take(3)
            .copied()
            .collect(),
        correlation_map,
    }
}

/// Main integration workflow
pub async fn process_market_data(
    market_data: Vec<MarketDataPoint>,
) -> Result<MarketPrediction, Box<dyn std::error::Error>> {
    // Step 1: Convert market data to tensor format
    let input_tensor = market_data_to_tensor(market_data.clone(), 20)?;
    
    // Step 2: Create phase packet for transmission
    let packet = create_market_packet(input_tensor, 10);
    
    // Step 3: Send through the system (simplified - actual implementation would use bus)
    // This would normally go through:
    // - csf_bus for routing to ADP component
    // - csf_clogic for adaptive processing
    // - csf_kernel for quantum temporal calculations
    
    // Step 4: Simulate receiving processed output
    let output_tensor = simulate_quantum_processing(input_tensor)?;
    
    // Step 5: Parse output for visualization
    let symbol = market_data.first()
        .map(|d| d.symbol.clone())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    
    let prediction = parse_prediction_output(output_tensor, symbol);
    
    Ok(prediction)
}

/// Simulate quantum processing (placeholder for actual system processing)
fn simulate_quantum_processing(
    mut tensor: RelationalTensor<f64>,
) -> Result<RelationalTensor<f64>, Box<dyn std::error::Error>> {
    // Apply quantum correlations (simplified simulation)
    tensor.metadata.coherence_factor = 0.85;
    
    // Add entanglement data
    for i in 0..5 {
        tensor.metadata.set_entanglement(
            ComponentId::custom(i),
            0.7 + (i as f64) * 0.05,
        );
    }
    
    // Transform data (simplified prediction logic)
    let mut output_data = vec![0.0; 10];
    if let Some(slice) = tensor.data.as_slice() {
        if !slice.is_empty() {
            output_data[0] = slice.last().copied().unwrap_or(0.0) * 1.02; // Predicted price
            output_data[1] = 0.6; // Trend indicator
            output_data[2] = output_data[0] * 0.98; // Support 1
            output_data[3] = output_data[0] * 0.96; // Support 2
            output_data[4] = output_data[0] * 0.94; // Support 3
            output_data[5] = output_data[0] * 1.02; // Resistance 1
            output_data[6] = output_data[0] * 1.04; // Resistance 2
            output_data[7] = output_data[0] * 1.06; // Resistance 3
        }
    }
    
    RelationalTensor::new(output_data, vec![10])
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_conversion() {
        let data = vec![
            MarketDataPoint {
                timestamp: 1000,
                symbol: "BTC/USD".to_string(),
                open: 50000.0,
                high: 51000.0,
                low: 49000.0,
                close: 50500.0,
                volume: 1000.0,
                bid: 50400.0,
                ask: 50600.0,
            },
        ];
        
        let tensor = market_data_to_tensor(data, 1).unwrap();
        assert_eq!(tensor.shape, vec![1, 7]);
    }
}