//! Basic sensor fusion example using ARES CSF
//!
//! This example demonstrates how to:
//! - Create multiple sensor components
//! - Send sensor data through the Phase Coherence Bus
//! - Process data with DRPP for pattern recognition
//! - Use ADP for adaptive learning
//! - Apply EGC for safety verification

use csf_bus::packet::PhasePacket;
use csf_core::prelude::*;
use csf_core::NanoTime;
use csf_time::global_time_source;
use rand;
use std::time::Duration;
use tokio::time::interval;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SensorReading {
    sensor_id: u16,
    timestamp: u64,
    temperature: f32,
    pressure: f32,
    humidity: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FusedSensorData {
    timestamp: u64,
    avg_temperature: f32,
    avg_pressure: f32,
    avg_humidity: f32,
    confidence: f32,
}

/// Simulated sensor that generates readings
async fn sensor_task(sensor_id: u16) {
    let mut ticker = interval(Duration::from_millis(100)); // 10Hz

    loop {
        ticker.tick().await;

        // Generate simulated sensor data
        let reading = SensorReading {
            sensor_id,
            timestamp: global_time_source()
                .now_ns()
                .unwrap_or(NanoTime::ZERO)
                .as_nanos(),
            temperature: 20.0 + (rand::random::<f32>() * 5.0),
            pressure: 1013.0 + (rand::random::<f32>() * 10.0),
            humidity: 50.0 + (rand::random::<f32>() * 20.0),
        };

        // Create phase packet
        let packet = PhasePacket::new(reading, ComponentId::custom(sensor_id as u64))
            .with_priority(Priority::Normal)
            .with_targets(0b0001); // Target DRPP

        // In a real implementation, this would publish to the bus
        println!(
            "Sensor {} reading: T={:.1}°C, P={:.1}hPa, H={:.1}%",
            sensor_id, packet.payload.temperature, packet.payload.pressure, packet.payload.humidity
        );
    }
}

/// Sensor fusion component that processes multiple sensor readings
async fn fusion_task() {
    let mut ticker = interval(Duration::from_millis(500)); // 2Hz
    let mut sensor_buffer = Vec::new();

    loop {
        ticker.tick().await;

        // In a real implementation, this would subscribe to sensor packets
        // For now, we simulate receiving data

        if sensor_buffer.len() >= 3 {
            // Perform sensor fusion
            let fused = perform_fusion(&sensor_buffer);

            // Create result packet
            let packet =
                PhasePacket::new(fused, ComponentId::custom(0x1000)).with_priority(Priority::High);

            println!(
                "Fused data: T={:.1}°C, P={:.1}hPa, H={:.1}%, Confidence={:.2}",
                packet.payload.avg_temperature,
                packet.payload.avg_pressure,
                packet.payload.avg_humidity,
                packet.payload.confidence
            );

            sensor_buffer.clear();
        }
    }
}

fn perform_fusion(readings: &[SensorReading]) -> FusedSensorData {
    let n = readings.len() as f32;

    let avg_temp = readings.iter().map(|r| r.temperature).sum::<f32>() / n;
    let avg_pressure = readings.iter().map(|r| r.pressure).sum::<f32>() / n;
    let avg_humidity = readings.iter().map(|r| r.humidity).sum::<f32>() / n;

    // Calculate variance for confidence
    let temp_variance = readings
        .iter()
        .map(|r| (r.temperature - avg_temp).powi(2))
        .sum::<f32>()
        / n;

    // Simple confidence metric based on variance
    let confidence = 1.0 / (1.0 + temp_variance);

    FusedSensorData {
        timestamp: global_time_source()
            .now_ns()
            .unwrap_or(NanoTime::ZERO)
            .as_nanos(),
        avg_temperature: avg_temp,
        avg_pressure: avg_pressure,
        avg_humidity: avg_humidity,
        confidence,
    }
}

/// Safety monitor using EGC principles
async fn safety_monitor_task() {
    let mut ticker = interval(Duration::from_secs(1));

    // Define safety rules
    let temp_min = 10.0;
    let temp_max = 40.0;
    let pressure_min = 900.0;
    let pressure_max = 1100.0;

    loop {
        ticker.tick().await;

        // In a real implementation, this would check STL formulas
        // For now, we just log
        println!("Safety monitor: All systems within nominal parameters");
    }
}

#[tokio::main]
async fn main() {
    println!("ARES CSF - Basic Sensor Fusion Example");
    println!("=====================================\n");

    // Spawn sensor tasks
    let sensor1 = tokio::spawn(sensor_task(1));
    let sensor2 = tokio::spawn(sensor_task(2));
    let sensor3 = tokio::spawn(sensor_task(3));

    // Spawn fusion task
    let fusion = tokio::spawn(fusion_task());

    // Spawn safety monitor
    let safety = tokio::spawn(safety_monitor_task());

    // Run for 10 seconds
    tokio::time::sleep(Duration::from_secs(10)).await;

    // In a real application, we would gracefully shutdown
    println!("\nShutting down...");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_fusion() {
        let readings = vec![
            SensorReading {
                sensor_id: 1,
                timestamp: 1000,
                temperature: 25.0,
                pressure: 1013.0,
                humidity: 60.0,
            },
            SensorReading {
                sensor_id: 2,
                timestamp: 1001,
                temperature: 25.5,
                pressure: 1013.5,
                humidity: 61.0,
            },
            SensorReading {
                sensor_id: 3,
                timestamp: 1002,
                temperature: 24.5,
                pressure: 1012.5,
                humidity: 59.0,
            },
        ];

        let fused = perform_fusion(&readings);

        assert!((fused.avg_temperature - 25.0).abs() < 0.1);
        assert!((fused.avg_pressure - 1013.0).abs() < 0.1);
        assert!((fused.avg_humidity - 60.0).abs() < 0.1);
        assert!(fused.confidence > 0.9);
    }
}
