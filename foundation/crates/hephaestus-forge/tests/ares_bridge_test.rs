'''//! Tests for the ARES Bridge integration

use hephaestus_forge::{
    ares_bridge::{AresSystemBridge, BridgeConfig},
    HephaestusForge, ForgeConfigBuilder, OperationalMode,
};
use std::sync::Arc;

#[tokio::test]
async fn test_ares_bridge_initialization() {
    let config = ForgeConfigBuilder::new()
        .mode(OperationalMode::Autonomous)
        .build()
        .expect("Failed to build config");

    let forge = HephaestusForge::new_async_public(config).await
        .expect("Failed to create forge");

    let bridge_config = BridgeConfig::default();
    let bridge = AresSystemBridge::new(Arc::new(forge), bridge_config).await
        .expect("Failed to create ARES bridge");

    // Verify that the bridge is initialized correctly
    assert_eq!(bridge.get_connected_systems().await.len(), 0);
}
''