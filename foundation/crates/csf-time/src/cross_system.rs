//! Cross-system temporal synchronization for enterprise determinism
//!
//! This module implements enterprise-grade temporal synchronization protocols
//! for maintaining consistency across heterogeneous distributed systems.

use crate::{
    clock::{DistributedCoordinationState, HlcClock},
    consensus::{ConsensusAlgorithm, TemporalConsensusCoordinator},
    distributed::DistributedSynchronizer,
    global_hlc, LogicalTime, TimeError,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Cross-system temporal synchronization coordinator
#[derive(Debug)]
pub struct CrossSystemSynchronizer {
    /// Local system identifier
    system_id: u64,
    /// System-to-system time offset mappings
    system_offsets: Arc<RwLock<HashMap<u64, SystemTimeOffset>>>,
    /// Cross-system synchronization state
    sync_state: Arc<RwLock<CrossSystemSyncState>>,
    /// Distributed synchronizer for local coordination
    local_synchronizer: Arc<DistributedSynchronizer>,
    /// Maximum allowed drift between systems (nanoseconds)
    max_drift_ns: u64,
}

/// Time offset between different systems
#[derive(Debug, Clone)]
pub struct SystemTimeOffset {
    /// Target system identifier
    pub target_system_id: u64,
    /// Time offset in nanoseconds (positive means target is ahead)
    pub offset_ns: i64,
    /// Confidence in the offset measurement (0.0 to 1.0)
    pub confidence: f64,
    /// Last measurement timestamp
    pub measured_at: LogicalTime,
    /// Network round-trip time to target system
    pub rtt_ns: u64,
}

/// Cross-system synchronization state
#[derive(Debug, Clone)]
pub struct CrossSystemSyncState {
    /// Known peer systems
    pub peer_systems: HashMap<u64, PeerSystemInfo>,
    /// Global synchronization epoch across all systems
    pub global_sync_epoch: u64,
    /// Last successful cross-system synchronization
    pub last_global_sync: LogicalTime,
    /// Active synchronization sessions
    pub active_sync_sessions: Vec<SyncSession>,
}

/// Information about a peer system
#[derive(Debug, Clone)]
pub struct PeerSystemInfo {
    /// System identifier
    pub system_id: u64,
    /// System type for protocol compatibility
    pub system_type: SystemType,
    /// Last known time from this system
    pub last_known_time: LogicalTime,
    /// System health status
    pub status: SystemStatus,
    /// Network endpoints for communication
    pub endpoints: Vec<String>,
    /// Supported synchronization protocols
    pub supported_protocols: Vec<SyncProtocol>,
}

/// Type of system for protocol compatibility
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemType {
    /// ARES ChronoFabric native system
    ChronoFabric,
    /// Legacy NTP-based system
    NtpBased,
    /// IEEE 1588 PTP system
    Ptp,
    /// Custom enterprise system
    Custom(String),
}

/// System health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    /// System is operational and synchronized
    Online,
    /// System is operational but with degraded performance
    Degraded,
    /// System is offline or unreachable
    Offline,
    /// System is being synchronized
    Syncing,
}

/// Supported synchronization protocols
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncProtocol {
    /// Native ChronoFabric HLC protocol
    ChronoFabricHlc,
    /// Network Time Protocol (NTP)
    Ntp,
    /// Precision Time Protocol (IEEE 1588)
    Ptp,
    /// Enterprise hybrid protocol
    EnterpriseHybrid,
}

/// Active synchronization session
#[derive(Debug, Clone)]
pub struct SyncSession {
    /// Session identifier
    pub session_id: u64,
    /// Systems participating in this session
    pub participant_systems: Vec<u64>,
    /// Protocol being used for synchronization
    pub protocol: SyncProtocol,
    /// Session start time
    pub started_at: LogicalTime,
    /// Expected completion time
    pub target_completion: LogicalTime,
    /// Current session status
    pub status: SyncSessionStatus,
}

/// Synchronization session status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSessionStatus {
    /// Session is initializing
    Initializing,
    /// Session is actively synchronizing
    Active,
    /// Session completed successfully
    Completed,
    /// Session failed
    Failed,
    /// Session timed out
    TimedOut,
}

impl CrossSystemSynchronizer {
    /// Create new cross-system synchronizer
    pub fn new(
        system_id: u64,
        local_synchronizer: Arc<DistributedSynchronizer>,
        max_drift_ns: u64,
    ) -> Self {
        Self {
            system_id,
            system_offsets: Arc::new(RwLock::new(HashMap::new())),
            sync_state: Arc::new(RwLock::new(CrossSystemSyncState {
                peer_systems: HashMap::new(),
                global_sync_epoch: 0,
                last_global_sync: LogicalTime::zero(system_id),
                active_sync_sessions: Vec::new(),
            })),
            local_synchronizer,
            max_drift_ns,
        }
    }

    /// Register a peer system for cross-system synchronization
    pub async fn register_peer_system(&self, peer_info: PeerSystemInfo) -> Result<(), TimeError> {
        let mut state = self.sync_state.write();
        
        state.peer_systems.insert(peer_info.system_id, peer_info.clone());
        
        info!(
            system_id = self.system_id,
            peer_system_id = peer_info.system_id,
            system_type = ?peer_info.system_type,
            endpoints = ?peer_info.endpoints,
            "Registered peer system for cross-system synchronization"
        );
        
        Ok(())
    }

    /// Measure time offset to peer system
    pub async fn measure_system_offset(&self, target_system_id: u64) -> Result<SystemTimeOffset, TimeError> {
        let start_time = std::time::Instant::now();
        
        // Simulate cross-system time measurement
        // In real implementation, this would use network protocols
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        let rtt_ns = start_time.elapsed().as_nanos() as u64;
        
        let hlc = global_hlc()?;
        let measured_at = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        // Simulate measured offset (in real implementation, this would be protocol-specific)
        let offset_ns = (rtt_ns as i64) / 2; // Simple RTT/2 approximation
        
        let system_offset = SystemTimeOffset {
            target_system_id,
            offset_ns,
            confidence: 0.85, // High confidence for local measurement
            measured_at,
            rtt_ns,
        };
        
        // Store the measured offset
        self.system_offsets.write().insert(target_system_id, system_offset.clone());
        
        debug!(
            system_id = self.system_id,
            target_system_id = target_system_id,
            offset_ns = offset_ns,
            rtt_ns = rtt_ns,
            confidence = system_offset.confidence,
            "Measured cross-system time offset"
        );
        
        Ok(system_offset)
    }

    /// Execute enterprise cross-system synchronization
    pub async fn execute_cross_system_sync(&self, target_systems: Vec<u64>) -> Result<LogicalTime, TimeError> {
        let hlc = global_hlc()?;
        let session_start = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        // Create synchronization session
        let session_id = session_start.physical.wrapping_add(session_start.logical);
        let session = SyncSession {
            session_id,
            participant_systems: target_systems.clone(),
            protocol: SyncProtocol::EnterpriseHybrid,
            started_at: session_start,
            target_completion: LogicalTime::new(
                session_start.physical + 5_000_000_000, // 5 second timeout
                session_start.logical,
                session_start.node_id,
            ),
            status: SyncSessionStatus::Initializing,
        };
        
        // Add session to active sessions
        {
            let mut state = self.sync_state.write();
            state.active_sync_sessions.push(session.clone());
        }
        
        // Measure offsets to all target systems
        let mut measured_offsets = Vec::new();
        for &target_system in &target_systems {
            match self.measure_system_offset(target_system).await {
                Ok(offset) => {
                    if offset.offset_ns.unsigned_abs() > self.max_drift_ns {
                        warn!(
                            system_id = self.system_id,
                            target_system_id = target_system,
                            offset_ns = offset.offset_ns,
                            max_drift_ns = self.max_drift_ns,
                            "Cross-system drift exceeds maximum allowed"
                        );
                    }
                    measured_offsets.push(offset);
                }
                Err(e) => {
                    warn!(
                        system_id = self.system_id,
                        target_system_id = target_system,
                        error = %e,
                        "Failed to measure offset to target system"
                    );
                }
            }
        }
        
        // Calculate synchronized time
        let sync_time = self.calculate_synchronized_time(&measured_offsets).await?;
        
        // Update global sync state
        {
            let mut state = self.sync_state.write();
            state.global_sync_epoch += 1;
            state.last_global_sync = sync_time;
            
            // Update session status
            if let Some(session) = state.active_sync_sessions.iter_mut().find(|s| s.session_id == session_id) {
                session.status = SyncSessionStatus::Completed;
            }
        }
        
        info!(
            system_id = self.system_id,
            session_id = session_id,
            sync_time = %sync_time,
            target_systems = ?target_systems,
            offset_count = measured_offsets.len(),
            "Cross-system synchronization completed successfully"
        );
        
        Ok(sync_time)
    }

    /// Calculate synchronized time from multiple system offsets
    async fn calculate_synchronized_time(&self, offsets: &[SystemTimeOffset]) -> Result<LogicalTime, TimeError> {
        let hlc = global_hlc()?;
        let local_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        if offsets.is_empty() {
            return Ok(local_time);
        }
        
        // Calculate weighted average offset based on confidence
        let total_weight: f64 = offsets.iter().map(|o| o.confidence).sum();
        let weighted_offset: f64 = offsets
            .iter()
            .map(|o| o.offset_ns as f64 * o.confidence)
            .sum::<f64>() / total_weight;
        
        // Apply offset to local time
        let synchronized_physical = if weighted_offset >= 0.0 {
            local_time.physical.saturating_add(weighted_offset as u64)
        } else {
            local_time.physical.saturating_sub((-weighted_offset) as u64)
        };
        
        let synchronized_time = LogicalTime::new(
            synchronized_physical,
            local_time.logical + 1, // Advance logical time for synchronization event
            local_time.node_id,
        );
        
        debug!(
            system_id = self.system_id,
            local_time = %local_time,
            weighted_offset = weighted_offset,
            synchronized_time = %synchronized_time,
            offset_count = offsets.len(),
            "Calculated cross-system synchronized time"
        );
        
        Ok(synchronized_time)
    }

    /// Check cross-system drift and trigger synchronization if needed
    pub async fn check_and_sync_if_needed(&self) -> Result<Option<LogicalTime>, TimeError> {
        // Check if any system has excessive drift
        let excessive_drift_info = {
            let offsets = self.system_offsets.read();
            offsets.values().find(|offset| offset.offset_ns.unsigned_abs() > self.max_drift_ns)
                .map(|offset| (offset.target_system_id, offset.offset_ns))
        };
        
        if let Some((target_system_id, offset_ns)) = excessive_drift_info {
            warn!(
                system_id = self.system_id,
                target_system_id = target_system_id,
                offset_ns = offset_ns,
                max_drift_ns = self.max_drift_ns,
                "Cross-system drift detected, triggering synchronization"
            );
            
            // Get all peer system IDs
            let peer_systems: Vec<u64> = {
                let state = self.sync_state.read();
                state.peer_systems.keys().copied().collect()
            };
            
            // Execute synchronization
            let sync_time = self.execute_cross_system_sync(peer_systems).await?;
            return Ok(Some(sync_time));
        }
        
        Ok(None) // No synchronization needed
    }

    /// Get cross-system synchronization statistics
    pub fn get_sync_stats(&self) -> CrossSystemSyncStats {
        let state = self.sync_state.read();
        let offsets = self.system_offsets.read();
        
        let max_drift = offsets.values()
            .map(|o| o.offset_ns.unsigned_abs())
            .max()
            .unwrap_or(0);
        
        let avg_confidence = if offsets.is_empty() {
            0.0
        } else {
            offsets.values().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64
        };
        
        CrossSystemSyncStats {
            system_id: self.system_id,
            peer_system_count: state.peer_systems.len(),
            max_drift_ns: max_drift,
            avg_offset_confidence: avg_confidence,
            global_sync_epoch: state.global_sync_epoch,
            active_sync_sessions: state.active_sync_sessions.len(),
        }
    }

    /// Cleanup completed and expired synchronization sessions
    pub async fn cleanup_sync_sessions(&self) -> Result<usize, TimeError> {
        let hlc = global_hlc()?;
        let current_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        let mut state = self.sync_state.write();
        let original_count = state.active_sync_sessions.len();
        
        state.active_sync_sessions.retain(|session| {
            match session.status {
                SyncSessionStatus::Completed | SyncSessionStatus::Failed | SyncSessionStatus::TimedOut => false,
                SyncSessionStatus::Initializing | SyncSessionStatus::Active => {
                    // Check if session has timed out
                    !current_time.happens_before(session.target_completion)
                }
            }
        });
        
        let removed_count = original_count - state.active_sync_sessions.len();
        
        if removed_count > 0 {
            debug!(
                system_id = self.system_id,
                removed_count = removed_count,
                current_time = %current_time,
                "Cleaned up completed/expired sync sessions"
            );
        }
        
        Ok(removed_count)
    }
}

/// Statistics for cross-system synchronization monitoring
#[derive(Debug, Clone)]
pub struct CrossSystemSyncStats {
    /// Local system identifier
    pub system_id: u64,
    /// Number of known peer systems
    pub peer_system_count: usize,
    /// Maximum drift detected (nanoseconds)
    pub max_drift_ns: u64,
    /// Average confidence of offset measurements
    pub avg_offset_confidence: f64,
    /// Current global synchronization epoch
    pub global_sync_epoch: u64,
    /// Number of active synchronization sessions
    pub active_sync_sessions: usize,
}

/// Enterprise cross-system temporal coordinator
pub struct EnterpriseTemporalCoordinator {
    /// Cross-system synchronizer
    cross_system_sync: Arc<CrossSystemSynchronizer>,
    /// Consensus coordinator for agreement protocols
    consensus_coordinator: Arc<TemporalConsensusCoordinator>,
    /// Automatic synchronization interval (milliseconds)
    auto_sync_interval_ms: u64,
    /// Background synchronization task handle
    sync_task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl EnterpriseTemporalCoordinator {
    /// Create new enterprise temporal coordinator
    pub fn new(
        system_id: u64,
        local_synchronizer: Arc<DistributedSynchronizer>,
        max_drift_ns: u64,
        auto_sync_interval_ms: u64,
    ) -> Self {
        let cross_system_sync = Arc::new(CrossSystemSynchronizer::new(
            system_id,
            local_synchronizer.clone(),
            max_drift_ns,
        ));
        
        let consensus_coordinator = Arc::new(TemporalConsensusCoordinator::new(
            system_id,
            local_synchronizer,
            ConsensusAlgorithm::EnterpriseHybrid,
        ));
        
        Self {
            cross_system_sync,
            consensus_coordinator,
            auto_sync_interval_ms,
            sync_task_handle: None,
        }
    }

    /// Start automatic cross-system synchronization
    pub async fn start_auto_sync(&mut self) -> Result<(), TimeError> {
        if self.sync_task_handle.is_some() {
            return Err(TimeError::InvalidOperation {
                operation: "start_auto_sync".to_string(),
                reason: "Auto sync already running".to_string(),
            });
        }
        
        let cross_system_sync = Arc::clone(&self.cross_system_sync);
        let interval_ms = self.auto_sync_interval_ms;
        
        let handle = tokio::task::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                match cross_system_sync.check_and_sync_if_needed().await {
                    Ok(Some(sync_time)) => {
                        info!(
                            system_id = cross_system_sync.system_id,
                            sync_time = %sync_time,
                            "Automatic cross-system synchronization completed"
                        );
                    }
                    Ok(None) => {
                        debug!(
                            system_id = cross_system_sync.system_id,
                            "Cross-system synchronization check - no sync needed"
                        );
                    }
                    Err(e) => {
                        warn!(
                            system_id = cross_system_sync.system_id,
                            error = %e,
                            "Automatic cross-system synchronization failed"
                        );
                    }
                }
                
                // Cleanup expired sessions
                if let Err(e) = cross_system_sync.cleanup_sync_sessions().await {
                    warn!(
                        system_id = cross_system_sync.system_id,
                        error = %e,
                        "Failed to cleanup sync sessions"
                    );
                }
            }
        });
        
        self.sync_task_handle = Some(handle);
        
        info!(
            system_id = self.cross_system_sync.system_id,
            interval_ms = interval_ms,
            "Started automatic cross-system synchronization"
        );
        
        Ok(())
    }

    /// Stop automatic synchronization
    pub async fn stop_auto_sync(&mut self) {
        if let Some(handle) = self.sync_task_handle.take() {
            handle.abort();
            
            info!(
                system_id = self.cross_system_sync.system_id,
                "Stopped automatic cross-system synchronization"
            );
        }
    }

    /// Get comprehensive synchronization status
    pub fn get_comprehensive_status(&self) -> ComprehensiveTemporalStatus {
        let cross_system_stats = self.cross_system_sync.get_sync_stats();
        let consensus_stats = self.consensus_coordinator.get_consensus_stats();
        let distributed_state = self.cross_system_sync.local_synchronizer.get_state_snapshot();
        
        ComprehensiveTemporalStatus {
            system_id: self.cross_system_sync.system_id,
            cross_system_stats,
            consensus_stats,
            distributed_state,
            auto_sync_running: self.sync_task_handle.is_some(),
        }
    }
}

/// Comprehensive temporal status for enterprise monitoring
#[derive(Debug, Clone)]
pub struct ComprehensiveTemporalStatus {
    /// System identifier
    pub system_id: u64,
    /// Cross-system synchronization statistics
    pub cross_system_stats: CrossSystemSyncStats,
    /// Consensus statistics
    pub consensus_stats: crate::consensus::ConsensusStats,
    /// Distributed coordination state
    pub distributed_state: DistributedCoordinationState,
    /// Whether auto-sync is currently running
    pub auto_sync_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_simulated_time_source, NanoTime};

    #[tokio::test]
    async fn test_cross_system_synchronizer_creation() {
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let cross_sync = CrossSystemSynchronizer::new(100, local_sync, 1_000_000);
        
        assert_eq!(cross_sync.system_id, 100);
        assert_eq!(cross_sync.max_drift_ns, 1_000_000);
    }

    #[tokio::test]
    async fn test_peer_system_registration() {
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let cross_sync = CrossSystemSynchronizer::new(100, local_sync, 1_000_000);
        
        let peer_info = PeerSystemInfo {
            system_id: 200,
            system_type: SystemType::ChronoFabric,
            last_known_time: LogicalTime::new(1000, 0, 200),
            status: SystemStatus::Online,
            endpoints: vec!["tcp://192.168.1.100:8080".to_string()],
            supported_protocols: vec![SyncProtocol::ChronoFabricHlc, SyncProtocol::EnterpriseHybrid],
        };
        
        cross_sync.register_peer_system(peer_info).await.expect("Should register peer system");
        
        let state = cross_sync.sync_state.read();
        assert!(state.peer_systems.contains_key(&200));
    }

    #[tokio::test]
    async fn test_system_offset_measurement() {
        initialize_simulated_time_source(NanoTime::from_nanos(5000));
        
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let cross_sync = CrossSystemSynchronizer::new(100, local_sync, 1_000_000);
        
        let offset = cross_sync.measure_system_offset(200).await.expect("Should measure offset");
        
        assert_eq!(offset.target_system_id, 200);
        assert!(offset.confidence > 0.0);
        assert!(offset.rtt_ns > 0);
    }

    #[test]
    fn test_enterprise_temporal_coordinator_creation() {
        let local_sync = Arc::new(DistributedSynchronizer::new(1, 5000));
        let coordinator = EnterpriseTemporalCoordinator::new(100, local_sync, 1_000_000, 10000);
        
        assert_eq!(coordinator.cross_system_sync.system_id, 100);
        assert_eq!(coordinator.auto_sync_interval_ms, 10000);
        assert!(coordinator.sync_task_handle.is_none());
    }
}