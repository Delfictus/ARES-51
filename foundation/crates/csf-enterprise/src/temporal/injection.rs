//! Enterprise TimeSource Dependency Injection
//! 
//! Provides standardized patterns for injecting TimeSource dependencies
//! across enterprise components with proper lifecycle management.

use std::sync::Arc;
use csf_time::TimeSource;
use super::{TemporalMetricsCollector, TemporalAuditLogger};

/// Enterprise temporal dependency container
#[derive(Clone)]
pub struct EnterpriseTemporalContext {
    pub time_source: Arc<dyn TimeSource>,
    pub metrics_collector: Arc<TemporalMetricsCollector>,
    pub audit_logger: Arc<TemporalAuditLogger>,
}

impl EnterpriseTemporalContext {
    pub fn new(time_source: Arc<dyn TimeSource>) -> Self {
        Self {
            time_source,
            metrics_collector: Arc::new(TemporalMetricsCollector::new()),
            audit_logger: Arc::new(TemporalAuditLogger::new()),
        }
    }
    
    pub fn production() -> Self {
        Self::new(Arc::new(csf_time::SystemTimeSource::new()))
    }
    
    pub fn testing() -> Self {
        Self::new(Arc::new(csf_time::MockTimeSource::new()))
    }
}

/// Macro for enterprise temporal operation with automatic audit and metrics
#[macro_export]
macro_rules! enterprise_temporal_op {
    ($ctx:expr, $op_name:expr, $body:block) => {{
        let operation_id = uuid::Uuid::new_v4();
        let start_time = $ctx.time_source.now_ns();
        
        $ctx.audit_logger.log_operation_start(operation_id, $op_name, start_time).await?;
        $ctx.metrics_collector.record_operation_start($op_name, start_time);
        
        let result = $body;
        
        let end_time = $ctx.time_source.now_ns();
        let duration_ns = end_time - start_time;
        
        $ctx.audit_logger.log_operation_complete(
            operation_id, 
            end_time, 
            duration_ns,
            result.is_ok()
        ).await?;
        
        $ctx.metrics_collector.record_operation_complete($op_name, end_time, duration_ns);
        
        result
    }};
}
