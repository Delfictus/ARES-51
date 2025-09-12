//! Rules of Engagement (ROE) lifecycle management and reporting

use crate::{EnterpriseError, EnterpriseResult, RoeConfig};
use csf_sil::SilCore;
use csf_time::{hardware_timestamp, NanoTime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Rules of Engagement manager
pub struct RoeManager {
    config: RoeConfig,
    sil_core: Arc<SilCore>,
    active_rules: Arc<RwLock<HashMap<Uuid, RuleOfEngagement>>>,
    violations: Arc<RwLock<Vec<RoeViolation>>>,
    audit_log: Arc<RwLock<Vec<RoeAuditEntry>>>,
}

impl RoeManager {
    /// Create new ROE manager
    pub async fn new(config: RoeConfig, sil_core: Arc<SilCore>) -> EnterpriseResult<Self> {
        let manager = Self {
            config,
            sil_core,
            active_rules: Arc::new(RwLock::new(HashMap::new())),
            violations: Arc::new(RwLock::new(Vec::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        };

        // Load default ROE
        manager.load_default_rules().await?;
        
        Ok(manager)
    }

    /// Start ROE manager
    pub async fn start(&self) -> EnterpriseResult<()> {
        tracing::info!("Starting Rules of Engagement manager");
        
        // Start compliance monitoring
        self.start_compliance_monitoring().await;
        
        Ok(())
    }

    /// Stop ROE manager
    pub async fn stop(&self) -> EnterpriseResult<()> {
        tracing::info!("Stopping Rules of Engagement manager");
        Ok(())
    }
    
    /// Get current authenticated user from context
    async fn get_current_user(&self) -> Option<String> {
        // In a production system, this would extract from auth context
        // For now, check environment or use default
        std::env::var("ARES_USER").ok().or_else(|| {
            std::env::var("USER").ok()
        })
    }

    /// Create new rule of engagement
    pub async fn create_rule(
        &self,
        name: String,
        description: String,
        rule_config: RuleConfig,
    ) -> EnterpriseResult<RuleOfEngagement> {
        let rule_id = Uuid::new_v4();
        
        let rule = RuleOfEngagement {
            id: rule_id,
            name: name.clone(),
            description,
            config: rule_config,
            status: RuleStatus::Active,
            created_at: hardware_timestamp(),
            created_by: self.get_current_user().await.unwrap_or_else(|| "system".to_string()),
            last_modified: hardware_timestamp(),
            version: 1,
            violations_count: 0,
        };

        // Store rule
        self.active_rules.write().await.insert(rule_id, rule.clone());
        
        // Log creation in SIL
        self.log_roe_action(RoeAction::Created {
            rule_id,
            rule_name: name,
        }).await?;

        // Add audit entry
        self.add_audit_entry(AuditAction::RuleCreated {
            rule_id,
            rule_name: rule.name.clone(),
        }).await;

        tracing::info!("Created ROE: {} ({})", rule.name, rule_id);
        
        Ok(rule)
    }

    /// Modify existing rule
    pub async fn modify_rule(
        &self,
        rule_id: Uuid,
        modifications: RuleModifications,
    ) -> EnterpriseResult<RuleOfEngagement> {
        let mut rules = self.active_rules.write().await;
        let rule = rules.get_mut(&rule_id).ok_or_else(|| {
            EnterpriseError::RoeViolation {
                rule: format!("Rule {}", rule_id),
                details: "Rule not found".to_string(),
            }
        })?;

        // Apply modifications
        if let Some(name) = modifications.name {
            rule.name = name;
        }
        if let Some(description) = modifications.description {
            rule.description = description;
        }
        if let Some(config) = modifications.config {
            rule.config = config;
        }
        if let Some(status) = modifications.status {
            rule.status = status;
        }

        rule.last_modified = hardware_timestamp();
        rule.version += 1;

        let updated_rule = rule.clone();

        // Log modification in SIL
        self.log_roe_action(RoeAction::Modified {
            rule_id,
            rule_name: updated_rule.name.clone(),
            changes: serde_json::to_string(&modifications).unwrap_or_else(|_| "unknown".to_string()),
        }).await?;

        // Add audit entry
        self.add_audit_entry(AuditAction::RuleModified {
            rule_id,
            rule_name: updated_rule.name.clone(),
            version: updated_rule.version,
        }).await;

        Ok(updated_rule)
    }

    /// Activate or deactivate rule
    pub async fn toggle_rule(&self, rule_id: Uuid, active: bool) -> EnterpriseResult<()> {
        let mut rules = self.active_rules.write().await;
        let rule = rules.get_mut(&rule_id).ok_or_else(|| {
            EnterpriseError::RoeViolation {
                rule: format!("Rule {}", rule_id),
                details: "Rule not found".to_string(),
            }
        })?;

        let old_status = rule.status.clone();
        rule.status = if active { RuleStatus::Active } else { RuleStatus::Inactive };
        rule.last_modified = hardware_timestamp();

        // Log status change in SIL
        self.log_roe_action(RoeAction::StatusChanged {
            rule_id,
            rule_name: rule.name.clone(),
            old_status: format!("{:?}", old_status),
            new_status: format!("{:?}", rule.status),
        }).await?;

        Ok(())
    }

    /// Check operation against all active ROE
    pub async fn check_compliance(
        &self,
        operation: &Operation,
    ) -> EnterpriseResult<ComplianceResult> {
        let rules = self.active_rules.read().await;
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        for rule in rules.values() {
            if rule.status != RuleStatus::Active {
                continue;
            }

            let result = self.evaluate_rule_compliance(&rule.config, operation).await?;
            
            match result.compliance_level {
                ComplianceLevel::Compliant => {},
                ComplianceLevel::Warning => {
                    warnings.push(ComplianceWarning {
                        rule_id: rule.id,
                        rule_name: rule.name.clone(),
                        message: result.message,
                        severity: WarningSeverity::Low,
                    });
                },
                ComplianceLevel::Violation => {
                    let violation = RoeViolation {
                        id: Uuid::new_v4(),
                        rule_id: rule.id,
                        rule_name: rule.name.clone(),
                        operation_id: operation.id,
                        violation_type: ViolationType::RuleViolation,
                        description: result.message,
                        severity: ViolationSeverity::Medium,
                        detected_at: hardware_timestamp(),
                        resolved: false,
                    };
                    
                    violations.push(violation.clone());
                    self.violations.write().await.push(violation);
                },
            }
        }

        let overall_compliance = if violations.is_empty() {
            if warnings.is_empty() {
                ComplianceLevel::Compliant
            } else {
                ComplianceLevel::Warning
            }
        } else {
            ComplianceLevel::Violation
        };

        Ok(ComplianceResult {
            overall_compliance,
            violations,
            warnings,
            checked_rules: rules.len(),
            timestamp: hardware_timestamp(),
        })
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(
        &self,
        report_type: ReportType,
        time_range: Option<std::time::Duration>,
    ) -> EnterpriseResult<ComplianceReport> {
        let end_time = hardware_timestamp();
        let start_time = if let Some(range) = time_range {
            end_time - csf_time::Duration::from_std(range).unwrap()
        } else {
            end_time - csf_time::Duration::from_secs(86400) // 24 hours default
        };

        let violations = self.violations.read().await;
        let audit_entries = self.audit_log.read().await;
        
        // Filter by time range
        let period_violations: Vec<_> = violations.iter()
            .filter(|v| v.detected_at >= start_time && v.detected_at <= end_time)
            .cloned()
            .collect();
            
        let period_audit_entries: Vec<_> = audit_entries.iter()
            .filter(|e| e.timestamp >= start_time && e.timestamp <= end_time)
            .cloned()
            .collect();

        // Generate statistics
        let stats = ComplianceStats {
            total_operations: period_audit_entries.len(),
            compliant_operations: period_audit_entries.len() - period_violations.len(),
            violations: period_violations.len(),
            compliance_rate: if period_audit_entries.is_empty() {
                1.0
            } else {
                (period_audit_entries.len() - period_violations.len()) as f64 / period_audit_entries.len() as f64
            },
            most_violated_rules: self.calculate_most_violated_rules(&period_violations),
        };

        let report = ComplianceReport {
            report_type,
            time_range: end_time - start_time,
            generated_at: hardware_timestamp(),
            statistics: stats,
            violations: period_violations,
            audit_entries: period_audit_entries,
            recommendations: self.generate_recommendations(&period_violations).await,
        };

        Ok(report)
    }

    /// Load default ROE rules
    async fn load_default_rules(&self) -> EnterpriseResult<()> {
        // Standard Operations ROE
        let standard_roe = RuleConfig {
            engagement_level: EngagementLevel::Standard,
            authorized_actions: vec![
                "data_processing".to_string(),
                "analysis".to_string(),
                "reporting".to_string(),
            ],
            restrictions: vec![
                "no_external_network".to_string(),
                "data_retention_limit".to_string(),
            ],
            escalation_thresholds: self.config.escalation_thresholds.clone(),
            approval_required: false,
            max_processing_time: std::time::Duration::from_secs(3600),
            data_classification_limits: vec!["unclassified".to_string(), "internal".to_string()],
        };

        self.create_rule(
            "Standard Operations".to_string(),
            "Standard operational procedures for routine data processing".to_string(),
            standard_roe,
        ).await?;

        // Emergency Response ROE
        let emergency_roe = RuleConfig {
            engagement_level: EngagementLevel::Emergency,
            authorized_actions: vec![
                "emergency_processing".to_string(),
                "priority_analysis".to_string(),
                "immediate_response".to_string(),
                "external_notification".to_string(),
            ],
            restrictions: vec![],
            escalation_thresholds: vec![0.5, 0.8, 0.95],
            approval_required: true,
            max_processing_time: std::time::Duration::from_secs(300),
            data_classification_limits: vec!["unclassified".to_string(), "confidential".to_string()],
        };

        self.create_rule(
            "Emergency Response".to_string(),
            "Emergency procedures for critical situations requiring immediate response".to_string(),
            emergency_roe,
        ).await?;

        Ok(())
    }

    /// Start compliance monitoring background task
    async fn start_compliance_monitoring(&self) {
        let violations = self.violations.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Monitor for patterns in violations
                let current_violations = violations.read().await;
                
                // Check for escalation conditions
                let recent_violations: Vec<_> = current_violations.iter()
                    .filter(|v| {
                        let age = hardware_timestamp() - v.detected_at;
                        age.as_secs() < 3600 // Last hour
                    })
                    .collect();

                if recent_violations.len() > 5 {
                    tracing::warn!("High violation rate detected: {} violations in last hour", recent_violations.len());
                }
            }
        });
    }

    /// Evaluate rule compliance for operation
    async fn evaluate_rule_compliance(
        &self,
        rule_config: &RuleConfig,
        operation: &Operation,
    ) -> EnterpriseResult<RuleEvaluation> {
        // Check if action is authorized
        if !rule_config.authorized_actions.contains(&operation.action_type) {
            return Ok(RuleEvaluation {
                compliance_level: ComplianceLevel::Violation,
                message: format!("Action '{}' not authorized by ROE", operation.action_type),
            });
        }

        // Check data classification limits
        if let Some(classification) = &operation.data_classification {
            if !rule_config.data_classification_limits.contains(classification) {
                return Ok(RuleEvaluation {
                    compliance_level: ComplianceLevel::Violation,
                    message: format!("Data classification '{}' exceeds ROE limits", classification),
                });
            }
        }

        // Check processing time limits
        if let Some(estimated_duration) = operation.estimated_duration {
            if estimated_duration > rule_config.max_processing_time {
                return Ok(RuleEvaluation {
                    compliance_level: ComplianceLevel::Warning,
                    message: format!("Estimated duration exceeds recommended limit"),
                });
            }
        }

        // Check restrictions
        for restriction in &rule_config.restrictions {
            if operation.flags.contains(restriction) {
                return Ok(RuleEvaluation {
                    compliance_level: ComplianceLevel::Violation,
                    message: format!("Operation violates restriction: {}", restriction),
                });
            }
        }

        Ok(RuleEvaluation {
            compliance_level: ComplianceLevel::Compliant,
            message: "Operation compliant with ROE".to_string(),
        })
    }

    /// Log ROE action to SIL
    async fn log_roe_action(&self, action: RoeAction) -> EnterpriseResult<()> {
        let log_data = serde_json::to_vec(&action).map_err(|e| {
            EnterpriseError::Internal {
                details: format!("Failed to serialize ROE action: {}", e),
            }
        })?;

        let packet_id = csf_shared_types::PacketId::new_v4();
        
        self.sil_core.commit(packet_id, &log_data).await.map_err(|e| {
            EnterpriseError::Internal {
                details: format!("Failed to log ROE action to SIL: {}", e),
            }
        })?;

        Ok(())
    }

    /// Add audit entry
    async fn add_audit_entry(&self, action: AuditAction) {
        let entry = RoeAuditEntry {
            id: Uuid::new_v4(),
            action,
            timestamp: hardware_timestamp(),
            user: self.get_current_user().await.unwrap_or_else(|| "system".to_string()),
        };

        self.audit_log.write().await.push(entry);
    }

    /// Calculate most violated rules
    fn calculate_most_violated_rules(&self, violations: &[RoeViolation]) -> Vec<(Uuid, String, usize)> {
        let mut counts: HashMap<Uuid, (String, usize)> = HashMap::new();
        
        for violation in violations {
            let entry = counts.entry(violation.rule_id)
                .or_insert((violation.rule_name.clone(), 0));
            entry.1 += 1;
        }

        let mut sorted: Vec<_> = counts.into_iter()
            .map(|(id, (name, count))| (id, name, count))
            .collect();
        
        sorted.sort_by(|a, b| b.2.cmp(&a.2));
        sorted.into_iter().take(5).collect()
    }

    /// Generate recommendations based on violations
    async fn generate_recommendations(&self, violations: &[RoeViolation]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if violations.len() > 10 {
            recommendations.push("Consider reviewing and updating ROE policies due to high violation rate".to_string());
        }

        let high_severity_violations = violations.iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::High | ViolationSeverity::Critical))
            .count();

        if high_severity_violations > 0 {
            recommendations.push("Immediate attention required for high-severity violations".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("ROE compliance is within acceptable parameters".to_string());
        }

        recommendations
    }

    /// Get all active rules
    pub async fn get_active_rules(&self) -> Vec<RuleOfEngagement> {
        self.active_rules.read().await.values().cloned().collect()
    }

    /// Get violations by criteria
    pub async fn get_violations(
        &self,
        since: Option<NanoTime>,
        severity: Option<ViolationSeverity>,
    ) -> Vec<RoeViolation> {
        let violations = self.violations.read().await;
        
        violations.iter()
            .filter(|v| {
                if let Some(since_time) = since {
                    if v.detected_at < since_time {
                        return false;
                    }
                }
                
                if let Some(sev) = &severity {
                    if &v.severity != sev {
                        return false;
                    }
                }
                
                true
            })
            .cloned()
            .collect()
    }
}

/// Rule of Engagement definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleOfEngagement {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub config: RuleConfig,
    pub status: RuleStatus,
    pub created_at: NanoTime,
    pub created_by: String,
    pub last_modified: NanoTime,
    pub version: u32,
    pub violations_count: usize,
}

/// Rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConfig {
    pub engagement_level: EngagementLevel,
    pub authorized_actions: Vec<String>,
    pub restrictions: Vec<String>,
    pub escalation_thresholds: Vec<f64>,
    pub approval_required: bool,
    pub max_processing_time: std::time::Duration,
    pub data_classification_limits: Vec<String>,
}

/// Engagement levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EngagementLevel {
    Passive,
    Standard,
    Elevated,
    Emergency,
    Critical,
}

/// Rule status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RuleStatus {
    Active,
    Inactive,
    Suspended,
    Deprecated,
}

/// Rule modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleModifications {
    pub name: Option<String>,
    pub description: Option<String>,
    pub config: Option<RuleConfig>,
    pub status: Option<RuleStatus>,
}

/// Operation to be checked against ROE
#[derive(Debug, Clone)]
pub struct Operation {
    pub id: Uuid,
    pub action_type: String,
    pub data_classification: Option<String>,
    pub estimated_duration: Option<std::time::Duration>,
    pub flags: Vec<String>,
    pub priority: u8,
    pub initiated_by: String,
}

/// Rule evaluation result
#[derive(Debug)]
struct RuleEvaluation {
    compliance_level: ComplianceLevel,
    message: String,
}

/// Compliance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceLevel {
    Compliant,
    Warning,
    Violation,
}

/// Compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    pub overall_compliance: ComplianceLevel,
    pub violations: Vec<RoeViolation>,
    pub warnings: Vec<ComplianceWarning>,
    pub checked_rules: usize,
    pub timestamp: NanoTime,
}

/// ROE violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoeViolation {
    pub id: Uuid,
    pub rule_id: Uuid,
    pub rule_name: String,
    pub operation_id: Uuid,
    pub violation_type: ViolationType,
    pub description: String,
    pub severity: ViolationSeverity,
    pub detected_at: NanoTime,
    pub resolved: bool,
}

/// Violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    RuleViolation,
    PolicyViolation,
    SecurityViolation,
    ComplianceViolation,
}

/// Violation severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance warning
#[derive(Debug, Clone)]
pub struct ComplianceWarning {
    pub rule_id: Uuid,
    pub rule_name: String,
    pub message: String,
    pub severity: WarningSeverity,
}

/// Warning severity
#[derive(Debug, Clone)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// ROE audit entry
#[derive(Debug, Clone)]
pub struct RoeAuditEntry {
    pub id: Uuid,
    pub action: AuditAction,
    pub timestamp: NanoTime,
    pub user: String,
}

/// Audit actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    RuleCreated { rule_id: Uuid, rule_name: String },
    RuleModified { rule_id: Uuid, rule_name: String, version: u32 },
    RuleDeleted { rule_id: Uuid, rule_name: String },
    ViolationDetected { violation_id: Uuid, rule_id: Uuid },
    ViolationResolved { violation_id: Uuid },
    ComplianceCheck { operation_id: Uuid, result: String },
}

/// ROE actions for SIL logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoeAction {
    Created { rule_id: Uuid, rule_name: String },
    Modified { rule_id: Uuid, rule_name: String, changes: String },
    StatusChanged { rule_id: Uuid, rule_name: String, old_status: String, new_status: String },
    Deleted { rule_id: Uuid, rule_name: String },
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub report_type: ReportType,
    pub time_range: csf_time::Duration,
    pub generated_at: NanoTime,
    pub statistics: ComplianceStats,
    pub violations: Vec<RoeViolation>,
    pub audit_entries: Vec<RoeAuditEntry>,
    pub recommendations: Vec<String>,
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Summary,
    Detailed,
    Audit,
    Executive,
}

/// Compliance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStats {
    pub total_operations: usize,
    pub compliant_operations: usize,
    pub violations: usize,
    pub compliance_rate: f64,
    pub most_violated_rules: Vec<(Uuid, String, usize)>,
}