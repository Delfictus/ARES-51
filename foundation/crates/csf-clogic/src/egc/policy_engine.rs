//! Policy engine for EGC

use csf_core::prelude::*;
use dashmap::DashMap;

/// Policy engine for evaluating and enforcing governance policies
pub struct PolicyEngine {
    /// Active policies
    policies: DashMap<PolicyId, Policy>,

    /// Policy evaluation cache
    cache: DashMap<PacketId, PolicyResult>,

    /// Configuration
    max_policies: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PolicyId(u64);

impl PolicyId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Governance policy
#[derive(Debug, Clone)]
pub struct Policy {
    /// Policy ID
    pub id: PolicyId,

    /// Policy name
    pub name: String,

    /// Policy type
    pub policy_type: PolicyType,

    /// Conditions for policy application
    pub conditions: Vec<PolicyCondition>,

    /// Actions to take when policy matches
    pub actions: Vec<PolicyAction>,

    /// Policy priority (higher = more important)
    pub priority: u8,

    /// Is policy active?
    pub active: bool,

    /// Creation timestamp
    pub created_at: NanoTime,
}

#[derive(Debug, Clone)]
pub enum PolicyType {
    Security,
    Performance,
    Resource,
    Compliance,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct PolicyCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    Matches,
}

#[derive(Debug, Clone)]
pub struct PolicyAction {
    pub action_type: ActionType,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum ActionType {
    Allow,
    Deny,
    Redirect,
    Log,
    Alert,
    Modify,
}

/// Policy evaluation result
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// Is packet compliant with policies?
    pub compliant: bool,

    /// Requires governance decision?
    pub requires_decision: bool,

    /// Applied rules
    pub applied_rules: Vec<PolicyId>,

    /// Decision options if required
    pub options: Vec<super::DecisionOption>,

    /// Actions to execute
    pub actions: Vec<PolicyAction>,
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new(config: &super::EgcConfig) -> Self {
        Self {
            policies: DashMap::new(),
            cache: DashMap::new(),
            max_policies: config.max_policies,
        }
    }

    /// Add a policy
    pub fn add_policy(&self, policy: Policy) -> anyhow::Result<PolicyId> {
        if self.policies.len() >= self.max_policies {
            return Err(anyhow::anyhow!("Maximum number of policies reached"));
        }

        let id = policy.id;
        self.policies.insert(id, policy);
        Ok(id)
    }

    /// Remove a policy
    pub fn remove_policy(&self, id: PolicyId) -> Option<Policy> {
        self.policies.remove(&id).map(|(_, p)| p)
    }

    /// Evaluate packet against policies
    pub async fn evaluate(&self, packet: &PhasePacket) -> anyhow::Result<PolicyResult> {
        // Check cache first
        if let Some(cached) = self.cache.get(&packet.header.packet_id) {
            return Ok(cached.clone());
        }

        let mut result = PolicyResult {
            compliant: true,
            requires_decision: false,
            applied_rules: Vec::new(),
            options: Vec::new(),
            actions: Vec::new(),
        };

        // Collect applicable policies
        let mut applicable_policies: Vec<_> = self
            .policies
            .iter()
            .filter(|entry| entry.value().active)
            .map(|entry| entry.value().clone())
            .collect();

        // Sort by priority (descending)
        applicable_policies.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Evaluate each policy
        for policy in applicable_policies {
            if self.evaluate_conditions(&policy.conditions, packet) {
                result.applied_rules.push(policy.id);

                // Process actions
                for action in &policy.actions {
                    match action.action_type {
                        ActionType::Allow => {
                            result.compliant = true;
                        }
                        ActionType::Deny => {
                            result.compliant = false;
                            result.requires_decision = true;
                        }
                        _ => {
                            result.actions.push(action.clone());
                        }
                    }
                }
            }
        }

        // Generate decision options if needed
        if result.requires_decision {
            result.options = vec![
                super::DecisionOption {
                    id: "allow".to_string(),
                    description: "Allow packet despite policy violation".to_string(),
                    impact: 0.3,
                },
                super::DecisionOption {
                    id: "deny".to_string(),
                    description: "Deny packet per policy".to_string(),
                    impact: 0.1,
                },
                super::DecisionOption {
                    id: "modify".to_string(),
                    description: "Modify packet to comply with policy".to_string(),
                    impact: 0.5,
                },
            ];
        }

        // Cache result
        self.cache.insert(packet.header.packet_id, result.clone());

        Ok(result)
    }

    /// Evaluate policy conditions
    fn evaluate_conditions(&self, conditions: &[PolicyCondition], packet: &PhasePacket) -> bool {
        conditions
            .iter()
            .all(|condition| self.evaluate_condition(condition, packet))
    }

    /// Evaluate single condition
    fn evaluate_condition(&self, condition: &PolicyCondition, packet: &PhasePacket) -> bool {
        // Extract field value from packet
        let field_value = match condition.field.as_str() {
            "priority" => serde_json::json!(packet.header.priority),
            "packet_type" => serde_json::json!(format!("{:?}", packet.header.packet_type)),
            "source" => serde_json::json!(packet.header.source_node),
            "destination" => serde_json::json!(packet.header.destination_node),
            _ => {
                // Check metadata
                packet
                    .payload
                    .metadata
                    .get(&condition.field)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
            }
        };

        // Evaluate operator
        match condition.operator {
            ConditionOperator::Equals => field_value == condition.value,
            ConditionOperator::NotEquals => field_value != condition.value,
            ConditionOperator::GreaterThan => {
                if let (Some(a), Some(b)) = (field_value.as_f64(), condition.value.as_f64()) {
                    a > b
                } else {
                    false
                }
            }
            ConditionOperator::LessThan => {
                if let (Some(a), Some(b)) = (field_value.as_f64(), condition.value.as_f64()) {
                    a < b
                } else {
                    false
                }
            }
            ConditionOperator::Contains => {
                if let (Some(s), Some(pattern)) = (field_value.as_str(), condition.value.as_str()) {
                    s.contains(pattern)
                } else {
                    false
                }
            }
            ConditionOperator::Matches => {
                // Simple pattern matching
                if let (Some(s), Some(pattern)) = (field_value.as_str(), condition.value.as_str()) {
                    s.starts_with(pattern) || s.ends_with(pattern)
                } else {
                    false
                }
            }
        }
    }

    /// Get all active policies
    pub async fn get_active_policies(&self) -> Vec<Policy> {
        self.policies
            .iter()
            .filter(|entry| entry.value().active)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Create default policies
    pub fn create_default_policies(&self) {
        // High priority traffic policy
        self.add_policy(Policy {
            id: PolicyId::new(),
            name: "High Priority Traffic".to_string(),
            policy_type: PolicyType::Performance,
            conditions: vec![PolicyCondition {
                field: "priority".to_string(),
                operator: ConditionOperator::GreaterThan,
                value: serde_json::json!(200),
            }],
            actions: vec![
                PolicyAction {
                    action_type: ActionType::Allow,
                    parameters: serde_json::json!({}),
                },
                PolicyAction {
                    action_type: ActionType::Log,
                    parameters: serde_json::json!({
                        "level": "info",
                        "message": "High priority packet processed"
                    }),
                },
            ],
            priority: 10,
            active: true,
            created_at: hardware_timestamp(),
        })
        .ok();

        // Resource limit policy
        self.add_policy(Policy {
            id: PolicyId::new(),
            name: "Resource Limit".to_string(),
            policy_type: PolicyType::Resource,
            conditions: vec![PolicyCondition {
                field: "payload_size".to_string(),
                operator: ConditionOperator::GreaterThan,
                value: serde_json::json!(1048576), // 1MB
            }],
            actions: vec![
                PolicyAction {
                    action_type: ActionType::Deny,
                    parameters: serde_json::json!({}),
                },
                PolicyAction {
                    action_type: ActionType::Alert,
                    parameters: serde_json::json!({
                        "severity": "warning",
                        "message": "Large payload detected"
                    }),
                },
            ],
            priority: 8,
            active: true,
            created_at: hardware_timestamp(),
        })
        .ok();
    }
}
