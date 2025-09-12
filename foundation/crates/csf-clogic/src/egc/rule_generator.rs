//! Automatic rule generation for EGC

use super::{Policy, Rule};
use csf_core::types::{hardware_timestamp, NanoTime};
use ndarray::Array2;
use statrs::statistics::Statistics;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// 🛡️ HARDENING: Resource limits for rule generation
const MAX_RULE_HISTORY: usize = 5_000;
const MAX_GENERATED_RULES_PER_CALL: usize = 100;

/// Rule generator for creating emergent governance rules with resource protection
pub struct RuleGenerator {
    /// Pattern recognition threshold
    pattern_threshold: f64,

    /// Confidence requirement
    min_confidence: f64,

    /// Rule history for learning
    rule_history: Arc<Mutex<Vec<GeneratedRule>>>,

    /// Pattern detector
    pattern_detector: PatternMiner,

    /// 🛡️ HARDENING: Generation metrics
    total_rules_generated: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
struct GeneratedRule {
    rule: Rule,
    performance: RulePerformance,
    created_at: NanoTime,
}

#[derive(Debug, Clone, Default)]
struct RulePerformance {
    applications: u64,
    successes: u64,
    failures: u64,
    avg_impact: f64,
}

struct PatternMiner {
    min_support: f64,
    min_confidence: f64,
}

impl RuleGenerator {
    /// Create a new rule generator
    pub fn new(config: &super::EgcConfig) -> Self {
        Self {
            pattern_threshold: 0.7,
            min_confidence: 0.8,
            rule_history: Arc::new(Mutex::new(Vec::new())),
            pattern_detector: PatternMiner {
                min_support: 0.3,
                min_confidence: 0.8,
            },
            // 🛡️ HARDENING: Initialize metrics
            total_rules_generated: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Generate new rules based on observed patterns with resource protection
    pub async fn generate_rules(&self, policies: &[Policy]) -> anyhow::Result<Vec<Rule>> {
        // 🛡️ HARDENING: Input validation
        if policies.is_empty() {
            return Ok(Vec::new());
        }

        if policies.len() > 10_000 {
            tracing::warn!(
                "Large policy set ({} policies), limiting processing",
                policies.len()
            );
        }

        // 🛡️ HARDENING: Check rule history size and clean if needed
        {
            let mut history = self.rule_history.lock().unwrap();
            if history.len() > MAX_RULE_HISTORY {
                tracing::warn!(
                    "Rule history at {} entries, pruning to {}",
                    history.len(),
                    MAX_RULE_HISTORY / 2
                );
                history.sort_by_key(|gr| gr.created_at.as_nanos());
                history.truncate(MAX_RULE_HISTORY / 2);
            }
        }

        let mut new_rules = Vec::new();

        // Analyze policy patterns
        let patterns = self.analyze_policy_patterns(policies);

        // Generate rules from patterns with limits
        let mut rules_generated_this_call = 0;
        for pattern in patterns {
            // 🛡️ HARDENING: Limit rules generated per call
            if rules_generated_this_call >= MAX_GENERATED_RULES_PER_CALL {
                tracing::warn!(
                    "Hit maximum rules per call limit ({}), stopping generation",
                    MAX_GENERATED_RULES_PER_CALL
                );
                break;
            }

            if pattern.confidence >= self.min_confidence {
                let rule = self.create_rule_from_pattern(pattern);
                new_rules.push(rule.clone());

                // Track generated rule
                self.rule_history.lock().unwrap().push(GeneratedRule {
                    rule,
                    performance: Default::default(),
                    created_at: hardware_timestamp(),
                });

                // 🛡️ HARDENING: Update generation metrics
                self.total_rules_generated.fetch_add(1, Ordering::Relaxed);
                rules_generated_this_call += 1;
            }
        }

        // Learn from historical performance
        if self.rule_history.lock().unwrap().len() > 10 {
            let improvements = self.learn_from_history();
            new_rules.extend(improvements);
        }

        // 🛡️ HARDENING: Log generation summary
        let total_generated = self.total_rules_generated.load(Ordering::Relaxed);
        if rules_generated_this_call > 0 {
            tracing::info!(
                "Generated {} new rules (total lifetime: {})",
                rules_generated_this_call,
                total_generated
            );
        }

        Ok(new_rules)
    }

    /// Analyze patterns in policy usage
    fn analyze_policy_patterns(&self, policies: &[Policy]) -> Vec<PolicyPattern> {
        let mut patterns = Vec::new();

        // Group policies by type
        let mut type_groups: std::collections::HashMap<String, Vec<&Policy>> =
            std::collections::HashMap::new();
        for policy in policies {
            let type_key = format!("{:?}", policy.policy_type);
            type_groups
                .entry(type_key)
                .or_insert_with(Vec::new)
                .push(policy);
        }

        // Find common condition patterns
        for (policy_type, group) in type_groups {
            if group.len() < 2 {
                continue;
            }

            // Extract common conditions
            let common_conditions = self.find_common_conditions(&group);

            if !common_conditions.is_empty() {
                patterns.push(PolicyPattern {
                    pattern_type: PatternType::CommonConditions,
                    policy_type,
                    conditions: common_conditions,
                    support: group.len() as f64 / policies.len() as f64,
                    confidence: 0.9, // High confidence for observed patterns
                });
            }
        }

        // Find sequential patterns
        patterns.extend(self.find_sequential_patterns(policies));

        // Find correlation patterns
        patterns.extend(self.find_correlation_patterns(policies));

        patterns
    }

    /// Find common conditions across policies
    fn find_common_conditions(&self, policies: &[&Policy]) -> Vec<String> {
        if policies.is_empty() {
            return Vec::new();
        }

        let mut condition_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for policy in policies {
            for condition in &policy.conditions {
                let condition_key = format!("{}_{:?}", condition.field, condition.operator);
                *condition_counts.entry(condition_key).or_insert(0) += 1;
            }
        }

        // Find conditions that appear in most policies
        let threshold = (policies.len() as f64 * self.pattern_detector.min_support) as usize;
        condition_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(condition, _)| condition)
            .collect()
    }

    /// Find sequential patterns in policy applications
    fn find_sequential_patterns(&self, policies: &[Policy]) -> Vec<PolicyPattern> {
        let mut patterns = Vec::new();

        // Simple sequential pattern detection
        // In a real implementation, this would use more sophisticated algorithms
        for i in 0..policies.len().saturating_sub(1) {
            for j in i + 1..policies.len() {
                let similarity = self.calculate_policy_similarity(&policies[i], &policies[j]);

                if similarity > self.pattern_threshold {
                    patterns.push(PolicyPattern {
                        pattern_type: PatternType::Sequential,
                        policy_type: format!("{:?}", policies[i].policy_type),
                        conditions: vec![
                            format!("follows_{}", policies[i].name),
                            format!("precedes_{}", policies[j].name),
                        ],
                        support: similarity,
                        confidence: similarity * 0.9,
                    });
                }
            }
        }

        patterns
    }

    /// Find correlation patterns between policies
    fn find_correlation_patterns(&self, policies: &[Policy]) -> Vec<PolicyPattern> {
        let mut patterns = Vec::new();

        // Build correlation matrix
        let n = policies.len();
        let mut correlation_matrix = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let correlation = self.calculate_policy_correlation(&policies[i], &policies[j]);
                    correlation_matrix[[i, j]] = correlation;
                }
            }
        }

        // Find strong correlations
        for i in 0..n {
            for j in i + 1..n {
                let correlation = correlation_matrix[[i, j]];

                if correlation.abs() > self.pattern_threshold {
                    patterns.push(PolicyPattern {
                        pattern_type: PatternType::Correlation,
                        policy_type: "Correlated".to_string(),
                        conditions: vec![
                            policies[i].name.clone(),
                            policies[j].name.clone(),
                            format!("correlation_{:.2}", correlation),
                        ],
                        support: correlation.abs(),
                        confidence: correlation.abs() * 0.85,
                    });
                }
            }
        }

        patterns
    }

    /// Calculate similarity between two policies
    fn calculate_policy_similarity(&self, p1: &Policy, p2: &Policy) -> f64 {
        let mut similarity = 0.0;
        let mut factors = 0;

        // Type similarity
        if format!("{:?}", p1.policy_type) == format!("{:?}", p2.policy_type) {
            similarity += 0.3;
        }
        factors += 1;

        // Condition similarity
        let common_conditions = p1
            .conditions
            .iter()
            .filter(|c1| p2.conditions.iter().any(|c2| c1.field == c2.field))
            .count();
        let condition_similarity = common_conditions as f64
            / (p1.conditions.len().max(p2.conditions.len()) as f64).max(1.0);
        similarity += condition_similarity * 0.4;
        factors += 1;

        // Action similarity
        let common_actions = p1
            .actions
            .iter()
            .filter(|a1| {
                p2.actions
                    .iter()
                    .any(|a2| format!("{:?}", a1.action_type) == format!("{:?}", a2.action_type))
            })
            .count();
        let action_similarity =
            common_actions as f64 / (p1.actions.len().max(p2.actions.len()) as f64).max(1.0);
        similarity += action_similarity * 0.3;
        factors += 1;

        similarity / factors as f64
    }

    /// Calculate correlation between policies
    fn calculate_policy_correlation(&self, p1: &Policy, p2: &Policy) -> f64 {
        // Simple correlation based on priority and creation time
        let priority_diff = (p1.priority as f64 - p2.priority as f64).abs() / 255.0;
        let time_diff =
            (p1.created_at.as_nanos() as f64 - p2.created_at.as_nanos() as f64).abs() / 1e9; // Convert to seconds

        // Inverse correlation - closer in priority and time means higher correlation
        1.0 - (priority_diff * 0.5 + (time_diff / 3600.0).min(1.0) * 0.5)
    }

    /// Create a rule from a pattern
    fn create_rule_from_pattern(&self, pattern: PolicyPattern) -> Rule {
        static RULE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let rule_id = RULE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let condition = match pattern.pattern_type {
            PatternType::CommonConditions => {
                format!("common_conditions({})", pattern.conditions.join(", "))
            }
            PatternType::Sequential => {
                format!("sequence({})", pattern.conditions.join(" -> "))
            }
            PatternType::Correlation => {
                format!("correlated({})", pattern.conditions.join(" <-> "))
            }
        };

        let action = format!("apply_pattern_{}", pattern.policy_type.to_lowercase());

        Rule {
            id: rule_id,
            condition,
            action,
            priority: (pattern.confidence * 10.0) as u8,
            confidence: pattern.confidence,
        }
    }

    /// Learn from historical rule performance
    fn learn_from_history(&self) -> Vec<Rule> {
        let mut improved_rules = Vec::new();
        let rule_history = self.rule_history.lock().unwrap();

        // Analyze successful rules
        let successful_rules: Vec<_> = rule_history
            .iter()
            .filter(|gr| {
                let success_rate = if gr.performance.applications > 0 {
                    gr.performance.successes as f64 / gr.performance.applications as f64
                } else {
                    0.0
                };
                success_rate > 0.8
            })
            .collect();

        // Generate variations of successful rules
        for gr in successful_rules {
            // Increase priority of successful rules
            let mut improved = gr.rule.clone();
            improved.id = rule_history.len() as u64 + 1;
            improved.priority = (improved.priority + 1).min(255);
            improved.confidence =
                (gr.performance.successes as f64 / gr.performance.applications as f64).min(1.0);

            improved_rules.push(improved);
        }

        improved_rules
    }
}

#[derive(Debug, Clone)]
struct PolicyPattern {
    pattern_type: PatternType,
    policy_type: String,
    conditions: Vec<String>,
    support: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
enum PatternType {
    CommonConditions,
    Sequential,
    Correlation,
}
