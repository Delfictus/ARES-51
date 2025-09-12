//! Intent validation and confirmation system

use crate::{EnterpriseError, EnterpriseResult, IntentConfig};
use csf_protocol::PhasePacket;
use csf_time::hardware_timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Intent validation system
pub struct IntentValidator {
    config: IntentConfig,
    pending_intents: Arc<RwLock<HashMap<Uuid, PendingIntent>>>,
    confirmed_intents: Arc<RwLock<HashMap<Uuid, ConfirmedIntent>>>,
    use_case_templates: HashMap<String, UseCaseTemplate>,
}

impl IntentValidator {
    /// Create new intent validator
    pub fn new(config: IntentConfig) -> EnterpriseResult<Self> {
        let use_case_templates = Self::load_use_case_templates();
        
        Ok(Self {
            config,
            pending_intents: Arc::new(RwLock::new(HashMap::new())),
            confirmed_intents: Arc::new(RwLock::new(HashMap::new())),
            use_case_templates,
        })
    }

    /// Analyze data and generate intent suggestions
    pub async fn analyze_intent(
        &self,
        packets: &[PhasePacket],
        use_case: &str,
    ) -> EnterpriseResult<IntentAnalysis> {
        let analysis_id = Uuid::new_v4();
        
        // Analyze data characteristics
        let data_characteristics = self.analyze_data_characteristics(packets).await?;
        
        // Get use case template
        let template = self.use_case_templates.get(use_case)
            .cloned()
            .unwrap_or_else(|| self.infer_use_case_template(&data_characteristics));

        // Generate suggested actions
        let suggested_actions = self.generate_suggested_actions(&data_characteristics, &template).await?;
        
        // Generate clarifying questions
        let questions = self.generate_clarifying_questions(&data_characteristics, &template, &suggested_actions).await?;
        
        let analysis = IntentAnalysis {
            id: analysis_id,
            use_case: use_case.to_string(),
            data_characteristics,
            suggested_actions,
            clarifying_questions: questions,
            confidence_score: self.calculate_confidence_score(&suggested_actions),
            created_at: hardware_timestamp(),
        };

        // Create pending intent if confidence is below auto-approval threshold
        if analysis.confidence_score < self.config.auto_approve_threshold {
            let pending = PendingIntent {
                id: analysis_id,
                analysis: analysis.clone(),
                status: IntentStatus::AwaitingConfirmation,
                created_at: hardware_timestamp(),
                expires_at: hardware_timestamp() + csf_time::Duration::from_secs(3600), // 1 hour
            };
            
            self.pending_intents.write().await.insert(analysis_id, pending);
        }

        Ok(analysis)
    }

    /// Confirm intent with user responses
    pub async fn confirm_intent(
        &self,
        intent_id: Uuid,
        responses: HashMap<String, String>,
        modifications: Option<HashMap<String, serde_json::Value>>,
    ) -> EnterpriseResult<ConfirmedIntent> {
        let mut pending_intents = self.pending_intents.write().await;
        let pending = pending_intents.remove(&intent_id).ok_or_else(|| {
            EnterpriseError::IntentValidation {
                reason: format!("Intent {} not found or already processed", intent_id),
            }
        })?;

        // Process user responses
        let mut finalized_actions = pending.analysis.suggested_actions.clone();
        
        if let Some(mods) = modifications {
            for (key, value) in mods {
                // Apply modifications to actions
                for action in &mut finalized_actions {
                    if let Some(params) = action.parameters.get_mut(&key) {
                        *params = value.clone();
                    }
                }
            }
        }

        // Validate responses against questions
        let validation_score = self.validate_responses(&pending.analysis.clarifying_questions, &responses)?;
        
        let confirmed = ConfirmedIntent {
            id: intent_id,
            original_analysis: pending.analysis,
            user_responses: responses,
            finalized_actions,
            validation_score,
            confirmed_at: hardware_timestamp(),
            status: if validation_score >= self.config.confidence_threshold {
                ConfirmationStatus::Approved
            } else {
                ConfirmationStatus::RequiresReview
            },
        };

        self.confirmed_intents.write().await.insert(intent_id, confirmed.clone());
        
        Ok(confirmed)
    }

    /// Get pending intents
    pub async fn get_pending_intents(&self) -> Vec<PendingIntent> {
        self.pending_intents.read().await.values().cloned().collect()
    }

    /// Get confirmed intents
    pub async fn get_confirmed_intents(&self, limit: Option<usize>) -> Vec<ConfirmedIntent> {
        let intents: Vec<_> = self.confirmed_intents.read().await.values().cloned().collect();
        
        if let Some(limit) = limit {
            intents.into_iter().take(limit).collect()
        } else {
            intents
        }
    }

    /// Analyze data characteristics
    async fn analyze_data_characteristics(&self, packets: &[PhasePacket]) -> EnterpriseResult<DataCharacteristics> {
        let mut characteristics = DataCharacteristics {
            total_records: packets.len(),
            data_types: HashMap::new(),
            field_coverage: HashMap::new(),
            temporal_patterns: Vec::new(),
            data_quality: 1.0,
            complexity_score: 0.0,
        };

        // Analyze field types and coverage
        for packet in packets {
            for (key, value) in &packet.payload.metadata {
                let field_type = match value {
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                    serde_json::Value::Null => "null",
                };
                
                *characteristics.data_types.entry(field_type.to_string()).or_insert(0) += 1;
                *characteristics.field_coverage.entry(key.clone()).or_insert(0) += 1;
            }
        }

        // Calculate complexity score
        characteristics.complexity_score = self.calculate_complexity_score(&characteristics);

        Ok(characteristics)
    }

    /// Generate suggested actions based on data and template
    async fn generate_suggested_actions(
        &self,
        characteristics: &DataCharacteristics,
        template: &UseCaseTemplate,
    ) -> EnterpriseResult<Vec<SuggestedAction>> {
        let mut actions = Vec::new();

        // Generate actions based on template and data characteristics
        for template_action in &template.default_actions {
            let mut parameters = template_action.default_parameters.clone();
            
            // Customize parameters based on data characteristics
            if characteristics.total_records > 1000 {
                parameters.insert("batch_processing".to_string(), serde_json::Value::Bool(true));
                parameters.insert("batch_size".to_string(), serde_json::Value::Number(serde_json::Number::from(1000)));
            }
            
            if characteristics.complexity_score > 0.7 {
                parameters.insert("advanced_analysis".to_string(), serde_json::Value::Bool(true));
            }

            actions.push(SuggestedAction {
                id: Uuid::new_v4(),
                action_type: template_action.action_type.clone(),
                description: template_action.description.clone(),
                parameters,
                estimated_duration: template_action.estimated_duration,
                confidence: self.calculate_action_confidence(characteristics, template_action),
                dependencies: template_action.dependencies.clone(),
            });
        }

        Ok(actions)
    }

    /// Generate clarifying questions
    async fn generate_clarifying_questions(
        &self,
        characteristics: &DataCharacteristics,
        template: &UseCaseTemplate,
        actions: &[SuggestedAction],
    ) -> EnterpriseResult<Vec<ClarifyingQuestion>> {
        let mut questions = Vec::new();

        // Generate questions based on low-confidence actions
        for action in actions {
            if action.confidence < self.config.confidence_threshold {
                questions.push(ClarifyingQuestion {
                    id: Uuid::new_v4(),
                    question: format!("Do you want to {} with the following parameters: {:?}?", 
                        action.description, action.parameters),
                    question_type: QuestionType::Confirmation,
                    related_action: Some(action.id),
                    options: vec!["yes".to_string(), "no".to_string(), "modify".to_string()],
                    required: true,
                });
            }
        }

        // Add template-specific questions
        for template_question in &template.clarifying_questions {
            if self.should_ask_question(template_question, characteristics) {
                questions.push(ClarifyingQuestion {
                    id: Uuid::new_v4(),
                    question: template_question.question.clone(),
                    question_type: template_question.question_type.clone(),
                    related_action: None,
                    options: template_question.options.clone(),
                    required: template_question.required,
                });
            }
        }

        // Limit number of questions
        questions.truncate(self.config.max_questions);

        Ok(questions)
    }

    /// Calculate confidence score for analysis
    fn calculate_confidence_score(&self, actions: &[SuggestedAction]) -> f64 {
        if actions.is_empty() {
            return 0.0;
        }

        let total_confidence: f64 = actions.iter().map(|a| a.confidence).sum();
        total_confidence / actions.len() as f64
    }

    /// Calculate action confidence
    fn calculate_action_confidence(
        &self,
        characteristics: &DataCharacteristics,
        template_action: &TemplateAction,
    ) -> f64 {
        let mut confidence = 0.8; // Base confidence

        // Adjust based on data quality
        confidence *= characteristics.data_quality;

        // Adjust based on data size appropriateness
        if characteristics.total_records >= template_action.min_records {
            confidence += 0.1;
        } else {
            confidence -= 0.2;
        }

        // Adjust based on complexity match
        let complexity_match = 1.0 - (characteristics.complexity_score - template_action.complexity_requirement).abs();
        confidence *= complexity_match;

        confidence.clamp(0.0, 1.0)
    }

    /// Calculate data complexity score
    fn calculate_complexity_score(&self, characteristics: &DataCharacteristics) -> f64 {
        let mut score = 0.0;

        // Factor in number of unique data types
        score += (characteristics.data_types.len() as f64 / 10.0).min(1.0) * 0.3;

        // Factor in field diversity
        score += (characteristics.field_coverage.len() as f64 / 50.0).min(1.0) * 0.4;

        // Factor in record count
        score += (characteristics.total_records as f64 / 10000.0).min(1.0) * 0.3;

        score.clamp(0.0, 1.0)
    }

    /// Determine if a template question should be asked
    fn should_ask_question(&self, question: &TemplateQuestion, characteristics: &DataCharacteristics) -> bool {
        // Simple logic - ask if data complexity or size meets criteria
        if let Some(min_complexity) = question.min_complexity {
            if characteristics.complexity_score < min_complexity {
                return false;
            }
        }

        if let Some(min_records) = question.min_records {
            if characteristics.total_records < min_records {
                return false;
            }
        }

        true
    }

    /// Validate user responses
    fn validate_responses(
        &self,
        questions: &[ClarifyingQuestion],
        responses: &HashMap<String, String>,
    ) -> EnterpriseResult<f64> {
        let mut total_score = 0.0;
        let mut question_count = 0;

        for question in questions {
            if let Some(response) = responses.get(&question.id.to_string()) {
                let score = match &question.question_type {
                    QuestionType::Confirmation => {
                        if matches!(response.to_lowercase().as_str(), "yes" | "y" | "true") {
                            1.0
                        } else if matches!(response.to_lowercase().as_str(), "no" | "n" | "false") {
                            0.5 // Not necessarily wrong, just different
                        } else {
                            0.8 // Modify response
                        }
                    }
                    QuestionType::Choice => {
                        if question.options.contains(response) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    QuestionType::Numeric => {
                        if response.parse::<f64>().is_ok() {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    QuestionType::Text => 0.8, // Assume reasonable text response
                };
                
                total_score += score;
                question_count += 1;
            } else if question.required {
                return Err(EnterpriseError::IntentValidation {
                    reason: format!("Required question {} not answered", question.id),
                });
            }
        }

        if question_count == 0 {
            Ok(1.0)
        } else {
            Ok(total_score / question_count as f64)
        }
    }

    /// Load use case templates
    fn load_use_case_templates() -> HashMap<String, UseCaseTemplate> {
        let mut templates = HashMap::new();

        // Financial Analytics template
        templates.insert("financial_analytics".to_string(), UseCaseTemplate {
            name: "Financial Analytics".to_string(),
            description: "Advanced financial data analysis and risk assessment".to_string(),
            default_actions: vec![
                TemplateAction {
                    action_type: "statistical_analysis".to_string(),
                    description: "Perform statistical analysis on financial data".to_string(),
                    default_parameters: {
                        let mut params = HashMap::new();
                        params.insert("analysis_type".to_string(), serde_json::Value::String("comprehensive".to_string()));
                        params.insert("risk_metrics".to_string(), serde_json::Value::Bool(true));
                        params
                    },
                    estimated_duration: std::time::Duration::from_secs(300),
                    min_records: 100,
                    complexity_requirement: 0.6,
                    dependencies: vec![],
                },
                TemplateAction {
                    action_type: "pattern_recognition".to_string(),
                    description: "Detect patterns and anomalies in financial data".to_string(),
                    default_parameters: {
                        let mut params = HashMap::new();
                        params.insert("pattern_types".to_string(), serde_json::Value::Array(vec![
                            serde_json::Value::String("trends".to_string()),
                            serde_json::Value::String("anomalies".to_string()),
                            serde_json::Value::String("correlations".to_string()),
                        ]));
                        params
                    },
                    estimated_duration: std::time::Duration::from_secs(600),
                    min_records: 500,
                    complexity_requirement: 0.7,
                    dependencies: vec!["statistical_analysis".to_string()],
                },
            ],
            clarifying_questions: vec![
                TemplateQuestion {
                    question: "What is your primary analysis objective? (risk_assessment, trend_analysis, compliance_check)".to_string(),
                    question_type: QuestionType::Choice,
                    options: vec!["risk_assessment".to_string(), "trend_analysis".to_string(), "compliance_check".to_string()],
                    required: true,
                    min_complexity: Some(0.5),
                    min_records: Some(50),
                },
                TemplateQuestion {
                    question: "What time horizon should be considered for analysis? (days)".to_string(),
                    question_type: QuestionType::Numeric,
                    options: vec![],
                    required: false,
                    min_complexity: None,
                    min_records: None,
                },
            ],
        });

        // Scientific Research template
        templates.insert("scientific_research".to_string(), UseCaseTemplate {
            name: "Scientific Research".to_string(),
            description: "Advanced scientific data analysis and hypothesis testing".to_string(),
            default_actions: vec![
                TemplateAction {
                    action_type: "quantum_analysis".to_string(),
                    description: "Apply quantum-enhanced analysis to research data".to_string(),
                    default_parameters: {
                        let mut params = HashMap::new();
                        params.insert("quantum_depth".to_string(), serde_json::Value::Number(serde_json::Number::from(5)));
                        params.insert("coherence_threshold".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.95).unwrap()));
                        params
                    },
                    estimated_duration: std::time::Duration::from_secs(900),
                    min_records: 200,
                    complexity_requirement: 0.8,
                    dependencies: vec![],
                },
            ],
            clarifying_questions: vec![
                TemplateQuestion {
                    question: "What scientific domain does this data represent? (physics, chemistry, biology, other)".to_string(),
                    question_type: QuestionType::Choice,
                    options: vec!["physics".to_string(), "chemistry".to_string(), "biology".to_string(), "other".to_string()],
                    required: true,
                    min_complexity: Some(0.6),
                    min_records: Some(100),
                },
            ],
        });

        // Defense Intelligence template
        templates.insert("defense_intelligence".to_string(), UseCaseTemplate {
            name: "Defense Intelligence".to_string(),
            description: "Advanced intelligence analysis and threat assessment".to_string(),
            default_actions: vec![
                TemplateAction {
                    action_type: "threat_analysis".to_string(),
                    description: "Comprehensive threat analysis and pattern detection".to_string(),
                    default_parameters: {
                        let mut params = HashMap::new();
                        params.insert("classification_level".to_string(), serde_json::Value::String("unclassified".to_string()));
                        params.insert("threat_vectors".to_string(), serde_json::Value::Array(vec![
                            serde_json::Value::String("cyber".to_string()),
                            serde_json::Value::String("physical".to_string()),
                            serde_json::Value::String("hybrid".to_string()),
                        ]));
                        params
                    },
                    estimated_duration: std::time::Duration::from_secs(1200),
                    min_records: 50,
                    complexity_requirement: 0.9,
                    dependencies: vec![],
                },
            ],
            clarifying_questions: vec![
                TemplateQuestion {
                    question: "What is the data classification level? (unclassified, confidential, secret)".to_string(),
                    question_type: QuestionType::Choice,
                    options: vec!["unclassified".to_string(), "confidential".to_string(), "secret".to_string()],
                    required: true,
                    min_complexity: None,
                    min_records: None,
                },
                TemplateQuestion {
                    question: "What is the analysis urgency level? (routine, priority, immediate)".to_string(),
                    question_type: QuestionType::Choice,
                    options: vec!["routine".to_string(), "priority".to_string(), "immediate".to_string()],
                    required: true,
                    min_complexity: None,
                    min_records: None,
                },
            ],
        });

        templates
    }

    /// Infer use case template from data characteristics
    fn infer_use_case_template(&self, characteristics: &DataCharacteristics) -> UseCaseTemplate {
        // Simple inference logic
        if characteristics.complexity_score > 0.8 {
            self.use_case_templates.get("scientific_research")
                .cloned()
                .unwrap_or_else(|| self.default_template())
        } else if characteristics.data_types.contains_key("number") && characteristics.total_records > 500 {
            self.use_case_templates.get("financial_analytics")
                .cloned()
                .unwrap_or_else(|| self.default_template())
        } else {
            self.default_template()
        }
    }

    /// Default template for unknown use cases
    fn default_template(&self) -> UseCaseTemplate {
        UseCaseTemplate {
            name: "General Analysis".to_string(),
            description: "General purpose data analysis".to_string(),
            default_actions: vec![
                TemplateAction {
                    action_type: "data_analysis".to_string(),
                    description: "Perform general data analysis".to_string(),
                    default_parameters: HashMap::new(),
                    estimated_duration: std::time::Duration::from_secs(180),
                    min_records: 1,
                    complexity_requirement: 0.0,
                    dependencies: vec![],
                },
            ],
            clarifying_questions: vec![
                TemplateQuestion {
                    question: "What specific analysis would you like to perform on this data?".to_string(),
                    question_type: QuestionType::Text,
                    options: vec![],
                    required: false,
                    min_complexity: None,
                    min_records: None,
                },
            ],
        }
    }
}

/// Intent analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAnalysis {
    pub id: Uuid,
    pub use_case: String,
    pub data_characteristics: DataCharacteristics,
    pub suggested_actions: Vec<SuggestedAction>,
    pub clarifying_questions: Vec<ClarifyingQuestion>,
    pub confidence_score: f64,
    pub created_at: csf_time::NanoTime,
}

/// Data characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub total_records: usize,
    pub data_types: HashMap<String, usize>,
    pub field_coverage: HashMap<String, usize>,
    pub temporal_patterns: Vec<String>,
    pub data_quality: f64,
    pub complexity_score: f64,
}

/// Suggested action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedAction {
    pub id: Uuid,
    pub action_type: String,
    pub description: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub estimated_duration: std::time::Duration,
    pub confidence: f64,
    pub dependencies: Vec<String>,
}

/// Clarifying question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarifyingQuestion {
    pub id: Uuid,
    pub question: String,
    pub question_type: QuestionType,
    pub related_action: Option<Uuid>,
    pub options: Vec<String>,
    pub required: bool,
}

/// Question types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    Confirmation,
    Choice,
    Numeric,
    Text,
}

/// Pending intent awaiting confirmation
#[derive(Debug, Clone)]
pub struct PendingIntent {
    pub id: Uuid,
    pub analysis: IntentAnalysis,
    pub status: IntentStatus,
    pub created_at: csf_time::NanoTime,
    pub expires_at: csf_time::NanoTime,
}

/// Intent status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentStatus {
    AwaitingConfirmation,
    UnderReview,
    Expired,
}

/// Confirmed intent ready for execution
#[derive(Debug, Clone)]
pub struct ConfirmedIntent {
    pub id: Uuid,
    pub original_analysis: IntentAnalysis,
    pub user_responses: HashMap<String, String>,
    pub finalized_actions: Vec<SuggestedAction>,
    pub validation_score: f64,
    pub confirmed_at: csf_time::NanoTime,
    pub status: ConfirmationStatus,
}

/// Confirmation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfirmationStatus {
    Approved,
    RequiresReview,
    Rejected,
}

/// Use case template
#[derive(Debug, Clone)]
pub struct UseCaseTemplate {
    pub name: String,
    pub description: String,
    pub default_actions: Vec<TemplateAction>,
    pub clarifying_questions: Vec<TemplateQuestion>,
}

/// Template action
#[derive(Debug, Clone)]
pub struct TemplateAction {
    pub action_type: String,
    pub description: String,
    pub default_parameters: HashMap<String, serde_json::Value>,
    pub estimated_duration: std::time::Duration,
    pub min_records: usize,
    pub complexity_requirement: f64,
    pub dependencies: Vec<String>,
}

/// Template question
#[derive(Debug, Clone)]
pub struct TemplateQuestion {
    pub question: String,
    pub question_type: QuestionType,
    pub options: Vec<String>,
    pub required: bool,
    pub min_complexity: Option<f64>,
    pub min_records: Option<usize>,
}