//! Neural Language Processor - Natural language understanding using neuromorphic computing

use anyhow::Result;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use csf_clogic::{CLogicSystem, DynamicResonancePatternProcessor, EmotionalModelingSystem};
use super::backend::NeuromorphicBackend;
use super::LearningConfig;

/// Natural language processing through neuromorphic networks
pub struct NeuralLanguageProcessor {
    /// Neuromorphic backend for spike processing
    backend: Arc<RwLock<NeuromorphicBackend>>,
    
    /// C-LOGIC system for pattern recognition
    clogic_system: Arc<CLogicSystem>,
    
    /// Text encoding utilities
    text_encoder: TextToSpikeEncoder,
    
    /// Command pattern library
    command_patterns: Arc<RwLock<CommandPatternLibrary>>,
    
    /// Learning configuration
    learning_config: LearningConfig,
    
    /// Processing metrics
    metrics: Arc<RwLock<NLPMetrics>>,
}

#[derive(Debug, Clone)]
pub struct CommandIntent {
    /// Interpreted command
    pub command: String,
    
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    
    /// Context information
    pub context: CommandContext,
    
    /// Alternative interpretations
    pub alternatives: Vec<AlternativeIntent>,
    
    /// Required confirmations
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone)]
pub struct CommandContext {
    /// Domain classification
    pub domain: CommandDomain,
    
    /// Urgency level (0.0 - 1.0)
    pub urgency: f64,
    
    /// Emotional tone
    pub sentiment: CommandSentiment,
    
    /// Detected parameters
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum CommandDomain {
    System,      // General system commands
    Defense,     // Security/defense operations
    Quantum,     // Quantum system operations
    Learning,    // Learning and training commands
    Status,      // Status and monitoring
    Unknown,     // Unclassified
}

#[derive(Debug, Clone)]
pub struct CommandSentiment {
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f64,
    
    /// Arousal level (0.0 to 1.0)
    pub arousal: f64,
    
    /// Operator emotional state inference
    pub operator_state: OperatorState,
}

#[derive(Debug, Clone)]
pub enum OperatorState {
    Calm,
    Stressed,
    Urgent,
    Learning,
    Troubleshooting,
}

#[derive(Debug, Clone)]
pub struct AlternativeIntent {
    pub command: String,
    pub confidence: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
struct NLPMetrics {
    pub total_processed: u64,
    pub avg_confidence: f64,
    pub domain_distribution: HashMap<String, u64>,
    pub processing_time_ms: f64,
}

impl Default for NLPMetrics {
    fn default() -> Self {
        Self {
            total_processed: 0,
            avg_confidence: 0.0,
            domain_distribution: HashMap::new(),
            processing_time_ms: 0.0,
        }
    }
}

/// Text to spike encoding for neuromorphic processing
pub struct TextToSpikeEncoder {
    /// Vocabulary mapping
    vocab_map: HashMap<String, usize>,
    
    /// Character encoding for unknown words
    char_encoding: HashMap<char, usize>,
    
    /// Maximum spike vector size
    max_size: usize,
}

impl TextToSpikeEncoder {
    pub fn new(max_size: usize) -> Self {
        // Build basic vocabulary from common command words
        let mut vocab_map = HashMap::new();
        let command_words = vec![
            "show", "status", "quantum", "coherence", "system", "run", "stop", 
            "start", "configure", "deploy", "monitor", "analyze", "optimize",
            "fix", "debug", "check", "scan", "detect", "coordinate", "learn",
            "train", "help", "reset", "backup", "restore", "update", "install",
        ];
        
        for (i, word) in command_words.iter().enumerate() {
            vocab_map.insert(word.to_string(), i);
        }
        
        // Character encoding for OOV words
        let mut char_encoding = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz0123456789-_".chars().enumerate() {
            char_encoding.insert(c, i + 1000); // Offset to avoid vocab conflicts
        }
        
        Self {
            vocab_map,
            char_encoding,
            max_size,
        }
    }
    
    /// Convert text to spike pattern suitable for neuromorphic processing
    pub fn encode_text(&self, text: &str) -> Array1<f64> {
        let mut spike_pattern = Array1::zeros(self.max_size);
        
        // Tokenize and normalize
        let tokens: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_'))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        
        debug!("Tokenized input: {:?}", tokens);
        
        // Encode tokens as spike frequencies
        for (pos, token) in tokens.iter().enumerate() {
            if let Some(&vocab_idx) = self.vocab_map.get(token) {
                // Known vocabulary word - high frequency
                if vocab_idx < self.max_size {
                    spike_pattern[vocab_idx] = 50.0; // 50Hz base rate
                    
                    // Position encoding (early words get higher frequency)
                    let position_boost = 1.0 + (0.5 * (1.0 - pos as f64 / tokens.len() as f64));
                    spike_pattern[vocab_idx] *= position_boost;
                }
            } else {
                // Unknown word - encode character by character
                for (char_pos, c) in token.chars().enumerate() {
                    if let Some(&char_idx) = self.char_encoding.get(&c) {
                        if char_idx < self.max_size {
                            spike_pattern[char_idx] = 20.0 + (char_pos as f64 * 2.0);
                        }
                    }
                }
            }
        }
        
        // Add context features
        self.add_context_features(&mut spike_pattern, text);
        
        spike_pattern
    }
    
    fn add_context_features(&self, spike_pattern: &mut Array1<f64>, text: &str) {
        let text_lower = text.to_lowercase();
        
        // Question detection
        if text.contains('?') {
            if let Some(question_idx) = self.get_feature_index("question") {
                spike_pattern[question_idx] = 60.0;
            }
        }
        
        // Urgency detection
        let urgent_words = ["urgent", "critical", "emergency", "immediately", "asap"];
        if urgent_words.iter().any(|&word| text_lower.contains(word)) {
            if let Some(urgent_idx) = self.get_feature_index("urgent") {
                spike_pattern[urgent_idx] = 80.0;
            }
        }
        
        // Command type detection
        let command_prefixes = ["show", "run", "stop", "start", "configure"];
        for prefix in command_prefixes {
            if text_lower.starts_with(prefix) {
                if let Some(cmd_idx) = self.get_feature_index(&format!("cmd_{}", prefix)) {
                    spike_pattern[cmd_idx] = 70.0;
                }
            }
        }
    }
    
    fn get_feature_index(&self, feature: &str) -> Option<usize> {
        // Reserved indices for context features (8000-9999)
        let feature_offset = 8000;
        match feature {
            "question" => Some(feature_offset),
            "urgent" => Some(feature_offset + 1),
            "cmd_show" => Some(feature_offset + 2),
            "cmd_run" => Some(feature_offset + 3),
            "cmd_stop" => Some(feature_offset + 4),
            "cmd_start" => Some(feature_offset + 5),
            "cmd_configure" => Some(feature_offset + 6),
            _ => None,
        }
    }
}

/// Command pattern library for mapping neural responses to commands
pub struct CommandPatternLibrary {
    /// Known patterns: neural response -> command mapping
    patterns: HashMap<String, CommandPattern>,
    
    /// Learning examples for improving patterns
    learning_examples: Vec<LearningExample>,
}

#[derive(Debug, Clone)]
pub struct CommandPattern {
    /// Command template
    pub command: String,
    
    /// Neural response pattern
    pub neural_signature: Array1<f64>,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Usage count
    pub usage_count: u64,
}

#[derive(Debug, Clone)]
pub struct LearningExample {
    pub input_text: String,
    pub expected_command: String,
    pub neural_response: Array1<f64>,
    pub was_correct: bool,
    pub timestamp: std::time::SystemTime,
}

impl CommandPatternLibrary {
    pub fn new() -> Self {
        Self {
            patterns: Self::create_default_patterns(),
            learning_examples: Vec::new(),
        }
    }
    
    fn create_default_patterns() -> HashMap<String, CommandPattern> {
        let mut patterns = HashMap::new();
        
        // Default command patterns (will be replaced by learned patterns)
        let defaults = vec![
            ("show_status", "csf status"),
            ("quantum_metrics", "csf quantum status --detailed"),
            ("system_health", "csf health check"),
            ("optimize_performance", "csf optimize --auto"),
            ("backup_config", "csf backup create --config"),
        ];
        
        for (pattern_id, command) in defaults {
            patterns.insert(pattern_id.to_string(), CommandPattern {
                command: command.to_string(),
                neural_signature: Array1::zeros(10000), // Will be learned
                success_rate: 0.5, // Neutral starting point
                usage_count: 0,
            });
        }
        
        patterns
    }
    
    /// Find best matching command for neural response
    pub fn match_response(&self, neural_response: &Array1<f64>) -> Option<CommandIntent> {
        let mut best_match: Option<(&String, f64)> = None;
        
        // Calculate similarity with known patterns
        for (pattern_id, pattern) in &self.patterns {
            let similarity = self.calculate_similarity(neural_response, &pattern.neural_signature);
            
            if similarity > 0.6 { // Minimum confidence threshold
                match best_match {
                    Some((_, best_sim)) if similarity > best_sim => {
                        best_match = Some((pattern_id, similarity));
                    },
                    None => {
                        best_match = Some((pattern_id, similarity));
                    },
                    _ => {}
                }
            }
        }
        
        if let Some((pattern_id, confidence)) = best_match {
            let pattern = &self.patterns[pattern_id];
            
            Some(CommandIntent {
                command: pattern.command.clone(),
                confidence,
                context: CommandContext {
                    domain: self.classify_domain(&pattern.command),
                    urgency: self.assess_urgency(neural_response),
                    sentiment: CommandSentiment {
                        valence: 0.0,
                        arousal: 0.0,
                        operator_state: OperatorState::Calm,
                    },
                    parameters: HashMap::new(),
                },
                alternatives: vec![],
                requires_confirmation: confidence < 0.8,
            })
        } else {
            None
        }
    }
    
    /// Learn from user feedback
    pub fn learn_pattern(&mut self, input: &str, neural_response: Array1<f64>, correct_command: &str) {
        debug!("Learning pattern: '{}' -> '{}'", input, correct_command);
        
        // Create pattern ID from command
        let pattern_id = self.create_pattern_id(correct_command);
        
        // Update or create pattern
        let pattern = self.patterns.entry(pattern_id.clone()).or_insert_with(|| {
            CommandPattern {
                command: correct_command.to_string(),
                neural_signature: neural_response.clone(),
                success_rate: 1.0,
                usage_count: 1,
            }
        });
        
        // Update pattern with exponential moving average
        let alpha = 0.1; // Learning rate
        pattern.neural_signature = (1.0 - alpha) * &pattern.neural_signature + alpha * &neural_response;
        pattern.usage_count += 1;
        
        // Store learning example
        self.learning_examples.push(LearningExample {
            input_text: input.to_string(),
            expected_command: correct_command.to_string(),
            neural_response,
            was_correct: true,
            timestamp: std::time::SystemTime::now(),
        });
        
        // Keep learning examples bounded
        if self.learning_examples.len() > 10000 {
            self.learning_examples.remove(0);
        }
        
        info!("Pattern learned: {} (usage: {})", pattern_id, pattern.usage_count);
    }
    
    fn calculate_similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        // Cosine similarity between neural response vectors
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    fn create_pattern_id(&self, command: &str) -> String {
        // Create stable pattern ID from command structure
        let normalized = command
            .split_whitespace()
            .take(3) // First 3 words
            .collect::<Vec<&str>>()
            .join("_")
            .to_lowercase();
        
        format!("pattern_{}", normalized)
    }
    
    fn classify_domain(&self, command: &str) -> CommandDomain {
        let cmd_lower = command.to_lowercase();
        
        if cmd_lower.contains("quantum") || cmd_lower.contains("coherence") {
            CommandDomain::Quantum
        } else if cmd_lower.contains("threat") || cmd_lower.contains("defense") || cmd_lower.contains("security") {
            CommandDomain::Defense
        } else if cmd_lower.contains("learn") || cmd_lower.contains("train") || cmd_lower.contains("teach") {
            CommandDomain::Learning
        } else if cmd_lower.contains("status") || cmd_lower.contains("health") || cmd_lower.contains("monitor") {
            CommandDomain::Status
        } else if cmd_lower.starts_with("csf") {
            CommandDomain::System
        } else {
            CommandDomain::Unknown
        }
    }
    
    fn assess_urgency(&self, neural_response: &Array1<f64>) -> f64 {
        // Look for high-frequency spikes indicating urgency
        let high_activity_threshold = 50.0;
        let urgent_spikes = neural_response.iter()
            .filter(|&&spike| spike > high_activity_threshold)
            .count();
        
        let urgency_ratio = urgent_spikes as f64 / neural_response.len() as f64;
        urgency_ratio.min(1.0)
    }
}

impl NeuralLanguageProcessor {
    pub async fn new(
        backend: Arc<RwLock<NeuromorphicBackend>>,
        clogic_system: Arc<CLogicSystem>,
        learning_config: LearningConfig,
    ) -> Result<Self> {
        info!("Initializing Neural Language Processor");
        
        let text_encoder = TextToSpikeEncoder::new(10000);
        let command_patterns = Arc::new(RwLock::new(CommandPatternLibrary::new()));
        let metrics = Arc::new(RwLock::new(NLPMetrics::default()));
        
        Ok(Self {
            backend,
            clogic_system,
            text_encoder,
            command_patterns,
            learning_config,
            metrics,
        })
    }
    
    /// Process natural language input and return command intent
    pub async fn process_input(&self, input: &str) -> Result<CommandIntent> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing natural language: '{}'", input);
        
        // Stage 1: Convert text to spike pattern
        let spike_pattern = self.text_encoder.encode_text(input);
        
        // Stage 2: Process through neuromorphic backend
        let neural_response = {
            let backend = self.backend.read().await;
            backend.process_spikes(spike_pattern).await?
        };
        
        // Stage 3: Analyze through C-LOGIC DRPP for pattern recognition
        let drpp_state = self.clogic_system.get_state().await;
        let pattern_context = self.analyze_with_drpp(&neural_response, &drpp_state.drpp_state);
        
        // Stage 4: Emotional analysis through EMS
        let sentiment = self.analyze_sentiment(&drpp_state.ems_state, input);
        
        // Stage 5: Match to known command patterns
        let mut intent = {
            let patterns = self.command_patterns.read().await;
            patterns.match_response(&neural_response)
        }.unwrap_or_else(|| {
            // Fallback: create unknown intent
            CommandIntent {
                command: format!("unknown: {}", input),
                confidence: 0.0,
                context: CommandContext {
                    domain: CommandDomain::Unknown,
                    urgency: 0.5,
                    sentiment: sentiment.clone(),
                    parameters: HashMap::new(),
                },
                alternatives: vec![],
                requires_confirmation: true,
            }
        });
        
        // Enhance intent with context analysis
        intent.context.sentiment = sentiment;
        intent.context.urgency = self.calculate_urgency(input, &neural_response);
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_processed += 1;
            metrics.avg_confidence = (metrics.avg_confidence + intent.confidence) / 2.0;
            metrics.processing_time_ms = (metrics.processing_time_ms + processing_time) / 2.0;
            
            let domain_key = format!("{:?}", intent.context.domain);
            *metrics.domain_distribution.entry(domain_key).or_insert(0) += 1;
        }
        
        debug!("Processed intent: {:?}", intent);
        Ok(intent)
    }
    
    /// Learn from user correction
    pub async fn learn_from_correction(&self, input: &str, correct_command: &str) -> Result<()> {
        debug!("Learning from correction: '{}' -> '{}'", input, correct_command);
        
        // Re-encode the input to get neural response
        let spike_pattern = self.text_encoder.encode_text(input);
        let neural_response = {
            let backend = self.backend.read().await;
            backend.process_spikes(spike_pattern).await?
        };
        
        // Update pattern library
        {
            let mut patterns = self.command_patterns.write().await;
            patterns.learn_pattern(input, neural_response.clone(), correct_command);
        }
        
        // Train the neuromorphic backend
        {
            let backend = self.backend.read().await;
            let training_error = backend.train(input, correct_command).await?;
            debug!("Training error: {:.4}", training_error);
        }
        
        Ok(())
    }
    
    fn analyze_with_drpp(&self, neural_response: &Array1<f64>, drpp_state: &csf_clogic::drpp::DrppState) -> HashMap<String, f64> {
        let mut context = HashMap::new();
        
        // Analyze coherence with DRPP oscillator patterns
        let response_coherence = self.calculate_response_coherence(neural_response);
        context.insert("neural_coherence".to_string(), response_coherence);
        
        // Use DRPP detected patterns for context
        context.insert("drpp_coherence".to_string(), drpp_state.coherence);
        context.insert("detected_patterns".to_string(), drpp_state.detected_patterns.len() as f64);
        
        context
    }
    
    fn analyze_sentiment(&self, ems_state: &csf_clogic::ems::EmsState, input: &str) -> CommandSentiment {
        // Use EMS emotional analysis
        let valence = ems_state.valence;
        let arousal = ems_state.arousal;
        
        // Infer operator state from text and emotional context
        let operator_state = if input.contains('?') && arousal > 0.5 {
            OperatorState::Troubleshooting
        } else if arousal > 0.7 {
            OperatorState::Urgent
        } else if valence < -0.3 {
            OperatorState::Stressed
        } else if input.contains("learn") || input.contains("teach") {
            OperatorState::Learning
        } else {
            OperatorState::Calm
        };
        
        CommandSentiment {
            valence,
            arousal,
            operator_state,
        }
    }
    
    fn calculate_urgency(&self, input: &str, neural_response: &Array1<f64>) -> f64 {
        let mut urgency = 0.0;
        
        // Text-based urgency indicators
        let urgent_keywords = ["urgent", "critical", "emergency", "immediately", "now", "asap"];
        for keyword in urgent_keywords {
            if input.to_lowercase().contains(keyword) {
                urgency += 0.3;
            }
        }
        
        // Neural response intensity
        let response_intensity = neural_response.mean().unwrap_or(0.0) / 100.0;
        urgency += response_intensity * 0.4;
        
        // Exclamation marks
        urgency += (input.matches('!').count() as f64) * 0.1;
        
        urgency.min(1.0)
    }
    
    fn calculate_response_coherence(&self, response: &Array1<f64>) -> f64 {
        // Calculate coherence of neural response
        let mean = response.mean().unwrap_or(0.0);
        let variance = response.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / response.len() as f64;
        
        // Higher coherence = lower variance relative to mean
        if mean > 0.0 {
            1.0 / (1.0 + variance / mean)
        } else {
            0.0
        }
    }
    
    /// Get processing metrics
    pub async fn get_metrics(&self) -> NLPMetrics {
        self.metrics.read().await.clone()
    }
}