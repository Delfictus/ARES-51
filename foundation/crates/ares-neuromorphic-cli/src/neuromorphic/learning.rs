//! Learning subsystem for dynamic neuromorphic adaptation

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::backend::NeuromorphicBackend;
use super::LearningConfig;

/// Dynamic learning mode for neuromorphic command processing
pub struct LearnMode {
    /// Learning state
    active: Arc<AtomicBool>,
    
    /// Neuromorphic backend for training
    backend: Arc<RwLock<super::backend::NeuromorphicBackend>>,
    
    /// Configuration
    config: LearningConfig,
    
    /// Training samples
    training_samples: Arc<RwLock<Vec<TrainingSample>>>,
    
    /// Learning metrics
    metrics: Arc<RwLock<LearningMetrics>>,
    
    /// Pattern confidence tracking
    pattern_confidence: Arc<RwLock<HashMap<String, f64>>>,
}

#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Original natural language input
    pub input: String,
    
    /// Correct command mapping
    pub correct_command: String,
    
    /// Whether the mapping was successful
    pub was_successful: bool,
    
    /// Timestamp of the sample
    pub timestamp: std::time::SystemTime,
    
    /// Context when sample was created
    pub context: TrainingContext,
}

#[derive(Debug, Clone)]
pub struct TrainingContext {
    /// System state when sample was created
    pub system_load: f64,
    
    /// Operator emotional state
    pub operator_state: String,
    
    /// Session duration
    pub session_duration_minutes: u64,
}

#[derive(Debug, Clone)]
pub struct LearningMetrics {
    /// Total training samples collected
    pub total_samples: u64,
    
    /// Average confidence improvement
    pub avg_confidence_improvement: f64,
    
    /// Learning rate adaptation history
    pub learning_rate_history: Vec<(std::time::SystemTime, f64)>,
    
    /// Pattern accuracy by domain
    pub domain_accuracy: HashMap<String, f64>,
    
    /// Recently learned patterns
    pub recent_patterns: Vec<String>,
    
    /// Learning session statistics
    pub session_stats: SessionStats,
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub session_start: std::time::SystemTime,
    pub commands_learned: u64,
    pub corrections_made: u64,
    pub average_learning_time_ms: f64,
}

impl Default for LearningMetrics {
    fn default() -> Self {
        Self {
            total_samples: 0,
            avg_confidence_improvement: 0.0,
            learning_rate_history: Vec::new(),
            domain_accuracy: HashMap::new(),
            recent_patterns: Vec::new(),
            session_stats: SessionStats {
                session_start: std::time::SystemTime::now(),
                commands_learned: 0,
                corrections_made: 0,
                average_learning_time_ms: 0.0,
            },
        }
    }
}

impl LearnMode {
    pub async fn new(
        backend: Arc<RwLock<NeuromorphicBackend>>,
        config: LearningConfig,
    ) -> Result<Self> {
        info!("Initializing learning subsystem");
        
        Ok(Self {
            active: Arc::new(AtomicBool::new(false)),
            backend,
            config,
            training_samples: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(LearningMetrics::default())),
            pattern_confidence: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Toggle learning mode on/off
    pub async fn toggle(&self) -> Result<bool> {
        let was_active = self.active.load(Ordering::Relaxed);
        let now_active = !was_active;
        self.active.store(now_active, Ordering::Relaxed);
        
        if now_active {
            info!("🧠 LEARNING MODE ACTIVATED");
            self.activate_learning().await?;
        } else {
            info!("📚 LEARNING MODE DEACTIVATED - Consolidating knowledge");
            self.deactivate_learning().await?;
        }
        
        Ok(now_active)
    }
    
    /// Check if learning mode is currently active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
    
    /// Record a training sample from user interaction
    pub async fn record_training_sample(
        &self,
        input: &str,
        correct_command: &str,
        was_successful: bool,
        context: TrainingContext,
    ) -> Result<()> {
        if !self.is_active() {
            return Ok(());
        }
        
        debug!("Recording training sample: '{}' -> '{}'", input, correct_command);
        
        let sample = TrainingSample {
            input: input.to_string(),
            correct_command: correct_command.to_string(),
            was_successful,
            timestamp: std::time::SystemTime::now(),
            context,
        };
        
        // Store sample
        {
            let mut samples = self.training_samples.write().await;
            samples.push(sample.clone());
            
            // Keep samples bounded
            if samples.len() > self.config.max_patterns {
                samples.remove(0);
            }
        }
        
        // Train the neuromorphic backend
        let training_error = {
            let backend = self.backend.read().await;
            backend.train(input, correct_command).await?
        };
        
        // Update pattern confidence
        {
            let mut confidence = self.pattern_confidence.write().await;
            let pattern_key = self.create_pattern_key(input);
            let current_confidence = confidence.get(&pattern_key).unwrap_or(&0.5);
            
            // Update confidence based on training error
            let new_confidence = if was_successful {
                (current_confidence + (1.0 - training_error)).min(1.0)
            } else {
                (current_confidence - 0.1).max(0.0)
            };
            
            confidence.insert(pattern_key.clone(), new_confidence);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_samples += 1;
            metrics.session_stats.commands_learned += 1;
            
            if !was_successful {
                metrics.session_stats.corrections_made += 1;
            }
            
            // Track recent patterns
            let pattern_key = self.create_pattern_key(input);
            metrics.recent_patterns.push(pattern_key);
            if metrics.recent_patterns.len() > 20 {
                metrics.recent_patterns.remove(0);
            }
        }
        
        info!("📝 Pattern learned: '{}' -> '{}' (error: {:.3})", 
              input, correct_command, training_error);
        
        Ok(())
    }
    
    /// Get current learning metrics
    pub async fn get_metrics(&self) -> LearningMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Export learned patterns for backup/analysis
    pub async fn export_patterns(&self) -> Result<Vec<TrainingSample>> {
        let samples = self.training_samples.read().await;
        Ok(samples.clone())
    }
    
    /// Import training patterns (for initialization or transfer learning)
    pub async fn import_patterns(&self, samples: Vec<TrainingSample>) -> Result<()> {
        info!("Importing {} training patterns", samples.len());
        
        for sample in samples {
            self.record_training_sample(
                &sample.input,
                &sample.correct_command,
                sample.was_successful,
                sample.context,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Adaptive learning rate based on performance
    pub async fn adapt_learning_rate(&self) -> Result<f64> {
        let metrics = self.metrics.read().await;
        let confidence_map = self.pattern_confidence.read().await;
        
        // Calculate average confidence across all patterns
        let avg_confidence = if confidence_map.is_empty() {
            0.5
        } else {
            confidence_map.values().sum::<f64>() / confidence_map.len() as f64
        };
        
        // Adapt learning rate based on performance
        let base_rate = self.config.learning_rate;
        let adapted_rate = if avg_confidence < 0.6 {
            // Low confidence - increase learning rate
            base_rate * 1.5
        } else if avg_confidence > 0.9 {
            // High confidence - decrease learning rate to prevent overfitting
            base_rate * 0.8
        } else {
            // Good performance - keep current rate
            base_rate
        };
        
        debug!("Adapted learning rate: {:.4} (avg confidence: {:.3})", 
               adapted_rate, avg_confidence);
        
        Ok(adapted_rate)
    }
    
    async fn activate_learning(&self) -> Result<()> {
        info!("Activating neuromorphic learning systems");
        
        // Initialize learning in the backend
        {
            let backend = self.backend.read().await;
            backend.initialize_learning().await?;
        }
        
        // Reset session statistics
        {
            let mut metrics = self.metrics.write().await;
            metrics.session_stats = SessionStats {
                session_start: std::time::SystemTime::now(),
                commands_learned: 0,
                corrections_made: 0,
                average_learning_time_ms: 0.0,
            };
        }
        
        info!("✓ Learning systems activated");
        Ok(())
    }
    
    async fn deactivate_learning(&self) -> Result<()> {
        info!("Deactivating learning mode and consolidating knowledge");
        
        // Consolidate patterns by merging similar ones
        self.consolidate_patterns().await?;
        
        // Generate learning session report
        let metrics = self.metrics.read().await;
        let session_duration = metrics.session_stats.session_start
            .elapsed()
            .unwrap_or_default()
            .as_secs() / 60; // minutes
        
        info!("📊 Learning session complete:");
        info!("   Duration: {} minutes", session_duration);
        info!("   Commands learned: {}", metrics.session_stats.commands_learned);
        info!("   Corrections made: {}", metrics.session_stats.corrections_made);
        info!("   Total patterns: {}", metrics.total_samples);
        
        Ok(())
    }
    
    async fn consolidate_patterns(&self) -> Result<()> {
        debug!("Consolidating learned patterns");
        
        let samples = self.training_samples.read().await;
        let mut pattern_groups: HashMap<String, Vec<&TrainingSample>> = HashMap::new();
        
        // Group similar patterns
        for sample in samples.iter() {
            let pattern_key = self.create_pattern_key(&sample.input);
            pattern_groups.entry(pattern_key).or_default().push(sample);
        }
        
        // Merge patterns with multiple examples
        let mut consolidated_count = 0;
        for (pattern_key, group) in pattern_groups {
            if group.len() > 1 {
                // Find the most successful command for this pattern
                let best_command = group.iter()
                    .filter(|s| s.was_successful)
                    .max_by_key(|s| s.timestamp)
                    .map(|s| &s.correct_command)
                    .or_else(|| group.last().map(|s| &s.correct_command));
                
                if let Some(command) = best_command {
                    consolidated_count += 1;
                    debug!("Consolidated pattern '{}' -> '{}'", pattern_key, command);
                }
            }
        }
        
        info!("📋 Consolidated {} pattern groups", consolidated_count);
        Ok(())
    }
    
    fn create_pattern_key(&self, input: &str) -> String {
        // Create normalized pattern key for grouping similar inputs
        input
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2) // Filter short words
            .take(5) // Take first 5 significant words
            .collect::<Vec<&str>>()
            .join("_")
    }
}