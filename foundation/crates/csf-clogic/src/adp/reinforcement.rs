//! Reinforcement learning for ADP

use anyhow::Result;
use ndarray::Array1;
use parking_lot::RwLock;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;

use super::Action;

/// RL configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RlConfig {
    /// Learning rate
    pub alpha: f64,
    /// Discount factor
    pub gamma: f64,
    /// Exploration rate
    pub epsilon: f64,
    /// Epsilon decay
    pub epsilon_decay: f64,
    /// Minimum epsilon
    pub epsilon_min: f64,
    /// Experience replay buffer size
    pub buffer_size: usize,
    /// Batch size for learning
    pub batch_size: usize,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            alpha: 0.001,
            gamma: 0.95,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            buffer_size: 10000,
            batch_size: 32,
        }
    }
}

/// Reinforcement learner using Q-learning
pub struct ReinforcementLearner {
    config: RlConfig,
    q_table: Arc<RwLock<HashMap<StateKey, Array1<f64>>>>,
    action_space: Vec<usize>,
    epsilon: Arc<RwLock<f64>>,
}

/// Discretized state representation
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct StateKey(Vec<i32>);

impl ReinforcementLearner {
    pub fn new(config: RlConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            q_table: Arc::new(RwLock::new(HashMap::new())),
            action_space: vec![0, 1, 2, 3, 4], // 5 possible actions
            epsilon: Arc::new(RwLock::new(config.epsilon)),
        })
    }

    pub async fn select_action(&self, state: &Array1<f64>) -> Result<(usize, f64)> {
        let state_key = self.discretize_state(state);
        let mut rng = rand::thread_rng();

        // Epsilon-greedy action selection
        let epsilon = *self.epsilon.read();
        let action = if rng.gen::<f64>() < epsilon {
            // Explore: random action
            self.action_space[rng.gen_range(0..self.action_space.len())]
        } else {
            // Exploit: best action from Q-table
            let q_values = self.get_q_values(&state_key);
            self.get_best_action(&q_values)
        };

        // Get Q-value as confidence
        let q_values = self.get_q_values(&state_key);
        let confidence = if action < q_values.len() {
            q_values[action].tanh().abs() // Normalize to [0, 1]
        } else {
            0.5
        };

        Ok((action, confidence))
    }

    pub async fn update_policy(
        &self,
        state: &Array1<f64>,
        action: Action,
        reward: f64,
        next_state: &Array1<f64>,
        done: bool,
    ) -> Result<()> {
        let state_key = self.discretize_state(state);
        let next_state_key = self.discretize_state(next_state);
        let action_idx = self.action_to_index(&action);

        // Get current Q-value
        let mut q_table = self.q_table.write();

        // Get max Q-value for next state first
        let next_q_values = if done {
            Array1::zeros(self.action_space.len())
        } else {
            q_table
                .get(&next_state_key)
                .cloned()
                .unwrap_or_else(|| Array1::zeros(self.action_space.len()))
        };

        // Now get current Q-values
        let q_values = q_table
            .entry(state_key.clone())
            .or_insert_with(|| Array1::zeros(self.action_space.len()));

        let max_next_q = next_q_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Q-learning update
        let old_q = q_values[action_idx];
        let new_q = old_q + self.config.alpha * (reward + self.config.gamma * max_next_q - old_q);
        q_values[action_idx] = new_q;

        // Decay epsilon
        let mut epsilon = self.epsilon.write();
        *epsilon = (*epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        Ok(())
    }

    fn discretize_state(&self, state: &Array1<f64>) -> StateKey {
        // Simple discretization - in production would use more sophisticated approach
        let discretized: Vec<i32> = state
            .iter()
            .take(10) // Use first 10 features
            .map(|&val| {
                if val < -1.0 {
                    -2
                } else if val < -0.5 {
                    -1
                } else if val < 0.5 {
                    0
                } else if val < 1.0 {
                    1
                } else {
                    2
                }
            })
            .collect();

        StateKey(discretized)
    }

    fn get_q_values(&self, state_key: &StateKey) -> Array1<f64> {
        self.q_table
            .read()
            .get(state_key)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.action_space.len()))
    }

    fn get_best_action(&self, q_values: &Array1<f64>) -> usize {
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (idx, &value) in q_values.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_action = idx;
            }
        }

        best_action
    }

    fn action_to_index(&self, action: &Action) -> usize {
        match action {
            Action::Route { .. } => 0,
            Action::Modify { .. } => 1,
            Action::Drop { .. } => 2,
            Action::Buffer { .. } => 3,
            _ => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reinforcement_learner() {
        let config = RlConfig::default();
        let learner = ReinforcementLearner::new(config).unwrap();

        let state = Array1::from_vec(vec![0.5; 20]);
        let (action, confidence) = learner.select_action(&state).await.unwrap();

        assert!(action < 5);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}
