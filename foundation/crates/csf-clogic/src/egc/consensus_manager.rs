//! Consensus management for EGC

use super::{ConsensusResult, Decision, DecisionId};
use csf_core::types::{hardware_timestamp, NanoTime};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Consensus manager for distributed decision making
pub struct ConsensusManager {
    /// Consensus algorithm
    algorithm: ConsensusAlgorithm,

    /// Voting records
    voting_records: DashMap<DecisionId, VotingRecord>,

    /// Participant registry
    participants: Arc<RwLock<Vec<Participant>>>,

    /// Consensus threshold
    threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ConsensusAlgorithm {
    SimpleMajority,
    SuperMajority,
    Byzantine,
    WeightedVoting,
}

#[derive(Debug, Clone)]
struct VotingRecord {
    decision_id: DecisionId,
    votes: DashMap<String, Vote>,
    start_time: NanoTime,
    consensus_reached: bool,
}

#[derive(Debug, Clone)]
struct Vote {
    participant_id: String,
    option_id: String,
    weight: f64,
    timestamp: NanoTime,
}

#[derive(Debug, Clone)]
struct Participant {
    id: String,
    voting_weight: f64,
    reputation: f64,
    active: bool,
}

impl ConsensusManager {
    /// Create a new consensus manager
    pub fn new(config: &super::EgcConfig) -> Self {
        let algorithm = match config.governance_model {
            super::GovernanceModel::Consensus => ConsensusAlgorithm::SimpleMajority,
            super::GovernanceModel::Hierarchical => ConsensusAlgorithm::WeightedVoting,
            super::GovernanceModel::Market => ConsensusAlgorithm::WeightedVoting,
            super::GovernanceModel::Hybrid => ConsensusAlgorithm::Byzantine,
        };

        // Initialize with default participants
        let participants = vec![
            Participant {
                id: "node_0".to_string(),
                voting_weight: 1.0,
                reputation: 1.0,
                active: true,
            },
            Participant {
                id: "node_1".to_string(),
                voting_weight: 1.0,
                reputation: 1.0,
                active: true,
            },
            Participant {
                id: "node_2".to_string(),
                voting_weight: 1.0,
                reputation: 1.0,
                active: true,
            },
        ];

        Self {
            algorithm,
            voting_records: DashMap::new(),
            participants: Arc::new(RwLock::new(participants)),
            threshold: config.consensus_threshold,
        }
    }

    /// Start consensus process for a decision
    pub fn start_consensus(&self, decision: &Decision) -> anyhow::Result<()> {
        let record = VotingRecord {
            decision_id: decision.id,
            votes: DashMap::new(),
            start_time: hardware_timestamp(),
            consensus_reached: false,
        };

        self.voting_records.insert(decision.id, record);
        Ok(())
    }

    /// Submit a vote
    pub async fn submit_vote(
        &self,
        decision_id: DecisionId,
        participant_id: String,
        option_id: String,
    ) -> anyhow::Result<()> {
        let participants = self.participants.read().await;
        let participant = participants
            .iter()
            .find(|p| p.id == participant_id && p.active)
            .ok_or_else(|| anyhow::anyhow!("Invalid or inactive participant"))?;

        let vote = Vote {
            participant_id: participant_id.clone(),
            option_id,
            weight: participant.voting_weight * participant.reputation,
            timestamp: hardware_timestamp(),
        };

        if let Some(record) = self.voting_records.get_mut(&decision_id) {
            record.votes.insert(participant_id, vote);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Decision not found"))
        }
    }

    /// Reach consensus on a decision
    pub async fn reach_consensus(
        &self,
        decision_id: DecisionId,
        timeout_ms: u64,
    ) -> anyhow::Result<ConsensusResult> {
        let start_time = hardware_timestamp();
        let deadline = start_time + NanoTime::from_nanos(timeout_ms * 1_000_000);

        // Simulate voting process
        // In a real implementation, this would collect votes from distributed nodes
        let participants = self.participants.read().await;
        for (i, participant) in participants.iter().enumerate() {
            if participant.active {
                // Simulate vote based on participant index
                let option_id = if i % 2 == 0 { "allow" } else { "deny" };
                self.submit_vote(decision_id, participant.id.clone(), option_id.to_string())
                    .await?;
            }
        }
        drop(participants);

        // Wait for consensus or timeout
        loop {
            if let Some(result) = self.check_consensus(decision_id).await? {
                return Ok(result);
            }

            if hardware_timestamp() > deadline {
                return Err(anyhow::anyhow!("Consensus timeout"));
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    /// Check if consensus has been reached
    async fn check_consensus(
        &self,
        decision_id: DecisionId,
    ) -> anyhow::Result<Option<ConsensusResult>> {
        let record = self
            .voting_records
            .get(&decision_id)
            .ok_or_else(|| anyhow::anyhow!("Decision not found"))?;

        if record.consensus_reached {
            // Already reached consensus
            return Ok(None);
        }

        // Count votes by option
        let mut vote_counts: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        let mut total_weight = 0.0;

        for vote_entry in record.votes.iter() {
            let vote = vote_entry.value();
            *vote_counts.entry(vote.option_id.clone()).or_insert(0.0) += vote.weight;
            total_weight += vote.weight;
        }

        // Check if any option has reached threshold
        match self.algorithm {
            ConsensusAlgorithm::SimpleMajority => {
                for (option_id, weight) in vote_counts.iter() {
                    if *weight / total_weight > 0.5 {
                        return Ok(Some(ConsensusResult {
                            decision_id,
                            selected_option: option_id.clone(),
                            consensus_strength: *weight / total_weight,
                            timestamp: hardware_timestamp(),
                        }));
                    }
                }
            }
            ConsensusAlgorithm::SuperMajority | ConsensusAlgorithm::Byzantine => {
                for (option_id, weight) in vote_counts.iter() {
                    if *weight / total_weight >= self.threshold {
                        return Ok(Some(ConsensusResult {
                            decision_id,
                            selected_option: option_id.clone(),
                            consensus_strength: *weight / total_weight,
                            timestamp: hardware_timestamp(),
                        }));
                    }
                }
            }
            ConsensusAlgorithm::WeightedVoting => {
                // Find option with highest weight
                if let Some((option_id, weight)) = vote_counts
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                {
                    if *weight / total_weight >= self.threshold {
                        return Ok(Some(ConsensusResult {
                            decision_id,
                            selected_option: option_id.clone(),
                            consensus_strength: *weight / total_weight,
                            timestamp: hardware_timestamp(),
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Add a participant
    pub async fn add_participant(&self, id: String, voting_weight: f64) -> anyhow::Result<()> {
        let mut participants = self.participants.write().await;

        if participants.iter().any(|p| p.id == id) {
            return Err(anyhow::anyhow!("Participant already exists"));
        }

        participants.push(Participant {
            id,
            voting_weight,
            reputation: 1.0,
            active: true,
        });

        Ok(())
    }

    /// Update participant reputation
    pub async fn update_reputation(
        &self,
        participant_id: &str,
        reputation_delta: f64,
    ) -> anyhow::Result<()> {
        let mut participants = self.participants.write().await;

        if let Some(participant) = participants.iter_mut().find(|p| p.id == participant_id) {
            participant.reputation = (participant.reputation + reputation_delta).clamp(0.0, 2.0);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Participant not found"))
        }
    }

    /// Get voting statistics
    pub fn get_voting_stats(&self, decision_id: DecisionId) -> Option<VotingStats> {
        self.voting_records.get(&decision_id).map(|record| {
            let mut stats = VotingStats {
                total_votes: record.votes.len(),
                vote_distribution: std::collections::HashMap::new(),
                elapsed_time_ms: (hardware_timestamp() - record.start_time).as_nanos() as u64
                    / 1_000_000,
            };

            for vote_entry in record.votes.iter() {
                let vote = vote_entry.value();
                *stats
                    .vote_distribution
                    .entry(vote.option_id.clone())
                    .or_insert(0) += 1;
            }

            stats
        })
    }
}

#[derive(Debug, Clone)]
pub struct VotingStats {
    pub total_votes: usize,
    pub vote_distribution: std::collections::HashMap<String, usize>,
    pub elapsed_time_ms: u64,
}
