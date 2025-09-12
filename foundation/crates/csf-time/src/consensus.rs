//! Enterprise consensus protocols for distributed temporal consistency
//!
//! This module implements enterprise-grade consensus algorithms for ensuring
//! temporal consistency and deterministic execution across distributed nodes.

use crate::{
    clock::HlcClock,
    distributed::DistributedSynchronizer,
    global_hlc, LogicalTime, TimeError,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Enterprise temporal consensus algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusAlgorithm {
    /// Byzantine Fault Tolerant consensus for critical systems
    ByzantineFaultTolerant,
    /// RAFT consensus for typical distributed coordination
    Raft,
    /// Enterprise hybrid consensus optimized for temporal determinism
    EnterpriseHybrid,
}

/// Consensus proposal for distributed temporal state
#[derive(Debug, Clone)]
pub struct ConsensusProposal {
    /// Unique proposal identifier
    pub proposal_id: u64,
    /// Proposing node identifier
    pub proposer_node_id: u64,
    /// Proposed logical time for consensus
    pub proposed_time: LogicalTime,
    /// Algorithm to use for this consensus
    pub algorithm: ConsensusAlgorithm,
    /// Nodes that must participate in consensus
    pub participant_nodes: Vec<u64>,
    /// Proposal creation timestamp
    pub created_at: LogicalTime,
    /// Consensus timeout in nanoseconds
    pub timeout_ns: u64,
}

/// Consensus vote from a participating node
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    /// Voting node identifier
    pub voter_node_id: u64,
    /// Proposal being voted on
    pub proposal_id: u64,
    /// Vote decision
    pub vote: VoteDecision,
    /// Logical time when vote was cast
    pub vote_time: LogicalTime,
    /// Optional justification for the vote
    pub justification: Option<String>,
}

/// Vote decision for consensus proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteDecision {
    /// Accept the proposal
    Accept,
    /// Reject the proposal
    Reject,
    /// Abstain from voting (neutral)
    Abstain,
}

/// Result of consensus execution
#[derive(Debug, Clone)]
pub enum ConsensusResult {
    /// Consensus achieved successfully
    Consensus {
        /// Final agreed-upon logical time
        agreed_time: LogicalTime,
        /// Nodes that participated in consensus
        participants: Vec<u64>,
        /// Final vote tally
        vote_tally: VoteTally,
    },
    /// Consensus failed to achieve agreement
    Failed {
        /// Reason for consensus failure
        reason: String,
        /// Partial vote results
        partial_votes: Vec<ConsensusVote>,
    },
    /// Consensus timed out
    Timeout {
        /// Time at which consensus timed out
        timeout_at: LogicalTime,
        /// Votes received before timeout
        received_votes: Vec<ConsensusVote>,
    },
}

/// Vote tally for consensus tracking
#[derive(Debug, Clone)]
pub struct VoteTally {
    /// Number of accept votes
    pub accept_count: usize,
    /// Number of reject votes
    pub reject_count: usize,
    /// Number of abstain votes
    pub abstain_count: usize,
    /// Total votes cast
    pub total_votes: usize,
}

/// Enterprise temporal consensus coordinator
#[derive(Debug)]
pub struct TemporalConsensusCoordinator {
    /// Local node identifier
    node_id: u64,
    /// Active consensus proposals
    active_proposals: Arc<RwLock<HashMap<u64, ConsensusProposal>>>,
    /// Received votes for proposals
    proposal_votes: Arc<RwLock<HashMap<u64, Vec<ConsensusVote>>>>,
    /// Distributed synchronizer for coordination
    #[allow(dead_code)]
    synchronizer: Arc<DistributedSynchronizer>,
    /// Default consensus algorithm
    default_algorithm: ConsensusAlgorithm,
}

impl TemporalConsensusCoordinator {
    /// Create new temporal consensus coordinator
    pub fn new(
        node_id: u64,
        synchronizer: Arc<DistributedSynchronizer>,
        default_algorithm: ConsensusAlgorithm,
    ) -> Self {
        Self {
            node_id,
            active_proposals: Arc::new(RwLock::new(HashMap::new())),
            proposal_votes: Arc::new(RwLock::new(HashMap::new())),
            synchronizer,
            default_algorithm,
        }
    }

    /// Propose a logical time for distributed consensus
    pub async fn propose_consensus(&self, proposed_time: LogicalTime, participant_nodes: Vec<u64>, timeout_ms: u64) -> Result<u64, TimeError> {
        let hlc = global_hlc()?;
        let current_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        let proposal_id = current_time.physical.wrapping_add(current_time.logical);
        
        let proposal = ConsensusProposal {
            proposal_id,
            proposer_node_id: self.node_id,
            proposed_time,
            algorithm: self.default_algorithm,
            participant_nodes: participant_nodes.clone(),
            created_at: current_time,
            timeout_ns: timeout_ms * 1_000_000,
        };
        
        // Store the proposal
        self.active_proposals.write().insert(proposal_id, proposal.clone());
        self.proposal_votes.write().insert(proposal_id, Vec::new());
        
        info!(
            node_id = self.node_id,
            proposal_id = proposal_id,
            proposed_time = %proposed_time,
            participants = ?participant_nodes,
            algorithm = ?self.default_algorithm,
            "Created consensus proposal for logical time"
        );
        
        Ok(proposal_id)
    }

    /// Cast a vote on a consensus proposal
    pub async fn cast_vote(&self, proposal_id: u64, vote: VoteDecision, justification: Option<String>) -> Result<(), TimeError> {
        let hlc = global_hlc()?;
        let vote_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        let consensus_vote = ConsensusVote {
            voter_node_id: self.node_id,
            proposal_id,
            vote,
            vote_time,
            justification,
        };
        
        // Record the vote
        let mut votes = self.proposal_votes.write();
        if let Some(proposal_votes) = votes.get_mut(&proposal_id) {
            // Remove any previous vote from this node
            proposal_votes.retain(|v| v.voter_node_id != self.node_id);
            proposal_votes.push(consensus_vote);
            
            debug!(
                node_id = self.node_id,
                proposal_id = proposal_id,
                vote = ?vote,
                vote_time = %vote_time,
                "Cast consensus vote"
            );
        } else {
            return Err(TimeError::SystemTimeError {
                details: format!("Consensus proposal {} not found", proposal_id),
            });
        }
        
        Ok(())
    }

    /// Execute consensus and return result
    pub async fn execute_consensus(&self, proposal_id: u64) -> Result<ConsensusResult, TimeError> {
        let proposal = {
            let proposals = self.active_proposals.read();
            proposals.get(&proposal_id).cloned()
                .ok_or_else(|| TimeError::SystemTimeError {
                    details: format!("Consensus proposal {} not found", proposal_id),
                })?
        };
        
        let start_time = std::time::Instant::now();
        let timeout_duration = Duration::from_nanos(proposal.timeout_ns);
        
        loop {
            // Check votes
            let (vote_tally, votes) = {
                let votes_map = self.proposal_votes.read();
                let votes = votes_map.get(&proposal_id).cloned().unwrap_or_default();
                let tally = self.calculate_vote_tally(&votes);
                (tally, votes)
            };
            
            // Check if we have enough votes for consensus
            let required_votes = proposal.participant_nodes.len().div_ceil(2); // Majority
            
            if vote_tally.total_votes >= required_votes {
                match proposal.algorithm {
                    ConsensusAlgorithm::ByzantineFaultTolerant => {
                        // BFT requires 2/3 majority
                        let bft_threshold = ((proposal.participant_nodes.len() + 1) * 2) / 3;
                        if vote_tally.accept_count >= bft_threshold {
                            return Ok(ConsensusResult::Consensus {
                                agreed_time: proposal.proposed_time,
                                participants: proposal.participant_nodes,
                                vote_tally,
                            });
                        }
                    }
                    ConsensusAlgorithm::Raft | ConsensusAlgorithm::EnterpriseHybrid => {
                        // Simple majority
                        if vote_tally.accept_count > vote_tally.reject_count {
                            return Ok(ConsensusResult::Consensus {
                                agreed_time: proposal.proposed_time,
                                participants: proposal.participant_nodes,
                                vote_tally,
                            });
                        }
                    }
                }
                
                // If we have enough votes but no consensus, it's a failure
                return Ok(ConsensusResult::Failed {
                    reason: "Consensus rejected by majority vote".to_string(),
                    partial_votes: votes,
                });
            }
            
            // Check timeout
            if start_time.elapsed() > timeout_duration {
                let hlc = global_hlc()?;
                let timeout_time = {
                    let clock = hlc.read();
                    HlcClock::current_time(&*clock)?
                };
                
                warn!(
                    node_id = self.node_id,
                    proposal_id = proposal_id,
                    elapsed_ms = start_time.elapsed().as_millis(),
                    "Consensus timed out"
                );
                
                return Ok(ConsensusResult::Timeout {
                    timeout_at: timeout_time,
                    received_votes: votes,
                });
            }
            
            // Wait before next check
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Calculate vote tally for a set of votes
    fn calculate_vote_tally(&self, votes: &[ConsensusVote]) -> VoteTally {
        let mut accept_count = 0;
        let mut reject_count = 0;
        let mut abstain_count = 0;
        
        for vote in votes {
            match vote.vote {
                VoteDecision::Accept => accept_count += 1,
                VoteDecision::Reject => reject_count += 1,
                VoteDecision::Abstain => abstain_count += 1,
            }
        }
        
        VoteTally {
            accept_count,
            reject_count,
            abstain_count,
            total_votes: votes.len(),
        }
    }

    /// Cleanup expired proposals and votes
    pub async fn cleanup_expired_proposals(&self) -> Result<usize, TimeError> {
        let hlc = global_hlc()?;
        let current_time = {
            let clock = hlc.read();
            HlcClock::current_time(&*clock)?
        };
        
        let mut proposals = self.active_proposals.write();
        let mut votes = self.proposal_votes.write();
        
        let original_count = proposals.len();
        
        // Remove expired proposals
        let expired_proposals: Vec<u64> = proposals
            .iter()
            .filter(|(_, proposal)| {
                let elapsed_ns = current_time.physical.saturating_sub(proposal.created_at.physical);
                elapsed_ns > proposal.timeout_ns
            })
            .map(|(&id, _)| id)
            .collect();
        
        for proposal_id in &expired_proposals {
            proposals.remove(proposal_id);
            votes.remove(proposal_id);
        }
        
        let removed_count = original_count - proposals.len();
        
        if removed_count > 0 {
            debug!(
                node_id = self.node_id,
                removed_count = removed_count,
                current_time = %current_time,
                "Cleaned up expired consensus proposals"
            );
        }
        
        Ok(removed_count)
    }

    /// Get consensus statistics for monitoring
    pub fn get_consensus_stats(&self) -> ConsensusStats {
        let proposals = self.active_proposals.read();
        let votes = self.proposal_votes.read();
        
        let total_proposals = proposals.len();
        let total_votes: usize = votes.values().map(|v| v.len()).sum();
        
        ConsensusStats {
            active_proposals: total_proposals,
            total_votes_cast: total_votes,
            node_id: self.node_id,
        }
    }
}

/// Statistics for consensus monitoring
#[derive(Debug, Clone)]
pub struct ConsensusStats {
    /// Number of active consensus proposals
    pub active_proposals: usize,
    /// Total votes cast across all proposals
    pub total_votes_cast: usize,
    /// Node identifier for this coordinator
    pub node_id: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{initialize_simulated_time_source, NanoTime};

    #[tokio::test]
    async fn test_consensus_coordinator_creation() {
        let synchronizer = Arc::new(DistributedSynchronizer::new(1, 5000));
        let coordinator = TemporalConsensusCoordinator::new(
            1,
            synchronizer,
            ConsensusAlgorithm::EnterpriseHybrid,
        );
        
        assert_eq!(coordinator.node_id, 1);
        assert_eq!(coordinator.default_algorithm, ConsensusAlgorithm::EnterpriseHybrid);
    }

    #[tokio::test]
    async fn test_consensus_proposal() {
        initialize_simulated_time_source(NanoTime::from_nanos(3000));
        
        let synchronizer = Arc::new(DistributedSynchronizer::new(1, 5000));
        let coordinator = TemporalConsensusCoordinator::new(
            1,
            synchronizer,
            ConsensusAlgorithm::Raft,
        );
        
        let proposed_time = LogicalTime::new(3000, 0, 1);
        let participants = vec![2, 3, 4];
        
        let proposal_id = coordinator
            .propose_consensus(proposed_time, participants, 1000)
            .await
            .expect("Should create consensus proposal");
        
        assert!(proposal_id > 0);
        
        // Verify proposal was stored
        let proposals = coordinator.active_proposals.read();
        assert!(proposals.contains_key(&proposal_id));
    }

    #[tokio::test]
    async fn test_consensus_voting() {
        initialize_simulated_time_source(NanoTime::from_nanos(4000));
        
        let synchronizer = Arc::new(DistributedSynchronizer::new(2, 5000));
        let coordinator = TemporalConsensusCoordinator::new(
            2,
            synchronizer,
            ConsensusAlgorithm::EnterpriseHybrid,
        );
        
        let proposed_time = LogicalTime::new(4000, 0, 2);
        let participants = vec![3, 4];
        
        let proposal_id = coordinator
            .propose_consensus(proposed_time, participants, 1000)
            .await
            .expect("Should create proposal");
        
        // Cast vote
        coordinator
            .cast_vote(proposal_id, VoteDecision::Accept, Some("Approved".to_string()))
            .await
            .expect("Should cast vote");
        
        // Verify vote was recorded
        let votes = coordinator.proposal_votes.read();
        let proposal_votes = votes.get(&proposal_id).expect("Should have votes for proposal");
        assert_eq!(proposal_votes.len(), 1);
        assert_eq!(proposal_votes[0].vote, VoteDecision::Accept);
    }

    #[test]
    fn test_vote_tally_calculation() {
        let synchronizer = Arc::new(DistributedSynchronizer::new(1, 5000));
        let coordinator = TemporalConsensusCoordinator::new(
            1,
            synchronizer,
            ConsensusAlgorithm::Raft,
        );
        
        let votes = vec![
            ConsensusVote {
                voter_node_id: 2,
                proposal_id: 1,
                vote: VoteDecision::Accept,
                vote_time: LogicalTime::new(1000, 0, 2),
                justification: None,
            },
            ConsensusVote {
                voter_node_id: 3,
                proposal_id: 1,
                vote: VoteDecision::Reject,
                vote_time: LogicalTime::new(1000, 1, 3),
                justification: None,
            },
            ConsensusVote {
                voter_node_id: 4,
                proposal_id: 1,
                vote: VoteDecision::Accept,
                vote_time: LogicalTime::new(1000, 2, 4),
                justification: None,
            },
        ];
        
        let tally = coordinator.calculate_vote_tally(&votes);
        assert_eq!(tally.accept_count, 2);
        assert_eq!(tally.reject_count, 1);
        assert_eq!(tally.abstain_count, 0);
        assert_eq!(tally.total_votes, 3);
    }
}
