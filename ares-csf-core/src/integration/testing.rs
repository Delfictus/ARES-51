//! Testing framework for CSF components.

use crate::error::{Error, Result};
use crate::types::{ComponentId, Timestamp};
use serde::{Deserialize, Serialize};

/// Test scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrppTestScenario {
    /// Scenario name
    pub name: String,
    /// Test description
    pub description: String,
    /// Components to test
    pub components: Vec<ComponentId>,
    /// Test parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Expected outcomes
    pub expected_outcomes: Vec<TestExpectation>,
}

/// Test expectation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExpectation {
    /// What to check
    pub check_type: String,
    /// Expected value
    pub expected_value: String,
    /// Tolerance for numeric comparisons
    pub tolerance: Option<f64>,
}

/// Test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Scenario that was run
    pub scenario: DrppTestScenario,
    /// Test execution timestamp
    pub timestamp: Timestamp,
    /// Overall test status
    pub status: TestStatus,
    /// Individual test outcomes
    pub outcomes: Vec<TestOutcome>,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed
    Passed,
    /// Test failed
    Failed,
    /// Test was skipped
    Skipped,
    /// Test encountered an error
    Error,
}

/// Individual test outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOutcome {
    /// Test name
    pub test_name: String,
    /// Test status
    pub status: TestStatus,
    /// Actual value observed
    pub actual_value: String,
    /// Expected value
    pub expected_value: String,
    /// Error message if failed
    pub error_message: Option<String>,
}

impl TestResults {
    /// Calculate overall pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        
        let passed = self.outcomes.iter()
            .filter(|outcome| outcome.status == TestStatus::Passed)
            .count();
        
        (passed as f64 / self.outcomes.len() as f64) * 100.0
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.status == TestStatus::Passed && 
        self.outcomes.iter().all(|outcome| outcome.status == TestStatus::Passed)
    }
}