//! A production-grade Temporal Coherence Framework for the ARES CSF.
//!
//! This module provides the `CausalityEnforcementEngine`, which is responsible for
//! tracking dependencies between distributed operations and ensuring they are
//! executed in a causally consistent order. This is a critical component for
//! achieving ChronoSynclastic determinism.

use crate::{TimeSource, SystemTimeSource};
use csf_shared_types::{TaskId, NanoTime};
use std::sync::Arc;
use thiserror::Error;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};

// --- Core Data Structures ---

/// Represents an operation with explicit causal dependencies.
#[derive(Debug, Clone)]
pub struct CausalOperation<T> {
    /// The unique identifier for this operation's resulting task.
    pub task_id: TaskId,
    /// A list of `TaskId`s that must be completed before this operation can run.
    pub dependencies: Vec<TaskId>,
    /// The actual payload or logic to be executed.
    pub payload: T,
}

/// A graph representing the causal dependencies between tasks.
#[derive(Debug, Default)]
pub struct CausalDependencyGraph {
    /// Adjacency list for forward edges (task -> dependents).
    nodes: DashMap<TaskId, HashSet<TaskId>>,
    /// Adjacency list for backward edges (task -> dependencies).
    reverse_nodes: DashMap<TaskId, HashSet<TaskId>>,
}

impl CausalDependencyGraph {
    /// Builds a dependency graph from a slice of causal operations.
    pub fn build<T>(operations: &[CausalOperation<T>]) -> Result<Self, CausalityError> {
        let graph = Self::default();
        let task_ids: HashSet<TaskId> = operations.iter().map(|op| op.task_id).collect();

        for op in operations {
            graph.nodes.insert(op.task_id, HashSet::new());
            graph.reverse_nodes.insert(op.task_id, op.dependencies.iter().cloned().collect());

            for dep_id in &op.dependencies {
                // Ensure the dependency exists within the set of operations.
                if !task_ids.contains(dep_id) {
                    return Err(CausalityError::MissingDependency { task_id: op.task_id, dependency_id: *dep_id });
                }
                // Add the forward edge from the dependency to the current task.
                graph.nodes.entry(*dep_id).or_default().insert(op.task_id);
            }
        }
        Ok(graph)
    }
}


/// A list of causality violations detected in a dependency graph.
#[derive(Debug, Clone)]
pub struct CausalityViolation {
    /// The ID of the task where the violation was detected.
    pub task_id: TaskId,
    /// A description of the violation (e.g., "Cyclic dependency detected").
    pub reason: String,
}

// --- Error Types ---

/// Errors that can occur during causality enforcement.
#[derive(Debug, Error)]
pub enum CausalityError {
    /// One or more causality violations (e.g., cycles) were detected.
    #[error("Causality violations detected")]
    ViolationsDetected(Vec<CausalityViolation>),
    /// A required dependency was not found in the graph.
    #[error("Missing dependency {dependency_id:?} for task {task_id:?}")]
    MissingDependency {
        task_id: TaskId,
        dependency_id: TaskId,
    },
    /// An IO or other external error occurred.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// --- Engine and Components ---

/// Detects violations within a `CausalDependencyGraph`.
#[derive(Debug, Default)]
pub struct CausalityViolationDetector;

impl CausalityViolationDetector {
    /// Detects cycles in the dependency graph using a depth-first search.
    /// Returns a list of all violations found.
    pub fn detect_violations(&self, graph: &CausalDependencyGraph) -> Vec<CausalityViolation> {
        let mut violations = Vec::new();
        let mut visiting = HashSet::new(); // Nodes currently in the recursion stack.
        let mut visited = HashSet::new();  // Nodes that have been fully processed.

        for task_id in graph.nodes.iter().map(|entry| *entry.key()) {
            if !visited.contains(&task_id) {
                if let Err(violation) = self.dfs_visit(task_id, graph, &mut visiting, &mut visited) {
                    violations.push(violation);
                    // Note: We could stop after the first error, but collecting all
                    // violations can be useful for debugging complex scenarios.
                }
            }
        }
        violations
    }

    /// Recursive DFS helper for cycle detection.
    fn dfs_visit(
        &self,
        task_id: TaskId,
        graph: &CausalDependencyGraph,
        visiting: &mut HashSet<TaskId>,
        visited: &mut HashSet<TaskId>,
    ) -> Result<(), CausalityViolation> {
        visiting.insert(task_id);

        if let Some(dependents) = graph.nodes.get(&task_id) {
            for dependent_id in dependents.iter() {
                if visiting.contains(dependent_id) {
                    return Err(CausalityViolation { task_id: *dependent_id, reason: "Cyclic dependency detected".to_string() });
                }
                if !visited.contains(dependent_id) {
                    self.dfs_visit(*dependent_id, graph, visiting, visited)?;
                }
            }
        }
        visiting.remove(&task_id);
        visited.insert(task_id);
        Ok(())
    }
}

/// The core engine for ensuring temporal coherence.
///
/// It orchestrates the building of a dependency graph, detects violations,
/// and produces a causally-correct execution plan.
pub struct CausalityEnforcementEngine {
    time_source: Arc<dyn TimeSource>,
    violation_detector: CausalityViolationDetector,
}

impl CausalityEnforcementEngine {
    /// Creates a new `CausalityEnforcementEngine`.
    pub fn new(time_source: Arc<dyn TimeSource>) -> Self {
        Self {
            time_source,
            violation_detector: CausalityViolationDetector::default(),
        }
    }

    /// Enforces causal ordering for a set of distributed operations.
    ///
    /// This function performs the full pipeline:
    /// 1. Builds a dependency graph.
    /// 2. Detects causality violations (like cycles).
    /// 3. Produces a topologically sorted execution plan.
    ///
    /// # Returns
    /// A `Result` containing either a topologically sorted list of `TaskId`s
    /// representing a valid execution plan, or a `CausalityError`.
    #[tracing::instrument(name = "enforce_causal_ordering", skip(self, operations))]
    pub fn enforce_causal_ordering<T>(
        &self,
        operations: &[CausalOperation<T>],
    ) -> Result<Vec<TaskId>, CausalityError> {
        let _start_time = self.time_source.now_ns();

        // Phase 1: Build causal dependency graph
        let dependency_graph = CausalDependencyGraph::build(operations)?;

        // Phase 2: Detect potential violations
        let violations = self.violation_detector.detect_violations(&dependency_graph);
        if !violations.is_empty() {
            return Err(CausalityError::ViolationsDetected(violations));
        }

        // Phase 3: Create execution plan via topological sort.
        let execution_plan = self.create_execution_plan(&dependency_graph)?;

        // The actual execution of the plan would happen here, but for now,
        // we return the plan itself.

        Ok(execution_plan)
    }

    /// Creates a valid execution plan by performing a topological sort of the graph.
    fn create_execution_plan(&self, graph: &CausalDependencyGraph) -> Result<Vec<TaskId>, CausalityError> {
        let mut in_degree: HashMap<TaskId, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut sorted_order = Vec::new();

        for task_id in graph.nodes.iter().map(|e| *e.key()) {
            let degree = graph.reverse_nodes.get(&task_id).map_or(0, |deps| deps.len());
            in_degree.insert(task_id, degree);
            if degree == 0 {
                queue.push_back(task_id);
            }
        }

        while let Some(task_id) = queue.pop_front() {
            sorted_order.push(task_id);
            if let Some(dependents) = graph.nodes.get(&task_id) {
                for dependent_id in dependents.iter() {
                    let degree = in_degree.entry(*dependent_id).or_insert(0);
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*dependent_id);
                    }
                }
            }
        }

        if sorted_order.len() != graph.nodes.len() {
            // This should be caught by the cycle detector, but serves as a safeguard.
            return Err(CausalityError::ViolationsDetected(vec![CausalityViolation {
                task_id: TaskId::new(), // Placeholder ID
                reason: "Topological sort failed; graph may contain a cycle.".to_string(),
            }]));
        }

        Ok(sorted_order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemTimeSource;

    // Implicit From<u64> for TaskId for easier testing.
    impl From<u64> for TaskId {
        fn from(id: u64) -> Self {
            TaskId(id)
        }
    }

    fn new_op(id: u64, deps: Vec<u64>) -> CausalOperation<()> {
        CausalOperation {
            task_id: TaskId::from(id),
            dependencies: deps.into_iter().map(TaskId::from).collect(),
            payload: (),
        }
    }

    #[test]
    fn test_valid_acyclic_graph_enforcement() {
        let time_source = Arc::new(SystemTimeSource);
        let engine = CausalityEnforcementEngine::new(time_source);
        let operations = vec![
            new_op(1, vec![]),
            new_op(2, vec![1]),
            new_op(3, vec![1]),
            new_op(4, vec![2, 3]),
        ];

        let plan = engine.enforce_causal_ordering(&operations).unwrap();

        assert_eq!(plan.len(), 4);
        let pos1 = plan.iter().position(|&id| id == TaskId::from(1)).unwrap();
        let pos2 = plan.iter().position(|&id| id == TaskId::from(2)).unwrap();
        let pos3 = plan.iter().position(|&id| id == TaskId::from(3)).unwrap();
        let pos4 = plan.iter().position(|&id| id == TaskId::from(4)).unwrap();

        assert!(pos1 < pos2, "Task 1 must come before Task 2");
        assert!(pos1 < pos3, "Task 1 must come before Task 3");
        assert!(pos2 < pos4, "Task 2 must come before Task 4");
        assert!(pos3 < pos4, "Task 3 must come before Task 4");
    }

    #[test]
    fn test_cycle_detection() {
        let time_source = Arc::new(SystemTimeSource);
        let engine = CausalityEnforcementEngine::new(time_source);
        let operations = vec![
            new_op(1, vec![3]), // Cycle: 1 -> 2 -> 3 -> 1
            new_op(2, vec![1]),
            new_op(3, vec![2]),
        ];

        let result = engine.enforce_causal_ordering(&operations);
        assert!(matches!(result, Err(CausalityError::ViolationsDetected(_))));

        if let Err(CausalityError::ViolationsDetected(violations)) = result {
            assert!(!violations.is_empty());
            assert!(violations.iter().any(|v| v.reason.contains("Cyclic dependency")));
        }
    }

    #[test]
    fn test_missing_dependency_error() {
        let time_source = Arc::new(SystemTimeSource);
        let engine = CausalityEnforcementEngine::new(time_source);
        let operations = vec![
            new_op(1, vec![]),
            new_op(2, vec![3]), // Dependency '3' is not in the set of operations.
        ];

        let result = engine.enforce_causal_ordering(&operations);
        assert!(matches!(result, Err(CausalityError::MissingDependency { .. })));
    }
}
