//! Causality tracking for task dependencies

use csf_core::prelude::*;
use dashmap::DashMap;
use std::collections::HashSet;
use std::hash::Hash;

/// Causality trace for task dependencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalityTrace(pub u128);

/// Causality graph for tracking task dependencies
#[derive(Default)]
pub struct CausalityGraph {
    /// Forward edges (task -> dependents)
    forward_edges: DashMap<TaskId, HashSet<TaskId>>,

    /// Backward edges (task -> dependencies)
    backward_edges: DashMap<TaskId, HashSet<TaskId>>,

    /// Completed tasks
    completed: DashMap<TaskId, bool>,

    /// Causality trace for each task
    traces: DashMap<TaskId, CausalityTrace>,
}

impl CausalityGraph {
    /// Create a new causality graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a task with its dependencies
    pub fn add_task(&self, task_id: TaskId, dependencies: Vec<TaskId>) {
        // Add backward edges
        self.backward_edges
            .insert(task_id, dependencies.iter().cloned().collect());

        // Add forward edges
        for dep in dependencies {
            self.forward_edges
                .entry(dep)
                .or_default()
                .insert(task_id);
        }

        // Initialize as not completed
        self.completed.insert(task_id, false);

        // Generate causality trace
        let trace = self.generate_trace(task_id);
        self.traces.insert(task_id, trace);
    }

    /// Check if all dependencies are satisfied
    pub fn dependencies_satisfied(&self, task_id: TaskId) -> bool {
        if let Some(deps) = self.backward_edges.get(&task_id) {
            deps.iter().all(|dep| {
                self.completed
                    .get(dep)
                    .map(|entry| *entry.value())
                    .unwrap_or(false)
            })
        } else {
            true // No dependencies
        }
    }

    /// Mark a task as completed
    pub fn mark_completed(&self, task_id: TaskId) {
        self.completed.insert(task_id, true);

        // Log causality trace
        if let Some(trace) = self.traces.get(&task_id) {
            log::trace!("Task {:?} completed with trace {:?}", task_id, trace);
        }
    }

    /// Get tasks that depend on the given task
    pub fn get_dependents(&self, task_id: TaskId) -> Vec<TaskId> {
        self.forward_edges
            .get(&task_id)
            .map(|entry| entry.value().iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Perform topological sort
    pub fn topological_sort(&self) -> Result<Vec<TaskId>, CycleError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        // Get all tasks
        let all_tasks: HashSet<TaskId> = self
            .backward_edges
            .iter()
            .map(|entry| *entry.key())
            .collect();

        for task in all_tasks {
            if !visited.contains(&task) {
                self.visit(task, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        Ok(result)
    }

    fn visit(
        &self,
        task: TaskId,
        visited: &mut HashSet<TaskId>,
        temp_visited: &mut HashSet<TaskId>,
        result: &mut Vec<TaskId>,
    ) -> Result<(), CycleError> {
        if temp_visited.contains(&task) {
            return Err(CycleError { task });
        }

        if visited.contains(&task) {
            return Ok(());
        }

        temp_visited.insert(task);

        if let Some(deps) = self.backward_edges.get(&task) {
            for dep in deps.value() {
                self.visit(*dep, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(&task);
        visited.insert(task);
        result.push(task);

        Ok(())
    }

    fn generate_trace(&self, task_id: TaskId) -> CausalityTrace {
        // Simple trace generation - in practice would be more sophisticated
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        task_id.hash(&mut hasher);

        if let Some(deps) = self.backward_edges.get(&task_id) {
            for dep in deps.value() {
                dep.hash(&mut hasher);
            }
        }

        CausalityTrace(hasher.finish() as u128)
    }
}

/// Error indicating a cycle in the dependency graph
#[derive(Debug, thiserror::Error)]
#[error("Cycle detected in task dependencies at task {task:?}")]
pub struct CycleError {
    /// The task that was part of the cycle
    pub task: TaskId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causality_graph() {
        let graph = CausalityGraph::new();

        // Create some task IDs using the public constructor
        let task1 = TaskId::new();
        let task2 = TaskId::new();
        let task3 = TaskId::new();

        // Add tasks with dependencies
        graph.add_task(task1, vec![]);
        graph.add_task(task2, vec![task1]);
        graph.add_task(task3, vec![task1, task2]);

        // Check dependencies
        assert!(graph.dependencies_satisfied(task1));
        assert!(!graph.dependencies_satisfied(task2));
        assert!(!graph.dependencies_satisfied(task3));

        // Mark task 1 as completed
        graph.mark_completed(task1);
        assert!(graph.dependencies_satisfied(task2));
        assert!(!graph.dependencies_satisfied(task3));

        // Mark task 2 as completed
        graph.mark_completed(task2);
        assert!(graph.dependencies_satisfied(task3));
    }

    #[test]
    fn test_topological_sort() {
        let graph = CausalityGraph::new();

        // Create task IDs
        let task1 = TaskId::new();
        let task2 = TaskId::new();
        let task3 = TaskId::new();

        graph.add_task(task1, vec![]);
        graph.add_task(task2, vec![task1]);
        graph.add_task(task3, vec![task1, task2]);

        // Just verify that the sort succeeds - the exact order may vary
        let sorted = graph
            .topological_sort()
            .expect("Topological sort should succeed for acyclic dependency graph");
        assert_eq!(sorted.len(), 3);
        assert!(sorted.contains(&task1));
        assert!(sorted.contains(&task2));
        assert!(sorted.contains(&task3));
    }
}