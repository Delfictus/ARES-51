//! Advanced dependency resolution system
//!
//! This module provides sophisticated dependency resolution with topological sorting,
//! circular dependency detection, and conflict resolution for the CSF Runtime system.

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{Direction, Graph};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::core::ComponentId;
use crate::error::{DependencyError, RuntimeError, RuntimeResult};

/// Advanced dependency resolver with cycle detection and conflict resolution
#[derive(Debug)]
pub struct DependencyResolver {
    /// Component dependency graph
    dependency_graph: DependencyGraph,
    /// Dependency constraints and rules
    constraints: DependencyConstraints,
    /// Resolution cache for performance
    resolution_cache: HashMap<ComponentId, Vec<ComponentId>>,
}

/// Dependency graph implementation using petgraph
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Graph structure (directed)
    graph: Graph<ComponentId, DependencyType>,
    /// Component ID to node index mapping
    node_map: HashMap<ComponentId, NodeIndex>,
    /// Reverse mapping for efficient lookups
    index_map: HashMap<NodeIndex, ComponentId>,
}

/// Type of dependency relationship
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DependencyType {
    /// Hard dependency - component cannot start without this
    Required,
    /// Soft dependency - component prefers this but can work without it
    Optional,
    /// Initialization dependency - needed only during startup
    Initialization,
    /// Runtime dependency - needed during operation
    Runtime,
    /// Configuration dependency - shares configuration
    Configuration,
    /// Performance dependency - optimization relationship
    Performance,
}

/// Dependency constraints and validation rules
#[derive(Debug, Default)]
pub struct DependencyConstraints {
    /// Maximum dependency depth allowed
    max_depth: usize,
    /// Maximum dependencies per component
    max_dependencies_per_component: usize,
    /// Forbidden dependency pairs (circular prevention)
    forbidden_pairs: HashSet<(ComponentId, ComponentId)>,
    /// Required dependency pairs (architectural constraints)
    required_pairs: HashSet<(ComponentId, ComponentId)>,
    /// Dependency type constraints
    type_constraints: HashMap<ComponentId, HashSet<DependencyType>>,
}

/// Circular dependency error with detailed cycle information
#[derive(Debug, Clone)]
pub struct CircularDependencyError {
    /// The cycle chain
    pub cycle: Vec<ComponentId>,
    /// Dependency types in the cycle
    pub cycle_types: Vec<DependencyType>,
    /// Suggested resolution strategies
    pub resolution_strategies: Vec<ResolutionStrategy>,
}

/// Dependency resolution strategies
#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    /// Break cycle by making one dependency optional
    MakeOptional { from: ComponentId, to: ComponentId },
    /// Introduce mediator component
    IntroduceMediator { mediator: String },
    /// Defer initialization of one component
    DeferInitialization { component: ComponentId },
    /// Use event-driven communication instead of direct dependency
    UseEventDriven { components: Vec<ComponentId> },
}

/// Dependency analysis results
#[derive(Debug)]
pub struct DependencyAnalysis {
    /// Topologically sorted component order
    pub startup_order: Vec<ComponentId>,
    /// Shutdown order (reverse of startup)
    pub shutdown_order: Vec<ComponentId>,
    /// Critical path components
    pub critical_path: Vec<ComponentId>,
    /// Dependency depth per component
    pub component_depths: HashMap<ComponentId, usize>,
    /// Strongly connected components (potential cycles)
    pub strong_components: Vec<Vec<ComponentId>>,
    /// Optimization opportunities
    pub optimizations: Vec<OptimizationOpportunity>,
}

/// Dependency optimization opportunities
#[derive(Debug, Clone)]
pub enum OptimizationOpportunity {
    /// Components that can be initialized in parallel
    ParallelInitialization { components: Vec<ComponentId> },
    /// Lazy loading opportunity
    LazyLoading {
        component: ComponentId,
        trigger: String,
    },
    /// Dependency injection optimization
    DependencyInjection {
        component: ComponentId,
        dependencies: Vec<ComponentId>,
    },
    /// Configuration sharing optimization
    ConfigurationSharing { components: Vec<ComponentId> },
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            dependency_graph: DependencyGraph::new(),
            constraints: DependencyConstraints::default(),
            resolution_cache: HashMap::new(),
        }
    }

    /// Create resolver with custom constraints
    pub fn with_constraints(constraints: DependencyConstraints) -> Self {
        Self {
            dependency_graph: DependencyGraph::new(),
            constraints,
            resolution_cache: HashMap::new(),
        }
    }

    /// Add a component to the dependency graph
    pub fn add_component(&mut self, component_id: ComponentId) -> RuntimeResult<()> {
        self.dependency_graph.add_component(component_id)?;
        self.invalidate_cache();
        Ok(())
    }

    /// Add a dependency relationship between components
    pub fn add_dependency(
        &mut self,
        from: ComponentId,
        to: ComponentId,
        dependency_type: DependencyType,
    ) -> RuntimeResult<()> {
        // Validate constraint before adding
        self.validate_dependency(&from, &to, &dependency_type)?;

        self.dependency_graph
            .add_dependency(from.clone(), to.clone(), dependency_type)?;

        // Check for cycles after adding dependency
        if let Err(cycle_error) = self.check_for_cycles() {
            // Rollback the dependency addition
            self.dependency_graph.remove_dependency(&from, &to)?;
            return Err(RuntimeError::Dependency(
                DependencyError::CircularDependency {
                    chain: cycle_error.cycle,
                },
            ));
        }

        self.invalidate_cache();
        Ok(())
    }

    /// Remove a dependency relationship
    pub fn remove_dependency(&mut self, from: &ComponentId, to: &ComponentId) -> RuntimeResult<()> {
        self.dependency_graph.remove_dependency(from, to)?;
        self.invalidate_cache();
        Ok(())
    }

    /// Resolve dependencies and generate startup order
    pub fn resolve_dependencies(&mut self) -> RuntimeResult<DependencyAnalysis> {
        // Check for circular dependencies first
        if let Err(error) = self.check_for_cycles() {
            return Err(RuntimeError::Dependency(
                DependencyError::CircularDependency { chain: error.cycle },
            ));
        }

        // Perform topological sort
        let startup_order = self.topological_sort()?;
        let shutdown_order = startup_order.iter().rev().cloned().collect();

        // Calculate dependency depths
        let component_depths = self.calculate_depths(&startup_order)?;

        // Find critical path
        let critical_path = self.find_critical_path(&component_depths)?;

        // Find strongly connected components
        let strong_components = self.find_strongly_connected_components();

        // Identify optimization opportunities
        let optimizations = self.identify_optimizations(&startup_order, &component_depths)?;

        Ok(DependencyAnalysis {
            startup_order,
            shutdown_order,
            critical_path,
            component_depths,
            strong_components,
            optimizations,
        })
    }

    /// Validate a potential dependency against constraints
    fn validate_dependency(
        &self,
        from: &ComponentId,
        to: &ComponentId,
        dependency_type: &DependencyType,
    ) -> RuntimeResult<()> {
        // Check forbidden pairs
        if self
            .constraints
            .forbidden_pairs
            .contains(&(from.clone(), to.clone()))
        {
            return Err(RuntimeError::Dependency(
                DependencyError::ConflictingDependencies {
                    conflicts: vec![format!("{} -> {} is forbidden", from, to)],
                },
            ));
        }

        // Check type constraints
        if let Some(allowed_types) = self.constraints.type_constraints.get(from) {
            if !allowed_types.contains(dependency_type) {
                return Err(RuntimeError::Dependency(
                    DependencyError::ConflictingDependencies {
                        conflicts: vec![format!(
                            "{} cannot have {:?} dependency",
                            from, dependency_type
                        )],
                    },
                ));
            }
        }

        // Check maximum dependencies per component
        let current_deps = self.dependency_graph.get_dependencies(from).len();
        if current_deps >= self.constraints.max_dependencies_per_component {
            return Err(RuntimeError::Dependency(
                DependencyError::ConflictingDependencies {
                    conflicts: vec![format!(
                        "{} exceeds maximum dependencies limit of {}",
                        from, self.constraints.max_dependencies_per_component
                    )],
                },
            ));
        }

        Ok(())
    }

    /// Check for circular dependencies
    fn check_for_cycles(&self) -> Result<(), CircularDependencyError> {
        if is_cyclic_directed(&self.dependency_graph.graph) {
            // Find the actual cycle
            let cycle = match self.find_cycle() {
                Ok(cycle) => cycle,
                Err(_) => {
                    return Err(CircularDependencyError {
                        cycle: vec![],
                        cycle_types: vec![],
                        resolution_strategies: vec![],
                    })
                }
            };
            let cycle_types = self.get_cycle_types(&cycle);
            let resolution_strategies = self.generate_resolution_strategies(&cycle, &cycle_types);

            return Err(CircularDependencyError {
                cycle,
                cycle_types,
                resolution_strategies,
            });
        }
        Ok(())
    }

    /// Find a specific cycle in the graph
    fn find_cycle(&self) -> Result<Vec<ComponentId>, RuntimeError> {
        // Use DFS to find cycle
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut cycle_path = Vec::new();

        for node_index in self.dependency_graph.graph.node_indices() {
            let component_id = &self.dependency_graph.index_map[&node_index];
            if !visited.contains(component_id) {
                if self.dfs_cycle_detection(
                    component_id,
                    &mut visited,
                    &mut rec_stack,
                    &mut cycle_path,
                )? {
                    return Ok(cycle_path);
                }
            }
        }

        Err(RuntimeError::Dependency(
            DependencyError::ConflictingDependencies {
                conflicts: vec!["No cycle found in supposedly cyclic graph".to_string()],
            },
        ))
    }

    /// DFS-based cycle detection
    fn dfs_cycle_detection(
        &self,
        component: &ComponentId,
        visited: &mut HashSet<ComponentId>,
        rec_stack: &mut HashSet<ComponentId>,
        cycle_path: &mut Vec<ComponentId>,
    ) -> Result<bool, RuntimeError> {
        visited.insert(component.clone());
        rec_stack.insert(component.clone());
        cycle_path.push(component.clone());

        let dependencies = self.dependency_graph.get_dependencies(component);
        for (dep_component, _) in dependencies {
            if !visited.contains(&dep_component) {
                if self.dfs_cycle_detection(&dep_component, visited, rec_stack, cycle_path)? {
                    return Ok(true);
                }
            } else if rec_stack.contains(&dep_component) {
                // Found cycle - truncate path to cycle
                if let Some(cycle_start) = cycle_path.iter().position(|c| c == &dep_component) {
                    cycle_path.drain(0..cycle_start);
                    cycle_path.push(dep_component);
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(component);
        cycle_path.pop();
        Ok(false)
    }

    /// Get dependency types for a cycle
    fn get_cycle_types(&self, cycle: &[ComponentId]) -> Vec<DependencyType> {
        let mut types = Vec::new();
        for i in 0..cycle.len() {
            let from = &cycle[i];
            let to = &cycle[(i + 1) % cycle.len()];
            if let Some((_, dep_type)) = self
                .dependency_graph
                .get_dependencies(from)
                .iter()
                .find(|(comp, _)| comp == to)
            {
                types.push(dep_type.clone());
            }
        }
        types
    }

    /// Generate resolution strategies for a cycle
    fn generate_resolution_strategies(
        &self,
        cycle: &[ComponentId],
        cycle_types: &[DependencyType],
    ) -> Vec<ResolutionStrategy> {
        let mut strategies = Vec::new();

        // Strategy 1: Make optional dependencies optional
        for (i, dep_type) in cycle_types.iter().enumerate() {
            if matches!(
                dep_type,
                DependencyType::Runtime | DependencyType::Performance
            ) {
                strategies.push(ResolutionStrategy::MakeOptional {
                    from: cycle[i].clone(),
                    to: cycle[(i + 1) % cycle.len()].clone(),
                });
            }
        }

        // Strategy 2: Defer initialization
        for component in cycle {
            strategies.push(ResolutionStrategy::DeferInitialization {
                component: component.clone(),
            });
        }

        // Strategy 3: Use event-driven communication
        if cycle.len() > 2 {
            strategies.push(ResolutionStrategy::UseEventDriven {
                components: cycle.to_vec(),
            });
        }

        strategies
    }

    /// Perform topological sort
    fn topological_sort(&self) -> RuntimeResult<Vec<ComponentId>> {
        match toposort(&self.dependency_graph.graph, None) {
            Ok(sorted) => Ok(sorted
                .into_iter()
                .map(|idx| self.dependency_graph.index_map[&idx].clone())
                .collect()),
            Err(_) => Err(RuntimeError::Dependency(
                DependencyError::CircularDependency {
                    chain: vec![], // Will be filled by caller
                },
            )),
        }
    }

    /// Calculate dependency depths for each component
    fn calculate_depths(
        &self,
        startup_order: &[ComponentId],
    ) -> RuntimeResult<HashMap<ComponentId, usize>> {
        let mut depths = HashMap::new();

        for component in startup_order {
            let max_dep_depth = self
                .dependency_graph
                .get_dependencies(component)
                .iter()
                .map(|(dep, _)| depths.get(dep).unwrap_or(&0))
                .max()
                .unwrap_or(&0);

            let depth = max_dep_depth + 1;
            if depth > self.constraints.max_depth {
                return Err(RuntimeError::Dependency(DependencyError::DepthExceeded {
                    max_depth: self.constraints.max_depth,
                }));
            }

            depths.insert(component.clone(), depth);
        }

        Ok(depths)
    }

    /// Find critical path through dependencies
    fn find_critical_path(
        &self,
        depths: &HashMap<ComponentId, usize>,
    ) -> RuntimeResult<Vec<ComponentId>> {
        let max_depth = depths.values().max().unwrap_or(&0);
        let mut critical_path = Vec::new();

        // Find components at maximum depth
        let end_components: Vec<_> = depths
            .iter()
            .filter(|(_, &depth)| depth == *max_depth)
            .map(|(comp, _)| comp.clone())
            .collect();

        if let Some(end_component) = end_components.first() {
            self.build_critical_path(end_component, depths, &mut critical_path);
        }

        critical_path.reverse();
        Ok(critical_path)
    }

    /// Build critical path recursively
    fn build_critical_path(
        &self,
        component: &ComponentId,
        depths: &HashMap<ComponentId, usize>,
        path: &mut Vec<ComponentId>,
    ) {
        path.push(component.clone());

        let current_depth = depths.get(component).unwrap_or(&0);
        if *current_depth == 1 {
            return; // Reached root
        }

        // Find dependency with maximum depth
        let max_dep = self
            .dependency_graph
            .get_dependencies(component)
            .iter()
            .max_by_key(|(dep, _)| depths.get(dep).unwrap_or(&0))
            .map(|(dep, _)| dep.clone());

        if let Some(dep) = max_dep {
            self.build_critical_path(&dep, depths, path);
        }
    }

    /// Find strongly connected components
    fn find_strongly_connected_components(&self) -> Vec<Vec<ComponentId>> {
        // This is a simplified implementation
        // In a real system, we'd use Tarjan's algorithm
        vec![]
    }

    /// Identify optimization opportunities
    fn identify_optimizations(
        &self,
        startup_order: &[ComponentId],
        depths: &HashMap<ComponentId, usize>,
    ) -> RuntimeResult<Vec<OptimizationOpportunity>> {
        let mut optimizations = Vec::new();

        // Find parallel initialization opportunities
        let mut depth_groups: HashMap<usize, Vec<ComponentId>> = HashMap::new();
        for (component, &depth) in depths {
            depth_groups
                .entry(depth)
                .or_insert_with(Vec::new)
                .push(component.clone());
        }

        for (_, components) in depth_groups {
            if components.len() > 1 {
                optimizations.push(OptimizationOpportunity::ParallelInitialization { components });
            }
        }

        // Find lazy loading opportunities
        for component in startup_order {
            let dependencies = self.dependency_graph.get_dependencies(component);
            let optional_deps: Vec<_> = dependencies
                .iter()
                .filter(|(_, dep_type)| matches!(dep_type, DependencyType::Optional))
                .map(|(dep, _)| dep.clone())
                .collect();

            if !optional_deps.is_empty() {
                optimizations.push(OptimizationOpportunity::LazyLoading {
                    component: component.clone(),
                    trigger: format!("on_demand_for_{}", component.short_name()),
                });
            }
        }

        Ok(optimizations)
    }

    /// Invalidate resolution cache
    fn invalidate_cache(&mut self) {
        self.resolution_cache.clear();
    }
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_map: HashMap::new(),
            index_map: HashMap::new(),
        }
    }

    /// Add a component to the graph
    pub fn add_component(&mut self, component_id: ComponentId) -> RuntimeResult<()> {
        if self.node_map.contains_key(&component_id) {
            return Ok(()); // Already exists
        }

        let node_index = self.graph.add_node(component_id.clone());
        self.node_map.insert(component_id.clone(), node_index);
        self.index_map.insert(node_index, component_id);

        Ok(())
    }

    /// Add a dependency edge
    pub fn add_dependency(
        &mut self,
        from: ComponentId,
        to: ComponentId,
        dependency_type: DependencyType,
    ) -> RuntimeResult<()> {
        // Ensure both components exist
        self.add_component(from.clone())?;
        self.add_component(to.clone())?;

        let from_idx = self.node_map[&from];
        let to_idx = self.node_map[&to];

        self.graph.add_edge(from_idx, to_idx, dependency_type);
        Ok(())
    }

    /// Remove a dependency edge
    pub fn remove_dependency(&mut self, from: &ComponentId, to: &ComponentId) -> RuntimeResult<()> {
        let from_idx = self.node_map.get(from).ok_or_else(|| {
            RuntimeError::Dependency(DependencyError::UnresolvableDependency {
                component: from.clone(),
                dependency: to.clone(),
            })
        })?;
        let to_idx = self.node_map.get(to).ok_or_else(|| {
            RuntimeError::Dependency(DependencyError::UnresolvableDependency {
                component: from.clone(),
                dependency: to.clone(),
            })
        })?;

        // Find and remove edge
        if let Some(edge) = self.graph.find_edge(*from_idx, *to_idx) {
            self.graph.remove_edge(edge);
        }

        Ok(())
    }

    /// Get all dependencies for a component
    pub fn get_dependencies(&self, component: &ComponentId) -> Vec<(ComponentId, DependencyType)> {
        if let Some(&node_idx) = self.node_map.get(component) {
            self.graph
                .edges_directed(node_idx, Direction::Outgoing)
                .map(|edge| {
                    let target_component = &self.index_map[&edge.target()];
                    (target_component.clone(), edge.weight().clone())
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all components that depend on this component
    pub fn get_dependents(&self, component: &ComponentId) -> Vec<(ComponentId, DependencyType)> {
        if let Some(&node_idx) = self.node_map.get(component) {
            self.graph
                .edges_directed(node_idx, Direction::Incoming)
                .map(|edge| {
                    let source_component = &self.index_map[&edge.source()];
                    (source_component.clone(), edge.weight().clone())
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyConstraints {
    /// Create new dependency constraints
    pub fn new() -> Self {
        Self {
            max_depth: crate::MAX_DEPENDENCY_DEPTH,
            max_dependencies_per_component: 50,
            forbidden_pairs: HashSet::new(),
            required_pairs: HashSet::new(),
            type_constraints: HashMap::new(),
        }
    }

    /// Set maximum dependency depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set maximum dependencies per component
    pub fn with_max_dependencies_per_component(mut self, max_deps: usize) -> Self {
        self.max_dependencies_per_component = max_deps;
        self
    }

    /// Add a forbidden dependency pair
    pub fn forbid_dependency(mut self, from: ComponentId, to: ComponentId) -> Self {
        self.forbidden_pairs.insert((from, to));
        self
    }

    /// Add a required dependency pair
    pub fn require_dependency(mut self, from: ComponentId, to: ComponentId) -> Self {
        self.required_pairs.insert((from, to));
        self
    }
}

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyType::Required => write!(f, "Required"),
            DependencyType::Optional => write!(f, "Optional"),
            DependencyType::Initialization => write!(f, "Initialization"),
            DependencyType::Runtime => write!(f, "Runtime"),
            DependencyType::Configuration => write!(f, "Configuration"),
            DependencyType::Performance => write!(f, "Performance"),
        }
    }
}

impl fmt::Display for CircularDependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circular dependency detected: ")?;
        for (i, component) in self.cycle.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{}", component)?;
        }
        if !self.cycle.is_empty() {
            write!(f, " -> {}", self.cycle[0])?;
        }
        Ok(())
    }
}

impl std::error::Error for CircularDependencyError {}
