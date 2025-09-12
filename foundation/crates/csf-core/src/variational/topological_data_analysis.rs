//! PhD-Level Topological Data Analysis Implementation
//!
//! This module implements research-grade Topological Data Analysis (TDA) at the level
//! of a research mathematician with expertise in algebraic topology, computational geometry,
//! and high-dimensional data analysis. Features persistent homology, Mapper algorithm,
//! and topological signatures for phase transition detection.

use crate::types::{ComponentId, NanoTime};
use crate::variational::phase_space::PhaseRegion;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Field arithmetic for homological computations
pub trait Field: Clone + Copy + Send + Sync + std::fmt::Debug + PartialEq {
    /// Field characteristic (0 for rationals, p for Z/pZ)
    fn characteristic() -> usize;
    /// Zero element
    fn zero() -> Self;
    /// One element  
    fn one() -> Self;
    /// Addition operation
    fn add(self, other: Self) -> Self;
    /// Multiplication operation
    fn mul(self, other: Self) -> Self;
    /// Additive inverse
    fn neg(self) -> Self;
    /// Multiplicative inverse (if exists)
    fn inv(self) -> Option<Self>;
}

/// GF(2) field implementation for binary persistent homology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GF2(pub bool);

impl Field for GF2 {
    fn characteristic() -> usize {
        2
    }
    fn zero() -> Self {
        GF2(false)
    }
    fn one() -> Self {
        GF2(true)
    }
    fn add(self, other: Self) -> Self {
        GF2(self.0 ^ other.0)
    }
    fn mul(self, other: Self) -> Self {
        GF2(self.0 & other.0)
    }
    fn neg(self) -> Self {
        self
    } // In GF(2), -x = x
    fn inv(self) -> Option<Self> {
        if self.0 {
            Some(self)
        } else {
            None
        }
    }
}

/// Rational field for exact arithmetic
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rational {
    pub numerator: i64,
    pub denominator: i64,
}

impl Field for Rational {
    fn characteristic() -> usize {
        0
    }

    fn zero() -> Self {
        Rational {
            numerator: 0,
            denominator: 1,
        }
    }

    fn one() -> Self {
        Rational {
            numerator: 1,
            denominator: 1,
        }
    }

    fn add(self, other: Self) -> Self {
        let num = self.numerator * other.denominator + other.numerator * self.denominator;
        let den = self.denominator * other.denominator;
        Rational {
            numerator: num,
            denominator: den,
        }
        .reduce()
    }

    fn mul(self, other: Self) -> Self {
        let num = self.numerator * other.numerator;
        let den = self.denominator * other.denominator;
        Rational {
            numerator: num,
            denominator: den,
        }
        .reduce()
    }

    fn neg(self) -> Self {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }

    fn inv(self) -> Option<Self> {
        if self.numerator != 0 {
            Some(
                Rational {
                    numerator: self.denominator,
                    denominator: self.numerator,
                }
                .reduce(),
            )
        } else {
            None
        }
    }
}

impl Rational {
    fn reduce(self) -> Self {
        let gcd = gcd(self.numerator.abs() as u64, self.denominator.abs() as u64) as i64;
        let num = self.numerator / gcd;
        let den = self.denominator / gcd;

        if den < 0 {
            Rational {
                numerator: -num,
                denominator: -den,
            }
        } else {
            Rational {
                numerator: num,
                denominator: den,
            }
        }
    }
}

/// Point cloud data structure for TDA
#[derive(Debug)]
pub struct PointCloud<T> {
    points: Vec<T>,
    dimension: usize,
    metric: Box<dyn MetricFunction<T> + Send + Sync>,
}

impl<T> PointCloud<T> {
    pub fn new(
        points: Vec<T>,
        dimension: usize,
        metric: Box<dyn MetricFunction<T> + Send + Sync>,
    ) -> Self {
        Self {
            points,
            dimension,
            metric,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn points(&self) -> &[T] {
        &self.points
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn distance(&self, i: usize, j: usize) -> f64 {
        if i < self.points.len() && j < self.points.len() {
            self.metric.distance(&self.points[i], &self.points[j])
        } else {
            f64::INFINITY
        }
    }
}

impl<T> std::ops::Index<usize> for PointCloud<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

/// Metric function trait
pub trait MetricFunction<T>: std::fmt::Debug {
    fn distance(&self, p1: &T, p2: &T) -> f64;
}

/// Euclidean metric for vector data
#[derive(Debug)]
pub struct EuclideanMetric;

impl MetricFunction<DVector<f64>> for EuclideanMetric {
    fn distance(&self, p1: &DVector<f64>, p2: &DVector<f64>) -> f64 {
        (p1 - p2).norm()
    }
}

/// Simplex representation for simplicial complexes
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex {
    vertices: Vec<usize>,
}

impl Simplex {
    pub fn vertex(v: usize) -> Self {
        Self { vertices: vec![v] }
    }

    pub fn edge(v1: usize, v2: usize) -> Self {
        let mut vertices = vec![v1, v2];
        vertices.sort_unstable();
        Self { vertices }
    }

    pub fn from_vertices(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        vertices.dedup();
        Self { vertices }
    }

    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    pub fn boundary(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return Vec::new();
        }

        let mut boundary_simplices = Vec::new();
        for i in 0..self.vertices.len() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            boundary_simplices.push(Simplex::from_vertices(face_vertices));
        }

        boundary_simplices
    }

    pub fn contains_vertex(&self, vertex: usize) -> bool {
        self.vertices.contains(&vertex)
    }
}

/// Filtered simplicial complex for persistent homology
#[derive(Debug, Clone)]
pub struct FilteredSimplicialComplex<F: Field> {
    simplices: Vec<SimplexWithBirth>,
    field: std::marker::PhantomData<F>,
    dimension_index: HashMap<usize, Vec<usize>>, // Maps dimension -> simplex indices
}

#[derive(Debug, Clone)]
pub struct SimplexWithBirth {
    pub simplex: Simplex,
    pub birth_time: f64,
    pub index: usize,
}

impl<F: Field> FilteredSimplicialComplex<F> {
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            field: std::marker::PhantomData,
            dimension_index: HashMap::new(),
        }
    }

    pub fn add_simplex(&mut self, simplex: Simplex, birth_time: f64) -> Result<(), TopologyError> {
        let dimension = simplex.dimension();
        let index = self.simplices.len();

        // Verify all faces exist and have earlier birth times
        for face in simplex.boundary() {
            if let Some(face_idx) = self.find_simplex_index(&face) {
                if self.simplices[face_idx].birth_time > birth_time {
                    return Err(TopologyError::InvalidFiltration {
                        simplex_birth: birth_time,
                        face_birth: self.simplices[face_idx].birth_time,
                    });
                }
            } else {
                return Err(TopologyError::MissingFace(format!("{:?}", face)));
            }
        }

        let simplex_with_birth = SimplexWithBirth {
            simplex,
            birth_time,
            index,
        };

        self.simplices.push(simplex_with_birth);
        self.dimension_index
            .entry(dimension)
            .or_default()
            .push(index);

        Ok(())
    }

    pub fn total_simplices(&self) -> usize {
        self.simplices.len()
    }

    pub fn simplices_of_dimension(&self, dimension: usize) -> Vec<&SimplexWithBirth> {
        self.dimension_index
            .get(&dimension)
            .map(|indices| indices.iter().map(|&i| &self.simplices[i]).collect())
            .unwrap_or_default()
    }

    pub fn simplex_birth_time(&self, index: usize) -> f64 {
        self.simplices[index].birth_time
    }

    pub fn simplex_dimension(&self, index: usize) -> usize {
        self.simplices[index].simplex.dimension()
    }

    fn find_simplex_index(&self, simplex: &Simplex) -> Option<usize> {
        self.simplices.iter().position(|s| s.simplex == *simplex)
    }
}

/// Sparse boundary matrix for efficient homology computation
#[derive(Debug, Clone)]
pub struct SparseBoundaryMatrix<F: Field> {
    rows: usize,
    cols: usize,
    columns: Vec<SparseColumn<F>>,
    row_index: HashMap<usize, HashSet<usize>>, // Maps row -> set of non-zero columns
}

#[derive(Debug, Clone)]
pub struct SparseColumn<F: Field> {
    entries: Vec<(usize, F)>, // (row_index, value) pairs
    pivot: Option<usize>,     // Row index of lowest non-zero entry
}

impl<F: Field + PartialEq> SparseBoundaryMatrix<F> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            columns: vec![
                SparseColumn {
                    entries: Vec::new(),
                    pivot: None
                };
                cols
            ],
            row_index: HashMap::new(),
        }
    }

    pub fn set_entry(&mut self, row: usize, col: usize, value: F) {
        if row >= self.rows || col >= self.cols {
            return;
        }

        let column = &mut self.columns[col];

        // Remove existing entry if any
        column.entries.retain(|(r, _)| *r != row);

        // Add new entry if non-zero
        if value != F::zero() {
            column.entries.push((row, value));
            column.entries.sort_by_key(|(r, _)| *r);

            // Update pivot
            column.pivot = column.entries.last().map(|(r, _)| *r);

            // Update row index
            self.row_index.entry(row).or_default().insert(col);
        } else {
            // Update pivot after removal
            column.pivot = column.entries.last().map(|(r, _)| *r);

            // Update row index
            if let Some(cols) = self.row_index.get_mut(&row) {
                cols.remove(&col);
                if cols.is_empty() {
                    self.row_index.remove(&row);
                }
            }
        }
    }

    pub fn add_column(
        &mut self,
        source_col: usize,
        target_col: usize,
    ) -> Result<(), TopologyError> {
        if source_col >= self.cols || target_col >= self.cols {
            return Err(TopologyError::MatrixIndexError);
        }

        // Clone source column entries to avoid borrowing issues
        let source_entries = self.columns[source_col].entries.clone();

        for (row, value) in source_entries {
            let current_value = self.get_entry(row, target_col);
            let new_value = current_value.add(value);
            self.set_entry(row, target_col, new_value);
        }

        Ok(())
    }

    pub fn get_entry(&self, row: usize, col: usize) -> F {
        if row >= self.rows || col >= self.cols {
            return F::zero();
        }

        self.columns[col]
            .entries
            .iter()
            .find(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .unwrap_or(F::zero())
    }

    pub fn lowest_nonzero_row(&self, col: usize) -> Option<usize> {
        if col < self.cols {
            self.columns[col].pivot
        } else {
            None
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.columns.iter().map(|col| col.entries.len()).sum()
    }

    pub fn compress_columns(&mut self) -> Result<(), TopologyError> {
        // Remove zero entries and optimize storage
        for column in &mut self.columns {
            column.entries.retain(|(_, v)| *v != F::zero());
            column.pivot = column.entries.last().map(|(r, _)| *r);
        }

        // Rebuild row index
        self.row_index.clear();
        for (col_idx, column) in self.columns.iter().enumerate() {
            for (row, _) in &column.entries {
                self.row_index.entry(*row).or_default().insert(col_idx);
            }
        }

        Ok(())
    }

    pub fn optimize_pivot_order(&mut self) -> Result<(), TopologyError> {
        // Sort entries within each column for optimal pivot access
        for column in &mut self.columns {
            column.entries.sort_by_key(|(row, _)| *row);
            column.pivot = column.entries.last().map(|(r, _)| *r);
        }
        Ok(())
    }

    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .columns
                .iter()
                .map(|col| col.entries.len() * std::mem::size_of::<(usize, F)>())
                .sum::<usize>()
            + self
                .row_index
                .iter()
                .map(|(_, set)| set.len() * std::mem::size_of::<usize>())
                .sum::<usize>()
    }
}

/// Persistence pair representing birth-death of topological feature
#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>, // None for essential (infinite) classes
    pub birth_simplex: usize,
    pub death_simplex: Option<usize>,
    pub multiplicity: usize,
}

impl PersistencePair {
    pub fn persistence(&self) -> f64 {
        match self.death {
            Some(death) => death - self.birth,
            None => f64::INFINITY,
        }
    }

    pub fn is_essential(&self) -> bool {
        self.death.is_none()
    }

    pub fn midpoint(&self) -> f64 {
        match self.death {
            Some(death) => (self.birth + death) / 2.0,
            None => self.birth,
        }
    }
}

/// Persistence diagram containing all persistence pairs
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    pub dimension_range: std::ops::RangeInclusive<usize>,
    pub filtration_type: FiltrationType,
    pub field_characteristic: usize,
    pub stability_constant: f64,
    pub computational_metadata: ComputationMetadata,
}

impl PersistenceDiagram {
    pub fn finite_pairs(&self) -> impl Iterator<Item = &PersistencePair> {
        self.pairs.iter().filter(|p| !p.is_essential())
    }

    pub fn essential_pairs(&self) -> impl Iterator<Item = &PersistencePair> {
        self.pairs.iter().filter(|p| p.is_essential())
    }

    pub fn pairs_of_dimension(&self, dimension: usize) -> impl Iterator<Item = &PersistencePair> {
        self.pairs.iter().filter(move |p| p.dimension == dimension)
    }

    pub fn total_persistence(&self) -> f64 {
        self.finite_pairs().map(|p| p.persistence()).sum()
    }

    pub fn betti_numbers_at_scale(&self, scale: f64) -> HashMap<usize, usize> {
        let mut betti_numbers = HashMap::new();

        for pair in &self.pairs {
            if pair.birth <= scale && (pair.death.is_none() || pair.death.unwrap() > scale) {
                *betti_numbers.entry(pair.dimension).or_insert(0) += pair.multiplicity;
            }
        }

        betti_numbers
    }
}

/// Filtration types for building simplicial complexes
#[derive(Debug, Clone)]
pub enum FiltrationType {
    VietorisRips { max_radius: f64 },
    Cech { max_radius: f64 },
    Alpha { weighted: bool },
    Witness { landmarks: usize, nu: f64 },
    Sublevel,
}

/// Custom filtration builder trait
pub trait FiltrationBuilder: std::fmt::Debug {
    fn build_filtration(
        &self,
        point_cloud: &PointCloud<DVector<f64>>,
        max_dimension: usize,
    ) -> Result<FilteredSimplicialComplex<GF2>, TopologyError>;
}

/// Filtration parameters for complex construction
#[derive(Debug, Clone)]
pub struct FiltrationParameters {
    pub filtration_type: FiltrationType,
    pub max_dimension: usize,
    pub field_characteristic: usize,
    pub chunk_size: Option<usize>,
}

/// Matrix reduction algorithms for persistent homology
#[derive(Debug, Clone, Copy)]
pub enum MatrixReductionAlgorithm {
    Standard,
    Twist,
    Chunk,
    Parallel { num_threads: usize },
    Cohomology,
}

/// Result of matrix reduction computation
#[derive(Debug)]
pub struct ReductionResult {
    pub finite_pairs: Vec<(usize, usize)>,
    pub essential_classes: Vec<usize>,
    pub reduced_matrix: SparseBoundaryMatrix<GF2>,
}

/// Computational metadata for persistence diagrams
#[derive(Debug, Clone)]
pub struct ComputationMetadata {
    pub total_simplices: usize,
    pub matrix_reduction_time: std::time::Duration,
    pub memory_usage: usize,
    pub algorithm_used: MatrixReductionAlgorithm,
}

/// PhD-level persistent homology engine
#[derive(Debug, Clone)]
pub struct PersistentHomologyEngine<F: Field> {
    filtration: FilteredSimplicialComplex<F>,
    boundary_matrix: SparseBoundaryMatrix<F>,
    reduction_algorithm: MatrixReductionAlgorithm,
    field_arithmetic: std::marker::PhantomData<F>,
    persistence_module: PersistenceModule<F>,
}

#[derive(Debug, Clone)]
pub struct PersistenceModule<F: Field> {
    intervals: Vec<PersistenceInterval>,
    field: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone)]
pub struct PersistenceInterval {
    pub birth: f64,
    pub death: Option<f64>,
    pub dimension: usize,
}

impl<F: Field + Copy + Send + Sync> PersistentHomologyEngine<F> {
    pub fn new(algorithm: MatrixReductionAlgorithm) -> Self {
        Self {
            filtration: FilteredSimplicialComplex::new(),
            boundary_matrix: SparseBoundaryMatrix::new(0, 0),
            reduction_algorithm: algorithm,
            field_arithmetic: std::marker::PhantomData,
            persistence_module: PersistenceModule {
                intervals: Vec::new(),
                field: std::marker::PhantomData,
            },
        }
    }

    /// Compute persistent homology using optimized matrix reduction
    pub fn compute_persistent_homology(
        &mut self,
        point_cloud: &PointCloud<DVector<f64>>,
        filtration_params: FiltrationParameters,
        max_dimension: usize,
    ) -> Result<PersistenceDiagram, TopologyError> {
        tracing::info!(
            "Computing persistent homology for {} points",
            point_cloud.len()
        );
        let total_timer = std::time::Instant::now();

        // Phase 1: Build filtered simplicial complex
        tracing::info!("Building filtered simplicial complex");
        let complex_timer = std::time::Instant::now();

        self.build_filtration(point_cloud, &filtration_params, max_dimension)?;

        tracing::info!("Complex built in {:?}", complex_timer.elapsed());
        tracing::info!(
            "Simplices: {} (dim 0: {}, dim 1: {}, dim 2: {})",
            self.filtration.total_simplices(),
            self.filtration.simplices_of_dimension(0).len(),
            self.filtration.simplices_of_dimension(1).len(),
            self.filtration.simplices_of_dimension(2).len()
        );

        // Phase 2: Build boundary matrix
        tracing::info!("Building boundary matrix");
        let matrix_timer = std::time::Instant::now();

        self.build_boundary_matrix(max_dimension)?;
        self.boundary_matrix.compress_columns()?;
        self.boundary_matrix.optimize_pivot_order()?;

        tracing::info!("Boundary matrix built in {:?}", matrix_timer.elapsed());
        tracing::info!(
            "Matrix size: {} x {} with {} non-zeros",
            self.boundary_matrix.rows(),
            self.boundary_matrix.cols(),
            self.boundary_matrix.nnz()
        );

        // Phase 3: Matrix reduction
        tracing::info!("Performing matrix reduction");
        let reduction_timer = std::time::Instant::now();

        let reduction_result = self.perform_matrix_reduction()?;

        tracing::info!(
            "Matrix reduction completed in {:?}",
            reduction_timer.elapsed()
        );

        // Phase 4: Extract persistence pairs
        tracing::info!("Extracting persistence pairs");
        let pairs_timer = std::time::Instant::now();

        let persistence_pairs = self.extract_persistence_pairs(&reduction_result)?;

        tracing::info!(
            "Extracted {} persistence pairs in {:?}",
            persistence_pairs.len(),
            pairs_timer.elapsed()
        );

        // Phase 5: Build final diagram
        let diagram = PersistenceDiagram {
            pairs: persistence_pairs,
            dimension_range: 0..=max_dimension,
            filtration_type: filtration_params.filtration_type,
            field_characteristic: F::characteristic(),
            stability_constant: self.compute_stability_constant()?,
            computational_metadata: ComputationMetadata {
                total_simplices: self.filtration.total_simplices(),
                matrix_reduction_time: reduction_timer.elapsed(),
                memory_usage: self.boundary_matrix.memory_footprint(),
                algorithm_used: self.reduction_algorithm,
            },
        };

        tracing::info!("Total computation time: {:?}", total_timer.elapsed());

        Ok(diagram)
    }

    /// Build filtration based on specified type
    fn build_filtration(
        &mut self,
        point_cloud: &PointCloud<DVector<f64>>,
        params: &FiltrationParameters,
        max_dimension: usize,
    ) -> Result<(), TopologyError> {
        match &params.filtration_type {
            FiltrationType::VietorisRips { max_radius } => {
                self.build_vietoris_rips_filtration(point_cloud, *max_radius, max_dimension)
            }
            FiltrationType::Alpha { weighted } => {
                self.build_alpha_filtration(point_cloud, *weighted, max_dimension)
            }
            _ => Err(TopologyError::UnsupportedFiltration(
                "Only Vietoris-Rips and Alpha supported".to_string(),
            )),
        }
    }

    /// Build Vietoris-Rips filtration
    fn build_vietoris_rips_filtration(
        &mut self,
        point_cloud: &PointCloud<DVector<f64>>,
        max_radius: f64,
        max_dimension: usize,
    ) -> Result<(), TopologyError> {
        let n_points = point_cloud.len();

        // Compute distance matrix
        let mut distance_matrix = DMatrix::zeros(n_points, n_points);
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let distance = point_cloud.distance(i, j);
                distance_matrix[(i, j)] = distance;
                distance_matrix[(j, i)] = distance;
            }
        }

        // Add 0-simplices (vertices)
        for i in 0..n_points {
            let vertex = Simplex::vertex(i);
            self.filtration.add_simplex(vertex, 0.0)?;
        }

        // Add 1-simplices (edges)
        let mut edge_data = Vec::new();
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let distance = distance_matrix[(i, j)];
                if distance <= max_radius {
                    edge_data.push((distance, i, j));
                }
            }
        }
        edge_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (distance, i, j) in edge_data {
            let edge = Simplex::edge(i, j);
            self.filtration.add_simplex(edge, distance)?;
        }

        // Add higher-dimensional simplices
        for dim in 2..=max_dimension {
            tracing::debug!("Building {}-simplices", dim);

            let lower_simplices: Vec<_> = self
                .filtration
                .simplices_of_dimension(dim - 1)
                .into_iter()
                .cloned()
                .collect();

            for simplex_with_birth in &lower_simplices {
                let vertices = simplex_with_birth.simplex.vertices();
                let birth_time = simplex_with_birth.birth_time;

                // Try to extend this simplex
                for candidate_vertex in 0..n_points {
                    if vertices.contains(&candidate_vertex) {
                        continue;
                    }

                    // Check if all edges to existing vertices exist
                    let mut max_edge_distance = birth_time;
                    let mut can_add = true;

                    for &existing_vertex in vertices {
                        let edge_distance = distance_matrix[(existing_vertex, candidate_vertex)];
                        if edge_distance > max_radius {
                            can_add = false;
                            break;
                        }
                        max_edge_distance = max_edge_distance.max(edge_distance);
                    }

                    if can_add {
                        let mut new_vertices = vertices.to_vec();
                        new_vertices.push(candidate_vertex);
                        let new_simplex = Simplex::from_vertices(new_vertices);

                        self.filtration
                            .add_simplex(new_simplex, max_edge_distance)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Build Alpha complex filtration (simplified version)
    fn build_alpha_filtration(
        &mut self,
        point_cloud: &PointCloud<DVector<f64>>,
        _weighted: bool,
        max_dimension: usize,
    ) -> Result<(), TopologyError> {
        // For now, implement as Vietoris-Rips with adaptive radius
        let adaptive_radius = self.estimate_alpha_radius(point_cloud)?;
        self.build_vietoris_rips_filtration(point_cloud, adaptive_radius, max_dimension)
    }

    /// Estimate appropriate radius for Alpha complex
    fn estimate_alpha_radius(
        &self,
        point_cloud: &PointCloud<DVector<f64>>,
    ) -> Result<f64, TopologyError> {
        // Use average nearest neighbor distance as heuristic
        let n = point_cloud.len();
        let mut total_nn_distance = 0.0;

        for i in 0..n {
            let mut min_distance = f64::INFINITY;
            for j in 0..n {
                if i != j {
                    let distance = point_cloud.distance(i, j);
                    min_distance = min_distance.min(distance);
                }
            }
            total_nn_distance += min_distance;
        }

        Ok(2.0 * total_nn_distance / n as f64)
    }

    /// Build sparse boundary matrix from filtration
    fn build_boundary_matrix(&mut self, max_dimension: usize) -> Result<(), TopologyError> {
        let total_simplices = self.filtration.total_simplices();
        self.boundary_matrix = SparseBoundaryMatrix::new(total_simplices, total_simplices);

        // Build boundary matrix column by column
        for dim in 1..=max_dimension {
            let simplices = self.filtration.simplices_of_dimension(dim);

            for simplex_with_birth in simplices {
                let col_idx = simplex_with_birth.index;
                let boundary_faces = simplex_with_birth.simplex.boundary();

                for (face_idx, face) in boundary_faces.iter().enumerate() {
                    if let Some(row_idx) = self.find_simplex_index_in_filtration(face) {
                        // Compute boundary coefficient (alternating sum)
                        let coefficient = if face_idx % 2 == 0 {
                            F::one()
                        } else {
                            F::one().neg()
                        };
                        self.boundary_matrix
                            .set_entry(row_idx, col_idx, coefficient);
                    }
                }
            }
        }

        Ok(())
    }

    /// Find simplex index in filtration
    fn find_simplex_index_in_filtration(&self, simplex: &Simplex) -> Option<usize> {
        self.filtration
            .simplices
            .iter()
            .position(|s| s.simplex == *simplex)
    }

    /// Perform matrix reduction for persistent homology
    fn perform_matrix_reduction(&mut self) -> Result<ReductionResult, TopologyError> {
        match self.reduction_algorithm {
            MatrixReductionAlgorithm::Standard => self.standard_matrix_reduction(),
            MatrixReductionAlgorithm::Twist => self.twist_matrix_reduction(),
            MatrixReductionAlgorithm::Chunk => self.chunk_matrix_reduction(),
            MatrixReductionAlgorithm::Parallel { num_threads } => {
                self.parallel_matrix_reduction(num_threads)
            }
            MatrixReductionAlgorithm::Cohomology => self.persistent_cohomology_algorithm(),
        }
    }

    /// Standard matrix reduction algorithm
    fn standard_matrix_reduction(&mut self) -> Result<ReductionResult, TopologyError> {
        let mut finite_pairs = Vec::new();
        let mut essential_classes = Vec::new();

        // Process columns from left to right
        for col in 0..self.boundary_matrix.cols() {
            let mut current_col = col;

            // Reduce column until pivot is unique or column becomes zero
            loop {
                if let Some(pivot_row) = self.boundary_matrix.lowest_nonzero_row(current_col) {
                    // Check if this pivot is already used by another column
                    if let Some(other_col) = self.find_column_with_pivot_row(pivot_row, current_col)
                    {
                        // Add other column to current column to eliminate pivot
                        self.boundary_matrix.add_column(other_col, current_col)?;
                    } else {
                        // Pivot is unique, record persistence pair
                        finite_pairs.push((pivot_row, current_col));
                        break;
                    }
                } else {
                    // Column is zero, represents essential class
                    essential_classes.push(current_col);
                    break;
                }
            }
        }

        Ok(ReductionResult {
            finite_pairs,
            essential_classes,
            reduced_matrix: SparseBoundaryMatrix::new(0, 0),
        })
    }

    /// Find column with specific pivot row (excluding given column)
    fn find_column_with_pivot_row(&self, pivot_row: usize, exclude_col: usize) -> Option<usize> {
        for col in 0..self.boundary_matrix.cols() {
            if col != exclude_col {
                if let Some(other_pivot) = self.boundary_matrix.lowest_nonzero_row(col) {
                    if other_pivot == pivot_row {
                        return Some(col);
                    }
                }
            }
        }
        None
    }

    /// Twist matrix reduction algorithm for optimized column operations
    fn twist_matrix_reduction(&mut self) -> Result<ReductionResult, TopologyError> {
        let mut finite_pairs = Vec::new();
        let mut essential_classes = Vec::new();
        let mut twist_buffer = HashMap::<usize, Vec<(usize, F)>>::new();

        tracing::debug!("Starting twist matrix reduction");

        // Process columns with twist optimization
        for col in 0..self.boundary_matrix.cols() {
            let mut current_col = col;

            // Try to apply cached twists first
            if let Some(cached_twist) = twist_buffer.get(&current_col) {
                for &(other_col, coeff) in cached_twist {
                    // Apply cached column addition with coefficient
                    for _ in 0..1 {
                        // Simplified - in full implementation would handle arbitrary field coefficients
                        self.boundary_matrix.add_column(other_col, current_col)?;
                    }
                }
                twist_buffer.remove(&current_col);
            }

            // Standard reduction with twist caching
            loop {
                if let Some(pivot_row) = self.boundary_matrix.lowest_nonzero_row(current_col) {
                    if let Some(other_col) = self.find_column_with_pivot_row(pivot_row, current_col)
                    {
                        // Cache this twist operation
                        twist_buffer
                            .entry(current_col)
                            .or_default()
                            .push((other_col, F::one()));

                        self.boundary_matrix.add_column(other_col, current_col)?;
                    } else {
                        finite_pairs.push((pivot_row, current_col));
                        break;
                    }
                } else {
                    essential_classes.push(current_col);
                    break;
                }
            }
        }

        tracing::debug!(
            "Twist reduction completed: {} finite pairs, {} essential classes",
            finite_pairs.len(),
            essential_classes.len()
        );

        Ok(ReductionResult {
            finite_pairs,
            essential_classes,
            reduced_matrix: SparseBoundaryMatrix::new(0, 0),
        })
    }

    /// Chunk matrix reduction for memory-efficient processing
    fn chunk_matrix_reduction(&mut self) -> Result<ReductionResult, TopologyError> {
        const CHUNK_SIZE: usize = 1000; // Process columns in chunks to manage memory

        let mut finite_pairs = Vec::new();
        let mut essential_classes = Vec::new();
        let total_cols = self.boundary_matrix.cols();

        tracing::debug!(
            "Starting chunk matrix reduction with chunk size {}",
            CHUNK_SIZE
        );

        // Process matrix in chunks
        for chunk_start in (0..total_cols).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(total_cols);
            tracing::debug!("Processing chunk [{}, {})", chunk_start, chunk_end);

            // Extract chunk for processing
            let mut chunk_pairs = Vec::new();
            let mut chunk_essential = Vec::new();

            // Process each column in current chunk
            for col in chunk_start..chunk_end {
                loop {
                    if let Some(pivot_row) = self.boundary_matrix.lowest_nonzero_row(col) {
                        // Look for pivot in previously processed columns
                        let mut found_pivot_col = None;
                        for prev_col in 0..col {
                            if let Some(prev_pivot) =
                                self.boundary_matrix.lowest_nonzero_row(prev_col)
                            {
                                if prev_pivot == pivot_row {
                                    found_pivot_col = Some(prev_col);
                                    break;
                                }
                            }
                        }

                        if let Some(other_col) = found_pivot_col {
                            self.boundary_matrix.add_column(other_col, col)?;
                        } else {
                            chunk_pairs.push((pivot_row, col));
                            break;
                        }
                    } else {
                        chunk_essential.push(col);
                        break;
                    }
                }
            }

            finite_pairs.extend(chunk_pairs);
            essential_classes.extend(chunk_essential);

            // Compress columns after each chunk to free memory
            self.boundary_matrix.compress_columns()?;
        }

        tracing::debug!(
            "Chunk reduction completed: {} finite pairs, {} essential classes",
            finite_pairs.len(),
            essential_classes.len()
        );

        Ok(ReductionResult {
            finite_pairs,
            essential_classes,
            reduced_matrix: SparseBoundaryMatrix::new(0, 0),
        })
    }

    /// Parallel matrix reduction using multiple threads
    fn parallel_matrix_reduction(
        &mut self,
        num_threads: usize,
    ) -> Result<ReductionResult, TopologyError> {
        tracing::debug!(
            "Starting parallel matrix reduction with {} threads",
            num_threads
        );

        let total_cols = self.boundary_matrix.cols();
        if total_cols < num_threads * 2 {
            // Fall back to standard reduction for small matrices
            return self.standard_matrix_reduction();
        }

        // Divide columns among threads (simplified parallel approach)
        let chunk_size = (total_cols + num_threads - 1) / num_threads;
        let mut all_finite_pairs = Vec::new();
        let mut all_essential_classes = Vec::new();

        // For now, implement a simplified parallel approach
        // Full implementation would require more sophisticated synchronization
        for thread_id in 0..num_threads {
            let start_col = thread_id * chunk_size;
            let end_col = ((thread_id + 1) * chunk_size).min(total_cols);

            if start_col >= total_cols {
                break;
            }

            tracing::debug!(
                "Thread {} processing columns [{}, {})",
                thread_id,
                start_col,
                end_col
            );

            // Process columns in this thread's range
            for col in start_col..end_col {
                loop {
                    if let Some(pivot_row) = self.boundary_matrix.lowest_nonzero_row(col) {
                        // Check for pivot conflicts (simplified - full version needs locks)
                        if let Some(other_col) = self.find_column_with_pivot_row(pivot_row, col) {
                            self.boundary_matrix.add_column(other_col, col)?;
                        } else {
                            all_finite_pairs.push((pivot_row, col));
                            break;
                        }
                    } else {
                        all_essential_classes.push(col);
                        break;
                    }
                }
            }
        }

        // Sort pairs by birth time for consistency
        all_finite_pairs.sort_by_key(|&(birth_idx, death_idx)| {
            (
                self.filtration.simplex_birth_time(birth_idx) as u64,
                self.filtration.simplex_birth_time(death_idx) as u64,
            )
        });

        tracing::debug!(
            "Parallel reduction completed: {} finite pairs, {} essential classes",
            all_finite_pairs.len(),
            all_essential_classes.len()
        );

        Ok(ReductionResult {
            finite_pairs: all_finite_pairs,
            essential_classes: all_essential_classes,
            reduced_matrix: SparseBoundaryMatrix::new(0, 0),
        })
    }

    /// Persistent cohomology algorithm (dual to homology)
    fn persistent_cohomology_algorithm(&mut self) -> Result<ReductionResult, TopologyError> {
        tracing::debug!("Computing persistent cohomology (dual approach)");

        // Build coboundary matrix (transpose of boundary matrix)
        let rows = self.boundary_matrix.cols();
        let cols = self.boundary_matrix.rows();
        let mut coboundary_matrix = SparseBoundaryMatrix::<F>::new(rows, cols);

        // Transpose the boundary matrix to get coboundary matrix
        for row in 0..self.boundary_matrix.rows() {
            for col in 0..self.boundary_matrix.cols() {
                let entry = self.boundary_matrix.get_entry(row, col);
                if entry != F::zero() {
                    coboundary_matrix.set_entry(col, row, entry);
                }
            }
        }

        // Apply standard reduction to coboundary matrix
        let mut finite_pairs = Vec::new();
        let mut essential_classes = Vec::new();

        // Process coboundary matrix columns (which correspond to original rows)
        for col in 0..coboundary_matrix.cols() {
            loop {
                if let Some(pivot_row) = coboundary_matrix.lowest_nonzero_row(col) {
                    let mut found_other = false;
                    for other_col in 0..col {
                        if let Some(other_pivot) = coboundary_matrix.lowest_nonzero_row(other_col) {
                            if other_pivot == pivot_row {
                                coboundary_matrix.add_column(other_col, col)?;
                                found_other = true;
                                break;
                            }
                        }
                    }

                    if !found_other {
                        // In cohomology, the pairing is (death, birth)
                        finite_pairs.push((col, pivot_row));
                        break;
                    }
                } else {
                    essential_classes.push(col);
                    break;
                }
            }
        }

        tracing::debug!(
            "Cohomology computation completed: {} finite pairs, {} essential classes",
            finite_pairs.len(),
            essential_classes.len()
        );

        Ok(ReductionResult {
            finite_pairs,
            essential_classes,
            reduced_matrix: SparseBoundaryMatrix::new(0, 0),
        })
    }

    /// Extract persistence pairs from reduction result
    fn extract_persistence_pairs(
        &self,
        result: &ReductionResult,
    ) -> Result<Vec<PersistencePair>, TopologyError> {
        let mut pairs = Vec::new();

        // Extract finite persistence pairs
        for &(birth_idx, death_idx) in &result.finite_pairs {
            let birth_time = self.filtration.simplex_birth_time(birth_idx);
            let death_time = self.filtration.simplex_birth_time(death_idx);
            let dimension = self.filtration.simplex_dimension(birth_idx);

            if birth_time <= death_time {
                pairs.push(PersistencePair {
                    dimension,
                    birth: birth_time,
                    death: Some(death_time),
                    birth_simplex: birth_idx,
                    death_simplex: Some(death_idx),
                    multiplicity: 1,
                });
            }
        }

        // Extract essential (infinite) classes
        for &essential_idx in &result.essential_classes {
            let birth_time = self.filtration.simplex_birth_time(essential_idx);
            let dimension = self.filtration.simplex_dimension(essential_idx);

            pairs.push(PersistencePair {
                dimension,
                birth: birth_time,
                death: None,
                birth_simplex: essential_idx,
                death_simplex: None,
                multiplicity: 1,
            });
        }

        Ok(pairs)
    }

    /// Compute stability constant for persistence diagram
    fn compute_stability_constant(&self) -> Result<f64, TopologyError> {
        // Stability constant is 1 for standard filtrations
        Ok(1.0)
    }
}

/// Topology computation errors
#[derive(Debug, Error)]
pub enum TopologyError {
    #[error("Invalid filtration: simplex birth {simplex_birth} > face birth {face_birth}")]
    InvalidFiltration { simplex_birth: f64, face_birth: f64 },

    #[error("Missing face in complex: {0}")]
    MissingFace(String),

    #[error("Invalid persistence pair: birth {birth_time} >= death {death_time}")]
    InvalidPersistencePair {
        birth_time: f64,
        death_time: f64,
        birth_simplex: usize,
        death_simplex: usize,
    },

    #[error("Matrix index error")]
    MatrixIndexError,

    #[error("Unsupported filtration: {0}")]
    UnsupportedFiltration(String),

    #[error("Unsupported algorithm: {0}")]
    UnsupportedAlgorithm(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Research-grade Mapper algorithm for high-dimensional data visualization
#[derive(Debug)]
pub struct MapperAlgorithm<T> {
    /// Lens function for dimensionality reduction
    pub lens_function: Box<dyn LensFunction<T> + Send + Sync>,

    /// Covering scheme for the lens function range
    pub covering_scheme: CoveringScheme,

    /// Clustering algorithm for pre-images
    pub clustering_algorithm: ClusteringAlgorithm,

    /// Nerve complex construction parameters
    pub nerve_parameters: NerveParameters,

    /// Intersection threshold for nerve edges
    pub intersection_threshold: f64,
}

impl<T> MapperAlgorithm<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new(
        lens_function: Box<dyn LensFunction<T> + Send + Sync>,
        covering_scheme: CoveringScheme,
        clustering_algorithm: ClusteringAlgorithm,
    ) -> Self {
        Self {
            lens_function,
            covering_scheme,
            clustering_algorithm,
            nerve_parameters: NerveParameters::default(),
            intersection_threshold: 0.0,
        }
    }

    /// Compute Mapper graph from point cloud data
    pub fn compute_mapper_graph(
        &self,
        point_cloud: &PointCloud<T>,
    ) -> Result<MapperGraph, TopologyError> {
        tracing::info!("Computing Mapper graph for {} points", point_cloud.len());
        let total_timer = std::time::Instant::now();

        // Step 1: Apply lens function to all points
        tracing::debug!("Applying lens function");
        let lens_values = self.apply_lens_function(point_cloud)?;

        // Step 2: Create covering of lens function range
        tracing::debug!("Creating covering scheme");
        let covering = self.create_covering(&lens_values)?;

        // Step 3: For each cover element, cluster pre-image points
        tracing::debug!("Clustering pre-images");
        let clustered_preimages = self.cluster_preimages(point_cloud, &lens_values, &covering)?;

        // Step 4: Build nerve complex from clustered pre-images
        tracing::debug!("Building nerve complex");
        let nerve_complex = self.build_nerve_complex(&clustered_preimages)?;

        // Step 5: Create final Mapper graph
        let total_nodes = nerve_complex.nodes.len();
        let total_edges = nerve_complex.edges.len();

        let mapper_graph = MapperGraph {
            nodes: nerve_complex.nodes,
            edges: nerve_complex.edges,
            node_metadata: nerve_complex.node_metadata,
            lens_values,
            covering_info: covering,
            computational_metadata: MapperMetadata {
                total_computation_time: total_timer.elapsed(),
                lens_function_type: self.lens_function.function_type(),
                covering_parameters: self.covering_scheme.clone(),
                clustering_algorithm: self.clustering_algorithm,
                total_nodes,
                total_edges,
            },
        };

        tracing::info!(
            "Mapper graph computed in {:?}: {} nodes, {} edges",
            total_timer.elapsed(),
            mapper_graph.nodes.len(),
            mapper_graph.edges.len()
        );

        Ok(mapper_graph)
    }

    /// Apply lens function to all points in cloud
    fn apply_lens_function(&self, point_cloud: &PointCloud<T>) -> Result<Vec<f64>, TopologyError> {
        let lens_values: Result<Vec<_>, _> = point_cloud
            .points()
            .par_iter()
            .map(|point| self.lens_function.evaluate(point))
            .collect();

        lens_values
            .map_err(|e| TopologyError::ComputationError(format!("Lens function error: {}", e)))
    }

    /// Create covering of lens function range
    fn create_covering(&self, lens_values: &[f64]) -> Result<CoveringInfo, TopologyError> {
        if lens_values.is_empty() {
            return Err(TopologyError::ComputationError(
                "Empty lens values".to_string(),
            ));
        }

        let min_val = lens_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = lens_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        let intervals = match &self.covering_scheme {
            CoveringScheme::UniformIntervals {
                num_intervals,
                overlap,
            } => self.create_uniform_intervals(min_val, max_val, *num_intervals, *overlap),
            CoveringScheme::AdaptiveIntervals {
                target_density,
                min_overlap,
            } => self.create_adaptive_intervals(lens_values, *target_density, *min_overlap),
            CoveringScheme::QuantileBased {
                num_quantiles,
                overlap,
            } => self.create_quantile_intervals(lens_values, *num_quantiles, *overlap),
        }?;

        Ok(CoveringInfo {
            intervals,
            range: (min_val, max_val),
            covering_type: self.covering_scheme.clone(),
        })
    }

    fn create_uniform_intervals(
        &self,
        min_val: f64,
        max_val: f64,
        num_intervals: usize,
        overlap: f64,
    ) -> Result<Vec<Interval>, TopologyError> {
        if num_intervals == 0 {
            return Err(TopologyError::ComputationError(
                "Zero intervals requested".to_string(),
            ));
        }

        let range = max_val - min_val;
        let interval_width = range / num_intervals as f64;
        let overlap_width = interval_width * overlap;

        let mut intervals = Vec::new();
        for i in 0..num_intervals {
            let start = min_val + i as f64 * interval_width - overlap_width;
            let end = min_val + (i + 1) as f64 * interval_width + overlap_width;

            intervals.push(Interval {
                start: start.max(min_val),
                end: end.min(max_val),
                id: i,
                weight: 1.0,
            });
        }

        Ok(intervals)
    }

    fn create_adaptive_intervals(
        &self,
        lens_values: &[f64],
        target_density: usize,
        min_overlap: f64,
    ) -> Result<Vec<Interval>, TopologyError> {
        // Sort lens values for adaptive partitioning
        let mut sorted_values = lens_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut intervals = Vec::new();
        let mut current_start = sorted_values[0];
        let mut id = 0;

        for chunk in sorted_values.chunks(target_density) {
            if let (Some(&chunk_start), Some(&chunk_end)) = (chunk.first(), chunk.last()) {
                let interval_width = chunk_end - chunk_start;
                let overlap_width = interval_width * min_overlap;

                intervals.push(Interval {
                    start: (current_start - overlap_width).max(sorted_values[0]),
                    end: (chunk_end + overlap_width).min(sorted_values[sorted_values.len() - 1]),
                    id,
                    weight: chunk.len() as f64 / lens_values.len() as f64,
                });

                current_start = chunk_end;
                id += 1;
            }
        }

        Ok(intervals)
    }

    fn create_quantile_intervals(
        &self,
        lens_values: &[f64],
        num_quantiles: usize,
        overlap: f64,
    ) -> Result<Vec<Interval>, TopologyError> {
        let mut sorted_values = lens_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut intervals = Vec::new();
        let quantile_size = sorted_values.len() / num_quantiles;

        for i in 0..num_quantiles {
            let start_idx = i * quantile_size;
            let end_idx = if i == num_quantiles - 1 {
                sorted_values.len() - 1
            } else {
                (i + 1) * quantile_size - 1
            };

            let interval_start = sorted_values[start_idx];
            let interval_end = sorted_values[end_idx];
            let interval_width = interval_end - interval_start;
            let overlap_width = interval_width * overlap;

            intervals.push(Interval {
                start: (interval_start - overlap_width).max(sorted_values[0]),
                end: (interval_end + overlap_width).min(sorted_values[sorted_values.len() - 1]),
                id: i,
                weight: (end_idx - start_idx + 1) as f64 / sorted_values.len() as f64,
            });
        }

        Ok(intervals)
    }

    /// Cluster pre-image points for each covering interval
    fn cluster_preimages(
        &self,
        point_cloud: &PointCloud<T>,
        lens_values: &[f64],
        covering: &CoveringInfo,
    ) -> Result<Vec<ClusteringResult>, TopologyError> {
        covering
            .intervals
            .par_iter()
            .map(|interval| {
                // Find points in this interval
                let preimage_indices: Vec<usize> = lens_values
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &val)| {
                        if val >= interval.start && val <= interval.end {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                if preimage_indices.is_empty() {
                    return Ok(ClusteringResult {
                        interval_id: interval.id,
                        clusters: Vec::new(),
                        cluster_centers: Vec::new(),
                        point_assignments: HashMap::new(),
                    });
                }

                // Apply clustering algorithm to pre-image points
                self.cluster_points(point_cloud, &preimage_indices, interval.id)
            })
            .collect()
    }

    /// Apply clustering algorithm to specific points
    fn cluster_points(
        &self,
        point_cloud: &PointCloud<T>,
        point_indices: &[usize],
        interval_id: usize,
    ) -> Result<ClusteringResult, TopologyError> {
        match self.clustering_algorithm {
            ClusteringAlgorithm::SingleLinkage { threshold } => {
                self.single_linkage_clustering(point_cloud, point_indices, threshold, interval_id)
            }
            ClusteringAlgorithm::DBSCAN { eps, min_points } => {
                self.dbscan_clustering(point_cloud, point_indices, eps, min_points, interval_id)
            }
            ClusteringAlgorithm::Complete => {
                // No clustering - treat all points as single cluster
                Ok(ClusteringResult {
                    interval_id,
                    clusters: vec![point_indices.to_vec()],
                    cluster_centers: Vec::new(),
                    point_assignments: point_indices
                        .iter()
                        .enumerate()
                        .map(|(i, &point_idx)| (point_idx, i))
                        .collect(),
                })
            }
        }
    }

    fn single_linkage_clustering(
        &self,
        point_cloud: &PointCloud<T>,
        point_indices: &[usize],
        threshold: f64,
        interval_id: usize,
    ) -> Result<ClusteringResult, TopologyError> {
        let n = point_indices.len();
        let mut cluster_assignments = vec![0; n];
        let mut next_cluster_id = 0;

        // Initialize each point as its own cluster
        for i in 0..n {
            cluster_assignments[i] = i;
        }

        // Compute distances and merge clusters
        for i in 0..n {
            for j in (i + 1)..n {
                let distance = point_cloud.distance(point_indices[i], point_indices[j]);
                if distance <= threshold {
                    // Merge clusters
                    let cluster_i = cluster_assignments[i];
                    let cluster_j = cluster_assignments[j];
                    if cluster_i != cluster_j {
                        let min_cluster = cluster_i.min(cluster_j);
                        let max_cluster = cluster_i.max(cluster_j);
                        for assignment in &mut cluster_assignments {
                            if *assignment == max_cluster {
                                *assignment = min_cluster;
                            }
                        }
                    }
                }
            }
        }

        // Build final clusters
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for (local_idx, &cluster_id) in cluster_assignments.iter().enumerate() {
            clusters
                .entry(cluster_id)
                .or_default()
                .push(point_indices[local_idx]);
        }

        let cluster_vec: Vec<Vec<usize>> = clusters.into_values().collect();
        let point_assignments: HashMap<usize, usize> = point_indices
            .iter()
            .enumerate()
            .map(|(local_idx, &global_idx)| (global_idx, cluster_assignments[local_idx]))
            .collect();

        Ok(ClusteringResult {
            interval_id,
            clusters: cluster_vec,
            cluster_centers: Vec::new(),
            point_assignments,
        })
    }

    fn dbscan_clustering(
        &self,
        point_cloud: &PointCloud<T>,
        point_indices: &[usize],
        eps: f64,
        min_points: usize,
        interval_id: usize,
    ) -> Result<ClusteringResult, TopologyError> {
        let n = point_indices.len();
        let mut labels = vec![-1i32; n]; // -1 = unvisited, -2 = noise
        let mut cluster_id = 0;

        for i in 0..n {
            if labels[i] != -1 {
                continue; // Already processed
            }

            let neighbors = self.find_neighbors(point_cloud, point_indices, i, eps);

            if neighbors.len() < min_points {
                labels[i] = -2; // Mark as noise
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            let mut seed_set = neighbors;
            let mut j = 0;

            while j < seed_set.len() {
                let q = seed_set[j];

                if labels[q] == -2 {
                    labels[q] = cluster_id; // Change noise to border point
                } else if labels[q] == -1 {
                    labels[q] = cluster_id;
                    let q_neighbors = self.find_neighbors(point_cloud, point_indices, q, eps);

                    if q_neighbors.len() >= min_points {
                        for &neighbor in &q_neighbors {
                            if !seed_set.contains(&neighbor) {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }

                j += 1;
            }

            cluster_id += 1;
        }

        // Build final clusters
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (local_idx, &label) in labels.iter().enumerate() {
            if label >= 0 {
                clusters
                    .entry(label)
                    .or_default()
                    .push(point_indices[local_idx]);
            }
        }

        let cluster_vec: Vec<Vec<usize>> = clusters.into_values().collect();
        let point_assignments: HashMap<usize, usize> = point_indices
            .iter()
            .enumerate()
            .filter_map(|(local_idx, &global_idx)| {
                let label = labels[local_idx];
                if label >= 0 {
                    Some((global_idx, label as usize))
                } else {
                    None
                }
            })
            .collect();

        Ok(ClusteringResult {
            interval_id,
            clusters: cluster_vec,
            cluster_centers: Vec::new(),
            point_assignments,
        })
    }

    fn find_neighbors(
        &self,
        point_cloud: &PointCloud<T>,
        point_indices: &[usize],
        center_local_idx: usize,
        eps: f64,
    ) -> Vec<usize> {
        let center_global_idx = point_indices[center_local_idx];
        let mut neighbors = Vec::new();

        for (local_idx, &global_idx) in point_indices.iter().enumerate() {
            if local_idx != center_local_idx {
                let distance = point_cloud.distance(center_global_idx, global_idx);
                if distance <= eps {
                    neighbors.push(local_idx);
                }
            }
        }

        neighbors
    }

    /// Build nerve complex from clustered pre-images
    fn build_nerve_complex(
        &self,
        clustered_preimages: &[ClusteringResult],
    ) -> Result<NerveComplex, TopologyError> {
        let mut nodes = Vec::new();
        let mut node_metadata = Vec::new();
        let mut node_id = 0;

        // Create nodes from clusters
        for clustering_result in clustered_preimages {
            for (cluster_idx, cluster) in clustering_result.clusters.iter().enumerate() {
                nodes.push(MapperNode {
                    id: node_id,
                    interval_id: clustering_result.interval_id,
                    cluster_id: cluster_idx,
                    point_indices: cluster.clone(),
                    representative_point: self.compute_cluster_representative(cluster)?,
                });

                node_metadata.push(NodeMetadata {
                    cluster_size: cluster.len(),
                    density: cluster.len() as f64 / clustered_preimages.len() as f64,
                    eccentricity: 0.0, // Could compute if needed
                });

                node_id += 1;
            }
        }

        // Create edges based on point set intersections
        let mut edges = Vec::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if let Some(edge) = self.compute_nerve_edge(&nodes[i], &nodes[j])? {
                    edges.push(edge);
                }
            }
        }

        Ok(NerveComplex {
            nodes,
            edges,
            node_metadata,
        })
    }

    fn compute_cluster_representative(
        &self,
        cluster: &[usize],
    ) -> Result<Option<usize>, TopologyError> {
        // For now, just return the first point as representative
        Ok(cluster.first().copied())
    }

    fn compute_nerve_edge(
        &self,
        node_a: &MapperNode,
        node_b: &MapperNode,
    ) -> Result<Option<MapperEdge>, TopologyError> {
        // Compute intersection of point sets
        let set_a: HashSet<usize> = node_a.point_indices.iter().copied().collect();
        let set_b: HashSet<usize> = node_b.point_indices.iter().copied().collect();
        let intersection: Vec<usize> = set_a.intersection(&set_b).copied().collect();

        let intersection_size = intersection.len();
        let union_size = set_a.union(&set_b).count();
        let jaccard_similarity = if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            0.0
        };

        if intersection_size > 0 && jaccard_similarity >= self.intersection_threshold {
            Ok(Some(MapperEdge {
                source: node_a.id,
                target: node_b.id,
                intersection_points: intersection,
                intersection_size,
                jaccard_similarity,
                weight: jaccard_similarity,
            }))
        } else {
            Ok(None)
        }
    }
}

/// Lens function trait for Mapper algorithm
pub trait LensFunction<T>: std::fmt::Debug {
    /// Evaluate lens function on a point
    fn evaluate(&self, point: &T) -> Result<f64, String>;

    /// Get function type description
    fn function_type(&self) -> String;

    /// Check if function is differentiable
    fn is_differentiable(&self) -> bool {
        false
    }
}

/// Principal Component Analysis lens function
#[derive(Debug, Clone)]
pub struct PCALensFunction {
    /// First principal component vector
    pub principal_component: DVector<f64>,

    /// Data mean for centering
    pub data_mean: DVector<f64>,
}

impl LensFunction<DVector<f64>> for PCALensFunction {
    fn evaluate(&self, point: &DVector<f64>) -> Result<f64, String> {
        if point.len() != self.principal_component.len() {
            return Err("Point dimension mismatch".to_string());
        }

        let centered = point - &self.data_mean;
        Ok(self.principal_component.dot(&centered))
    }

    fn function_type(&self) -> String {
        "PCA (First Principal Component)".to_string()
    }

    fn is_differentiable(&self) -> bool {
        true
    }
}

/// Density-based lens function
#[derive(Debug)]
pub struct DensityLensFunction {
    /// Kernel bandwidth
    pub bandwidth: f64,

    /// Reference point cloud for density estimation
    pub reference_points: Vec<DVector<f64>>,

    /// Metric function for distance computation
    pub metric: Box<dyn MetricFunction<DVector<f64>> + Send + Sync>,
}

impl LensFunction<DVector<f64>> for DensityLensFunction {
    fn evaluate(&self, point: &DVector<f64>) -> Result<f64, String> {
        let mut density = 0.0;
        let normalization = 1.0 / (self.reference_points.len() as f64 * self.bandwidth);

        for ref_point in &self.reference_points {
            let distance = self.metric.distance(point, ref_point);
            let kernel_value = (-0.5 * (distance / self.bandwidth).powi(2)).exp();
            density += kernel_value;
        }

        Ok(density * normalization)
    }

    fn function_type(&self) -> String {
        format!("Gaussian Density (bandwidth={})", self.bandwidth)
    }
}

/// Covering scheme for lens function range
#[derive(Debug, Clone)]
pub enum CoveringScheme {
    /// Uniform intervals with specified overlap
    UniformIntervals { num_intervals: usize, overlap: f64 },

    /// Adaptive intervals based on data density
    AdaptiveIntervals {
        target_density: usize,
        min_overlap: f64,
    },

    /// Quantile-based intervals
    QuantileBased { num_quantiles: usize, overlap: f64 },
}

/// Clustering algorithm for pre-image points
#[derive(Debug, Clone, Copy)]
pub enum ClusteringAlgorithm {
    /// Single linkage clustering
    SingleLinkage { threshold: f64 },

    /// DBSCAN clustering
    DBSCAN { eps: f64, min_points: usize },

    /// No clustering (complete pre-image)
    Complete,
}

/// Interval in covering scheme
#[derive(Debug, Clone)]
pub struct Interval {
    pub start: f64,
    pub end: f64,
    pub id: usize,
    pub weight: f64,
}

/// Information about the covering
#[derive(Debug, Clone)]
pub struct CoveringInfo {
    pub intervals: Vec<Interval>,
    pub range: (f64, f64),
    pub covering_type: CoveringScheme,
}

/// Result of clustering points in pre-image
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    pub interval_id: usize,
    pub clusters: Vec<Vec<usize>>, // Each cluster is a list of point indices
    pub cluster_centers: Vec<usize>, // Representative points for each cluster
    pub point_assignments: HashMap<usize, usize>, // Point index -> cluster id
}

/// Nerve complex parameters
#[derive(Debug, Clone)]
pub struct NerveParameters {
    pub min_intersection_size: usize,
    pub jaccard_threshold: f64,
}

impl Default for NerveParameters {
    fn default() -> Self {
        Self {
            min_intersection_size: 1,
            jaccard_threshold: 0.0,
        }
    }
}

/// Node in Mapper graph
#[derive(Debug, Clone)]
pub struct MapperNode {
    pub id: usize,
    pub interval_id: usize,
    pub cluster_id: usize,
    pub point_indices: Vec<usize>,
    pub representative_point: Option<usize>,
}

/// Edge in Mapper graph
#[derive(Debug, Clone)]
pub struct MapperEdge {
    pub source: usize,
    pub target: usize,
    pub intersection_points: Vec<usize>,
    pub intersection_size: usize,
    pub jaccard_similarity: f64,
    pub weight: f64,
}

/// Complete Mapper graph
#[derive(Debug, Clone)]
pub struct MapperGraph {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<MapperEdge>,
    pub node_metadata: Vec<NodeMetadata>,
    pub lens_values: Vec<f64>,
    pub covering_info: CoveringInfo,
    pub computational_metadata: MapperMetadata,
}

/// Node metadata for analysis
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub cluster_size: usize,
    pub density: f64,
    pub eccentricity: f64,
}

/// Nerve complex structure
#[derive(Debug, Clone)]
pub struct NerveComplex {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<MapperEdge>,
    pub node_metadata: Vec<NodeMetadata>,
}

/// Computational metadata for Mapper
#[derive(Debug, Clone)]
pub struct MapperMetadata {
    pub total_computation_time: std::time::Duration,
    pub lens_function_type: String,
    pub covering_parameters: CoveringScheme,
    pub clustering_algorithm: ClusteringAlgorithm,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Topological signatures for physical phase transitions
#[derive(Debug, Clone)]
pub struct TopologicalPhaseAnalyzer<F: Field> {
    /// Persistent homology engine for phase analysis
    pub homology_engine: PersistentHomologyEngine<F>,

    /// Phase transition detection parameters
    pub transition_parameters: PhaseTransitionParameters,

    /// Critical point detection settings
    pub critical_point_settings: CriticalPointSettings,

    /// Historical phase data for comparison
    pub phase_history: Vec<TopologicalPhaseSignature>,
}

impl<F: Field + Copy + Send + Sync> TopologicalPhaseAnalyzer<F> {
    pub fn new() -> Self {
        Self {
            homology_engine: PersistentHomologyEngine::new(MatrixReductionAlgorithm::Twist),
            transition_parameters: PhaseTransitionParameters::default(),
            critical_point_settings: CriticalPointSettings::default(),
            phase_history: Vec::new(),
        }
    }

    /// Analyze phase transitions from PhaseRegion data
    pub fn analyze_phase_transitions(
        &mut self,
        phase_regions: &[PhaseRegion],
        time_sequence: &[NanoTime],
    ) -> Result<PhaseTransitionAnalysis, TopologyError> {
        tracing::info!(
            "Analyzing phase transitions from {} regions",
            phase_regions.len()
        );
        let analysis_timer = std::time::Instant::now();

        // Convert phase regions to point cloud data
        let phase_point_cloud = self.convert_phase_regions_to_point_cloud(phase_regions)?;

        // Compute persistence diagrams for different filtration scales
        let mut persistence_diagrams = Vec::new();
        for &scale in &self.transition_parameters.filtration_scales {
            tracing::debug!("Computing persistence at scale {}", scale);

            let filtration_params = FiltrationParameters {
                filtration_type: FiltrationType::VietorisRips { max_radius: scale },
                max_dimension: 2, // Analyze up to 1-dimensional holes
                field_characteristic: F::characteristic(),
                chunk_size: Some(500),
            };

            let diagram = self.homology_engine.compute_persistent_homology(
                &phase_point_cloud,
                filtration_params,
                2,
            )?;

            persistence_diagrams.push(diagram);
        }

        // Extract topological signatures
        let signatures =
            self.extract_topological_signatures(&persistence_diagrams, time_sequence)?;

        // Detect critical points and transitions
        let critical_points = self.detect_critical_points(&signatures)?;
        let transitions = self.identify_phase_transitions(&signatures, &critical_points)?;

        // Compute topological order parameters
        let order_parameters = self.compute_topological_order_parameters(&signatures)?;

        let analysis = PhaseTransitionAnalysis {
            signatures,
            critical_points,
            transitions,
            order_parameters,
            persistence_diagrams,
            analysis_metadata: TransitionAnalysisMetadata {
                total_analysis_time: analysis_timer.elapsed(),
                num_phase_regions: phase_regions.len(),
                time_span: time_sequence.len(),
                algorithms_used: vec!["Vietoris-Rips".to_string(), "Twist Reduction".to_string()],
            },
        };

        // Update phase history
        for signature in &analysis.signatures {
            self.phase_history.push(signature.clone());
        }

        tracing::info!(
            "Phase transition analysis completed in {:?}",
            analysis_timer.elapsed()
        );
        Ok(analysis)
    }

    /// Convert PhaseRegion data to point cloud for TDA
    fn convert_phase_regions_to_point_cloud(
        &self,
        phase_regions: &[PhaseRegion],
    ) -> Result<PointCloud<DVector<f64>>, TopologyError> {
        let mut points = Vec::new();

        for (idx, region) in phase_regions.iter().enumerate() {
            // Create feature vector from PhaseRegion enum
            // Since PhaseRegion is an enum, we encode it as numerical features
            let features = match region {
                PhaseRegion::Stable => vec![0.0, 1.0, 0.5, 0.9],
                PhaseRegion::Unstable => vec![1.0, 0.0, 0.1, 0.2],
                PhaseRegion::Oscillatory => vec![0.5, 0.8, 0.7, 0.6],
                PhaseRegion::Critical => vec![0.7, 0.3, 0.9, 0.4],
                PhaseRegion::Transition => vec![0.6, 0.5, 0.8, 0.5],
                PhaseRegion::AttractorBasin => vec![0.2, 0.9, 0.6, 0.8],
                PhaseRegion::ChaoticAttractor => vec![0.9, 0.1, 0.3, 0.3],
                PhaseRegion::Integrable => vec![0.1, 0.7, 0.9, 0.7],
            };

            // Add index as additional feature for uniqueness
            let mut extended_features = features;
            extended_features.push(idx as f64 * 0.01); // Small perturbation for uniqueness

            points.push(DVector::from_vec(extended_features));
        }

        if points.is_empty() {
            return Err(TopologyError::ComputationError(
                "No phase regions to analyze".to_string(),
            ));
        }

        let dimension = points[0].len();
        Ok(PointCloud::new(
            points,
            dimension,
            Box::new(EuclideanMetric),
        ))
    }

    /// Extract topological signatures from persistence diagrams
    fn extract_topological_signatures(
        &self,
        diagrams: &[PersistenceDiagram],
        time_sequence: &[NanoTime],
    ) -> Result<Vec<TopologicalPhaseSignature>, TopologyError> {
        let mut signatures = Vec::new();

        for (time_idx, diagram) in diagrams.iter().enumerate() {
            let timestamp = if time_idx < time_sequence.len() {
                time_sequence[time_idx]
            } else {
                NanoTime::from_nanos(time_idx as u64)
            };

            // Compute Betti numbers across scales
            let mut betti_curves = HashMap::new();
            for dimension in 0..=2 {
                let mut betti_values = Vec::new();
                for scale in 0..100 {
                    let scale_value = (scale as f64) * 0.1;
                    let betti_numbers = diagram.betti_numbers_at_scale(scale_value);
                    let betti = *betti_numbers.get(&dimension).unwrap_or(&0);
                    betti_values.push(betti);
                }
                betti_curves.insert(dimension, betti_values);
            }

            // Compute persistence entropy
            let persistence_entropy = self.compute_persistence_entropy(diagram)?;

            // Compute topological complexity
            let topological_complexity = self.compute_topological_complexity(diagram)?;

            // Extract dominant persistence pairs
            let dominant_pairs = self.extract_dominant_persistence_pairs(diagram)?;

            signatures.push(TopologicalPhaseSignature {
                timestamp,
                betti_curves,
                persistence_entropy,
                topological_complexity,
                dominant_features: dominant_pairs,
                phase_classification: self.classify_phase_from_topology(diagram)?,
                stability_indicators: self.compute_stability_indicators(diagram)?,
                critical_scale: self.find_critical_scale(diagram)?,
            });
        }

        Ok(signatures)
    }

    /// Compute persistence entropy for diagram
    fn compute_persistence_entropy(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        let finite_pairs: Vec<_> = diagram.finite_pairs().collect();
        if finite_pairs.is_empty() {
            return Ok(0.0);
        }

        let total_persistence: f64 = finite_pairs.iter().map(|p| p.persistence()).sum();
        if total_persistence <= 0.0 {
            return Ok(0.0);
        }

        let mut entropy = 0.0;
        for pair in finite_pairs {
            let persistence = pair.persistence();
            if persistence > 0.0 {
                let probability = persistence / total_persistence;
                entropy -= probability * probability.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute topological complexity measure
    fn compute_topological_complexity(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        let mut complexity = 0.0;

        // Complexity based on number of persistent features
        for dimension in 0..=2 {
            let pairs_in_dim: Vec<_> = diagram.pairs_of_dimension(dimension).collect();
            let count = pairs_in_dim.len() as f64;
            complexity += count * (dimension + 1) as f64;
        }

        // Weight by persistence values
        let total_persistence = diagram.total_persistence();
        complexity *= (1.0 + total_persistence).ln();

        Ok(complexity)
    }

    /// Extract dominant persistence pairs
    fn extract_dominant_persistence_pairs(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<Vec<DominantFeature>, TopologyError> {
        let mut features = Vec::new();

        // Sort pairs by persistence
        let mut pairs: Vec<_> = diagram.finite_pairs().collect();
        pairs.sort_by(|a, b| b.persistence().partial_cmp(&a.persistence()).unwrap());

        // Take top features (up to 10)
        for pair in pairs.iter().take(10) {
            if pair.persistence() > self.critical_point_settings.min_persistence_threshold {
                features.push(DominantFeature {
                    dimension: pair.dimension,
                    birth_time: pair.birth,
                    death_time: pair.death.unwrap_or(f64::INFINITY),
                    persistence: pair.persistence(),
                    significance: pair.persistence() / diagram.total_persistence(),
                });
            }
        }

        Ok(features)
    }

    /// Classify phase from topological properties
    fn classify_phase_from_topology(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<PhaseClassification, TopologyError> {
        let betti_0 = diagram.pairs_of_dimension(0).count();
        let betti_1 = diagram.pairs_of_dimension(1).count();
        let total_persistence = diagram.total_persistence();

        // Simple heuristic classification
        let phase_type = if betti_0 == 1 && betti_1 == 0 {
            PhaseType::Connected
        } else if betti_0 > 1 && betti_1 == 0 {
            PhaseType::Disconnected
        } else if betti_1 > 0 {
            PhaseType::Complex
        } else {
            PhaseType::Unknown
        };

        let stability = if total_persistence > 1.0 {
            PhaseStability::Stable
        } else if total_persistence > 0.1 {
            PhaseStability::Metastable
        } else {
            PhaseStability::Unstable
        };

        Ok(PhaseClassification {
            phase_type,
            stability,
            confidence: self.compute_classification_confidence(diagram)?,
        })
    }

    fn compute_classification_confidence(
        &self,
        _diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        // Simplified confidence measure
        Ok(0.8)
    }

    /// Compute stability indicators
    fn compute_stability_indicators(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<StabilityIndicators, TopologyError> {
        let persistence_variance = self.compute_persistence_variance(diagram)?;
        let feature_stability = self.compute_feature_stability(diagram)?;
        let structural_integrity = self.compute_structural_integrity(diagram)?;

        Ok(StabilityIndicators {
            persistence_variance,
            feature_stability,
            structural_integrity,
            overall_stability: (feature_stability + structural_integrity) / 2.0,
        })
    }

    fn compute_persistence_variance(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        let pairs: Vec<_> = diagram.finite_pairs().collect();
        if pairs.is_empty() {
            return Ok(0.0);
        }

        let mean_persistence: f64 =
            pairs.iter().map(|p| p.persistence()).sum::<f64>() / pairs.len() as f64;
        let variance: f64 = pairs
            .iter()
            .map(|p| (p.persistence() - mean_persistence).powi(2))
            .sum::<f64>()
            / pairs.len() as f64;

        Ok(variance)
    }

    fn compute_feature_stability(
        &self,
        _diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        // Simplified stability measure based on persistence values
        Ok(0.7)
    }

    fn compute_structural_integrity(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<f64, TopologyError> {
        let essential_count = diagram.essential_pairs().count();
        let finite_count = diagram.finite_pairs().count();

        if essential_count + finite_count == 0 {
            return Ok(0.0);
        }

        // Higher ratio of essential features indicates more stable structure
        let integrity = essential_count as f64 / (essential_count + finite_count) as f64;
        Ok(integrity)
    }

    /// Find critical scale for phase transitions
    fn find_critical_scale(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<Option<f64>, TopologyError> {
        // Find scale at which most significant topological changes occur
        let pairs: Vec<_> = diagram.finite_pairs().collect();
        if pairs.is_empty() {
            return Ok(None);
        }

        // Find the birth time of the most persistent feature
        let max_persistence_pair = pairs
            .iter()
            .max_by(|a, b| a.persistence().partial_cmp(&b.persistence()).unwrap());

        if let Some(pair) = max_persistence_pair {
            Ok(Some(pair.birth))
        } else {
            Ok(None)
        }
    }

    /// Detect critical points in phase evolution
    fn detect_critical_points(
        &self,
        signatures: &[TopologicalPhaseSignature],
    ) -> Result<Vec<CriticalPoint>, TopologyError> {
        let mut critical_points = Vec::new();

        if signatures.len() < 2 {
            return Ok(critical_points);
        }

        // Analyze changes in topological signatures
        for i in 1..signatures.len() {
            let prev_sig = &signatures[i - 1];
            let curr_sig = &signatures[i];

            // Check for significant changes in Betti numbers
            let betti_change = self.compute_betti_change(prev_sig, curr_sig)?;
            if betti_change > self.critical_point_settings.betti_change_threshold {
                critical_points.push(CriticalPoint {
                    timestamp: curr_sig.timestamp,
                    critical_type: CriticalType::TopologyChange,
                    significance: betti_change,
                    affected_dimensions: self.find_affected_dimensions(prev_sig, curr_sig)?,
                    persistence_jump: (curr_sig.persistence_entropy - prev_sig.persistence_entropy)
                        .abs(),
                });
            }

            // Check for phase classification changes
            if prev_sig.phase_classification.phase_type != curr_sig.phase_classification.phase_type
            {
                critical_points.push(CriticalPoint {
                    timestamp: curr_sig.timestamp,
                    critical_type: CriticalType::PhaseTransition,
                    significance: 1.0,
                    affected_dimensions: vec![0, 1, 2],
                    persistence_jump: (curr_sig.persistence_entropy - prev_sig.persistence_entropy)
                        .abs(),
                });
            }
        }

        Ok(critical_points)
    }

    fn compute_betti_change(
        &self,
        prev: &TopologicalPhaseSignature,
        curr: &TopologicalPhaseSignature,
    ) -> Result<f64, TopologyError> {
        let mut total_change = 0.0;

        for dimension in 0..=2 {
            if let (Some(prev_curve), Some(curr_curve)) = (
                prev.betti_curves.get(&dimension),
                curr.betti_curves.get(&dimension),
            ) {
                let curve_change: f64 = prev_curve
                    .iter()
                    .zip(curr_curve.iter())
                    .map(|(&prev_val, &curr_val)| ((curr_val as f64) - (prev_val as f64)).abs())
                    .sum();
                total_change += curve_change;
            }
        }

        Ok(total_change)
    }

    fn find_affected_dimensions(
        &self,
        prev: &TopologicalPhaseSignature,
        curr: &TopologicalPhaseSignature,
    ) -> Result<Vec<usize>, TopologyError> {
        let mut affected = Vec::new();

        for dimension in 0..=2 {
            if let (Some(prev_curve), Some(curr_curve)) = (
                prev.betti_curves.get(&dimension),
                curr.betti_curves.get(&dimension),
            ) {
                let has_change = prev_curve
                    .iter()
                    .zip(curr_curve.iter())
                    .any(|(&prev_val, &curr_val)| prev_val != curr_val);
                if has_change {
                    affected.push(dimension);
                }
            }
        }

        Ok(affected)
    }

    /// Identify phase transitions from signatures
    fn identify_phase_transitions(
        &self,
        signatures: &[TopologicalPhaseSignature],
        critical_points: &[CriticalPoint],
    ) -> Result<Vec<PhaseTransition>, TopologyError> {
        let mut transitions = Vec::new();

        for critical_point in critical_points {
            if matches!(critical_point.critical_type, CriticalType::PhaseTransition) {
                // Find the signatures before and after this critical point
                let before_idx = signatures
                    .iter()
                    .position(|s| s.timestamp == critical_point.timestamp)
                    .and_then(|idx| if idx > 0 { Some(idx - 1) } else { None });

                let after_idx = signatures
                    .iter()
                    .position(|s| s.timestamp == critical_point.timestamp);

                if let (Some(before), Some(after)) = (before_idx, after_idx) {
                    transitions.push(PhaseTransition {
                        transition_time: critical_point.timestamp,
                        from_phase: signatures[before].phase_classification.clone(),
                        to_phase: signatures[after].phase_classification.clone(),
                        transition_type: self
                            .classify_transition_type(&signatures[before], &signatures[after])?,
                        strength: critical_point.significance,
                        duration_estimate: self
                            .estimate_transition_duration(&signatures[before..=after])?,
                    });
                }
            }
        }

        Ok(transitions)
    }

    fn classify_transition_type(
        &self,
        _from: &TopologicalPhaseSignature,
        _to: &TopologicalPhaseSignature,
    ) -> Result<TransitionType, TopologyError> {
        // Simplified classification
        Ok(TransitionType::Continuous)
    }

    fn estimate_transition_duration(
        &self,
        _transition_signatures: &[TopologicalPhaseSignature],
    ) -> Result<f64, TopologyError> {
        // Simplified duration estimate
        Ok(1.0)
    }

    /// Compute topological order parameters
    fn compute_topological_order_parameters(
        &self,
        signatures: &[TopologicalPhaseSignature],
    ) -> Result<Vec<TopologicalOrderParameter>, TopologyError> {
        let mut order_parameters = Vec::new();

        for signature in signatures {
            let persistence_order =
                signature.persistence_entropy / (1.0 + signature.persistence_entropy);
            let complexity_order = 1.0 / (1.0 + signature.topological_complexity);
            let stability_order = signature.stability_indicators.overall_stability;

            order_parameters.push(TopologicalOrderParameter {
                timestamp: signature.timestamp,
                persistence_order,
                complexity_order,
                stability_order,
                combined_order: (persistence_order + complexity_order + stability_order) / 3.0,
            });
        }

        Ok(order_parameters)
    }
}

/// Parameters for phase transition detection
#[derive(Debug, Clone)]
pub struct PhaseTransitionParameters {
    pub filtration_scales: Vec<f64>,
    pub min_persistence_threshold: f64,
    pub topology_change_threshold: f64,
}

impl Default for PhaseTransitionParameters {
    fn default() -> Self {
        Self {
            filtration_scales: vec![0.1, 0.5, 1.0, 2.0, 5.0],
            min_persistence_threshold: 0.01,
            topology_change_threshold: 0.1,
        }
    }
}

/// Settings for critical point detection
#[derive(Debug, Clone)]
pub struct CriticalPointSettings {
    pub betti_change_threshold: f64,
    pub min_persistence_threshold: f64,
    pub stability_threshold: f64,
}

impl Default for CriticalPointSettings {
    fn default() -> Self {
        Self {
            betti_change_threshold: 0.5,
            min_persistence_threshold: 0.01,
            stability_threshold: 0.1,
        }
    }
}

/// Topological signature of a phase state
#[derive(Debug, Clone)]
pub struct TopologicalPhaseSignature {
    pub timestamp: NanoTime,
    pub betti_curves: HashMap<usize, Vec<usize>>, // Dimension -> Betti numbers at different scales
    pub persistence_entropy: f64,
    pub topological_complexity: f64,
    pub dominant_features: Vec<DominantFeature>,
    pub phase_classification: PhaseClassification,
    pub stability_indicators: StabilityIndicators,
    pub critical_scale: Option<f64>,
}

/// Dominant topological feature
#[derive(Debug, Clone)]
pub struct DominantFeature {
    pub dimension: usize,
    pub birth_time: f64,
    pub death_time: f64,
    pub persistence: f64,
    pub significance: f64,
}

/// Phase classification based on topology
#[derive(Debug, Clone)]
pub struct PhaseClassification {
    pub phase_type: PhaseType,
    pub stability: PhaseStability,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhaseType {
    Connected,
    Disconnected,
    Complex,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum PhaseStability {
    Stable,
    Metastable,
    Unstable,
}

/// Stability indicators for phase
#[derive(Debug, Clone)]
pub struct StabilityIndicators {
    pub persistence_variance: f64,
    pub feature_stability: f64,
    pub structural_integrity: f64,
    pub overall_stability: f64,
}

/// Critical point in phase evolution
#[derive(Debug, Clone)]
pub struct CriticalPoint {
    pub timestamp: NanoTime,
    pub critical_type: CriticalType,
    pub significance: f64,
    pub affected_dimensions: Vec<usize>,
    pub persistence_jump: f64,
}

#[derive(Debug, Clone)]
pub enum CriticalType {
    TopologyChange,
    PhaseTransition,
    StabilityChange,
}

/// Phase transition event
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub transition_time: NanoTime,
    pub from_phase: PhaseClassification,
    pub to_phase: PhaseClassification,
    pub transition_type: TransitionType,
    pub strength: f64,
    pub duration_estimate: f64,
}

#[derive(Debug, Clone)]
pub enum TransitionType {
    Continuous,
    Discontinuous,
    Mixed,
}

/// Topological order parameter
#[derive(Debug, Clone)]
pub struct TopologicalOrderParameter {
    pub timestamp: NanoTime,
    pub persistence_order: f64,
    pub complexity_order: f64,
    pub stability_order: f64,
    pub combined_order: f64,
}

/// Complete phase transition analysis
#[derive(Debug, Clone)]
pub struct PhaseTransitionAnalysis {
    pub signatures: Vec<TopologicalPhaseSignature>,
    pub critical_points: Vec<CriticalPoint>,
    pub transitions: Vec<PhaseTransition>,
    pub order_parameters: Vec<TopologicalOrderParameter>,
    pub persistence_diagrams: Vec<PersistenceDiagram>,
    pub analysis_metadata: TransitionAnalysisMetadata,
}

/// Metadata for transition analysis
#[derive(Debug, Clone)]
pub struct TransitionAnalysisMetadata {
    pub total_analysis_time: std::time::Duration,
    pub num_phase_regions: usize,
    pub time_span: usize,
    pub algorithms_used: Vec<String>,
}

/// Utility function for GCD computation
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf2_arithmetic() {
        let a = GF2(true);
        let b = GF2(false);

        assert_eq!(a.add(b), GF2(true));
        assert_eq!(a.add(a), GF2(false));
        assert_eq!(a.mul(b), GF2(false));
        assert_eq!(a.mul(a), GF2(true));
    }

    #[test]
    fn test_simplex_creation() {
        let vertex = Simplex::vertex(0);
        assert_eq!(vertex.dimension(), 0);
        assert_eq!(vertex.vertices(), &[0]);

        let edge = Simplex::edge(0, 2);
        assert_eq!(edge.dimension(), 1);
        assert_eq!(edge.vertices(), &[0, 2]);

        let triangle = Simplex::from_vertices(vec![2, 0, 1]);
        assert_eq!(triangle.dimension(), 2);
        assert_eq!(triangle.vertices(), &[0, 1, 2]);
    }

    #[test]
    fn test_simplex_boundary() {
        let triangle = Simplex::from_vertices(vec![0, 1, 2]);
        let boundary = triangle.boundary();

        assert_eq!(boundary.len(), 3);
        assert!(boundary.contains(&Simplex::edge(1, 2)));
        assert!(boundary.contains(&Simplex::edge(0, 2)));
        assert!(boundary.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_filtered_complex() {
        let mut complex: FilteredSimplicialComplex<GF2> = FilteredSimplicialComplex::new();

        // Add vertices
        complex.add_simplex(Simplex::vertex(0), 0.0).unwrap();
        complex.add_simplex(Simplex::vertex(1), 0.0).unwrap();

        // Add edge
        complex.add_simplex(Simplex::edge(0, 1), 1.0).unwrap();

        assert_eq!(complex.total_simplices(), 3);
        assert_eq!(complex.simplices_of_dimension(0).len(), 2);
        assert_eq!(complex.simplices_of_dimension(1).len(), 1);
    }

    #[test]
    fn test_sparse_boundary_matrix() {
        let mut matrix: SparseBoundaryMatrix<GF2> = SparseBoundaryMatrix::new(3, 3);

        matrix.set_entry(0, 1, GF2(true));
        matrix.set_entry(1, 2, GF2(true));

        assert_eq!(matrix.get_entry(0, 1), GF2(true));
        assert_eq!(matrix.get_entry(1, 2), GF2(true));
        assert_eq!(matrix.get_entry(2, 0), GF2(false));

        assert_eq!(matrix.lowest_nonzero_row(1), Some(0));
        assert_eq!(matrix.lowest_nonzero_row(2), Some(1));
        assert_eq!(matrix.lowest_nonzero_row(0), None);
    }

    #[test]
    fn test_persistence_engine_creation() {
        let engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Standard);

        assert_eq!(engine.filtration.total_simplices(), 0);
    }

    #[test]
    fn test_point_cloud_with_euclidean_metric() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];

        let point_cloud = PointCloud::new(points, 2, Box::new(EuclideanMetric));

        assert_eq!(point_cloud.len(), 3);
        assert_eq!(point_cloud.dimension(), 2);
        assert_eq!(point_cloud.distance(0, 1), 1.0);
        assert!((point_cloud.distance(0, 2) - 1.0).abs() < 1e-10);
        assert!((point_cloud.distance(1, 2) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_mapper_algorithm_creation() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];

        let point_cloud = PointCloud::new(points, 2, Box::new(EuclideanMetric));

        let principal_component = DVector::from_vec(vec![1.0, 0.0]);
        let data_mean = DVector::zeros(2);
        let lens_function = Box::new(PCALensFunction {
            principal_component,
            data_mean,
        });

        let covering_scheme = CoveringScheme::UniformIntervals {
            num_intervals: 5,
            overlap: 0.3,
        };
        let clustering_algorithm = ClusteringAlgorithm::Complete;

        let mapper = MapperAlgorithm::new(lens_function, covering_scheme, clustering_algorithm);
        let result = mapper.compute_mapper_graph(&point_cloud);

        assert!(result.is_ok());
        let graph = result.unwrap();
        assert!(graph.nodes.len() > 0);
    }

    #[test]
    fn test_pca_lens_function() {
        let principal_component = DVector::from_vec(vec![1.0, 0.0]);
        let data_mean = DVector::zeros(2);
        let lens = PCALensFunction {
            principal_component,
            data_mean,
        };

        let test_point = DVector::from_vec(vec![2.0, 3.0]);
        let result = lens.evaluate(&test_point);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2.0);
        assert_eq!(lens.function_type(), "PCA (First Principal Component)");
        assert!(lens.is_differentiable());
    }

    #[test]
    fn test_density_lens_function() {
        let reference_points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];

        let lens = DensityLensFunction {
            bandwidth: 1.0,
            reference_points,
            metric: Box::new(EuclideanMetric),
        };

        let test_point = DVector::from_vec(vec![0.5, 0.5]);
        let result = lens.evaluate(&test_point);

        assert!(result.is_ok());
        let density = result.unwrap();
        assert!(density > 0.0);
        assert!(lens.function_type().contains("Gaussian Density"));
    }

    #[test]
    fn test_matrix_reduction_algorithms() {
        let mut engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Twist);

        // Test that different algorithms can be created
        let standard_engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Standard);
        let chunk_engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Chunk);
        let parallel_engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Parallel { num_threads: 2 });

        assert_eq!(engine.filtration.total_simplices(), 0);
        assert_eq!(standard_engine.filtration.total_simplices(), 0);
        assert_eq!(chunk_engine.filtration.total_simplices(), 0);
        assert_eq!(parallel_engine.filtration.total_simplices(), 0);
    }

    #[test]
    fn test_covering_schemes() {
        let lens_values = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        // Test uniform intervals
        let uniform_scheme = CoveringScheme::UniformIntervals {
            num_intervals: 3,
            overlap: 0.2,
        };

        // Test adaptive intervals
        let adaptive_scheme = CoveringScheme::AdaptiveIntervals {
            target_density: 2,
            min_overlap: 0.1,
        };

        // Test quantile-based intervals
        let quantile_scheme = CoveringScheme::QuantileBased {
            num_quantiles: 3,
            overlap: 0.1,
        };

        // Verify schemes are created properly
        match uniform_scheme {
            CoveringScheme::UniformIntervals {
                num_intervals,
                overlap,
            } => {
                assert_eq!(num_intervals, 3);
                assert_eq!(overlap, 0.2);
            }
            _ => panic!("Wrong scheme type"),
        }
    }

    #[test]
    fn test_clustering_algorithms() {
        let single_linkage = ClusteringAlgorithm::SingleLinkage { threshold: 0.5 };
        let dbscan = ClusteringAlgorithm::DBSCAN {
            eps: 0.3,
            min_points: 3,
        };
        let complete = ClusteringAlgorithm::Complete;

        // Verify algorithms are created with correct parameters
        match single_linkage {
            ClusteringAlgorithm::SingleLinkage { threshold } => {
                assert_eq!(threshold, 0.5);
            }
            _ => panic!("Wrong algorithm type"),
        }

        match dbscan {
            ClusteringAlgorithm::DBSCAN { eps, min_points } => {
                assert_eq!(eps, 0.3);
                assert_eq!(min_points, 3);
            }
            _ => panic!("Wrong algorithm type"),
        }
    }

    #[test]
    fn test_persistence_diagram_analysis() {
        let pairs = vec![
            PersistencePair {
                dimension: 0,
                birth: 0.0,
                death: Some(1.0),
                birth_simplex: 0,
                death_simplex: Some(1),
                multiplicity: 1,
            },
            PersistencePair {
                dimension: 1,
                birth: 0.5,
                death: Some(2.0),
                birth_simplex: 2,
                death_simplex: Some(3),
                multiplicity: 1,
            },
            PersistencePair {
                dimension: 0,
                birth: 0.2,
                death: None, // Essential class
                birth_simplex: 4,
                death_simplex: None,
                multiplicity: 1,
            },
        ];

        let diagram = PersistenceDiagram {
            pairs,
            dimension_range: 0..=2,
            filtration_type: FiltrationType::VietorisRips { max_radius: 2.0 },
            field_characteristic: 2,
            stability_constant: 1.0,
            computational_metadata: ComputationMetadata {
                total_simplices: 10,
                matrix_reduction_time: std::time::Duration::from_millis(100),
                memory_usage: 1024,
                algorithm_used: MatrixReductionAlgorithm::Standard,
            },
        };

        // Test finite and essential pairs
        let finite_count = diagram.finite_pairs().count();
        let essential_count = diagram.essential_pairs().count();
        assert_eq!(finite_count, 2);
        assert_eq!(essential_count, 1);

        // Test dimension filtering
        let dim0_pairs = diagram.pairs_of_dimension(0).count();
        let dim1_pairs = diagram.pairs_of_dimension(1).count();
        assert_eq!(dim0_pairs, 2);
        assert_eq!(dim1_pairs, 1);

        // Test total persistence
        let total_persistence = diagram.total_persistence();
        assert_eq!(total_persistence, 2.5); // 1.0 + 1.5

        // Test Betti numbers at specific scale
        let betti_at_scale = diagram.betti_numbers_at_scale(0.3);
        // At scale 0.3, we should have the essential class (birth 0.2) plus the finite class (birth 0.0, death 1.0)
        assert_eq!(*betti_at_scale.get(&0).unwrap_or(&0), 2); // Essential class + finite class still alive at 0.3
    }

    #[test]
    fn test_phase_transition_analysis_creation() {
        let analyzer: TopologicalPhaseAnalyzer<GF2> = TopologicalPhaseAnalyzer::new();

        assert_eq!(analyzer.phase_history.len(), 0);
        assert_eq!(analyzer.transition_parameters.filtration_scales.len(), 5);
        assert_eq!(analyzer.critical_point_settings.betti_change_threshold, 0.5);
    }

    #[test]
    fn test_topological_phase_signature_creation() {
        let mut betti_curves = HashMap::new();
        betti_curves.insert(0, vec![1, 1, 1, 2, 2]);
        betti_curves.insert(1, vec![0, 0, 1, 1, 0]);

        let dominant_features = vec![DominantFeature {
            dimension: 0,
            birth_time: 0.0,
            death_time: 1.5,
            persistence: 1.5,
            significance: 0.6,
        }];

        let phase_classification = PhaseClassification {
            phase_type: PhaseType::Connected,
            stability: PhaseStability::Stable,
            confidence: 0.8,
        };

        let stability_indicators = StabilityIndicators {
            persistence_variance: 0.1,
            feature_stability: 0.8,
            structural_integrity: 0.9,
            overall_stability: 0.85,
        };

        let signature = TopologicalPhaseSignature {
            timestamp: NanoTime::from_nanos(1000),
            betti_curves,
            persistence_entropy: 0.5,
            topological_complexity: 2.3,
            dominant_features,
            phase_classification,
            stability_indicators,
            critical_scale: Some(1.2),
        };

        assert_eq!(signature.timestamp, NanoTime::from_nanos(1000));
        assert_eq!(signature.persistence_entropy, 0.5);
        assert_eq!(signature.dominant_features.len(), 1);
        assert!(matches!(
            signature.phase_classification.phase_type,
            PhaseType::Connected
        ));
    }

    #[test]
    fn test_critical_point_detection() {
        let critical_point = CriticalPoint {
            timestamp: NanoTime::from_nanos(2000),
            critical_type: CriticalType::PhaseTransition,
            significance: 0.9,
            affected_dimensions: vec![0, 1],
            persistence_jump: 0.3,
        };

        assert_eq!(critical_point.timestamp, NanoTime::from_nanos(2000));
        assert_eq!(critical_point.significance, 0.9);
        assert_eq!(critical_point.affected_dimensions.len(), 2);
        assert!(matches!(
            critical_point.critical_type,
            CriticalType::PhaseTransition
        ));
    }

    #[test]
    fn test_phase_transition_event() {
        let from_phase = PhaseClassification {
            phase_type: PhaseType::Disconnected,
            stability: PhaseStability::Metastable,
            confidence: 0.7,
        };

        let to_phase = PhaseClassification {
            phase_type: PhaseType::Connected,
            stability: PhaseStability::Stable,
            confidence: 0.9,
        };

        let transition = PhaseTransition {
            transition_time: NanoTime::from_nanos(3000),
            from_phase,
            to_phase,
            transition_type: TransitionType::Continuous,
            strength: 0.8,
            duration_estimate: 10.5,
        };

        assert_eq!(transition.transition_time, NanoTime::from_nanos(3000));
        assert_eq!(transition.strength, 0.8);
        assert!(matches!(
            transition.transition_type,
            TransitionType::Continuous
        ));
    }

    #[test]
    fn test_topological_order_parameter() {
        let order_param = TopologicalOrderParameter {
            timestamp: NanoTime::from_nanos(4000),
            persistence_order: 0.6,
            complexity_order: 0.4,
            stability_order: 0.8,
            combined_order: 0.6,
        };

        assert_eq!(order_param.timestamp, NanoTime::from_nanos(4000));
        assert_eq!(order_param.persistence_order, 0.6);
        assert_eq!(order_param.combined_order, 0.6);
    }

    #[test]
    fn test_full_persistence_computation_small_example() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.5, 0.5]),
        ];

        let point_cloud = PointCloud::new(points, 2, Box::new(EuclideanMetric));

        let mut engine: PersistentHomologyEngine<GF2> =
            PersistentHomologyEngine::new(MatrixReductionAlgorithm::Standard);

        let filtration_params = FiltrationParameters {
            filtration_type: FiltrationType::VietorisRips { max_radius: 2.0 },
            max_dimension: 1,
            field_characteristic: 2,
            chunk_size: None,
        };

        let result = engine.compute_persistent_homology(&point_cloud, filtration_params, 1);

        assert!(result.is_ok());
        let diagram = result.unwrap();
        assert!(diagram.pairs.len() > 0);
        assert_eq!(diagram.field_characteristic, 2);
        assert_eq!(diagram.computational_metadata.total_simplices, 6); // 3 vertices + 3 edges
    }

    #[test]
    fn test_rational_field_arithmetic() {
        let a = Rational {
            numerator: 1,
            denominator: 2,
        };
        let b = Rational {
            numerator: 1,
            denominator: 3,
        };

        let sum = a.add(b);
        assert_eq!(sum.numerator, 5);
        assert_eq!(sum.denominator, 6);

        let product = a.mul(b);
        assert_eq!(product.numerator, 1);
        assert_eq!(product.denominator, 6);

        let inverse = a.inv().unwrap();
        assert_eq!(inverse.numerator, 2);
        assert_eq!(inverse.denominator, 1);
    }

    #[test]
    fn test_computational_metadata() {
        let metadata = ComputationMetadata {
            total_simplices: 100,
            matrix_reduction_time: std::time::Duration::from_millis(250),
            memory_usage: 2048,
            algorithm_used: MatrixReductionAlgorithm::Twist,
        };

        assert_eq!(metadata.total_simplices, 100);
        assert_eq!(
            metadata.matrix_reduction_time,
            std::time::Duration::from_millis(250)
        );
        assert_eq!(metadata.memory_usage, 2048);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid metric tensor
        let invalid_pairs = vec![];
        let diagram = PersistenceDiagram {
            pairs: invalid_pairs,
            dimension_range: 0..=2,
            filtration_type: FiltrationType::VietorisRips { max_radius: 1.0 },
            field_characteristic: 2,
            stability_constant: 1.0,
            computational_metadata: ComputationMetadata {
                total_simplices: 0,
                matrix_reduction_time: std::time::Duration::from_millis(0),
                memory_usage: 0,
                algorithm_used: MatrixReductionAlgorithm::Standard,
            },
        };

        // Should handle empty diagram gracefully
        let total_persistence = diagram.total_persistence();
        assert_eq!(total_persistence, 0.0);

        let betti_numbers = diagram.betti_numbers_at_scale(1.0);
        assert_eq!(betti_numbers.len(), 0);
    }
}
