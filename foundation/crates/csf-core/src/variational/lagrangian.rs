//! Lagrangian Mechanics Implementation
//!
//! Implements Lagrangian formalism for the DRPP system, providing the mathematical
//! framework for variational principles, action minimization, and emergent dynamics.
//! This is where the theoretical physics meets computational implementation.

use crate::types::NanoTime;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;

/// Quantum-inspired Lagrangian for relational dynamics
///
/// This structure implements the Lagrangian formalism adapted for quantum-like
/// relational processing. It combines classical Lagrangian mechanics with
/// quantum field theory concepts and relational processing principles.
#[derive(Debug, Clone)]
pub struct QuantumLagrangian {
    /// Kinetic energy matrix (generalized mass matrix)
    pub kinetic_matrix: DMatrix<f64>,

    /// Potential energy field coefficients
    pub potential_coefficients: Array2<f64>,

    /// Interaction coupling constants between components
    pub coupling_constants: DVector<f64>,

    /// Temporal correlation parameters
    pub temporal_params: TemporalParameters,

    /// Field interaction parameters
    pub field_params: FieldParameters,

    /// System dimensionality
    pub dimensions: usize,

    /// Current generalized coordinates
    current_coordinates: DVector<f64>,

    /// Current generalized velocities  
    current_velocities: DVector<f64>,

    /// Action history for path integral calculations
    action_history: Vec<ActionPoint>,
}

/// Temporal parameters for quantum-like behavior
#[derive(Debug, Clone)]
pub struct TemporalParameters {
    /// Quantum time uncertainty parameter (ΔtΔE ≥ ℏ/2)
    pub time_uncertainty: f64,

    /// Phase coherence time scale
    pub coherence_time: NanoTime,

    /// Decoherence rate (1/lifetime)
    pub decoherence_rate: f64,

    /// Temporal correlation strength
    pub correlation_strength: f64,

    /// Memory correlation length (how far back correlations extend)
    pub memory_length: usize,

    /// Retardation effects parameter
    pub retardation_parameter: f64,
}

/// Field interaction parameters for field-theoretic formulation
#[derive(Debug, Clone)]
pub struct FieldParameters {
    /// Field coupling strength matrix
    pub coupling_matrix: DMatrix<f64>,

    /// Field mass matrix (gives field propagation properties)
    pub mass_matrix: DMatrix<f64>,

    /// Self-interaction parameters (φ³, φ⁴ terms, etc.)
    pub self_interaction: Vec<f64>,

    /// Background field configuration
    pub background_field: DVector<f64>,

    /// Gauge parameters (if applicable)
    pub gauge_parameters: DVector<f64>,
}

/// Action point for path integral calculations
#[derive(Debug, Clone)]
pub struct ActionPoint {
    /// Time at this point
    pub time: NanoTime,

    /// System configuration (generalized coordinates)
    pub configuration: DVector<f64>,

    /// Configuration derivatives (generalized velocities)
    pub velocity: DVector<f64>,

    /// Action value at this point
    pub action: f64,

    /// Lagrangian value at this point
    pub lagrangian: f64,
}

/// Euler-Lagrange equation solver
///
/// This structure implements numerical methods for solving the Euler-Lagrange
/// equations that arise from variational principles.
#[derive(Debug, Clone)]
pub struct EulerLagrangeSolver {
    /// The Lagrangian to solve for
    lagrangian: QuantumLagrangian,

    /// Integration method parameters
    integration_params: IntegrationParameters,

    /// Current solution trajectory
    trajectory: Vec<ActionPoint>,

    /// Convergence criteria
    #[allow(dead_code)]
    convergence_tolerance: f64,

    /// Maximum iterations for iterative methods
    #[allow(dead_code)]
    max_iterations: usize,
}

/// Integration parameters for numerical methods
#[derive(Debug, Clone)]
pub struct IntegrationParameters {
    /// Time step size
    pub dt: f64,

    /// Integration method type
    pub method: IntegrationMethod,

    /// Symplectic integration flag (preserves phase space structure)
    pub symplectic: bool,

    /// Adaptive step size control
    pub adaptive: bool,

    /// Error tolerance for adaptive methods
    pub error_tolerance: f64,
}

/// Integration method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    /// 4th order Runge-Kutta
    RungeKutta4,

    /// Velocity-Verlet (symplectic)
    VelocityVerlet,

    /// Leapfrog method (symplectic)
    Leapfrog,

    /// Implicit Euler (stable)
    ImplicitEuler,

    /// Adams-Bashforth predictor-corrector
    AdamsBashforth,
}

impl QuantumLagrangian {
    /// Create a new quantum Lagrangian
    pub fn new(dimensions: usize) -> Self {
        Self {
            kinetic_matrix: DMatrix::identity(dimensions, dimensions),
            potential_coefficients: Array2::zeros((dimensions, dimensions)),
            coupling_constants: DVector::from_element(dimensions, 1.0),
            temporal_params: TemporalParameters {
                time_uncertainty: 1e-15,                    // femtosecond scale
                coherence_time: NanoTime::from_nanos(1000), // nanosecond coherence
                decoherence_rate: 0.001,
                correlation_strength: 1.0,
                memory_length: 100,
                retardation_parameter: 0.1,
            },
            field_params: FieldParameters {
                coupling_matrix: DMatrix::zeros(dimensions, dimensions),
                mass_matrix: DMatrix::identity(dimensions, dimensions),
                self_interaction: vec![0.0, 0.0, 0.1], // φ², φ³, φ⁴ terms
                background_field: DVector::zeros(dimensions),
                gauge_parameters: DVector::zeros(dimensions),
            },
            dimensions,
            current_coordinates: DVector::zeros(dimensions),
            current_velocities: DVector::zeros(dimensions),
            action_history: Vec::new(),
        }
    }

    /// Compute the Lagrangian value L = T - V
    pub fn compute_lagrangian(&self, coordinates: &DVector<f64>, velocities: &DVector<f64>) -> f64 {
        let kinetic = self.compute_kinetic_energy(velocities);
        let potential = self.compute_potential_energy(coordinates);
        let interaction = self.compute_interaction_energy(coordinates);
        let field = self.compute_field_energy(coordinates);

        kinetic - potential - interaction - field
    }

    /// Compute kinetic energy T = (1/2) * v^T * M * v
    pub fn compute_kinetic_energy(&self, velocities: &DVector<f64>) -> f64 {
        if velocities.len() != self.kinetic_matrix.nrows() {
            return 0.0;
        }

        let result = velocities.transpose() * &self.kinetic_matrix * velocities;
        0.5 * result[(0, 0)]
    }

    /// Compute potential energy V
    pub fn compute_potential_energy(&self, coordinates: &DVector<f64>) -> f64 {
        let mut potential = 0.0;

        // Quadratic potential terms
        for i in 0..coordinates.len().min(self.potential_coefficients.nrows()) {
            for j in 0..coordinates.len().min(self.potential_coefficients.ncols()) {
                potential +=
                    0.5 * self.potential_coefficients[(i, j)] * coordinates[i] * coordinates[j];
            }
        }

        potential
    }

    /// Compute interaction energy between components
    pub fn compute_interaction_energy(&self, coordinates: &DVector<f64>) -> f64 {
        let mut interaction = 0.0;

        // Pairwise interactions
        for i in 0..coordinates.len() {
            for j in (i + 1)..coordinates.len() {
                if i < self.coupling_constants.len() && j < self.coupling_constants.len() {
                    let coupling_strength = self.coupling_constants[i] * self.coupling_constants[j];
                    interaction += coupling_strength * coordinates[i] * coordinates[j];
                }
            }
        }

        interaction
    }

    /// Compute field theory contribution
    pub fn compute_field_energy(&self, coordinates: &DVector<f64>) -> f64 {
        let mut field_energy = 0.0;

        // Mass terms: (1/2) * m² * φ²
        if self.field_params.mass_matrix.nrows() == coordinates.len() {
            let result = coordinates.transpose() * &self.field_params.mass_matrix * coordinates;
            field_energy += 0.5 * result[(0, 0)];
        }

        // Self-interaction terms
        if self.field_params.self_interaction.len() >= 3 {
            for &coord in coordinates.iter() {
                // φ² term
                if self.field_params.self_interaction.len() > 0 {
                    field_energy += 0.5 * self.field_params.self_interaction[0] * coord * coord;
                }
                // φ³ term
                if self.field_params.self_interaction.len() > 1 {
                    field_energy +=
                        (1.0 / 6.0) * self.field_params.self_interaction[1] * coord * coord * coord;
                }
                // φ⁴ term
                if self.field_params.self_interaction.len() > 2 {
                    field_energy += 0.25
                        * self.field_params.self_interaction[2]
                        * coord
                        * coord
                        * coord
                        * coord;
                }
            }
        }

        field_energy
    }

    /// Compute Euler-Lagrange equations: d/dt(∂L/∂q̇) - ∂L/∂q = 0
    pub fn compute_euler_lagrange(
        &self,
        coordinates: &DVector<f64>,
        velocities: &DVector<f64>,
    ) -> DVector<f64> {
        let mut equations = DVector::zeros(coordinates.len());

        // ∂L/∂q̇ = M * q̇ (from kinetic energy)
        if velocities.len() == self.kinetic_matrix.nrows() {
            let _momentum = &self.kinetic_matrix * velocities;

            // d/dt(∂L/∂q̇) = M * q̈ (assuming constant mass matrix)
            // For now, we compute the force F = ∂L/∂q and return F (acceleration = M⁻¹F)

            // ∂L/∂q = -∂V/∂q - ∂U/∂q - ∂F/∂q
            for i in 0..coordinates.len() {
                let mut force = 0.0;

                // Potential energy contribution
                for j in 0..coordinates.len().min(self.potential_coefficients.ncols()) {
                    if i < self.potential_coefficients.nrows() {
                        force -= self.potential_coefficients[(i, j)] * coordinates[j];
                    }
                }

                // Interaction energy contribution
                for j in 0..coordinates.len() {
                    if i != j
                        && i < self.coupling_constants.len()
                        && j < self.coupling_constants.len()
                    {
                        let coupling = self.coupling_constants[i] * self.coupling_constants[j];
                        force -= coupling * coordinates[j];
                    }
                }

                // Field energy contribution (mass and self-interaction terms)
                if i < self.field_params.mass_matrix.nrows() {
                    for j in 0..coordinates.len().min(self.field_params.mass_matrix.ncols()) {
                        force -= self.field_params.mass_matrix[(i, j)] * coordinates[j];
                    }
                }

                // Self-interaction derivatives
                if self.field_params.self_interaction.len() >= 3 {
                    let coord = coordinates[i];
                    // d/dφ(φ²) = 2φ
                    if self.field_params.self_interaction.len() > 0 {
                        force -= self.field_params.self_interaction[0] * coord;
                    }
                    // d/dφ(φ³) = 3φ²
                    if self.field_params.self_interaction.len() > 1 {
                        force -= 0.5 * self.field_params.self_interaction[1] * coord * coord;
                    }
                    // d/dφ(φ⁴) = 4φ³
                    if self.field_params.self_interaction.len() > 2 {
                        force -= self.field_params.self_interaction[2] * coord * coord * coord;
                    }
                }

                equations[i] = force;
            }
        }

        equations
    }

    /// Compute action integral S = ∫L dt over time interval
    pub fn compute_action(
        &self,
        start_time: NanoTime,
        end_time: NanoTime,
        trajectory: &[ActionPoint],
    ) -> f64 {
        if trajectory.is_empty() {
            return 0.0;
        }

        let mut action = 0.0;
        let dt =
            (end_time.as_nanos() - start_time.as_nanos()) as f64 / trajectory.len() as f64 * 1e-9;

        for point in trajectory {
            let lagrangian = self.compute_lagrangian(&point.configuration, &point.velocity);
            action += lagrangian * dt;
        }

        action
    }

    /// Update system state
    pub fn update_state(
        &mut self,
        coordinates: DVector<f64>,
        velocities: DVector<f64>,
        time: NanoTime,
    ) {
        self.current_coordinates = coordinates.clone();
        self.current_velocities = velocities.clone();

        // Add to action history
        let lagrangian = self.compute_lagrangian(&coordinates, &velocities);
        let action_point = ActionPoint {
            time,
            configuration: coordinates,
            velocity: velocities,
            action: 0.0, // Will be computed later from integral
            lagrangian,
        };

        self.action_history.push(action_point);

        // Limit history size
        if self.action_history.len() > self.temporal_params.memory_length {
            self.action_history.remove(0);
        }
    }

    /// Apply quantum uncertainty principle constraints
    pub fn apply_uncertainty_constraints(&mut self, _dt: f64) {
        // Heisenberg uncertainty principle: Δx * Δp ≥ ℏ/2
        // For our system: Δq * Δ(∂L/∂q̇) ≥ ℏ/2

        let hbar = 1.0545718e-34; // Planck constant / 2π
        let uncertainty_floor = hbar / 2.0;

        // Apply minimum uncertainty to coordinates and momenta
        for i in 0..self.current_coordinates.len() {
            let coordinate_uncertainty =
                self.temporal_params.time_uncertainty * self.current_velocities[i].abs();

            if coordinate_uncertainty < uncertainty_floor {
                // Add quantum fluctuations to maintain uncertainty principle
                let fluctuation = (uncertainty_floor - coordinate_uncertainty).sqrt();
                self.current_coordinates[i] += fluctuation * (rand::random::<f64>() - 0.5) * 2.0;
            }
        }
    }

    /// Compute quantum phase evolution
    pub fn compute_phase_evolution(&self, dt: f64) -> f64 {
        let lagrangian =
            self.compute_lagrangian(&self.current_coordinates, &self.current_velocities);

        // Phase evolution: φ(t+dt) = φ(t) + (i/ℏ) * ∫L dt
        let hbar = 1.0545718e-34;
        lagrangian * dt / hbar
    }

    /// Get current system energy (Hamiltonian)
    pub fn compute_hamiltonian(&self) -> f64 {
        let kinetic = self.compute_kinetic_energy(&self.current_velocities);
        let potential = self.compute_potential_energy(&self.current_coordinates);
        let interaction = self.compute_interaction_energy(&self.current_coordinates);
        let field = self.compute_field_energy(&self.current_coordinates);

        kinetic + potential + interaction + field
    }
}

impl EulerLagrangeSolver {
    /// Create a new Euler-Lagrange solver
    pub fn new(lagrangian: QuantumLagrangian) -> Self {
        let integration_params = IntegrationParameters {
            dt: 1e-6, // microsecond time step
            method: IntegrationMethod::VelocityVerlet,
            symplectic: true,
            adaptive: false,
            error_tolerance: 1e-8,
        };

        Self {
            lagrangian,
            integration_params,
            trajectory: Vec::new(),
            convergence_tolerance: 1e-10,
            max_iterations: 10000,
        }
    }

    /// Solve Euler-Lagrange equations with initial conditions
    pub fn solve(
        &mut self,
        initial_coordinates: DVector<f64>,
        initial_velocities: DVector<f64>,
        time_span: (NanoTime, NanoTime),
    ) -> Result<Vec<ActionPoint>, String> {
        let (start_time, end_time) = time_span;
        let total_time = (end_time.as_nanos() - start_time.as_nanos()) as f64 * 1e-9;
        let num_steps = (total_time / self.integration_params.dt) as usize;

        let mut trajectory = Vec::with_capacity(num_steps);
        let mut coordinates = initial_coordinates;
        let mut velocities = initial_velocities;
        let mut current_time = start_time;

        for step in 0..num_steps {
            // Record current state
            let lagrangian = self
                .lagrangian
                .compute_lagrangian(&coordinates, &velocities);
            let action_point = ActionPoint {
                time: current_time,
                configuration: coordinates.clone(),
                velocity: velocities.clone(),
                action: 0.0, // Will be computed from integral
                lagrangian,
            };
            trajectory.push(action_point);

            // Integrate one step forward
            match self.integration_params.method {
                IntegrationMethod::VelocityVerlet => {
                    self.velocity_verlet_step(&mut coordinates, &mut velocities)?;
                }
                IntegrationMethod::RungeKutta4 => {
                    self.runge_kutta_step(&mut coordinates, &mut velocities)?;
                }
                IntegrationMethod::Leapfrog => {
                    self.leapfrog_step(&mut coordinates, &mut velocities)?;
                }
                _ => {
                    return Err("Integration method not yet implemented".to_string());
                }
            }

            // Update time
            current_time = NanoTime::from_nanos(
                current_time.as_nanos() + (self.integration_params.dt * 1e9) as u64,
            );

            // Apply quantum constraints
            if step % 100 == 0 {
                // Apply periodically to avoid computational overhead
                self.lagrangian
                    .apply_uncertainty_constraints(self.integration_params.dt);
            }
        }

        // Compute action integrals for all points
        self.compute_action_integrals(&mut trajectory, (start_time, end_time));

        self.trajectory = trajectory.clone();
        Ok(trajectory)
    }

    /// Velocity-Verlet integration step (symplectic)
    fn velocity_verlet_step(
        &self,
        coordinates: &mut DVector<f64>,
        velocities: &mut DVector<f64>,
    ) -> Result<(), String> {
        let dt = self.integration_params.dt;

        // Compute current acceleration
        let forces = self
            .lagrangian
            .compute_euler_lagrange(coordinates, velocities);
        let accelerations = if self.lagrangian.kinetic_matrix.nrows() == forces.len() {
            match self.lagrangian.kinetic_matrix.clone().try_inverse() {
                Some(inv_mass) => inv_mass * forces,
                None => return Err("Singular mass matrix - cannot invert".to_string()),
            }
        } else {
            forces // Assume unit mass
        };

        // Update coordinates: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        *coordinates += &*velocities * dt + &accelerations * (0.5 * dt * dt);

        // Compute new acceleration at updated position
        let new_forces = self
            .lagrangian
            .compute_euler_lagrange(coordinates, velocities);
        let new_accelerations = if self.lagrangian.kinetic_matrix.nrows() == new_forces.len() {
            match self.lagrangian.kinetic_matrix.clone().try_inverse() {
                Some(inv_mass) => inv_mass * new_forces,
                None => return Err("Singular mass matrix - cannot invert".to_string()),
            }
        } else {
            new_forces
        };

        // Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        *velocities += (&accelerations + &new_accelerations) * (0.5 * dt);

        Ok(())
    }

    /// 4th order Runge-Kutta step
    fn runge_kutta_step(
        &self,
        coordinates: &mut DVector<f64>,
        velocities: &mut DVector<f64>,
    ) -> Result<(), String> {
        let dt = self.integration_params.dt;

        // This is more complex for 2nd order differential equations
        // We need to treat it as a system of 1st order equations
        // Let y = [q, q̇], then dy/dt = [q̇, q̈] where q̈ = M⁻¹F(q,q̇)

        let n = coordinates.len();
        let mut y = DVector::zeros(2 * n);

        // Pack state vector
        for i in 0..n {
            y[i] = coordinates[i];
            y[n + i] = velocities[i];
        }

        // Compute k1
        let k1 = self.compute_derivative(&y)?;

        // Compute k2
        let y2 = &y + &k1 * (dt / 2.0);
        let k2 = self.compute_derivative(&y2)?;

        // Compute k3
        let y3 = &y + &k2 * (dt / 2.0);
        let k3 = self.compute_derivative(&y3)?;

        // Compute k4
        let y4 = &y + &k3 * dt;
        let k4 = self.compute_derivative(&y4)?;

        // Final update
        let dy = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        y += dy;

        // Unpack state vector
        for i in 0..n {
            coordinates[i] = y[i];
            velocities[i] = y[n + i];
        }

        Ok(())
    }

    /// Compute derivative for RK4 method
    fn compute_derivative(&self, y: &DVector<f64>) -> Result<DVector<f64>, String> {
        let n = y.len() / 2;
        let mut dy = DVector::zeros(y.len());

        // Extract coordinates and velocities
        let coordinates = y.rows(0, n).clone_owned();
        let velocities = y.rows(n, n).clone_owned();

        // dy[0..n] = velocities (dq/dt = q̇)
        for i in 0..n {
            dy[i] = velocities[i];
        }

        // dy[n..2n] = accelerations (dq̇/dt = q̈ = M⁻¹F)
        let forces = self
            .lagrangian
            .compute_euler_lagrange(&coordinates, &velocities);
        let accelerations = if self.lagrangian.kinetic_matrix.nrows() == forces.len() {
            match self.lagrangian.kinetic_matrix.clone().try_inverse() {
                Some(inv_mass) => inv_mass * forces,
                None => return Err("Singular mass matrix - cannot invert".to_string()),
            }
        } else {
            forces
        };

        for i in 0..n {
            dy[n + i] = accelerations[i];
        }

        Ok(dy)
    }

    /// Leapfrog integration step (symplectic)
    fn leapfrog_step(
        &self,
        coordinates: &mut DVector<f64>,
        velocities: &mut DVector<f64>,
    ) -> Result<(), String> {
        let dt = self.integration_params.dt;

        // Leapfrog: kick-drift-kick
        // 1. Half kick: v(t+dt/2) = v(t) + a(t) * dt/2
        let forces = self
            .lagrangian
            .compute_euler_lagrange(coordinates, velocities);
        let accelerations = if self.lagrangian.kinetic_matrix.nrows() == forces.len() {
            match self.lagrangian.kinetic_matrix.clone().try_inverse() {
                Some(inv_mass) => inv_mass * forces,
                None => return Err("Singular mass matrix - cannot invert".to_string()),
            }
        } else {
            forces
        };

        *velocities += &accelerations * (dt / 2.0);

        // 2. Drift: x(t+dt) = x(t) + v(t+dt/2) * dt
        *coordinates += &*velocities * dt;

        // 3. Half kick: v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
        let new_forces = self
            .lagrangian
            .compute_euler_lagrange(coordinates, velocities);
        let new_accelerations = if self.lagrangian.kinetic_matrix.nrows() == new_forces.len() {
            match self.lagrangian.kinetic_matrix.clone().try_inverse() {
                Some(inv_mass) => inv_mass * new_forces,
                None => return Err("Singular mass matrix - cannot invert".to_string()),
            }
        } else {
            new_forces
        };

        *velocities += &new_accelerations * (dt / 2.0);

        Ok(())
    }

    /// Compute action integrals for trajectory points
    fn compute_action_integrals(
        &self,
        trajectory: &mut [ActionPoint],
        time_span: (NanoTime, NanoTime),
    ) {
        let (start_time, end_time) = time_span;
        let dt =
            (end_time.as_nanos() - start_time.as_nanos()) as f64 / trajectory.len() as f64 * 1e-9;

        let mut cumulative_action = 0.0;
        for point in trajectory.iter_mut() {
            cumulative_action += point.lagrangian * dt;
            point.action = cumulative_action;
        }
    }

    /// Get the solved trajectory
    pub fn trajectory(&self) -> &[ActionPoint] {
        &self.trajectory
    }

    /// Set integration parameters
    pub fn set_integration_params(&mut self, params: IntegrationParameters) {
        self.integration_params = params;
    }
}

// Add a simple implementation of rand::random for the quantum fluctuations
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        let mut hasher = DefaultHasher::new();
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        let hash = hasher.finish();
        let normalized = (hash as f64) / (u64::MAX as f64);
        T::from(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_lagrangian_creation() {
        let lagrangian = QuantumLagrangian::new(3);
        assert_eq!(lagrangian.dimensions, 3);
        assert_eq!(lagrangian.kinetic_matrix.nrows(), 3);
        assert_eq!(lagrangian.kinetic_matrix.ncols(), 3);
    }

    #[test]
    fn test_lagrangian_computation() {
        let lagrangian = QuantumLagrangian::new(2);
        let coordinates = DVector::from_vec(vec![1.0, 2.0]);
        let velocities = DVector::from_vec(vec![0.5, -0.5]);

        let l_value = lagrangian.compute_lagrangian(&coordinates, &velocities);
        assert!(l_value.is_finite());
    }

    #[test]
    fn test_kinetic_energy() {
        let lagrangian = QuantumLagrangian::new(2);
        let velocities = DVector::from_vec(vec![1.0, 1.0]);

        let kinetic = lagrangian.compute_kinetic_energy(&velocities);
        assert_eq!(kinetic, 1.0); // 0.5 * (1² + 1²) = 1.0 for identity mass matrix
    }

    #[test]
    fn test_euler_lagrange_equations() {
        let lagrangian = QuantumLagrangian::new(2);
        let coordinates = DVector::from_vec(vec![1.0, 0.0]);
        let velocities = DVector::from_vec(vec![0.0, 1.0]);

        let equations = lagrangian.compute_euler_lagrange(&coordinates, &velocities);
        assert_eq!(equations.len(), 2);
        assert!(equations.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_action_computation() {
        let lagrangian = QuantumLagrangian::new(2);
        let start_time = NanoTime::from_nanos(0);
        let end_time = NanoTime::from_nanos(1000);

        let trajectory = vec![ActionPoint {
            time: start_time,
            configuration: DVector::from_vec(vec![1.0, 0.0]),
            velocity: DVector::from_vec(vec![0.0, 1.0]),
            action: 0.0,
            lagrangian: 0.5,
        }];

        let action = lagrangian.compute_action(start_time, end_time, &trajectory);
        assert!(action.is_finite());
    }

    #[test]
    fn test_solver_creation() {
        let lagrangian = QuantumLagrangian::new(2);
        let solver = EulerLagrangeSolver::new(lagrangian);
        assert_eq!(solver.max_iterations, 10000);
    }

    #[test]
    fn test_hamiltonian_computation() {
        let mut lagrangian = QuantumLagrangian::new(2);
        lagrangian.current_coordinates = DVector::from_vec(vec![1.0, 0.0]);
        lagrangian.current_velocities = DVector::from_vec(vec![0.0, 1.0]);

        let hamiltonian = lagrangian.compute_hamiltonian();
        assert!(hamiltonian.is_finite());
        assert!(hamiltonian >= 0.0); // Energy should be non-negative for this configuration
    }
}
