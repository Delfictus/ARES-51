//! Dynamical system implementation for Energy Management System

use anyhow::Result;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Dynamical system configuration
#[derive(Debug, Clone)]
pub struct DynamicsConfig {
    /// System dimension
    pub dimension: usize,

    /// Time step
    pub dt: f64,

    /// Lyapunov function parameters
    pub lyapunov_params: LyapunovParams,

    /// Control parameters
    pub control_params: ControlParams,

    /// Stability margin
    pub stability_margin: f64,

    /// Enable adaptive control
    pub adaptive_control: bool,
}

#[derive(Debug, Clone)]
pub struct LyapunovParams {
    /// Q matrix for quadratic Lyapunov function
    pub q_matrix: Array2<f64>,

    /// Decay rate
    pub decay_rate: f64,

    /// Barrier function weight
    pub barrier_weight: f64,
}

#[derive(Debug, Clone)]
pub struct ControlParams {
    /// Control gain matrix
    pub k_gain: Array2<f64>,

    /// Maximum control effort
    pub u_max: f64,

    /// Control horizon
    pub horizon: usize,

    /// Prediction steps
    pub prediction_steps: usize,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        let dim = 10;
        Self {
            dimension: dim,
            dt: 0.001,
            lyapunov_params: LyapunovParams {
                q_matrix: Array2::eye(dim),
                decay_rate: 0.1,
                barrier_weight: 1.0,
            },
            control_params: ControlParams {
                k_gain: Array2::eye(dim) * 0.5,
                u_max: 10.0,
                horizon: 50,
                prediction_steps: 10,
            },
            stability_margin: 0.1,
            adaptive_control: true,
        }
    }
}

/// Energy-aware dynamical system
pub struct EnergyDynamicalSystem {
    config: DynamicsConfig,
    state: Array1<f64>,
    energy_function: Box<dyn EnergyFunction>,
    controller: Box<dyn Controller>,
    observer: StateObserver,
    stability_analyzer: StabilityAnalyzer,
}

/// Energy function trait
pub trait EnergyFunction: Send + Sync {
    /// Compute energy at state
    fn energy(&self, state: &Array1<f64>) -> f64;

    /// Compute energy gradient
    fn gradient(&self, state: &Array1<f64>) -> Array1<f64>;

    /// Compute energy Hessian
    fn hessian(&self, state: &Array1<f64>) -> Array2<f64>;
}

/// Controller trait
pub trait Controller: Send + Sync {
    /// Compute control input
    fn control(&mut self, state: &Array1<f64>, reference: &Array1<f64>) -> Array1<f64>;

    /// Update controller parameters
    fn update(&mut self, state: &Array1<f64>, error: &Array1<f64>);
}

impl EnergyDynamicalSystem {
    /// Create new dynamical system
    pub fn new(config: DynamicsConfig) -> Result<Self> {
        let state = Array1::zeros(config.dimension);
        let energy_function = Box::new(QuadraticEnergy::new(&config));
        let controller = Box::new(LyapunovController::new(&config));
        let observer = StateObserver::new(&config);
        let stability_analyzer = StabilityAnalyzer::new(&config);

        Ok(Self {
            config,
            state,
            energy_function,
            controller,
            observer,
            stability_analyzer,
        })
    }

    /// Step the dynamical system
    pub fn step(
        &mut self,
        external_input: &Array1<f64>,
        reference: &Array1<f64>,
    ) -> Result<SystemState> {
        // Estimate full state from observations
        let estimated_state = self.observer.estimate(&self.state);

        // Compute control input
        let control = self.controller.control(&estimated_state, reference);

        // Apply control limits
        let control = self.limit_control(&control);

        // Compute dynamics
        let dynamics = self.compute_dynamics(&self.state, &control, external_input)?;

        // Integrate state
        self.state = &self.state + &dynamics * self.config.dt;

        // Update observer
        self.observer.update(&self.state);

        // Update controller if adaptive
        if self.config.adaptive_control {
            let error = reference - &self.state;
            self.controller.update(&self.state, &error);
        }

        // Analyze stability
        let stability = self.stability_analyzer.analyze(&self.state, &dynamics)?;

        // Compute energy
        let energy = self.energy_function.energy(&self.state);

        Ok(SystemState {
            state: self.state.clone(),
            control,
            energy,
            stability,
            estimated_state,
        })
    }

    /// Compute system dynamics dx/dt = f(x, u, w)
    fn compute_dynamics(
        &self,
        state: &Array1<f64>,
        control: &Array1<f64>,
        external: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // Nonlinear dynamics with energy dissipation
        let linear_part = self.compute_linear_dynamics(state);
        let nonlinear_part = self.compute_nonlinear_dynamics(state);
        let dissipation = self.compute_dissipation(state);

        Ok(linear_part + nonlinear_part + control + external - dissipation)
    }

    /// Linear dynamics component
    fn compute_linear_dynamics(&self, state: &Array1<f64>) -> Array1<f64> {
        // A matrix defines linear dynamics
        let a_matrix = self.get_system_matrix();
        a_matrix.dot(state)
    }

    /// Nonlinear dynamics component
    fn compute_nonlinear_dynamics(&self, state: &Array1<f64>) -> Array1<f64> {
        let mut nonlinear = Array1::zeros(self.config.dimension);

        // Example: Van der Pol-like nonlinearity
        for i in 0..self.config.dimension {
            if i > 0 {
                let mu = 0.1;
                nonlinear[i] = mu * (1.0 - state[i - 1].powi(2)) * state[i];
            }
        }

        // Coupling terms
        for i in 1..self.config.dimension - 1 {
            nonlinear[i] += 0.05 * (state[i - 1] - 2.0 * state[i] + state[i + 1]);
        }

        nonlinear
    }

    /// Energy dissipation term
    fn compute_dissipation(&self, state: &Array1<f64>) -> Array1<f64> {
        // Gradient-based dissipation
        let gradient = self.energy_function.gradient(state);
        gradient * self.config.lyapunov_params.decay_rate
    }

    /// Get system matrix
    fn get_system_matrix(&self) -> Array2<f64> {
        let n = self.config.dimension;
        let mut a = Array2::zeros((n, n));

        // Tridiagonal structure with energy-aware coupling
        for i in 0..n {
            a[[i, i]] = -0.5;
            if i > 0 {
                a[[i, i - 1]] = 0.2;
            }
            if i < n - 1 {
                a[[i, i + 1]] = 0.2;
            }
        }

        a
    }

    /// Limit control input
    fn limit_control(&self, control: &Array1<f64>) -> Array1<f64> {
        let u_max = self.config.control_params.u_max;
        control.mapv(|u| u.max(-u_max).min(u_max))
    }

    /// Set system state
    pub fn set_state(&mut self, state: Array1<f64>) {
        self.state = state;
        self.observer.reset(&self.state);
    }

    /// Get current energy
    pub fn get_energy(&self) -> f64 {
        self.energy_function.energy(&self.state)
    }

    /// Predict future trajectory
    pub fn predict_trajectory(&self, steps: usize, reference: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut trajectory = Vec::with_capacity(steps);
        let mut pred_state = self.state.clone();
        // For prediction, use a simplified control law without modifying state

        for _ in 0..steps {
            // Use a simplified control approach for prediction
            let control = Array1::zeros(self.config.dimension);
            let dynamics = self
                .compute_dynamics(&pred_state, &control, &Array1::zeros(self.config.dimension))
                .unwrap_or_else(|_| Array1::zeros(self.config.dimension));

            pred_state = &pred_state + &dynamics * self.config.dt;
            trajectory.push(pred_state.clone());
        }

        trajectory
    }
}

/// System state information
#[derive(Debug, Clone)]
pub struct SystemState {
    pub state: Array1<f64>,
    pub control: Array1<f64>,
    pub energy: f64,
    pub stability: StabilityInfo,
    pub estimated_state: Array1<f64>,
}

/// Stability information
#[derive(Debug, Clone)]
pub struct StabilityInfo {
    pub lyapunov_value: f64,
    pub lyapunov_derivative: f64,
    pub is_stable: bool,
    pub stability_margin: f64,
    pub largest_eigenvalue: Complex64,
}

/// Quadratic energy function
struct QuadraticEnergy {
    q_matrix: Array2<f64>,
}

impl QuadraticEnergy {
    fn new(config: &DynamicsConfig) -> Self {
        Self {
            q_matrix: config.lyapunov_params.q_matrix.clone(),
        }
    }
}

impl EnergyFunction for QuadraticEnergy {
    fn energy(&self, state: &Array1<f64>) -> f64 {
        0.5 * state.dot(&self.q_matrix.dot(state))
    }

    fn gradient(&self, state: &Array1<f64>) -> Array1<f64> {
        self.q_matrix.dot(state)
    }

    fn hessian(&self, _state: &Array1<f64>) -> Array2<f64> {
        self.q_matrix.clone()
    }
}

/// Lyapunov-based controller
struct LyapunovController {
    k_gain: Array2<f64>,
    adaptive_gain: Array2<f64>,
}

impl LyapunovController {
    fn new(config: &DynamicsConfig) -> Self {
        Self {
            k_gain: config.control_params.k_gain.clone(),
            adaptive_gain: Array2::zeros(config.control_params.k_gain.dim()),
        }
    }
}

impl Controller for LyapunovController {
    fn control(&mut self, state: &Array1<f64>, reference: &Array1<f64>) -> Array1<f64> {
        let error = reference - state;
        let nominal_control = self.k_gain.dot(&error);
        let adaptive_control = self.adaptive_gain.dot(&error);

        nominal_control + adaptive_control
    }

    fn update(&mut self, _state: &Array1<f64>, error: &Array1<f64>) {
        // Simple adaptation law
        let learning_rate = 0.001;
        let update = error
            .clone()
            .insert_axis(ndarray::Axis(1))
            .dot(&error.clone().insert_axis(ndarray::Axis(0)));

        self.adaptive_gain = &self.adaptive_gain + &(update * learning_rate);

        // Limit adaptive gain
        self.adaptive_gain.mapv_inplace(|g| g.max(-1.0).min(1.0));
    }
}

/// State observer for estimation
struct StateObserver {
    kalman_gain: Array2<f64>,
    estimate: Array1<f64>,
    covariance: Array2<f64>,
}

impl StateObserver {
    fn new(config: &DynamicsConfig) -> Self {
        let n = config.dimension;
        Self {
            kalman_gain: Array2::eye(n) * 0.1,
            estimate: Array1::zeros(n),
            covariance: Array2::eye(n),
        }
    }

    fn estimate(&self, measurement: &Array1<f64>) -> Array1<f64> {
        // Simple estimation (in practice, use Kalman filter)
        let innovation = measurement - &self.estimate;
        &self.estimate + self.kalman_gain.dot(&innovation)
    }

    fn update(&mut self, measurement: &Array1<f64>) {
        self.estimate = self.estimate(measurement);
        // Update covariance (simplified)
        self.covariance = &self.covariance * 0.99 + Array2::<f64>::eye(self.estimate.len()) * 0.01;
    }

    fn reset(&mut self, state: &Array1<f64>) {
        self.estimate = state.clone();
        self.covariance = Array2::eye(state.len());
    }
}

/// Stability analyzer
struct StabilityAnalyzer {
    lyapunov_matrix: Array2<f64>,
    stability_margin: f64,
}

impl StabilityAnalyzer {
    fn new(config: &DynamicsConfig) -> Self {
        Self {
            lyapunov_matrix: config.lyapunov_params.q_matrix.clone(),
            stability_margin: config.stability_margin,
        }
    }

    fn analyze(&self, state: &Array1<f64>, dynamics: &Array1<f64>) -> Result<StabilityInfo> {
        // Lyapunov function value
        let v = 0.5 * state.dot(&self.lyapunov_matrix.dot(state));

        // Lyapunov derivative
        let v_dot = state.dot(&self.lyapunov_matrix.dot(dynamics));

        // Compute eigenvalues (simplified - in practice use LAPACK)
        let largest_eigenvalue = self.estimate_largest_eigenvalue(&self.lyapunov_matrix);

        // Check stability
        let is_stable = v_dot < -self.stability_margin * v;
        let margin = if v > 1e-10 { -v_dot / v } else { f64::INFINITY };

        Ok(StabilityInfo {
            lyapunov_value: v,
            lyapunov_derivative: v_dot,
            is_stable,
            stability_margin: margin,
            largest_eigenvalue,
        })
    }

    fn estimate_largest_eigenvalue(&self, matrix: &Array2<f64>) -> Complex64 {
        // Power iteration (simplified)
        let n = matrix.nrows();
        let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

        for _ in 0..20 {
            v = matrix.dot(&v);
            let norm = v.dot(&v).sqrt();
            if norm > 1e-10 {
                v /= norm;
            }
        }

        let eigenvalue = v.dot(&matrix.dot(&v));
        Complex64::new(eigenvalue, 0.0)
    }
}

/// Phase space analyzer for dynamical systems
pub struct PhaseSpaceAnalyzer {
    dimension: usize,
    trajectory_buffer: Vec<Array1<f64>>,
    max_buffer_size: usize,
}

impl PhaseSpaceAnalyzer {
    pub fn new(dimension: usize, max_buffer_size: usize) -> Self {
        Self {
            dimension,
            trajectory_buffer: Vec::with_capacity(max_buffer_size),
            max_buffer_size,
        }
    }

    /// Add state to trajectory
    pub fn add_state(&mut self, state: &Array1<f64>) {
        self.trajectory_buffer.push(state.clone());
        if self.trajectory_buffer.len() > self.max_buffer_size {
            self.trajectory_buffer.remove(0);
        }
    }

    /// Compute Poincaré section
    pub fn poincare_section(
        &self,
        plane_normal: &Array1<f64>,
        plane_point: &Array1<f64>,
    ) -> Vec<Array1<f64>> {
        let mut section_points = Vec::new();

        for i in 1..self.trajectory_buffer.len() {
            let prev = &self.trajectory_buffer[i - 1];
            let curr = &self.trajectory_buffer[i];

            // Check if trajectory crosses the plane
            let prev_dist = (prev - plane_point).dot(plane_normal);
            let curr_dist = (curr - plane_point).dot(plane_normal);

            if prev_dist * curr_dist < 0.0 {
                // Linear interpolation to find intersection
                let t = prev_dist / (prev_dist - curr_dist);
                let intersection = prev + &((curr - prev) * t);
                section_points.push(intersection);
            }
        }

        section_points
    }

    /// Estimate Lyapunov exponents
    pub fn lyapunov_exponents(&self, dt: f64) -> Vec<f64> {
        if self.trajectory_buffer.len() < 10 {
            return vec![0.0; self.dimension];
        }

        let mut exponents = vec![0.0; self.dimension];
        let mut tangent_vectors = Array2::eye(self.dimension);

        for i in 1..self.trajectory_buffer.len() {
            // Approximate Jacobian using finite differences
            let jacobian = self.approximate_jacobian(i, dt);

            // Evolve tangent vectors
            tangent_vectors = jacobian.dot(&tangent_vectors);

            // QR decomposition for orthonormalization
            let (q, r) = self.qr_decomposition(&tangent_vectors);
            tangent_vectors = q;

            // Update exponents
            for j in 0..self.dimension {
                exponents[j] += r[[j, j]].abs().ln() / (i as f64 * dt);
            }
        }

        exponents
            .iter_mut()
            .for_each(|e| *e /= self.trajectory_buffer.len() as f64);
        exponents
    }

    /// Approximate Jacobian at time index
    fn approximate_jacobian(&self, idx: usize, dt: f64) -> Array2<f64> {
        let n = self.dimension;
        let mut jacobian = Array2::zeros((n, n));

        if idx == 0 || idx >= self.trajectory_buffer.len() - 1 {
            return Array2::eye(n);
        }

        let prev = &self.trajectory_buffer[idx - 1];
        let next = &self.trajectory_buffer[idx + 1];

        // Central difference approximation
        for i in 0..n {
            for j in 0..n {
                let mut prev_perturbed = prev.clone();
                let mut next_perturbed = next.clone();

                let eps = 1e-6;
                prev_perturbed[j] += eps;
                next_perturbed[j] += eps;

                jacobian[[i, j]] =
                    (next_perturbed[i] - next[i] - prev_perturbed[i] + prev[i]) / (2.0 * eps * dt);
            }
        }

        jacobian
    }

    /// Simple QR decomposition
    fn qr_decomposition(&self, a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (m, n) = a.dim();
        let mut q = a.clone();
        let mut r = Array2::zeros((n, n));

        // Gram-Schmidt orthogonalization
        for j in 0..n {
            let mut v = q.column(j).to_owned();

            for i in 0..j {
                let qi = q.column(i);
                let rij = qi.dot(&v);
                r[[i, j]] = rij;
                v = v - &(qi.to_owned() * rij);
            }

            r[[j, j]] = v.dot(&v).sqrt();
            if r[[j, j]] > 1e-10 {
                v /= r[[j, j]];
                q.column_mut(j).assign(&v);
            }
        }

        (q, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamical_system() {
        let config = DynamicsConfig::default();
        let mut system = EnergyDynamicalSystem::new(config).unwrap();

        let reference = Array1::ones(10) * 0.5;
        let external = Array1::zeros(10);

        let state = system.step(&external, &reference).unwrap();
        assert!(state.energy >= 0.0);
    }

    #[test]
    fn test_stability_analysis() {
        let config = DynamicsConfig::default();
        let analyzer = StabilityAnalyzer::new(&config);

        let state = Array1::ones(10) * 0.1;
        let dynamics = Array1::ones(10) * -0.01;

        let stability = analyzer.analyze(&state, &dynamics).unwrap();
        assert!(stability.lyapunov_derivative < 0.0);
    }

    #[test]
    fn test_phase_space_analyzer() {
        let mut analyzer = PhaseSpaceAnalyzer::new(3, 1000);

        // Add trajectory points
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let state = Array1::from_vec(vec![t.cos(), t.sin(), (t * 0.5).cos()]);
            analyzer.add_state(&state);
        }

        // Compute Poincaré section
        let normal = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let point = Array1::zeros(3);
        let section = analyzer.poincare_section(&normal, &point);

        assert!(!section.is_empty());
    }
}
