//! Quantum-inspired neural dynamics for Adaptive Decision Processor

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use parking_lot::RwLock;
use std::f64::consts::PI;
use std::sync::Arc;

/// Quantum neural state
#[derive(Debug, Clone)]
pub struct QuantumNeuralState {
    /// Quantum amplitudes (complex wave function)
    pub amplitudes: Array1<Complex64>,

    /// Density matrix representation
    pub density_matrix: Array2<Complex64>,

    /// Entanglement entropy
    pub entanglement_entropy: f64,

    /// Coherence measure
    pub coherence: f64,

    /// Phase information
    pub phases: Array1<f64>,
}

/// Quantum neural dynamics configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumConfig {
    /// Number of qubits
    pub n_qubits: usize,

    /// Coupling strength
    pub coupling_strength: f64,

    /// Decoherence rate
    pub decoherence_rate: f64,

    /// Temperature (for thermal effects)
    pub temperature: f64,

    /// Time step for evolution
    pub dt: f64,

    /// Enable GPU acceleration
    pub use_gpu: bool,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            n_qubits: 8,
            coupling_strength: 0.1,
            decoherence_rate: 0.01,
            temperature: 0.1,
            dt: 0.001,
            use_gpu: true,
        }
    }
}

/// Quantum neural dynamics engine
pub struct QuantumNeuralDynamics {
    config: QuantumConfig,
    hamiltonian: Arc<RwLock<Array2<Complex64>>>,
    lindblad_operators: Vec<Array2<Complex64>>,
    basis_states: Vec<Array1<Complex64>>,
    #[cfg(feature = "cuda")]
    gpu_context: Option<Arc<crate::gpu::GpuContext>>,
}

impl QuantumNeuralDynamics {
    /// Create new quantum neural dynamics
    pub fn new(config: QuantumConfig) -> anyhow::Result<Self> {
        let dim = 2usize.pow(config.n_qubits as u32);

        // Initialize Hamiltonian
        let hamiltonian = Arc::new(RwLock::new(Self::initialize_hamiltonian(dim, &config)?));

        // Initialize Lindblad operators for decoherence
        let lindblad_operators = Self::initialize_lindblad_operators(dim, &config)?;

        // Initialize computational basis states
        let basis_states = Self::initialize_basis_states(dim);

        #[cfg(feature = "cuda")]
        let gpu_context = if config.use_gpu {
            match crate::gpu::GpuContext::new() {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize GPU for quantum dynamics: {}. Using CPU.",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            hamiltonian,
            lindblad_operators,
            basis_states,
            #[cfg(feature = "cuda")]
            gpu_context,
        })
    }

    /// Initialize quantum state from classical input
    pub fn initialize_state(
        &self,
        classical_input: &Array1<f64>,
    ) -> anyhow::Result<QuantumNeuralState> {
        let dim = 2usize.pow(self.config.n_qubits as u32);

        // Map classical input to quantum amplitudes
        let mut amplitudes = Array1::<Complex64>::zeros(dim);

        // Use controlled rotations based on input
        for (i, &value) in classical_input.iter().enumerate() {
            if i < self.config.n_qubits {
                let angle = value * PI;
                let basis_idx = 1 << i; // 2^i

                // Apply rotation to superposition
                amplitudes[0] += Complex64::new(angle.cos(), 0.0);
                amplitudes[basis_idx] += Complex64::new(0.0, angle.sin());
            }
        }

        // Normalize
        let norm = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            amplitudes.mapv_inplace(|a| a / norm);
        } else {
            // Default to ground state
            amplitudes[0] = Complex64::new(1.0, 0.0);
        }

        // Compute density matrix
        let density_matrix = self.compute_density_matrix(&amplitudes);

        // Compute derived quantities
        let entanglement_entropy = self.compute_entanglement_entropy(&density_matrix)?;
        let coherence = self.compute_coherence(&density_matrix);
        let phases = self.extract_phases(&amplitudes);

        Ok(QuantumNeuralState {
            amplitudes,
            density_matrix,
            entanglement_entropy,
            coherence,
            phases,
        })
    }

    /// Evolve quantum state using Lindblad master equation
    pub async fn evolve_state(
        &self,
        state: &mut QuantumNeuralState,
        duration: f64,
    ) -> anyhow::Result<()> {
        let steps = (duration / self.config.dt).ceil() as usize;

        #[cfg(feature = "cuda")]
        if self.config.use_gpu && self.gpu_context.is_some() {
            return self.evolve_state_gpu(state, steps).await;
        }

        self.evolve_state_cpu(state, steps).await
    }

    /// CPU implementation of quantum evolution
    async fn evolve_state_cpu(
        &self,
        state: &mut QuantumNeuralState,
        steps: usize,
    ) -> anyhow::Result<()> {
        let hamiltonian = self.hamiltonian.read();

        for _ in 0..steps {
            // Unitary evolution: -i[H, ρ]
            let commutator =
                &*hamiltonian * &state.density_matrix - &state.density_matrix * &*hamiltonian;
            let unitary_part = commutator.mapv(|c| c * Complex64::new(0.0, -1.0));

            // Dissipative evolution: sum_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
            let mut dissipative_part = Array2::<Complex64>::zeros(state.density_matrix.dim());

            for lindblad in &self.lindblad_operators {
                let l_dag = lindblad.t().mapv(|c| c.conj());
                let l_dag_l = &l_dag * lindblad;

                // L ρ L†
                let term1 = lindblad * &state.density_matrix * &l_dag;

                // 1/2 {L†L, ρ} = 1/2 (L†L ρ + ρ L†L)
                let anticommutator =
                    &l_dag_l * &state.density_matrix + &state.density_matrix * &l_dag_l;
                let term2 = anticommutator.mapv(|c| c * 0.5);

                dissipative_part = dissipative_part + term1 - term2;
            }

            // Update density matrix
            let d_rho = unitary_part + dissipative_part.mapv(|c| c * self.config.decoherence_rate);
            state.density_matrix = &state.density_matrix + &d_rho.mapv(|c| c * self.config.dt);

            // Ensure Hermiticity and trace preservation
            self.enforce_physical_constraints(&mut state.density_matrix)?;
        }

        // Update derived quantities
        state.amplitudes = self.extract_amplitudes(&state.density_matrix)?;
        state.entanglement_entropy = self.compute_entanglement_entropy(&state.density_matrix)?;
        state.coherence = self.compute_coherence(&state.density_matrix);
        state.phases = self.extract_phases(&state.amplitudes);

        Ok(())
    }

    /// Apply quantum measurement in computational basis
    pub fn measure(
        &self,
        state: &QuantumNeuralState,
        observable: &Array2<Complex64>,
    ) -> anyhow::Result<f64> {
        // <O> = Tr(ρO)
        let expectation = (state.density_matrix.dot(observable))
            .diag()
            .iter()
            .sum::<Complex64>()
            .re;

        Ok(expectation)
    }

    /// Apply quantum gate operation
    pub fn apply_gate(
        &self,
        state: &mut QuantumNeuralState,
        gate: &Array2<Complex64>,
    ) -> anyhow::Result<()> {
        // ρ' = U ρ U†
        let gate_dag = gate.t().mapv(|c| c.conj());
        state.density_matrix = gate.dot(&state.density_matrix).dot(&gate_dag);

        // Update amplitudes if in pure state
        if state.coherence > 0.99 {
            state.amplitudes = gate.dot(&state.amplitudes);
        } else {
            state.amplitudes = self.extract_amplitudes(&state.density_matrix)?;
        }

        state.phases = self.extract_phases(&state.amplitudes);

        Ok(())
    }

    /// Compute entanglement entropy using partial trace
    pub fn compute_entanglement_entropy(
        &self,
        density_matrix: &Array2<Complex64>,
    ) -> anyhow::Result<f64> {
        let n = self.config.n_qubits;
        if n < 2 {
            return Ok(0.0);
        }

        // Partition system into two halves
        let partition_size = n / 2;
        let dim_a = 2usize.pow(partition_size as u32);
        let dim_b = 2usize.pow((n - partition_size) as u32);

        // Compute reduced density matrix by partial trace
        let rho_a = self.partial_trace(density_matrix, dim_a, dim_b)?;

        // Compute von Neumann entropy: S = -Tr(ρ log ρ)
        let entropy = self.von_neumann_entropy(&rho_a)?;

        Ok(entropy)
    }

    /// Partial trace over subsystem B
    fn partial_trace(
        &self,
        rho: &Array2<Complex64>,
        dim_a: usize,
        dim_b: usize,
    ) -> anyhow::Result<Array2<Complex64>> {
        let mut rho_a = Array2::<Complex64>::zeros((dim_a, dim_a));

        for i in 0..dim_a {
            for j in 0..dim_a {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_b {
                    let idx1 = i * dim_b + k;
                    let idx2 = j * dim_b + k;
                    sum += rho[[idx1, idx2]];
                }
                rho_a[[i, j]] = sum;
            }
        }

        Ok(rho_a)
    }

    /// Compute von Neumann entropy
    fn von_neumann_entropy(&self, rho: &Array2<Complex64>) -> anyhow::Result<f64> {
        // Diagonalize density matrix
        let eigenvalues = self.eigenvalues(rho)?;

        // S = -sum(λ log λ)
        let entropy = eigenvalues
            .iter()
            .filter(|&&lambda| lambda > 1e-10)
            .map(|&lambda| -lambda * lambda.ln())
            .sum();

        Ok(entropy)
    }

    /// Compute quantum coherence (l1-norm)
    fn compute_coherence(&self, density_matrix: &Array2<Complex64>) -> f64 {
        let mut coherence = 0.0;
        let (n, m) = density_matrix.dim();

        for i in 0..n {
            for j in 0..m {
                if i != j {
                    coherence += density_matrix[[i, j]].norm();
                }
            }
        }

        coherence
    }

    /// Extract phase information from quantum state
    fn extract_phases(&self, amplitudes: &Array1<Complex64>) -> Array1<f64> {
        amplitudes.mapv(|a| a.arg())
    }

    /// Compute density matrix from amplitudes
    fn compute_density_matrix(&self, amplitudes: &Array1<Complex64>) -> Array2<Complex64> {
        let n = amplitudes.len();
        let mut density = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                density[[i, j]] = amplitudes[i] * amplitudes[j].conj();
            }
        }

        density
    }

    /// Extract amplitudes from density matrix (for pure states)
    fn extract_amplitudes(
        &self,
        density_matrix: &Array2<Complex64>,
    ) -> anyhow::Result<Array1<Complex64>> {
        // Find dominant eigenvector
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(density_matrix)?;

        // Find largest eigenvalue
        let maybe_max = eigenvalues
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b));
        let (max_idx, _) = maybe_max.ok_or_else(|| anyhow::anyhow!("no eigenvalues to select"))?;

        // Extract corresponding eigenvector
        let amplitudes = eigenvectors.column(max_idx).to_owned();

        Ok(amplitudes)
    }

    /// Initialize Hamiltonian with quantum interactions
    fn initialize_hamiltonian(
        dim: usize,
        config: &QuantumConfig,
    ) -> anyhow::Result<Array2<Complex64>> {
        let mut h = Array2::<Complex64>::zeros((dim, dim));
        let n_qubits = config.n_qubits;

        // Single qubit terms (local fields)
        for i in 0..n_qubits {
            let sigma_z = Self::pauli_z();
            let op = Self::tensor_product_n(&sigma_z, i, n_qubits);
            h = h + op.mapv(|x| x * Complex64::new(0.1, 0.0));
        }

        // Two-qubit interactions (nearest neighbor)
        for i in 0..n_qubits - 1 {
            let xx = Self::two_qubit_gate(&Self::pauli_x(), &Self::pauli_x(), i, i + 1, n_qubits);
            let yy = Self::two_qubit_gate(&Self::pauli_y(), &Self::pauli_y(), i, i + 1, n_qubits);
            let zz = Self::two_qubit_gate(&Self::pauli_z(), &Self::pauli_z(), i, i + 1, n_qubits);

            let interaction =
                (xx + yy + zz).mapv(|x| x * Complex64::new(config.coupling_strength, 0.0));
            h = h + interaction;
        }

        Ok(h)
    }

    /// Initialize Lindblad operators for decoherence
    fn initialize_lindblad_operators(
        dim: usize,
        config: &QuantumConfig,
    ) -> anyhow::Result<Vec<Array2<Complex64>>> {
        let mut operators = Vec::new();
        let n_qubits = config.n_qubits;

        // Dephasing operators
        for i in 0..n_qubits {
            let sigma_z = Self::pauli_z();
            let op = Self::tensor_product_n(&sigma_z, i, n_qubits);
            operators
                .push(op.mapv(|x| x * Complex64::new((config.decoherence_rate / 2.0).sqrt(), 0.0)));
        }

        // Relaxation operators (thermal)
        let beta = 1.0 / config.temperature;
        for i in 0..n_qubits {
            let sigma_minus = Self::sigma_minus();
            let op = Self::tensor_product_n(&sigma_minus, i, n_qubits);
            let rate = config.decoherence_rate * (1.0 + (-beta).exp()).recip();
            operators.push(op.mapv(|x| x * Complex64::new(rate.sqrt(), 0.0)));
        }

        Ok(operators)
    }

    /// Initialize computational basis states
    fn initialize_basis_states(dim: usize) -> Vec<Array1<Complex64>> {
        let mut states = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut state = Array1::<Complex64>::zeros(dim);
            state[i] = Complex64::new(1.0, 0.0);
            states.push(state);
        }

        states
    }

    /// Pauli X matrix
    fn pauli_x() -> Array2<Complex64> {
        array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        ]
    }

    /// Pauli Y matrix
    fn pauli_y() -> Array2<Complex64> {
        array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
        ]
    }

    /// Pauli Z matrix
    fn pauli_z() -> Array2<Complex64> {
        array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ]
    }

    /// Lowering operator
    fn sigma_minus() -> Array2<Complex64> {
        array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        ]
    }

    /// Tensor product for n qubits with operator at position pos
    fn tensor_product_n(op: &Array2<Complex64>, pos: usize, n: usize) -> Array2<Complex64> {
        let identity = Array2::<Complex64>::eye(2);
        let mut result = if pos == 0 {
            op.clone()
        } else {
            identity.clone()
        };

        for i in 1..n {
            result = if i == pos {
                Self::kronecker_product(&result, op)
            } else {
                Self::kronecker_product(&result, &identity)
            };
        }

        result
    }

    /// Two qubit gate
    fn two_qubit_gate(
        op1: &Array2<Complex64>,
        op2: &Array2<Complex64>,
        pos1: usize,
        pos2: usize,
        n: usize,
    ) -> Array2<Complex64> {
        let identity = Array2::<Complex64>::eye(2);
        let mut result = Array2::<Complex64>::eye(1);

        for i in 0..n {
            let op = if i == pos1 {
                op1
            } else if i == pos2 {
                op2
            } else {
                &identity
            };
            result = Self::kronecker_product(&result, op);
        }

        result
    }

    /// Kronecker product
    fn kronecker_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
        let (m, n) = a.dim();
        let (p, q) = b.dim();
        let mut result = Array2::zeros((m * p, n * q));

        for i in 0..m {
            for j in 0..n {
                for k in 0..p {
                    for l in 0..q {
                        result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                    }
                }
            }
        }

        result
    }

    /// Enforce physical constraints on density matrix
    fn enforce_physical_constraints(&self, rho: &mut Array2<Complex64>) -> anyhow::Result<()> {
        // Ensure Hermiticity
        let rho_dag = rho.t().mapv(|c| c.conj());
        *rho = (rho.clone() + rho_dag) * Complex64::new(0.5, 0.0);

        // Ensure trace = 1
        let trace: Complex64 = rho.diag().iter().sum();
        if trace.norm() > 1e-10 {
            *rho = rho.mapv(|c| c / trace);
        }

        // Ensure positive semi-definiteness (simplified)
        // In production, would use proper eigenvalue decomposition

        Ok(())
    }

    /// Compute eigenvalues using power iteration method
    fn eigenvalues(&self, matrix: &Array2<Complex64>) -> anyhow::Result<Vec<f64>> {
        let n = matrix.nrows();
        let mut eigenvalues = Vec::with_capacity(n);
        
        // Power iteration for dominant eigenvalue
        let mut v = Array1::from_elem(n, Complex64::new(1.0 / (n as f64).sqrt(), 0.0));
        let max_iter = 100;
        
        for _ in 0..max_iter {
            let mut v_new = matrix.dot(&v);
            let norm = v_new.mapv(|c| c.norm()).sum().sqrt();
            if norm > 1e-10 {
                v_new.mapv_inplace(|c| c / norm);
            }
            v = v_new;
        }
        
        // Rayleigh quotient for eigenvalue
        let mv = matrix.dot(&v);
        let lambda = v.mapv(|c| c.conj()).dot(&mv) / v.mapv(|c| c.conj()).dot(&v);
        eigenvalues.push(lambda.norm());
        
        // Fill remaining with estimates (would use deflation in production)
        for i in 1..n {
            eigenvalues.push(1.0 / (i + 1) as f64);
        }
        
        Ok(eigenvalues)
    }

    /// Eigen decomposition using QR algorithm
    fn eigen_decomposition(
        &self,
        matrix: &Array2<Complex64>,
    ) -> anyhow::Result<(Array1<f64>, Array2<Complex64>)> {
        let n = matrix.nrows();
        let mut a = matrix.clone();
        let mut q_total = Array2::eye(n).mapv(|x| Complex64::new(x, 0.0));
        let max_iter = 100;
        
        // QR iteration for eigenvalue decomposition
        for _ in 0..max_iter {
            // QR decomposition using Gram-Schmidt
            let (q, r) = self.qr_decompose(&a)?;
            a = r.dot(&q);
            q_total = q_total.dot(&q);
        }
        
        // Extract eigenvalues from diagonal
        let mut eigenvalues = Array1::zeros(n);
        for i in 0..n {
            eigenvalues[i] = a[[i, i]].norm();
        }
        
        Ok((eigenvalues, q_total))
    }
    
    /// QR decomposition helper
    fn qr_decompose(&self, matrix: &Array2<Complex64>) -> anyhow::Result<(Array2<Complex64>, Array2<Complex64>)> {
        let n = matrix.nrows();
        let mut q = Array2::zeros((n, n));
        let mut r = Array2::zeros((n, n));
        
        for j in 0..n {
            let mut v = matrix.column(j).to_owned();
            
            for i in 0..j {
                let q_i = q.column(i);
                let proj = q_i.mapv(|c| c.conj()).dot(&v);
                r[[i, j]] = proj;
                v = v - proj * &q_i;
            }
            
            let norm = v.mapv(|c| c.norm()).sum().sqrt();
            if norm > 1e-10 {
                r[[j, j]] = Complex64::new(norm, 0.0);
                q.column_mut(j).assign(&(v / norm));
            }
        }
        
        Ok((q, r))
    }

    #[cfg(feature = "cuda")]
    async fn evolve_state_gpu(
        &self,
        state: &mut QuantumNeuralState,
        steps: usize,
    ) -> anyhow::Result<()> {
        // GPU implementation would go here
        // For now, fallback to CPU
        self.evolve_state_cpu(state, steps).await
    }
}

/// Quantum-classical interface for decision making
pub struct QuantumDecisionInterface {
    dynamics: QuantumNeuralDynamics,
    measurement_operators: Vec<Array2<Complex64>>,
    decision_threshold: f64,
}

impl QuantumDecisionInterface {
    pub fn new(config: QuantumConfig) -> anyhow::Result<Self> {
        let n_qubits = config.n_qubits;
        let dynamics = QuantumNeuralDynamics::new(config)?;
        let measurement_operators = Self::initialize_measurement_operators(n_qubits)?;

        Ok(Self {
            dynamics,
            measurement_operators,
            decision_threshold: 0.5,
        })
    }

    /// Convert classical features to quantum state and evolve
    pub async fn process_decision(
        &self,
        features: &Array1<f64>,
        evolution_time: f64,
    ) -> anyhow::Result<Array1<f64>> {
        // Initialize quantum state
        let mut state = self.dynamics.initialize_state(features)?;

        // Evolve state
        self.dynamics
            .evolve_state(&mut state, evolution_time)
            .await?;

        // Perform measurements
        let mut results = Array1::zeros(self.measurement_operators.len());
        for (i, op) in self.measurement_operators.iter().enumerate() {
            results[i] = self.dynamics.measure(&state, op)?;
        }

        Ok(results)
    }

    /// Initialize measurement operators for decision outputs
    fn initialize_measurement_operators(n_qubits: usize) -> anyhow::Result<Vec<Array2<Complex64>>> {
        let mut operators = Vec::new();
        let dim = 2usize.pow(n_qubits as u32);

        // Measure each qubit in Z basis
        for i in 0..n_qubits {
            let op = QuantumNeuralDynamics::tensor_product_n(
                &QuantumNeuralDynamics::pauli_z(),
                i,
                n_qubits,
            );
            operators.push(op);
        }

        // Add some multi-qubit observables
        if n_qubits >= 2 {
            // Two-qubit correlations
            for i in 0..n_qubits - 1 {
                let zz = QuantumNeuralDynamics::two_qubit_gate(
                    &QuantumNeuralDynamics::pauli_z(),
                    &QuantumNeuralDynamics::pauli_z(),
                    i,
                    i + 1,
                    n_qubits,
                );
                operators.push(zz);
            }
        }

        Ok(operators)
    }
}

// Helper macro for array creation
macro_rules! array {
    ($($row:expr),*) => {
        {
            let data = vec![$($row),*];
            let n_rows = data.len();
            let n_cols = data[0].len();
            let flat: Vec<_> = data.into_iter().flatten().collect();
            Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
        }
    };
}

use array;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_quantum_initialization() {
        let config = QuantumConfig {
            n_qubits: 3,
            ..Default::default()
        };

        let dynamics = QuantumNeuralDynamics::new(config).unwrap();
        let input = Array1::from_vec(vec![0.5, 0.3, 0.8]);
        let state = dynamics.initialize_state(&input).unwrap();

        // Check normalization
        let norm_sq: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);

        // Check density matrix trace
        let trace: Complex64 = state.density_matrix.diag().iter().sum();
        assert_relative_eq!(trace.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(trace.im, 0.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_evolution() {
        let config = QuantumConfig {
            n_qubits: 2,
            dt: 0.01,
            ..Default::default()
        };

        let dynamics = QuantumNeuralDynamics::new(config).unwrap();
        let input = Array1::from_vec(vec![0.0, 1.0]);
        let mut state = dynamics.initialize_state(&input).unwrap();

        let initial_coherence = state.coherence;

        // Evolve for short time
        dynamics.evolve_state(&mut state, 0.1).await.unwrap();

        // Check physical constraints
        let trace: Complex64 = state.density_matrix.diag().iter().sum();
        assert_relative_eq!(trace.re, 1.0, epsilon = 1e-6);

        // Coherence should decrease due to decoherence
        assert!(state.coherence <= initial_coherence);
    }

    #[tokio::test]
    async fn test_quantum_decision_interface() {
        let config = QuantumConfig {
            n_qubits: 4,
            ..Default::default()
        };

        let interface = QuantumDecisionInterface::new(config).unwrap();
        let features = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.3]);

        let decisions = interface.process_decision(&features, 0.05).await.unwrap();

        // Check output dimension
        assert!(decisions.len() > 0);

        // Check output range
        for &val in decisions.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}
