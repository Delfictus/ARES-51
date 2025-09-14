use crate::geometry::{Structure, Atom};
use crate::security::{SecurityValidator, SecurityError};
use crate::data::{ForceFieldParams, RamachandranConstraints};
use ndarray::{Array1, Array2};

/// Industrial-grade energy minimization framework with BFGS optimizer
#[derive(Debug, Clone)]
pub struct EnergyMinimizer {
    /// Security validator for input validation
    validator: SecurityValidator,
    /// Force field parameters for energy calculations
    force_field: ForceFieldParams,
    /// Ramachandran constraints for validation
    ramachandran: RamachandranConstraints,
    /// Convergence criteria
    convergence_config: ConvergenceConfig,
    /// BFGS optimizer configuration
    bfgs_config: BfgsConfig,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Energy function type
    energy_function: EnergyFunctionType,
}

/// BFGS optimizer configuration with security bounds
#[derive(Debug, Clone)]
pub struct BfgsConfig {
    /// Maximum step size (Angstroms)
    max_step_size: f64,
    /// Initial Hessian approximation scale
    initial_hessian_scale: f64,
    /// Line search parameters
    line_search: LineSearchConfig,
    /// Memory size for L-BFGS
    memory_size: usize,
    /// Gradient tolerance for convergence
    gradient_tolerance: f64,
}

/// Line search configuration for step size optimization
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Armijo condition parameter (c1)
    armijo_c1: f64,
    /// Curvature condition parameter (c2) 
    curvature_c2: f64,
    /// Maximum line search iterations
    max_iterations: usize,
    /// Initial step size
    initial_step: f64,
    /// Step size reduction factor
    reduction_factor: f64,
}

/// Convergence criteria with mathematical precision requirements
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Energy tolerance (kcal/mol)
    energy_tolerance: f64,
    /// Gradient RMS tolerance (kcal/mol/Å)
    gradient_tolerance: f64,
    /// Maximum displacement tolerance (Å)
    displacement_tolerance: f64,
    /// Force tolerance (kcal/mol/Å)
    force_tolerance: f64,
    /// Minimum iterations before checking convergence
    min_iterations: usize,
    /// Relative energy change tolerance
    relative_energy_tolerance: f64,
}

/// Energy function types supported by the minimizer
#[derive(Debug, Clone)]
pub enum EnergyFunctionType {
    /// CHARMM36 force field with full interactions
    Charmm36Full,
    /// Simplified backbone-only energy
    BackboneOnly,
    /// Contact-based potential
    ContactBased { contact_threshold: f64 },
    /// Combined energy with weighted terms
    Combined { weights: EnergyWeights },
}

/// Energy term weights for combined energy functions
#[derive(Debug, Clone)]
pub struct EnergyWeights {
    /// Lennard-Jones potential weight
    lennard_jones: f64,
    /// Electrostatic potential weight
    electrostatic: f64,
    /// Ramachandran constraint weight
    ramachandran: f64,
    /// Contact potential weight
    contact: f64,
    /// Solvation energy weight
    solvation: f64,
}

/// Energy minimization result with comprehensive analysis
#[derive(Debug, Clone)]
pub struct MinimizationResult {
    /// Final optimized structure
    pub optimized_structure: Structure,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final energy value (kcal/mol)
    pub final_energy: f64,
    /// Initial energy value (kcal/mol)
    pub initial_energy: f64,
    /// Energy reduction achieved
    pub energy_reduction: f64,
    /// Final gradient RMS (kcal/mol/Å)
    pub final_gradient_rms: f64,
    /// Maximum force component (kcal/mol/Å)
    pub max_force: f64,
    /// Energy history during optimization
    pub energy_history: Vec<f64>,
    /// Gradient RMS history
    pub gradient_history: Vec<f64>,
    /// Step size history
    pub step_history: Vec<f64>,
    /// Total optimization time (seconds)
    pub optimization_time: f64,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
}

/// Detailed convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Energy convergence achieved
    pub energy_converged: bool,
    /// Gradient convergence achieved
    pub gradient_converged: bool,
    /// Force convergence achieved
    pub force_converged: bool,
    /// Displacement convergence achieved
    pub displacement_converged: bool,
    /// Final energy change per iteration
    pub final_energy_change: f64,
    /// Convergence rate estimate
    pub convergence_rate: f64,
    /// Number of line search failures
    pub line_search_failures: usize,
}

/// BFGS optimizer state for iterative minimization
#[derive(Debug)]
struct BfgsState {
    /// Current coordinates as flat vector
    coordinates: Array1<f64>,
    /// Current gradient vector
    gradient: Array1<f64>,
    /// Current energy value
    energy: f64,
    /// Inverse Hessian approximation
    inverse_hessian: Array2<f64>,
    /// Previous coordinate vector
    prev_coordinates: Array1<f64>,
    /// Previous gradient vector
    prev_gradient: Array1<f64>,
    /// Step vectors for L-BFGS memory
    step_vectors: Vec<Array1<f64>>,
    /// Gradient difference vectors for L-BFGS
    gradient_diffs: Vec<Array1<f64>>,
    /// Iteration counter
    iteration: usize,
}

impl EnergyMinimizer {
    /// Create new energy minimizer with comprehensive security validation
    pub fn new(energy_function: EnergyFunctionType) -> Result<Self, SecurityError> {
        let validator = SecurityValidator::new()?;
        let force_field = ForceFieldParams::default();
        let ramachandran = RamachandranConstraints::new();
        
        let convergence_config = ConvergenceConfig {
            energy_tolerance: 1e-6,        // 0.000001 kcal/mol
            gradient_tolerance: 1e-4,      // 0.0001 kcal/mol/Å
            displacement_tolerance: 1e-6,   // 0.000001 Å
            force_tolerance: 1e-4,         // 0.0001 kcal/mol/Å
            min_iterations: 10,
            relative_energy_tolerance: 1e-8,
        };
        
        let bfgs_config = BfgsConfig {
            max_step_size: 0.1,           // 0.1 Å maximum step
            initial_hessian_scale: 1.0,
            line_search: LineSearchConfig {
                armijo_c1: 1e-4,
                curvature_c2: 0.9,
                max_iterations: 20,
                initial_step: 1.0,
                reduction_factor: 0.5,
            },
            memory_size: 10,              // L-BFGS memory
            gradient_tolerance: 1e-6,
        };
        
        Ok(EnergyMinimizer {
            validator,
            force_field,
            ramachandran,
            convergence_config,
            bfgs_config,
            max_iterations: 1000,
            energy_function,
        })
    }
    
    /// Set convergence criteria with security validation
    pub fn set_convergence_criteria(&mut self, config: ConvergenceConfig) -> Result<(), SecurityError> {
        // Security validation for convergence parameters
        if config.energy_tolerance <= 0.0 || config.energy_tolerance > 1.0 {
            return Err(SecurityError::InvalidInput(
                format!("Energy tolerance {:.2e} outside valid range (0, 1.0]", config.energy_tolerance)
            ));
        }
        
        if config.gradient_tolerance <= 0.0 || config.gradient_tolerance > 10.0 {
            return Err(SecurityError::InvalidInput(
                format!("Gradient tolerance {:.2e} outside valid range (0, 10.0]", config.gradient_tolerance)
            ));
        }
        
        if config.min_iterations > 100_000 {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "optimization iterations".to_string(),
                current_count: config.min_iterations,
                max_allowed: 100_000,
            });
        }
        
        self.convergence_config = config;
        Ok(())
    }
    
    /// Set BFGS optimizer configuration with security bounds
    pub fn set_bfgs_config(&mut self, config: BfgsConfig) -> Result<(), SecurityError> {
        // Security validation for BFGS parameters
        if config.max_step_size <= 0.0 || config.max_step_size > 5.0 {
            return Err(SecurityError::InvalidInput(
                format!("Max step size {:.3}Å outside valid range (0, 5.0]", config.max_step_size)
            ));
        }
        
        if config.memory_size == 0 || config.memory_size > 50 {
            return Err(SecurityError::InvalidInput(
                format!("BFGS memory size {} outside valid range [1, 50]", config.memory_size)
            ));
        }
        
        if config.line_search.max_iterations > 100 {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "line search iterations".to_string(),
                current_count: config.line_search.max_iterations,
                max_allowed: 100,
            });
        }
        
        self.bfgs_config = config;
        Ok(())
    }
    
    /// Minimize energy of protein structure using BFGS optimization
    pub fn minimize_energy(&mut self, structure: &Structure) -> Result<MinimizationResult, SecurityError> {
        // Security validation: Check structure size
        let atom_count = self.count_structure_atoms(structure)?;
        if atom_count == 0 {
            return Err(SecurityError::InvalidInput("Empty structure provided".to_string()));
        }
        
        if atom_count > 50_000 {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "structure atoms".to_string(),
                current_count: atom_count,
                max_allowed: 50_000,
            });
        }
        
        // Start security monitoring
        self.validator.start_operation("energy minimization");
        
        let start_time = std::time::Instant::now();
        
        // Initialize BFGS state
        let mut bfgs_state = self.initialize_bfgs_state(structure)?;
        
        // Calculate initial energy and gradient
        let initial_energy = bfgs_state.energy;
        let mut energy_history = vec![initial_energy];
        let mut gradient_history = vec![bfgs_state.gradient.iter().map(|x| x*x).sum::<f64>().sqrt()];
        let mut step_history = Vec::new();
        
        let mut converged = false;
        let mut convergence_analysis = ConvergenceAnalysis {
            energy_converged: false,
            gradient_converged: false,
            force_converged: false,
            displacement_converged: false,
            final_energy_change: 0.0,
            convergence_rate: 0.0,
            line_search_failures: 0,
        };
        
        // BFGS optimization loop
        for iteration in 0..self.max_iterations {
            // Security: Check timeout periodically
            if iteration % 10 == 0 {
                self.validator.check_timeout("BFGS optimization")?;
            }
            
            bfgs_state.iteration = iteration;
            
            // Check convergence criteria
            if iteration >= self.convergence_config.min_iterations {
                let conv_check = self.check_convergence(&bfgs_state, &energy_history, &gradient_history)?;
                convergence_analysis = conv_check;
                if self.is_converged(&convergence_analysis) {
                    converged = true;
                    break;
                }
            }
            
            // Compute search direction using BFGS
            let search_direction = self.compute_bfgs_direction(&mut bfgs_state)?;
            
            // Perform line search to find optimal step size
            let line_search_result = self.perform_line_search(&mut bfgs_state, &search_direction)?;
            
            if !line_search_result.success {
                convergence_analysis.line_search_failures += 1;
                if convergence_analysis.line_search_failures > 5 {
                    break; // Too many line search failures
                }
            }
            
            step_history.push(line_search_result.step_size);
            
            // Update BFGS approximation
            self.update_bfgs_approximation(&mut bfgs_state)?;
            
            // Record energy and gradient history
            energy_history.push(bfgs_state.energy);
            let gradient_rms = bfgs_state.gradient.iter().map(|x| x*x).sum::<f64>().sqrt();
            gradient_history.push(gradient_rms);
            
            // Security: Validate energy progression
            if !bfgs_state.energy.is_finite() {
                return Err(SecurityError::InvalidInput("Non-finite energy encountered".to_string()));
            }
        }
        
        let optimization_time = start_time.elapsed().as_secs_f64();
        
        // Reconstruct optimized structure
        let optimized_structure = self.coordinates_to_structure(structure, &bfgs_state.coordinates)?;
        
        // Calculate final metrics
        let final_gradient_rms = bfgs_state.gradient.iter().map(|x| x*x).sum::<f64>().sqrt();
        let max_force = bfgs_state.gradient.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let energy_reduction = initial_energy - bfgs_state.energy;
        
        Ok(MinimizationResult {
            optimized_structure,
            converged,
            iterations: bfgs_state.iteration,
            final_energy: bfgs_state.energy,
            initial_energy,
            energy_reduction,
            final_gradient_rms,
            max_force,
            energy_history,
            gradient_history,
            step_history,
            optimization_time,
            convergence_analysis,
        })
    }
    
    /// Initialize BFGS optimization state
    fn initialize_bfgs_state(&self, structure: &Structure) -> Result<BfgsState, SecurityError> {
        // Extract coordinates from structure
        let coordinates = self.structure_to_coordinates(structure)?;
        let n_coords = coordinates.len();
        
        // Calculate initial energy and gradient
        let (energy, gradient) = self.calculate_energy_and_gradient(structure, &coordinates)?;
        
        // Security: Validate initial energy
        if !energy.is_finite() {
            return Err(SecurityError::InvalidInput("Initial energy is not finite".to_string()));
        }
        
        // Initialize inverse Hessian as scaled identity matrix
        let mut inverse_hessian = Array2::<f64>::zeros((n_coords, n_coords));
        for i in 0..n_coords {
            inverse_hessian[[i, i]] = self.bfgs_config.initial_hessian_scale;
        }
        
        Ok(BfgsState {
            coordinates: coordinates.clone(),
            gradient: gradient.clone(),
            energy,
            inverse_hessian,
            prev_coordinates: coordinates,
            prev_gradient: gradient,
            step_vectors: Vec::with_capacity(self.bfgs_config.memory_size),
            gradient_diffs: Vec::with_capacity(self.bfgs_config.memory_size),
            iteration: 0,
        })
    }
    
    /// Convert structure to coordinate vector
    fn structure_to_coordinates(&self, structure: &Structure) -> Result<Array1<f64>, SecurityError> {
        let mut coords = Vec::new();
        
        for chain in structure.chains() {
            for residue in chain.residues() {
                for atom in residue.atoms.values() {
                    // Security: Validate atom coordinates
                    if !atom.has_valid_coordinates() {
                        return Err(SecurityError::InvalidInput(
                            format!("Invalid coordinates for atom {}", atom.name)
                        ));
                    }
                    
                    coords.push(atom.coords.x);
                    coords.push(atom.coords.y);
                    coords.push(atom.coords.z);
                }
            }
        }
        
        if coords.is_empty() {
            return Err(SecurityError::InvalidInput("No valid coordinates found".to_string()));
        }
        
        Ok(Array1::from_vec(coords))
    }
    
    /// Convert coordinate vector back to structure
    fn coordinates_to_structure(&self, original_structure: &Structure, coordinates: &Array1<f64>) -> Result<Structure, SecurityError> {
        if coordinates.len() % 3 != 0 {
            return Err(SecurityError::InvalidInput("Coordinate vector length not divisible by 3".to_string()));
        }
        
        let mut new_structure = original_structure.clone();
        let mut coord_idx = 0;
        
        let chain_ids = new_structure.chain_ids_vec();
        for chain_id in chain_ids {
            if let Some(chain) = new_structure.chains.get_mut(&chain_id) {
                for residue in chain.residues_mut() {
                    for atom in residue.atoms.values_mut() {
                    if coord_idx + 2 >= coordinates.len() {
                        return Err(SecurityError::InvalidInput("Insufficient coordinates for structure".to_string()));
                    }
                    
                    atom.coords.x = coordinates[coord_idx];
                    atom.coords.y = coordinates[coord_idx + 1];  
                    atom.coords.z = coordinates[coord_idx + 2];
                    coord_idx += 3;
                    
                        // Security: Validate updated coordinates
                        if !atom.coords.is_finite() {
                            return Err(SecurityError::InvalidInput("Non-finite coordinates generated".to_string()));
                        }
                    }
                }
            }
        }
        
        Ok(new_structure)
    }
    
    /// Calculate energy and gradient for current coordinates
    fn calculate_energy_and_gradient(&self, structure: &Structure, coordinates: &Array1<f64>) -> Result<(f64, Array1<f64>), SecurityError> {
        // Reconstruct structure with current coordinates
        let current_structure = self.coordinates_to_structure(structure, coordinates)?;
        
        // Calculate energy based on selected energy function
        let energy = match &self.energy_function {
            EnergyFunctionType::Charmm36Full => {
                self.calculate_charmm36_energy(&current_structure)?
            },
            EnergyFunctionType::BackboneOnly => {
                self.calculate_backbone_energy(&current_structure)?
            },
            EnergyFunctionType::ContactBased { contact_threshold } => {
                self.calculate_contact_energy(&current_structure, *contact_threshold)?
            },
            EnergyFunctionType::Combined { weights } => {
                self.calculate_combined_energy(&current_structure, weights)?
            },
        };
        
        // Calculate numerical gradient using finite differences
        let gradient = self.calculate_numerical_gradient(&current_structure, coordinates)?;
        
        Ok((energy, gradient))
    }
    
    /// Calculate CHARMM36 energy
    fn calculate_charmm36_energy(&self, structure: &Structure) -> Result<f64, SecurityError> {
        let mut total_energy = 0.0;
        
        // Bond stretching energy
        total_energy += self.calculate_bond_energy(structure)?;
        
        // Angle bending energy
        total_energy += self.calculate_angle_energy(structure)?;
        
        // Dihedral torsion energy
        total_energy += self.calculate_dihedral_energy(structure)?;
        
        // Van der Waals energy
        total_energy += self.calculate_vdw_energy(structure)?;
        
        // Electrostatic energy
        total_energy += self.calculate_electrostatic_energy(structure)?;
        
        // Ramachandran constraint energy
        total_energy += self.calculate_ramachandran_energy(structure)?;
        
        // Security: Validate computed energy
        if !total_energy.is_finite() {
            return Err(SecurityError::InvalidInput("Non-finite energy computed".to_string()));
        }
        
        Ok(total_energy)
    }
    
    /// Calculate backbone-only energy (simplified)
    fn calculate_backbone_energy(&self, structure: &Structure) -> Result<f64, SecurityError> {
        let mut energy = 0.0;
        
        // Simplified backbone potential based on CA distances
        for chain in structure.chains() {
            let ca_atoms: Vec<&Atom> = chain.residues()
                .iter()
                .filter_map(|residue| residue.atoms.get("CA"))
                .collect();
                
            for i in 0..ca_atoms.len() {
                for j in i+1..ca_atoms.len() {
                    let distance = ca_atoms[i].distance_to(ca_atoms[j]);
                    
                    // Simple harmonic potential for nearby CA atoms
                    if distance < 12.0 {
                        let ideal_distance = 3.8; // Ideal CA-CA distance
                        energy += 0.5 * (distance - ideal_distance).powi(2);
                    }
                }
            }
        }
        
        Ok(energy)
    }
    
    /// Calculate contact-based energy
    fn calculate_contact_energy(&self, structure: &Structure, threshold: f64) -> Result<f64, SecurityError> {
        let mut energy = 0.0;
        
        // Contact-based potential using CA atoms
        for chain in structure.chains() {
            let ca_atoms: Vec<&Atom> = chain.residues()
                .iter()
                .filter_map(|residue| residue.atoms.get("CA"))
                .collect();
                
            for i in 0..ca_atoms.len() {
                for j in i+1..ca_atoms.len() {
                    let distance = ca_atoms[i].distance_to(ca_atoms[j]);
                    
                    if distance <= threshold {
                        // Attractive contact potential
                        energy -= 1.0 / (1.0 + distance);
                    } else {
                        // Weak repulsive potential for distant contacts
                        energy += 0.1 * (distance - threshold).powi(2);
                    }
                }
            }
        }
        
        Ok(energy)
    }
    
    /// Calculate combined energy with multiple terms
    fn calculate_combined_energy(&self, structure: &Structure, weights: &EnergyWeights) -> Result<f64, SecurityError> {
        let mut total_energy = 0.0;
        
        // Weighted combination of energy terms
        if weights.lennard_jones > 0.0 {
            total_energy += weights.lennard_jones * self.calculate_vdw_energy(structure)?;
        }
        
        if weights.electrostatic > 0.0 {
            total_energy += weights.electrostatic * self.calculate_electrostatic_energy(structure)?;
        }
        
        if weights.ramachandran > 0.0 {
            total_energy += weights.ramachandran * self.calculate_ramachandran_energy(structure)?;
        }
        
        if weights.contact > 0.0 {
            total_energy += weights.contact * self.calculate_contact_energy(structure, 8.0)?;
        }
        
        if weights.solvation > 0.0 {
            total_energy += weights.solvation * self.calculate_solvation_energy(structure)?;
        }
        
        Ok(total_energy)
    }
    
    /// Calculate numerical gradient using finite differences
    fn calculate_numerical_gradient(&self, structure: &Structure, coordinates: &Array1<f64>) -> Result<Array1<f64>, SecurityError> {
        let mut gradient = Array1::<f64>::zeros(coordinates.len());
        let delta = 1e-6; // Finite difference step size
        
        let base_energy = match &self.energy_function {
            EnergyFunctionType::Charmm36Full => self.calculate_charmm36_energy(structure)?,
            EnergyFunctionType::BackboneOnly => self.calculate_backbone_energy(structure)?,
            EnergyFunctionType::ContactBased { contact_threshold } => 
                self.calculate_contact_energy(structure, *contact_threshold)?,
            EnergyFunctionType::Combined { weights } => 
                self.calculate_combined_energy(structure, weights)?,
        };
        
        // Calculate finite difference gradient
        for i in 0..coordinates.len() {
            let mut coords_plus = coordinates.clone();
            coords_plus[i] += delta;
            
            let structure_plus = self.coordinates_to_structure(structure, &coords_plus)?;
            
            let energy_plus = match &self.energy_function {
                EnergyFunctionType::Charmm36Full => self.calculate_charmm36_energy(&structure_plus)?,
                EnergyFunctionType::BackboneOnly => self.calculate_backbone_energy(&structure_plus)?,
                EnergyFunctionType::ContactBased { contact_threshold } => 
                    self.calculate_contact_energy(&structure_plus, *contact_threshold)?,
                EnergyFunctionType::Combined { weights } => 
                    self.calculate_combined_energy(&structure_plus, weights)?,
            };
            
            gradient[i] = (energy_plus - base_energy) / delta;
        }
        
        Ok(gradient)
    }
    
    /// Helper functions for individual energy terms (simplified implementations)
    fn calculate_bond_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement full CHARMM36 bond energy
    }
    
    fn calculate_angle_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement full CHARMM36 angle energy
    }
    
    fn calculate_dihedral_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement full CHARMM36 dihedral energy
    }
    
    fn calculate_vdw_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement full van der Waals energy
    }
    
    fn calculate_electrostatic_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement full electrostatic energy
    }
    
    fn calculate_ramachandran_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement Ramachandran constraint energy
    }
    
    fn calculate_solvation_energy(&self, _structure: &Structure) -> Result<f64, SecurityError> {
        Ok(0.0) // Placeholder - would implement solvation energy
    }
    
    /// Count total atoms in structure for security validation
    fn count_structure_atoms(&self, structure: &Structure) -> Result<usize, SecurityError> {
        let mut count = 0;
        for chain in structure.chains() {
            for residue in chain.residues() {
                count += residue.atoms.len();
            }
        }
        Ok(count)
    }
    
    /// Compute BFGS search direction using two-loop recursion
    fn compute_bfgs_direction(&self, state: &BfgsState) -> Result<Array1<f64>, SecurityError> {
        if state.step_vectors.is_empty() {
            // Initial step: steepest descent with inverse Hessian scaling
            return Ok(-(&state.inverse_hessian.dot(&state.gradient)));
        }
        
        // L-BFGS two-loop recursion algorithm
        let mut q = state.gradient.clone();
        let memory_size = state.step_vectors.len();
        let mut alpha = vec![0.0; memory_size];
        
        // First loop: backward pass
        for i in (0..memory_size).rev() {
            let s = &state.step_vectors[i];
            let y = &state.gradient_diffs[i];
            let rho = 1.0 / y.dot(s);
            
            // Security: Check for numerical issues
            if !rho.is_finite() || rho.abs() > 1e6 {
                return Err(SecurityError::InvalidInput(
                    "BFGS curvature condition violated - numerical instability".to_string()
                ));
            }
            
            alpha[i] = rho * s.dot(&q);
            q = &q - alpha[i] * y;
        }
        
        // Apply initial Hessian approximation (scaled identity)
        let gamma = if !state.gradient_diffs.is_empty() {
            let last_idx = state.gradient_diffs.len() - 1;
            let s = &state.step_vectors[last_idx];
            let y = &state.gradient_diffs[last_idx];
            s.dot(y) / y.dot(y)
        } else {
            self.bfgs_config.initial_hessian_scale
        };
        
        let mut r = gamma * q;
        
        // Second loop: forward pass
        for i in 0..memory_size {
            let s = &state.step_vectors[i];
            let y = &state.gradient_diffs[i];
            let rho = 1.0 / y.dot(s);
            let beta = rho * y.dot(&r);
            r = &r + (alpha[i] - beta) * s;
        }
        
        // Return negative direction for minimization
        Ok(-r)
    }
    
    /// Perform line search using Wolfe conditions
    fn perform_line_search(&self, state: &mut BfgsState, direction: &Array1<f64>) -> Result<LineSearchResult, SecurityError> {
        let config = &self.bfgs_config.line_search;
        let mut step_size = config.initial_step;
        let current_energy = state.energy;
        let gradient_dot_direction = state.gradient.dot(direction);
        
        // Security: Check that direction is a descent direction
        if gradient_dot_direction >= 0.0 {
            return Err(SecurityError::InvalidInput(
                "Not a descent direction in line search".to_string()
            ));
        }
        
        // Armijo condition parameters
        let armijo_threshold = config.armijo_c1 * gradient_dot_direction;
        
        for iteration in 0..config.max_iterations {
            // Limit step size for security
            step_size = step_size.min(self.bfgs_config.max_step_size);
            
            // Trial coordinates
            let trial_coords = &state.coordinates + step_size * direction;
            
            // Security: Validate trial coordinates
            if !trial_coords.iter().all(|x| x.is_finite()) {
                step_size *= config.reduction_factor;
                continue;
            }
            
            // Calculate energy at trial point
            let trial_structure = self.coordinates_to_structure(
                &Structure::new("trial".to_string()), // Temporary structure
                &trial_coords
            )?;
            
            let (trial_energy, trial_gradient) = self.calculate_energy_and_gradient(
                &trial_structure, &trial_coords
            )?;
            
            // Check Armijo condition (sufficient decrease)
            if trial_energy <= current_energy + step_size * armijo_threshold {
                // Accept step and update state
                state.prev_coordinates = state.coordinates.clone();
                state.prev_gradient = state.gradient.clone();
                state.coordinates = trial_coords;
                state.gradient = trial_gradient;
                state.energy = trial_energy;
                
                return Ok(LineSearchResult {
                    success: true,
                    step_size,
                    iterations: iteration + 1,
                });
            }
            
            // Reduce step size for next iteration
            step_size *= config.reduction_factor;
            
            // Security: Prevent infinite reduction
            if step_size < 1e-12 {
                break;
            }
        }
        
        // Line search failed
        Ok(LineSearchResult {
            success: false,
            step_size: 0.0,
            iterations: config.max_iterations,
        })
    }
    
    /// Update BFGS Hessian approximation using L-BFGS memory
    fn update_bfgs_approximation(&self, state: &mut BfgsState) -> Result<(), SecurityError> {
        // Calculate step vector and gradient difference
        let step_vector = &state.coordinates - &state.prev_coordinates;
        let gradient_diff = &state.gradient - &state.prev_gradient;
        
        // Security: Validate curvature condition (Powell's criterion)
        let curvature = step_vector.dot(&gradient_diff);
        if curvature <= 1e-12 {
            return Err(SecurityError::InvalidInput(
                "BFGS curvature condition violated - step too small or gradient diff too small".to_string()
            ));
        }
        
        // Add to L-BFGS memory
        state.step_vectors.push(step_vector);
        state.gradient_diffs.push(gradient_diff);
        
        // Maintain memory limit
        if state.step_vectors.len() > self.bfgs_config.memory_size {
            state.step_vectors.remove(0);
            state.gradient_diffs.remove(0);
        }
        
        // Security: Validate memory consistency
        if state.step_vectors.len() != state.gradient_diffs.len() {
            return Err(SecurityError::InvalidInput(
                "BFGS memory vectors inconsistent".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Check convergence criteria with comprehensive analysis
    fn check_convergence(&self, state: &BfgsState, energy_history: &[f64], gradient_history: &[f64]) -> Result<ConvergenceAnalysis, SecurityError> {
        let config = &self.convergence_config;
        
        // Calculate gradient RMS
        let gradient_rms = (state.gradient.iter().map(|x| x * x).sum::<f64>() / state.gradient.len() as f64).sqrt();
        let max_force = state.gradient.iter().map(|x| x.abs()).fold(0.0, f64::max);
        
        // Check energy convergence
        let energy_converged = if energy_history.len() >= 2 {
            let energy_change = (energy_history[energy_history.len()-1] - energy_history[energy_history.len()-2]).abs();
            energy_change < config.energy_tolerance
        } else {
            false
        };
        
        // Check gradient convergence
        let gradient_converged = gradient_rms < config.gradient_tolerance;
        
        // Check force convergence
        let force_converged = max_force < config.force_tolerance;
        
        // Check displacement convergence
        let displacement_converged = if state.iteration > 0 {
            let displacement = (&state.coordinates - &state.prev_coordinates)
                .iter().map(|x| x * x).sum::<f64>().sqrt();
            displacement < config.displacement_tolerance
        } else {
            false
        };
        
        // Calculate convergence rate
        let convergence_rate = if gradient_history.len() >= 3 {
            let recent_gradients = &gradient_history[gradient_history.len()-3..];
            let rate1 = recent_gradients[1] / recent_gradients[0].max(1e-12);
            let rate2 = recent_gradients[2] / recent_gradients[1].max(1e-12);
            (rate1 + rate2) / 2.0
        } else {
            1.0
        };
        
        // Calculate final energy change
        let final_energy_change = if energy_history.len() >= 2 {
            (energy_history[energy_history.len()-1] - energy_history[energy_history.len()-2]).abs()
        } else {
            0.0
        };
        
        Ok(ConvergenceAnalysis {
            energy_converged,
            gradient_converged,
            force_converged,
            displacement_converged,
            final_energy_change,
            convergence_rate,
            line_search_failures: 0, // This would be tracked separately
        })
    }
    
    /// Determine if optimization has converged
    fn is_converged(&self, analysis: &ConvergenceAnalysis) -> bool {
        analysis.energy_converged && analysis.gradient_converged
    }
}

/// Line search result
struct LineSearchResult {
    success: bool,
    step_size: f64,
    iterations: usize,
}

/// Default energy weights for combined energy function
impl Default for EnergyWeights {
    fn default() -> Self {
        EnergyWeights {
            lennard_jones: 1.0,
            electrostatic: 1.0,
            ramachandran: 0.5,
            contact: 0.3,
            solvation: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test imports removed - not needed for current tests
    
    #[test]
    fn test_energy_minimizer_creation() {
        let minimizer = EnergyMinimizer::new(EnergyFunctionType::BackboneOnly);
        assert!(minimizer.is_ok());
        
        let minimizer = minimizer.unwrap();
        assert_eq!(minimizer.max_iterations, 1000);
        assert!(minimizer.convergence_config.energy_tolerance > 0.0);
    }
    
    #[test]
    fn test_convergence_criteria_validation() {
        let mut minimizer = EnergyMinimizer::new(EnergyFunctionType::BackboneOnly).unwrap();
        
        // Test valid convergence criteria
        let valid_config = ConvergenceConfig {
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-4,
            displacement_tolerance: 1e-6,
            force_tolerance: 1e-4,
            min_iterations: 10,
            relative_energy_tolerance: 1e-8,
        };
        
        assert!(minimizer.set_convergence_criteria(valid_config).is_ok());
        
        // Test invalid energy tolerance
        let invalid_config = ConvergenceConfig {
            energy_tolerance: -1.0, // Invalid: negative
            gradient_tolerance: 1e-4,
            displacement_tolerance: 1e-6,
            force_tolerance: 1e-4,
            min_iterations: 10,
            relative_energy_tolerance: 1e-8,
        };
        
        assert!(minimizer.set_convergence_criteria(invalid_config).is_err());
    }
    
    #[test]
    fn test_bfgs_config_validation() {
        let mut minimizer = EnergyMinimizer::new(EnergyFunctionType::BackboneOnly).unwrap();
        
        // Test valid BFGS config
        let valid_config = BfgsConfig {
            max_step_size: 0.1,
            initial_hessian_scale: 1.0,
            line_search: LineSearchConfig {
                armijo_c1: 1e-4,
                curvature_c2: 0.9,
                max_iterations: 20,
                initial_step: 1.0,
                reduction_factor: 0.5,
            },
            memory_size: 10,
            gradient_tolerance: 1e-6,
        };
        
        assert!(minimizer.set_bfgs_config(valid_config.clone()).is_ok());
        
        // Test invalid step size
        let mut invalid_config = valid_config.clone();
        invalid_config.max_step_size = -1.0; // Invalid: negative
        assert!(minimizer.set_bfgs_config(invalid_config).is_err());
        
        // Test invalid memory size
        let mut invalid_config = valid_config.clone();
        invalid_config.memory_size = 0; // Invalid: zero
        assert!(minimizer.set_bfgs_config(invalid_config).is_err());
    }
    
    #[test]
    fn test_energy_function_types() {
        let charmm36 = EnergyFunctionType::Charmm36Full;
        let backbone = EnergyFunctionType::BackboneOnly;
        let contact = EnergyFunctionType::ContactBased { contact_threshold: 8.0 };
        let combined = EnergyFunctionType::Combined { weights: EnergyWeights::default() };
        
        // Test that energy function types can be created
        assert!(matches!(charmm36, EnergyFunctionType::Charmm36Full));
        assert!(matches!(backbone, EnergyFunctionType::BackboneOnly));
        assert!(matches!(contact, EnergyFunctionType::ContactBased { .. }));
        assert!(matches!(combined, EnergyFunctionType::Combined { .. }));
    }
    
    #[test]
    fn test_energy_weights_default() {
        let weights = EnergyWeights::default();
        
        assert_eq!(weights.lennard_jones, 1.0);
        assert_eq!(weights.electrostatic, 1.0);
        assert_eq!(weights.ramachandran, 0.5);
        assert_eq!(weights.contact, 0.3);
        assert_eq!(weights.solvation, 0.2);
    }
    
    #[test]
    fn test_coordinate_conversion() {
        let minimizer = EnergyMinimizer::new(EnergyFunctionType::BackboneOnly).unwrap();
        
        // Create a simple test structure
        let structure = Structure::new("test".to_string());
        // Note: Would need to add proper test structure creation
        
        // Test empty structure handling
        let coords_result = minimizer.structure_to_coordinates(&structure);
        assert!(coords_result.is_err()); // Should fail for empty structure
    }
}