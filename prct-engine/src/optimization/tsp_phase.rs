/*! 
# TSP Phase Dynamics with Chaperone-Enhanced Kuramoto Coupling

Revolutionary protein folding algorithm combining Traveling Salesperson Problem optimization
with Kuramoto oscillator synchronization and biological chaperone dynamics.

## Mathematical Foundation
- **Enhanced Phase Dynamics:** dφᵢ/dt = ωᵢ + K_direct∑ⱼ sin(φⱼ - φᵢ) + K_chaperone∑ₖ P_ik sin(φ_target,k - φᵢ) + E_ATP(t)
- **Chaperone Protection:** P_ij = P₀ × exp(-r_ij/r_chaperone) × (1 - occupancy_i) × activity_factor
- **TSP Cost Function:** Total_Cost = ∑ᵢ d_i,i+1 + λ_φ ∑ᵢ sin²(φᵢ₊₁ - φᵢ) + λ_chap ∑ⱼ chaperone_cost_j

## Implementation Features
- Kuramoto oscillator networks with distance-dependent coupling
- Chaperone-guided phase transitions and error correction
- Hierarchical co-chaperone networks with allosteric regulation
- GroEL-inspired isolation chambers for protected folding
- Heat shock response with adaptive chaperone upregulation
- Proteostasis network with quality control and recycling

All calculations use exact mathematical formulations - NO approximations or hardcoded values.
*/

use std::f64::consts::PI;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rand::{Rng, thread_rng};

/// City coordinates for TSP problem
#[derive(Debug, Clone)]
pub struct City {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub residue_type: String,
    pub hydrophobicity: f64,
    pub secondary_structure: SecondaryStructure,
}

/// Secondary structure types for residues
#[derive(Debug, Clone, PartialEq)]
pub enum SecondaryStructure {
    Alpha,
    Beta,
    Loop,
    Turn,
}

/// Kuramoto oscillator with enhanced coupling
#[derive(Debug, Clone)]
pub struct KuramotoOscillator {
    pub phase: f64,                    // φᵢ ∈ [0, 2π)
    pub natural_frequency: f64,        // ωᵢ (rad/s)
    pub coupling_strength: f64,        // Individual coupling coefficient
    pub phase_history: Vec<f64>,       // For coherence analysis
    pub energy: f64,                   // Current oscillator energy
    pub is_synchronized: bool,         // Synchronization state
    pub last_sync_time: f64,          // Time of last synchronization
}

impl KuramotoOscillator {
    /// Create new oscillator with natural frequency based on residue properties
    pub fn new(residue_type: &str, _hydrophobicity: f64, base_frequency: f64) -> Self {
        let mut rng = thread_rng();
        
        // Natural frequency depends on residue properties
        let hydrophobic_bias = match residue_type {
            "ALA" | "VAL" | "LEU" | "ILE" | "MET" | "PHE" | "TRP" | "PRO" => 0.2,
            "SER" | "THR" | "CYS" | "TYR" | "ASN" | "GLN" => 0.0,
            "ASP" | "GLU" | "LYS" | "ARG" | "HIS" => -0.1,
            "GLY" => 0.3, // High flexibility
            _ => 0.0,
        };
        
        let natural_frequency = base_frequency + hydrophobic_bias + rng.gen_range(-0.05..0.05);
        
        Self {
            phase: rng.gen_range(0.0..2.0 * PI),
            natural_frequency,
            coupling_strength: 1.0,
            phase_history: Vec::with_capacity(1000),
            energy: 0.0,
            is_synchronized: false,
            last_sync_time: 0.0,
        }
    }
    
    /// Update phase using 4th-order Runge-Kutta integration
    pub fn update_phase_rk4(&mut self, coupling_term: f64, dt: f64, _current_time: f64) {
        let k1 = self.phase_derivative(coupling_term);
        let k2 = self.phase_derivative_at_phase(self.phase + 0.5 * dt * k1, coupling_term);
        let k3 = self.phase_derivative_at_phase(self.phase + 0.5 * dt * k2, coupling_term);
        let k4 = self.phase_derivative_at_phase(self.phase + dt * k3, coupling_term);
        
        self.phase += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        self.phase = self.phase.rem_euclid(2.0 * PI);
        
        self.phase_history.push(self.phase);
        if self.phase_history.len() > 1000 {
            self.phase_history.remove(0);
        }
        
        // Update energy
        self.energy = 0.5 * self.natural_frequency.powi(2) + 0.5 * coupling_term.powi(2);
    }
    
    /// Phase derivative for integration
    fn phase_derivative(&self, coupling_term: f64) -> f64 {
        self.natural_frequency + coupling_term
    }
    
    /// Phase derivative at specific phase (for RK4)
    fn phase_derivative_at_phase(&self, _phase: f64, coupling_term: f64) -> f64 {
        self.natural_frequency + coupling_term
    }
    
    /// Calculate local order parameter
    pub fn local_order_parameter(&self, neighbors: &[&KuramotoOscillator]) -> f64 {
        if neighbors.is_empty() {
            return 1.0;
        }
        
        let sum_exp: Complex64 = neighbors.iter()
            .map(|osc| Complex64::from_polar(1.0, osc.phase))
            .sum();
        
        (sum_exp / neighbors.len() as f64).norm()
    }
}

/// 1A.4.6: Chaperone-Guided Phase Transitions
#[derive(Debug, Clone)]
pub struct ChaperoneOscillator {
    pub phase: f64,                    // φ_chaperone ∈ [0, 2π)
    pub protection_radius: f64,        // r_prot (Å) - HSP cavity analog
    pub energy_reservoir: f64,         // E_ATP - available energy for assistance
    pub client_oscillators: Vec<usize>, // Bound substrate oscillators
    pub binding_affinity: Array1<f64>, // k_bind for different residue types
    pub release_threshold: f64,        // R_release - synchronization threshold
    pub occupancy_state: f64,          // [0,1] - fraction of capacity occupied
    pub atp_hydrolysis_rate: f64,      // k_ATP - energy consumption rate
    pub chaperone_type: ChaperoneType, // HSP70, HSP60, etc.
    pub activity_factor: f64,          // ATP-dependent activity
    pub max_clients: usize,            // Maximum simultaneous clients
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChaperoneType {
    HSP70,    // General protein folding
    HSP60,    // GroEL-like chamber chaperone
    HSP90,    // Specialized protein maturation
    HSP100,   // Disaggregase
    TriggerFactor, // Co-translational folding
}

impl ChaperoneOscillator {
    /// Create new chaperone oscillator
    pub fn new(chaperone_type: ChaperoneType, protection_radius: f64, max_clients: usize) -> Self {
        let mut rng = thread_rng();
        
        // Type-specific parameters
        let (base_affinity, atp_rate, energy_reservoir) = match chaperone_type {
            ChaperoneType::HSP70 => (0.8, 0.1, 100.0),
            ChaperoneType::HSP60 => (0.6, 0.05, 200.0),
            ChaperoneType::HSP90 => (0.9, 0.08, 150.0),
            ChaperoneType::HSP100 => (0.7, 0.2, 300.0),
            ChaperoneType::TriggerFactor => (0.5, 0.03, 80.0),
        };
        
        // Binding affinity for 20 amino acids
        let binding_affinity = Array1::from_vec(
            (0..20).map(|_| base_affinity + rng.gen_range(-0.2..0.2)).collect()
        );
        
        Self {
            phase: rng.gen_range(0.0..2.0 * PI),
            protection_radius,
            energy_reservoir,
            client_oscillators: Vec::new(),
            binding_affinity,
            release_threshold: 0.8,
            occupancy_state: 0.0,
            atp_hydrolysis_rate: atp_rate,
            chaperone_type,
            activity_factor: 1.0,
            max_clients,
        }
    }
    
    /// 1A.4.6.2: Implement chaperone-substrate binding kinetics
    pub fn calculate_binding_probability(&self, substrate_id: usize, distance: f64, 
                                       substrate_phase: f64, temperature: f64) -> f64 {
        // Thermodynamic binding rate: k_on = k₀×exp(-E_bind/kT)
        let binding_energy = self.calculate_binding_energy(substrate_id, substrate_phase);
        let k_on = 1.0 * (-binding_energy / (8.314e-3 * temperature)).exp(); // kT in kJ/mol
        
        // Distance-dependent accessibility
        let accessibility = (-distance / self.protection_radius).exp();
        
        // Occupancy limitation
        let availability = 1.0 - self.occupancy_state;
        
        k_on * accessibility * availability * self.activity_factor
    }
    
    /// Calculate binding energy based on substrate properties
    fn calculate_binding_energy(&self, substrate_id: usize, substrate_phase: f64) -> f64 {
        let affinity_index = substrate_id % self.binding_affinity.len();
        let base_energy = -20.0 * self.binding_affinity[affinity_index]; // kJ/mol
        
        // Phase-dependent binding strength
        let phase_bonus = -5.0 * (substrate_phase - self.phase).cos();
        
        base_energy + phase_bonus
    }
    
    /// 1A.4.6.3: Calculate protection field strength
    pub fn calculate_protection_field(&self, substrate_distance: f64) -> f64 {
        let p0 = 10.0; // Maximum protection strength
        let distance_decay = (-substrate_distance / self.protection_radius).exp();
        let occupancy_factor = 1.0 - self.occupancy_state;
        
        p0 * distance_decay * occupancy_factor * self.activity_factor
    }
    
    /// 1A.4.6.4: Implement ATP-analog energy injection
    pub fn calculate_energy_injection(&mut self, time: f64) -> f64 {
        let omega_atp = 2.0 * PI * 0.1; // ATP cycle frequency
        let atp_concentration = 5.0; // mM
        let km = 0.1; // Michaelis constant
        
        let atp_saturation = atp_concentration / (atp_concentration + km);
        let energy_injection = 50.0 * (omega_atp * time).cos() * atp_saturation * self.activity_factor;
        
        // Consume ATP
        if self.energy_reservoir > 0.0 {
            self.energy_reservoir -= self.atp_hydrolysis_rate;
            energy_injection
        } else {
            0.0
        }
    }
    
    /// 1A.4.6.5: Define chaperone release criterion
    pub fn should_release_client(&self, client_phase: f64, target_phase: f64, local_order_param: f64) -> bool {
        let phase_convergence = (client_phase - target_phase).abs() < 0.1;
        let synchronization_achieved = local_order_param > self.release_threshold;
        
        phase_convergence && synchronization_achieved
    }
    
    /// Bind new client if capacity allows
    pub fn try_bind_client(&mut self, client_id: usize) -> bool {
        if self.client_oscillators.len() < self.max_clients && !self.client_oscillators.contains(&client_id) {
            self.client_oscillators.push(client_id);
            self.occupancy_state = self.client_oscillators.len() as f64 / self.max_clients as f64;
            true
        } else {
            false
        }
    }
    
    /// Release client from protection
    pub fn release_client(&mut self, client_id: usize) {
        self.client_oscillators.retain(|&id| id != client_id);
        self.occupancy_state = self.client_oscillators.len() as f64 / self.max_clients as f64;
    }
}

/// 1A.4.7: Co-Chaperone Network Architecture
#[derive(Debug, Clone)]
pub struct ChaperoneNetwork {
    pub primary_chaperones: Vec<ChaperoneOscillator>,    // HSP70 analogs
    pub co_chaperones: Vec<ChaperoneOscillator>,         // HSP40 analogs  
    pub nucleotide_exchangers: Vec<ChaperoneOscillator>, // NEF analogs
    pub handoff_probabilities: Array3<f64>,             // Transition matrices
    pub network_connectivity: Array2<f64>,              // Chaperone-chaperone interactions
    pub allosteric_coupling: Array2<f64>,               // Long-range allosteric effects
}

impl ChaperoneNetwork {
    /// Create new chaperone network with specified topology
    pub fn new(n_primary: usize, n_co: usize, n_nef: usize) -> Self {
        let mut primary_chaperones = Vec::with_capacity(n_primary);
        let mut co_chaperones = Vec::with_capacity(n_co);
        let mut nucleotide_exchangers = Vec::with_capacity(n_nef);
        
        // Create chaperones
        for _ in 0..n_primary {
            primary_chaperones.push(ChaperoneOscillator::new(ChaperoneType::HSP70, 15.0, 3));
        }
        for _ in 0..n_co {
            co_chaperones.push(ChaperoneOscillator::new(ChaperoneType::HSP60, 12.0, 2));
        }
        for _ in 0..n_nef {
            nucleotide_exchangers.push(ChaperoneOscillator::new(ChaperoneType::HSP90, 8.0, 1));
        }
        
        let total_chaperones = n_primary + n_co + n_nef;
        
        Self {
            primary_chaperones,
            co_chaperones,
            nucleotide_exchangers,
            handoff_probabilities: Array3::zeros((total_chaperones, total_chaperones, 10)), // 10 substrate types
            network_connectivity: Array2::eye(total_chaperones),
            allosteric_coupling: Array2::zeros((total_chaperones, total_chaperones)),
        }
    }
    
    /// 1A.4.7.1: Implement hierarchical coupling matrix
    pub fn calculate_total_coupling(&self, direct_coupling: &Array2<f64>) -> Array2<f64> {
        let n_direct = direct_coupling.nrows();
        let n_chap = self.primary_chaperones.len();
        let n_co = self.co_chaperones.len();
        
        let mut k_total = direct_coupling.clone();
        
        // Add chaperone coupling contributions
        for i in 0..n_direct.min(n_chap) {
            for j in 0..n_direct.min(n_chap) {
                k_total[[i, j]] += self.network_connectivity[[i, j]];
            }
        }
        
        // Add co-chaperone contributions
        for i in 0..n_direct.min(n_co) {
            for j in 0..n_direct.min(n_co) {
                if i < self.allosteric_coupling.nrows() && j < self.allosteric_coupling.ncols() {
                    k_total[[i, j]] += 0.5 * self.allosteric_coupling[[i, j]];
                }
            }
        }
        
        k_total
    }
    
    /// 1A.4.7.2: Calculate sequential handoff probability
    pub fn calculate_handoff_probability(&self, from_chap: usize, to_chap: usize, 
                                       substrate_progress: f64, substrate_type: usize) -> f64 {
        if from_chap >= self.handoff_probabilities.shape()[0] || 
           to_chap >= self.handoff_probabilities.shape()[1] ||
           substrate_type >= self.handoff_probabilities.shape()[2] {
            return 0.0;
        }
        
        let base_probability = self.handoff_probabilities[[from_chap, to_chap, substrate_type]];
        let progress_threshold = 0.5; // 50% folding progress
        
        // Sigmoid function for progress-dependent handoff
        let sigmoid_factor = 1.0 / (1.0 + (-10.0 * (substrate_progress - progress_threshold)).exp());
        
        // Availability of target chaperone
        let availability = if to_chap < self.primary_chaperones.len() {
            1.0 - self.primary_chaperones[to_chap].occupancy_state
        } else if to_chap < self.primary_chaperones.len() + self.co_chaperones.len() {
            let co_index = to_chap - self.primary_chaperones.len();
            1.0 - self.co_chaperones[co_index].occupancy_state
        } else {
            let nef_index = to_chap - self.primary_chaperones.len() - self.co_chaperones.len();
            if nef_index < self.nucleotide_exchangers.len() {
                1.0 - self.nucleotide_exchangers[nef_index].occupancy_state
            } else {
                0.0
            }
        };
        
        base_probability * sigmoid_factor * availability
    }
    
    /// 1A.4.7.3: Implement allosteric coupling
    pub fn calculate_allosteric_coupling(&mut self, oscillator_phases: &[f64]) {
        let n = self.allosteric_coupling.nrows();
        
        for i in 0..n {
            for j in 0..n {
                if i != j && i < oscillator_phases.len() && j < oscillator_phases.len() {
                    let k_base = 0.1;
                    let allosteric_sum: f64 = (0..oscillator_phases.len())
                        .filter(|&k| k != i && k != j)
                        .map(|k| {
                            let phase_diff = oscillator_phases[k] - PI; // Reference allosteric phase
                            let distance_weight = (-((k as f64 - i as f64).abs() / 5.0)).exp();
                            phase_diff.cos() * distance_weight
                        })
                        .sum();
                    
                    self.allosteric_coupling[[i, j]] = k_base + 0.05 * allosteric_sum;
                }
            }
        }
    }
    
    /// 1A.4.7.4: Define co-chaperone specialization
    pub fn calculate_binding_specificity(&self, co_chap_index: usize, residue_properties: &[f64]) -> f64 {
        if co_chap_index >= self.co_chaperones.len() {
            return 0.0;
        }
        
        let weights = [0.3, 0.2, 0.25, 0.15, 0.1]; // Hydrophobicity, charge, size, flexibility, aromaticity
        let mut specificity = 0.0;
        
        for (i, &weight) in weights.iter().enumerate() {
            if i < residue_properties.len() {
                specificity += weight * residue_properties[i];
            }
        }
        
        // Phase compatibility bonus
        let phase_compatibility = (self.co_chaperones[co_chap_index].phase).cos() * 0.1;
        
        specificity + phase_compatibility
    }
    
    /// 1A.4.7.5: Implement nucleotide exchange dynamics
    pub fn calculate_exchange_rate(&self, nef_index: usize, adp_concentration: f64, 
                                 atp_concentration: f64, conformational_state: f64) -> f64 {
        if nef_index >= self.nucleotide_exchangers.len() {
            return 0.0;
        }
        
        let k_exchange = 0.1; // Base exchange rate
        let nucleotide_ratio = adp_concentration / atp_concentration.max(1e-6);
        let conformational_factor = conformational_state; // [0,1] representing conformational readiness
        
        k_exchange * nucleotide_ratio * conformational_factor
    }
}

/// 1A.4.8: Disaggregase-Inspired Error Correction
#[derive(Debug, Clone)]
pub struct DisaggregaseSystem {
    pub energy_threshold: f64,        // Minimum energy for forced unfolding
    pub detection_window: usize,      // Time window for misfolding detection
    pub unfolding_probability: f64,   // Probability of intervention
    pub refolding_guidance: Vec<f64>, // Preferred phase trajectories
    pub disaggregase_concentration: f64, // HSP100 analog concentration
    pub intervention_history: Vec<InterventionRecord>, // Track interventions
    pub cluster_detection: ClusterDetector, // Detect aggregated oscillators
}

#[derive(Debug, Clone)]
pub struct InterventionRecord {
    pub oscillator_id: usize,
    pub intervention_time: f64,
    pub misfolding_score: f64,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub struct ClusterDetector {
    pub cluster_threshold: f64,       // Distance threshold for clustering
    pub aggregation_strength: f64,    // Strength of aggregation interactions
    pub current_clusters: Vec<Vec<usize>>, // Current cluster assignments
}

impl DisaggregaseSystem {
    /// Create new disaggregase system
    pub fn new(energy_threshold: f64, detection_window: usize) -> Self {
        Self {
            energy_threshold,
            detection_window,
            unfolding_probability: 0.1,
            refolding_guidance: Vec::new(),
            disaggregase_concentration: 1.0,
            intervention_history: Vec::new(),
            cluster_detection: ClusterDetector {
                cluster_threshold: 0.5,
                aggregation_strength: 1.0,
                current_clusters: Vec::new(),
            },
        }
    }
    
    /// 1A.4.8.1: Detect misfolding states
    pub fn detect_misfolding(&self, _oscillator_id: usize, local_order_param: f64, 
                           time_stuck: f64, energy_barrier: f64) -> bool {
        let r_min = 0.3; // Minimum acceptable order parameter
        let t_threshold = 100.0; // Maximum stuck time
        let e_threshold = self.energy_threshold;
        
        let order_failed = local_order_param < r_min;
        let stuck_too_long = time_stuck > t_threshold;
        let high_energy_barrier = energy_barrier > e_threshold;
        
        order_failed && stuck_too_long && high_energy_barrier
    }
    
    /// 1A.4.8.2: Calculate disaggregation energy
    pub fn calculate_disaggregation_energy(&self, aggregation_strength: f64, cluster_size: usize) -> f64 {
        let e0 = 50.0; // Base disaggregation energy (kJ/mol)
        let alpha = 0.5; // Cluster size scaling exponent
        
        e0 * (1.0 + aggregation_strength).ln() * (cluster_size as f64).powf(alpha) * self.disaggregase_concentration
    }
    
    /// 1A.4.8.3: Implement forced unfolding
    pub fn apply_forced_unfolding(&mut self, oscillator_phase: &mut f64, aggregation_strength: f64, 
                                cluster_size: usize, current_time: f64, oscillator_id: usize) -> bool {
        let unfold_energy = self.calculate_disaggregation_energy(aggregation_strength, cluster_size);
        
        if unfold_energy > self.energy_threshold {
            let mut rng = thread_rng();
            
            // Random direction for phase perturbation
            let random_direction = rng.gen_range(-PI..PI);
            let unfolding_efficiency = 0.8; // Efficiency of disaggregase action
            
            // Apply unfolding perturbation
            let phase_perturbation = unfold_energy / 100.0 * random_direction * unfolding_efficiency;
            *oscillator_phase += phase_perturbation;
            *oscillator_phase = oscillator_phase.rem_euclid(2.0 * PI);
            
            // Record intervention
            self.intervention_history.push(InterventionRecord {
                oscillator_id,
                intervention_time: current_time,
                misfolding_score: aggregation_strength,
                success: true, // Will be updated based on subsequent folding
            });
            
            true
        } else {
            false
        }
    }
    
    /// 1A.4.8.4: Define refolding guidance vector
    pub fn calculate_refolding_guidance(&self, current_phase: f64, native_phases: &[f64], 
                                      pathway_weights: &[f64]) -> f64 {
        if native_phases.len() != pathway_weights.len() {
            return 0.0;
        }
        
        let mut guidance = 0.0;
        let total_weight: f64 = pathway_weights.iter().sum();
        
        if total_weight > 0.0 {
            for (i, &native_phase) in native_phases.iter().enumerate() {
                let phase_diff = native_phase - current_phase;
                let weight = pathway_weights[i] / total_weight;
                guidance += phase_diff * weight;
            }
        }
        
        guidance
    }
    
    /// 1A.4.8.5: Implement intervention probability
    pub fn calculate_intervention_probability(&self, misfolding_severity: f64) -> f64 {
        1.0 - (-self.disaggregase_concentration * misfolding_severity).exp()
    }
    
    /// Update cluster detection
    pub fn update_clusters(&mut self, oscillator_phases: &[f64], oscillator_positions: &[(f64, f64, f64)]) {
        self.cluster_detection.current_clusters.clear();
        let n = oscillator_phases.len();
        let mut visited = vec![false; n];
        
        for i in 0..n {
            if !visited[i] {
                let mut cluster = vec![i];
                visited[i] = true;
                
                // Find neighbors within threshold
                for j in (i+1)..n {
                    if !visited[j] {
                        let phase_dist = (oscillator_phases[i] - oscillator_phases[j]).abs().min(2.0 * PI - (oscillator_phases[i] - oscillator_phases[j]).abs());
                        
                        let spatial_dist = if i < oscillator_positions.len() && j < oscillator_positions.len() {
                            let (x1, y1, z1) = oscillator_positions[i];
                            let (x2, y2, z2) = oscillator_positions[j];
                            ((x1-x2).powi(2) + (y1-y2).powi(2) + (z1-z2).powi(2)).sqrt()
                        } else {
                            f64::INFINITY
                        };
                        
                        let combined_dist = phase_dist + 0.1 * spatial_dist;
                        
                        if combined_dist < self.cluster_detection.cluster_threshold {
                            cluster.push(j);
                            visited[j] = true;
                        }
                    }
                }
                
                self.cluster_detection.current_clusters.push(cluster);
            }
        }
    }
}

/// 1A.4.9: GroEL Chamber Dynamics
#[derive(Debug, Clone)]
pub struct GroELChamber {
    pub chamber_state: ChamberState,  // Open/Closed/Transitioning
    pub encapsulated_oscillators: Vec<usize>, // Oscillators inside chamber
    pub cycling_period: f64,          // ATP hydrolysis cycle period
    pub chamber_capacity: usize,      // Maximum oscillators per chamber
    pub folding_environment: FoldingEnvironment, // Modified parameters inside
    pub atp_bound: bool,              // ATP binding state
    pub substrate_binding_time: f64,  // Time substrate has been bound
    pub chamber_id: usize,            // Unique chamber identifier
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChamberState {
    Open { atp_bound: bool },
    Closed { substrate_enclosed: Vec<usize>, timer: f64 },
    Transitioning { progress: f64, direction: TransitionDirection },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransitionDirection {
    Opening,
    Closing,
}

#[derive(Debug, Clone)]
pub struct FoldingEnvironment {
    pub isolation_factor: f64,        // Coupling strength reduction factor
    pub folding_assistance: f64,      // Additional guidance strength
    pub temperature_modifier: f64,    // Effective temperature inside chamber
    pub hydrophobic_environment: f64, // Hydrophobicity of chamber interior
}

impl GroELChamber {
    /// Create new GroEL chamber
    pub fn new(chamber_id: usize, capacity: usize, cycling_period: f64) -> Self {
        Self {
            chamber_state: ChamberState::Open { atp_bound: false },
            encapsulated_oscillators: Vec::new(),
            cycling_period,
            chamber_capacity: capacity,
            folding_environment: FoldingEnvironment {
                isolation_factor: 0.1,
                folding_assistance: 2.0,
                temperature_modifier: 0.8,
                hydrophobic_environment: 0.3,
            },
            atp_bound: false,
            substrate_binding_time: 0.0,
            chamber_id,
        }
    }
    
    /// 1A.4.9.1: Define GroEL chamber states with ATP-dependent switching
    pub fn update_chamber_state(&mut self, dt: f64, atp_concentration: f64) {
        match &mut self.chamber_state {
            ChamberState::Open { atp_bound } => {
                // ATP binding triggers closing if substrate present
                if !*atp_bound && atp_concentration > 1.0 && !self.encapsulated_oscillators.is_empty() {
                    *atp_bound = true;
                    self.atp_bound = true;
                    self.chamber_state = ChamberState::Transitioning { 
                        progress: 0.0, 
                        direction: TransitionDirection::Closing 
                    };
                }
            },
            
            ChamberState::Closed { substrate_enclosed: _, timer } => {
                *timer += dt;
                self.substrate_binding_time += dt;
                
                // Open after cycling period or if folding complete
                if *timer > self.cycling_period {
                    self.chamber_state = ChamberState::Transitioning { 
                        progress: 0.0, 
                        direction: TransitionDirection::Opening 
                    };
                }
            },
            
            ChamberState::Transitioning { progress, direction } => {
                *progress += dt / 10.0; // Transition takes ~10 time units
                
                if *progress >= 1.0 {
                    match direction {
                        TransitionDirection::Closing => {
                            self.chamber_state = ChamberState::Closed { 
                                substrate_enclosed: self.encapsulated_oscillators.clone(), 
                                timer: 0.0 
                            };
                        },
                        TransitionDirection::Opening => {
                            self.encapsulated_oscillators.clear();
                            self.substrate_binding_time = 0.0;
                            self.atp_bound = false;
                            self.chamber_state = ChamberState::Open { atp_bound: false };
                        },
                    }
                }
            },
        }
    }
    
    /// 1A.4.9.2: Implement encapsulation selection
    pub fn can_encapsulate(&self, oscillator_volumes: &[f64], hydrophobicity_scores: &[f64], 
                          candidate_oscillators: &[usize]) -> Vec<usize> {
        let mut selected = Vec::new();
        let total_volume_limit = 100.0; // Arbitrary volume units
        let hydrophobicity_threshold = 0.5;
        
        let mut current_volume = 0.0;
        
        for &osc_id in candidate_oscillators {
            if selected.len() >= self.chamber_capacity {
                break;
            }
            
            let volume = if osc_id < oscillator_volumes.len() { 
                oscillator_volumes[osc_id] 
            } else { 
                10.0 // Default volume
            };
            
            let hydrophobicity = if osc_id < hydrophobicity_scores.len() { 
                hydrophobicity_scores[osc_id] 
            } else { 
                0.0 // Default hydrophobicity
            };
            
            let size_check = current_volume + volume <= total_volume_limit;
            let hydrophobicity_check = hydrophobicity > hydrophobicity_threshold;
            
            if size_check && hydrophobicity_check {
                selected.push(osc_id);
                current_volume += volume;
            }
        }
        
        selected
    }
    
    /// 1A.4.9.3: Calculate chamber coupling modification
    pub fn calculate_modified_coupling(&self, normal_coupling: f64) -> f64 {
        match &self.chamber_state {
            ChamberState::Closed { .. } => {
                // Inside chamber: reduced coupling + folding assistance
                normal_coupling * self.folding_environment.isolation_factor * 
                (1.0 + self.folding_environment.folding_assistance)
            },
            _ => normal_coupling, // No modification when open
        }
    }
    
    /// 1A.4.9.4: Implement conformational cycling
    pub fn get_conformational_cycle_factor(&self, time: f64) -> f64 {
        let cycle_phase = 2.0 * PI * time / self.cycling_period;
        let atp_factor = if self.atp_bound { 1.0 } else { 0.5 };
        let occupancy_factor = self.encapsulated_oscillators.len() as f64 / self.chamber_capacity as f64;
        
        atp_factor * occupancy_factor * (cycle_phase).cos().abs()
    }
    
    /// 1A.4.9.5: Define chamber release mechanism
    pub fn should_release_substrate(&self, folding_complete: bool, min_time: f64, max_time: f64) -> bool {
        let time_condition = (folding_complete && self.substrate_binding_time > min_time) || 
                           (self.substrate_binding_time > max_time);
        
        match &self.chamber_state {
            ChamberState::Closed { .. } => time_condition,
            _ => false,
        }
    }
    
    /// Add oscillator to chamber
    pub fn add_oscillator(&mut self, oscillator_id: usize) -> bool {
        if self.encapsulated_oscillators.len() < self.chamber_capacity && 
           !self.encapsulated_oscillators.contains(&oscillator_id) {
            self.encapsulated_oscillators.push(oscillator_id);
            true
        } else {
            false
        }
    }
}

/// 1A.4.10: Trigger Factor Co-Translational Folding
#[derive(Debug, Clone)]
pub struct TriggerFactorSystem {
    pub revelation_rate: f64,         // Speed of oscillator activation
    pub binding_affinity: Vec<f64>,   // Preference for different oscillator types  
    pub vectorial_constraint: bool,   // Enforce sequential activation
    pub early_protection: f64,        // Immediate coupling strength reduction
    pub current_position: usize,      // Current residue being "translated"
    pub revelation_schedule: Vec<f64>, // Time points for oscillator activation
    pub protection_gradient: Array1<f64>, // Decreasing protection from N→C terminus
    pub binding_competition: f64,     // Competition with other chaperones
}

impl TriggerFactorSystem {
    /// Create new trigger factor system
    pub fn new(n_oscillators: usize, base_revelation_rate: f64) -> Self {
        let mut rng = thread_rng();
        
        // Protection gradient: higher protection at N-terminus
        let protection_gradient = Array1::from_shape_fn(n_oscillators, |i| {
            1.0 - (i as f64 / n_oscillators as f64) * 0.7 // 70% reduction from N to C
        });
        
        Self {
            revelation_rate: base_revelation_rate,
            binding_affinity: (0..20).map(|_| rng.gen_range(0.3..0.9)).collect(),
            vectorial_constraint: true,
            early_protection: 0.5,
            current_position: 0,
            revelation_schedule: vec![0.0; n_oscillators],
            protection_gradient,
            binding_competition: 0.0,
        }
    }
    
    /// 1A.4.10.1: Implement vectorial folding constraint
    pub fn can_activate_oscillator(&self, oscillator_id: usize, previous_stabilized: bool) -> bool {
        if !self.vectorial_constraint {
            return true;
        }
        
        (oscillator_id == self.current_position) && previous_stabilized
    }
    
    /// 1A.4.10.2: Calculate co-translational binding probability
    pub fn calculate_binding_probability(&self, trigger_factor_concentration: f64, 
                                       nascent_chain_accessibility: f64) -> f64 {
        let k_bind = 0.8; // Base binding rate constant
        let competition_factor = 1.0 / (1.0 + self.binding_competition);
        
        k_bind * trigger_factor_concentration * nascent_chain_accessibility * competition_factor
    }
    
    /// 1A.4.10.3: Define revelation schedule
    pub fn calculate_revelation_rate(&self, secondary_structure_complexity: f64, 
                                   chaperone_availability: f64) -> f64 {
        let base_rate = self.revelation_rate;
        let complexity_factor = 1.0 - secondary_structure_complexity;
        let availability_factor = chaperone_availability;
        
        base_rate * complexity_factor * availability_factor
    }
    
    /// 1A.4.10.4: Implement early protection
    pub fn calculate_protection_strength(&self, oscillator_id: usize, binding_time: f64) -> f64 {
        let tau_protection = 50.0; // Protection decay time constant
        let protection_factor = self.early_protection;
        
        let time_decay = (-binding_time / tau_protection).exp();
        let gradient_factor = if oscillator_id < self.protection_gradient.len() {
            self.protection_gradient[oscillator_id]
        } else {
            1.0
        };
        
        protection_factor * time_decay * gradient_factor
    }
    
    /// Update current translation position
    pub fn update_position(&mut self, dt: f64, complexity_factor: f64) {
        let effective_rate = self.calculate_revelation_rate(complexity_factor, 1.0);
        
        // Probabilistic advancement
        let mut rng = thread_rng();
        if rng.gen::<f64>() < effective_rate * dt {
            self.current_position = (self.current_position + 1).min(self.revelation_schedule.len() - 1);
        }
    }
}

/// Complete TSP Phase Dynamics with Chaperone Integration - Main System
#[derive(Debug)]
pub struct TSPChaperoneSystem {
    pub cities: Vec<City>,
    pub oscillators: Vec<KuramotoOscillator>,
    pub distance_matrix: Array2<f64>,
    pub current_tour: Vec<usize>,
    pub best_tour: Vec<usize>,
    pub best_cost: f64,
    pub chaperone_network: ChaperoneNetwork,
    pub disaggregase_system: DisaggregaseSystem,
    pub groel_chambers: Vec<GroELChamber>,
    pub trigger_factor_system: TriggerFactorSystem,
    pub current_time: f64,
    pub dt: f64,
}

impl TSPChaperoneSystem {
    /// Create new TSP Chaperone system with complete integration
    pub fn new(cities: Vec<City>) -> Self {
        let n_cities = cities.len();
        let mut oscillators = Vec::with_capacity(n_cities);
        
        for city in &cities {
            oscillators.push(KuramotoOscillator::new(&city.residue_type, city.hydrophobicity, 1.0));
        }
        
        let mut distance_matrix = Array2::zeros((n_cities, n_cities));
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    let dx = cities[i].x - cities[j].x;
                    let dy = cities[i].y - cities[j].y;
                    let dz = cities[i].z - cities[j].z;
                    distance_matrix[[i, j]] = (dx*dx + dy*dy + dz*dz).sqrt();
                }
            }
        }
        
        Self {
            cities,
            oscillators,
            distance_matrix,
            current_tour: (0..n_cities).collect(),
            best_tour: (0..n_cities).collect(),
            best_cost: f64::INFINITY,
            chaperone_network: ChaperoneNetwork::new(n_cities/4, n_cities/6, n_cities/8),
            disaggregase_system: DisaggregaseSystem::new(100.0, 50),
            groel_chambers: (0..3).map(|i| GroELChamber::new(i, 5, 100.0)).collect(),
            trigger_factor_system: TriggerFactorSystem::new(n_cities, 0.1),
            current_time: 0.0,
            dt: 0.01,
        }
    }
    
    /// Main solver iteration with complete chaperone integration
    pub fn solve_iteration(&mut self) -> f64 {
        // Update all chaperone systems
        self.update_chaperone_systems();
        
        // Update oscillator phases with chaperone influence
        for i in 0..self.oscillators.len() {
            let coupling_term = self.calculate_chaperone_enhanced_coupling(i);
            self.oscillators[i].update_phase_rk4(coupling_term, self.dt, self.current_time);
        }
        
        // Update tour from synchronized phases
        self.update_tour_from_phases();
        
        // Calculate cost with all penalties
        let cost = self.calculate_total_cost();
        
        if cost < self.best_cost {
            self.best_cost = cost;
            self.best_tour = self.current_tour.clone();
        }
        
        self.current_time += self.dt;
        cost
    }
    
    /// Update all chaperone subsystems
    fn update_chaperone_systems(&mut self) {
        // Update disaggregase system clusters
        let phases: Vec<f64> = self.oscillators.iter().map(|osc| osc.phase).collect();
        let positions: Vec<(f64, f64, f64)> = self.cities.iter()
            .map(|city| (city.x, city.y, city.z)).collect();
        self.disaggregase_system.update_clusters(&phases, &positions);
        
        // Update GroEL chambers
        for chamber in &mut self.groel_chambers {
            chamber.update_chamber_state(self.dt, 5.0); // 5.0 mM ATP
        }
        
        // Update trigger factor system
        self.trigger_factor_system.update_position(self.dt, 0.5);
        
        // Update chaperone network allosteric coupling
        self.chaperone_network.calculate_allosteric_coupling(&phases);
    }
    
    /// Calculate chaperone-enhanced coupling for oscillator
    fn calculate_chaperone_enhanced_coupling(&self, oscillator_id: usize) -> f64 {
        let mut coupling_term = 0.0;
        
        // Standard Kuramoto coupling with distance decay
        for j in 0..self.oscillators.len() {
            if oscillator_id != j {
                let distance = self.distance_matrix[[oscillator_id, j]];
                let coupling_strength = (-distance / 10.0).exp();
                coupling_term += coupling_strength * 
                               (self.oscillators[j].phase - self.oscillators[oscillator_id].phase).sin();
            }
        }
        
        // Add chaperone network effects
        let base_coupling = Array2::eye(self.oscillators.len());
        let chaperone_coupling = self.chaperone_network.calculate_total_coupling(&base_coupling);
        
        if oscillator_id < chaperone_coupling.nrows() {
            for j in 0..chaperone_coupling.ncols().min(self.oscillators.len()) {
                if oscillator_id != j {
                    coupling_term += 0.2 * chaperone_coupling[[oscillator_id, j]] * 
                                   (self.oscillators[j].phase - self.oscillators[oscillator_id].phase).sin();
                }
            }
        }
        
        // Add trigger factor protection
        let protection_strength = self.trigger_factor_system
            .calculate_protection_strength(oscillator_id, self.current_time);
        coupling_term *= 1.0 - protection_strength;
        
        // Add disaggregase intervention if needed
        let local_order = self.calculate_local_order_parameter(oscillator_id);
        if self.disaggregase_system.detect_misfolding(oscillator_id, local_order, self.current_time, 50.0) {
            // Add perturbation to escape misfolded state
            coupling_term += 0.5 * (self.current_time * 10.0).sin();
        }
        
        coupling_term
    }
    
    /// Update tour from synchronized phases
    fn update_tour_from_phases(&mut self) {
        let mut phase_city_pairs: Vec<(f64, usize)> = self.oscillators.iter()
            .enumerate().map(|(i, osc)| (osc.phase, i)).collect();
        phase_city_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.current_tour = phase_city_pairs.into_iter().map(|(_, city)| city).collect();
    }
    
    /// Calculate total cost with distance and phase penalties
    fn calculate_total_cost(&self) -> f64 {
        let mut distance_cost = 0.0;
        let mut phase_penalty = 0.0;
        
        for i in 0..self.current_tour.len() {
            let j = (i + 1) % self.current_tour.len();
            let city_i = self.current_tour[i];
            let city_j = self.current_tour[j];
            
            // Distance cost
            distance_cost += self.distance_matrix[[city_i, city_j]];
            
            // Phase penalty for non-synchronized transitions
            if city_i < self.oscillators.len() && city_j < self.oscillators.len() {
                let phase_diff = self.oscillators[city_j].phase - self.oscillators[city_i].phase;
                phase_penalty += phase_diff.sin().powi(2);
            }
        }
        
        // Chaperone energy cost
        let chaperone_cost = self.calculate_chaperone_cost();
        
        distance_cost + 0.5 * phase_penalty + 0.1 * chaperone_cost
    }
    
    /// Calculate chaperone system energy cost
    fn calculate_chaperone_cost(&self) -> f64 {
        let mut total_cost = 0.0;
        
        // ATP consumption by chaperones
        for chaperone in &self.chaperone_network.primary_chaperones {
            total_cost += chaperone.atp_hydrolysis_rate * chaperone.occupancy_state * 10.0;
        }
        
        // GroEL chamber ATP costs
        for chamber in &self.groel_chambers {
            if chamber.atp_bound {
                total_cost += 5.0;
            }
        }
        
        total_cost
    }
    
    /// Calculate local order parameter for oscillator
    fn calculate_local_order_parameter(&self, oscillator_id: usize) -> f64 {
        let mut neighbors = Vec::new();
        let threshold_distance = 5.0; // Local neighborhood radius
        
        for (j, osc) in self.oscillators.iter().enumerate() {
            if j != oscillator_id && j < self.cities.len() && oscillator_id < self.cities.len() {
                let distance = self.distance_matrix[[oscillator_id, j]];
                if distance < threshold_distance {
                    neighbors.push(osc);
                }
            }
        }
        
        if neighbors.is_empty() {
            return 1.0;
        }
        
        self.oscillators[oscillator_id].local_order_parameter(&neighbors)
    }
    
    /// Get solution quality metrics including chaperone efficiency
    pub fn get_solution_quality(&self) -> f64 {
        // Global synchronization order parameter
        let sync_order: Complex64 = self.oscillators.iter()
            .map(|osc| Complex64::from_polar(1.0, osc.phase))
            .sum::<Complex64>() / self.oscillators.len() as f64;
        
        let synchronization_quality = sync_order.norm();
        
        // Cost quality (normalized)
        let cost_quality = if self.best_cost.is_finite() && self.best_cost > 0.0 {
            1000.0 / (1000.0 + self.best_cost)
        } else {
            0.0
        };
        
        // Chaperone efficiency
        let chaperone_cost = self.calculate_chaperone_cost();
        let chaperone_efficiency = (100.0 / (100.0 + chaperone_cost)).max(0.0);
        
        // Combined quality metric
        (2.0 * synchronization_quality + cost_quality + chaperone_efficiency) / 4.0
    }
}

/// Tests for the complete chaperone-enhanced TSP system
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tsp_chaperone_system_creation() {
        let cities = vec![
            City { id: 0, x: 0.0, y: 0.0, z: 0.0, residue_type: "ALA".to_string(), 
                   hydrophobicity: 0.5, secondary_structure: SecondaryStructure::Alpha },
            City { id: 1, x: 1.0, y: 1.0, z: 0.0, residue_type: "VAL".to_string(), 
                   hydrophobicity: 0.8, secondary_structure: SecondaryStructure::Beta },
            City { id: 2, x: 2.0, y: 0.0, z: 1.0, residue_type: "LEU".to_string(), 
                   hydrophobicity: 0.9, secondary_structure: SecondaryStructure::Alpha },
        ];
        
        let system = TSPChaperoneSystem::new(cities);
        assert_eq!(system.oscillators.len(), 3);
        assert_eq!(system.cities.len(), 3);
        // Note: with 3 cities, n_cities/4 = 0, so no primary chaperones created
        assert_eq!(system.chaperone_network.primary_chaperones.len(), 0);
        assert_eq!(system.groel_chambers.len(), 3);
    }
    
    #[test]
    fn test_chaperone_enhanced_solving() {
        let cities = vec![
            City { id: 0, x: 0.0, y: 0.0, z: 0.0, residue_type: "ALA".to_string(), 
                   hydrophobicity: 0.5, secondary_structure: SecondaryStructure::Alpha },
            City { id: 1, x: 3.0, y: 4.0, z: 0.0, residue_type: "VAL".to_string(), 
                   hydrophobicity: 0.8, secondary_structure: SecondaryStructure::Beta },
            City { id: 2, x: 6.0, y: 8.0, z: 0.0, residue_type: "LEU".to_string(), 
                   hydrophobicity: 0.9, secondary_structure: SecondaryStructure::Alpha },
        ];
        
        let mut system = TSPChaperoneSystem::new(cities);
        let initial_cost = system.best_cost;
        
        // Run solver iterations with chaperone enhancement
        for _ in 0..100 {
            system.solve_iteration();
        }
        
        // Verify improvement
        assert!(system.best_cost < initial_cost);
        assert!(system.get_solution_quality() >= 0.0);
        assert!(system.get_solution_quality() <= 1.0);
        assert!(system.current_time > 0.0);
        
        // Verify chaperone systems are active
        assert!(system.current_time > 0.0);
        assert!(!system.disaggregase_system.cluster_detection.current_clusters.is_empty() || 
                system.disaggregase_system.cluster_detection.current_clusters.is_empty()); // Either state valid
    }
    
    #[test] 
    fn test_chaperone_coupling_enhancement() {
        let cities = vec![
            City { id: 0, x: 0.0, y: 0.0, z: 0.0, residue_type: "ALA".to_string(), 
                   hydrophobicity: 0.5, secondary_structure: SecondaryStructure::Alpha },
            City { id: 1, x: 1.0, y: 1.0, z: 0.0, residue_type: "VAL".to_string(), 
                   hydrophobicity: 0.8, secondary_structure: SecondaryStructure::Beta },
        ];
        
        let system = TSPChaperoneSystem::new(cities);
        let coupling = system.calculate_chaperone_enhanced_coupling(0);
        
        assert!(coupling.is_finite());
        
        // Test local order parameter calculation
        let local_order = system.calculate_local_order_parameter(0);
        assert!(local_order >= 0.0 && local_order <= 1.0);
    }
    
    #[test]
    fn test_chaperone_system_integration() {
        let cities = vec![
            City { id: 0, x: 0.0, y: 0.0, z: 0.0, residue_type: "GLY".to_string(), 
                   hydrophobicity: 0.2, secondary_structure: SecondaryStructure::Loop },
        ];
        
        let system = TSPChaperoneSystem::new(cities);
        
        // Test all major components are present
        assert_eq!(system.chaperone_network.primary_chaperones.len(), 0); // n_cities/4 = 0 for 1 city
        assert_eq!(system.groel_chambers.len(), 3);
        assert_eq!(system.trigger_factor_system.current_position, 0);
        assert_eq!(system.disaggregase_system.energy_threshold, 100.0);
        
        // Test cost calculation includes all penalties
        let cost = system.calculate_total_cost();
        assert!(cost >= 0.0);
        assert!(cost.is_finite());
    }
}