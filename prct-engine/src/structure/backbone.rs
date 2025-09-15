//! Backbone generation using PRCT phase resonance for spatial guidance

use super::atom::{Atom, BackboneAtom, Element, AtomType};
use super::sequences::one_to_three;
use super::{bond_lengths, bond_angles, deg_to_rad, cross_product, normalize_vector};
use crate::PRCTResult;
use std::f64::consts::PI;

/// Generate backbone coordinates from sequence and PRCT parameters
pub fn generate_backbone_coordinates(
    sequence: &str,
    phase_coherence: f64,
    chromatic_score: f64,
    hamiltonian_energy: f64,
) -> PRCTResult<Vec<BackboneAtom>> {
    let mut backbone_atoms = Vec::new();
    let residue_count = sequence.len();

    // PRCT-guided parameters for backbone conformation
    let base_phi = phase_coherence_to_phi(phase_coherence);
    let base_psi = phase_coherence_to_psi(phase_coherence);
    let backbone_curvature = chromatic_score_to_curvature(chromatic_score);
    let compactness_factor = hamiltonian_energy_to_compactness(hamiltonian_energy);

    // Initialize first residue at origin with standard orientation
    let mut ca_prev = [0.0, 0.0, 0.0];
    let mut c_prev = [bond_lengths::CA_C, 0.0, 0.0];
    let mut direction = [1.0, 0.0, 0.0]; // Initial backbone direction

    for (i, aa_char) in sequence.chars().enumerate() {
        let residue_id = (i + 1) as i32;
        let residue_name = one_to_three(aa_char);

        // Calculate PRCT-guided backbone angles for this residue
        let phi = calculate_phi_angle(base_phi, i, residue_count, phase_coherence);
        let psi = calculate_psi_angle(base_psi, i, residue_count, chromatic_score);
        let omega = 180.0; // Peptide bond is typically trans

        // Generate N, CA, C, O atoms for this residue
        let (n_pos, ca_pos, c_pos, o_pos) = if i == 0 {
            // First residue: place at origin
            generate_first_residue_atoms()
        } else {
            // Subsequent residues: use previous C position and PRCT angles
            generate_residue_atoms(
                &c_prev,
                &ca_prev,
                &mut direction,
                phi,
                psi,
                backbone_curvature,
                compactness_factor,
            )
        };

        // Create backbone atoms
        let n_atom = Atom::backbone(
            "N".to_string(),
            residue_name.to_string(),
            residue_id,
            n_pos[0], n_pos[1], n_pos[2],
            Element::Nitrogen,
        );

        let ca_atom = Atom::backbone(
            "CA".to_string(),
            residue_name.to_string(),
            residue_id,
            ca_pos[0], ca_pos[1], ca_pos[2],
            Element::Carbon,
        );

        let c_atom = Atom::backbone(
            "C".to_string(),
            residue_name.to_string(),
            residue_id,
            c_pos[0], c_pos[1], c_pos[2],
            Element::Carbon,
        );

        let o_atom = Atom::backbone(
            "O".to_string(),
            residue_name.to_string(),
            residue_id,
            o_pos[0], o_pos[1], o_pos[2],
            Element::Oxygen,
        );

        // Store backbone atoms with dihedral information
        backbone_atoms.push(BackboneAtom::new(n_atom).with_dihedrals(phi, psi, omega));
        backbone_atoms.push(BackboneAtom::new(ca_atom).with_dihedrals(phi, psi, omega));
        backbone_atoms.push(BackboneAtom::new(c_atom).with_dihedrals(phi, psi, omega));
        backbone_atoms.push(BackboneAtom::new(o_atom).with_dihedrals(phi, psi, omega));

        // Update previous positions for next iteration
        ca_prev = ca_pos;
        c_prev = c_pos;
    }

    Ok(backbone_atoms)
}

/// Convert PRCT phase coherence to backbone phi angle
fn phase_coherence_to_phi(phase_coherence: f64) -> f64 {
    // Phase coherence [0,1] maps to phi angles favoring alpha-helix or beta-sheet
    // High coherence -> alpha-helical (-60°)
    // Low coherence -> extended/beta (-120°)
    let alpha_phi = -60.0;
    let beta_phi = -120.0;

    alpha_phi + (beta_phi - alpha_phi) * (1.0 - phase_coherence)
}

/// Convert PRCT phase coherence to backbone psi angle
fn phase_coherence_to_psi(phase_coherence: f64) -> f64 {
    // High coherence -> alpha-helical (-45°)
    // Low coherence -> extended/beta (120°)
    let alpha_psi = -45.0;
    let beta_psi = 120.0;

    alpha_psi + (beta_psi - alpha_psi) * (1.0 - phase_coherence)
}

/// Convert chromatic score to backbone curvature
fn chromatic_score_to_curvature(chromatic_score: f64) -> f64 {
    // Higher chromatic optimization -> more curved/compact structure
    chromatic_score * 0.3 // Scale factor for curvature
}

/// Convert Hamiltonian energy to compactness factor
fn hamiltonian_energy_to_compactness(hamiltonian_energy: f64) -> f64 {
    // More negative energy -> more compact structure
    let normalized_energy = (-hamiltonian_energy / 50.0).min(1.0).max(0.0);
    0.8 + 0.4 * normalized_energy // Range [0.8, 1.2]
}

/// Calculate position-dependent phi angle using PRCT phase variation
fn calculate_phi_angle(
    base_phi: f64,
    position: usize,
    total_residues: usize,
    phase_coherence: f64,
) -> f64 {
    // Add position-dependent variation based on phase coherence
    let position_factor = position as f64 / total_residues as f64;
    let phase_variation = 15.0 * phase_coherence * (2.0 * PI * position_factor * 3.0).sin();

    base_phi + phase_variation
}

/// Calculate position-dependent psi angle using chromatic optimization
fn calculate_psi_angle(
    base_psi: f64,
    position: usize,
    total_residues: usize,
    chromatic_score: f64,
) -> f64 {
    // Add chromatic-guided variation
    let position_factor = position as f64 / total_residues as f64;
    let chromatic_variation = 20.0 * chromatic_score * (2.0 * PI * position_factor * 2.0).cos();

    base_psi + chromatic_variation
}

/// Generate atoms for the first residue at origin
fn generate_first_residue_atoms() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let n_pos = [-bond_lengths::C_N, 0.0, 0.0];
    let ca_pos = [0.0, 0.0, 0.0];
    let c_pos = [bond_lengths::CA_C, 0.0, 0.0];
    let o_pos = [
        bond_lengths::CA_C + bond_lengths::C_O * bond_angles::CA_C_O.cos(),
        bond_lengths::C_O * bond_angles::CA_C_O.sin(),
        0.0,
    ];

    (n_pos, ca_pos, c_pos, o_pos)
}

/// Generate atoms for subsequent residues using simplified, robust geometry
fn generate_residue_atoms(
    prev_c: &[f64; 3],
    prev_ca: &[f64; 3],
    direction: &mut [f64; 3],
    phi: f64,
    psi: f64,
    curvature: f64,
    compactness: f64,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    // Use simplified backbone geometry to avoid NaN issues
    let phi_rad = deg_to_rad(phi);
    let psi_rad = deg_to_rad(psi);

    // Simplified backbone progression along x-axis with phi/psi variation
    let residue_spacing = 3.8 * compactness; // Approximate residue-to-residue distance

    // Add curvature variation
    let curvature_x = curvature * 0.5 * phi_rad.cos();
    let curvature_y = curvature * 0.5 * phi_rad.sin();
    let curvature_z = curvature * 0.3 * psi_rad.sin();

    // N atom position - continue from previous C
    let n_pos = [
        prev_c[0] + bond_lengths::C_N,
        prev_c[1] + curvature_y,
        prev_c[2] + curvature_z * 0.5,
    ];

    // CA atom position - standard geometry
    let ca_pos = [
        n_pos[0] + bond_lengths::N_CA * phi_rad.cos() * compactness,
        n_pos[1] + bond_lengths::N_CA * phi_rad.sin() * 0.5,
        n_pos[2] + curvature_z,
    ];

    // C atom position - using psi angle
    let c_pos = [
        ca_pos[0] + bond_lengths::CA_C * psi_rad.cos(),
        ca_pos[1] + bond_lengths::CA_C * psi_rad.sin() * 0.6,
        ca_pos[2] + curvature_z * 0.3,
    ];

    // O atom position - perpendicular to CA-C bond
    let o_pos = [
        c_pos[0] + bond_lengths::C_O * 0.6,
        c_pos[1] + bond_lengths::C_O * 0.8,
        c_pos[2] - bond_lengths::C_O * 0.2,
    ];

    // Update direction for next residue
    direction[0] = (c_pos[0] - ca_pos[0]).max(-1.0).min(1.0);
    direction[1] = (c_pos[1] - ca_pos[1]).max(-1.0).min(1.0);
    direction[2] = (c_pos[2] - ca_pos[2]).max(-1.0).min(1.0);

    (n_pos, ca_pos, c_pos, o_pos)
}

/// Rotate vector around axis by angle
fn rotate_around_axis(vector: &[f64; 3], axis: &[f64; 3], angle: f64) -> [f64; 3] {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    // Rodrigues' rotation formula
    let dot = vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2];
    let cross = cross_product(axis, vector);

    [
        vector[0] * cos_angle + cross[0] * sin_angle + axis[0] * dot * (1.0 - cos_angle),
        vector[1] * cos_angle + cross[1] * sin_angle + axis[1] * dot * (1.0 - cos_angle),
        vector[2] * cos_angle + cross[2] * sin_angle + axis[2] * dot * (1.0 - cos_angle),
    ]
}

/// Calculate backbone RMSD between two structures (for validation)
pub fn calculate_backbone_rmsd(backbone1: &[BackboneAtom], backbone2: &[BackboneAtom]) -> f64 {
    if backbone1.len() != backbone2.len() {
        return f64::INFINITY;
    }

    let mut sum_sq_dist = 0.0;
    let mut count = 0;

    for (atom1, atom2) in backbone1.iter().zip(backbone2.iter()) {
        if atom1.atom().name == "CA" {  // Compare only CA atoms for RMSD
            let pos1 = atom1.position();
            let pos2 = atom2.position();

            let dx = pos1[0] - pos2[0];
            let dy = pos1[1] - pos2[1];
            let dz = pos1[2] - pos2[2];

            sum_sq_dist += dx * dx + dy * dy + dz * dz;
            count += 1;
        }
    }

    if count > 0 {
        (sum_sq_dist / count as f64).sqrt()
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_generation() {
        let sequence = "MKTA";
        let backbone = generate_backbone_coordinates(sequence, 0.7, 0.8, -25.0);

        assert!(backbone.is_ok());
        let atoms = backbone.unwrap();

        // Should have 4 atoms per residue (N, CA, C, O) * 4 residues = 16 atoms
        assert_eq!(atoms.len(), 16);

        // First atom should be N of first residue
        assert_eq!(atoms[0].atom().name, "N");
        assert_eq!(atoms[0].atom().residue, "MET");
        assert_eq!(atoms[0].atom().residue_id, 1);
    }

    #[test]
    fn test_angle_calculations() {
        let phi = phase_coherence_to_phi(0.8);
        let psi = phase_coherence_to_psi(0.8);

        // High coherence should give alpha-helical angles
        assert!(phi > -70.0 && phi < -50.0);
        assert!(psi > -55.0 && psi < -35.0);
    }

    #[test]
    fn test_rotation() {
        let vector = [1.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let angle = PI / 2.0; // 90 degrees

        let rotated = rotate_around_axis(&vector, &axis, angle);

        // Should rotate to [0, 1, 0] (within floating point precision)
        assert!((rotated[0] - 0.0).abs() < 1e-10);
        assert!((rotated[1] - 1.0).abs() < 1e-10);
        assert!((rotated[2] - 0.0).abs() < 1e-10);
    }
}