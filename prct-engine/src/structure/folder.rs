//! Main PRCT folder integrating all structure generation components

use super::*;
use super::backbone::generate_backbone_coordinates;
use super::sidechains::place_side_chains;
use super::sequences::{get_sequence, CASP_SEQUENCES};
use super::pdb_writer::center_structure;
use crate::PRCTResult;

/// Main PRCT structure generation engine
pub struct PRCTFolder {
    /// Enable debugging output
    pub debug: bool,
}

impl PRCTFolder {
    /// Create a new PRCT folder
    pub fn new() -> Self {
        Self {
            debug: false,
        }
    }

    /// Create new folder with debugging enabled
    pub fn with_debug() -> Self {
        Self {
            debug: true,
        }
    }

    /// Fold a protein sequence to 3D coordinates using PRCT algorithm
    pub fn fold_to_coordinates(
        &self,
        target_id: &str,
        phase_coherence: f64,
        chromatic_score: f64,
        hamiltonian_energy: f64,
        tsp_energy: f64,
    ) -> PRCTResult<Structure3D> {
        // Get sequence for target
        let sequence = get_sequence(target_id)
            .ok_or_else(|| crate::PRCTError::StructureGeneration(
                format!("Sequence not found for target: {}", target_id)
            ))?;

        if self.debug {
            println!("Folding target: {} (length: {})", target_id, sequence.len());
            println!("PRCT parameters: phase={:.3}, chromatic={:.3}, hamiltonian={:.1}, tsp={:.1}",
                     phase_coherence, chromatic_score, hamiltonian_energy, tsp_energy);
        }

        // Generate backbone coordinates using phase resonance
        let backbone_atoms = generate_backbone_coordinates(
            sequence,
            phase_coherence,
            chromatic_score,
            hamiltonian_energy,
        )?;

        if self.debug {
            println!("Generated {} backbone atoms", backbone_atoms.len());
        }

        // Place side chains using Hamiltonian energy guidance
        let all_atoms = place_side_chains(
            &backbone_atoms,
            sequence,
            hamiltonian_energy,
            tsp_energy,
        )?;

        if self.debug {
            println!("Generated {} total atoms", all_atoms.len());
        }

        // Calculate prediction confidence from PRCT metrics
        let confidence = calculate_prediction_confidence(
            phase_coherence,
            chromatic_score,
            hamiltonian_energy,
            sequence.len(),
        );

        // Create 3D structure
        let mut structure = Structure3D::new(
            all_atoms,
            target_id.to_string(),
            confidence,
            hamiltonian_energy + tsp_energy,
        );

        // Center structure at origin
        center_structure(&mut structure);

        Ok(structure)
    }

    /// Fold all CASP sequences using provided PRCT metrics
    pub fn fold_all_casp_targets(
        &self,
        prct_results: &[(String, f64, f64, f64, f64)], // (target_id, phase_coherence, chromatic_score, hamiltonian_energy, tsp_energy)
    ) -> PRCTResult<Vec<Structure3D>> {
        let mut structures = Vec::new();

        for (target_id, phase_coherence, chromatic_score, hamiltonian_energy, tsp_energy) in prct_results {
            if self.debug {
                println!("\nProcessing target: {}", target_id);
            }

            match self.fold_to_coordinates(
                target_id,
                *phase_coherence,
                *chromatic_score,
                *hamiltonian_energy,
                *tsp_energy,
            ) {
                Ok(structure) => {
                    if self.debug {
                        println!("Successfully folded {} with {} atoms",
                                target_id, structure.atom_count());
                    }
                    structures.push(structure);
                }
                Err(e) => {
                    eprintln!("Failed to fold target {}: {}", target_id, e);
                    // Continue with other targets
                }
            }
        }

        if self.debug {
            println!("\nSuccessfully folded {}/{} targets",
                     structures.len(), prct_results.len());
        }

        Ok(structures)
    }

    /// Generate structures from validation results JSON
    pub fn fold_from_validation_results(
        &self,
        validation_results: &serde_json::Value,
    ) -> PRCTResult<Vec<Structure3D>> {
        let target_predictions = validation_results
            .get("target_predictions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| crate::PRCTError::StructureGeneration(
                "Invalid validation results format".to_string()
            ))?;

        let mut prct_results = Vec::new();

        for prediction in target_predictions {
            let target_id = prediction
                .get("target_id")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();

            let phase_coherence = prediction
                .get("phase_coherence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            let chromatic_score = prediction
                .get("chromatic_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            let hamiltonian_energy = prediction
                .get("hamiltonian_energy")
                .and_then(|v| v.as_f64())
                .unwrap_or(-20.0);

            let tsp_energy = prediction
                .get("tsp_energy")
                .and_then(|v| v.as_f64())
                .unwrap_or(-10.0);

            prct_results.push((
                target_id,
                phase_coherence,
                chromatic_score,
                hamiltonian_energy,
                tsp_energy,
            ));
        }

        self.fold_all_casp_targets(&prct_results)
    }

    /// Generate ensemble of structures with parameter variations
    pub fn fold_ensemble(
        &self,
        target_id: &str,
        base_phase_coherence: f64,
        base_chromatic_score: f64,
        base_hamiltonian_energy: f64,
        base_tsp_energy: f64,
        ensemble_size: usize,
    ) -> PRCTResult<Vec<Structure3D>> {
        let mut structures = Vec::new();

        for i in 0..ensemble_size {
            // Add small variations to parameters for ensemble diversity
            let variation = (i as f64) / (ensemble_size as f64);
            let phase_variation = 0.1 * (variation * 2.0 - 1.0); // ±0.1
            let chromatic_variation = 0.1 * (variation * 2.0 - 1.0);
            let energy_variation = 2.0 * (variation * 2.0 - 1.0); // ±2.0 kcal/mol

            let varied_phase = (base_phase_coherence + phase_variation).clamp(0.0_f64, 1.0_f64);
            let varied_chromatic = (base_chromatic_score + chromatic_variation).clamp(0.0_f64, 1.0_f64);
            let varied_hamiltonian = base_hamiltonian_energy + energy_variation;
            let varied_tsp = base_tsp_energy + energy_variation * 0.5;

            match self.fold_to_coordinates(
                target_id,
                varied_phase,
                varied_chromatic,
                varied_hamiltonian,
                varied_tsp,
            ) {
                Ok(structure) => structures.push(structure),
                Err(e) => {
                    if self.debug {
                        println!("Failed ensemble member {}: {}", i, e);
                    }
                }
            }
        }

        Ok(structures)
    }

    /// Validate generated structure quality
    pub fn validate_structure(&self, structure: &Structure3D) -> StructureQuality {
        let mut quality = StructureQuality::new();

        // Check atom counts
        quality.total_atoms = structure.atom_count();
        quality.backbone_atoms = structure.backbone_atoms().len();
        quality.sidechain_atoms = structure.sidechain_atoms().len();

        // Check for reasonable coordinate ranges
        let (min_coords, max_coords) = calculate_coordinate_bounds(&structure.atoms);
        quality.coordinate_range = [
            max_coords[0] - min_coords[0],
            max_coords[1] - min_coords[1],
            max_coords[2] - min_coords[2],
        ];

        // Check for reasonable bond lengths
        quality.bond_statistics = analyze_bond_lengths(&structure.atoms);

        // Overall quality score
        quality.overall_score = calculate_overall_quality_score(&quality);

        quality
    }
}

impl Default for PRCTFolder {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate prediction confidence from PRCT parameters
fn calculate_prediction_confidence(
    phase_coherence: f64,
    chromatic_score: f64,
    hamiltonian_energy: f64,
    sequence_length: usize,
) -> f64 {
    // Higher phase coherence and chromatic score indicate better prediction
    let base_confidence = (phase_coherence + chromatic_score) / 2.0;

    // More negative energy generally indicates more stable structure
    let energy_factor = if hamiltonian_energy < 0.0 {
        ((-hamiltonian_energy) / 50.0).min(0.3)
    } else {
        0.0
    };

    // Longer sequences are generally harder to predict accurately
    let length_penalty = (sequence_length as f64 / 500.0).min(0.2);

    (base_confidence + energy_factor - length_penalty).clamp(0.0_f64, 1.0_f64)
}

/// Structure quality assessment
#[derive(Debug, Clone)]
pub struct StructureQuality {
    pub total_atoms: usize,
    pub backbone_atoms: usize,
    pub sidechain_atoms: usize,
    pub coordinate_range: [f64; 3],
    pub bond_statistics: BondStatistics,
    pub overall_score: f64,
}

impl StructureQuality {
    fn new() -> Self {
        Self {
            total_atoms: 0,
            backbone_atoms: 0,
            sidechain_atoms: 0,
            coordinate_range: [0.0, 0.0, 0.0],
            bond_statistics: BondStatistics::default(),
            overall_score: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BondStatistics {
    pub mean_bond_length: f64,
    pub bond_length_std: f64,
    pub unusual_bonds: usize,
}

/// Calculate coordinate bounds
fn calculate_coordinate_bounds(atoms: &[Atom]) -> ([f64; 3], [f64; 3]) {
    if atoms.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let mut min_coords = [f64::INFINITY; 3];
    let mut max_coords = [f64::NEG_INFINITY; 3];

    for atom in atoms {
        min_coords[0] = min_coords[0].min(atom.x);
        min_coords[1] = min_coords[1].min(atom.y);
        min_coords[2] = min_coords[2].min(atom.z);

        max_coords[0] = max_coords[0].max(atom.x);
        max_coords[1] = max_coords[1].max(atom.y);
        max_coords[2] = max_coords[2].max(atom.z);
    }

    (min_coords, max_coords)
}

/// Analyze bond lengths in structure
fn analyze_bond_lengths(atoms: &[Atom]) -> BondStatistics {
    let mut bond_lengths = Vec::new();
    let mut unusual_bonds = 0;

    // Check consecutive backbone atoms
    for i in 0..atoms.len().saturating_sub(1) {
        let atom1 = &atoms[i];
        let atom2 = &atoms[i + 1];

        // Only check bonds within same residue or consecutive residues
        if (atom1.residue_id == atom2.residue_id) ||
           (atom2.residue_id == atom1.residue_id + 1 && atom1.name == "C" && atom2.name == "N") {
            let distance = atom1.distance_to(atom2);
            bond_lengths.push(distance);

            // Flag unusual bond lengths (< 0.8Å or > 2.5Å)
            if distance < 0.8 || distance > 2.5 {
                unusual_bonds += 1;
            }
        }
    }

    let mean_bond_length = if !bond_lengths.is_empty() {
        bond_lengths.iter().sum::<f64>() / bond_lengths.len() as f64
    } else {
        0.0
    };

    let bond_length_std = if bond_lengths.len() > 1 {
        let variance = bond_lengths.iter()
            .map(|&length| (length - mean_bond_length).powi(2))
            .sum::<f64>() / bond_lengths.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };

    BondStatistics {
        mean_bond_length,
        bond_length_std,
        unusual_bonds,
    }
}

/// Calculate overall structure quality score
fn calculate_overall_quality_score(quality: &StructureQuality) -> f64 {
    let mut score = 1.0_f64;

    // Penalize unreasonable coordinate ranges
    let max_range = quality.coordinate_range.iter().fold(0.0f64, |a, &b| a.max(b));
    if max_range > 200.0 {  // Unusually extended structure
        score *= 0.8;
    }
    if max_range < 10.0 {   // Unusually compact structure
        score *= 0.9;
    }

    // Penalize unusual bond statistics
    if quality.bond_statistics.mean_bond_length < 1.0 || quality.bond_statistics.mean_bond_length > 2.0 {
        score *= 0.7;
    }

    if quality.bond_statistics.unusual_bonds > quality.total_atoms / 20 {
        score *= 0.6; // More than 5% unusual bonds
    }

    score.clamp(0.0_f64, 1.0_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prct_folder_creation() {
        let folder = PRCTFolder::new();
        assert!(!folder.debug);

        let debug_folder = PRCTFolder::with_debug();
        assert!(debug_folder.debug);
    }

    #[test]
    fn test_confidence_calculation() {
        let confidence = calculate_prediction_confidence(0.8, 0.7, -25.0, 100);
        assert!(confidence > 0.5 && confidence <= 1.0);

        // Test with poor parameters
        let low_confidence = calculate_prediction_confidence(0.2, 0.3, 10.0, 500);
        assert!(low_confidence < confidence);
    }

    #[test]
    fn test_coordinate_bounds() {
        let atoms = vec![
            Atom::backbone("CA".to_string(), "ALA".to_string(), 1,
                          0.0, 0.0, 0.0, Element::Carbon),
            Atom::backbone("CA".to_string(), "ALA".to_string(), 2,
                          5.0, 3.0, -2.0, Element::Carbon),
        ];

        let (min_coords, max_coords) = calculate_coordinate_bounds(&atoms);
        assert_eq!(min_coords, [0.0, 0.0, -2.0]);
        assert_eq!(max_coords, [5.0, 3.0, 0.0]);
    }
}