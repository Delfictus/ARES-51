//! Side chain placement using PRCT Hamiltonian energy guidance

use super::atom::{Atom, BackboneAtom, Element};
use super::sequences::{one_to_three, is_hydrophobic, is_charged, is_polar};
use super::{bond_lengths, bond_angles, deg_to_rad, cross_product, normalize_vector};
use crate::PRCTResult;
use std::f64::consts::PI;

/// Amino acid side chain definitions and rotamer libraries
#[derive(Debug, Clone, PartialEq)]
pub enum AminoAcid {
    Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile,
    Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val,
}

impl AminoAcid {
    pub fn from_char(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'A' => AminoAcid::Ala, 'R' => AminoAcid::Arg, 'N' => AminoAcid::Asn,
            'D' => AminoAcid::Asp, 'C' => AminoAcid::Cys, 'Q' => AminoAcid::Gln,
            'E' => AminoAcid::Glu, 'G' => AminoAcid::Gly, 'H' => AminoAcid::His,
            'I' => AminoAcid::Ile, 'L' => AminoAcid::Leu, 'K' => AminoAcid::Lys,
            'M' => AminoAcid::Met, 'F' => AminoAcid::Phe, 'P' => AminoAcid::Pro,
            'S' => AminoAcid::Ser, 'T' => AminoAcid::Thr, 'W' => AminoAcid::Trp,
            'Y' => AminoAcid::Tyr, 'V' => AminoAcid::Val,
            _ => AminoAcid::Ala,
        }
    }

    pub fn three_letter_code(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "ALA", AminoAcid::Arg => "ARG", AminoAcid::Asn => "ASN",
            AminoAcid::Asp => "ASP", AminoAcid::Cys => "CYS", AminoAcid::Gln => "GLN",
            AminoAcid::Glu => "GLU", AminoAcid::Gly => "GLY", AminoAcid::His => "HIS",
            AminoAcid::Ile => "ILE", AminoAcid::Leu => "LEU", AminoAcid::Lys => "LYS",
            AminoAcid::Met => "MET", AminoAcid::Phe => "PHE", AminoAcid::Pro => "PRO",
            AminoAcid::Ser => "SER", AminoAcid::Thr => "THR", AminoAcid::Trp => "TRP",
            AminoAcid::Tyr => "TYR", AminoAcid::Val => "VAL",
        }
    }

    pub fn has_cb(&self) -> bool {
        !matches!(self, AminoAcid::Gly)
    }

    pub fn side_chain_atoms(&self) -> Vec<&'static str> {
        match self {
            AminoAcid::Ala => vec!["CB"],
            AminoAcid::Arg => vec!["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
            AminoAcid::Asn => vec!["CB", "CG", "OD1", "ND2"],
            AminoAcid::Asp => vec!["CB", "CG", "OD1", "OD2"],
            AminoAcid::Cys => vec!["CB", "SG"],
            AminoAcid::Gln => vec!["CB", "CG", "CD", "OE1", "NE2"],
            AminoAcid::Glu => vec!["CB", "CG", "CD", "OE1", "OE2"],
            AminoAcid::Gly => vec![], // No side chain
            AminoAcid::His => vec!["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
            AminoAcid::Ile => vec!["CB", "CG1", "CG2", "CD1"],
            AminoAcid::Leu => vec!["CB", "CG", "CD1", "CD2"],
            AminoAcid::Lys => vec!["CB", "CG", "CD", "CE", "NZ"],
            AminoAcid::Met => vec!["CB", "CG", "SD", "CE"],
            AminoAcid::Phe => vec!["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            AminoAcid::Pro => vec!["CB", "CG", "CD"],
            AminoAcid::Ser => vec!["CB", "OG"],
            AminoAcid::Thr => vec!["CB", "OG1", "CG2"],
            AminoAcid::Trp => vec!["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            AminoAcid::Tyr => vec!["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
            AminoAcid::Val => vec!["CB", "CG1", "CG2"],
        }
    }
}

/// Place side chains on backbone using PRCT Hamiltonian energy guidance
pub fn place_side_chains(
    backbone: &[BackboneAtom],
    sequence: &str,
    hamiltonian_energy: f64,
    tsp_energy: f64,
) -> PRCTResult<Vec<Atom>> {
    let mut all_atoms = Vec::new();

    // Add all backbone atoms first
    for backbone_atom in backbone {
        all_atoms.push(backbone_atom.atom().clone());
    }

    // Extract CA atoms for side chain placement
    let ca_atoms: Vec<&BackboneAtom> = backbone.iter()
        .filter(|atom| atom.atom().name == "CA")
        .collect();

    if ca_atoms.len() != sequence.len() {
        return Err(crate::PRCTError::StructureGeneration(
            "Mismatch between CA atoms and sequence length".to_string()
        ));
    }

    // PRCT energy-guided parameters
    let energy_compactness = hamiltonian_to_compactness(hamiltonian_energy);
    let tsp_ordering = tsp_to_rotamer_preference(tsp_energy);

    // Place side chains for each residue
    for (i, (ca_atom, aa_char)) in ca_atoms.iter().zip(sequence.chars()).enumerate() {
        let amino_acid = AminoAcid::from_char(aa_char);
        let residue_id = ca_atom.atom().residue_id;
        let residue_name = amino_acid.three_letter_code();

        if !amino_acid.has_cb() {
            continue; // Skip glycine (no side chain)
        }

        // Get neighboring CA positions for environmental context
        let (prev_ca, next_ca) = get_neighboring_ca_positions(&ca_atoms, i);

        // Place side chain atoms using PRCT guidance
        let side_chain_atoms = place_amino_acid_side_chain(
            ca_atom,
            prev_ca.as_ref(),
            next_ca.as_ref(),
            &amino_acid,
            residue_id,
            residue_name,
            energy_compactness,
            tsp_ordering,
            i,
            sequence.len(),
        )?;

        all_atoms.extend(side_chain_atoms);
    }

    Ok(all_atoms)
}

/// Convert Hamiltonian energy to compactness factor for side chain placement
fn hamiltonian_to_compactness(hamiltonian_energy: f64) -> f64 {
    // More negative energy -> more compact side chains
    let normalized = (-hamiltonian_energy / 30.0).min(2.0).max(0.0);
    0.7 + 0.3 * normalized // Range [0.7, 1.0]
}

/// Convert TSP energy to rotamer preference
fn tsp_to_rotamer_preference(tsp_energy: f64) -> f64 {
    // TSP energy guides rotamer selection
    let normalized = (-tsp_energy / 15.0).min(1.0).max(0.0);
    normalized // Range [0.0, 1.0]
}

/// Get neighboring CA atom positions for environmental context
fn get_neighboring_ca_positions(
    ca_atoms: &[&BackboneAtom],
    index: usize,
) -> (Option<[f64; 3]>, Option<[f64; 3]>) {
    let prev_ca = if index > 0 {
        Some(ca_atoms[index - 1].position())
    } else {
        None
    };

    let next_ca = if index < ca_atoms.len() - 1 {
        Some(ca_atoms[index + 1].position())
    } else {
        None
    };

    (prev_ca, next_ca)
}

/// Place side chain atoms for a specific amino acid
fn place_amino_acid_side_chain(
    ca_atom: &BackboneAtom,
    prev_ca: Option<&[f64; 3]>,
    next_ca: Option<&[f64; 3]>,
    amino_acid: &AminoAcid,
    residue_id: i32,
    residue_name: &str,
    compactness: f64,
    rotamer_pref: f64,
    position: usize,
    total_residues: usize,
) -> PRCTResult<Vec<Atom>> {
    let mut side_chain_atoms = Vec::new();
    let ca_pos = ca_atom.position();

    // Calculate local coordinate frame
    let backbone_direction = calculate_backbone_direction(&ca_pos, prev_ca, next_ca);
    let perpendicular = calculate_perpendicular_direction(&backbone_direction);

    // Place CB atom first (common to all amino acids except glycine)
    let cb_pos = place_cb_atom(&ca_pos, &backbone_direction, &perpendicular, compactness);
    let cb_atom = Atom::sidechain(
        "CB".to_string(),
        residue_name.to_string(),
        residue_id,
        cb_pos[0], cb_pos[1], cb_pos[2],
        Element::Carbon,
    );
    side_chain_atoms.push(cb_atom);

    // Place remaining side chain atoms based on amino acid type
    match amino_acid {
        AminoAcid::Ala => {
            // Only CB - already placed
        },
        AminoAcid::Ser => {
            let og_pos = place_ser_og(&cb_pos, &ca_pos, rotamer_pref);
            side_chain_atoms.push(Atom::sidechain(
                "OG".to_string(), residue_name.to_string(), residue_id,
                og_pos[0], og_pos[1], og_pos[2], Element::Oxygen,
            ));
        },
        AminoAcid::Thr => {
            place_thr_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref);
        },
        AminoAcid::Cys => {
            let sg_pos = place_cys_sg(&cb_pos, &ca_pos, rotamer_pref);
            side_chain_atoms.push(Atom::sidechain(
                "SG".to_string(), residue_name.to_string(), residue_id,
                sg_pos[0], sg_pos[1], sg_pos[2], Element::Sulfur,
            ));
        },
        AminoAcid::Val => {
            place_val_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref);
        },
        AminoAcid::Leu => {
            place_leu_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref, compactness);
        },
        AminoAcid::Ile => {
            place_ile_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref);
        },
        AminoAcid::Met => {
            place_met_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref, compactness);
        },
        AminoAcid::Phe => {
            place_phe_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref, compactness);
        },
        AminoAcid::Tyr => {
            place_tyr_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref, compactness);
        },
        AminoAcid::Trp => {
            place_trp_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, residue_name, residue_id, rotamer_pref, compactness);
        },
        _ => {
            // For complex amino acids, use simplified placement
            place_generic_side_chain(&mut side_chain_atoms, &cb_pos, &ca_pos, amino_acid, residue_name, residue_id, rotamer_pref, compactness);
        }
    }

    Ok(side_chain_atoms)
}

/// Calculate backbone direction vector
fn calculate_backbone_direction(
    ca_pos: &[f64; 3],
    prev_ca: Option<&[f64; 3]>,
    next_ca: Option<&[f64; 3]>,
) -> [f64; 3] {
    let mut direction = match (prev_ca, next_ca) {
        (Some(prev), Some(next)) => {
            // Use average direction
            [
                (next[0] - prev[0]) / 2.0,
                (next[1] - prev[1]) / 2.0,
                (next[2] - prev[2]) / 2.0,
            ]
        },
        (Some(prev), None) => {
            // Use direction from previous
            [
                ca_pos[0] - prev[0],
                ca_pos[1] - prev[1],
                ca_pos[2] - prev[2],
            ]
        },
        (None, Some(next)) => {
            // Use direction to next
            [
                next[0] - ca_pos[0],
                next[1] - ca_pos[1],
                next[2] - ca_pos[2],
            ]
        },
        (None, None) => {
            // Default direction
            [1.0, 0.0, 0.0]
        }
    };

    normalize_vector(&mut direction);
    direction
}

/// Calculate perpendicular direction for side chain placement
fn calculate_perpendicular_direction(backbone_direction: &[f64; 3]) -> [f64; 3] {
    let up = [0.0, 0.0, 1.0];
    let mut perpendicular = cross_product(backbone_direction, &up);
    normalize_vector(&mut perpendicular);
    perpendicular
}

/// Place CB atom using tetrahedral geometry
fn place_cb_atom(
    ca_pos: &[f64; 3],
    backbone_dir: &[f64; 3],
    perpendicular: &[f64; 3],
    compactness: f64,
) -> [f64; 3] {
    // Tetrahedral angle ~109.5°
    let tetrahedral_angle = deg_to_rad(109.5);
    let cb_distance = bond_lengths::CA_CB * compactness;

    // Direction for CB placement
    let cb_direction = [
        -backbone_dir[0] * tetrahedral_angle.cos() + perpendicular[0] * tetrahedral_angle.sin(),
        -backbone_dir[1] * tetrahedral_angle.cos() + perpendicular[1] * tetrahedral_angle.sin(),
        -backbone_dir[2] * tetrahedral_angle.cos() + perpendicular[2] * tetrahedral_angle.sin(),
    ];

    [
        ca_pos[0] + cb_direction[0] * cb_distance,
        ca_pos[1] + cb_direction[1] * cb_distance,
        ca_pos[2] + cb_direction[2] * cb_distance,
    ]
}

/// Place serine OG atom
fn place_ser_og(cb_pos: &[f64; 3], ca_pos: &[f64; 3], rotamer_pref: f64) -> [f64; 3] {
    let cb_ca_vec = [
        ca_pos[0] - cb_pos[0],
        ca_pos[1] - cb_pos[1],
        ca_pos[2] - cb_pos[2],
    ];

    // Rotate around CB-CA bond based on rotamer preference
    let chi_angle = deg_to_rad(-60.0 + 120.0 * rotamer_pref); // Range: -60° to +60°
    let rotated_direction = rotate_around_axis(&[1.0, 0.0, 0.0], &cb_ca_vec, chi_angle);

    [
        cb_pos[0] + rotated_direction[0] * 1.41, // C-O bond length
        cb_pos[1] + rotated_direction[1] * 1.41,
        cb_pos[2] + rotated_direction[2] * 1.41,
    ]
}

/// Simplified implementations for other amino acids
fn place_thr_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64) {
    // OG1
    let og1_pos = [cb_pos[0] + 1.41, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("OG1".to_string(), residue_name.to_string(), residue_id,
        og1_pos[0], og1_pos[1], og1_pos[2], Element::Oxygen));

    // CG2
    let cg2_pos = [cb_pos[0], cb_pos[1] + 1.54, cb_pos[2]];
    atoms.push(Atom::sidechain("CG2".to_string(), residue_name.to_string(), residue_id,
        cg2_pos[0], cg2_pos[1], cg2_pos[2], Element::Carbon));
}

fn place_cys_sg(cb_pos: &[f64; 3], _ca_pos: &[f64; 3], rotamer_pref: f64) -> [f64; 3] {
    [
        cb_pos[0] + 1.82 * rotamer_pref.cos(), // C-S bond length
        cb_pos[1] + 1.82 * rotamer_pref.sin(),
        cb_pos[2],
    ]
}

fn place_val_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64) {
    let angle1 = deg_to_rad(109.5);
    let angle2 = deg_to_rad(-109.5);

    let cg1_pos = [
        cb_pos[0] + 1.54 * angle1.cos(),
        cb_pos[1] + 1.54 * angle1.sin(),
        cb_pos[2],
    ];

    let cg2_pos = [
        cb_pos[0] + 1.54 * angle2.cos(),
        cb_pos[1] + 1.54 * angle2.sin(),
        cb_pos[2],
    ];

    atoms.push(Atom::sidechain("CG1".to_string(), residue_name.to_string(), residue_id,
        cg1_pos[0], cg1_pos[1], cg1_pos[2], Element::Carbon));
    atoms.push(Atom::sidechain("CG2".to_string(), residue_name.to_string(), residue_id,
        cg2_pos[0], cg2_pos[1], cg2_pos[2], Element::Carbon));
}

// Placeholder implementations for complex amino acids
fn place_leu_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

fn place_ile_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64) {
    let cg1_pos = [cb_pos[0] + 1.54, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG1".to_string(), residue_name.to_string(), residue_id,
        cg1_pos[0], cg1_pos[1], cg1_pos[2], Element::Carbon));
}

fn place_met_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

fn place_phe_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

fn place_tyr_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

fn place_trp_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

fn place_generic_side_chain(atoms: &mut Vec<Atom>, cb_pos: &[f64; 3], _ca_pos: &[f64; 3], amino_acid: &AminoAcid, residue_name: &str, residue_id: i32, rotamer_pref: f64, compactness: f64) {
    // Place a generic CG atom for complex amino acids
    let cg_pos = [cb_pos[0] + 1.54 * compactness, cb_pos[1], cb_pos[2]];
    atoms.push(Atom::sidechain("CG".to_string(), residue_name.to_string(), residue_id,
        cg_pos[0], cg_pos[1], cg_pos[2], Element::Carbon));
}

/// Rotate vector around axis by angle (simplified implementation)
fn rotate_around_axis(vector: &[f64; 3], axis: &[f64; 3], angle: f64) -> [f64; 3] {
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    // Simplified rotation (assumes normalized axis)
    [
        vector[0] * cos_angle + axis[0] * sin_angle,
        vector[1] * cos_angle + axis[1] * sin_angle,
        vector[2] * cos_angle + axis[2] * sin_angle,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::atom::Element;

    #[test]
    fn test_amino_acid_classification() {
        let ala = AminoAcid::from_char('A');
        assert_eq!(ala, AminoAcid::Ala);
        assert_eq!(ala.three_letter_code(), "ALA");
        assert!(ala.has_cb());

        let gly = AminoAcid::from_char('G');
        assert_eq!(gly, AminoAcid::Gly);
        assert!(!gly.has_cb());
    }

    #[test]
    fn test_side_chain_atoms() {
        let ser = AminoAcid::Ser;
        let atoms = ser.side_chain_atoms();
        assert_eq!(atoms, vec!["CB", "OG"]);

        let phe = AminoAcid::Phe;
        let phe_atoms = phe.side_chain_atoms();
        assert!(phe_atoms.contains(&"CB"));
        assert!(phe_atoms.contains(&"CG"));
    }
}