//! PDB file format writer for protein structures

use super::atom::Atom;
use super::Structure3D;
use crate::PRCTResult;
use std::fmt::Write;
use chrono::Utc;

/// Write atoms to PDB format string
pub fn write_pdb_file(atoms: &[Atom], target_id: &str, confidence: f64) -> PRCTResult<String> {
    let mut pdb_content = String::new();

    // Write header
    write_pdb_header(&mut pdb_content, target_id, confidence)?;

    // Write atoms
    for (i, atom) in atoms.iter().enumerate() {
        write_atom_record(&mut pdb_content, atom, i + 1)?;
    }

    // Write footer
    write_pdb_footer(&mut pdb_content)?;

    Ok(pdb_content)
}

/// Write complete Structure3D to PDB format
pub fn write_pdb_structure(structure: &Structure3D) -> PRCTResult<String> {
    write_pdb_file(&structure.atoms, &structure.target_id, structure.confidence)
}

/// Write PDB header with metadata
fn write_pdb_header(content: &mut String, target_id: &str, confidence: f64) -> PRCTResult<()> {
    let now = Utc::now();
    let date = now.format("%d-%b-%y").to_string().to_uppercase();

    writeln!(content, "HEADER    PROTEIN FOLDING/PRCT ALGORITHM        {:<9} {:<4}",
             date, target_id).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB header write error: {}", e)))?;

    writeln!(content, "TITLE     PRCT ALGORITHM STRUCTURE PREDICTION FOR TARGET {}",
             target_id).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB title write error: {}", e)))?;

    writeln!(content, "COMPND    MOL_ID: 1;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB compnd write error: {}", e)))?;

    writeln!(content, "COMPND   2 MOLECULE: PROTEIN;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB compnd write error: {}", e)))?;

    writeln!(content, "COMPND   3 CHAIN: A;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB compnd write error: {}", e)))?;

    writeln!(content, "SOURCE    MOL_ID: 1;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB source write error: {}", e)))?;

    writeln!(content, "SOURCE   2 ORGANISM_SCIENTIFIC: SYNTHETIC CONSTRUCT;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB source write error: {}", e)))?;

    writeln!(content, "KEYWDS    PRCT ALGORITHM, PHASE RESONANCE, PROTEIN FOLDING").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB keywds write error: {}", e)))?;

    writeln!(content, "EXPDTA    THEORETICAL MODEL").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB expdta write error: {}", e)))?;

    writeln!(content, "MODEL        1").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB model write error: {}", e)))?;

    // Add confidence as REMARK
    writeln!(content, "REMARK   3 PRCT CONFIDENCE SCORE: {:.4}", confidence).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB remark write error: {}", e)))?;

    writeln!(content, "REMARK   3 ALGORITHM: PRCT (PHASE RESONANCE CHROMATIC TSP)").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB remark write error: {}", e)))?;

    Ok(())
}

/// Write individual atom record in PDB format
fn write_atom_record(content: &mut String, atom: &Atom, serial_number: usize) -> PRCTResult<()> {
    // PDB ATOM record format:
    // ATOM      1  N   ALA A   1      20.154  16.967  23.421  1.00 20.00           N
    // Columns:  1-6     7-11 13-16   17 18-21    31-38  39-46  47-54 55-60 61-66       77-78
    //          ATOM     serial name   alt residue  x      y      z     occ   b-fact     element

    let record_type = "ATOM";
    let serial = format!("{:5}", serial_number.min(99999));
    let atom_name = format!("{:>4}", atom.name); // Right-aligned in 4-char field
    let alt_loc = " ";
    let residue_name = format!("{:>3}", atom.residue);
    let chain_id = atom.chain_id;
    let residue_seq = format!("{:4}", atom.residue_id.min(9999));
    let insertion_code = " ";
    let x = format!("{:8.3}", atom.x);
    let y = format!("{:8.3}", atom.y);
    let z = format!("{:8.3}", atom.z);
    let occupancy = format!("{:6.2}", atom.occupancy);
    let temp_factor = format!("{:6.2}", atom.b_factor);
    let element_symbol = format!("{:>2}", atom.element.symbol());

    writeln!(content,
        "{:<6}{}{} {}{}{}{:<1}{}   {}{}{}{}{}{}{}",
        record_type,      // 1-6
        serial,          // 7-11
        atom_name,       // 12-16
        alt_loc,         // 17
        residue_name,    // 18-20
        chain_id,        // 21-21
        residue_seq,     // 22-25
        insertion_code,  // 26
        x,              // 30-37
        y,              // 38-45
        z,              // 46-53
        occupancy,      // 54-59
        temp_factor,    // 60-65
        "",             // 66-71 (segment identifier - unused)
        element_symbol  // 76-77
    ).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB atom write error: {}", e)))?;

    Ok(())
}

/// Write PDB footer
fn write_pdb_footer(content: &mut String) -> PRCTResult<()> {
    writeln!(content, "TER").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB ter write error: {}", e)))?;
    writeln!(content, "ENDMDL").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB endmdl write error: {}", e)))?;
    writeln!(content, "END").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB end write error: {}", e)))?;

    Ok(())
}

/// Write multiple structures as multi-model PDB file
pub fn write_multi_model_pdb(structures: &[Structure3D]) -> PRCTResult<String> {
    let mut pdb_content = String::new();

    if structures.is_empty() {
        return Err(crate::PRCTError::StructureGeneration(
            "No structures provided for multi-model PDB".to_string()
        ));
    }

    // Use first structure for header info
    let first_structure = &structures[0];
    let now = Utc::now();
    let date = now.format("%d-%b-%y").to_string().to_uppercase();

    writeln!(pdb_content, "HEADER    PROTEIN FOLDING/PRCT ALGORITHM        {:<9} {:<4}",
             date, first_structure.target_id).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB header write error: {}", e)))?;

    writeln!(pdb_content, "TITLE     PRCT ALGORITHM ENSEMBLE PREDICTIONS").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB title write error: {}", e)))?;

    writeln!(pdb_content, "COMPND    MOL_ID: 1;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB compnd write error: {}", e)))?;

    writeln!(pdb_content, "COMPND   2 MOLECULE: PROTEIN;").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB compnd write error: {}", e)))?;

    writeln!(pdb_content, "REMARK   3 ENSEMBLE OF {} PRCT PREDICTIONS", structures.len()).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB remark write error: {}", e)))?;

    // Write each model
    for (model_num, structure) in structures.iter().enumerate() {
        writeln!(pdb_content, "MODEL     {:4}", model_num + 1).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB model write error: {}", e)))?;

        writeln!(pdb_content, "REMARK   3 MODEL {} CONFIDENCE: {:.4}",
                model_num + 1, structure.confidence).map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB remark write error: {}", e)))?;

        // Write atoms for this model
        for (i, atom) in structure.atoms.iter().enumerate() {
            write_atom_record(&mut pdb_content, atom, i + 1)?;
        }

        writeln!(pdb_content, "TER").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB ter write error: {}", e)))?;
        writeln!(pdb_content, "ENDMDL").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB endmdl write error: {}", e)))?;
    }

    writeln!(pdb_content, "END").map_err(|e| crate::PRCTError::StructureGeneration(format!("PDB end write error: {}", e)))?;

    Ok(pdb_content)
}

/// Write structure with CASP-specific formatting
pub fn write_casp_prediction(
    structure: &Structure3D,
    group_name: &str,
    method: &str,
) -> PRCTResult<String> {
    let mut content = String::new();

    // CASP-specific header
    writeln!(content, "PFRMAT TS").map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP header write error: {}", e)))?;

    writeln!(content, "TARGET {}", structure.target_id).map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP target write error: {}", e)))?;

    writeln!(content, "AUTHOR {}", group_name).map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP author write error: {}", e)))?;

    writeln!(content, "METHOD {}", method).map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP method write error: {}", e)))?;

    writeln!(content, "MODEL  1").map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP model write error: {}", e)))?;

    // Write atoms
    for (i, atom) in structure.atoms.iter().enumerate() {
        write_atom_record(&mut content, atom, i + 1)?;
    }

    writeln!(content, "TER").map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP ter write error: {}", e)))?;
    writeln!(content, "END").map_err(|e| crate::PRCTError::StructureGeneration(format!("CASP end write error: {}", e)))?;

    Ok(content)
}

/// Calculate center of mass for structure positioning
pub fn calculate_center_of_mass(atoms: &[Atom]) -> [f64; 3] {
    if atoms.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut center = [0.0, 0.0, 0.0];
    for atom in atoms {
        center[0] += atom.x;
        center[1] += atom.y;
        center[2] += atom.z;
    }

    let n = atoms.len() as f64;
    [center[0] / n, center[1] / n, center[2] / n]
}

/// Center structure at origin
pub fn center_structure(structure: &mut Structure3D) {
    let center = calculate_center_of_mass(&structure.atoms);

    for atom in &mut structure.atoms {
        atom.x -= center[0];
        atom.y -= center[1];
        atom.z -= center[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::atom::{Atom, Element};

    #[test]
    fn test_pdb_atom_writing() {
        let atom = Atom::backbone(
            "CA".to_string(),
            "ALA".to_string(),
            1,
            1.234, 5.678, 9.012,
            Element::Carbon,
        );

        let mut content = String::new();
        let result = write_atom_record(&mut content, &atom, 1);
        assert!(result.is_ok());

        // Check that the line contains expected values
        assert!(content.contains("ATOM"));
        assert!(content.contains("CA"));
        assert!(content.contains("ALA"));
        assert!(content.contains("1.234"));
        assert!(content.contains("5.678"));
        assert!(content.contains("9.012"));
    }

    #[test]
    fn test_pdb_structure_writing() {
        let atoms = vec![
            Atom::backbone(
                "N".to_string(), "ALA".to_string(), 1,
                0.0, 0.0, 0.0, Element::Nitrogen,
            ),
            Atom::backbone(
                "CA".to_string(), "ALA".to_string(), 1,
                1.0, 1.0, 1.0, Element::Carbon,
            ),
        ];

        let structure = Structure3D::new(atoms, "T1000".to_string(), 0.85, -25.0);
        let pdb_content = write_pdb_structure(&structure);

        assert!(pdb_content.is_ok());
        let content = pdb_content.unwrap();

        assert!(content.contains("HEADER"));
        assert!(content.contains("T1000"));
        assert!(content.contains("ATOM"));
        assert!(content.contains("END"));
    }

    #[test]
    fn test_center_calculation() {
        let atoms = vec![
            Atom::backbone(
                "CA".to_string(), "ALA".to_string(), 1,
                0.0, 0.0, 0.0, Element::Carbon,
            ),
            Atom::backbone(
                "CA".to_string(), "ALA".to_string(), 2,
                2.0, 2.0, 2.0, Element::Carbon,
            ),
        ];

        let center = calculate_center_of_mass(&atoms);
        assert_eq!(center, [1.0, 1.0, 1.0]);
    }
}