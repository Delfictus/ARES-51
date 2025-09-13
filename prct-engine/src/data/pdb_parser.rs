// PDB file parser with exact format specification compliance
use crate::geometry::{Structure, Chain, ChainType, Residue, Atom, AminoAcid, StructureMetadata, ExperimentalMethod};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::collections::HashMap;
use std::str::FromStr;

/// PDB parser with zero-drift compliance
pub struct PDBParser;

impl PDBParser {
    /// Parse PDB file from path
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Structure, PDBParseError> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        
        let filename = path.as_ref().file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("UNKNOWN")
            .to_string();
            
        Self::parse_lines(reader.lines(), filename)
    }

    /// Parse PDB from string content
    pub fn parse_string(content: &str, id: String) -> Result<Structure, PDBParseError> {
        let lines = content.lines().map(|s| Ok(s.to_string()));
        Self::parse_lines(lines, id)
    }

    /// Parse PDB from string content without validation (used for testing)
    pub fn parse_string_no_validation(content: &str, id: String) -> Result<Structure, PDBParseError> {
        let lines = content.lines().map(|s| Ok(s.to_string()));
        Self::parse_lines_no_validation(lines, id)
    }

    /// Parse PDB from iterator of lines
    fn parse_lines<I>(lines: I, structure_id: String) -> Result<Structure, PDBParseError>
    where
        I: Iterator<Item = std::io::Result<String>>,
    {
        let mut structure = Structure::new(structure_id);
        let mut chains: HashMap<char, Chain> = HashMap::new();
        let mut current_residue_map: HashMap<(char, i32), Residue> = HashMap::new();
        let mut atom_counter = 0;

        for (line_num, line_result) in lines.enumerate() {
            let line = line_result?;
            let line_num = line_num + 1;

            // Skip short lines and comments
            if line.len() < 6 {
                continue;
            }

            let record_type = &line[0..6];

            match record_type {
                "HEADER" => {
                    Self::parse_header(&line, &mut structure.metadata)?;
                }
                "TITLE " => {
                    Self::parse_title(&line, &mut structure.metadata);
                }
                "EXPDTA" => {
                    Self::parse_expdta(&line, &mut structure.metadata);
                }
                "REMARK" => {
                    Self::parse_remark(&line, &mut structure.metadata);
                }
                "ATOM  " | "HETATM" => {
                    let atom = Atom::from_pdb_line(&line, atom_counter)
                        .map_err(|e| PDBParseError::AtomParse(line_num, e))?;
                    atom_counter += 1;

                    // Get or create chain
                    let chain = chains.entry(atom.chain_id)
                        .or_insert_with(|| Chain::protein(atom.chain_id));

                    // Get or create residue
                    let residue_key = (atom.chain_id, atom.residue_seq);
                    let residue = current_residue_map.entry(residue_key)
                        .or_insert_with(|| {
                            // Determine amino acid from residue name in ATOM line
                            let aa = if line.len() >= 20 {
                                let res_name = line[17..20].trim();
                                res_name.parse::<AminoAcid>().unwrap_or(AminoAcid::Unk)
                            } else {
                                AminoAcid::Unk
                            };
                            
                            Residue::new(atom.residue_seq, aa, atom.chain_id, atom.insertion_code)
                        });

                    residue.add_atom(atom);
                }
                "SEQRES" => {
                    Self::parse_seqres(&line, &mut structure)?;
                }
                "HELIX " => {
                    // Will implement in phase 1B.1.3
                }
                "SHEET " => {
                    // Will implement in phase 1B.1.3
                }
                "CRYST1" => {
                    // Will implement unit cell parsing
                }
                "END   " | "ENDMDL" => {
                    break;
                }
                _ => {
                    // Skip unrecognized records
                }
            }
        }

        // Add all residues to chains
        for ((chain_id, _), residue) in current_residue_map {
            if let Some(chain) = chains.get_mut(&chain_id) {
                chain.add_residue(residue);
            }
        }

        // Add chains to structure in alphabetical order
        let mut chain_ids: Vec<_> = chains.keys().cloned().collect();
        chain_ids.sort();
        
        for chain_id in chain_ids {
            if let Some(chain) = chains.remove(&chain_id) {
                structure.add_chain(chain);
            }
        }

        // Validate structure
        let errors = structure.validate();
        if !errors.is_empty() {
            return Err(PDBParseError::StructureValidation(errors));
        }

        Ok(structure)
    }

    /// Parse PDB from iterator of lines without validation
    fn parse_lines_no_validation<I>(lines: I, structure_id: String) -> Result<Structure, PDBParseError>
    where
        I: Iterator<Item = std::io::Result<String>>,
    {
        let mut structure = Structure::new(structure_id);
        let mut chains: HashMap<char, Chain> = HashMap::new();
        let mut current_residue_map: HashMap<(char, i32), Residue> = HashMap::new();
        let mut atom_counter = 0;

        for (line_num, line_result) in lines.enumerate() {
            let line = line_result?;
            let line_num = line_num + 1;

            // Skip short lines and comments
            if line.len() < 6 {
                continue;
            }

            let record_type = &line[0..6];

            match record_type {
                "HEADER" => {
                    Self::parse_header(&line, &mut structure.metadata)?;
                }
                "TITLE " => {
                    Self::parse_title(&line, &mut structure.metadata);
                }
                "EXPDTA" => {
                    Self::parse_expdta(&line, &mut structure.metadata);
                }
                "REMARK" => {
                    Self::parse_remark(&line, &mut structure.metadata);
                }
                "ATOM  " | "HETATM" => {
                    let atom = Atom::from_pdb_line(&line, atom_counter)
                        .map_err(|e| PDBParseError::AtomParse(line_num, e))?;
                    atom_counter += 1;

                    // Get or create chain
                    let chain = chains.entry(atom.chain_id)
                        .or_insert_with(|| Chain::protein(atom.chain_id));

                    // Get or create residue
                    let residue_key = (atom.chain_id, atom.residue_seq);
                    let residue = current_residue_map.entry(residue_key)
                        .or_insert_with(|| {
                            // Determine amino acid from residue name in ATOM line
                            let aa = if line.len() >= 20 {
                                let res_name = line[17..20].trim();
                                res_name.parse::<AminoAcid>().unwrap_or(AminoAcid::Unk)
                            } else {
                                AminoAcid::Unk
                            };
                            
                            Residue::new(atom.residue_seq, aa, atom.chain_id, atom.insertion_code)
                        });

                    residue.add_atom(atom);
                }
                "SEQRES" => {
                    Self::parse_seqres(&line, &mut structure)?;
                }
                "HELIX " => {
                    // Will implement in phase 1B.1.3
                }
                "SHEET " => {
                    // Will implement in phase 1B.1.3
                }
                "CRYST1" => {
                    // Will implement unit cell parsing
                }
                "END   " | "ENDMDL" => {
                    break;
                }
                _ => {
                    // Skip unrecognized records
                }
            }
        }

        // Add all residues to chains
        for ((chain_id, _), residue) in current_residue_map {
            if let Some(chain) = chains.get_mut(&chain_id) {
                chain.add_residue(residue);
            }
        }

        // Add chains to structure in alphabetical order, preserving SEQRES data
        let mut chain_ids: Vec<_> = chains.keys().cloned().collect();
        chain_ids.sort();
        
        for chain_id in chain_ids {
            if let Some(mut chain) = chains.remove(&chain_id) {
                // Preserve SEQRES data if it exists in the structure already
                if let Some(existing_chain) = structure.get_chain(chain_id) {
                    if let Some(seqres) = existing_chain.seqres_sequence() {
                        chain.set_seqres_sequence(seqres.to_string());
                    }
                }
                structure.add_chain(chain);
            }
        }

        // NO VALIDATION - return structure as-is
        Ok(structure)
    }

    /// Parse HEADER record
    fn parse_header(line: &str, metadata: &mut StructureMetadata) -> Result<(), PDBParseError> {
        if line.len() >= 50 {
            metadata.classification = line[10..50].trim().to_string();
        }
        if line.len() >= 59 {
            metadata.deposition_date = line[50..59].trim().to_string();
        }
        Ok(())
    }

    /// Parse TITLE record
    fn parse_title(line: &str, metadata: &mut StructureMetadata) {
        if line.len() > 10 {
            let title_part = line[10..].trim();
            if metadata.title.is_empty() {
                metadata.title = title_part.to_string();
            } else {
                metadata.title.push(' ');
                metadata.title.push_str(title_part);
            }
        }
    }

    /// Parse EXPDTA record
    fn parse_expdta(line: &str, metadata: &mut StructureMetadata) {
        if line.len() > 10 {
            let method_str = line[10..].trim().to_uppercase();
            metadata.experimental_method = match method_str.as_str() {
                s if s.contains("X-RAY") => ExperimentalMethod::XRayDiffraction,
                s if s.contains("NMR") => ExperimentalMethod::NMRSolution,
                s if s.contains("ELECTRON MICROSCOPY") => ExperimentalMethod::ElectronMicroscopy,
                s if s.contains("NEUTRON") => ExperimentalMethod::NeutronDiffraction,
                _ => ExperimentalMethod::Unknown,
            };
        }
    }

    /// Parse REMARK records for resolution and R-factors
    fn parse_remark(line: &str, metadata: &mut StructureMetadata) {
        if line.len() < 10 {
            return;
        }

        let remark_num_str = line[7..10].trim();
        let remark_text = if line.len() > 11 { line[11..].trim() } else { "" };

        match remark_num_str {
            "2" => {
                // REMARK   2 RESOLUTION.    1.50 ANGSTROMS.
                if remark_text.contains("RESOLUTION") {
                    if let Some(res_str) = Self::extract_number_from_text(remark_text) {
                        if let Ok(resolution) = res_str.parse::<f64>() {
                            if resolution > 0.0 && resolution < 10.0 {
                                metadata.resolution = Some(resolution);
                            }
                        }
                    }
                }
            }
            "3" => {
                // REMARK   3   R VALUE (WORKING SET) : 0.180
                if remark_text.contains("R VALUE") && remark_text.contains("WORKING") {
                    if let Some(r_str) = Self::extract_number_from_text(remark_text) {
                        if let Ok(r_work) = r_str.parse::<f64>() {
                            if r_work >= 0.0 && r_work <= 1.0 {
                                metadata.r_work = Some(r_work);
                            }
                        }
                    }
                }
                // REMARK   3   FREE R VALUE                     : 0.215
                if remark_text.contains("FREE R") {
                    if let Some(r_str) = Self::extract_number_from_text(remark_text) {
                        if let Ok(r_free) = r_str.parse::<f64>() {
                            if r_free >= 0.0 && r_free <= 1.0 {
                                metadata.r_free = Some(r_free);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Parse SEQRES record for sequence information
    fn parse_seqres(line: &str, structure: &mut Structure) -> Result<(), PDBParseError> {
        if line.len() < 19 {
            return Ok(()); // Skip malformed SEQRES records
        }

        // SEQRES format:
        // SEQRES   1 A  147  THR VAL ASN VAL LEU ALA LYS GLY ARG ASN GLY GLU
        // Columns: 1-6 (SEQRES), 8-10 (serial number), 12 (chain ID), 14-17 (num residues), 20+ (residues)
        
        let chain_id = line.chars().nth(11).ok_or_else(|| {
            PDBParseError::InvalidFormat("Missing chain ID in SEQRES record".to_string())
        })?;

        // Parse number of residues for validation
        let num_residues = if line.len() >= 17 {
            line[13..17].trim().parse::<usize>().unwrap_or(0)
        } else {
            0
        };

        // Extract residue names (start at column 20, each is 3 letters + space)
        if line.len() < 20 {
            return Ok(());
        }

        let residue_part = &line[19..];
        let residue_names: Vec<&str> = residue_part
            .split_whitespace()
            .take(13) // Maximum 13 residues per SEQRES record
            .collect();

        // Convert three-letter codes to one-letter codes
        let mut sequence_part = String::new();
        for res_name in residue_names {
            let one_letter = Self::three_letter_to_one_letter(res_name);
            sequence_part.push(one_letter);
        }

        // Get or create chain to store sequence
        if !structure.has_chain(chain_id) {
            let chain = Chain::new(chain_id, ChainType::Protein);
            structure.add_chain(chain);
        }

        // Store sequence information in chain's SEQRES field
        if let Some(chain) = structure.get_chain_mut(chain_id) {
            // Append to existing SEQRES sequence (multi-line SEQRES records)
            if let Some(existing_seq) = chain.seqres_sequence() {
                let combined_seq = format!("{}{}", existing_seq, sequence_part);
                chain.set_seqres_sequence(combined_seq);
            } else {
                chain.set_seqres_sequence(sequence_part.clone());
            }
            
            // Validate the length matches expected (allow some tolerance for incomplete SEQRES)
            if num_residues > 0 {
                let current_seqres_len = chain.seqres_sequence().unwrap_or("").len();
                if current_seqres_len as f64 / num_residues as f64 > 1.2 {
                    return Err(PDBParseError::InvalidFormat(format!(
                        "SEQRES residue count mismatch for chain {}: expected {}, current SEQRES length {}",
                        chain_id, num_residues, current_seqres_len
                    )));
                }
            }
        }

        Ok(())
    }

    /// Convert three-letter amino acid code to one-letter code
    fn three_letter_to_one_letter(three_letter: &str) -> char {
        match three_letter.trim().to_uppercase().as_str() {
            "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D',
            "CYS" => 'C', "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G',
            "HIS" => 'H', "ILE" => 'I', "LEU" => 'L', "LYS" => 'K',
            "MET" => 'M', "PHE" => 'F', "PRO" => 'P', "SER" => 'S',
            "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
            
            // Non-standard amino acids
            "SEC" => 'U',         // Selenocysteine
            "PYL" => 'O',         // Pyrrolysine
            "ASX" => 'B',         // Asparagine or Aspartic acid
            "GLX" => 'Z',         // Glutamine or Glutamic acid
            
            // Modified amino acids - map to standard equivalents
            "MSE" => 'M',         // Selenomethionine -> Methionine
            "CSE" => 'C',         // Selenocysteine -> Cysteine  
            "PCA" => 'E',         // Pyroglutamic acid -> Glutamic acid
            "HYP" => 'P',         // Hydroxyproline -> Proline
            "TPO" => 'T',         // Phosphothreonine -> Threonine
            "SEP" => 'S',         // Phosphoserine -> Serine
            "PTR" => 'Y',         // Phosphotyrosine -> Tyrosine
            
            // Common ligands and heteroatoms - mark as unknown
            "HOH" | "WAT" => 'X', // Water
            "SO4" | "PO4" => 'X', // Sulfate, Phosphate
            "CA" | "MG" | "ZN" | "FE" => 'X', // Metal ions
            "ATP" | "ADP" | "GTP" | "GDP" => 'X', // Nucleotides
            "NAD" | "FAD" | "COA" => 'X', // Cofactors
            
            // Unknown or unrecognized - use X
            _ => 'X',
        }
    }

    /// Extract first number from text
    fn extract_number_from_text(text: &str) -> Option<String> {
        let mut number = String::new();
        let mut found_digit = false;
        let mut found_decimal = false;

        for ch in text.chars() {
            if ch.is_ascii_digit() {
                number.push(ch);
                found_digit = true;
            } else if ch == '.' && found_digit && !found_decimal {
                number.push(ch);
                found_decimal = true;
            } else if found_digit {
                break;
            }
        }

        if found_digit { Some(number) } else { None }
    }

    /// Validate parsed structure
    pub fn validate_structure(structure: &Structure) -> Vec<String> {
        let mut warnings = Vec::new();

        if structure.chain_count() == 0 {
            warnings.push("Structure has no chains".to_string());
        }

        if structure.atom_count() == 0 {
            warnings.push("Structure has no atoms".to_string());
        }

        for chain in structure.chains() {
            if chain.is_empty() {
                warnings.push(format!("Chain {} is empty", chain.id));
            }

            let incomplete_residues = chain.incomplete_backbone_residues();
            if !incomplete_residues.is_empty() {
                warnings.push(format!(
                    "Chain {} has {} residues with incomplete backbone",
                    chain.id,
                    incomplete_residues.len()
                ));
            }
        }

        // Check resolution
        if let Some(resolution) = structure.metadata.resolution {
            if resolution > 3.0 {
                warnings.push(format!("Low resolution structure: {:.2} Å", resolution));
            }
        }

        // Check R-factors
        if let Some(r_work) = structure.metadata.r_work {
            if r_work > 0.3 {
                warnings.push(format!("High R-work: {:.3}", r_work));
            }
        }

        warnings
    }
}

/// PDB parsing errors
#[derive(Debug)]
pub enum PDBParseError {
    Io(std::io::Error),
    AtomParse(usize, crate::geometry::atom::PdbParseError),
    StructureValidation(Vec<crate::geometry::structure::StructureValidationError>),
    InvalidFormat(String),
    MissingData(String),
}

impl From<std::io::Error> for PDBParseError {
    fn from(err: std::io::Error) -> Self {
        PDBParseError::Io(err)
    }
}

impl std::fmt::Display for PDBParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PDBParseError::Io(err) => write!(f, "IO error: {}", err),
            PDBParseError::AtomParse(line, err) => write!(f, "Atom parsing error at line {}: {}", line, err),
            PDBParseError::StructureValidation(errors) => {
                write!(f, "Structure validation failed: {} errors", errors.len())
            }
            PDBParseError::InvalidFormat(msg) => write!(f, "Invalid PDB format: {}", msg),
            PDBParseError::MissingData(msg) => write!(f, "Missing required data: {}", msg),
        }
    }
}

impl std::error::Error for PDBParseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Element;

    const TEST_PDB_CONTENT: &str = r#"HEADER    HYDROLASE/HYDROLASE INHIBITOR           12-NOV-94   1HTM              
TITLE     HIV-1 PROTEASE IN COMPLEX WITH THE INHIBITOR CGP 53820          
EXPDTA    X-RAY DIFFRACTION                                                 
REMARK   2 RESOLUTION.    2.00 ANGSTROMS.                                   
REMARK   3   R VALUE     (WORKING SET) : 0.180                              
REMARK   3   FREE R VALUE             : 0.215                              
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  -8.605  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.710  16.882  -8.579  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.639  17.977  -9.142  1.00 20.00           O  
ATOM      5  CB  ALA A   1      18.876  15.038  -9.678  1.00 20.00           C  
END                                                                             
"#;

    #[test]
    fn test_parse_simple_pdb() {
        let structure = PDBParser::parse_string(TEST_PDB_CONTENT, "1HTM".to_string()).unwrap();
        
        assert_eq!(structure.id, "1HTM");
        assert_eq!(structure.chain_count(), 1);
        assert_eq!(structure.residue_count(), 1);
        assert_eq!(structure.atom_count(), 5);
        
        // Check metadata
        assert_eq!(structure.metadata.experimental_method, ExperimentalMethod::XRayDiffraction);
        assert_eq!(structure.metadata.resolution, Some(2.0));
        assert_eq!(structure.metadata.r_work, Some(0.180));
        assert_eq!(structure.metadata.r_free, Some(0.215));
        
        // Check chain
        let chain = structure.get_chain('A').unwrap();
        assert_eq!(chain.id, 'A');
        assert_eq!(chain.len(), 1);
        
        // Check residue
        let residue = chain.get_residue(1).unwrap();
        assert_eq!(residue.amino_acid, AminoAcid::Ala);
        assert_eq!(residue.atom_count(), 5);
        assert!(residue.has_complete_backbone());
    }

    #[test]
    fn test_atom_coordinates() {
        let structure = PDBParser::parse_string(TEST_PDB_CONTENT, "1HTM".to_string()).unwrap();
        let residue = structure.get_residue('A', 1).unwrap();
        
        let n_atom = residue.get_atom("N").unwrap();
        assert!((n_atom.coords.x - 20.154).abs() < 1e-6);
        assert!((n_atom.coords.y - 16.967).abs() < 1e-6);
        assert!((n_atom.coords.z + 8.901).abs() < 1e-6);
        
        let ca_atom = residue.get_atom("CA").unwrap();
        assert_eq!(ca_atom.element, Element::C);
        assert!((ca_atom.b_factor - 20.0).abs() < 1e-6);
        assert!((ca_atom.occupancy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metadata_parsing() {
        let structure = PDBParser::parse_string(TEST_PDB_CONTENT, "1HTM".to_string()).unwrap();
        
        assert!(structure.metadata.title.contains("HIV-1 PROTEASE"));
        assert!(structure.metadata.classification.contains("HYDROLASE"));
    }

    #[test]
    fn test_number_extraction() {
        assert_eq!(PDBParser::extract_number_from_text("RESOLUTION. 1.50 ANGSTROMS"), Some("1.50".to_string()));
        assert_eq!(PDBParser::extract_number_from_text("R VALUE : 0.180"), Some("0.180".to_string()));
        assert_eq!(PDBParser::extract_number_from_text("NO NUMBERS HERE"), None);
    }

    #[test]
    fn test_validation_warnings() {
        let structure = PDBParser::parse_string(TEST_PDB_CONTENT, "1HTM".to_string()).unwrap();
        let warnings = PDBParser::validate_structure(&structure);
        
        // Should have no major warnings for complete structure
        assert!(warnings.is_empty() || warnings.iter().all(|w| !w.contains("no chains")));
    }

    #[test]
    fn test_seqres_parsing() {
        let seqres_content = r#"HEADER    HYDROLASE/HYDROLASE INHIBITOR           20-NOV-91   1HTM              
SEQRES   1 A   99  PRO GLN ILE THR LEU TRP GLN ARG PRO LEU VAL THR ILE
SEQRES   2 A   99  LYS ILE GLY GLY GLN LEU LYS GLU ALA LEU LEU ASP THR
SEQRES   3 A   99  GLY ALA ASP ASP THR VAL LEU GLU GLU MET SER LEU PRO
SEQRES   4 A   99  GLY ARG TRP LYS PRO LYS MET ILE GLY GLY ILE GLY GLY
SEQRES   5 A   99  PHE ILE LYS VAL ARG GLN TYR ASP GLN ILE LEU ILE GLU
SEQRES   6 A   99  ILE CYS GLY HIS LYS ALA ILE GLY THR VAL LEU VAL GLY
SEQRES   7 A   99  PRO THR PRO VAL ASN ILE ILE GLY ARG ASN LEU LEU THR
SEQRES   8 A   99  GLN ILE GLY CYS THR LEU ASN PHE
SEQRES   1 B   99  PRO GLN ILE THR LEU TRP GLN ARG PRO LEU VAL THR ILE
SEQRES   2 B   99  LYS ILE GLY GLY GLN LEU LYS GLU ALA LEU LEU ASP THR
END"#;

        // Parse with no structure validation (chains will be empty but have SEQRES)
        let structure = match PDBParser::parse_string(seqres_content, "TEST".to_string()) {
            Ok(s) => s,
            Err(PDBParseError::StructureValidation(_errors)) => {
                // Expected - parse without validation to get the structure with SEQRES data
                PDBParser::parse_string_no_validation(seqres_content, "TEST".to_string()).unwrap()
            },
            Err(e) => panic!("Unexpected error: {:?}", e),
        };
        
        // Should have two chains
        assert_eq!(structure.chain_count(), 2);
        assert!(structure.has_chain('A'));
        assert!(structure.has_chain('B'));
        
        // Check chain A SEQRES sequence
        let chain_a = structure.get_chain('A').unwrap();
        let seqres_a = chain_a.seqres_sequence().unwrap();
        assert_eq!(seqres_a.len(), 99); // Should be 99 residues as specified
        assert!(seqres_a.starts_with("PQITLWQRPL")); // First 10 residues
        assert!(seqres_a.ends_with("CTLNF")); // Last 5 residues
        
        // Check chain B SEQRES sequence (partial)
        let chain_b = structure.get_chain('B').unwrap();
        let seqres_b = chain_b.seqres_sequence().unwrap();
        assert_eq!(seqres_b.len(), 26); // Only 2 lines parsed = 26 residues
        assert!(seqres_b.starts_with("PQITLWQRPL")); // Should match chain A start
    }

    #[test]
    fn test_three_letter_conversion() {
        // Standard amino acids
        assert_eq!(PDBParser::three_letter_to_one_letter("ALA"), 'A');
        assert_eq!(PDBParser::three_letter_to_one_letter("ARG"), 'R');
        assert_eq!(PDBParser::three_letter_to_one_letter("TRP"), 'W');
        
        // Non-standard amino acids
        assert_eq!(PDBParser::three_letter_to_one_letter("MSE"), 'M'); // Selenomethionine
        assert_eq!(PDBParser::three_letter_to_one_letter("SEC"), 'U'); // Selenocysteine
        assert_eq!(PDBParser::three_letter_to_one_letter("PYL"), 'O'); // Pyrrolysine
        
        // Modified amino acids
        assert_eq!(PDBParser::three_letter_to_one_letter("HYP"), 'P'); // Hydroxyproline
        assert_eq!(PDBParser::three_letter_to_one_letter("PCA"), 'E'); // Pyroglutamic acid
        
        // Unknown residues
        assert_eq!(PDBParser::three_letter_to_one_letter("UNK"), 'X');
        assert_eq!(PDBParser::three_letter_to_one_letter("XYZ"), 'X');
        
        // Case insensitive
        assert_eq!(PDBParser::three_letter_to_one_letter("ala"), 'A');
        assert_eq!(PDBParser::three_letter_to_one_letter("Arg"), 'R');
    }

    #[test]
    fn test_seqres_validation() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
SEQRES   1 A    5  ARG ALA ASN ASP CYS
ATOM      1  N   ARG A   1      20.154  16.967  -8.901  1.00 20.00           N  
ATOM      2  CA  ARG A   1      21.618  16.507  -8.620  1.00 20.00           C  
ATOM      3  C   ARG A   1      22.602  17.661  -8.897  1.00 20.00           C  
ATOM      4  O   ARG A   1      23.820  17.556  -8.815  1.00 20.00           O  
ATOM      5  N   ALA A   2      22.090  18.831  -9.221  1.00 20.00           N  
ATOM      6  CA  ALA A   2      22.968  19.990  -9.519  1.00 20.00           C  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        let chain = structure.get_chain('A').unwrap();
        
        // SEQRES sequence should be "RANDC" (5 residues)
        let seqres = chain.seqres_sequence().unwrap();
        assert_eq!(seqres, "RANDC");
        
        // ATOM sequence should be "RA" (only 2 residues have ATOM records: ARG at pos 1, ALA at pos 2)
        let atom_seq = chain.sequence_one_letter();
        assert_eq!(atom_seq, "RA");
        
        // Sequence validation
        use crate::geometry::SequenceValidation;
        let validation = chain.validate_sequence_consistency();
        match validation {
            SequenceValidation::MissingResidues(count) => {
                assert_eq!(count, 3); // Missing 3 residues (N, D, C)
            },
            _ => panic!("Expected MissingResidues validation result"),
        }
        
        // Authoritative sequence should be SEQRES
        assert_eq!(chain.authoritative_sequence(), "RANDC");
    }
}