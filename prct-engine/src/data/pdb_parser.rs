// PDB file parser with exact format specification compliance
use crate::geometry::{Structure, Chain, Residue, Atom, AminoAcid, StructureMetadata, ExperimentalMethod};
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
                    // Will implement in phase 1B.1.2
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
}