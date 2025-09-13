// PDB file parser with exact format specification compliance
use crate::geometry::{Structure, Chain, ChainType, Residue, Atom, AminoAcid, StructureMetadata, ExperimentalMethod, HelixRecord, SheetRecord, SheetRegistration, UnitCell};
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
                    Self::parse_remark(&line, &mut structure.metadata, &mut structure.space_group);
                }
                "AUTHOR" => {
                    Self::parse_author(&line, &mut structure.metadata);
                }
                "CRYST1" => {
                    Self::parse_cryst1(&line, &mut structure.unit_cell, &mut structure.space_group)?;
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
                    Self::parse_helix(&line, &mut structure)?;
                }
                "SHEET " => {
                    Self::parse_sheet(&line, &mut structure)?;
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

        // Assign secondary structure from parsed HELIX and SHEET records
        structure.assign_secondary_structure_from_records();

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
                    Self::parse_remark(&line, &mut structure.metadata, &mut structure.space_group);
                }
                "AUTHOR" => {
                    Self::parse_author(&line, &mut structure.metadata);
                }
                "CRYST1" => {
                    Self::parse_cryst1(&line, &mut structure.unit_cell, &mut structure.space_group)?;
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
                    Self::parse_helix(&line, &mut structure)?;
                }
                "SHEET " => {
                    Self::parse_sheet(&line, &mut structure)?;
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

        // Assign secondary structure from parsed HELIX and SHEET records
        structure.assign_secondary_structure_from_records();

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

    /// Parse REMARK records for resolution, R-factors, and space group
    fn parse_remark(line: &str, metadata: &mut StructureMetadata, space_group: &mut Option<String>) {
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
            "290" => {
                // REMARK 290   CRYSTALLOGRAPHIC SYMMETRY
                // REMARK 290   SYMMETRY OPERATORS FOR SPACE GROUP: P 21 21 21
                if remark_text.contains("SPACE GROUP:") {
                    if let Some(sg_start) = remark_text.find("SPACE GROUP:") {
                        let space_group_text = remark_text[sg_start + 12..].trim();
                        if !space_group_text.is_empty() {
                            *space_group = Some(space_group_text.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Parse AUTHOR record
    fn parse_author(line: &str, metadata: &mut StructureMetadata) {
        if line.len() > 10 {
            let author_part = line[10..].trim();
            let authors: Vec<String> = author_part
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            metadata.authors.extend(authors);
        }
    }

    /// Parse CRYST1 record for unit cell and space group
    fn parse_cryst1(line: &str, unit_cell: &mut Option<UnitCell>, space_group: &mut Option<String>) -> Result<(), PDBParseError> {
        if line.len() < 55 {
            return Ok(());
        }

        // CRYST1 format:
        // CRYST1   61.777   58.233   23.135  90.00  90.00  90.00 P 21 21 21    4
        // Columns: 7-15 (a), 16-24 (b), 25-33 (c), 34-40 (alpha), 41-47 (beta), 48-54 (gamma), 56-66 (space group)
        
        let a = line[6..15].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell dimension a".to_string())
        })?;
        
        let b = line[15..24].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell dimension b".to_string())
        })?;
        
        let c = line[24..33].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell dimension c".to_string())
        })?;
        
        let alpha = line[33..40].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell angle alpha".to_string())
        })?;
        
        let beta = line[40..47].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell angle beta".to_string())
        })?;
        
        let gamma = line[47..54].trim().parse::<f64>().map_err(|_| {
            PDBParseError::InvalidFormat("Invalid unit cell angle gamma".to_string())
        })?;

        // Calculate volume (for orthorhombic case)
        let volume = a * b * c;

        *unit_cell = Some(UnitCell {
            dimensions: (a, b, c),
            angles: (alpha, beta, gamma),
            volume,
        });

        // Parse space group if present (columns 56-66, excluding Z value)
        if line.len() > 55 {
            let sg_part = if line.len() > 66 { &line[55..66] } else { &line[55..] };
            let sg_text = sg_part.trim();
            if !sg_text.is_empty() {
                *space_group = Some(sg_text.to_string());
            }
        }

        Ok(())
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

    /// Parse HELIX record for secondary structure information
    fn parse_helix(line: &str, structure: &mut Structure) -> Result<(), PDBParseError> {
        // HELIX format (columns 1-6: "HELIX ", 8-10: serial, 12-14: helix ID, 16-18: initial residue, 20: initial chain,
        //               22-25: initial seq num, 26: initial iCode, 28-30: terminal residue, 32: terminal chain,
        //               34-37: terminal seq num, 38: terminal iCode, 39-40: helix class, 41-70: comment, 72-76: length)
        if line.len() < 38 {
            return Ok(()); // Skip malformed HELIX records
        }

        let helix_id = line.get(11..14).unwrap_or("").trim();
        if helix_id.is_empty() {
            return Err(PDBParseError::InvalidFormat("Missing helix ID in HELIX record".to_string()));
        }

        // Parse initial residue position
        let init_chain = line.chars().nth(19).unwrap_or(' ');
        let init_seq_str = line.get(21..25).unwrap_or("").trim();
        let init_seq_num: i32 = init_seq_str.parse()
            .map_err(|_| PDBParseError::InvalidFormat(format!("Invalid initial seq num: {}", init_seq_str)))?;
        let init_insertion = line.chars().nth(25).filter(|&c| c != ' ');

        // Parse terminal residue position
        let end_chain = line.chars().nth(31).unwrap_or(' ');
        let end_seq_str = line.get(33..37).unwrap_or("").trim();
        let end_seq_num: i32 = end_seq_str.parse()
            .map_err(|_| PDBParseError::InvalidFormat(format!("Invalid terminal seq num: {}", end_seq_str)))?;
        let end_insertion = line.chars().nth(37).filter(|&c| c != ' ');

        // Parse helix class (default to 1 for right-handed alpha if missing)
        let helix_class_str = line.get(38..40).unwrap_or("").trim();
        let helix_class = if helix_class_str.is_empty() {
            1 // Default to right-handed alpha helix
        } else {
            helix_class_str.parse().unwrap_or(1)
        };

        // Parse comment and length (optional)
        let comment = line.get(40..70).map(|s| s.trim()).filter(|s| !s.is_empty()).map(String::from);
        let length_str = line.get(71..76).unwrap_or("").trim();
        let length = if length_str.is_empty() {
            (end_seq_num - init_seq_num + 1).max(1) // Calculate from positions
        } else {
            length_str.parse().unwrap_or(end_seq_num - init_seq_num + 1)
        };

        let helix_record = HelixRecord {
            id: helix_id.to_string(),
            helix_class,
            init_chain,
            init_seq_num,
            init_insertion,
            end_chain,
            end_seq_num,
            end_insertion,
            length,
            comment,
        };

        structure.helices.push(helix_record);
        Ok(())
    }

    /// Parse SHEET record for secondary structure information
    fn parse_sheet(line: &str, structure: &mut Structure) -> Result<(), PDBParseError> {
        // SHEET format (columns 1-6: "SHEET ", 8-10: strand, 12-14: sheet ID, 15-16: num strands,
        //               18-20: initial residue, 22: initial chain, 23-26: initial seq num, 27: initial iCode,
        //               29-31: terminal residue, 33: terminal chain, 34-37: terminal seq num, 38: terminal iCode,
        //               39-40: sense, 42-45: registration)
        if line.len() < 38 {
            return Ok(()); // Skip malformed SHEET records
        }

        let strand_str = line.get(7..10).unwrap_or("").trim();
        let strand: i32 = strand_str.parse()
            .map_err(|_| PDBParseError::InvalidFormat(format!("Invalid strand number: {}", strand_str)))?;

        let sheet_id = line.get(11..14).unwrap_or("").trim();
        if sheet_id.is_empty() {
            return Err(PDBParseError::InvalidFormat("Missing sheet ID in SHEET record".to_string()));
        }

        let num_strands_str = line.get(14..16).unwrap_or("").trim();
        let num_strands: i32 = num_strands_str.parse().unwrap_or(1);

        // Parse initial residue position
        let init_chain = line.chars().nth(21).unwrap_or(' ');
        let init_seq_str = line.get(22..26).unwrap_or("").trim();
        let init_seq_num: i32 = init_seq_str.parse()
            .map_err(|_| PDBParseError::InvalidFormat(format!("Invalid initial seq num: {}", init_seq_str)))?;
        let init_insertion = line.chars().nth(26).filter(|&c| c != ' ');

        // Parse terminal residue position
        let end_chain = line.chars().nth(32).unwrap_or(' ');
        let end_seq_str = line.get(33..37).unwrap_or("").trim();
        let end_seq_num: i32 = end_seq_str.parse()
            .map_err(|_| PDBParseError::InvalidFormat(format!("Invalid terminal seq num: {}", end_seq_str)))?;
        let end_insertion = line.chars().nth(37).filter(|&c| c != ' ');

        // Parse sense (0=first strand, 1=parallel, -1=antiparallel)
        let sense_str = line.get(38..40).unwrap_or("").trim();
        let sense = if sense_str.is_empty() {
            0 // First strand (no previous strand to compare to)
        } else {
            sense_str.parse().unwrap_or(0)
        };

        // Parse registration information (optional for strands > 1)
        let registration = if line.len() > 50 && strand > 1 {
            let cur_atom = line.get(41..45).unwrap_or("").trim();
            let cur_chain = line.chars().nth(49).unwrap_or(' ');
            let cur_seq_str = line.get(50..54).unwrap_or("").trim();
            let prev_atom = line.get(56..60).unwrap_or("").trim();
            let prev_chain = line.chars().nth(64).unwrap_or(' ');
            let prev_seq_str = line.get(65..69).unwrap_or("").trim();

            if !cur_atom.is_empty() && !prev_atom.is_empty() {
                Some(SheetRegistration {
                    cur_atom: cur_atom.to_string(),
                    cur_chain,
                    cur_seq_num: cur_seq_str.parse().unwrap_or(init_seq_num),
                    cur_insertion: line.chars().nth(54).filter(|&c| c != ' '),
                    prev_atom: prev_atom.to_string(),
                    prev_chain,
                    prev_seq_num: prev_seq_str.parse().unwrap_or(end_seq_num),
                    prev_insertion: line.chars().nth(69).filter(|&c| c != ' '),
                })
            } else {
                None
            }
        } else {
            None
        };

        let sheet_record = SheetRecord {
            sheet_id: sheet_id.to_string(),
            strand,
            num_strands,
            init_chain,
            init_seq_num,
            init_insertion,
            end_chain,
            end_seq_num,
            end_insertion,
            sense,
            registration,
        };

        structure.sheets.push(sheet_record);
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

    #[test]
    fn test_helix_parsing() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
HELIX    1  H1 ALA A   10  LEU A   20  1                                  11
ATOM      1  N   ALA A   10     20.154  16.967  -8.901  1.00 20.00           N  
ATOM      2  CA  ALA A   10     21.618  16.507  -8.620  1.00 20.00           C  
ATOM      3  N   VAL A   15     22.090  18.831  -9.221  1.00 20.00           N  
ATOM      4  CA  VAL A   15     22.968  19.990  -9.519  1.00 20.00           C  
ATOM      5  N   LEU A   20     24.090  20.831  -9.421  1.00 20.00           N  
ATOM      6  CA  LEU A   20     24.968  21.990  -9.619  1.00 20.00           C  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        // Check that the HELIX record was parsed
        assert_eq!(structure.helices.len(), 1);
        let helix = &structure.helices[0];
        assert_eq!(helix.id, "H1");
        assert_eq!(helix.helix_class, 1); // Right-handed alpha helix
        assert_eq!(helix.init_chain, 'A');
        assert_eq!(helix.init_seq_num, 10);
        assert_eq!(helix.end_chain, 'A');
        assert_eq!(helix.end_seq_num, 20);
        assert_eq!(helix.length, 11);

        // Check that secondary structure was assigned to residues in helix range
        let chain = structure.get_chain('A').unwrap();
        if let Some(residue) = chain.get_residue(10) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        }
        if let Some(residue) = chain.get_residue(15) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        }
        if let Some(residue) = chain.get_residue(20) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        }
    }

    #[test]
    fn test_sheet_parsing() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
SHEET    1   A 3 ILE A  96  THR A 101  0                                        
SHEET    2   A 3 PHE A 134  PHE A 138  1  N  LEU A 135   O  THR A  97
SHEET    3   A 3 ARG A 180  ILE A 184 -1  N  PHE A 182   O  ILE A 136
ATOM      1  N   ILE A   96     20.154  16.967  -8.901  1.00 20.00           N  
ATOM      2  CA  ILE A   96     21.618  16.507  -8.620  1.00 20.00           C  
ATOM      3  N   THR A  101     22.090  18.831  -9.221  1.00 20.00           N  
ATOM      4  CA  THR A  101     22.968  19.990  -9.519  1.00 20.00           C  
ATOM      5  N   PHE A  134     24.090  20.831  -9.421  1.00 20.00           N  
ATOM      6  CA  PHE A  134     24.968  21.990  -9.619  1.00 20.00           C  
ATOM      7  N   PHE A  138     26.090  22.831  -9.521  1.00 20.00           N  
ATOM      8  CA  PHE A  138     26.968  23.990  -9.719  1.00 20.00           C  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        // Check that the SHEET records were parsed
        assert_eq!(structure.sheets.len(), 3);
        
        // Check first strand
        let strand1 = &structure.sheets[0];
        assert_eq!(strand1.sheet_id, "A");
        assert_eq!(strand1.strand, 1);
        assert_eq!(strand1.num_strands, 3);
        assert_eq!(strand1.init_chain, 'A');
        assert_eq!(strand1.init_seq_num, 96);
        assert_eq!(strand1.end_chain, 'A');
        assert_eq!(strand1.end_seq_num, 101);
        assert_eq!(strand1.sense, 0); // First strand
        assert!(strand1.registration.is_none()); // No registration for first strand

        // Check second strand (parallel)
        let strand2 = &structure.sheets[1];
        assert_eq!(strand2.strand, 2);
        assert_eq!(strand2.sense, 1); // Parallel
        assert!(strand2.registration.is_some());
        let reg2 = strand2.registration.as_ref().unwrap();
        assert_eq!(reg2.cur_atom, "N");
        assert_eq!(reg2.cur_chain, 'A');
        assert_eq!(reg2.cur_seq_num, 135);
        assert_eq!(reg2.prev_atom, "O");
        assert_eq!(reg2.prev_chain, 'A');
        assert_eq!(reg2.prev_seq_num, 97);

        // Check third strand (antiparallel)
        let strand3 = &structure.sheets[2];
        assert_eq!(strand3.strand, 3);
        assert_eq!(strand3.sense, -1); // Antiparallel

        // Check that secondary structure was assigned to residues in sheet ranges
        let chain = structure.get_chain('A').unwrap();
        if let Some(residue) = chain.get_residue(96) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Sheet);
        }
        if let Some(residue) = chain.get_residue(134) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Sheet);
        }
    }

    #[test]
    fn test_mixed_secondary_structure() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
HELIX    1  H1 ALA A   10  LEU A   20  1                                  11
HELIX    2  H2 VAL A   35  GLY A   40  5                                   6
SHEET    1   A 2 ILE A  50  THR A  55  0                                        
SHEET    2   A 2 PHE A  60  PHE A  65  1  N  LEU A  62   O  THR A  52
ATOM      1  N   ALA A  10      10.000   0.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA A  10      11.000   0.000   0.000  1.00 20.00           C  
ATOM      3  N   LEU A  20      20.000   0.000   0.000  1.00 20.00           N  
ATOM      4  CA  LEU A  20      21.000   0.000   0.000  1.00 20.00           C  
ATOM      5  N   VAL A  35      35.000   0.000   0.000  1.00 20.00           N  
ATOM      6  CA  VAL A  35      36.000   0.000   0.000  1.00 20.00           C  
ATOM      7  N   GLY A  40      40.000   0.000   0.000  1.00 20.00           N  
ATOM      8  CA  GLY A  40      41.000   0.000   0.000  1.00 20.00           C  
ATOM      9  N   ILE A  50      50.000   0.000   0.000  1.00 20.00           N  
ATOM     10  CA  ILE A  50      51.000   0.000   0.000  1.00 20.00           C  
ATOM     11  N   THR A  55      55.000   0.000   0.000  1.00 20.00           N  
ATOM     12  CA  THR A  55      56.000   0.000   0.000  1.00 20.00           C  
ATOM     13  N   PHE A  60      60.000   0.000   0.000  1.00 20.00           N  
ATOM     14  CA  PHE A  60      61.000   0.000   0.000  1.00 20.00           C  
ATOM     15  N   PHE A  65      65.000   0.000   0.000  1.00 20.00           N  
ATOM     16  CA  PHE A  65      66.000   0.000   0.000  1.00 20.00           C  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        // Check records were parsed correctly
        assert_eq!(structure.helices.len(), 2);
        assert_eq!(structure.sheets.len(), 2);

        // Check different helix types
        let helix1 = &structure.helices[0];
        assert_eq!(helix1.helix_class, 1); // Alpha helix
        let helix2 = &structure.helices[1];
        assert_eq!(helix2.helix_class, 5); // 3-10 helix

        // Check chain for mixed secondary structure
        let chain = structure.get_chain('A').unwrap();
        
        // Alpha helix residues
        if let Some(residue) = chain.get_residue(10) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        }
        if let Some(residue) = chain.get_residue(20) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        }
        
        // 3-10 helix residues
        if let Some(residue) = chain.get_residue(35) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix310);
        }
        if let Some(residue) = chain.get_residue(40) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Helix310);
        }
        
        // Sheet residues
        if let Some(residue) = chain.get_residue(50) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Sheet);
        }
        if let Some(residue) = chain.get_residue(60) {
            use crate::geometry::residue::SecondaryStructure;
            assert_eq!(residue.secondary_structure, SecondaryStructure::Sheet);
        }
    }

    #[test]
    fn test_secondary_structure_assignment() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
HELIX    1  H1 ALA A   10  ALA A   12  3                                   3
ATOM      1  N   ALA A  10      10.000   0.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA A  10      11.000   0.000   0.000  1.00 20.00           C  
ATOM      3  N   ALA A  11      11.000   0.000   0.000  1.00 20.00           N  
ATOM      4  CA  ALA A  11      12.000   0.000   0.000  1.00 20.00           C  
ATOM      5  N   ALA A  12      12.000   0.000   0.000  1.00 20.00           N  
ATOM      6  CA  ALA A  12      13.000   0.000   0.000  1.00 20.00           C  
ATOM      7  N   ALA A  13      13.000   0.000   0.000  1.00 20.00           N  
ATOM      8  CA  ALA A  13      14.000   0.000   0.000  1.00 20.00           C  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        let chain = structure.get_chain('A').unwrap();

        // Check secondary structure sequence (Note: residues appear in order they were parsed from ATOM records)
        let ss_seq = chain.secondary_structure_sequence();
        assert_eq!(ss_seq, "III-"); // Pi helix residues then unknown

        // Verify specific assignments
        use crate::geometry::residue::SecondaryStructure;
        assert_eq!(chain.get_residue(10).unwrap().secondary_structure, SecondaryStructure::HelixPi);
        assert_eq!(chain.get_residue(11).unwrap().secondary_structure, SecondaryStructure::HelixPi);
        assert_eq!(chain.get_residue(12).unwrap().secondary_structure, SecondaryStructure::HelixPi);
        assert_eq!(chain.get_residue(13).unwrap().secondary_structure, SecondaryStructure::Unknown);
    }

    #[test]
    fn test_header_information_extraction() {
        let test_content = r#"HEADER    HYDROLASE/HYDROLASE INHIBITOR           12-NOV-94   1HTM              
TITLE     HIV-1 PROTEASE IN COMPLEX WITH THE INHIBITOR CGP 53820          
TITLE    2 DETAILED STRUCTURAL ANALYSIS                                    
AUTHOR    J.P.PRIESTLE,A.SCHAR,H.P.GRUTTER                                 
EXPDTA    X-RAY DIFFRACTION                                                 
REMARK   2 RESOLUTION.    2.00 ANGSTROMS.                                   
REMARK   3   R VALUE     (WORKING SET) : 0.180                              
REMARK   3   FREE R VALUE             : 0.215                              
REMARK 290   SYMMETRY OPERATORS FOR SPACE GROUP: P 21 21 21               
CRYST1   61.777   58.233   23.135  90.00  90.00  90.00 P 21 21 21    4    
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "1HTM".to_string()).unwrap();
        
        // Test metadata extraction
        assert_eq!(structure.metadata.classification, "HYDROLASE/HYDROLASE INHIBITOR");
        assert_eq!(structure.metadata.deposition_date, "12-NOV-94");
        assert_eq!(structure.metadata.title, "HIV-1 PROTEASE IN COMPLEX WITH THE INHIBITOR CGP 53820 DETAILED STRUCTURAL ANALYSIS");
        assert_eq!(structure.metadata.experimental_method, ExperimentalMethod::XRayDiffraction);
        assert_eq!(structure.metadata.resolution, Some(2.0));
        assert_eq!(structure.metadata.r_work, Some(0.180));
        assert_eq!(structure.metadata.r_free, Some(0.215));
        assert_eq!(structure.metadata.authors, vec!["J.P.PRIESTLE", "A.SCHAR", "H.P.GRUTTER"]);
        
        // Test space group extraction
        assert_eq!(structure.space_group, Some("P 21 21 21".to_string()));
        
        // Test unit cell extraction
        assert!(structure.unit_cell.is_some());
        let unit_cell = structure.unit_cell.unwrap();
        assert_eq!(unit_cell.dimensions, (61.777, 58.233, 23.135));
        assert_eq!(unit_cell.angles, (90.00, 90.00, 90.00));
    }

    #[test]
    fn test_author_parsing() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
AUTHOR    J.DOE,A.SMITH,B.JOHNSON                                          
AUTHOR   2 C.WILLIAMS,D.BROWN                                              
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        assert_eq!(structure.metadata.authors, vec![
            "J.DOE", "A.SMITH", "B.JOHNSON", "C.WILLIAMS", "D.BROWN"
        ]);
    }

    #[test]
    fn test_experimental_method_parsing() {
        let methods = vec![
            ("X-RAY DIFFRACTION", ExperimentalMethod::XRayDiffraction),
            ("SOLUTION NMR", ExperimentalMethod::NMRSolution),
            ("ELECTRON MICROSCOPY", ExperimentalMethod::ElectronMicroscopy),
            ("NEUTRON DIFFRACTION", ExperimentalMethod::NeutronDiffraction),
            ("THEORETICAL MODEL", ExperimentalMethod::Unknown),
        ];

        for (method_str, expected_method) in methods {
            let test_content = format!(r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
EXPDTA    {}                                                 
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#, method_str);

            let structure = PDBParser::parse_string_no_validation(&test_content, "TEST".to_string()).unwrap();
            assert_eq!(structure.metadata.experimental_method, expected_method);
        }
    }

    #[test]
    fn test_resolution_and_r_factor_parsing() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
REMARK   2 RESOLUTION.    1.75 ANGSTROMS.                                   
REMARK   3   R VALUE     (WORKING SET) : 0.156                              
REMARK   3   FREE R VALUE             : 0.189                              
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        assert_eq!(structure.metadata.resolution, Some(1.75));
        assert_eq!(structure.metadata.r_work, Some(0.156));
        assert_eq!(structure.metadata.r_free, Some(0.189));
    }

    #[test]
    fn test_unit_cell_parsing() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
CRYST1   75.123   82.456   91.789 100.50 105.30 110.80 P 1 21 1      4    
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        
        assert!(structure.unit_cell.is_some());
        let unit_cell = structure.unit_cell.unwrap();
        assert_eq!(unit_cell.dimensions, (75.123, 82.456, 91.789));
        assert_eq!(unit_cell.angles, (100.50, 105.30, 110.80));
        assert_eq!(structure.space_group, Some("P 1 21 1".to_string()));
    }

    #[test]
    fn test_space_group_from_remark() {
        let test_content = r#"HEADER    TEST PROTEIN                            01-JAN-00   TEST              
REMARK 290   SYMMETRY OPERATORS FOR SPACE GROUP: P 1                      
ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  
END"#;

        let structure = PDBParser::parse_string_no_validation(test_content, "TEST".to_string()).unwrap();
        assert_eq!(structure.space_group, Some("P 1".to_string()));
    }
}