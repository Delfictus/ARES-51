// Protein chain data structures
use crate::geometry::{Residue, AminoAcid, Atom, Vector3};
use std::collections::HashMap;
use std::fmt;

/// Protein chain containing residues
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    /// Chain identifier (single character)
    pub id: char,
    
    /// Ordered list of residues
    pub residues: Vec<Residue>,
    
    /// Map from sequence number to residue index for fast lookup
    residue_map: HashMap<i32, usize>,
    
    /// Chain type (protein, DNA, RNA, etc.)
    pub chain_type: ChainType,
    
    /// Chain sequence as string (from ATOM records)
    pub sequence: String,
    
    /// SEQRES sequence from PDB header (authoritative sequence)
    pub seqres_sequence: Option<String>,
    
    /// Chain length (number of residues)
    pub length: usize,
}

/// Types of molecular chains
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainType {
    Protein,
    DNA,
    RNA,
    Ligand,
    Water,
    Ion,
    Unknown,
}

/// Validation result comparing SEQRES and ATOM-derived sequences
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceValidation {
    /// No SEQRES sequence available for comparison
    NoSeqres,
    
    /// Perfect match between SEQRES and ATOM sequences
    Perfect,
    
    /// ATOM sequence is missing residues compared to SEQRES
    MissingResidues(usize),
    
    /// Sequences are inconsistent
    Inconsistent {
        seqres_length: usize,
        atom_length: usize,
        matches: usize,
        mismatches: usize,
        length_difference: usize,
    },
}

impl Chain {
    /// Create new empty chain
    pub fn new(id: char, chain_type: ChainType) -> Self {
        Chain {
            id,
            residues: Vec::new(),
            residue_map: HashMap::new(),
            chain_type,
            sequence: String::new(),
            seqres_sequence: None,
            length: 0,
        }
    }

    /// Create protein chain
    pub fn protein(id: char) -> Self {
        Self::new(id, ChainType::Protein)
    }

    /// Add residue to chain
    pub fn add_residue(&mut self, residue: Residue) {
        let seq_num = residue.seq_num;
        let index = self.residues.len();
        
        self.residues.push(residue);
        self.residue_map.insert(seq_num, index);
        self.length = self.residues.len();
        
        // Update sequence string
        self.update_sequence();
    }

    /// Insert residue at specific position
    pub fn insert_residue(&mut self, index: usize, residue: Residue) {
        if index <= self.residues.len() {
            self.residues.insert(index, residue);
            self.rebuild_residue_map();
            self.update_sequence();
        }
    }

    /// Remove residue by sequence number
    pub fn remove_residue(&mut self, seq_num: i32) -> Option<Residue> {
        if let Some(&index) = self.residue_map.get(&seq_num) {
            let residue = self.residues.remove(index);
            self.rebuild_residue_map();
            self.update_sequence();
            Some(residue)
        } else {
            None
        }
    }

    /// Get residue by sequence number
    pub fn get_residue(&self, seq_num: i32) -> Option<&Residue> {
        self.residue_map.get(&seq_num)
            .and_then(|&index| self.residues.get(index))
    }

    /// Get mutable residue by sequence number
    pub fn get_residue_mut(&mut self, seq_num: i32) -> Option<&mut Residue> {
        if let Some(&index) = self.residue_map.get(&seq_num) {
            self.residues.get_mut(index)
        } else {
            None
        }
    }

    /// Get residue by index
    pub fn get_residue_by_index(&self, index: usize) -> Option<&Residue> {
        self.residues.get(index)
    }

    /// Get mutable residue by index
    pub fn get_residue_by_index_mut(&mut self, index: usize) -> Option<&mut Residue> {
        self.residues.get_mut(index)
    }

    /// Get first residue
    pub fn first_residue(&self) -> Option<&Residue> {
        self.residues.first()
    }

    /// Get last residue
    pub fn last_residue(&self) -> Option<&Residue> {
        self.residues.last()
    }

    /// Check if chain contains residue with sequence number
    pub fn has_residue(&self, seq_num: i32) -> bool {
        self.residue_map.contains_key(&seq_num)
    }

    /// Get all residues
    pub fn residues(&self) -> &[Residue] {
        &self.residues
    }

    /// Get all residues mutably
    pub fn residues_mut(&mut self) -> &mut [Residue] {
        &mut self.residues
    }

    /// Get residue iterator
    pub fn iter(&self) -> impl Iterator<Item = &Residue> {
        self.residues.iter()
    }

    /// Get mutable residue iterator
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Residue> {
        self.residues.iter_mut()
    }

    /// Get all atoms in chain
    pub fn atoms(&self) -> Vec<&Atom> {
        self.residues.iter()
            .flat_map(|residue| residue.atoms())
            .collect()
    }

    /// Get all CA atoms (alpha carbons) in order
    pub fn ca_atoms(&self) -> Vec<&Atom> {
        self.residues.iter()
            .filter_map(|residue| residue.ca_atom())
            .collect()
    }

    /// Get CA coordinates as vector
    pub fn ca_coordinates(&self) -> Vec<Vector3> {
        self.residues.iter()
            .filter_map(|residue| residue.ca_coords())
            .collect()
    }

    /// Get backbone atoms for each residue
    pub fn backbone_atoms(&self) -> Vec<Vec<Option<&Atom>>> {
        self.residues.iter()
            .map(|residue| residue.backbone_atoms())
            .collect()
    }

    /// Get heavy atoms only
    pub fn heavy_atoms(&self) -> Vec<&Atom> {
        self.residues.iter()
            .flat_map(|residue| residue.heavy_atoms())
            .collect()
    }

    /// Calculate chain length (number of residues)
    pub fn len(&self) -> usize {
        self.residues.len()
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }

    /// Get total atom count
    pub fn atom_count(&self) -> usize {
        self.residues.iter()
            .map(|residue| residue.atom_count())
            .sum()
    }

    /// Get heavy atom count
    pub fn heavy_atom_count(&self) -> usize {
        self.residues.iter()
            .map(|residue| residue.heavy_atom_count())
            .sum()
    }

    /// Calculate molecular weight
    pub fn molecular_weight(&self) -> f64 {
        self.residues.iter()
            .map(|residue| residue.molecular_weight())
            .sum()
    }

    /// Get sequence as one-letter codes
    pub fn sequence_one_letter(&self) -> String {
        self.residues.iter()
            .map(|residue| residue.amino_acid.one_letter())
            .collect()
    }

    /// Get sequence as three-letter codes
    pub fn sequence_three_letter(&self) -> Vec<String> {
        self.residues.iter()
            .map(|residue| residue.amino_acid.three_letter().to_string())
            .collect()
    }

    /// Set SEQRES sequence from PDB header
    pub fn set_seqres_sequence(&mut self, seqres: String) {
        self.seqres_sequence = Some(seqres);
    }

    /// Get SEQRES sequence (authoritative sequence from PDB header)
    pub fn seqres_sequence(&self) -> Option<&str> {
        self.seqres_sequence.as_deref()
    }

    /// Get authoritative sequence (SEQRES if available, otherwise from ATOM records)
    pub fn authoritative_sequence(&self) -> &str {
        self.seqres_sequence.as_deref().unwrap_or(&self.sequence)
    }

    /// Validate ATOM-derived sequence against SEQRES sequence
    pub fn validate_sequence_consistency(&self) -> SequenceValidation {
        let Some(seqres) = &self.seqres_sequence else {
            return SequenceValidation::NoSeqres;
        };

        let atom_seq = &self.sequence;
        
        if seqres == atom_seq {
            return SequenceValidation::Perfect;
        }

        // Check if ATOM sequence is a subsequence of SEQRES (common case)
        if seqres.contains(atom_seq) {
            let missing_count = seqres.len() - atom_seq.len();
            return SequenceValidation::MissingResidues(missing_count);
        }

        // Count mismatches
        let mut matches = 0;
        let mut mismatches = 0;
        let min_len = seqres.len().min(atom_seq.len());
        
        for (seqres_char, atom_char) in seqres.chars().zip(atom_seq.chars()) {
            if seqres_char == atom_char {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }

        let length_diff = (seqres.len() as i32 - atom_seq.len() as i32).abs() as usize;
        
        SequenceValidation::Inconsistent {
            seqres_length: seqres.len(),
            atom_length: atom_seq.len(),
            matches,
            mismatches,
            length_difference: length_diff,
        }
    }

    /// Update sequence string from residues
    fn update_sequence(&mut self) {
        self.sequence = self.sequence_one_letter();
        self.length = self.residues.len();
    }

    /// Rebuild residue map after modifications
    fn rebuild_residue_map(&mut self) {
        self.residue_map.clear();
        for (index, residue) in self.residues.iter().enumerate() {
            self.residue_map.insert(residue.seq_num, index);
        }
        self.length = self.residues.len();
    }

    /// Calculate center of mass for entire chain
    pub fn center_of_mass(&self) -> Vector3 {
        let mut total_mass = 0.0;
        let mut weighted_sum = Vector3::zero();

        for atom in self.atoms() {
            let mass = atom.atomic_mass();
            total_mass += mass;
            weighted_sum += atom.coords * mass;
        }

        if total_mass > 0.0 {
            weighted_sum / total_mass
        } else {
            Vector3::zero()
        }
    }

    /// Calculate geometric center
    pub fn geometric_center(&self) -> Vector3 {
        let atoms = self.atoms();
        if atoms.is_empty() {
            return Vector3::zero();
        }

        let sum: Vector3 = atoms.iter()
            .map(|atom| atom.coords)
            .fold(Vector3::zero(), |acc, coords| acc + coords);
            
        sum / atoms.len() as f64
    }

    /// Get bounding box (min and max coordinates)
    pub fn bounding_box(&self) -> Option<(Vector3, Vector3)> {
        let atoms = self.atoms();
        if atoms.is_empty() {
            return None;
        }

        let first_coords = atoms[0].coords;
        let mut min_coords = first_coords;
        let mut max_coords = first_coords;

        for atom in &atoms[1..] {
            min_coords = min_coords.component_min(&atom.coords);
            max_coords = max_coords.component_max(&atom.coords);
        }

        Some((min_coords, max_coords))
    }

    /// Get chain radius (maximum distance from center)
    pub fn radius(&self) -> f64 {
        let center = self.geometric_center();
        self.atoms().iter()
            .map(|atom| center.distance(&atom.coords))
            .fold(0.0, f64::max)
    }

    /// Get consecutive residue pairs for calculating dihedral angles
    pub fn residue_pairs(&self) -> Vec<(&Residue, &Residue)> {
        self.residues.windows(2)
            .map(|pair| (&pair[0], &pair[1]))
            .collect()
    }

    /// Get consecutive residue triplets for angle calculations
    pub fn residue_triplets(&self) -> Vec<(&Residue, &Residue, &Residue)> {
        self.residues.windows(3)
            .map(|triplet| (&triplet[0], &triplet[1], &triplet[2]))
            .collect()
    }

    /// Get consecutive residue quadruplets for dihedral calculations
    pub fn residue_quadruplets(&self) -> Vec<(&Residue, &Residue, &Residue, &Residue)> {
        self.residues.windows(4)
            .map(|quad| (&quad[0], &quad[1], &quad[2], &quad[3]))
            .collect()
    }

    /// Calculate phi/psi angles for all residues
    pub fn calculate_backbone_dihedrals(&mut self) {
        for i in 1..self.residues.len()-1 {
            if let Some(phi) = self.calculate_phi_angle(i) {
                self.residues[i].set_phi(phi);
            }
            if let Some(psi) = self.calculate_psi_angle(i) {
                self.residues[i].set_psi(psi);
            }
        }
    }

    /// Calculate phi angle for residue at index
    fn calculate_phi_angle(&self, index: usize) -> Option<f64> {
        if index == 0 || index >= self.residues.len() {
            return None;
        }

        let prev_res = &self.residues[index - 1];
        let curr_res = &self.residues[index];

        let c_prev = prev_res.get_atom("C")?;
        let n_curr = curr_res.get_atom("N")?;
        let ca_curr = curr_res.get_atom("CA")?;
        let c_curr = curr_res.get_atom("C")?;

        Some(Vector3::dihedral_angle(
            c_prev.coords,
            n_curr.coords,
            ca_curr.coords,
            c_curr.coords,
        ))
    }

    /// Calculate psi angle for residue at index
    fn calculate_psi_angle(&self, index: usize) -> Option<f64> {
        if index >= self.residues.len() - 1 {
            return None;
        }

        let curr_res = &self.residues[index];
        let next_res = &self.residues[index + 1];

        let n_curr = curr_res.get_atom("N")?;
        let ca_curr = curr_res.get_atom("CA")?;
        let c_curr = curr_res.get_atom("C")?;
        let n_next = next_res.get_atom("N")?;

        Some(Vector3::dihedral_angle(
            n_curr.coords,
            ca_curr.coords,
            c_curr.coords,
            n_next.coords,
        ))
    }

    /// Get secondary structure sequence
    pub fn secondary_structure_sequence(&self) -> String {
        self.residues.iter()
            .map(|residue| residue.secondary_structure_string())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Count residues by amino acid type
    pub fn amino_acid_composition(&self) -> HashMap<AminoAcid, usize> {
        let mut composition = HashMap::new();
        
        for residue in &self.residues {
            *composition.entry(residue.amino_acid).or_insert(0) += 1;
        }
        
        composition
    }

    /// Check if chain has complete backbone
    pub fn has_complete_backbone(&self) -> bool {
        self.residues.iter()
            .all(|residue| residue.has_complete_backbone())
    }

    /// Get residues with incomplete backbone
    pub fn incomplete_backbone_residues(&self) -> Vec<&Residue> {
        self.residues.iter()
            .filter(|residue| !residue.has_complete_backbone())
            .collect()
    }

    /// Validate chain structure
    pub fn validate(&self) -> Vec<ChainValidationError> {
        let mut errors = Vec::new();

        if self.residues.is_empty() {
            errors.push(ChainValidationError::EmptyChain);
            return errors;
        }

        // Check for sequence numbering issues
        let mut prev_seq_num = None;
        for residue in &self.residues {
            if let Some(prev) = prev_seq_num {
                if residue.seq_num <= prev {
                    errors.push(ChainValidationError::NonIncreasingSequenceNumbers);
                    break;
                }
            }
            prev_seq_num = Some(residue.seq_num);
        }

        // Check individual residues
        for (i, residue) in self.residues.iter().enumerate() {
            let residue_errors = residue.validate();
            for error in residue_errors {
                errors.push(ChainValidationError::ResidueError(i, error));
            }
        }

        // Check chain connectivity
        if !self.has_reasonable_connectivity() {
            errors.push(ChainValidationError::PoorConnectivity);
        }

        errors
    }

    /// Check if consecutive residues are reasonably connected
    fn has_reasonable_connectivity(&self) -> bool {
        for pair in self.residue_pairs() {
            let (res1, res2) = pair;
            
            if let (Some(c1), Some(n2)) = (res1.get_atom("C"), res2.get_atom("N")) {
                let distance = c1.distance_to(n2);
                // Peptide bond should be approximately 1.33 Å
                if distance < 1.0 || distance > 2.0 {
                    return false;
                }
            } else {
                return false; // Missing backbone atoms
            }
        }
        true
    }

    /// Sort residues by sequence number
    pub fn sort_by_sequence_number(&mut self) {
        self.residues.sort_by_key(|residue| residue.seq_num);
        self.rebuild_residue_map();
    }

    /// Get N-terminal residue
    pub fn n_terminus(&self) -> Option<&Residue> {
        self.residues.first()
    }

    /// Get C-terminal residue
    pub fn c_terminus(&self) -> Option<&Residue> {
        self.residues.last()
    }
}

/// Chain validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ChainValidationError {
    EmptyChain,
    NonIncreasingSequenceNumbers,
    ResidueError(usize, crate::geometry::residue::ResidueValidationError),
    PoorConnectivity,
    MissingTerminus,
}

impl fmt::Display for ChainValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainValidationError::EmptyChain => {
                write!(f, "Chain is empty")
            }
            ChainValidationError::NonIncreasingSequenceNumbers => {
                write!(f, "Sequence numbers are not increasing")
            }
            ChainValidationError::ResidueError(index, error) => {
                write!(f, "Error in residue {}: {}", index, error)
            }
            ChainValidationError::PoorConnectivity => {
                write!(f, "Poor connectivity between consecutive residues")
            }
            ChainValidationError::MissingTerminus => {
                write!(f, "Missing N-terminus or C-terminus")
            }
        }
    }
}

impl std::error::Error for ChainValidationError {}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Chain {} ({:?}): {} residues, {} atoms",
            self.id,
            self.chain_type,
            self.length,
            self.atom_count()
        )
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ChainType::Protein => "Protein",
            ChainType::DNA => "DNA",
            ChainType::RNA => "RNA",
            ChainType::Ligand => "Ligand",
            ChainType::Water => "Water",
            ChainType::Ion => "Ion",
            ChainType::Unknown => "Unknown",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Element, Vector3, Atom};

    fn create_test_residue(seq_num: i32, aa: AminoAcid) -> Residue {
        let mut residue = Residue::new(seq_num, aa, 'A', None);
        
        // Add basic backbone atoms
        let n = Atom::new(1, "N".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::N, 'A', seq_num);
        let ca = Atom::new(2, "CA".to_string(), Vector3::new(1.0, 0.0, 0.0), Element::C, 'A', seq_num);
        let c = Atom::new(3, "C".to_string(), Vector3::new(2.0, 0.0, 0.0), Element::C, 'A', seq_num);
        let o = Atom::new(4, "O".to_string(), Vector3::new(3.0, 0.0, 0.0), Element::O, 'A', seq_num);
        
        residue.add_atom(n);
        residue.add_atom(ca);
        residue.add_atom(c);
        residue.add_atom(o);
        
        residue
    }

    fn create_test_chain() -> Chain {
        let mut chain = Chain::protein('A');
        
        chain.add_residue(create_test_residue(1, AminoAcid::Met));
        chain.add_residue(create_test_residue(2, AminoAcid::Ala));
        chain.add_residue(create_test_residue(3, AminoAcid::Gly));
        
        chain
    }

    #[test]
    fn test_chain_creation() {
        let chain = Chain::protein('A');
        assert_eq!(chain.id, 'A');
        assert_eq!(chain.chain_type, ChainType::Protein);
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_residue_management() {
        let mut chain = create_test_chain();
        
        assert_eq!(chain.len(), 3);
        assert!(!chain.is_empty());
        assert!(chain.has_residue(1));
        assert!(chain.has_residue(2));
        assert!(chain.has_residue(3));
        assert!(!chain.has_residue(4));
        
        let first_residue = chain.get_residue(1).unwrap();
        assert_eq!(first_residue.amino_acid, AminoAcid::Met);
        
        let removed = chain.remove_residue(2);
        assert!(removed.is_some());
        assert_eq!(chain.len(), 2);
        assert!(!chain.has_residue(2));
    }

    #[test]
    fn test_sequence_operations() {
        let chain = create_test_chain();
        
        assert_eq!(chain.sequence_one_letter(), "MAG");
        assert_eq!(chain.sequence_three_letter(), vec!["MET", "ALA", "GLY"]);
        
        let composition = chain.amino_acid_composition();
        assert_eq!(composition.get(&AminoAcid::Met), Some(&1));
        assert_eq!(composition.get(&AminoAcid::Ala), Some(&1));
        assert_eq!(composition.get(&AminoAcid::Gly), Some(&1));
    }

    #[test]
    fn test_atom_access() {
        let chain = create_test_chain();
        
        assert_eq!(chain.atom_count(), 12); // 3 residues × 4 atoms each
        assert_eq!(chain.heavy_atom_count(), 12); // No hydrogens in test
        
        let ca_atoms = chain.ca_atoms();
        assert_eq!(ca_atoms.len(), 3);
        
        let ca_coords = chain.ca_coordinates();
        assert_eq!(ca_coords.len(), 3);
        assert_eq!(ca_coords[0], Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_geometric_properties() {
        let chain = create_test_chain();
        
        let center = chain.geometric_center();
        assert!(center.x > 0.0 && center.x < 3.0);
        
        let (min_coords, max_coords) = chain.bounding_box().unwrap();
        assert_eq!(min_coords.x, 0.0);
        assert_eq!(max_coords.x, 3.0);
        
        let radius = chain.radius();
        assert!(radius > 0.0);
        
        let mw = chain.molecular_weight();
        assert!(mw > 200.0); // Should be sum of amino acid molecular weights
    }

    #[test]
    fn test_structural_analysis() {
        let chain = create_test_chain();
        
        let pairs = chain.residue_pairs();
        assert_eq!(pairs.len(), 2); // 3 residues → 2 pairs
        
        let triplets = chain.residue_triplets();
        assert_eq!(triplets.len(), 1); // 3 residues → 1 triplet
        
        let quadruplets = chain.residue_quadruplets();
        assert_eq!(quadruplets.len(), 0); // Need 4+ residues
    }

    #[test]
    fn test_validation() {
        let chain = create_test_chain();
        let errors = chain.validate();
        
        // Should be valid chain
        let has_serious_errors = errors.iter().any(|e| {
            matches!(e, 
                ChainValidationError::EmptyChain | 
                ChainValidationError::NonIncreasingSequenceNumbers |
                ChainValidationError::PoorConnectivity
            )
        });
        assert!(!has_serious_errors);
    }

    #[test]
    fn test_terminus_identification() {
        let chain = create_test_chain();
        
        let n_term = chain.n_terminus().unwrap();
        assert_eq!(n_term.amino_acid, AminoAcid::Met);
        assert_eq!(n_term.seq_num, 1);
        
        let c_term = chain.c_terminus().unwrap();
        assert_eq!(c_term.amino_acid, AminoAcid::Gly);
        assert_eq!(c_term.seq_num, 3);
    }
}