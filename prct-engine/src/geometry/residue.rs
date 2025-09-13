// Residue data structures for protein sequences
use crate::geometry::{Atom, AminoAcid, Vector3};
use std::collections::HashMap;
use std::fmt;

/// Amino acid residue containing atoms
#[derive(Debug, Clone, PartialEq)]
pub struct Residue {
    /// Residue identifier
    pub id: String,
    
    /// Sequence number from PDB
    pub seq_num: i32,
    
    /// Insertion code (if any)
    pub insertion_code: Option<char>,
    
    /// Amino acid type
    pub amino_acid: AminoAcid,
    
    /// Chain identifier this residue belongs to
    pub chain_id: char,
    
    /// Map of atom name to atom
    pub atoms: HashMap<String, Atom>,
    
    /// Secondary structure assignment
    pub secondary_structure: SecondaryStructure,
    
    /// Phi angle (backbone dihedral)
    pub phi: Option<f64>,
    
    /// Psi angle (backbone dihedral)  
    pub psi: Option<f64>,
    
    /// Chi angles (side chain dihedrals)
    pub chi_angles: Vec<f64>,
    
    /// Accessible surface area
    pub accessible_surface_area: Option<f64>,
}

/// Secondary structure types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryStructure {
    /// Alpha helix
    Helix,
    
    /// Beta sheet/strand
    Sheet,
    
    /// 3-10 helix
    Helix310,
    
    /// Pi helix
    HelixPi,
    
    /// Beta turn
    Turn,
    
    /// Random coil/loop
    Coil,
    
    /// Bridge (isolated beta strand)
    Bridge,
    
    /// Unknown/unassigned
    Unknown,
}

impl Residue {
    /// Create new residue
    pub fn new(
        seq_num: i32,
        amino_acid: AminoAcid,
        chain_id: char,
        insertion_code: Option<char>,
    ) -> Self {
        let id = format!("{}:{}{}", 
            chain_id, 
            seq_num, 
            insertion_code.map(|c| c.to_string()).unwrap_or_default()
        );

        Residue {
            id,
            seq_num,
            insertion_code,
            amino_acid,
            chain_id,
            atoms: HashMap::new(),
            secondary_structure: SecondaryStructure::Unknown,
            phi: None,
            psi: None,
            chi_angles: Vec::new(),
            accessible_surface_area: None,
        }
    }

    /// Add atom to residue
    pub fn add_atom(&mut self, atom: Atom) {
        self.atoms.insert(atom.name.clone(), atom);
    }

    /// Get atom by name
    pub fn get_atom(&self, name: &str) -> Option<&Atom> {
        self.atoms.get(name)
    }

    /// Get mutable atom by name
    pub fn get_atom_mut(&mut self, name: &str) -> Option<&mut Atom> {
        self.atoms.get_mut(name)
    }

    /// Check if residue has atom with given name
    pub fn has_atom(&self, name: &str) -> bool {
        self.atoms.contains_key(name)
    }

    /// Remove atom by name
    pub fn remove_atom(&mut self, name: &str) -> Option<Atom> {
        self.atoms.remove(name)
    }

    /// Get all atoms
    pub fn atoms(&self) -> impl Iterator<Item = &Atom> {
        self.atoms.values()
    }

    /// Get all atoms mutably
    pub fn atoms_mut(&mut self) -> impl Iterator<Item = &mut Atom> {
        self.atoms.values_mut()
    }

    /// Get backbone atoms in order (N, CA, C, O)
    pub fn backbone_atoms(&self) -> Vec<Option<&Atom>> {
        vec![
            self.get_atom("N"),
            self.get_atom("CA"),
            self.get_atom("C"),
            self.get_atom("O"),
        ]
    }

    /// Get side chain atoms
    pub fn side_chain_atoms(&self) -> Vec<&Atom> {
        self.atoms()
            .filter(|atom| atom.is_side_chain())
            .collect()
    }

    /// Get heavy atoms (non-hydrogen)
    pub fn heavy_atoms(&self) -> Vec<&Atom> {
        self.atoms()
            .filter(|atom| atom.is_heavy_atom())
            .collect()
    }

    /// Check if residue has complete backbone
    pub fn has_complete_backbone(&self) -> bool {
        ["N", "CA", "C", "O"].iter()
            .all(|&name| self.has_atom(name))
    }

    /// Check if residue has minimal backbone (N, CA, C)
    pub fn has_minimal_backbone(&self) -> bool {
        ["N", "CA", "C"].iter()
            .all(|&name| self.has_atom(name))
    }

    /// Get CA atom (alpha carbon)
    pub fn ca_atom(&self) -> Option<&Atom> {
        self.get_atom("CA")
    }

    /// Get CA coordinates
    pub fn ca_coords(&self) -> Option<Vector3> {
        self.ca_atom().map(|atom| atom.coords)
    }

    /// Get CB atom (beta carbon, or CA for glycine)
    pub fn cb_atom(&self) -> Option<&Atom> {
        if self.amino_acid == AminoAcid::Gly {
            self.get_atom("CA") // Glycine has no CB, use CA
        } else {
            self.get_atom("CB")
        }
    }

    /// Get CB coordinates
    pub fn cb_coords(&self) -> Option<Vector3> {
        self.cb_atom().map(|atom| atom.coords)
    }

    /// Calculate center of mass
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

    /// Calculate geometric center (centroid)
    pub fn geometric_center(&self) -> Vector3 {
        let atoms: Vec<_> = self.atoms().collect();
        if atoms.is_empty() {
            return Vector3::zero();
        }

        let sum: Vector3 = atoms.iter()
            .map(|atom| atom.coords)
            .fold(Vector3::zero(), |acc, coords| acc + coords);
            
        sum / atoms.len() as f64
    }

    /// Get residue radius (maximum distance from center to any atom)
    pub fn radius(&self) -> f64 {
        let center = self.geometric_center();
        self.atoms()
            .map(|atom| center.distance(&atom.coords))
            .fold(0.0, f64::max)
    }

    /// Calculate molecular weight
    pub fn molecular_weight(&self) -> f64 {
        self.atoms()
            .map(|atom| atom.atomic_mass())
            .sum()
    }

    /// Get number of atoms
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Get number of heavy atoms
    pub fn heavy_atom_count(&self) -> usize {
        self.heavy_atoms().len()
    }

    /// Check if residue is terminal (N-terminal or C-terminal)
    pub fn is_terminal(&self) -> bool {
        self.has_atom("OXT") || // C-terminal oxygen
        self.atoms().any(|atom| atom.name.contains("H1") || atom.name.contains("H2") || atom.name.contains("H3")) // N-terminal hydrogens
    }

    /// Check if residue is N-terminal
    pub fn is_n_terminal(&self) -> bool {
        self.atoms().any(|atom| atom.name.contains("H1") || atom.name.contains("H2") || atom.name.contains("H3"))
    }

    /// Check if residue is C-terminal
    pub fn is_c_terminal(&self) -> bool {
        self.has_atom("OXT")
    }

    /// Set phi angle
    pub fn set_phi(&mut self, phi: f64) {
        self.phi = Some(phi);
    }

    /// Set psi angle
    pub fn set_psi(&mut self, psi: f64) {
        self.psi = Some(psi);
    }

    /// Get phi angle in degrees
    pub fn phi_degrees(&self) -> Option<f64> {
        self.phi.map(|angle| angle.to_degrees())
    }

    /// Get psi angle in degrees
    pub fn psi_degrees(&self) -> Option<f64> {
        self.psi.map(|angle| angle.to_degrees())
    }

    /// Add chi angle
    pub fn add_chi_angle(&mut self, chi: f64) {
        self.chi_angles.push(chi);
    }

    /// Get chi angles in degrees
    pub fn chi_angles_degrees(&self) -> Vec<f64> {
        self.chi_angles.iter()
            .map(|&angle| angle.to_degrees())
            .collect()
    }

    /// Check if residue is in allowed Ramachandran region
    pub fn is_ramachandran_allowed(&self) -> bool {
        if let (Some(phi), Some(psi)) = (self.phi, self.psi) {
            let phi_deg = phi.to_degrees();
            let psi_deg = psi.to_degrees();
            
            // Define allowed regions (approximate)
            // Alpha-helix region
            if phi_deg >= -120.0 && phi_deg <= -30.0 && psi_deg >= -80.0 && psi_deg <= 0.0 {
                return true;
            }
            
            // Beta-sheet region  
            if phi_deg >= -180.0 && phi_deg <= -90.0 && psi_deg >= 90.0 && psi_deg <= 180.0 {
                return true;
            }
            
            // Left-handed helix (mainly glycine)
            if self.amino_acid == AminoAcid::Gly && phi_deg >= 30.0 && phi_deg <= 90.0 && psi_deg >= -30.0 && psi_deg <= 90.0 {
                return true;
            }
            
            false
        } else {
            true // If angles not calculated, assume allowed
        }
    }

    /// Get secondary structure as string
    pub fn secondary_structure_string(&self) -> &'static str {
        match self.secondary_structure {
            SecondaryStructure::Helix => "H",
            SecondaryStructure::Sheet => "E", 
            SecondaryStructure::Helix310 => "G",
            SecondaryStructure::HelixPi => "I",
            SecondaryStructure::Turn => "T",
            SecondaryStructure::Coil => "C",
            SecondaryStructure::Bridge => "B",
            SecondaryStructure::Unknown => "-",
        }
    }

    /// Set secondary structure from DSSP code
    pub fn set_secondary_structure_from_dssp(&mut self, dssp_code: char) {
        self.secondary_structure = match dssp_code {
            'H' => SecondaryStructure::Helix,
            'B' => SecondaryStructure::Bridge,
            'E' => SecondaryStructure::Sheet,
            'G' => SecondaryStructure::Helix310,
            'I' => SecondaryStructure::HelixPi,
            'T' => SecondaryStructure::Turn,
            'S' => SecondaryStructure::Turn,
            ' ' | '-' => SecondaryStructure::Coil,
            _ => SecondaryStructure::Unknown,
        };
    }

    /// Validate residue structure
    pub fn validate(&self) -> Vec<ResidueValidationError> {
        let mut errors = Vec::new();

        // Check for required backbone atoms
        if !self.has_minimal_backbone() {
            errors.push(ResidueValidationError::IncompleteBackbone);
        }

        // Check atom coordinates
        for atom in self.atoms() {
            if !atom.has_valid_coordinates() {
                errors.push(ResidueValidationError::InvalidAtomCoordinates(atom.name.clone()));
            }
            
            if !atom.has_valid_b_factor() {
                errors.push(ResidueValidationError::InvalidBFactor(atom.name.clone()));
            }
            
            if !atom.has_valid_occupancy() {
                errors.push(ResidueValidationError::InvalidOccupancy(atom.name.clone()));
            }
        }

        // Check Ramachandran angles
        if !self.is_ramachandran_allowed() {
            errors.push(ResidueValidationError::DisallowedConformation);
        }

        errors
    }
}

/// Residue validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidueValidationError {
    IncompleteBackbone,
    InvalidAtomCoordinates(String),
    InvalidBFactor(String),
    InvalidOccupancy(String),
    DisallowedConformation,
    MissingRequiredAtom(String),
}

impl fmt::Display for ResidueValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResidueValidationError::IncompleteBackbone => {
                write!(f, "Incomplete backbone (missing N, CA, or C atoms)")
            }
            ResidueValidationError::InvalidAtomCoordinates(atom) => {
                write!(f, "Invalid coordinates for atom {}", atom)
            }
            ResidueValidationError::InvalidBFactor(atom) => {
                write!(f, "Invalid B-factor for atom {}", atom)
            }
            ResidueValidationError::InvalidOccupancy(atom) => {
                write!(f, "Invalid occupancy for atom {}", atom)
            }
            ResidueValidationError::DisallowedConformation => {
                write!(f, "Residue in disallowed Ramachandran region")
            }
            ResidueValidationError::MissingRequiredAtom(atom) => {
                write!(f, "Missing required atom {}", atom)
            }
        }
    }
}

impl std::error::Error for ResidueValidationError {}

impl fmt::Display for Residue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}:{} ({} atoms)",
            self.amino_acid.three_letter(),
            self.seq_num,
            self.chain_id,
            self.atom_count()
        )
    }
}

impl fmt::Display for SecondaryStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SecondaryStructure::Helix => "Helix",
            SecondaryStructure::Sheet => "Sheet",
            SecondaryStructure::Helix310 => "3-10 Helix",
            SecondaryStructure::HelixPi => "Pi Helix",
            SecondaryStructure::Turn => "Turn",
            SecondaryStructure::Coil => "Coil",
            SecondaryStructure::Bridge => "Bridge",
            SecondaryStructure::Unknown => "Unknown",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Element, Vector3};

    fn create_test_residue() -> Residue {
        let mut residue = Residue::new(1, AminoAcid::Ala, 'A', None);
        
        // Add backbone atoms
        let n = Atom::new(1, "N".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::N, 'A', 1);
        let ca = Atom::new(2, "CA".to_string(), Vector3::new(1.0, 0.0, 0.0), Element::C, 'A', 1);
        let c = Atom::new(3, "C".to_string(), Vector3::new(2.0, 0.0, 0.0), Element::C, 'A', 1);
        let o = Atom::new(4, "O".to_string(), Vector3::new(3.0, 0.0, 0.0), Element::O, 'A', 1);
        let cb = Atom::new(5, "CB".to_string(), Vector3::new(1.0, 1.0, 0.0), Element::C, 'A', 1);
        
        residue.add_atom(n);
        residue.add_atom(ca);
        residue.add_atom(c);
        residue.add_atom(o);
        residue.add_atom(cb);
        
        residue
    }

    #[test]
    fn test_residue_creation() {
        let residue = Residue::new(1, AminoAcid::Ala, 'A', None);
        assert_eq!(residue.seq_num, 1);
        assert_eq!(residue.amino_acid, AminoAcid::Ala);
        assert_eq!(residue.chain_id, 'A');
        assert_eq!(residue.id, "A:1");
    }

    #[test]
    fn test_atom_management() {
        let mut residue = create_test_residue();
        
        assert!(residue.has_atom("CA"));
        assert!(!residue.has_atom("CB2"));
        assert_eq!(residue.atom_count(), 5);
        
        let ca_atom = residue.get_atom("CA").unwrap();
        assert_eq!(ca_atom.element, Element::C);
        
        residue.remove_atom("CB");
        assert_eq!(residue.atom_count(), 4);
    }

    #[test]
    fn test_backbone_completeness() {
        let residue = create_test_residue();
        assert!(residue.has_complete_backbone());
        assert!(residue.has_minimal_backbone());
        
        let mut incomplete = Residue::new(1, AminoAcid::Ala, 'A', None);
        let ca = Atom::new(1, "CA".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::C, 'A', 1);
        incomplete.add_atom(ca);
        
        assert!(!incomplete.has_complete_backbone());
        assert!(!incomplete.has_minimal_backbone());
    }

    #[test]
    fn test_center_calculations() {
        let residue = create_test_residue();
        let geometric_center = residue.geometric_center();
        let center_of_mass = residue.center_of_mass();
        
        // Geometric center should be average of coordinates
        assert!((geometric_center.x - 1.4).abs() < 0.1);
        assert!((geometric_center.y - 0.2).abs() < 0.1);
        
        // Center of mass should be weighted by atomic masses
        assert!(center_of_mass.x > 0.0 && center_of_mass.x < 3.0);
    }

    #[test]
    fn test_secondary_structure() {
        let mut residue = create_test_residue();
        
        residue.set_secondary_structure_from_dssp('H');
        assert_eq!(residue.secondary_structure, SecondaryStructure::Helix);
        assert_eq!(residue.secondary_structure_string(), "H");
        
        residue.set_secondary_structure_from_dssp('E');
        assert_eq!(residue.secondary_structure, SecondaryStructure::Sheet);
    }

    #[test]
    fn test_dihedral_angles() {
        let mut residue = create_test_residue();
        
        residue.set_phi(-60.0_f64.to_radians());
        residue.set_psi(-45.0_f64.to_radians());
        
        assert!((residue.phi_degrees().unwrap() + 60.0).abs() < 1e-10);
        assert!((residue.psi_degrees().unwrap() + 45.0).abs() < 1e-10);
        assert!(residue.is_ramachandran_allowed());
    }

    #[test]
    fn test_residue_validation() {
        let residue = create_test_residue();
        let errors = residue.validate();
        
        // Should be valid with complete backbone and good coordinates
        assert!(errors.is_empty());
        
        // Test with incomplete residue
        let incomplete = Residue::new(1, AminoAcid::Ala, 'A', None);
        let errors = incomplete.validate();
        assert!(!errors.is_empty());
        assert!(errors.contains(&ResidueValidationError::IncompleteBackbone));
    }

    #[test]
    fn test_molecular_properties() {
        let residue = create_test_residue();
        
        let mw = residue.molecular_weight();
        assert!(mw > 50.0); // Should be sum of atomic masses
        
        let radius = residue.radius();
        assert!(radius > 0.0);
        
        assert_eq!(residue.heavy_atom_count(), 5); // N, CA, C, O, CB
    }
}