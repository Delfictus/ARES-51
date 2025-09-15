//! Atomic structure definitions for 3D protein structures

use std::fmt;

/// Atom representation with 3D coordinates and chemical properties
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub name: String,           // Atom name: "CA", "N", "C", "O", "CB", etc.
    pub residue: String,        // Amino acid name: "ALA", "GLY", "PRO", etc.
    pub residue_id: i32,        // Residue sequence number: 1, 2, 3, ...
    pub chain_id: char,         // Chain identifier: 'A', 'B', etc.
    pub x: f64,                 // X coordinate in Angstroms
    pub y: f64,                 // Y coordinate in Angstroms
    pub z: f64,                 // Z coordinate in Angstroms
    pub occupancy: f64,         // Occupancy factor (usually 1.00)
    pub b_factor: f64,          // Temperature factor
    pub element: Element,       // Chemical element
    pub atom_type: AtomType,    // Backbone or side chain classification
}

impl Atom {
    pub fn new(
        name: String,
        residue: String,
        residue_id: i32,
        x: f64,
        y: f64,
        z: f64,
        element: Element,
        atom_type: AtomType,
    ) -> Self {
        Self {
            name,
            residue,
            residue_id,
            chain_id: 'A',
            x,
            y,
            z,
            occupancy: 1.0,
            b_factor: 20.0,
            element,
            atom_type,
        }
    }

    /// Create a backbone atom (N, CA, C, O)
    pub fn backbone(
        name: String,
        residue: String,
        residue_id: i32,
        x: f64,
        y: f64,
        z: f64,
        element: Element,
    ) -> Self {
        Self::new(name, residue, residue_id, x, y, z, element, AtomType::Backbone)
    }

    /// Create a side chain atom
    pub fn sidechain(
        name: String,
        residue: String,
        residue_id: i32,
        x: f64,
        y: f64,
        z: f64,
        element: Element,
    ) -> Self {
        Self::new(name, residue, residue_id, x, y, z, element, AtomType::SideChain)
    }

    /// Get position as 3D vector
    pub fn position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Set position from 3D vector
    pub fn set_position(&mut self, pos: [f64; 3]) {
        self.x = pos[0];
        self.y = pos[1];
        self.z = pos[2];
    }

    /// Calculate distance to another atom
    pub fn distance_to(&self, other: &Atom) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Check if this is a backbone atom
    pub fn is_backbone(&self) -> bool {
        matches!(self.atom_type, AtomType::Backbone)
    }

    /// Check if this is a side chain atom
    pub fn is_sidechain(&self) -> bool {
        matches!(self.atom_type, AtomType::SideChain)
    }

    /// Check if this is a CA atom
    pub fn is_ca(&self) -> bool {
        self.name == "CA" && self.is_backbone()
    }

    /// Check if this is an N atom
    pub fn is_n(&self) -> bool {
        self.name == "N" && self.is_backbone()
    }

    /// Check if this is a C atom
    pub fn is_c(&self) -> bool {
        self.name == "C" && self.is_backbone()
    }
}

/// Chemical elements commonly found in proteins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Hydrogen,
    Phosphorus,
    Iron,
    Zinc,
    Magnesium,
    Calcium,
}

impl Element {
    /// Get element symbol for PDB format
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::Carbon => "C",
            Element::Nitrogen => "N",
            Element::Oxygen => "O",
            Element::Sulfur => "S",
            Element::Hydrogen => "H",
            Element::Phosphorus => "P",
            Element::Iron => "FE",
            Element::Zinc => "ZN",
            Element::Magnesium => "MG",
            Element::Calcium => "CA",
        }
    }

    /// Get atomic number
    pub fn atomic_number(&self) -> u8 {
        match self {
            Element::Hydrogen => 1,
            Element::Carbon => 6,
            Element::Nitrogen => 7,
            Element::Oxygen => 8,
            Element::Phosphorus => 15,
            Element::Sulfur => 16,
            Element::Calcium => 20,
            Element::Iron => 26,
            Element::Zinc => 30,
            Element::Magnesium => 12,
        }
    }

    /// Determine element from atom name (PDB convention)
    pub fn from_atom_name(name: &str) -> Self {
        match name.chars().next().unwrap_or('C') {
            'C' => Element::Carbon,
            'N' => Element::Nitrogen,
            'O' => Element::Oxygen,
            'S' => Element::Sulfur,
            'H' => Element::Hydrogen,
            'P' => Element::Phosphorus,
            'F' if name.starts_with("FE") => Element::Iron,
            'Z' if name.starts_with("ZN") => Element::Zinc,
            'M' if name.starts_with("MG") => Element::Magnesium,
            _ => Element::Carbon, // Default to carbon
        }
    }

    /// Van der Waals radius in Angstroms
    pub fn vdw_radius(&self) -> f64 {
        match self {
            Element::Hydrogen => 1.20,
            Element::Carbon => 1.70,
            Element::Nitrogen => 1.55,
            Element::Oxygen => 1.52,
            Element::Sulfur => 1.80,
            Element::Phosphorus => 1.80,
            Element::Calcium => 2.31,
            Element::Iron => 2.00,
            Element::Zinc => 1.39,
            Element::Magnesium => 1.73,
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Classification of atom types in protein structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    Backbone,   // N, CA, C, O atoms forming protein backbone
    SideChain,  // All other atoms in amino acid residues
}

/// Backbone atom specifically (N, CA, C, O)
#[derive(Debug, Clone, PartialEq)]
pub struct BackboneAtom {
    pub atom: Atom,
    pub phi: Option<f64>,    // Phi dihedral angle
    pub psi: Option<f64>,    // Psi dihedral angle
    pub omega: Option<f64>,  // Omega dihedral angle (usually ~180°)
}

impl BackboneAtom {
    pub fn new(atom: Atom) -> Self {
        Self {
            atom,
            phi: None,
            psi: None,
            omega: None,
        }
    }

    pub fn with_dihedrals(mut self, phi: f64, psi: f64, omega: f64) -> Self {
        self.phi = Some(phi);
        self.psi = Some(psi);
        self.omega = Some(omega);
        self
    }

    /// Get the underlying atom
    pub fn atom(&self) -> &Atom {
        &self.atom
    }

    /// Get mutable reference to underlying atom
    pub fn atom_mut(&mut self) -> &mut Atom {
        &mut self.atom
    }

    /// Get position as 3D vector
    pub fn position(&self) -> [f64; 3] {
        self.atom.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let atom = Atom::backbone(
            "CA".to_string(),
            "ALA".to_string(),
            1,
            1.0, 2.0, 3.0,
            Element::Carbon,
        );

        assert_eq!(atom.name, "CA");
        assert_eq!(atom.residue, "ALA");
        assert_eq!(atom.residue_id, 1);
        assert_eq!(atom.position(), [1.0, 2.0, 3.0]);
        assert!(atom.is_backbone());
        assert!(atom.is_ca());
    }

    #[test]
    fn test_element_from_atom_name() {
        assert_eq!(Element::from_atom_name("CA"), Element::Carbon);
        assert_eq!(Element::from_atom_name("N"), Element::Nitrogen);
        assert_eq!(Element::from_atom_name("O"), Element::Oxygen);
        assert_eq!(Element::from_atom_name("SG"), Element::Sulfur);
    }

    #[test]
    fn test_distance_calculation() {
        let atom1 = Atom::backbone(
            "CA".to_string(), "ALA".to_string(), 1,
            0.0, 0.0, 0.0, Element::Carbon,
        );
        let atom2 = Atom::backbone(
            "N".to_string(), "ALA".to_string(), 1,
            1.0, 1.0, 1.0, Element::Nitrogen,
        );

        let distance = atom1.distance_to(&atom2);
        assert!((distance - 3_f64.sqrt()).abs() < 1e-10);
    }
}