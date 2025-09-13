// Atomic data structures for protein representations
use crate::geometry::{Vector3, Element};
use std::fmt;

/// Atomic data from PDB files
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Unique atom identifier within structure
    pub id: usize,
    
    /// Serial number from PDB file
    pub serial: i32,
    
    /// Atom name (e.g., "CA", "N", "O")
    pub name: String,
    
    /// Alternative location indicator
    pub alt_loc: Option<char>,
    
    /// 3D coordinates in Angstroms
    pub coords: Vector3,
    
    /// Occupancy factor [0.0, 1.0]
    pub occupancy: f64,
    
    /// Temperature factor (B-factor) in Ų
    pub b_factor: f64,
    
    /// Chemical element
    pub element: Element,
    
    /// Formal charge (if present)
    pub charge: Option<i8>,
    
    /// Chain identifier this atom belongs to
    pub chain_id: char,
    
    /// Residue sequence number
    pub residue_seq: i32,
    
    /// Insertion code for residue
    pub insertion_code: Option<char>,
}

impl Atom {
    /// Create new atom with minimal required information
    pub fn new(
        id: usize,
        name: String,
        coords: Vector3,
        element: Element,
        chain_id: char,
        residue_seq: i32,
    ) -> Self {
        Atom {
            id,
            serial: id as i32,
            name,
            alt_loc: None,
            coords,
            occupancy: 1.0,
            b_factor: 20.0, // Default B-factor
            element,
            charge: None,
            chain_id,
            residue_seq,
            insertion_code: None,
        }
    }

    /// Create atom from PDB ATOM record line
    pub fn from_pdb_line(line: &str, id: usize) -> Result<Atom, PdbParseError> {
        if line.len() < 54 {
            return Err(PdbParseError::LineTooShort);
        }

        // Extract fields according to PDB format specification
        let serial = parse_int(&line[6..11])?;
        let name = line[12..16].trim().to_string();
        let alt_loc = parse_char(&line[16..17]);
        let chain_id = line[21..22].chars().next().unwrap_or(' ');
        let residue_seq = parse_int(&line[22..26])?;
        let insertion_code = parse_char(&line[26..27]);

        // Parse coordinates
        let x = parse_float(&line[30..38])?;
        let y = parse_float(&line[38..46])?;
        let z = parse_float(&line[46..54])?;
        let coords = Vector3::new(x, y, z);

        // Validate coordinate bounds
        if !coords.is_finite() || coords.magnitude() > 10000.0 {
            return Err(PdbParseError::InvalidCoordinates);
        }

        // Parse optional fields
        let occupancy = if line.len() >= 60 {
            parse_float(&line[54..60]).unwrap_or(1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let b_factor = if line.len() >= 66 {
            parse_float(&line[60..66]).unwrap_or(20.0).max(0.0)
        } else {
            20.0
        };

        // Determine element
        let element = if line.len() >= 78 {
            let element_str = line[76..78].trim();
            if !element_str.is_empty() {
                element_str.parse().unwrap_or_else(|_| Element::from_atom_name(&name))
            } else {
                Element::from_atom_name(&name)
            }
        } else {
            Element::from_atom_name(&name)
        };

        // Parse charge if present
        let charge = if line.len() >= 80 {
            parse_charge(&line[78..80])
        } else {
            None
        };

        Ok(Atom {
            id,
            serial,
            name,
            alt_loc,
            coords,
            occupancy,
            b_factor,
            element,
            charge,
            chain_id,
            residue_seq,
            insertion_code,
        })
    }

    /// Calculate distance to another atom
    pub fn distance_to(&self, other: &Atom) -> f64 {
        self.coords.distance(&other.coords)
    }

    /// Calculate squared distance (faster, no sqrt)
    pub fn distance_squared_to(&self, other: &Atom) -> f64 {
        self.coords.distance_squared(&other.coords)
    }

    /// Check if atom is in contact with another (within van der Waals radii + tolerance)
    pub fn is_in_contact(&self, other: &Atom, tolerance: f64) -> bool {
        let contact_distance = self.element.vdw_radius() + other.element.vdw_radius() + tolerance;
        self.distance_to(other) <= contact_distance
    }

    /// Check if atom is a backbone atom
    pub fn is_backbone(&self) -> bool {
        matches!(self.name.as_str(), "N" | "CA" | "C" | "O" | "OXT")
    }

    /// Check if atom is a side chain atom
    pub fn is_side_chain(&self) -> bool {
        !self.is_backbone() && self.name != "H"
    }

    /// Check if atom is hydrogen
    pub fn is_hydrogen(&self) -> bool {
        self.element == Element::H
    }

    /// Check if atom is a heavy atom (non-hydrogen)
    pub fn is_heavy_atom(&self) -> bool {
        self.element != Element::H
    }

    /// Get van der Waals radius
    pub fn vdw_radius(&self) -> f64 {
        self.element.vdw_radius()
    }

    /// Get covalent radius  
    pub fn covalent_radius(&self) -> f64 {
        self.element.covalent_radius()
    }

    /// Get atomic mass
    pub fn atomic_mass(&self) -> f64 {
        self.element.atomic_mass()
    }

    /// Check if coordinates are reasonable
    pub fn has_valid_coordinates(&self) -> bool {
        self.coords.is_finite() && 
        self.coords.x.abs() < 1000.0 &&
        self.coords.y.abs() < 1000.0 &&
        self.coords.z.abs() < 1000.0
    }

    /// Check if B-factor is reasonable
    pub fn has_valid_b_factor(&self) -> bool {
        self.b_factor >= 0.0 && self.b_factor <= 200.0
    }

    /// Check if occupancy is valid
    pub fn has_valid_occupancy(&self) -> bool {
        self.occupancy >= 0.0 && self.occupancy <= 1.0
    }

    /// Validate all atom properties
    pub fn is_valid(&self) -> bool {
        self.has_valid_coordinates() && 
        self.has_valid_b_factor() && 
        self.has_valid_occupancy() &&
        !self.name.is_empty() &&
        self.element != Element::Unknown
    }

    /// Format atom as PDB ATOM record line
    pub fn to_pdb_line(&self) -> String {
        format!(
            "ATOM  {:5} {:4} {:1}{:3} {:1}{:4}{:1}   {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}          {:2}{:2}",
            self.serial,
            self.name,
            self.alt_loc.unwrap_or(' '),
            "", // Residue name (handled by residue)
            self.chain_id,
            self.residue_seq,
            self.insertion_code.unwrap_or(' '),
            self.coords.x,
            self.coords.y,
            self.coords.z,
            self.occupancy,
            self.b_factor,
            self.element.symbol(),
            format_charge(self.charge),
        )
    }
}

impl Element {
    /// Determine element from PDB atom name
    fn from_atom_name(name: &str) -> Element {
        let trimmed = name.trim();
        
        // Handle common PDB atom naming conventions
        if trimmed.starts_with('H') {
            Element::H
        } else if trimmed.starts_with('C') {
            Element::C
        } else if trimmed.starts_with('N') {
            Element::N
        } else if trimmed.starts_with('O') {
            Element::O
        } else if trimmed.starts_with('S') {
            Element::S
        } else if trimmed.starts_with('P') {
            Element::P
        } else if trimmed.contains("ZN") {
            Element::Zn
        } else if trimmed.contains("FE") {
            Element::Fe
        } else if trimmed.contains("MG") {
            Element::Mg
        } else if trimmed.contains("CA") && trimmed.len() > 2 {
            Element::Ca
        } else {
            // Try to parse as element symbol
            let first_char = trimmed.chars().next().unwrap_or(' ').to_string();
            first_char.parse().unwrap_or(Element::Unknown)
        }
    }
}

/// Parse integer from PDB field
fn parse_int(field: &str) -> Result<i32, PdbParseError> {
    field.trim().parse().map_err(|_| PdbParseError::InvalidInteger)
}

/// Parse float from PDB field
fn parse_float(field: &str) -> Result<f64, PdbParseError> {
    field.trim().parse().map_err(|_| PdbParseError::InvalidFloat)
}

/// Parse optional character from PDB field
fn parse_char(field: &str) -> Option<char> {
    field.trim().chars().next().filter(|&c| c != ' ')
}

/// Parse charge from PDB field (e.g., "2+", "1-")
fn parse_charge(field: &str) -> Option<i8> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return None;
    }
    
    if let Some(last_char) = trimmed.chars().last() {
        let sign = match last_char {
            '+' => 1,
            '-' => -1,
            _ => return None,
        };
        
        let magnitude_str = &trimmed[..trimmed.len()-1];
        if let Ok(magnitude) = magnitude_str.parse::<i8>() {
            Some(magnitude * sign)
        } else {
            None
        }
    } else {
        None
    }
}

/// Format charge for PDB output
fn format_charge(charge: Option<i8>) -> String {
    match charge {
        Some(c) if c > 0 => format!("{:}+", c.abs()),
        Some(c) if c < 0 => format!("{:}-", c.abs()),
        _ => "  ".to_string(),
    }
}

/// Errors that can occur during PDB parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdbParseError {
    LineTooShort,
    InvalidInteger,
    InvalidFloat,
    InvalidCoordinates,
    UnknownElement,
}

impl fmt::Display for PdbParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdbParseError::LineTooShort => write!(f, "PDB line too short"),
            PdbParseError::InvalidInteger => write!(f, "Invalid integer in PDB line"),
            PdbParseError::InvalidFloat => write!(f, "Invalid float in PDB line"),
            PdbParseError::InvalidCoordinates => write!(f, "Invalid coordinates"),
            PdbParseError::UnknownElement => write!(f, "Unknown chemical element"),
        }
    }
}

impl std::error::Error for PdbParseError {}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} {} {} ({:.3}, {:.3}, {:.3})",
            self.chain_id,
            self.residue_seq,
            self.name,
            self.element.symbol(),
            self.coords.x,
            self.coords.y,
            self.coords.z
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let coords = Vector3::new(1.0, 2.0, 3.0);
        let atom = Atom::new(1, "CA".to_string(), coords, Element::C, 'A', 1);
        
        assert_eq!(atom.id, 1);
        assert_eq!(atom.name, "CA");
        assert_eq!(atom.coords, coords);
        assert_eq!(atom.element, Element::C);
        assert_eq!(atom.chain_id, 'A');
        assert_eq!(atom.residue_seq, 1);
    }

    #[test]
    fn test_atom_distance() {
        let atom1 = Atom::new(1, "CA".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::C, 'A', 1);
        let atom2 = Atom::new(2, "CB".to_string(), Vector3::new(3.0, 4.0, 0.0), Element::C, 'A', 1);
        
        assert!((atom1.distance_to(&atom2) - 5.0).abs() < 1e-10);
        assert!((atom1.distance_squared_to(&atom2) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_atom_properties() {
        let atom = Atom::new(1, "CA".to_string(), Vector3::new(1.0, 2.0, 3.0), Element::C, 'A', 1);
        
        assert!(atom.is_backbone());
        assert!(!atom.is_side_chain());
        assert!(!atom.is_hydrogen());
        assert!(atom.is_heavy_atom());
        assert!(atom.has_valid_coordinates());
        assert!(atom.is_valid());
    }

    #[test]
    fn test_pdb_parsing() {
        let pdb_line = "ATOM      1  N   ALA A   1      20.154  16.967  -8.901  1.00 20.00           N  ";
        let atom = Atom::from_pdb_line(pdb_line, 1).unwrap();
        
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "N");
        assert_eq!(atom.chain_id, 'A');
        assert_eq!(atom.residue_seq, 1);
        assert_eq!(atom.element, Element::N);
        assert!((atom.coords.x - 20.154).abs() < 1e-6);
        assert!((atom.coords.y - 16.967).abs() < 1e-6);
        assert!((atom.coords.z + 8.901).abs() < 1e-6);
    }

    #[test]
    fn test_charge_parsing() {
        assert_eq!(parse_charge("2+"), Some(2));
        assert_eq!(parse_charge("1-"), Some(-1));
        assert_eq!(parse_charge("  "), None);
        assert_eq!(parse_charge(""), None);
    }

    #[test]
    fn test_element_from_atom_name() {
        assert_eq!(Element::from_atom_name("CA"), Element::C);
        assert_eq!(Element::from_atom_name("N"), Element::N);
        assert_eq!(Element::from_atom_name("O"), Element::O);
        assert_eq!(Element::from_atom_name("1HB"), Element::H);
        assert_eq!(Element::from_atom_name("ZN"), Element::Zn);
    }

    #[test]
    fn test_contact_detection() {
        let atom1 = Atom::new(1, "CA".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::C, 'A', 1);
        let atom2 = Atom::new(2, "CB".to_string(), Vector3::new(1.5, 0.0, 0.0), Element::C, 'A', 1);
        
        // Carbon vdW radius is 1.7 Å, so 2 carbons at 1.5 Å are in contact
        assert!(atom1.is_in_contact(&atom2, 0.1));
        
        let atom3 = Atom::new(3, "CD".to_string(), Vector3::new(10.0, 0.0, 0.0), Element::C, 'A', 1);
        assert!(!atom1.is_in_contact(&atom3, 0.1));
    }
}