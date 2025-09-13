// Chemical elements for atomic data
use std::fmt;
use std::str::FromStr;

/// Chemical elements found in biological macromolecules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,  // Hydrogen
    C,  // Carbon
    N,  // Nitrogen
    O,  // Oxygen
    S,  // Sulfur
    P,  // Phosphorus
    F,  // Fluorine
    Cl, // Chlorine
    Br, // Bromine
    I,  // Iodine
    Na, // Sodium
    K,  // Potassium
    Mg, // Magnesium
    Ca, // Calcium
    Zn, // Zinc
    Fe, // Iron
    Cu, // Copper
    Mn, // Manganese
    Mo, // Molybdenum
    Co, // Cobalt
    Ni, // Nickel
    Se, // Selenium
    Unknown, // For unrecognized elements
}

impl Element {
    /// Get atomic number
    pub fn atomic_number(&self) -> u8 {
        match self {
            Element::H => 1,
            Element::C => 6,
            Element::N => 7,
            Element::O => 8,
            Element::F => 9,
            Element::Na => 11,
            Element::Mg => 12,
            Element::P => 15,
            Element::S => 16,
            Element::Cl => 17,
            Element::K => 19,
            Element::Ca => 20,
            Element::Mn => 25,
            Element::Fe => 26,
            Element::Co => 27,
            Element::Ni => 28,
            Element::Cu => 29,
            Element::Zn => 30,
            Element::Se => 34,
            Element::Br => 35,
            Element::Mo => 42,
            Element::I => 53,
            Element::Unknown => 0,
        }
    }

    /// Get atomic mass (in atomic mass units)
    pub fn atomic_mass(&self) -> f64 {
        match self {
            Element::H => 1.008,
            Element::C => 12.011,
            Element::N => 14.007,
            Element::O => 15.999,
            Element::F => 18.998,
            Element::Na => 22.990,
            Element::Mg => 24.305,
            Element::P => 30.974,
            Element::S => 32.065,
            Element::Cl => 35.453,
            Element::K => 39.098,
            Element::Ca => 40.078,
            Element::Mn => 54.938,
            Element::Fe => 55.845,
            Element::Co => 58.933,
            Element::Ni => 58.693,
            Element::Cu => 63.546,
            Element::Zn => 65.38,
            Element::Se => 78.96,
            Element::Br => 79.904,
            Element::Mo => 95.96,
            Element::I => 126.904,
            Element::Unknown => 0.0,
        }
    }

    /// Get van der Waals radius (in Angstroms)
    pub fn vdw_radius(&self) -> f64 {
        match self {
            Element::H => 1.20,
            Element::C => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::F => 1.47,
            Element::Na => 2.27,
            Element::Mg => 1.73,
            Element::P => 1.80,
            Element::S => 1.80,
            Element::Cl => 1.75,
            Element::K => 2.75,
            Element::Ca => 2.31,
            Element::Mn => 2.05,
            Element::Fe => 2.05,
            Element::Co => 2.00,
            Element::Ni => 1.63,
            Element::Cu => 1.40,
            Element::Zn => 1.39,
            Element::Se => 1.90,
            Element::Br => 1.85,
            Element::Mo => 2.10,
            Element::I => 1.98,
            Element::Unknown => 2.00, // Default radius
        }
    }

    /// Get covalent radius (in Angstroms)
    pub fn covalent_radius(&self) -> f64 {
        match self {
            Element::H => 0.31,
            Element::C => 0.76,
            Element::N => 0.71,
            Element::O => 0.66,
            Element::F => 0.57,
            Element::Na => 1.66,
            Element::Mg => 1.41,
            Element::P => 1.07,
            Element::S => 1.05,
            Element::Cl => 0.99,
            Element::K => 2.03,
            Element::Ca => 1.76,
            Element::Mn => 1.39,
            Element::Fe => 1.32,
            Element::Co => 1.26,
            Element::Ni => 1.24,
            Element::Cu => 1.32,
            Element::Zn => 1.22,
            Element::Se => 1.20,
            Element::Br => 1.20,
            Element::Mo => 1.54,
            Element::I => 1.39,
            Element::Unknown => 1.00,
        }
    }

    /// Check if element is a metal
    pub fn is_metal(&self) -> bool {
        matches!(self, 
            Element::Na | Element::K | Element::Mg | Element::Ca |
            Element::Zn | Element::Fe | Element::Cu | Element::Mn |
            Element::Mo | Element::Co | Element::Ni
        )
    }

    /// Check if element is a halogen
    pub fn is_halogen(&self) -> bool {
        matches!(self, Element::F | Element::Cl | Element::Br | Element::I)
    }

    /// Check if element is typically found in biological systems
    pub fn is_biological(&self) -> bool {
        matches!(self,
            Element::H | Element::C | Element::N | Element::O |
            Element::S | Element::P | Element::Na | Element::K |
            Element::Mg | Element::Ca | Element::Zn | Element::Fe |
            Element::Cu | Element::Mn | Element::Se
        )
    }

    /// Check if element is hydrogen
    pub fn is_hydrogen(&self) -> bool {
        matches!(self, Element::H)
    }

    /// Get all biological elements
    pub fn biological_elements() -> Vec<Element> {
        vec![
            Element::H, Element::C, Element::N, Element::O,
            Element::S, Element::P, Element::Na, Element::K,
            Element::Mg, Element::Ca, Element::Zn, Element::Fe,
            Element::Cu, Element::Mn, Element::Se,
        ]
    }

    /// Get element symbol as string
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::F => "F",
            Element::Cl => "Cl",
            Element::Br => "Br",
            Element::I => "I",
            Element::Na => "Na",
            Element::K => "K",
            Element::Mg => "Mg",
            Element::Ca => "Ca",
            Element::Zn => "Zn",
            Element::Fe => "Fe",
            Element::Cu => "Cu",
            Element::Mn => "Mn",
            Element::Mo => "Mo",
            Element::Co => "Co",
            Element::Ni => "Ni",
            Element::Se => "Se",
            Element::Unknown => "X",
        }
    }

    /// Get element name
    pub fn name(&self) -> &'static str {
        match self {
            Element::H => "Hydrogen",
            Element::C => "Carbon",
            Element::N => "Nitrogen",
            Element::O => "Oxygen",
            Element::S => "Sulfur",
            Element::P => "Phosphorus",
            Element::F => "Fluorine",
            Element::Cl => "Chlorine",
            Element::Br => "Bromine",
            Element::I => "Iodine",
            Element::Na => "Sodium",
            Element::K => "Potassium",
            Element::Mg => "Magnesium",
            Element::Ca => "Calcium",
            Element::Zn => "Zinc",
            Element::Fe => "Iron",
            Element::Cu => "Copper",
            Element::Mn => "Manganese",
            Element::Mo => "Molybdenum",
            Element::Co => "Cobalt",
            Element::Ni => "Nickel",
            Element::Se => "Selenium",
            Element::Unknown => "Unknown",
        }
    }
}

impl FromStr for Element {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "H" => Ok(Element::H),
            "C" => Ok(Element::C),
            "N" => Ok(Element::N),
            "O" => Ok(Element::O),
            "S" => Ok(Element::S),
            "P" => Ok(Element::P),
            "F" => Ok(Element::F),
            "CL" => Ok(Element::Cl),
            "BR" => Ok(Element::Br),
            "I" => Ok(Element::I),
            "NA" => Ok(Element::Na),
            "K" => Ok(Element::K),
            "MG" => Ok(Element::Mg),
            "CA" => Ok(Element::Ca),
            "ZN" => Ok(Element::Zn),
            "FE" => Ok(Element::Fe),
            "CU" => Ok(Element::Cu),
            "MN" => Ok(Element::Mn),
            "MO" => Ok(Element::Mo),
            "CO" => Ok(Element::Co),
            "NI" => Ok(Element::Ni),
            "SE" => Ok(Element::Se),
            _ => Ok(Element::Unknown),
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Parse element from PDB atom name (handles cases like " CA ", "CA  ", etc.)
pub fn element_from_pdb_name(atom_name: &str) -> Element {
    // PDB atom names can be left or right justified
    let name = atom_name.trim();
    
    // Handle common cases first
    if name.len() >= 2 {
        // Check if first character is element symbol
        let first_char = &name[0..1];
        if let Ok(element) = Element::from_str(first_char) {
            if element != Element::Unknown {
                return element;
            }
        }
        
        // Check if first two characters are element symbol
        let first_two = &name[0..2];
        if let Ok(element) = Element::from_str(first_two) {
            if element != Element::Unknown {
                return element;
            }
        }
    }
    
    // Try the whole string
    Element::from_str(name).unwrap_or(Element::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_properties() {
        assert_eq!(Element::C.atomic_number(), 6);
        assert_eq!(Element::N.atomic_number(), 7);
        assert_eq!(Element::O.atomic_number(), 8);
        
        assert!((Element::C.atomic_mass() - 12.011).abs() < 1e-3);
        assert!((Element::C.vdw_radius() - 1.70).abs() < 1e-2);
        assert!((Element::C.covalent_radius() - 0.76).abs() < 1e-2);
    }

    #[test]
    fn test_element_classification() {
        assert!(Element::Fe.is_metal());
        assert!(!Element::C.is_metal());
        
        assert!(Element::Cl.is_halogen());
        assert!(!Element::C.is_halogen());
        
        assert!(Element::C.is_biological());
        assert!(Element::N.is_biological());
        assert!(Element::O.is_biological());
    }

    #[test]
    fn test_element_parsing() {
        assert_eq!("C".parse::<Element>().unwrap(), Element::C);
        assert_eq!("CA".parse::<Element>().unwrap(), Element::Ca);
        assert_eq!("cl".parse::<Element>().unwrap(), Element::Cl);
        assert_eq!("unknown".parse::<Element>().unwrap(), Element::Unknown);
    }

    #[test]
    fn test_pdb_name_parsing() {
        assert_eq!(element_from_pdb_name(" C  "), Element::C);
        assert_eq!(element_from_pdb_name("CA  "), Element::C);
        assert_eq!(element_from_pdb_name(" N  "), Element::N);
        assert_eq!(element_from_pdb_name("ZN  "), Element::Zn);
        assert_eq!(element_from_pdb_name("FE2+"), Element::Fe);
    }

    #[test]
    fn test_element_display() {
        assert_eq!(Element::C.to_string(), "C");
        assert_eq!(Element::Ca.to_string(), "Ca");
        assert_eq!(Element::Unknown.to_string(), "X");
    }
}