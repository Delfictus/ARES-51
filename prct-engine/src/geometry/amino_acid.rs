// Amino acid types and properties
use std::fmt;
use std::str::FromStr;

/// Standard amino acids found in proteins
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AminoAcid {
    // Standard 20 amino acids
    Ala, // Alanine
    Arg, // Arginine  
    Asn, // Asparagine
    Asp, // Aspartic acid
    Cys, // Cysteine
    Gln, // Glutamine
    Glu, // Glutamic acid
    Gly, // Glycine
    His, // Histidine
    Ile, // Isoleucine
    Leu, // Leucine
    Lys, // Lysine
    Met, // Methionine
    Phe, // Phenylalanine
    Pro, // Proline
    Ser, // Serine
    Thr, // Threonine
    Trp, // Tryptophan
    Tyr, // Tyrosine
    Val, // Valine
    
    // Non-standard amino acids
    Sec, // Selenocysteine (21st amino acid)
    Pyl, // Pyrrolysine (22nd amino acid)
    
    // Modified/unknown residues
    Asx, // Asparagine or Aspartic acid (ambiguous)
    Glx, // Glutamine or Glutamic acid (ambiguous)
    Unk, // Unknown amino acid
}

impl AminoAcid {
    /// Get three-letter code
    pub fn three_letter(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "ALA",
            AminoAcid::Arg => "ARG",
            AminoAcid::Asn => "ASN",
            AminoAcid::Asp => "ASP",
            AminoAcid::Cys => "CYS",
            AminoAcid::Gln => "GLN",
            AminoAcid::Glu => "GLU",
            AminoAcid::Gly => "GLY",
            AminoAcid::His => "HIS",
            AminoAcid::Ile => "ILE",
            AminoAcid::Leu => "LEU",
            AminoAcid::Lys => "LYS",
            AminoAcid::Met => "MET",
            AminoAcid::Phe => "PHE",
            AminoAcid::Pro => "PRO",
            AminoAcid::Ser => "SER",
            AminoAcid::Thr => "THR",
            AminoAcid::Trp => "TRP",
            AminoAcid::Tyr => "TYR",
            AminoAcid::Val => "VAL",
            AminoAcid::Sec => "SEC",
            AminoAcid::Pyl => "PYL",
            AminoAcid::Asx => "ASX",
            AminoAcid::Glx => "GLX",
            AminoAcid::Unk => "UNK",
        }
    }

    /// Get one-letter code
    pub fn one_letter(&self) -> char {
        match self {
            AminoAcid::Ala => 'A',
            AminoAcid::Arg => 'R',
            AminoAcid::Asn => 'N',
            AminoAcid::Asp => 'D',
            AminoAcid::Cys => 'C',
            AminoAcid::Gln => 'Q',
            AminoAcid::Glu => 'E',
            AminoAcid::Gly => 'G',
            AminoAcid::His => 'H',
            AminoAcid::Ile => 'I',
            AminoAcid::Leu => 'L',
            AminoAcid::Lys => 'K',
            AminoAcid::Met => 'M',
            AminoAcid::Phe => 'F',
            AminoAcid::Pro => 'P',
            AminoAcid::Ser => 'S',
            AminoAcid::Thr => 'T',
            AminoAcid::Trp => 'W',
            AminoAcid::Tyr => 'Y',
            AminoAcid::Val => 'V',
            AminoAcid::Sec => 'U',
            AminoAcid::Pyl => 'O',
            AminoAcid::Asx => 'B',
            AminoAcid::Glx => 'Z',
            AminoAcid::Unk => 'X',
        }
    }

    /// Get full name
    pub fn full_name(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "Alanine",
            AminoAcid::Arg => "Arginine",
            AminoAcid::Asn => "Asparagine",
            AminoAcid::Asp => "Aspartic acid",
            AminoAcid::Cys => "Cysteine",
            AminoAcid::Gln => "Glutamine",
            AminoAcid::Glu => "Glutamic acid",
            AminoAcid::Gly => "Glycine",
            AminoAcid::His => "Histidine",
            AminoAcid::Ile => "Isoleucine",
            AminoAcid::Leu => "Leucine",
            AminoAcid::Lys => "Lysine",
            AminoAcid::Met => "Methionine",
            AminoAcid::Phe => "Phenylalanine",
            AminoAcid::Pro => "Proline",
            AminoAcid::Ser => "Serine",
            AminoAcid::Thr => "Threonine",
            AminoAcid::Trp => "Tryptophan",
            AminoAcid::Tyr => "Tyrosine",
            AminoAcid::Val => "Valine",
            AminoAcid::Sec => "Selenocysteine",
            AminoAcid::Pyl => "Pyrrolysine",
            AminoAcid::Asx => "Asparagine or Aspartic acid",
            AminoAcid::Glx => "Glutamine or Glutamic acid",
            AminoAcid::Unk => "Unknown amino acid",
        }
    }

    /// Get molecular weight (in g/mol)
    pub fn molecular_weight(&self) -> f64 {
        match self {
            AminoAcid::Ala => 89.094,
            AminoAcid::Arg => 174.203,
            AminoAcid::Asn => 132.119,
            AminoAcid::Asp => 133.104,
            AminoAcid::Cys => 121.154,
            AminoAcid::Gln => 146.146,
            AminoAcid::Glu => 147.131,
            AminoAcid::Gly => 75.067,
            AminoAcid::His => 155.156,
            AminoAcid::Ile => 131.175,
            AminoAcid::Leu => 131.175,
            AminoAcid::Lys => 146.189,
            AminoAcid::Met => 149.208,
            AminoAcid::Phe => 165.192,
            AminoAcid::Pro => 115.132,
            AminoAcid::Ser => 105.093,
            AminoAcid::Thr => 119.120,
            AminoAcid::Trp => 204.228,
            AminoAcid::Tyr => 181.191,
            AminoAcid::Val => 117.148,
            AminoAcid::Sec => 168.053,
            AminoAcid::Pyl => 255.313,
            AminoAcid::Asx => 132.61, // Average of Asn and Asp
            AminoAcid::Glx => 146.64, // Average of Gln and Glu
            AminoAcid::Unk => 110.0,  // Rough average
        }
    }

    /// Get side chain polarity
    pub fn polarity(&self) -> Polarity {
        match self {
            AminoAcid::Ala | AminoAcid::Val | AminoAcid::Ile | AminoAcid::Leu |
            AminoAcid::Met | AminoAcid::Phe | AminoAcid::Trp | AminoAcid::Pro => Polarity::Nonpolar,
            
            AminoAcid::Ser | AminoAcid::Thr | AminoAcid::Asn | AminoAcid::Gln |
            AminoAcid::Tyr | AminoAcid::Cys => Polarity::Polar,
            
            AminoAcid::Asp | AminoAcid::Glu => Polarity::NegativelyCharged,
            
            AminoAcid::Lys | AminoAcid::Arg => Polarity::PositivelyCharged,
            
            AminoAcid::His => Polarity::PositivelyCharged, // At physiological pH
            
            AminoAcid::Gly => Polarity::Special, // No side chain
            
            AminoAcid::Sec => Polarity::Polar,
            AminoAcid::Pyl => Polarity::PositivelyCharged,
            AminoAcid::Asx => Polarity::Polar, // Could be charged
            AminoAcid::Glx => Polarity::Polar, // Could be charged
            AminoAcid::Unk => Polarity::Unknown,
        }
    }

    /// Get hydrophobicity index (Kyte-Doolittle scale)
    pub fn hydrophobicity(&self) -> f64 {
        match self {
            AminoAcid::Ala => 1.8,
            AminoAcid::Arg => -4.5,
            AminoAcid::Asn => -3.5,
            AminoAcid::Asp => -3.5,
            AminoAcid::Cys => 2.5,
            AminoAcid::Gln => -3.5,
            AminoAcid::Glu => -3.5,
            AminoAcid::Gly => -0.4,
            AminoAcid::His => -3.2,
            AminoAcid::Ile => 4.5,
            AminoAcid::Leu => 3.8,
            AminoAcid::Lys => -3.9,
            AminoAcid::Met => 1.9,
            AminoAcid::Phe => 2.8,
            AminoAcid::Pro => -1.6,
            AminoAcid::Ser => -0.8,
            AminoAcid::Thr => -0.7,
            AminoAcid::Trp => -0.9,
            AminoAcid::Tyr => -1.3,
            AminoAcid::Val => 4.2,
            AminoAcid::Sec => 2.5,  // Similar to Cys
            AminoAcid::Pyl => -3.0, // Estimated
            AminoAcid::Asx => -3.5, // Similar to Asn
            AminoAcid::Glx => -3.5, // Similar to Gln
            AminoAcid::Unk => 0.0,  // Neutral
        }
    }

    /// Check if amino acid is standard (one of the 20)
    pub fn is_standard(&self) -> bool {
        matches!(self,
            AminoAcid::Ala | AminoAcid::Arg | AminoAcid::Asn | AminoAcid::Asp |
            AminoAcid::Cys | AminoAcid::Gln | AminoAcid::Glu | AminoAcid::Gly |
            AminoAcid::His | AminoAcid::Ile | AminoAcid::Leu | AminoAcid::Lys |
            AminoAcid::Met | AminoAcid::Phe | AminoAcid::Pro | AminoAcid::Ser |
            AminoAcid::Thr | AminoAcid::Trp | AminoAcid::Tyr | AminoAcid::Val
        )
    }

    /// Check if amino acid can form disulfide bonds
    pub fn can_form_disulfide(&self) -> bool {
        matches!(self, AminoAcid::Cys | AminoAcid::Sec)
    }

    /// Check if amino acid is aromatic
    pub fn is_aromatic(&self) -> bool {
        matches!(self, AminoAcid::Phe | AminoAcid::Trp | AminoAcid::Tyr | AminoAcid::His)
    }

    /// Get secondary structure propensity (alpha-helix, beta-sheet preferences)
    pub fn secondary_structure_propensity(&self) -> (f64, f64) {
        // (helix propensity, sheet propensity) - Chou-Fasman parameters
        match self {
            AminoAcid::Ala => (1.42, 0.83),
            AminoAcid::Arg => (0.98, 0.93),
            AminoAcid::Asn => (0.67, 0.89),
            AminoAcid::Asp => (1.01, 0.54),
            AminoAcid::Cys => (0.70, 1.19),
            AminoAcid::Gln => (1.11, 1.10),
            AminoAcid::Glu => (1.51, 0.37),
            AminoAcid::Gly => (0.57, 0.75),
            AminoAcid::His => (1.00, 0.87),
            AminoAcid::Ile => (1.08, 1.60),
            AminoAcid::Leu => (1.21, 1.30),
            AminoAcid::Lys => (1.16, 0.74),
            AminoAcid::Met => (1.45, 1.05),
            AminoAcid::Phe => (1.13, 1.38),
            AminoAcid::Pro => (0.57, 0.55),
            AminoAcid::Ser => (0.77, 0.75),
            AminoAcid::Thr => (0.83, 1.19),
            AminoAcid::Trp => (1.08, 1.37),
            AminoAcid::Tyr => (0.69, 1.47),
            AminoAcid::Val => (1.06, 1.70),
            _ => (1.00, 1.00), // Default for non-standard
        }
    }

    /// Get all standard amino acids
    pub fn standard_amino_acids() -> Vec<AminoAcid> {
        vec![
            AminoAcid::Ala, AminoAcid::Arg, AminoAcid::Asn, AminoAcid::Asp,
            AminoAcid::Cys, AminoAcid::Gln, AminoAcid::Glu, AminoAcid::Gly,
            AminoAcid::His, AminoAcid::Ile, AminoAcid::Leu, AminoAcid::Lys,
            AminoAcid::Met, AminoAcid::Phe, AminoAcid::Pro, AminoAcid::Ser,
            AminoAcid::Thr, AminoAcid::Trp, AminoAcid::Tyr, AminoAcid::Val,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Nonpolar,
    Polar,
    PositivelyCharged,
    NegativelyCharged,
    Special,
    Unknown,
}

impl FromStr for AminoAcid {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "ALA" | "A" => Ok(AminoAcid::Ala),
            "ARG" | "R" => Ok(AminoAcid::Arg),
            "ASN" | "N" => Ok(AminoAcid::Asn),
            "ASP" | "D" => Ok(AminoAcid::Asp),
            "CYS" | "C" => Ok(AminoAcid::Cys),
            "GLN" | "Q" => Ok(AminoAcid::Gln),
            "GLU" | "E" => Ok(AminoAcid::Glu),
            "GLY" | "G" => Ok(AminoAcid::Gly),
            "HIS" | "H" => Ok(AminoAcid::His),
            "ILE" | "I" => Ok(AminoAcid::Ile),
            "LEU" | "L" => Ok(AminoAcid::Leu),
            "LYS" | "K" => Ok(AminoAcid::Lys),
            "MET" | "M" => Ok(AminoAcid::Met),
            "PHE" | "F" => Ok(AminoAcid::Phe),
            "PRO" | "P" => Ok(AminoAcid::Pro),
            "SER" | "S" => Ok(AminoAcid::Ser),
            "THR" | "T" => Ok(AminoAcid::Thr),
            "TRP" | "W" => Ok(AminoAcid::Trp),
            "TYR" | "Y" => Ok(AminoAcid::Tyr),
            "VAL" | "V" => Ok(AminoAcid::Val),
            "SEC" | "U" => Ok(AminoAcid::Sec),
            "PYL" | "O" => Ok(AminoAcid::Pyl),
            "ASX" | "B" => Ok(AminoAcid::Asx),
            "GLX" | "Z" => Ok(AminoAcid::Glx),
            "UNK" | "X" => Ok(AminoAcid::Unk),
            _ => Err(()),
        }
    }
}

impl fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.three_letter())
    }
}

/// Convert between one-letter and three-letter codes
pub fn one_to_three(one_letter: char) -> Option<AminoAcid> {
    match one_letter.to_ascii_uppercase() {
        'A' => Some(AminoAcid::Ala),
        'R' => Some(AminoAcid::Arg),
        'N' => Some(AminoAcid::Asn),
        'D' => Some(AminoAcid::Asp),
        'C' => Some(AminoAcid::Cys),
        'Q' => Some(AminoAcid::Gln),
        'E' => Some(AminoAcid::Glu),
        'G' => Some(AminoAcid::Gly),
        'H' => Some(AminoAcid::His),
        'I' => Some(AminoAcid::Ile),
        'L' => Some(AminoAcid::Leu),
        'K' => Some(AminoAcid::Lys),
        'M' => Some(AminoAcid::Met),
        'F' => Some(AminoAcid::Phe),
        'P' => Some(AminoAcid::Pro),
        'S' => Some(AminoAcid::Ser),
        'T' => Some(AminoAcid::Thr),
        'W' => Some(AminoAcid::Trp),
        'Y' => Some(AminoAcid::Tyr),
        'V' => Some(AminoAcid::Val),
        'U' => Some(AminoAcid::Sec),
        'O' => Some(AminoAcid::Pyl),
        'B' => Some(AminoAcid::Asx),
        'Z' => Some(AminoAcid::Glx),
        'X' => Some(AminoAcid::Unk),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amino_acid_codes() {
        assert_eq!(AminoAcid::Ala.three_letter(), "ALA");
        assert_eq!(AminoAcid::Ala.one_letter(), 'A');
        assert_eq!(AminoAcid::Trp.three_letter(), "TRP");
        assert_eq!(AminoAcid::Trp.one_letter(), 'W');
    }

    #[test]
    fn test_amino_acid_properties() {
        assert!((AminoAcid::Ala.molecular_weight() - 89.094).abs() < 1e-2);
        assert!(AminoAcid::Ala.hydrophobicity() > 0.0); // Hydrophobic
        assert!(AminoAcid::Asp.hydrophobicity() < 0.0); // Hydrophilic
        
        assert_eq!(AminoAcid::Ala.polarity(), Polarity::Nonpolar);
        assert_eq!(AminoAcid::Asp.polarity(), Polarity::NegativelyCharged);
        assert_eq!(AminoAcid::Lys.polarity(), Polarity::PositivelyCharged);
    }

    #[test]
    fn test_amino_acid_classification() {
        assert!(AminoAcid::Ala.is_standard());
        assert!(!AminoAcid::Sec.is_standard());
        
        assert!(AminoAcid::Cys.can_form_disulfide());
        assert!(!AminoAcid::Ala.can_form_disulfide());
        
        assert!(AminoAcid::Phe.is_aromatic());
        assert!(!AminoAcid::Ala.is_aromatic());
    }

    #[test]
    fn test_amino_acid_parsing() {
        assert_eq!("ALA".parse::<AminoAcid>().unwrap(), AminoAcid::Ala);
        assert_eq!("A".parse::<AminoAcid>().unwrap(), AminoAcid::Ala);
        assert_eq!("trp".parse::<AminoAcid>().unwrap(), AminoAcid::Trp);
        assert!("INVALID".parse::<AminoAcid>().is_err());
    }

    #[test]
    fn test_one_to_three_conversion() {
        assert_eq!(one_to_three('A'), Some(AminoAcid::Ala));
        assert_eq!(one_to_three('w'), Some(AminoAcid::Trp));
        assert_eq!(one_to_three('Z'), None);
    }

    #[test]
    fn test_secondary_structure_propensity() {
        let (helix, sheet) = AminoAcid::Ala.secondary_structure_propensity();
        assert!(helix > 1.0); // Ala favors helices
        
        let (helix, sheet) = AminoAcid::Val.secondary_structure_propensity();
        assert!(sheet > helix); // Val favors sheets
    }
}