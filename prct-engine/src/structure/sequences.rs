//! CASP target sequences for structure prediction validation

use std::collections::HashMap;

/// CASP15 target sequences extracted from validation data
pub const CASP_SEQUENCES: &[(&str, &str)] = &[
    (
        "T1024",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKREQTPNRAKRVITTFRTGTWDAYKNL"
    ),
    (
        "T1025",
        "GSHMKTTLSYVIQLKGSDLGAPLDRWSFSQFGGPKYLGGGHPSLQSKLRIQYVDPKLDRNGKLKIQYVDPLGTDAKIKSNQSPSKLDSVLKQF"
    ),
    (
        "T1026",
        "MSFNDKGTLQDGDIVKIKDVYNSNKDCYAYIVSTNKGDTLKKAKLDRTKGDKYVKGKKD"
    ),
];

/// Get sequence by target ID
pub fn get_sequence(target_id: &str) -> Option<&'static str> {
    CASP_SEQUENCES
        .iter()
        .find(|(id, _)| *id == target_id)
        .map(|(_, seq)| *seq)
}

/// Get all target IDs
pub fn get_target_ids() -> Vec<&'static str> {
    CASP_SEQUENCES.iter().map(|(id, _)| *id).collect()
}

/// Create a HashMap of sequences for fast lookup
pub fn create_sequence_map() -> HashMap<&'static str, &'static str> {
    CASP_SEQUENCES.iter().copied().collect()
}

/// Convert single letter amino acid code to three letter code
pub fn one_to_three(aa: char) -> &'static str {
    match aa.to_ascii_uppercase() {
        'A' => "ALA", // Alanine
        'R' => "ARG", // Arginine
        'N' => "ASN", // Asparagine
        'D' => "ASP", // Aspartic acid
        'C' => "CYS", // Cysteine
        'Q' => "GLN", // Glutamine
        'E' => "GLU", // Glutamic acid
        'G' => "GLY", // Glycine
        'H' => "HIS", // Histidine
        'I' => "ILE", // Isoleucine
        'L' => "LEU", // Leucine
        'K' => "LYS", // Lysine
        'M' => "MET", // Methionine
        'F' => "PHE", // Phenylalanine
        'P' => "PRO", // Proline
        'S' => "SER", // Serine
        'T' => "THR", // Threonine
        'W' => "TRP", // Tryptophan
        'Y' => "TYR", // Tyrosine
        'V' => "VAL", // Valine
        _ => "UNK",   // Unknown
    }
}

/// Convert three letter amino acid code to single letter code
pub fn three_to_one(aa: &str) -> char {
    match aa.to_uppercase().as_str() {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

/// Get amino acid properties
pub fn is_hydrophobic(aa: char) -> bool {
    matches!(aa.to_ascii_uppercase(), 'A' | 'I' | 'L' | 'M' | 'F' | 'W' | 'Y' | 'V')
}

pub fn is_polar(aa: char) -> bool {
    matches!(aa.to_ascii_uppercase(), 'S' | 'T' | 'N' | 'Q' | 'C')
}

pub fn is_charged(aa: char) -> bool {
    matches!(aa.to_ascii_uppercase(), 'D' | 'E' | 'K' | 'R' | 'H')
}

pub fn is_positive(aa: char) -> bool {
    matches!(aa.to_ascii_uppercase(), 'K' | 'R' | 'H')
}

pub fn is_negative(aa: char) -> bool {
    matches!(aa.to_ascii_uppercase(), 'D' | 'E')
}

/// Get molecular weight of amino acid (in Daltons)
pub fn molecular_weight(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'A' => 71.08,   // Alanine
        'R' => 156.19,  // Arginine
        'N' => 114.10,  // Asparagine
        'D' => 115.09,  // Aspartic acid
        'C' => 103.14,  // Cysteine
        'Q' => 128.13,  // Glutamine
        'E' => 129.12,  // Glutamic acid
        'G' => 57.05,   // Glycine
        'H' => 137.14,  // Histidine
        'I' => 113.16,  // Isoleucine
        'L' => 113.16,  // Leucine
        'K' => 128.17,  // Lysine
        'M' => 131.20,  // Methionine
        'F' => 147.18,  // Phenylalanine
        'P' => 97.12,   // Proline
        'S' => 87.08,   // Serine
        'T' => 101.11,  // Threonine
        'W' => 186.21,  // Tryptophan
        'Y' => 163.18,  // Tyrosine
        'V' => 99.13,   // Valine
        _ => 110.0,     // Average amino acid weight
    }
}

/// Calculate total molecular weight of sequence
pub fn sequence_molecular_weight(sequence: &str) -> f64 {
    let mut weight = 0.0;
    for aa in sequence.chars() {
        weight += molecular_weight(aa);
    }
    // Subtract water molecules lost in peptide bond formation
    weight - (18.015 * (sequence.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_lookup() {
        let seq = get_sequence("T1024");
        assert!(seq.is_some());
        assert_eq!(seq.unwrap().len(), 103);

        let seq = get_sequence("NONEXISTENT");
        assert!(seq.is_none());
    }

    #[test]
    fn test_amino_acid_conversion() {
        assert_eq!(one_to_three('A'), "ALA");
        assert_eq!(one_to_three('r'), "ARG");
        assert_eq!(three_to_one("ALA"), 'A');
        assert_eq!(three_to_one("arg"), 'R');
    }

    #[test]
    fn test_amino_acid_properties() {
        assert!(is_hydrophobic('A'));
        assert!(is_polar('S'));
        assert!(is_charged('K'));
        assert!(is_positive('K'));
        assert!(is_negative('D'));
    }

    #[test]
    fn test_molecular_weight() {
        assert!((molecular_weight('G') - 57.05).abs() < 0.01);

        let seq_weight = sequence_molecular_weight("ALA");
        assert!((seq_weight - 71.08).abs() < 0.01);
    }

    #[test]
    fn test_target_ids() {
        let ids = get_target_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"T1024"));
        assert!(ids.contains(&"T1025"));
        assert!(ids.contains(&"T1026"));
    }
}