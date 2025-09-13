// Protein structure data types containing multiple chains
use crate::geometry::{Chain, Residue, Atom, Vector3, ChainType};
use std::collections::HashMap;
use std::fmt;

/// Complete protein structure containing multiple chains
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    /// Structure identifier (PDB ID)
    pub id: String,
    
    /// Map of chain ID to chain
    pub chains: HashMap<char, Chain>,
    
    /// Ordered list of chain IDs
    pub chain_order: Vec<char>,
    
    /// Structure metadata
    pub metadata: StructureMetadata,
    
    /// Unit cell parameters (for crystal structures)
    pub unit_cell: Option<UnitCell>,
    
    /// Space group information
    pub space_group: Option<String>,
    
    /// Biological assembly information
    pub biological_assemblies: Vec<BiologicalAssembly>,
    
    /// HELIX records from PDB
    pub helices: Vec<HelixRecord>,
    
    /// SHEET records from PDB
    pub sheets: Vec<SheetRecord>,
}

/// Structure metadata from PDB header
#[derive(Debug, Clone, PartialEq)]
pub struct StructureMetadata {
    /// Classification (enzyme, transport protein, etc.)
    pub classification: String,
    
    /// Deposition date
    pub deposition_date: String,
    
    /// Structure title
    pub title: String,
    
    /// Experimental method (X-ray, NMR, cryo-EM, etc.)
    pub experimental_method: ExperimentalMethod,
    
    /// Resolution in Angstroms
    pub resolution: Option<f64>,
    
    /// R-factor (working set)
    pub r_work: Option<f64>,
    
    /// R-factor (free set)
    pub r_free: Option<f64>,
    
    /// Authors
    pub authors: Vec<String>,
    
    /// Keywords
    pub keywords: Vec<String>,
    
    /// Source organism
    pub organism: Option<String>,
    
    /// Expression system
    pub expression_system: Option<String>,
}

/// Unit cell parameters for crystal structures
#[derive(Debug, Clone, PartialEq)]
pub struct UnitCell {
    /// Cell dimensions (a, b, c) in Angstroms
    pub dimensions: (f64, f64, f64),
    
    /// Cell angles (α, β, γ) in degrees
    pub angles: (f64, f64, f64),
    
    /// Cell volume in Ų
    pub volume: f64,
}

/// Experimental methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentalMethod {
    XRayDiffraction,
    NMRSolution,
    NMRSolid,
    ElectronMicroscopy,
    ElectronCrystallography,
    FiberDiffraction,
    NeutronDiffraction,
    SolutionScattering,
    TheoreticalModel,
    Unknown,
}

/// Biological assembly information
#[derive(Debug, Clone, PartialEq)]
pub struct BiologicalAssembly {
    /// Assembly identifier
    pub id: String,
    
    /// Assembly method (author_defined_assembly, software_defined_assembly)
    pub method: String,
    
    /// Oligomeric details
    pub oligomeric_details: String,
    
    /// Number of biological assemblies
    pub oligomeric_count: i32,
    
    /// Transformation matrices
    pub transformations: Vec<TransformationMatrix>,
}

/// 3D transformation matrix for biological assemblies
#[derive(Debug, Clone, PartialEq)]
pub struct TransformationMatrix {
    /// Rotation matrix (3x3)
    pub rotation: [[f64; 3]; 3],
    
    /// Translation vector
    pub translation: Vector3,
    
    /// Chain IDs to apply transformation to
    pub chain_ids: Vec<char>,
}

/// HELIX record information
#[derive(Debug, Clone, PartialEq)]
pub struct HelixRecord {
    /// Helix identifier
    pub id: String,
    
    /// Helix class (1=right-handed alpha, 2=omega, 3=pi, 5=3-10)
    pub helix_class: i32,
    
    /// Initial residue
    pub init_chain: char,
    pub init_seq_num: i32,
    pub init_insertion: Option<char>,
    
    /// Terminal residue
    pub end_chain: char,
    pub end_seq_num: i32,
    pub end_insertion: Option<char>,
    
    /// Helix length
    pub length: i32,
    
    /// Comment
    pub comment: Option<String>,
}

/// SHEET record information
#[derive(Debug, Clone, PartialEq)]
pub struct SheetRecord {
    /// Sheet identifier
    pub sheet_id: String,
    
    /// Strand number within sheet
    pub strand: i32,
    
    /// Number of strands in sheet
    pub num_strands: i32,
    
    /// Initial residue
    pub init_chain: char,
    pub init_seq_num: i32,
    pub init_insertion: Option<char>,
    
    /// Terminal residue
    pub end_chain: char,
    pub end_seq_num: i32,
    pub end_insertion: Option<char>,
    
    /// Strand sense (0=first, 1=parallel, -1=antiparallel)
    pub sense: i32,
    
    /// Registration information
    pub registration: Option<SheetRegistration>,
}

/// Beta sheet registration information
#[derive(Debug, Clone, PartialEq)]
pub struct SheetRegistration {
    /// Current strand atom
    pub cur_atom: String,
    pub cur_chain: char,
    pub cur_seq_num: i32,
    pub cur_insertion: Option<char>,
    
    /// Previous strand atom
    pub prev_atom: String,
    pub prev_chain: char,
    pub prev_seq_num: i32,
    pub prev_insertion: Option<char>,
}

impl Structure {
    /// Create new empty structure
    pub fn new(id: String) -> Self {
        Structure {
            id,
            chains: HashMap::new(),
            chain_order: Vec::new(),
            metadata: StructureMetadata::default(),
            unit_cell: None,
            space_group: None,
            biological_assemblies: Vec::new(),
            helices: Vec::new(),
            sheets: Vec::new(),
        }
    }

    /// Add chain to structure
    pub fn add_chain(&mut self, chain: Chain) {
        let chain_id = chain.id;
        if !self.chains.contains_key(&chain_id) {
            self.chain_order.push(chain_id);
        }
        self.chains.insert(chain_id, chain);
    }

    /// Get chain by ID
    pub fn get_chain(&self, chain_id: char) -> Option<&Chain> {
        self.chains.get(&chain_id)
    }

    /// Get mutable chain by ID
    pub fn get_chain_mut(&mut self, chain_id: char) -> Option<&mut Chain> {
        self.chains.get_mut(&chain_id)
    }

    /// Remove chain by ID
    pub fn remove_chain(&mut self, chain_id: char) -> Option<Chain> {
        if let Some(chain) = self.chains.remove(&chain_id) {
            self.chain_order.retain(|&id| id != chain_id);
            Some(chain)
        } else {
            None
        }
    }

    /// Check if structure has chain
    pub fn has_chain(&self, chain_id: char) -> bool {
        self.chains.contains_key(&chain_id)
    }

    /// Get all chain IDs in order
    pub fn chain_ids(&self) -> &[char] {
        &self.chain_order
    }

    /// Get all chains
    pub fn chains(&self) -> impl Iterator<Item = &Chain> {
        self.chain_order.iter().filter_map(|&id| self.chains.get(&id))
    }

    /// Get chain IDs for iterating mutably
    pub fn chain_ids_vec(&self) -> Vec<char> {
        self.chain_order.clone()
    }

    /// Get protein chains only
    pub fn protein_chains(&self) -> impl Iterator<Item = &Chain> {
        self.chains().filter(|chain| chain.chain_type == ChainType::Protein)
    }

    /// Get number of chains
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Get total number of residues
    pub fn residue_count(&self) -> usize {
        self.chains().map(|chain| chain.len()).sum()
    }

    /// Get total number of atoms
    pub fn atom_count(&self) -> usize {
        self.chains().map(|chain| chain.atom_count()).sum()
    }

    /// Get total number of heavy atoms
    pub fn heavy_atom_count(&self) -> usize {
        self.chains().map(|chain| chain.heavy_atom_count()).sum()
    }

    /// Get all atoms in structure
    pub fn atoms(&self) -> Vec<&Atom> {
        self.chains().flat_map(|chain| chain.atoms()).collect()
    }

    /// Get all heavy atoms
    pub fn heavy_atoms(&self) -> Vec<&Atom> {
        self.chains().flat_map(|chain| chain.heavy_atoms()).collect()
    }

    /// Get all CA atoms in order
    pub fn ca_atoms(&self) -> Vec<&Atom> {
        self.chains().flat_map(|chain| chain.ca_atoms()).collect()
    }

    /// Calculate total molecular weight
    pub fn molecular_weight(&self) -> f64 {
        self.chains().map(|chain| chain.molecular_weight()).sum()
    }

    /// Calculate center of mass for entire structure
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

    /// Get bounding box for entire structure
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

    /// Get structure radius
    pub fn radius(&self) -> f64 {
        let center = self.geometric_center();
        self.atoms().iter()
            .map(|atom| center.distance(&atom.coords))
            .fold(0.0, f64::max)
    }

    /// Get residue by chain and sequence number
    pub fn get_residue(&self, chain_id: char, seq_num: i32) -> Option<&Residue> {
        self.get_chain(chain_id)?.get_residue(seq_num)
    }

    /// Get mutable residue by chain and sequence number
    pub fn get_residue_mut(&mut self, chain_id: char, seq_num: i32) -> Option<&mut Residue> {
        self.get_chain_mut(chain_id)?.get_residue_mut(seq_num)
    }

    /// Find atom by chain, residue, and atom name
    pub fn find_atom(&self, chain_id: char, seq_num: i32, atom_name: &str) -> Option<&Atom> {
        self.get_residue(chain_id, seq_num)?.get_atom(atom_name)
    }

    /// Get combined sequence for all protein chains
    pub fn protein_sequence(&self) -> String {
        self.protein_chains()
            .map(|chain| chain.sequence_one_letter())
            .collect::<Vec<_>>()
            .join("/") // Separate chains with "/"
    }

    /// Calculate phi/psi angles for all protein chains
    pub fn calculate_backbone_dihedrals(&mut self) {
        let chain_ids: Vec<char> = self.chain_order.clone();
        for chain_id in chain_ids {
            if let Some(chain) = self.chains.get_mut(&chain_id) {
                if chain.chain_type == ChainType::Protein {
                    chain.calculate_backbone_dihedrals();
                }
            }
        }
    }

    /// Assign secondary structure from HELIX/SHEET records
    pub fn assign_secondary_structure_from_records(&mut self) {
        // Clone the records to avoid borrowing issues
        let helices = self.helices.clone();
        let sheets = self.sheets.clone();
        
        // Assign helix secondary structure
        for helix in helices {
            if let Some(chain) = self.get_chain_mut(helix.init_chain) {
                for seq_num in helix.init_seq_num..=helix.end_seq_num {
                    if let Some(residue) = chain.get_residue_mut(seq_num) {
                        residue.secondary_structure = match helix.helix_class {
                            1 => crate::geometry::residue::SecondaryStructure::Helix,
                            3 => crate::geometry::residue::SecondaryStructure::HelixPi,
                            5 => crate::geometry::residue::SecondaryStructure::Helix310,
                            _ => crate::geometry::residue::SecondaryStructure::Helix,
                        };
                    }
                }
            }
        }

        // Assign sheet secondary structure
        for sheet in sheets {
            if let Some(chain) = self.get_chain_mut(sheet.init_chain) {
                for seq_num in sheet.init_seq_num..=sheet.end_seq_num {
                    if let Some(residue) = chain.get_residue_mut(seq_num) {
                        residue.secondary_structure = crate::geometry::residue::SecondaryStructure::Sheet;
                    }
                }
            }
        }
    }

    /// Check if structure is reasonable (basic validation)
    pub fn validate(&self) -> Vec<StructureValidationError> {
        let mut errors = Vec::new();

        if self.chains.is_empty() {
            errors.push(StructureValidationError::NoChains);
            return errors;
        }

        // Validate each chain
        for (chain_id, chain) in &self.chains {
            let chain_errors = chain.validate();
            for error in chain_errors {
                errors.push(StructureValidationError::ChainError(*chain_id, error));
            }
        }

        // Check resolution
        if let Some(resolution) = self.metadata.resolution {
            if resolution <= 0.0 || resolution > 10.0 {
                errors.push(StructureValidationError::UnreasonableResolution(resolution));
            }
        }

        // Check R-factors
        if let Some(r_work) = self.metadata.r_work {
            if r_work < 0.0 || r_work > 1.0 {
                errors.push(StructureValidationError::UnreasonableRFactor(r_work));
            }
        }

        // Enhanced validation checks
        
        // 1. Coordinate bounds checking
        self.check_coordinate_bounds(&mut errors);
        
        // 2. Missing atom detection
        self.check_missing_atoms(&mut errors);
        
        // 3. Geometric consistency validation
        self.check_geometric_consistency(&mut errors);
        
        // 4. Backbone geometry validation
        self.check_backbone_geometry(&mut errors);

        errors
    }

    /// Check coordinate bounds (reasonable limits: -999 to 999 Å)
    fn check_coordinate_bounds(&self, errors: &mut Vec<StructureValidationError>) {
        const MAX_COORD: f64 = 999.0;
        
        for chain in self.chains.values() {
            for residue in &chain.residues {
                for (_atom_name, atom) in &residue.atoms {
                    let pos = atom.coords;
                    if pos.x.abs() > MAX_COORD || pos.y.abs() > MAX_COORD || pos.z.abs() > MAX_COORD {
                        errors.push(StructureValidationError::CoordinatesOutOfBounds(pos.x, pos.y, pos.z));
                        return; // Only report first occurrence to avoid spam
                    }
                }
            }
        }
    }

    /// Check for missing atoms (protein residues should have N, CA, C backbone atoms)
    fn check_missing_atoms(&self, errors: &mut Vec<StructureValidationError>) {
        let mut missing_atoms = 0;
        let mut total_expected = 0;
        
        for chain in self.protein_chains() {
            for residue in chain.residues() {
                total_expected += 3; // N, CA, C
                
                let has_n = residue.get_atom("N").is_some();
                let has_ca = residue.get_atom("CA").is_some();
                let has_c = residue.get_atom("C").is_some();
                
                if !has_n { missing_atoms += 1; }
                if !has_ca { missing_atoms += 1; }
                if !has_c { missing_atoms += 1; }
            }
        }
        
        if total_expected > 0 {
            let missing_percentage = 100.0 * missing_atoms as f64 / total_expected as f64;
            if missing_percentage > 10.0 { // More than 10% missing is concerning
                errors.push(StructureValidationError::TooManyMissingAtoms(missing_atoms, total_expected));
            }
        }
    }

    /// Check geometric consistency (atom clashes and bond lengths)
    fn check_geometric_consistency(&self, errors: &mut Vec<StructureValidationError>) {
        let mut clashes = 0;
        let mut bad_bonds = 0;
        
        // Collect all atoms for clash detection
        let mut all_atoms = Vec::new();
        for chain in self.chains.values() {
            for residue in &chain.residues {
                for (_atom_name, atom) in &residue.atoms {
                    all_atoms.push((atom, chain.id, residue.seq_num));
                }
            }
        }
        
        // Check for severe atom clashes (< 1.0 Å between non-bonded heavy atoms)
        for i in 0..all_atoms.len() {
            for j in (i+1)..all_atoms.len() {
                let (atom1, chain1, res1) = &all_atoms[i];
                let (atom2, chain2, res2) = &all_atoms[j];
                
                // Skip bonded atoms (same residue or adjacent residues)
                if chain1 == chain2 && (res1 == res2 || (res1 - res2).abs() <= 1) {
                    continue;
                }
                
                // Skip hydrogens
                if atom1.element.is_hydrogen() || atom2.element.is_hydrogen() {
                    continue;
                }
                
                let distance = atom1.coords.distance(&atom2.coords);
                if distance < 1.0 {
                    clashes += 1;
                }
            }
        }
        
        // Check backbone bond lengths (N-CA: ~1.46Å, CA-C: ~1.52Å, C-N: ~1.33Å)
        for chain in self.protein_chains() {
            for residue in chain.residues() {
                if let (Some(n), Some(ca)) = (residue.get_atom("N"), residue.get_atom("CA")) {
                    let dist = n.coords.distance(&ca.coords);
                    if dist < 1.2 || dist > 1.8 { // Expected ~1.46Å ± 0.26Å
                        bad_bonds += 1;
                    }
                }
                
                if let (Some(ca), Some(c)) = (residue.get_atom("CA"), residue.get_atom("C")) {
                    let dist = ca.coords.distance(&c.coords);
                    if dist < 1.2 || dist > 1.8 { // Expected ~1.52Å ± 0.26Å
                        bad_bonds += 1;
                    }
                }
            }
        }
        
        if clashes > 10 {
            errors.push(StructureValidationError::AtomClashes(clashes));
        }
        
        if bad_bonds > 5 {
            errors.push(StructureValidationError::UnreasonableBondLengths(bad_bonds));
        }
    }

    /// Check backbone geometry (phi/psi angles)
    fn check_backbone_geometry(&self, errors: &mut Vec<StructureValidationError>) {
        let mut total_residues = 0;
        let mut poor_geometry = 0;
        
        for chain in self.protein_chains() {
            let residues = chain.residues();
            
            for i in 1..(residues.len()-1) {
                let prev = &residues[i-1];
                let curr = &residues[i];
                let next = &residues[i+1];
                
                // Calculate phi and psi angles
                if let (Some(phi), Some(psi)) = (
                    self.calculate_phi_angle(prev, curr),
                    self.calculate_psi_angle(curr, next)
                ) {
                    total_residues += 1;
                    
                    // Very basic Ramachandran check (simplified)
                    // Poor geometry if angles are in obviously disallowed regions
                    let phi_deg = phi.to_degrees();
                    let psi_deg = psi.to_degrees();
                    
                    // Extremely crude check - just flag very bad angles
                    if phi_deg > 0.0 && psi_deg > 0.0 && 
                       !(phi_deg > 120.0 && phi_deg < 180.0 && psi_deg > 120.0 && psi_deg < 180.0) {
                        poor_geometry += 1;
                    }
                }
            }
        }
        
        if total_residues > 0 {
            let poor_percentage = 100.0 * poor_geometry as f64 / total_residues as f64;
            if poor_percentage > 25.0 { // More than 25% poor geometry is concerning
                errors.push(StructureValidationError::PoorBackboneGeometry(poor_percentage));
            }
        }
    }

    /// Calculate phi dihedral angle (C-1, N, CA, C)
    fn calculate_phi_angle(&self, prev: &crate::geometry::residue::Residue, curr: &crate::geometry::residue::Residue) -> Option<f64> {
        let c_prev = prev.get_atom("C")?;
        let n_curr = curr.get_atom("N")?;
        let ca_curr = curr.get_atom("CA")?;
        let c_curr = curr.get_atom("C")?;
        
        Some(crate::geometry::vector3::calculate_dihedral_angle(
            &c_prev.coords,
            &n_curr.coords,
            &ca_curr.coords,
            &c_curr.coords
        ))
    }

    /// Calculate psi dihedral angle (N, CA, C, N+1)
    fn calculate_psi_angle(&self, curr: &crate::geometry::residue::Residue, next: &crate::geometry::residue::Residue) -> Option<f64> {
        let n_curr = curr.get_atom("N")?;
        let ca_curr = curr.get_atom("CA")?;
        let c_curr = curr.get_atom("C")?;
        let n_next = next.get_atom("N")?;
        
        Some(crate::geometry::vector3::calculate_dihedral_angle(
            &n_curr.coords,
            &ca_curr.coords,
            &c_curr.coords,
            &n_next.coords
        ))
    }

    /// Get structure quality metrics
    pub fn quality_metrics(&self) -> StructureQuality {
        let total_residues = self.residue_count();
        let protein_chains: Vec<_> = self.protein_chains().collect();
        
        // Count residues with backbone
        let complete_backbone = protein_chains.iter()
            .map(|chain| chain.residues())
            .flatten()
            .filter(|residue| residue.has_complete_backbone())
            .count();

        let backbone_completeness = if total_residues > 0 {
            complete_backbone as f64 / total_residues as f64
        } else {
            0.0
        };

        // Count reasonable Ramachandran conformations
        let allowed_conformations = protein_chains.iter()
            .map(|chain| chain.residues())
            .flatten()
            .filter(|residue| residue.is_ramachandran_allowed())
            .count();

        let ramachandran_quality = if total_residues > 0 {
            allowed_conformations as f64 / total_residues as f64
        } else {
            0.0
        };

        StructureQuality {
            resolution: self.metadata.resolution,
            r_work: self.metadata.r_work,
            r_free: self.metadata.r_free,
            backbone_completeness,
            ramachandran_quality,
            total_residues,
            total_atoms: self.atom_count(),
        }
    }

    /// Check if structure is high resolution (< 2.0 Å)
    pub fn is_high_resolution(&self) -> bool {
        self.metadata.resolution.map_or(false, |res| res < 2.0)
    }

    /// Check if structure has good R-factor (< 0.25)
    pub fn has_good_r_factor(&self) -> bool {
        self.metadata.r_work.map_or(false, |r| r < 0.25)
    }
}

/// Structure quality metrics
#[derive(Debug, Clone, PartialEq)]
pub struct StructureQuality {
    pub resolution: Option<f64>,
    pub r_work: Option<f64>,
    pub r_free: Option<f64>,
    pub backbone_completeness: f64,
    pub ramachandran_quality: f64,
    pub total_residues: usize,
    pub total_atoms: usize,
}

/// Structure validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum StructureValidationError {
    NoChains,
    ChainError(char, crate::geometry::chain::ChainValidationError),
    UnreasonableResolution(f64),
    UnreasonableRFactor(f64),
    MissingMetadata,
    CoordinatesOutOfBounds(f64, f64, f64), // x, y, z coordinates beyond reasonable limits
    TooManyMissingAtoms(usize, usize), // missing_count, total_expected
    AtomClashes(usize), // number of severe atom clashes detected
    UnreasonableBondLengths(usize), // number of bonds with unreasonable lengths
    PoorBackboneGeometry(f64), // percentage of residues with poor phi/psi angles
}

impl fmt::Display for StructureValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructureValidationError::NoChains => {
                write!(f, "Structure has no chains")
            }
            StructureValidationError::ChainError(chain_id, error) => {
                write!(f, "Error in chain {}: {}", chain_id, error)
            }
            StructureValidationError::UnreasonableResolution(res) => {
                write!(f, "Unreasonable resolution: {:.2} Å", res)
            }
            StructureValidationError::UnreasonableRFactor(r) => {
                write!(f, "Unreasonable R-factor: {:.3}", r)
            }
            StructureValidationError::MissingMetadata => {
                write!(f, "Missing required metadata")
            }
            StructureValidationError::CoordinatesOutOfBounds(x, y, z) => {
                write!(f, "Coordinates out of bounds: ({:.2}, {:.2}, {:.2})", x, y, z)
            }
            StructureValidationError::TooManyMissingAtoms(missing, total) => {
                write!(f, "Too many missing atoms: {}/{} ({:.1}%)", 
                       missing, total, 100.0 * *missing as f64 / *total as f64)
            }
            StructureValidationError::AtomClashes(count) => {
                write!(f, "Detected {} severe atom clashes", count)
            }
            StructureValidationError::UnreasonableBondLengths(count) => {
                write!(f, "Found {} bonds with unreasonable lengths", count)
            }
            StructureValidationError::PoorBackboneGeometry(percentage) => {
                write!(f, "Poor backbone geometry in {:.1}% of residues", percentage)
            }
        }
    }
}

impl std::error::Error for StructureValidationError {}

impl Default for StructureMetadata {
    fn default() -> Self {
        StructureMetadata {
            classification: String::new(),
            deposition_date: String::new(),
            title: String::new(),
            experimental_method: ExperimentalMethod::Unknown,
            resolution: None,
            r_work: None,
            r_free: None,
            authors: Vec::new(),
            keywords: Vec::new(),
            organism: None,
            expression_system: None,
        }
    }
}

impl fmt::Display for Structure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Structure {} ({:?}): {} chains, {} residues, {} atoms",
            self.id,
            self.metadata.experimental_method,
            self.chain_count(),
            self.residue_count(),
            self.atom_count()
        )
    }
}

impl fmt::Display for ExperimentalMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ExperimentalMethod::XRayDiffraction => "X-ray diffraction",
            ExperimentalMethod::NMRSolution => "NMR solution",
            ExperimentalMethod::NMRSolid => "NMR solid-state",
            ExperimentalMethod::ElectronMicroscopy => "Electron microscopy",
            ExperimentalMethod::ElectronCrystallography => "Electron crystallography",
            ExperimentalMethod::FiberDiffraction => "Fiber diffraction",
            ExperimentalMethod::NeutronDiffraction => "Neutron diffraction",
            ExperimentalMethod::SolutionScattering => "Solution scattering",
            ExperimentalMethod::TheoreticalModel => "Theoretical model",
            ExperimentalMethod::Unknown => "Unknown",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{AminoAcid, Element, Atom, Residue};

    fn create_test_structure() -> Structure {
        let mut structure = Structure::new("TEST".to_string());
        structure.metadata.experimental_method = ExperimentalMethod::XRayDiffraction;
        structure.metadata.resolution = Some(1.5);
        structure.metadata.r_work = Some(0.18);

        // Create test chain
        let mut chain = Chain::protein('A');
        let mut residue = Residue::new(1, AminoAcid::Ala, 'A', None);
        
        // Add atoms
        let n = Atom::new(1, "N".to_string(), Vector3::new(0.0, 0.0, 0.0), Element::N, 'A', 1);
        let ca = Atom::new(2, "CA".to_string(), Vector3::new(1.0, 0.0, 0.0), Element::C, 'A', 1);
        let c = Atom::new(3, "C".to_string(), Vector3::new(2.0, 0.0, 0.0), Element::C, 'A', 1);
        let o = Atom::new(4, "O".to_string(), Vector3::new(3.0, 0.0, 0.0), Element::O, 'A', 1);
        
        residue.add_atom(n);
        residue.add_atom(ca);
        residue.add_atom(c);
        residue.add_atom(o);
        
        chain.add_residue(residue);
        structure.add_chain(chain);
        
        structure
    }

    #[test]
    fn test_structure_creation() {
        let structure = Structure::new("1ABC".to_string());
        assert_eq!(structure.id, "1ABC");
        assert_eq!(structure.chain_count(), 0);
        assert_eq!(structure.residue_count(), 0);
        assert_eq!(structure.atom_count(), 0);
    }

    #[test]
    fn test_chain_management() {
        let mut structure = create_test_structure();
        
        assert_eq!(structure.chain_count(), 1);
        assert!(structure.has_chain('A'));
        assert!(!structure.has_chain('B'));
        
        let chain_ids = structure.chain_ids();
        assert_eq!(chain_ids, &['A']);
        
        let chain_a = structure.get_chain('A').unwrap();
        assert_eq!(chain_a.id, 'A');
        assert_eq!(chain_a.len(), 1);
    }

    #[test]
    fn test_atom_access() {
        let structure = create_test_structure();
        
        assert_eq!(structure.atom_count(), 4);
        assert_eq!(structure.heavy_atom_count(), 4);
        
        let atoms = structure.atoms();
        assert_eq!(atoms.len(), 4);
        
        let ca_atoms = structure.ca_atoms();
        assert_eq!(ca_atoms.len(), 1);
        assert_eq!(ca_atoms[0].name, "CA");
    }

    #[test]
    fn test_residue_access() {
        let structure = create_test_structure();
        
        let residue = structure.get_residue('A', 1).unwrap();
        assert_eq!(residue.amino_acid, AminoAcid::Ala);
        assert_eq!(residue.seq_num, 1);
        
        let atom = structure.find_atom('A', 1, "CA").unwrap();
        assert_eq!(atom.name, "CA");
        assert_eq!(atom.element, Element::C);
    }

    #[test]
    fn test_geometric_properties() {
        let structure = create_test_structure();
        
        let center = structure.geometric_center();
        assert!(center.x >= 0.0 && center.x <= 3.0);
        
        let (min_coords, max_coords) = structure.bounding_box().unwrap();
        assert_eq!(min_coords.x, 0.0);
        assert_eq!(max_coords.x, 3.0);
        
        let radius = structure.radius();
        assert!(radius > 0.0);
        
        let mw = structure.molecular_weight();
        assert!(mw > 50.0); // Should be at least the molecular weight of atoms
    }

    #[test]
    fn test_quality_metrics() {
        let structure = create_test_structure();
        let quality = structure.quality_metrics();
        
        assert_eq!(quality.resolution, Some(1.5));
        assert_eq!(quality.r_work, Some(0.18));
        assert_eq!(quality.total_residues, 1);
        assert_eq!(quality.total_atoms, 4);
        assert!(quality.backbone_completeness > 0.9); // Should have complete backbone
    }

    #[test]
    fn test_validation() {
        let structure = create_test_structure();
        let errors = structure.validate();
        
        // Should be a valid structure
        let has_major_errors = errors.iter().any(|e| {
            matches!(e, 
                StructureValidationError::NoChains |
                StructureValidationError::UnreasonableResolution(_) |
                StructureValidationError::UnreasonableRFactor(_)
            )
        });
        assert!(!has_major_errors);
    }

    #[test]
    fn test_quality_checks() {
        let structure = create_test_structure();
        
        assert!(structure.is_high_resolution()); // 1.5 Å < 2.0 Å
        assert!(structure.has_good_r_factor()); // 0.18 < 0.25
    }

    #[test]
    fn test_secondary_structure_records() {
        let mut structure = create_test_structure();
        
        // Add helix record
        let helix = HelixRecord {
            id: "H1".to_string(),
            helix_class: 1,
            init_chain: 'A',
            init_seq_num: 1,
            init_insertion: None,
            end_chain: 'A',
            end_seq_num: 1,
            end_insertion: None,
            length: 1,
            comment: None,
        };
        structure.helices.push(helix);
        
        structure.assign_secondary_structure_from_records();
        
        let residue = structure.get_residue('A', 1).unwrap();
        assert_eq!(residue.secondary_structure, crate::geometry::residue::SecondaryStructure::Helix);
    }

    #[test]
    fn test_protein_sequence() {
        let structure = create_test_structure();
        let sequence = structure.protein_sequence();
        assert_eq!(sequence, "A"); // Single alanine residue
    }
}