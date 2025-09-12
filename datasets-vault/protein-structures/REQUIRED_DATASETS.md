# Required Protein Structure Datasets for CapoAI
## ZERO DRIFT IMPLEMENTATION - REAL DATA ONLY

### Primary Datasets (MANDATORY)

#### 1. Protein Data Bank (PDB) - Complete Collection
**Location**: https://www.rcsb.org/downloads/ftp
**Size**: ~200,000+ experimentally determined structures
**Format**: PDB/PDBx-mmCIF files
**Download Command**: 
```bash
rsync -rlpt -v -z --delete rsync.rcsb.org::ftp_data/structures/divided/pdb/ ./pdb/
```

**Required Fields Per Structure**:
- **Atomic Coordinates**: (x, y, z) positions for all atoms
- **Resolution**: X-ray/NMR resolution in Angstroms  
- **R-factor**: Reliability metric (must be <0.25)
- **B-factors**: Atomic displacement parameters
- **Secondary Structure**: HELIX/SHEET records
- **SEQRES**: Complete amino acid sequence
- **Contact Maps**: Atom-atom distances <8Å
- **Ramachandran Angles**: φ/ψ backbone torsions

#### 2. CASP (Critical Assessment of Protein Structure Prediction)
**Location**: https://predictioncenter.org/download_area/
**Datasets**: CASP13, CASP14, CASP15 targets
**Size**: ~100 blind prediction targets per competition

**Required Data Per Target**:
- **Target Sequence**: FASTA format amino acid sequence
- **Native Structure**: Experimental PDB coordinates
- **Difficulty Classification**: Easy/Medium/Hard/Template-free
- **GDT-TS Scores**: Global Distance Test scores for all predictors
- **RMSD Values**: Root Mean Square Deviation measurements
- **Template Availability**: Homology modeling templates

#### 3. AlphaFold Database
**Location**: https://alphafold.ebi.ac.uk/
**Size**: 200M+ predicted protein structures
**Format**: PDB format with confidence scores

**Required Fields**:
- **Confidence Scores**: Per-residue reliability (0-100)
- **Predicted Coordinates**: (x, y, z) positions
- **UniProt Mapping**: Cross-reference to sequence databases
- **Organism Classification**: Taxonomic information
- **Domain Annotations**: Structural domain boundaries

#### 4. Molecular Dynamics Trajectories
**Sources**: 
- Folding@home: https://foldingathome.org/
- DESRES Anton trajectories: https://www.deshawresearch.com/
- GPCRmd: http://www.gpcrmd.org/

**Required Data**:
- **Trajectory Frames**: Atomic coordinates vs time
- **Time Resolution**: 2fs timesteps minimum
- **Force Field**: CHARMM36, AMBER99sb-disp, or OPLS-AA
- **Solvation**: Explicit water + ions
- **Temperature**: 300K ± 10K simulation conditions
- **Pressure**: 1 atm simulation pressure

### Specialized Datasets

#### 5. Drug-Target Interaction Networks
**Sources**: 
- ChEMBL: https://www.ebi.ac.uk/chembl/
- DrugBank: https://go.drugbank.com/
- BindingDB: https://www.bindingdb.org/

**Required Fields**:
- **Binding Affinities**: Kd, Ki, IC50 values in nM/μM
- **Chemical Structures**: SMILES/SDF molecular representations
- **Target Sequences**: Protein amino acid sequences
- **Binding Sites**: Residue-level interaction annotations
- **Thermodynamic Data**: ΔG, ΔH, ΔS values where available

#### 6. Folding Kinetics Data
**Sources**:
- Protein Folding Database: http://pfd.med.monash.edu.au/
- Literature compilation: Manually curated from papers

**Required Measurements**:
- **Folding Rates**: kf in s⁻¹
- **Unfolding Rates**: ku in s⁻¹  
- **Thermodynamic Stability**: ΔGf in kcal/mol
- **Chevron Plot Data**: Rate vs denaturant concentration
- **Temperature Dependencies**: Arrhenius parameters

### Validation Benchmarks

#### 7. SCOP (Structural Classification of Proteins)
**Location**: https://scop.mrc-lmb.cam.ac.uk/
**Purpose**: Hierarchical protein structure classification
**Required**: Family/superfamily/fold classifications for each PDB entry

#### 8. CATH Database
**Location**: https://www.cathdb.info/
**Purpose**: Alternative structural classification
**Required**: Class/Architecture/Topology/Homology levels

#### 9. Membrane Protein Datasets
**Sources**:
- PDBTM: http://pdbtm.enzim.hu/
- OPM: https://opm.phar.umich.edu/
**Special Requirements**: Lipid bilayer positioning data

### Data Processing Requirements

#### Real-Time Processing Pipeline
```rust
struct ProteinDataProcessor {
    pdb_cache: HashMap<String, ProteinStructure>,
    casp_targets: Vec<CaspTarget>,
    alphafold_predictions: HashMap<String, AlphaFoldStructure>,
    md_trajectories: Vec<TrajectoryData>,
    binding_affinities: HashMap<String, Vec<BindingData>>,
}

impl ProteinDataProcessor {
    fn load_complete_dataset(&mut self) -> Result<(), DataError> {
        // Load ALL required datasets
        // NO stubs, NO placeholders, NO mock data
    }
    
    fn validate_data_quality(&self) -> DataQualityReport {
        // REAL validation metrics
        // Resolution checks, R-factor validation
        // Coordinate completeness verification
    }
    
    fn compute_performance_benchmarks(&self) -> BenchmarkResults {
        // MEASURED performance against baselines
        // Statistical significance testing
        // Cross-validation results
    }
}
```

#### Quality Control Metrics
- **Resolution Cutoff**: <3.0Å for X-ray structures
- **R-factor Limit**: <0.25 for reliable structures  
- **Completeness**: >95% non-hydrogen atoms present
- **Geometry**: <5% Ramachandran outliers
- **B-factor**: <100Å² average displacement

### Storage Requirements
- **Total Size**: ~500TB for complete datasets
- **Access Pattern**: Random access for structure lookup
- **Backup Strategy**: 3-2-1 rule (3 copies, 2 media, 1 offsite)
- **Compression**: gzip for PDB files, HDF5 for trajectories

### Update Schedule
- **PDB**: Weekly synchronization
- **CASP**: Biennial competition updates  
- **AlphaFold**: Quarterly releases
- **Literature**: Monthly manual curation

**NO SYNTHETIC DATA - ALL DATASETS MUST BE EXPERIMENTAL OR VALIDATED PREDICTIONS**