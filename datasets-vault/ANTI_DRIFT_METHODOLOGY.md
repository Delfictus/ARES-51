# CapoAI Anti-Drift Methodology
## ZERO TOLERANCE FOR ARCHITECTURAL DRIFT

### Core Principles (From Screenshot Analysis)

**1. One component per request (not entire system)**
- Single focused implementation per development cycle
- ~1000-3000 lines per component maximum
- Complete with real data, real tests, real benchmarks

**2. Exact specifications (mathematical equations)**
- All equations must be provided with scientific notation
- Algorithm pseudocode included with complexity analysis
- Performance targets specified with measurable constraints
- Memory constraints defined with exact byte allocations

**3. Concrete constraints (performance targets)**
- Latency: <100μs p50, <1ms p99 for protein folding operations
- Throughput: >10,000 protein evaluations/second
- Memory: <16GB working set for 10k residue proteins
- Accuracy: RMSD <2.0Å for 90% of test proteins

**4. Runnable requirement ("include main() with output")**
- Every component must have executable main() function
- Real output demonstrating variable results
- Performance measurements printed with actual timing
- Assertions verify targets are met

### What You Should Actually Do:

**1. "Write the proof generation function with these exact specs..."**
- PRCT phase resonance equations with full mathematical derivation
- Protein folding energy landscape computation with Ramachandran validation
- Performance benchmarks against AlphaFold2 and RoseTTAFold

**2. "Write the verification function for those proofs..."**
- Statistical significance testing (p < 0.001 required)
- Cross-validation with independent test sets
- Ablation studies proving each algorithmic component

**3. "Write validator selection using those proofs..."**
- Automated dataset selection from PDB, CASP, UniProt
- Quality filtering based on resolution, R-factor, confidence scores
- Balanced sampling across protein families and structures

**4. "Connect these three functions into a working demo..."**
- End-to-end protein folding pipeline
- Real-time performance monitoring
- Comparison dashboard showing superiority over existing methods

### FORBIDDEN PATTERNS (ARCHITECTURAL DRIFT CAUSES):

❌ **Hardcoded return values**: `fn spike_rate() -> f64 { 50.0 }`
❌ **Random logic**: `if rand::random() < 0.95 { success }`
❌ **Simulated metrics**: `cpu_usage = 50.0 + activity * 30.0`
❌ **TODO placeholders**: `// TODO: implement later`
❌ **Generic errors**: `Result<T, Box<dyn Error>>`
❌ **Mock data**: Using fake protein structures or synthetic datasets
❌ **Placeholder functions**: Functions that return default values

### REQUIRED PATTERNS (ZERO DRIFT GUARANTEE):

✅ **Computed values from real data**
✅ **Deterministic algorithms based on actual scientific equations**
✅ **System metrics from actual measurements**
✅ **Complete implementations with no stubs**
✅ **Specific error types with detailed diagnostics**
✅ **Real protein datasets from PDB, CASP, UniProt**
✅ **Measured performance against established benchmarks**

### SUCCESS VALIDATION CRITERIA:

Every implementation must pass these tests:
1. **Different inputs → different outputs** (no hardcoding)
2. **Performance measured, not estimated** (real benchmarks)
3. **All assertions pass** (target metrics achieved)
4. **main() produces variable output** (demonstrates real computation)
5. **Zero TODO/FIXME comments** (complete implementation)
6. **Statistical significance proven** (p < 0.001 vs baselines)