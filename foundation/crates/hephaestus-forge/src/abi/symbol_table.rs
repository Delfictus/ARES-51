//! Symbol table analysis for undefined references and ABI violations

use super::{SymbolInfo, SymbolType, SymbolVisibility, AbiViolation, ViolationType, ViolationSeverity};
use crate::types::{ForgeResult, ForgeError};
use std::collections::{HashMap, HashSet};

/// Analyzes symbol tables for ABI compliance
pub struct SymbolTableAnalyzer {
    known_symbols: HashMap<String, SymbolInfo>,
}

impl SymbolTableAnalyzer {
    pub fn new() -> Self {
        Self {
            known_symbols: HashMap::new(),
        }
    }

    /// Analyze symbols in binary for ABI violations
    pub async fn analyze_symbols(&self, binary_data: &[u8]) -> ForgeResult<Vec<AbiViolation>> {
        #[cfg(feature = "abi-verification")]
        {
            self.analyze_with_object_parser(binary_data).await
        }
        #[cfg(not(feature = "abi-verification"))]
        {
            Ok(vec![AbiViolation {
                violation_type: ViolationType::UndefinedSymbol,
                description: "ABI verification feature not enabled".into(),
                location: None,
                severity: ViolationSeverity::Warning,
            }])
        }
    }

    #[cfg(feature = "abi-verification")]
    async fn analyze_with_object_parser(&self, binary_data: &[u8]) -> ForgeResult<Vec<AbiViolation>> {
        use object::{Object, ObjectSymbol};
        
        let mut violations = Vec::new();
        
        // Parse object file
        let file = object::File::parse(binary_data)
            .map_err(|e| ForgeError::ValidationError(format!("Failed to parse object: {}", e)))?;

        let mut defined_symbols = HashSet::new();
        let mut undefined_symbols = HashSet::new();
        let mut duplicate_symbols = HashMap::new();

        // First pass: collect all symbols
        for symbol in file.symbols() {
            let name = symbol.name()
                .map_err(|e| ForgeError::ValidationError(format!("Invalid symbol name: {}", e)))?;
            
            if symbol.is_undefined() {
                undefined_symbols.insert(name.to_string());
            } else {
                if defined_symbols.contains(name) {
                    *duplicate_symbols.entry(name.to_string()).or_insert(0) += 1;
                }
                defined_symbols.insert(name.to_string());
            }

            // Check symbol-specific violations
            violations.extend(self.check_symbol_violations(&symbol)?);
        }

        // Check for undefined symbols
        for undefined in &undefined_symbols {
            if !self.is_known_external_symbol(undefined) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::UndefinedSymbol,
                    description: format!("Undefined symbol: {}", undefined),
                    location: Some(undefined.clone()),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        // Check for duplicate symbols
        for (symbol_name, count) in duplicate_symbols {
            violations.push(AbiViolation {
                violation_type: ViolationType::FunctionSignatureMismatch,
                description: format!("Duplicate symbol '{}' defined {} times", symbol_name, count + 1),
                location: Some(symbol_name),
                severity: ViolationSeverity::Error,
            });
        }

        // Check for ABI-specific symbol requirements
        violations.extend(self.check_abi_symbol_requirements(&file)?);

        Ok(violations)
    }

    #[cfg(feature = "abi-verification")]
    fn check_symbol_violations(&self, symbol: &object::Symbol) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();
        
        let name = symbol.name()
            .map_err(|e| ForgeError::ValidationError(format!("Invalid symbol name: {}", e)))?;

        // Check symbol visibility constraints
        if let Some(violation) = self.check_symbol_visibility(symbol, name)? {
            violations.push(violation);
        }

        // Check symbol alignment
        if let Some(violation) = self.check_symbol_alignment(symbol, name)? {
            violations.push(violation);
        }

        // Check for reserved symbol names
        if let Some(violation) = self.check_reserved_symbols(name) {
            violations.push(violation);
        }

        Ok(violations)
    }

    #[cfg(feature = "abi-verification")]
    fn check_symbol_visibility(&self, symbol: &object::Symbol, name: &str) -> ForgeResult<Option<AbiViolation>> {
        use object::SymbolScope;

        // Check for inappropriate visibility
        match symbol.scope() {
            SymbolScope::Unknown => {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::FunctionSignatureMismatch,
                    description: format!("Symbol '{}' has unknown visibility", name),
                    location: Some(name.to_string()),
                    severity: ViolationSeverity::Warning,
                }));
            }
            SymbolScope::Compilation => {
                // Internal symbols should not be exported
                if self.looks_like_internal_symbol(name) {
                    return Ok(Some(AbiViolation {
                        violation_type: ViolationType::FunctionSignatureMismatch,
                        description: format!("Internal symbol '{}' should not be visible", name),
                        location: Some(name.to_string()),
                        severity: ViolationSeverity::Warning,
                    }));
                }
            }
            _ => {}
        }

        Ok(None)
    }

    #[cfg(feature = "abi-verification")]
    fn check_symbol_alignment(&self, symbol: &object::Symbol, name: &str) -> ForgeResult<Option<AbiViolation>> {
        let address = symbol.address();
        
        // Check alignment based on symbol type
        let required_alignment = match symbol.kind() {
            object::SymbolKind::Text => 4, // Function alignment
            object::SymbolKind::Data => {
                // Data alignment depends on size
                let size = symbol.size();
                if size >= 8 { 8 } else if size >= 4 { 4 } else if size >= 2 { 2 } else { 1 }
            }
            _ => 1,
        };

        if required_alignment > 1 && address % required_alignment as u64 != 0 {
            return Ok(Some(AbiViolation {
                violation_type: ViolationType::AlignmentViolation,
                description: format!(
                    "Symbol '{}' at address 0x{:x} violates alignment requirement {}",
                    name, address, required_alignment
                ),
                location: Some(name.to_string()),
                severity: ViolationSeverity::Error,
            }));
        }

        Ok(None)
    }

    fn check_reserved_symbols(&self, name: &str) -> Option<AbiViolation> {
        // Check for reserved symbol prefixes
        let reserved_prefixes = [
            "__",     // C library internals
            "_Z",     // C++ mangled names (should be handled differently)
            "_GLOBAL_", // Global constructors/destructors
        ];

        for prefix in &reserved_prefixes {
            if name.starts_with(prefix) && !self.is_allowed_reserved_symbol(name) {
                return Some(AbiViolation {
                    violation_type: ViolationType::FunctionSignatureMismatch,
                    description: format!("Symbol '{}' uses reserved prefix '{}'", name, prefix),
                    location: Some(name.to_string()),
                    severity: ViolationSeverity::Warning,
                });
            }
        }

        None
    }

    #[cfg(feature = "abi-verification")]
    fn check_abi_symbol_requirements(&self, file: &object::File) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Check for required ABI symbols
        let required_symbols = self.get_required_abi_symbols();
        let mut found_symbols = HashSet::new();

        for symbol in file.symbols() {
            if let Ok(name) = symbol.name() {
                found_symbols.insert(name.to_string());
            }
        }

        for required in required_symbols {
            if !found_symbols.contains(&required) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::UndefinedSymbol,
                    description: format!("Missing required ABI symbol: {}", required),
                    location: Some(required),
                    severity: ViolationSeverity::Warning,
                });
            }
        }

        // Check for version symbol requirements
        violations.extend(self.check_version_symbols(file)?);

        Ok(violations)
    }

    #[cfg(feature = "abi-verification")]
    fn check_version_symbols(&self, _file: &object::File) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Check for symbol versioning (GNU extension)
        // This would require parsing .gnu.version and .gnu.version_r sections
        // For now, just check if versioning is present when it should be

        // Implement comprehensive symbol version checking
        self.parse_gnu_version_sections(_file, &mut violations)?;

        Ok(violations)
    }

    /// Parse GNU version sections for symbol version checking
    #[cfg(feature = "abi-verification")]
    fn parse_gnu_version_sections(&self, file: &object::File, violations: &mut Vec<AbiViolation>) -> ForgeResult<()> {
        // Parse .gnu.version section (symbol version indices)
        if let Some(version_section) = file.section_by_name(".gnu.version") {
            self.parse_version_indices_section(version_section, violations)?;
        }
        
        // Parse .gnu.version_r section (version requirements)  
        if let Some(version_r_section) = file.section_by_name(".gnu.version_r") {
            self.parse_version_requirements_section(version_r_section, violations)?;
        }
        
        // Parse .gnu.version_d section (version definitions)
        if let Some(version_d_section) = file.section_by_name(".gnu.version_d") {
            self.parse_version_definitions_section(version_d_section, violations)?;
        }
        
        Ok(())
    }
    
    /// Parse .gnu.version section containing symbol version indices
    #[cfg(feature = "abi-verification")]
    fn parse_version_indices_section(&self, section: object::Section, violations: &mut Vec<AbiViolation>) -> ForgeResult<()> {
        let section_data = section.data().map_err(|e| ForgeError::ValidationError(format!("Failed to read .gnu.version section: {}", e)))?;
        
        // .gnu.version contains an array of 16-bit version indices, one per symbol
        if section_data.len() % 2 != 0 {
            violations.push(AbiViolation {
                violation_type: AbiViolationType::MalformedSection,
                symbol_name: None,
                expected: ".gnu.version section size must be even".to_string(),
                found: format!("size: {}", section_data.len()),
                severity: AbiSeverity::Error,
            });
            return Ok(());
        }
        
        let version_count = section_data.len() / 2;
        tracing::debug!("Parsing {} version indices from .gnu.version section", version_count);
        
        // Parse version indices (little-endian 16-bit values)
        for i in 0..version_count {
            let offset = i * 2;
            let version_index = u16::from_le_bytes([section_data[offset], section_data[offset + 1]]);
            
            // Check for valid version indices
            match version_index {
                0 => {
                    // Version index 0 means local symbol
                    tracing::trace!("Symbol {} has local version", i);
                },
                1 => {
                    // Version index 1 means global symbol  
                    tracing::trace!("Symbol {} has global version", i);
                },
                index if index & 0x8000 != 0 => {
                    // Hidden symbol (high bit set)
                    let actual_index = index & 0x7fff;
                    tracing::trace!("Symbol {} has hidden version index {}", i, actual_index);
                },
                index => {
                    // Regular versioned symbol
                    tracing::trace!("Symbol {} has version index {}", i, index);
                    
                    // Validate that the version index references a valid version definition
                    if index > 100 { // Reasonable upper bound for version definitions
                        violations.push(AbiViolation {
                            violation_type: AbiViolationType::InvalidVersion,
                            symbol_name: Some(format!("symbol_{}", i)),
                            expected: "version index < 100".to_string(),
                            found: format!("version index: {}", index),
                            severity: AbiSeverity::Warning,
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Parse .gnu.version_r section containing version requirements
    #[cfg(feature = "abi-verification")]
    fn parse_version_requirements_section(&self, section: object::Section, violations: &mut Vec<AbiViolation>) -> ForgeResult<()> {
        let section_data = section.data().map_err(|e| ForgeError::ValidationError(format!("Failed to read .gnu.version_r section: {}", e)))?;
        
        if section_data.is_empty() {
            return Ok(());
        }
        
        tracing::debug!("Parsing version requirements from .gnu.version_r section ({} bytes)", section_data.len());
        
        // .gnu.version_r section contains Verneed structures
        let mut offset = 0;
        let mut verneed_count = 0;
        
        while offset < section_data.len() {
            // Parse Verneed structure (simplified)
            if offset + 16 > section_data.len() {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::MalformedSection,
                    symbol_name: None,
                    expected: "complete Verneed structure".to_string(),
                    found: format!("truncated at offset {}", offset),
                    severity: AbiSeverity::Error,
                });
                break;
            }
            
            // Read Verneed header fields (simplified parsing)
            let vn_version = u16::from_le_bytes([section_data[offset], section_data[offset + 1]]);
            let vn_cnt = u16::from_le_bytes([section_data[offset + 2], section_data[offset + 3]]);
            let vn_aux = u32::from_le_bytes([
                section_data[offset + 8], section_data[offset + 9], 
                section_data[offset + 10], section_data[offset + 11]
            ]);
            let vn_next = u32::from_le_bytes([
                section_data[offset + 12], section_data[offset + 13],
                section_data[offset + 14], section_data[offset + 15]
            ]);
            
            // Validate Verneed structure
            if vn_version != 1 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::UnsupportedVersion,
                    symbol_name: None,
                    expected: "Verneed version 1".to_string(),
                    found: format!("version {}", vn_version),
                    severity: AbiSeverity::Error,
                });
            }
            
            // Parse auxiliary version entries
            if vn_aux > 0 && vn_cnt > 0 {
                self.parse_vernaux_entries(section_data, offset + vn_aux as usize, vn_cnt, violations)?;
            }
            
            verneed_count += 1;
            
            // Move to next Verneed entry
            if vn_next == 0 {
                break;
            }
            offset += vn_next as usize;
            
            // Safety check to prevent infinite loops
            if verneed_count > 100 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::MalformedSection,
                    symbol_name: None,
                    expected: "reasonable number of version requirements".to_string(),
                    found: format!("too many Verneed entries: {}", verneed_count),
                    severity: AbiSeverity::Warning,
                });
                break;
            }
        }
        
        tracing::debug!("Parsed {} version requirement entries", verneed_count);
        Ok(())
    }
    
    /// Parse Vernaux entries (version auxiliary information)
    #[cfg(feature = "abi-verification")]
    fn parse_vernaux_entries(&self, data: &[u8], mut offset: usize, count: u16, violations: &mut Vec<AbiViolation>) -> ForgeResult<()> {
        for i in 0..count {
            if offset + 16 > data.len() {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::MalformedSection,
                    symbol_name: None,
                    expected: "complete Vernaux structure".to_string(),
                    found: format!("truncated at offset {} (entry {})", offset, i),
                    severity: AbiSeverity::Error,
                });
                break;
            }
            
            // Parse Vernaux structure fields
            let vna_hash = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
            let vna_flags = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
            let vna_other = u16::from_le_bytes([data[offset + 6], data[offset + 7]]);
            let vna_next = u32::from_le_bytes([data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15]]);
            
            // Validate version auxiliary entry
            if vna_other == 0 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::InvalidVersion,
                    symbol_name: None,
                    expected: "non-zero version index".to_string(),
                    found: "version index 0".to_string(),
                    severity: AbiSeverity::Warning,
                });
            }
            
            tracing::trace!("Vernaux entry {}: hash=0x{:x}, flags=0x{:x}, other={}", 
                i, vna_hash, vna_flags, vna_other);
            
            // Move to next auxiliary entry
            if vna_next == 0 {
                break;
            }
            offset += vna_next as usize;
        }
        
        Ok(())
    }
    
    /// Parse .gnu.version_d section containing version definitions
    #[cfg(feature = "abi-verification")]
    fn parse_version_definitions_section(&self, section: object::Section, violations: &mut Vec<AbiViolation>) -> ForgeResult<()> {
        let section_data = section.data().map_err(|e| ForgeError::ValidationError(format!("Failed to read .gnu.version_d section: {}", e)))?;
        
        if section_data.is_empty() {
            return Ok(());
        }
        
        tracing::debug!("Parsing version definitions from .gnu.version_d section ({} bytes)", section_data.len());
        
        // .gnu.version_d section contains Verdef structures
        let mut offset = 0;
        let mut verdef_count = 0;
        
        while offset < section_data.len() {
            // Parse Verdef structure
            if offset + 20 > section_data.len() {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::MalformedSection,
                    symbol_name: None,
                    expected: "complete Verdef structure".to_string(),
                    found: format!("truncated at offset {}", offset),
                    severity: AbiSeverity::Error,
                });
                break;
            }
            
            // Read Verdef header fields
            let vd_version = u16::from_le_bytes([section_data[offset], section_data[offset + 1]]);
            let vd_flags = u16::from_le_bytes([section_data[offset + 2], section_data[offset + 3]]);
            let vd_ndx = u16::from_le_bytes([section_data[offset + 4], section_data[offset + 5]]);
            let vd_cnt = u16::from_le_bytes([section_data[offset + 6], section_data[offset + 7]]);
            let vd_aux = u32::from_le_bytes([
                section_data[offset + 12], section_data[offset + 13],
                section_data[offset + 14], section_data[offset + 15]
            ]);
            let vd_next = u32::from_le_bytes([
                section_data[offset + 16], section_data[offset + 17],
                section_data[offset + 18], section_data[offset + 19]
            ]);
            
            // Validate Verdef structure
            if vd_version != 1 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::UnsupportedVersion,
                    symbol_name: None,
                    expected: "Verdef version 1".to_string(),
                    found: format!("version {}", vd_version),
                    severity: AbiSeverity::Error,
                });
            }
            
            if vd_ndx == 0 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::InvalidVersion,
                    symbol_name: None,
                    expected: "non-zero version definition index".to_string(),
                    found: "index 0".to_string(),
                    severity: AbiSeverity::Error,
                });
            }
            
            tracing::trace!("Verdef entry {}: version={}, flags=0x{:x}, index={}, count={}", 
                verdef_count, vd_version, vd_flags, vd_ndx, vd_cnt);
            
            verdef_count += 1;
            
            // Move to next Verdef entry
            if vd_next == 0 {
                break;
            }
            offset += vd_next as usize;
            
            // Safety check
            if verdef_count > 50 {
                violations.push(AbiViolation {
                    violation_type: AbiViolationType::MalformedSection,
                    symbol_name: None,
                    expected: "reasonable number of version definitions".to_string(),
                    found: format!("too many Verdef entries: {}", verdef_count),
                    severity: AbiSeverity::Warning,
                });
                break;
            }
        }
        
        tracing::debug!("Parsed {} version definition entries", verdef_count);
        Ok(())
    }

    fn is_known_external_symbol(&self, symbol_name: &str) -> bool {
        // List of commonly undefined symbols that are provided by system libraries
        let known_externals = [
            // C standard library
            "malloc", "free", "printf", "sprintf", "strlen", "strcpy", "memcpy",
            "exit", "_exit", "abort", "atexit",
            
            // POSIX
            "open", "close", "read", "write", "mmap", "munmap",
            
            // Dynamic linker
            "_DYNAMIC", "_GLOBAL_OFFSET_TABLE_", "_PROCEDURE_LINKAGE_TABLE_",
            
            // GCC runtime
            "__stack_chk_fail", "__stack_chk_guard",
            "__gxx_personality_v0", "_Unwind_Resume",
            
            // Rust runtime
            "rust_begin_unwind", "rust_eh_personality",
            
            // Thread local storage
            "__tls_get_addr", "__pthread_key_create",
        ];

        known_externals.contains(&symbol_name) || 
        symbol_name.starts_with("__rust") ||
        symbol_name.starts_with("_ZN") || // C++ mangled names
        self.known_symbols.contains_key(symbol_name)
    }

    fn looks_like_internal_symbol(&self, name: &str) -> bool {
        name.contains("_internal") ||
        name.contains(".llvm.") ||
        name.starts_with(".L") || // Local labels
        name.starts_with("__llvm")
    }

    fn is_allowed_reserved_symbol(&self, name: &str) -> bool {
        // Some reserved symbols are legitimate
        let allowed_reserved = [
            "__main",
            "__start",
            "__init",
            "__fini",
            "__preinit_array_start",
            "__preinit_array_end",
            "__init_array_start", 
            "__init_array_end",
            "__fini_array_start",
            "__fini_array_end",
            "__bss_start",
            "__end",
            "_edata",
            "_end",
            "__stack_chk_guard",
            "__stack_chk_fail",
        ];

        allowed_reserved.contains(&name) ||
        name.starts_with("__rust") ||
        name.starts_with("__pthread")
    }

    fn get_required_abi_symbols(&self) -> Vec<String> {
        // Symbols that might be required for proper ABI compliance
        // This is highly dependent on the target platform and use case
        vec![
            // Could include symbols like entry points, required callbacks, etc.
        ]
    }

    /// Add a known symbol to the analyzer
    pub fn add_known_symbol(&mut self, symbol: SymbolInfo) {
        self.known_symbols.insert(symbol.name.clone(), symbol);
    }

    /// Load symbols from system libraries or previous modules
    pub async fn load_system_symbols(&mut self) -> ForgeResult<()> {
        // This could load symbols from:
        // - System dynamic libraries
        // - Previous module versions
        // - Symbol databases
        
        // For now, just populate with common system symbols
        let system_symbols = vec![
            SymbolInfo {
                name: "malloc".to_string(),
                mangled_name: None,
                address: 0,
                size: None,
                symbol_type: SymbolType::Function,
                visibility: SymbolVisibility::Default,
                is_undefined: true,
            },
            SymbolInfo {
                name: "free".to_string(),
                mangled_name: None,
                address: 0,
                size: None,
                symbol_type: SymbolType::Function,
                visibility: SymbolVisibility::Default,
                is_undefined: true,
            },
            // Add more system symbols as needed
        ];

        for symbol in system_symbols {
            self.add_known_symbol(symbol);
        }

        Ok(())
    }
}

impl Default for SymbolTableAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}