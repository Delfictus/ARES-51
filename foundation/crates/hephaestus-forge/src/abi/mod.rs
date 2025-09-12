//! ABI Verification Module
//! 
//! Comprehensive Application Binary Interface verification for synthesized modules.
//! Provides formal guarantees for ABI compatibility, type safety, and calling conventions.

pub mod signature;
pub mod layout;
pub mod calling_convention;
pub mod symbol_table;
pub mod compatibility;

use crate::types::{ForgeResult, ForgeError, VersionedModule};
use std::collections::HashMap;

pub use signature::SignatureValidator;
pub use layout::DataLayoutValidator;
pub use calling_convention::CallingConventionValidator;
pub use symbol_table::SymbolTableAnalyzer;
pub use compatibility::CompatibilityChecker;

/// Target platform architecture for ABI verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetArchitecture {
    X86_64,
    AArch64,
    X86,
    RISCV64,
    WASM32,
}

/// Calling convention types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallingConvention {
    SystemV,        // Unix x86-64
    Win64,          // Windows x64
    AAPCS,          // ARM AAPCS
    AAPCS64,        // ARM64 AAPCS
    C,              // Standard C
    Rust,           // Rust ABI (unstable)
    RustCall,       // Rust internal
    CDecl,          // cdecl (x86)
    StdCall,        // stdcall (Windows x86)
    FastCall,       // fastcall
}

/// Data type alignment requirements
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlignment {
    pub type_name: String,
    pub size: usize,
    pub alignment: usize,
    pub is_packed: bool,
}

/// Function signature for ABI verification
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub return_type: TypeInfo,
    pub parameters: Vec<Parameter>,
    pub calling_convention: CallingConvention,
    pub is_variadic: bool,
    pub mangled_name: Option<String>,
}

/// Parameter information
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Option<String>,
    pub type_info: TypeInfo,
    pub is_register: bool,
    pub register_location: Option<String>,
}

/// Comprehensive type information
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: String,
    pub size: usize,
    pub alignment: usize,
    pub kind: TypeKind,
    pub fields: Vec<FieldInfo>,
    pub is_union: bool,
}

/// Type classification
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Void,
    Integer { width: u8, signed: bool },
    Float { width: u8 },
    Pointer { pointee: Box<TypeInfo> },
    Array { element: Box<TypeInfo>, count: usize },
    Structure,
    Union,
    Function { signature: Box<FunctionSignature> },
    Enum { underlying: Box<TypeInfo> },
    Unknown,
}

/// Field information for structures and unions
#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub type_info: TypeInfo,
    pub offset: usize,
    pub bit_offset: Option<u8>,
    pub bit_size: Option<u8>,
}

/// Symbol information from symbol table
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub mangled_name: Option<String>,
    pub address: u64,
    pub size: Option<u64>,
    pub symbol_type: SymbolType,
    pub visibility: SymbolVisibility,
    pub is_undefined: bool,
}

/// Symbol type classification
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolType {
    Function,
    Object,
    Section,
    File,
    Common,
    TLS,
    GNU_IFunc,
    Unknown,
}

/// Symbol visibility
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolVisibility {
    Default,
    Hidden,
    Protected,
    Internal,
}

/// ABI compatibility report
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub is_compatible: bool,
    pub violations: Vec<AbiViolation>,
    pub warnings: Vec<AbiWarning>,
    pub verified_functions: usize,
    pub verified_types: usize,
    pub analysis_duration_ms: u64,
}

/// ABI violation details
#[derive(Debug, Clone)]
pub struct AbiViolation {
    pub violation_type: ViolationType,
    pub description: String,
    pub location: Option<String>,
    pub severity: ViolationSeverity,
}

/// Types of ABI violations
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    FunctionSignatureMismatch,
    CallingConventionMismatch,
    DataLayoutMismatch,
    UndefinedSymbol,
    TypeSizeMismatch,
    AlignmentViolation,
    VersionIncompatibility,
}

/// Severity levels for violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// ABI warning (non-fatal)
#[derive(Debug, Clone)]
pub struct AbiWarning {
    pub message: String,
    pub location: Option<String>,
}

/// Main ABI verifier
pub struct AbiVerifier {
    signature_validator: SignatureValidator,
    layout_validator: DataLayoutValidator,
    calling_conv_validator: CallingConventionValidator,
    symbol_analyzer: SymbolTableAnalyzer,
    compatibility_checker: CompatibilityChecker,
    target_arch: TargetArchitecture,
}

impl AbiVerifier {
    /// Create new ABI verifier for target architecture
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self {
            signature_validator: SignatureValidator::new(target_arch.clone()),
            layout_validator: DataLayoutValidator::new(target_arch.clone()),
            calling_conv_validator: CallingConventionValidator::new(target_arch.clone()),
            symbol_analyzer: SymbolTableAnalyzer::new(),
            compatibility_checker: CompatibilityChecker::new(target_arch.clone()),
            target_arch,
        }
    }

    /// Perform comprehensive ABI verification
    pub async fn verify_module_abi(
        &self,
        module: &VersionedModule,
        reference_module: Option<&VersionedModule>,
    ) -> ForgeResult<CompatibilityReport> {
        let start_time = std::time::Instant::now();

        // Parse module binary and extract DWARF debug information
        let debug_info = self.extract_debug_info(&module.code)?;
        
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut verified_functions = 0;
        let mut verified_types = 0;

        // 1. Validate function signatures
        match self.signature_validator.validate_signatures(&debug_info).await {
            Ok(results) => {
                verified_functions = results.len();
                for result in results {
                    if let Some(violation) = result.violation {
                        violations.push(violation);
                    }
                    warnings.extend(result.warnings);
                }
            }
            Err(e) => {
                violations.push(AbiViolation {
                    violation_type: ViolationType::FunctionSignatureMismatch,
                    description: format!("Function signature validation failed: {}", e),
                    location: None,
                    severity: ViolationSeverity::Error,
                });
            }
        }

        // 2. Validate data structure layouts
        match self.layout_validator.validate_layouts(&debug_info).await {
            Ok(results) => {
                verified_types = results.len();
                for result in results {
                    if let Some(violation) = result.violation {
                        violations.push(violation);
                    }
                    warnings.extend(result.warnings);
                }
            }
            Err(e) => {
                violations.push(AbiViolation {
                    violation_type: ViolationType::DataLayoutMismatch,
                    description: format!("Data layout validation failed: {}", e),
                    location: None,
                    severity: ViolationSeverity::Error,
                });
            }
        }

        // 3. Verify calling conventions
        if let Err(e) = self.calling_conv_validator.verify_calling_conventions(&debug_info).await {
            violations.push(AbiViolation {
                violation_type: ViolationType::CallingConventionMismatch,
                description: format!("Calling convention verification failed: {}", e),
                location: None,
                severity: ViolationSeverity::Error,
            });
        }

        // 4. Analyze symbol table for undefined references
        match self.symbol_analyzer.analyze_symbols(&module.code).await {
            Ok(symbol_violations) => {
                violations.extend(symbol_violations);
            }
            Err(e) => {
                violations.push(AbiViolation {
                    violation_type: ViolationType::UndefinedSymbol,
                    description: format!("Symbol analysis failed: {}", e),
                    location: None,
                    severity: ViolationSeverity::Error,
                });
            }
        }

        // 5. Cross-module compatibility if reference provided
        if let Some(reference) = reference_module {
            match self.compatibility_checker.check_compatibility(module, reference).await {
                Ok(compat_violations) => {
                    violations.extend(compat_violations);
                }
                Err(e) => {
                    violations.push(AbiViolation {
                        violation_type: ViolationType::VersionIncompatibility,
                        description: format!("Compatibility check failed: {}", e),
                        location: None,
                        severity: ViolationSeverity::Error,
                    });
                }
            }
        }

        let analysis_duration = start_time.elapsed().as_millis() as u64;
        
        // Determine overall compatibility
        let is_compatible = !violations.iter().any(|v| 
            matches!(v.severity, ViolationSeverity::Error | ViolationSeverity::Critical)
        );

        Ok(CompatibilityReport {
            is_compatible,
            violations,
            warnings,
            verified_functions,
            verified_types,
            analysis_duration_ms: analysis_duration,
        })
    }

    /// Extract debug information from module binary
    fn extract_debug_info(&self, binary_data: &[u8]) -> ForgeResult<DebugInfo> {
        #[cfg(feature = "abi-verification")]
        {
            self.parse_debug_info_with_gimli(binary_data)
        }
        #[cfg(not(feature = "abi-verification"))]
        {
            Err(ForgeError::ValidationError(
                "ABI verification feature not enabled. Enable 'abi-verification' feature.".into()
            ))
        }
    }

    #[cfg(feature = "abi-verification")]
    fn parse_debug_info_with_gimli(&self, binary_data: &[u8]) -> ForgeResult<DebugInfo> {
        use object::{Object, ObjectSection};
        use gimli::{Dwarf, EndianSlice, LittleEndian};

        // Parse object file
        let file = object::File::parse(binary_data)
            .map_err(|e| ForgeError::ValidationError(format!("Failed to parse object file: {}", e)))?;

        // Extract DWARF sections
        let dwarf = gimli::Dwarf::load(&file, |section| -> Result<_, gimli::Error> {
            match file.section_by_name(section.name()) {
                Some(section) => Ok(section.data().unwrap_or(&[])),
                None => Ok(&[]),
            }
        }).map_err(|e| ForgeError::ValidationError(format!("Failed to load DWARF: {}", e)))?;

        // Parse debug information
        Ok(DebugInfo::from_dwarf(dwarf, &file)?)
    }
}

/// Debug information extracted from DWARF
#[cfg(feature = "abi-verification")]
pub struct DebugInfo {
    pub functions: Vec<FunctionSignature>,
    pub types: Vec<TypeInfo>,
    pub symbols: Vec<SymbolInfo>,
}

#[cfg(not(feature = "abi-verification"))]
pub struct DebugInfo {
    pub functions: Vec<FunctionSignature>,
    pub types: Vec<TypeInfo>,
    pub symbols: Vec<SymbolInfo>,
}

impl DebugInfo {
    #[cfg(feature = "abi-verification")]
    fn from_dwarf(dwarf: gimli::Dwarf<&[u8]>, object_file: &object::File) -> ForgeResult<Self> {
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut symbols = Vec::new();

        // Parse compilation units
        let mut units = dwarf.units();
        while let Some(header) = units.next()
            .map_err(|e| ForgeError::ValidationError(format!("DWARF unit iteration failed: {}", e)))?
        {
            let unit = dwarf.unit(header)
                .map_err(|e| ForgeError::ValidationError(format!("Failed to load DWARF unit: {}", e)))?;

            // Parse DIEs (Debug Information Entries)
            let mut cursor = unit.entries();
            while let Some((_, entry)) = cursor.next_dfs()
                .map_err(|e| ForgeError::ValidationError(format!("DIE traversal failed: {}", e)))?
            {
                match entry.tag() {
                    gimli::DW_TAG_subprogram => {
                        if let Ok(function) = Self::parse_function_signature(&dwarf, &unit, entry) {
                            functions.push(function);
                        }
                    }
                    gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type | gimli::DW_TAG_class_type => {
                        if let Ok(type_info) = Self::parse_type_info(&dwarf, &unit, entry) {
                            types.push(type_info);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Parse symbol table
        for section in object_file.sections() {
            if section.kind() == object::SectionKind::Text {
                // Process symbols in text section
                for symbol in object_file.symbols() {
                    if let Ok(symbol_info) = Self::parse_symbol_info(&symbol) {
                        symbols.push(symbol_info);
                    }
                }
            }
        }

        Ok(DebugInfo {
            functions,
            types,
            symbols,
        })
    }

    #[cfg(feature = "abi-verification")]
    fn parse_function_signature(
        dwarf: &gimli::Dwarf<&[u8]>,
        unit: &gimli::Unit<&[u8]>,
        entry: &gimli::DebuggingInformationEntry<&[u8]>
    ) -> Result<FunctionSignature, gimli::Error> {
        let mut name = String::new();
        let mut return_type = TypeInfo::void();
        let mut parameters = Vec::new();
        let mut calling_convention = CallingConvention::C;
        let mut is_variadic = false;
        let mut mangled_name = None;

        // Parse function attributes
        let mut attrs = entry.attrs();
        while let Some(attr) = attrs.next()? {
            match attr.name() {
                gimli::DW_AT_name => {
                    if let gimli::AttributeValue::String(s) = attr.value() {
                        name = String::from_utf8_lossy(&dwarf.attr_string(unit, s)?).to_string();
                    }
                }
                gimli::DW_AT_linkage_name => {
                    if let gimli::AttributeValue::String(s) = attr.value() {
                        mangled_name = Some(String::from_utf8_lossy(&dwarf.attr_string(unit, s)?).to_string());
                    }
                }
                gimli::DW_AT_calling_convention => {
                    if let gimli::AttributeValue::Data1(cc) = attr.value() {
                        calling_convention = Self::decode_calling_convention(cc);
                    }
                }
                _ => {}
            }
        }

        Ok(FunctionSignature {
            name,
            return_type,
            parameters,
            calling_convention,
            is_variadic,
            mangled_name,
        })
    }

    #[cfg(feature = "abi-verification")]
    fn parse_type_info(
        dwarf: &gimli::Dwarf<&[u8]>,
        unit: &gimli::Unit<&[u8]>,
        entry: &gimli::DebuggingInformationEntry<&[u8]>
    ) -> Result<TypeInfo, gimli::Error> {
        let mut name = String::new();
        let mut size = 0;
        let mut alignment = 1;
        let mut fields = Vec::new();
        let is_union = entry.tag() == gimli::DW_TAG_union_type;

        // Parse type attributes
        let mut attrs = entry.attrs();
        while let Some(attr) = attrs.next()? {
            match attr.name() {
                gimli::DW_AT_name => {
                    if let gimli::AttributeValue::String(s) = attr.value() {
                        name = String::from_utf8_lossy(&dwarf.attr_string(unit, s)?).to_string();
                    }
                }
                gimli::DW_AT_byte_size => {
                    if let gimli::AttributeValue::Udata(s) = attr.value() {
                        size = s as usize;
                    }
                }
                gimli::DW_AT_alignment => {
                    if let gimli::AttributeValue::Udata(a) = attr.value() {
                        alignment = a as usize;
                    }
                }
                _ => {}
            }
        }

        Ok(TypeInfo {
            name,
            size,
            alignment,
            kind: TypeKind::Structure,
            fields,
            is_union,
        })
    }

    #[cfg(feature = "abi-verification")]
    fn parse_symbol_info(symbol: &object::Symbol) -> Result<SymbolInfo, object::Error> {
        let name = symbol.name()?.to_string();
        let mangled_name = None; // Could extract from symbol if available
        let address = symbol.address();
        let size = symbol.size();
        let is_undefined = symbol.is_undefined();

        let symbol_type = match symbol.kind() {
            object::SymbolKind::Text => SymbolType::Function,
            object::SymbolKind::Data => SymbolType::Object,
            object::SymbolKind::Section => SymbolType::Section,
            object::SymbolKind::File => SymbolType::File,
            object::SymbolKind::Common => SymbolType::Common,
            object::SymbolKind::Tls => SymbolType::TLS,
            _ => SymbolType::Unknown,
        };

        let visibility = match symbol.scope() {
            object::SymbolScope::Linkage => SymbolVisibility::Default,
            object::SymbolScope::Compilation => SymbolVisibility::Internal,
            _ => SymbolVisibility::Default,
        };

        Ok(SymbolInfo {
            name,
            mangled_name,
            address,
            size,
            symbol_type,
            visibility,
            is_undefined,
        })
    }

    #[cfg(feature = "abi-verification")]
    fn decode_calling_convention(cc: u8) -> CallingConvention {
        match cc {
            gimli::constants::DW_CC_normal => CallingConvention::C,
            gimli::constants::DW_CC_program => CallingConvention::SystemV,
            gimli::constants::DW_CC_nocall => CallingConvention::Rust,
            _ => CallingConvention::C,
        }
    }

    #[cfg(not(feature = "abi-verification"))]
    pub fn empty() -> Self {
        Self {
            functions: Vec::new(),
            types: Vec::new(),
            symbols: Vec::new(),
        }
    }
}

impl TypeInfo {
    pub fn void() -> Self {
        Self {
            name: "void".to_string(),
            size: 0,
            alignment: 1,
            kind: TypeKind::Void,
            fields: Vec::new(),
            is_union: false,
        }
    }
}

impl Default for TargetArchitecture {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        return TargetArchitecture::X86_64;
        #[cfg(target_arch = "aarch64")]
        return TargetArchitecture::AArch64;
        #[cfg(target_arch = "x86")]
        return TargetArchitecture::X86;
        #[cfg(target_arch = "riscv64")]
        return TargetArchitecture::RISCV64;
        #[cfg(target_arch = "wasm32")]
        return TargetArchitecture::WASM32;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "x86", target_arch = "riscv64", target_arch = "wasm32")))]
        return TargetArchitecture::X86_64; // Default fallback
    }
}