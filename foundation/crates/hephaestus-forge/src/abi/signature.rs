//! Function signature validation with DWARF debug information

use super::{
    FunctionSignature, DebugInfo, AbiViolation, AbiWarning, ViolationType, ViolationSeverity,
    TargetArchitecture, CallingConvention, TypeInfo, TypeKind
};
use crate::types::{ForgeResult, ForgeError};
use std::collections::HashMap;

/// Result of signature validation
#[derive(Debug)]
pub struct SignatureValidationResult {
    pub function_name: String,
    pub is_valid: bool,
    pub violation: Option<AbiViolation>,
    pub warnings: Vec<AbiWarning>,
}

/// Validates function signatures against ABI requirements
pub struct SignatureValidator {
    target_arch: TargetArchitecture,
    type_cache: HashMap<String, TypeInfo>,
}

impl SignatureValidator {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self {
            target_arch,
            type_cache: HashMap::new(),
        }
    }

    /// Validate all function signatures in debug information
    pub async fn validate_signatures(
        &self,
        debug_info: &DebugInfo,
    ) -> ForgeResult<Vec<SignatureValidationResult>> {
        let mut results = Vec::new();

        for function in &debug_info.functions {
            let result = self.validate_single_signature(function, debug_info).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Validate a single function signature
    async fn validate_single_signature(
        &self,
        signature: &FunctionSignature,
        debug_info: &DebugInfo,
    ) -> ForgeResult<SignatureValidationResult> {
        let mut warnings = Vec::new();
        let mut violation = None;

        // 1. Validate calling convention compatibility
        if let Some(conv_violation) = self.validate_calling_convention(signature) {
            violation = Some(conv_violation);
        }

        // 2. Validate return type
        if let Some(ret_violation) = self.validate_return_type(&signature.return_type).await? {
            violation = Some(ret_violation);
        }

        // 3. Validate parameters
        if let Some(param_violation) = self.validate_parameters(&signature.parameters).await? {
            if violation.is_none() {
                violation = Some(param_violation);
            }
        }

        // 4. Check for ABI-specific requirements
        self.check_abi_requirements(signature, &mut warnings);

        // 5. Validate type consistency
        if let Some(consistency_violation) = self.validate_type_consistency(signature, debug_info)? {
            if violation.is_none() {
                violation = Some(consistency_violation);
            }
        }

        let is_valid = violation.is_none();

        Ok(SignatureValidationResult {
            function_name: signature.name.clone(),
            is_valid,
            violation,
            warnings,
        })
    }

    /// Validate calling convention for target architecture
    fn validate_calling_convention(&self, signature: &FunctionSignature) -> Option<AbiViolation> {
        let expected_conventions = self.get_supported_calling_conventions();
        
        if !expected_conventions.contains(&signature.calling_convention) {
            return Some(AbiViolation {
                violation_type: ViolationType::CallingConventionMismatch,
                description: format!(
                    "Unsupported calling convention {:?} for architecture {:?}. Expected one of: {:?}",
                    signature.calling_convention,
                    self.target_arch,
                    expected_conventions
                ),
                location: Some(signature.name.clone()),
                severity: ViolationSeverity::Error,
            });
        }

        None
    }

    /// Get supported calling conventions for target architecture
    fn get_supported_calling_conventions(&self) -> Vec<CallingConvention> {
        match self.target_arch {
            TargetArchitecture::X86_64 => vec![
                CallingConvention::SystemV,
                CallingConvention::Win64,
                CallingConvention::C,
                CallingConvention::Rust,
                CallingConvention::RustCall,
            ],
            TargetArchitecture::AArch64 => vec![
                CallingConvention::AAPCS64,
                CallingConvention::C,
                CallingConvention::Rust,
                CallingConvention::RustCall,
            ],
            TargetArchitecture::X86 => vec![
                CallingConvention::CDecl,
                CallingConvention::StdCall,
                CallingConvention::FastCall,
                CallingConvention::C,
            ],
            TargetArchitecture::RISCV64 => vec![
                CallingConvention::C,
                CallingConvention::Rust,
                CallingConvention::RustCall,
            ],
            TargetArchitecture::WASM32 => vec![
                CallingConvention::C,
            ],
        }
    }

    /// Validate return type compatibility
    async fn validate_return_type(&self, return_type: &TypeInfo) -> ForgeResult<Option<AbiViolation>> {
        // Check if return type can be passed in registers vs memory
        let max_register_size = self.get_max_register_return_size();
        
        if return_type.size > max_register_size {
            // Large return types must be returned via hidden pointer parameter
            if !self.is_aggregate_type(&return_type.kind) {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::TypeSizeMismatch,
                    description: format!(
                        "Return type '{}' size ({} bytes) exceeds maximum register return size ({} bytes) but is not an aggregate type",
                        return_type.name,
                        return_type.size,
                        max_register_size
                    ),
                    location: None,
                    severity: ViolationSeverity::Warning,
                }));
            }
        }

        // Validate type alignment
        if let Some(alignment_violation) = self.validate_type_alignment(return_type)? {
            return Ok(Some(alignment_violation));
        }

        Ok(None)
    }

    /// Validate function parameters
    async fn validate_parameters(&self, parameters: &[super::Parameter]) -> ForgeResult<Option<AbiViolation>> {
        let max_params = self.get_max_register_parameters();
        let mut register_params = 0;

        for (i, param) in parameters.iter().enumerate() {
            // Check parameter passing mechanism
            if param.is_register {
                register_params += 1;
                if register_params > max_params {
                    return Ok(Some(AbiViolation {
                        violation_type: ViolationType::CallingConventionMismatch,
                        description: format!(
                            "Parameter {} exceeds maximum register parameters ({}) for target architecture",
                            i,
                            max_params
                        ),
                        location: param.name.clone(),
                        severity: ViolationSeverity::Error,
                    }));
                }
            }

            // Validate parameter type alignment
            if let Some(alignment_violation) = self.validate_type_alignment(&param.type_info)? {
                return Ok(Some(alignment_violation));
            }

            // Validate parameter size constraints
            if param.type_info.size == 0 && !matches!(param.type_info.kind, TypeKind::Void) {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::TypeSizeMismatch,
                    description: format!(
                        "Parameter '{}' has zero size but is not void type",
                        param.name.as_deref().unwrap_or("unnamed")
                    ),
                    location: param.name.clone(),
                    severity: ViolationSeverity::Error,
                }));
            }
        }

        Ok(None)
    }

    /// Validate type alignment requirements
    fn validate_type_alignment(&self, type_info: &TypeInfo) -> ForgeResult<Option<AbiViolation>> {
        let expected_alignment = self.calculate_type_alignment(&type_info.kind, type_info.size);
        
        if type_info.alignment != expected_alignment {
            return Ok(Some(AbiViolation {
                violation_type: ViolationType::AlignmentViolation,
                description: format!(
                    "Type '{}' has alignment {} but expected alignment {} for architecture {:?}",
                    type_info.name,
                    type_info.alignment,
                    expected_alignment,
                    self.target_arch
                ),
                location: None,
                severity: ViolationSeverity::Error,
            }));
        }

        Ok(None)
    }

    /// Calculate expected alignment for a type
    fn calculate_type_alignment(&self, type_kind: &TypeKind, size: usize) -> usize {
        match type_kind {
            TypeKind::Void => 1,
            TypeKind::Integer { width, .. } => {
                let byte_width = (*width as usize + 7) / 8;
                self.get_natural_alignment(byte_width)
            }
            TypeKind::Float { width } => {
                let byte_width = (*width as usize + 7) / 8;
                self.get_natural_alignment(byte_width)
            }
            TypeKind::Pointer { .. } => {
                self.get_pointer_alignment()
            }
            TypeKind::Array { element, .. } => {
                self.calculate_type_alignment(&element.kind, element.size)
            }
            TypeKind::Structure | TypeKind::Union => {
                // Structure alignment is the maximum alignment of its fields
                // For now, use natural alignment based on size
                self.get_natural_alignment(size)
            }
            TypeKind::Function { .. } => self.get_pointer_alignment(),
            TypeKind::Enum { underlying } => {
                self.calculate_type_alignment(&underlying.kind, underlying.size)
            }
            TypeKind::Unknown => 1,
        }
    }

    /// Get natural alignment for a given size
    fn get_natural_alignment(&self, size: usize) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 | TargetArchitecture::AArch64 => {
                match size {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    9..=16 => 16,
                    _ => 16, // Max alignment on most 64-bit systems
                }
            }
            TargetArchitecture::X86 => {
                match size {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8, // Max alignment on 32-bit systems
                }
            }
            TargetArchitecture::RISCV64 => {
                match size {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8,
                }
            }
            TargetArchitecture::WASM32 => {
                match size {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8,
                }
            }
        }
    }

    /// Get pointer alignment for target architecture
    fn get_pointer_alignment(&self) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 | TargetArchitecture::AArch64 | TargetArchitecture::RISCV64 => 8,
            TargetArchitecture::X86 | TargetArchitecture::WASM32 => 4,
        }
    }

    /// Get maximum size for return values passed in registers
    fn get_max_register_return_size(&self) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 => 16, // Two 64-bit registers
            TargetArchitecture::AArch64 => 16, // Two 64-bit registers
            TargetArchitecture::X86 => 8,     // Two 32-bit registers
            TargetArchitecture::RISCV64 => 16, // Two 64-bit registers
            TargetArchitecture::WASM32 => 4,   // Single 32-bit value
        }
    }

    /// Get maximum number of parameters passed in registers
    fn get_max_register_parameters(&self) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 => 6, // System V ABI: RDI, RSI, RDX, RCX, R8, R9
            TargetArchitecture::AArch64 => 8, // AAPCS64: X0-X7
            TargetArchitecture::X86 => 0,     // All parameters on stack for most conventions
            TargetArchitecture::RISCV64 => 8, // a0-a7
            TargetArchitecture::WASM32 => 0,  // Stack-based
        }
    }

    /// Check if type is an aggregate (struct, union, array)
    fn is_aggregate_type(&self, type_kind: &TypeKind) -> bool {
        matches!(type_kind, 
            TypeKind::Structure | 
            TypeKind::Union | 
            TypeKind::Array { .. }
        )
    }

    /// Check ABI-specific requirements and generate warnings
    fn check_abi_requirements(&self, signature: &FunctionSignature, warnings: &mut Vec<AbiWarning>) {
        // Check for variadic functions
        if signature.is_variadic {
            match signature.calling_convention {
                CallingConvention::C | CallingConvention::SystemV => {
                    // These support variadic functions
                }
                _ => {
                    warnings.push(AbiWarning {
                        message: format!(
                            "Variadic function with calling convention {:?} may not be portable",
                            signature.calling_convention
                        ),
                        location: Some(signature.name.clone()),
                    });
                }
            }
        }

        // Check for Rust ABI instability
        if matches!(signature.calling_convention, CallingConvention::Rust | CallingConvention::RustCall) {
            warnings.push(AbiWarning {
                message: "Rust ABI is unstable and may change between compiler versions".to_string(),
                location: Some(signature.name.clone()),
            });
        }

        // Check function name mangling
        if signature.mangled_name.is_some() {
            warnings.push(AbiWarning {
                message: "Function uses name mangling which may affect interoperability".to_string(),
                location: Some(signature.name.clone()),
            });
        }
    }

    /// Validate type consistency across the module
    fn validate_type_consistency(
        &self,
        signature: &FunctionSignature,
        debug_info: &DebugInfo,
    ) -> ForgeResult<Option<AbiViolation>> {
        // Check if return type exists in type definitions
        let return_type_name = &signature.return_type.name;
        if !return_type_name.is_empty() && return_type_name != "void" {
            let type_found = debug_info.types.iter()
                .any(|t| t.name == *return_type_name);
            
            if !type_found {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::FunctionSignatureMismatch,
                    description: format!(
                        "Return type '{}' not found in module type definitions",
                        return_type_name
                    ),
                    location: Some(signature.name.clone()),
                    severity: ViolationSeverity::Warning,
                }));
            }
        }

        // Check parameter types
        for param in &signature.parameters {
            let param_type_name = &param.type_info.name;
            if !param_type_name.is_empty() && param_type_name != "void" {
                let type_found = debug_info.types.iter()
                    .any(|t| t.name == *param_type_name);
                
                if !type_found {
                    return Ok(Some(AbiViolation {
                        violation_type: ViolationType::FunctionSignatureMismatch,
                        description: format!(
                            "Parameter type '{}' not found in module type definitions",
                            param_type_name
                        ),
                        location: Some(format!("{}::{}", 
                            signature.name,
                            param.name.as_deref().unwrap_or("unnamed")
                        )),
                        severity: ViolationSeverity::Warning,
                    }));
                }
            }
        }

        Ok(None)
    }
}