//! Cross-module ABI compatibility checking

use super::{
    TargetArchitecture, AbiViolation, ViolationType, ViolationSeverity,
    FunctionSignature, TypeInfo, TypeKind
};
use crate::types::{VersionedModule, ForgeResult, ForgeError};
use std::collections::HashMap;

/// Checks ABI compatibility between module versions
pub struct CompatibilityChecker {
    target_arch: TargetArchitecture,
}

impl CompatibilityChecker {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self { target_arch }
    }

    /// Check compatibility between two module versions
    pub async fn check_compatibility(
        &self,
        new_module: &VersionedModule,
        old_module: &VersionedModule,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Extract ABI information from both modules
        let new_abi = self.extract_abi_info(&new_module.code).await?;
        let old_abi = self.extract_abi_info(&old_module.code).await?;

        // Check function compatibility
        violations.extend(self.check_function_compatibility(&new_abi.functions, &old_abi.functions)?);

        // Check type compatibility
        violations.extend(self.check_type_compatibility(&new_abi.types, &old_abi.types)?);

        // Check symbol compatibility
        violations.extend(self.check_symbol_compatibility(&new_abi.exported_symbols, &old_abi.exported_symbols)?);

        // Check version-specific compatibility
        violations.extend(self.check_version_compatibility(new_module, old_module)?);

        Ok(violations)
    }

    async fn extract_abi_info(&self, binary_data: &[u8]) -> ForgeResult<AbiInfo> {
        #[cfg(feature = "abi-verification")]
        {
            self.extract_abi_with_debug_info(binary_data).await
        }
        #[cfg(not(feature = "abi-verification"))]
        {
            Ok(AbiInfo::empty())
        }
    }

    #[cfg(feature = "abi-verification")]
    async fn extract_abi_with_debug_info(&self, binary_data: &[u8]) -> ForgeResult<AbiInfo> {
        use object::{Object, ObjectSymbol};
        
        // Parse object file
        let file = object::File::parse(binary_data)
            .map_err(|e| ForgeError::ValidationError(format!("Failed to parse binary: {}", e)))?;

        let mut exported_symbols = HashMap::new();
        
        // Extract exported symbols
        for symbol in file.symbols() {
            if !symbol.is_undefined() && symbol.is_global() {
                if let Ok(name) = symbol.name() {
                    let symbol_info = ExportedSymbol {
                        name: name.to_string(),
                        address: symbol.address(),
                        size: symbol.size(),
                        symbol_type: match symbol.kind() {
                            object::SymbolKind::Text => ExportedSymbolType::Function,
                            object::SymbolKind::Data => ExportedSymbolType::Data,
                            _ => ExportedSymbolType::Other,
                        },
                    };
                    exported_symbols.insert(name.to_string(), symbol_info);
                }
            }
        }

        // For now, return simplified ABI info
        // In a full implementation, this would parse DWARF debug info
        Ok(AbiInfo {
            functions: Vec::new(), // Would be populated from debug info
            types: Vec::new(),     // Would be populated from debug info
            exported_symbols,
        })
    }

    fn check_function_compatibility(
        &self,
        new_functions: &[FunctionSignature],
        old_functions: &[FunctionSignature],
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();
        let old_func_map: HashMap<String, &FunctionSignature> = 
            old_functions.iter().map(|f| (f.name.clone(), f)).collect();

        for new_func in new_functions {
            if let Some(old_func) = old_func_map.get(&new_func.name) {
                // Check if function signature is compatible
                violations.extend(self.check_function_signature_compatibility(new_func, old_func)?);
            }
            // New functions are generally OK (additions don't break ABI)
        }

        // Check for removed functions
        let new_func_names: std::collections::HashSet<_> = 
            new_functions.iter().map(|f| &f.name).collect();
        
        for old_func in old_functions {
            if !new_func_names.contains(&old_func.name) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::FunctionSignatureMismatch,
                    description: format!("Function '{}' was removed from public API", old_func.name),
                    location: Some(old_func.name.clone()),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        Ok(violations)
    }

    fn check_function_signature_compatibility(
        &self,
        new_func: &FunctionSignature,
        old_func: &FunctionSignature,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Check calling convention compatibility
        if new_func.calling_convention != old_func.calling_convention {
            violations.push(AbiViolation {
                violation_type: ViolationType::CallingConventionMismatch,
                description: format!(
                    "Function '{}' calling convention changed from {:?} to {:?}",
                    new_func.name, old_func.calling_convention, new_func.calling_convention
                ),
                location: Some(new_func.name.clone()),
                severity: ViolationSeverity::Error,
            });
        }

        // Check return type compatibility
        if let Some(violation) = self.check_type_change_compatibility(&new_func.return_type, &old_func.return_type, &format!("{}::return", new_func.name))? {
            violations.push(violation);
        }

        // Check parameter compatibility
        if new_func.parameters.len() != old_func.parameters.len() {
            violations.push(AbiViolation {
                violation_type: ViolationType::FunctionSignatureMismatch,
                description: format!(
                    "Function '{}' parameter count changed from {} to {}",
                    new_func.name, old_func.parameters.len(), new_func.parameters.len()
                ),
                location: Some(new_func.name.clone()),
                severity: ViolationSeverity::Error,
            });
        } else {
            for (i, (new_param, old_param)) in new_func.parameters.iter().zip(old_func.parameters.iter()).enumerate() {
                if let Some(violation) = self.check_type_change_compatibility(
                    &new_param.type_info, 
                    &old_param.type_info, 
                    &format!("{}::param{}", new_func.name, i)
                )? {
                    violations.push(violation);
                }
            }
        }

        // Check variadic compatibility
        if new_func.is_variadic != old_func.is_variadic {
            violations.push(AbiViolation {
                violation_type: ViolationType::FunctionSignatureMismatch,
                description: format!(
                    "Function '{}' variadic property changed",
                    new_func.name
                ),
                location: Some(new_func.name.clone()),
                severity: ViolationSeverity::Error,
            });
        }

        Ok(violations)
    }

    fn check_type_compatibility(
        &self,
        new_types: &[TypeInfo],
        old_types: &[TypeInfo],
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();
        let old_type_map: HashMap<String, &TypeInfo> = 
            old_types.iter().map(|t| (t.name.clone(), t)).collect();

        for new_type in new_types {
            if let Some(old_type) = old_type_map.get(&new_type.name) {
                // Check type layout compatibility
                violations.extend(self.check_type_layout_compatibility(new_type, old_type)?);
            }
        }

        // Check for removed types (might be OK if not publicly exposed)
        let new_type_names: std::collections::HashSet<_> = 
            new_types.iter().map(|t| &t.name).collect();
        
        for old_type in old_types {
            if !new_type_names.contains(&old_type.name) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::TypeSizeMismatch,
                    description: format!("Type '{}' was removed", old_type.name),
                    location: Some(old_type.name.clone()),
                    severity: ViolationSeverity::Warning, // Might be OK if internal
                });
            }
        }

        Ok(violations)
    }

    fn check_type_layout_compatibility(
        &self,
        new_type: &TypeInfo,
        old_type: &TypeInfo,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Size must not change for ABI compatibility
        if new_type.size != old_type.size {
            violations.push(AbiViolation {
                violation_type: ViolationType::TypeSizeMismatch,
                description: format!(
                    "Type '{}' size changed from {} to {} bytes",
                    new_type.name, old_type.size, new_type.size
                ),
                location: Some(new_type.name.clone()),
                severity: ViolationSeverity::Error,
            });
        }

        // Alignment must not change
        if new_type.alignment != old_type.alignment {
            violations.push(AbiViolation {
                violation_type: ViolationType::AlignmentViolation,
                description: format!(
                    "Type '{}' alignment changed from {} to {}",
                    new_type.name, old_type.alignment, new_type.alignment
                ),
                location: Some(new_type.name.clone()),
                severity: ViolationSeverity::Error,
            });
        }

        // For structures, check field layout
        if matches!(new_type.kind, TypeKind::Structure) {
            violations.extend(self.check_structure_field_compatibility(new_type, old_type)?);
        }

        Ok(violations)
    }

    fn check_structure_field_compatibility(
        &self,
        new_type: &TypeInfo,
        old_type: &TypeInfo,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        let old_field_map: HashMap<String, &super::FieldInfo> = 
            old_type.fields.iter().map(|f| (f.name.clone(), f)).collect();

        // Check existing fields haven't changed
        for new_field in &new_type.fields {
            if let Some(old_field) = old_field_map.get(&new_field.name) {
                // Field offset must not change
                if new_field.offset != old_field.offset {
                    violations.push(AbiViolation {
                        violation_type: ViolationType::DataLayoutMismatch,
                        description: format!(
                            "Field '{}' offset changed from {} to {} in type '{}'",
                            new_field.name, old_field.offset, new_field.offset, new_type.name
                        ),
                        location: Some(format!("{}::{}", new_type.name, new_field.name)),
                        severity: ViolationSeverity::Error,
                    });
                }

                // Field type must be compatible
                if let Some(violation) = self.check_type_change_compatibility(
                    &new_field.type_info, 
                    &old_field.type_info,
                    &format!("{}::{}", new_type.name, new_field.name)
                )? {
                    violations.push(violation);
                }
            }
        }

        // Check for removed fields
        let new_field_names: std::collections::HashSet<_> = 
            new_type.fields.iter().map(|f| &f.name).collect();
        
        for old_field in &old_type.fields {
            if !new_field_names.contains(&old_field.name) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::DataLayoutMismatch,
                    description: format!(
                        "Field '{}' was removed from type '{}'",
                        old_field.name, new_type.name
                    ),
                    location: Some(format!("{}::{}", new_type.name, old_field.name)),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        Ok(violations)
    }

    fn check_type_change_compatibility(
        &self,
        new_type: &TypeInfo,
        old_type: &TypeInfo,
        location: &str,
    ) -> ForgeResult<Option<AbiViolation>> {
        // Types must be identical for ABI compatibility
        if new_type.name != old_type.name || 
           new_type.size != old_type.size ||
           new_type.alignment != old_type.alignment {
            return Ok(Some(AbiViolation {
                violation_type: ViolationType::TypeSizeMismatch,
                description: format!(
                    "Type changed at '{}': '{}' ({} bytes, align {}) -> '{}' ({} bytes, align {})",
                    location, old_type.name, old_type.size, old_type.alignment,
                    new_type.name, new_type.size, new_type.alignment
                ),
                location: Some(location.to_string()),
                severity: ViolationSeverity::Error,
            }));
        }

        Ok(None)
    }

    fn check_symbol_compatibility(
        &self,
        new_symbols: &HashMap<String, ExportedSymbol>,
        old_symbols: &HashMap<String, ExportedSymbol>,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Check for removed symbols
        for (name, old_symbol) in old_symbols {
            if !new_symbols.contains_key(name) {
                violations.push(AbiViolation {
                    violation_type: ViolationType::UndefinedSymbol,
                    description: format!("Exported symbol '{}' was removed", name),
                    location: Some(name.clone()),
                    severity: ViolationSeverity::Error,
                });
            } else if let Some(new_symbol) = new_symbols.get(name) {
                // Check symbol compatibility
                if old_symbol.symbol_type != new_symbol.symbol_type {
                    violations.push(AbiViolation {
                        violation_type: ViolationType::FunctionSignatureMismatch,
                        description: format!(
                            "Symbol '{}' type changed from {:?} to {:?}",
                            name, old_symbol.symbol_type, new_symbol.symbol_type
                        ),
                        location: Some(name.clone()),
                        severity: ViolationSeverity::Error,
                    });
                }

                // Size changes might be OK for data, but should be flagged
                if old_symbol.size != new_symbol.size {
                    let severity = match new_symbol.symbol_type {
                        ExportedSymbolType::Function => ViolationSeverity::Warning,
                        ExportedSymbolType::Data => ViolationSeverity::Error,
                        ExportedSymbolType::Other => ViolationSeverity::Warning,
                    };

                    violations.push(AbiViolation {
                        violation_type: ViolationType::TypeSizeMismatch,
                        description: format!(
                            "Symbol '{}' size changed from {} to {} bytes",
                            name, old_symbol.size, new_symbol.size
                        ),
                        location: Some(name.clone()),
                        severity,
                    });
                }
            }
        }

        Ok(violations)
    }

    fn check_version_compatibility(
        &self,
        new_module: &VersionedModule,
        old_module: &VersionedModule,
    ) -> ForgeResult<Vec<AbiViolation>> {
        let mut violations = Vec::new();

        // Check version numbers for semantic versioning compliance
        if new_module.version <= old_module.version {
            violations.push(AbiViolation {
                violation_type: ViolationType::VersionIncompatibility,
                description: format!(
                    "New module version {} is not greater than old version {}",
                    new_module.version, old_module.version
                ),
                location: None,
                severity: ViolationSeverity::Warning,
            });
        }

        // Check metadata compatibility
        if new_module.metadata.complexity_score > old_module.metadata.complexity_score * 2.0 {
            violations.push(AbiViolation {
                violation_type: ViolationType::VersionIncompatibility,
                description: format!(
                    "Module complexity increased significantly: {:.2} -> {:.2}",
                    old_module.metadata.complexity_score,
                    new_module.metadata.complexity_score
                ),
                location: None,
                severity: ViolationSeverity::Warning,
            });
        }

        Ok(violations)
    }
}

/// ABI information extracted from a module
struct AbiInfo {
    functions: Vec<FunctionSignature>,
    types: Vec<TypeInfo>,
    exported_symbols: HashMap<String, ExportedSymbol>,
}

impl AbiInfo {
    fn empty() -> Self {
        Self {
            functions: Vec::new(),
            types: Vec::new(),
            exported_symbols: HashMap::new(),
        }
    }
}

/// Information about an exported symbol
#[derive(Debug, Clone)]
struct ExportedSymbol {
    name: String,
    address: u64,
    size: u64,
    symbol_type: ExportedSymbolType,
}

#[derive(Debug, Clone, PartialEq)]
enum ExportedSymbolType {
    Function,
    Data,
    Other,
}