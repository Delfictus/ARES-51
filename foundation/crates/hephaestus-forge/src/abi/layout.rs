//! Data structure layout validation

use super::{
    TypeInfo, DebugInfo, AbiViolation, AbiWarning, ViolationType, ViolationSeverity,
    TargetArchitecture, TypeKind, FieldInfo
};
use crate::types::{ForgeResult, ForgeError};

/// Result of layout validation
#[derive(Debug)]
pub struct LayoutValidationResult {
    pub type_name: String,
    pub is_valid: bool,
    pub violation: Option<AbiViolation>,
    pub warnings: Vec<AbiWarning>,
}

/// Validates data structure layouts and memory organization
pub struct DataLayoutValidator {
    target_arch: TargetArchitecture,
}

impl DataLayoutValidator {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self { target_arch }
    }

    /// Validate all data layouts in debug information
    pub async fn validate_layouts(
        &self,
        debug_info: &DebugInfo,
    ) -> ForgeResult<Vec<LayoutValidationResult>> {
        let mut results = Vec::new();

        for type_info in &debug_info.types {
            let result = self.validate_single_layout(type_info).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Validate a single data structure layout
    async fn validate_single_layout(
        &self,
        type_info: &TypeInfo,
    ) -> ForgeResult<LayoutValidationResult> {
        let mut warnings = Vec::new();
        let mut violation = None;

        // 1. Validate overall size and alignment
        if let Some(size_violation) = self.validate_size_alignment(type_info)? {
            violation = Some(size_violation);
        }

        // 2. Validate field layouts for structures and unions
        if matches!(type_info.kind, TypeKind::Structure | TypeKind::Union) {
            if let Some(field_violation) = self.validate_field_layout(type_info)? {
                if violation.is_none() {
                    violation = Some(field_violation);
                }
            }
        }

        // 3. Check for padding and alignment issues
        self.check_padding_issues(type_info, &mut warnings);

        let is_valid = violation.is_none();

        Ok(LayoutValidationResult {
            type_name: type_info.name.clone(),
            is_valid,
            violation,
            warnings,
        })
    }

    /// Validate size and alignment requirements
    fn validate_size_alignment(&self, type_info: &TypeInfo) -> ForgeResult<Option<AbiViolation>> {
        // Check alignment is power of 2
        if type_info.alignment > 0 && (type_info.alignment & (type_info.alignment - 1)) != 0 {
            return Ok(Some(AbiViolation {
                violation_type: ViolationType::AlignmentViolation,
                description: format!(
                    "Type '{}' alignment {} is not a power of 2",
                    type_info.name,
                    type_info.alignment
                ),
                location: None,
                severity: ViolationSeverity::Error,
            }));
        }

        // Check maximum alignment limits
        let max_alignment = self.get_max_alignment();
        if type_info.alignment > max_alignment {
            return Ok(Some(AbiViolation {
                violation_type: ViolationType::AlignmentViolation,
                description: format!(
                    "Type '{}' alignment {} exceeds maximum alignment {} for architecture {:?}",
                    type_info.name,
                    type_info.alignment,
                    max_alignment,
                    self.target_arch
                ),
                location: None,
                severity: ViolationSeverity::Error,
            }));
        }

        // Validate size is multiple of alignment for structures
        if matches!(type_info.kind, TypeKind::Structure) && type_info.alignment > 0 {
            if type_info.size % type_info.alignment != 0 {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::DataLayoutMismatch,
                    description: format!(
                        "Structure '{}' size {} is not a multiple of alignment {}",
                        type_info.name,
                        type_info.size,
                        type_info.alignment
                    ),
                    location: None,
                    severity: ViolationSeverity::Error,
                }));
            }
        }

        Ok(None)
    }

    /// Validate field layout within structures and unions
    fn validate_field_layout(&self, type_info: &TypeInfo) -> ForgeResult<Option<AbiViolation>> {
        if type_info.fields.is_empty() {
            return Ok(None);
        }

        let mut current_offset = 0;
        let mut max_field_alignment = 1;

        for field in &type_info.fields {
            // Calculate expected field alignment
            let field_alignment = self.calculate_field_alignment(&field.type_info);
            max_field_alignment = max_field_alignment.max(field_alignment);

            // Check field alignment
            if field.offset % field_alignment != 0 {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::AlignmentViolation,
                    description: format!(
                        "Field '{}' in type '{}' at offset {} violates alignment requirement {}",
                        field.name,
                        type_info.name,
                        field.offset,
                        field_alignment
                    ),
                    location: Some(format!("{}::{}", type_info.name, field.name)),
                    severity: ViolationSeverity::Error,
                }));
            }

            // For structures, check field ordering
            if matches!(type_info.kind, TypeKind::Structure) {
                if field.offset < current_offset {
                    return Ok(Some(AbiViolation {
                        violation_type: ViolationType::DataLayoutMismatch,
                        description: format!(
                            "Field '{}' in structure '{}' has overlapping offset {}",
                            field.name,
                            type_info.name,
                            field.offset
                        ),
                        location: Some(format!("{}::{}", type_info.name, field.name)),
                        severity: ViolationSeverity::Error,
                    }));
                }
                current_offset = field.offset + field.type_info.size;
            }

            // For unions, all fields should start at offset 0
            if type_info.is_union && field.offset != 0 {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::DataLayoutMismatch,
                    description: format!(
                        "Union field '{}' in '{}' has non-zero offset {}",
                        field.name,
                        type_info.name,
                        field.offset
                    ),
                    location: Some(format!("{}::{}", type_info.name, field.name)),
                    severity: ViolationSeverity::Error,
                }));
            }

            // Validate bit fields
            if let (Some(bit_offset), Some(bit_size)) = (field.bit_offset, field.bit_size) {
                if let Err(e) = self.validate_bit_field(field, bit_offset, bit_size) {
                    return Ok(Some(AbiViolation {
                        violation_type: ViolationType::DataLayoutMismatch,
                        description: format!(
                            "Invalid bit field '{}' in '{}': {}",
                            field.name,
                            type_info.name,
                            e
                        ),
                        location: Some(format!("{}::{}", type_info.name, field.name)),
                        severity: ViolationSeverity::Error,
                    }));
                }
            }
        }

        // Validate overall structure alignment matches maximum field alignment
        if matches!(type_info.kind, TypeKind::Structure) {
            if type_info.alignment != max_field_alignment {
                return Ok(Some(AbiViolation {
                    violation_type: ViolationType::AlignmentViolation,
                    description: format!(
                        "Structure '{}' alignment {} does not match maximum field alignment {}",
                        type_info.name,
                        type_info.alignment,
                        max_field_alignment
                    ),
                    location: None,
                    severity: ViolationSeverity::Warning,
                }));
            }
        }

        Ok(None)
    }

    /// Calculate expected alignment for a field
    fn calculate_field_alignment(&self, type_info: &TypeInfo) -> usize {
        match &type_info.kind {
            TypeKind::Integer { width, .. } => {
                let byte_width = (*width as usize + 7) / 8;
                self.get_integer_alignment(byte_width)
            }
            TypeKind::Float { width } => {
                let byte_width = (*width as usize + 7) / 8;
                self.get_float_alignment(byte_width)
            }
            TypeKind::Pointer { .. } => self.get_pointer_alignment(),
            TypeKind::Array { element, .. } => {
                self.calculate_field_alignment(element)
            }
            TypeKind::Structure | TypeKind::Union => {
                // Use the type's specified alignment
                type_info.alignment
            }
            _ => 1,
        }
    }

    /// Get integer alignment for target architecture
    fn get_integer_alignment(&self, byte_width: usize) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 | TargetArchitecture::AArch64 => {
                match byte_width {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8,
                }
            }
            TargetArchitecture::X86 => {
                match byte_width {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 4, // 32-bit alignment
                    _ => 4,
                }
            }
            TargetArchitecture::RISCV64 => {
                match byte_width {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8,
                }
            }
            TargetArchitecture::WASM32 => {
                match byte_width {
                    1 => 1,
                    2 => 2,
                    3..=4 => 4,
                    5..=8 => 8,
                    _ => 8,
                }
            }
        }
    }

    /// Get float alignment for target architecture
    fn get_float_alignment(&self, byte_width: usize) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 | TargetArchitecture::AArch64 => {
                match byte_width {
                    4 => 4,  // float
                    8 => 8,  // double
                    10 | 16 => 16, // long double
                    _ => byte_width.min(16),
                }
            }
            TargetArchitecture::X86 => {
                match byte_width {
                    4 => 4,  // float
                    8 => 4,  // double (32-bit alignment)
                    10 | 12 => 4, // long double
                    _ => 4,
                }
            }
            TargetArchitecture::RISCV64 => {
                match byte_width {
                    4 => 4,  // float
                    8 => 8,  // double
                    16 => 16, // quad
                    _ => byte_width.min(16),
                }
            }
            TargetArchitecture::WASM32 => {
                match byte_width {
                    4 => 4,  // f32
                    8 => 8,  // f64
                    _ => byte_width,
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

    /// Get maximum alignment for target architecture
    fn get_max_alignment(&self) -> usize {
        match self.target_arch {
            TargetArchitecture::X86_64 | TargetArchitecture::AArch64 => 16,
            TargetArchitecture::X86 => 8,
            TargetArchitecture::RISCV64 => 16,
            TargetArchitecture::WASM32 => 8,
        }
    }

    /// Validate bit field constraints
    fn validate_bit_field(&self, field: &FieldInfo, bit_offset: u8, bit_size: u8) -> Result<(), String> {
        // Bit field size cannot exceed underlying type size
        let max_bits = field.type_info.size * 8;
        if bit_size as usize > max_bits {
            return Err(format!(
                "Bit field size {} exceeds underlying type size {} bits",
                bit_size, max_bits
            ));
        }

        // Bit offset + bit size cannot exceed underlying type
        if (bit_offset + bit_size) as usize > max_bits {
            return Err(format!(
                "Bit field extends beyond underlying type boundary ({} + {} > {})",
                bit_offset, bit_size, max_bits
            ));
        }

        // Zero-width bit fields must be unnamed
        if bit_size == 0 && !field.name.is_empty() {
            return Err("Zero-width bit field cannot be named".to_string());
        }

        Ok(())
    }

    /// Check for potential padding and alignment issues
    fn check_padding_issues(&self, type_info: &TypeInfo, warnings: &mut Vec<AbiWarning>) {
        if !matches!(type_info.kind, TypeKind::Structure) || type_info.fields.is_empty() {
            return;
        }

        let mut total_field_size = 0;
        let mut has_padding = false;

        // Calculate total size of fields and detect padding
        let mut sorted_fields = type_info.fields.clone();
        sorted_fields.sort_by_key(|f| f.offset);

        for (i, field) in sorted_fields.iter().enumerate() {
            // Check for padding before this field
            if field.offset > total_field_size {
                has_padding = true;
            }

            total_field_size = field.offset + field.type_info.size;

            // Check for excessive gaps between fields
            if i > 0 {
                let prev_field = &sorted_fields[i - 1];
                let prev_end = prev_field.offset + prev_field.type_info.size;
                let gap = field.offset - prev_end;
                
                if gap > field.type_info.alignment {
                    warnings.push(AbiWarning {
                        message: format!(
                            "Large gap ({} bytes) between fields '{}' and '{}', consider reordering",
                            gap, prev_field.name, field.name
                        ),
                        location: Some(format!("{}::{}", type_info.name, field.name)),
                    });
                }
            }
        }

        // Check for padding at end of structure
        if type_info.size > total_field_size {
            has_padding = true;
        }

        // Warn about structures with significant padding
        let padding_ratio = if type_info.size > 0 {
            (type_info.size - (type_info.size - total_field_size + type_info.size - total_field_size)) as f64 / type_info.size as f64
        } else {
            0.0
        };

        if has_padding && padding_ratio > 0.25 {
            warnings.push(AbiWarning {
                message: format!(
                    "Structure '{}' has significant padding ({:.1}% wasted space), consider field reordering",
                    type_info.name, padding_ratio * 100.0
                ),
                location: Some(type_info.name.clone()),
            });
        }

        // Suggest field reordering for better packing
        if type_info.fields.len() > 1 {
            let optimized_size = self.calculate_optimized_size(&type_info.fields);
            if optimized_size < type_info.size {
                warnings.push(AbiWarning {
                    message: format!(
                        "Structure '{}' could be optimized from {} to {} bytes by reordering fields",
                        type_info.name, type_info.size, optimized_size
                    ),
                    location: Some(type_info.name.clone()),
                });
            }
        }
    }

    /// Calculate optimized structure size with field reordering
    fn calculate_optimized_size(&self, fields: &[FieldInfo]) -> usize {
        let mut sorted_fields = fields.to_vec();
        
        // Sort by alignment (descending) then by size (descending) for optimal packing
        sorted_fields.sort_by(|a, b| {
            let a_align = self.calculate_field_alignment(&a.type_info);
            let b_align = self.calculate_field_alignment(&b.type_info);
            
            b_align.cmp(&a_align)
                .then_with(|| b.type_info.size.cmp(&a.type_info.size))
        });

        let mut offset = 0;
        let mut max_alignment = 1;

        for field in &sorted_fields {
            let field_alignment = self.calculate_field_alignment(&field.type_info);
            max_alignment = max_alignment.max(field_alignment);
            
            // Align offset to field alignment
            offset = (offset + field_alignment - 1) & !(field_alignment - 1);
            offset += field.type_info.size;
        }

        // Align total size to structure alignment
        (offset + max_alignment - 1) & !(max_alignment - 1)
    }
}