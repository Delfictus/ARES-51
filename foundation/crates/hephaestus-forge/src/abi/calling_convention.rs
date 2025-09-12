//! Calling convention verification for different platforms

use super::{
    DebugInfo, FunctionSignature, CallingConvention, TargetArchitecture, 
    TypeKind, Parameter
};
use crate::types::{ForgeResult, ForgeError};

/// Validates calling conventions for target platform
pub struct CallingConventionValidator {
    target_arch: TargetArchitecture,
}

impl CallingConventionValidator {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self { target_arch }
    }

    /// Verify calling conventions for all functions
    pub async fn verify_calling_conventions(&self, debug_info: &DebugInfo) -> ForgeResult<()> {
        for function in &debug_info.functions {
            self.verify_function_calling_convention(function)?;
        }
        Ok(())
    }

    /// Verify a single function's calling convention
    fn verify_function_calling_convention(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        match (&self.target_arch, &signature.calling_convention) {
            // X86-64 validations
            (TargetArchitecture::X86_64, CallingConvention::SystemV) => {
                self.verify_system_v_x64(signature)
            }
            (TargetArchitecture::X86_64, CallingConvention::Win64) => {
                self.verify_win64(signature)
            }
            
            // AArch64 validations  
            (TargetArchitecture::AArch64, CallingConvention::AAPCS64) => {
                self.verify_aapcs64(signature)
            }
            
            // X86 validations
            (TargetArchitecture::X86, CallingConvention::CDecl) => {
                self.verify_cdecl(signature)
            }
            (TargetArchitecture::X86, CallingConvention::StdCall) => {
                self.verify_stdcall(signature)
            }
            (TargetArchitecture::X86, CallingConvention::FastCall) => {
                self.verify_fastcall(signature)
            }
            
            // RISC-V validations
            (TargetArchitecture::RISCV64, CallingConvention::C) => {
                self.verify_riscv_c(signature)
            }
            
            // WebAssembly validations
            (TargetArchitecture::WASM32, CallingConvention::C) => {
                self.verify_wasm_c(signature)
            }
            
            // Generic C calling convention
            (_, CallingConvention::C) => {
                self.verify_generic_c(signature)
            }
            
            // Rust calling conventions (unstable)
            (_, CallingConvention::Rust | CallingConvention::RustCall) => {
                // Rust ABI is implementation-defined and unstable
                Ok(())
            }
            
            _ => {
                Err(ForgeError::ValidationError(format!(
                    "Unsupported calling convention {:?} for architecture {:?}",
                    signature.calling_convention,
                    self.target_arch
                )))
            }
        }
    }

    /// Verify System V ABI for x86-64
    fn verify_system_v_x64(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // System V x86-64 ABI specifics:
        // - Integer args in: RDI, RSI, RDX, RCX, R8, R9
        // - FP args in: XMM0-XMM7  
        // - Return in: RAX (int), XMM0 (fp), or memory for large types
        
        let mut integer_regs_used = 0;
        let mut fp_regs_used = 0;
        
        for param in &signature.parameters {
            match &param.type_info.kind {
                TypeKind::Integer { .. } | TypeKind::Pointer { .. } | TypeKind::Enum { .. } => {
                    if param.is_register {
                        integer_regs_used += 1;
                        if integer_regs_used > 6 {
                            return Err(ForgeError::ValidationError(
                                "Too many integer parameters for System V x64 ABI".into()
                            ));
                        }
                    }
                }
                TypeKind::Float { .. } => {
                    if param.is_register {
                        fp_regs_used += 1;
                        if fp_regs_used > 8 {
                            return Err(ForgeError::ValidationError(
                                "Too many floating-point parameters for System V x64 ABI".into()
                            ));
                        }
                    }
                }
                TypeKind::Structure | TypeKind::Union => {
                    // Structures > 16 bytes must be passed by reference
                    if param.type_info.size > 16 && param.is_register {
                        return Err(ForgeError::ValidationError(
                            "Large aggregate types must be passed by reference in System V x64 ABI".into()
                        ));
                    }
                }
                _ => {}
            }
        }

        // Verify return type
        self.verify_system_v_return_type(&signature.return_type)?;

        Ok(())
    }

    /// Verify Windows x64 ABI
    fn verify_win64(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // Windows x64 ABI specifics:
        // - First 4 args in: RCX, RDX, R8, R9
        // - FP args use same registers as integers
        // - All args > 8 bytes passed by reference
        
        let mut regs_used = 0;
        
        for param in &signature.parameters {
            if param.is_register {
                regs_used += 1;
                if regs_used > 4 {
                    return Err(ForgeError::ValidationError(
                        "Too many register parameters for Win64 ABI".into()
                    ));
                }
            }
            
            // Windows x64 passes all large types by reference
            if param.type_info.size > 8 && param.is_register {
                return Err(ForgeError::ValidationError(
                    "Types larger than 8 bytes must be passed by reference in Win64 ABI".into()
                ));
            }
        }

        Ok(())
    }

    /// Verify AAPCS64 (ARM64) ABI
    fn verify_aapcs64(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // AAPCS64 specifics:
        // - General purpose args in: X0-X7
        // - FP/SIMD args in: V0-V7
        // - Large types passed by reference
        
        let mut gp_regs_used = 0;
        let mut fp_regs_used = 0;
        
        for param in &signature.parameters {
            match &param.type_info.kind {
                TypeKind::Integer { .. } | TypeKind::Pointer { .. } => {
                    if param.is_register {
                        gp_regs_used += 1;
                        if gp_regs_used > 8 {
                            return Err(ForgeError::ValidationError(
                                "Too many general purpose parameters for AAPCS64".into()
                            ));
                        }
                    }
                }
                TypeKind::Float { .. } => {
                    if param.is_register {
                        fp_regs_used += 1;
                        if fp_regs_used > 8 {
                            return Err(ForgeError::ValidationError(
                                "Too many floating-point parameters for AAPCS64".into()
                            ));
                        }
                    }
                }
                TypeKind::Structure | TypeKind::Union => {
                    // HFA (Homogeneous Float Aggregates) have special rules
                    if self.is_hfa(&param.type_info) {
                        // HFA can use FP registers
                        if param.is_register {
                            fp_regs_used += self.count_hfa_elements(&param.type_info);
                            if fp_regs_used > 8 {
                                return Err(ForgeError::ValidationError(
                                    "HFA exhausts floating-point registers in AAPCS64".into()
                                ));
                            }
                        }
                    } else if param.type_info.size > 16 && param.is_register {
                        return Err(ForgeError::ValidationError(
                            "Large composite types must be passed by reference in AAPCS64".into()
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Verify cdecl calling convention (x86)
    fn verify_cdecl(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // cdecl: all parameters on stack, caller cleans up
        // Only return values can use registers
        
        for param in &signature.parameters {
            if param.is_register {
                return Err(ForgeError::ValidationError(
                    "cdecl passes all parameters on stack".into()
                ));
            }
        }

        // Variadic functions must use cdecl
        if signature.is_variadic {
            // This is correct for cdecl
        }

        Ok(())
    }

    /// Verify stdcall calling convention (x86)
    fn verify_stdcall(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // stdcall: all parameters on stack, callee cleans up
        // Cannot be variadic
        
        if signature.is_variadic {
            return Err(ForgeError::ValidationError(
                "stdcall cannot be used with variadic functions".into()
            ));
        }

        for param in &signature.parameters {
            if param.is_register {
                return Err(ForgeError::ValidationError(
                    "stdcall passes all parameters on stack".into()
                ));
            }
        }

        Ok(())
    }

    /// Verify fastcall calling convention (x86)
    fn verify_fastcall(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // fastcall: first two integer args in ECX, EDX
        
        let mut reg_params = 0;
        for param in &signature.parameters {
            if param.is_register {
                reg_params += 1;
                if reg_params > 2 {
                    return Err(ForgeError::ValidationError(
                        "fastcall can only pass first 2 parameters in registers".into()
                    ));
                }
                
                // Only integer-like types can go in registers
                match &param.type_info.kind {
                    TypeKind::Integer { .. } | TypeKind::Pointer { .. } => {}
                    _ => {
                        return Err(ForgeError::ValidationError(
                            "fastcall register parameters must be integer types".into()
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Verify RISC-V C calling convention
    fn verify_riscv_c(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // RISC-V C ABI:
        // - Integer args in: a0-a7 (x10-x17)
        // - FP args in: fa0-fa7 (f10-f17)
        
        let mut int_regs_used = 0;
        let mut fp_regs_used = 0;
        
        for param in &signature.parameters {
            match &param.type_info.kind {
                TypeKind::Integer { .. } | TypeKind::Pointer { .. } => {
                    if param.is_register {
                        int_regs_used += 1;
                        if int_regs_used > 8 {
                            return Err(ForgeError::ValidationError(
                                "Too many integer parameters for RISC-V C ABI".into()
                            ));
                        }
                    }
                }
                TypeKind::Float { .. } => {
                    if param.is_register {
                        fp_regs_used += 1;
                        if fp_regs_used > 8 {
                            return Err(ForgeError::ValidationError(
                                "Too many floating-point parameters for RISC-V C ABI".into()
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Verify WebAssembly C calling convention
    fn verify_wasm_c(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // WebAssembly is stack-based, no register constraints
        // But has type limitations
        
        for param in &signature.parameters {
            match &param.type_info.kind {
                TypeKind::Integer { width, .. } => {
                    if *width != 32 && *width != 64 {
                        return Err(ForgeError::ValidationError(
                            "WebAssembly only supports i32 and i64 integer types".into()
                        ));
                    }
                }
                TypeKind::Float { width } => {
                    if *width != 32 && *width != 64 {
                        return Err(ForgeError::ValidationError(
                            "WebAssembly only supports f32 and f64 float types".into()
                        ));
                    }
                }
                TypeKind::Pointer { .. } => {
                    // Pointers are represented as i32 or i64
                }
                _ => {
                    return Err(ForgeError::ValidationError(
                        "WebAssembly has limited type support".into()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Verify generic C calling convention
    fn verify_generic_c(&self, signature: &FunctionSignature) -> ForgeResult<()> {
        // Generic validation that applies to most C ABIs
        
        // Check for common issues
        if signature.is_variadic {
            // Variadic functions have special requirements
            if signature.parameters.is_empty() {
                return Err(ForgeError::ValidationError(
                    "Variadic function must have at least one named parameter".into()
                ));
            }
        }

        Ok(())
    }

    /// Verify System V return type constraints
    fn verify_system_v_return_type(&self, return_type: &super::TypeInfo) -> ForgeResult<()> {
        match &return_type.kind {
            TypeKind::Structure | TypeKind::Union => {
                if return_type.size > 16 {
                    // Large return types are returned via hidden pointer parameter
                    // This should be reflected in the calling convention
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Check if a type is a Homogeneous Float Aggregate (ARM64)
    fn is_hfa(&self, type_info: &super::TypeInfo) -> bool {
        match &type_info.kind {
            TypeKind::Float { .. } => true,
            TypeKind::Structure | TypeKind::Union => {
                if type_info.fields.is_empty() {
                    return false;
                }
                
                // All fields must be the same float type
                let first_field_kind = &type_info.fields[0].type_info.kind;
                if let TypeKind::Float { width } = first_field_kind {
                    type_info.fields.iter().all(|f| {
                        matches!(&f.type_info.kind, TypeKind::Float { width: w } if w == width)
                    })
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Count elements in HFA
    fn count_hfa_elements(&self, type_info: &super::TypeInfo) -> usize {
        match &type_info.kind {
            TypeKind::Float { .. } => 1,
            TypeKind::Structure | TypeKind::Union => type_info.fields.len(),
            _ => 0,
        }
    }
}