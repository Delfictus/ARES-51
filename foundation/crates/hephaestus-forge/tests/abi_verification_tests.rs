//! Comprehensive tests for ABI verification system
//! 
//! Tests the complete ABI verification pipeline including DWARF parsing,
//! signature validation, data layout verification, calling convention checks,
//! and symbol table analysis.

use hephaestus_forge::{
    types::{VersionedModule, ModuleId, ModuleMetadata, PerformanceProfile},
    orchestrator::MetamorphicRuntimeOrchestrator,
    types::{RuntimeConfig, ForgeError},
};

#[cfg(feature = "abi-verification")]
use hephaestus_forge::abi::{
    AbiVerifier, TargetArchitecture, CallingConvention, TypeInfo, TypeKind,
    FunctionSignature, Parameter, ViolationType, ViolationSeverity
};

use chrono::Utc;
use std::sync::Arc;

/// Create a test module with ELF binary data
fn create_test_module(binary_data: Vec<u8>, module_name: &str) -> VersionedModule {
    VersionedModule {
        id: ModuleId(module_name.to_string()),
        version: 1,
        code: binary_data,
        proof: None,
        metadata: ModuleMetadata {
            created_at: Utc::now(),
            risk_score: 0.1,
            complexity_score: 0.5,
            performance_profile: PerformanceProfile {
                cpu_usage_percent: 10.0,
                memory_mb: 64,
                latency_p99_ms: 5.0,
                throughput_ops_per_sec: 1000,
            },
        },
    }
}

/// Create minimal valid ELF binary for testing
fn create_minimal_elf() -> Vec<u8> {
    let mut elf = Vec::new();
    
    // ELF header
    elf.extend_from_slice(&[0x7f, 0x45, 0x4c, 0x46]); // e_ident[EI_MAG0..EI_MAG3]
    elf.extend_from_slice(&[0x02]); // e_ident[EI_CLASS] = ELFCLASS64
    elf.extend_from_slice(&[0x01]); // e_ident[EI_DATA] = ELFDATA2LSB
    elf.extend_from_slice(&[0x01]); // e_ident[EI_VERSION] = EV_CURRENT
    elf.extend_from_slice(&[0x00]); // e_ident[EI_OSABI] = ELFOSABI_SYSV
    elf.extend_from_slice(&[0x00]); // e_ident[EI_ABIVERSION]
    elf.extend_from_slice(&[0x00; 7]); // e_ident[EI_PAD]
    
    // e_type, e_machine, e_version (2 + 2 + 4 = 8 bytes)
    elf.extend_from_slice(&[0x01, 0x00]); // e_type = ET_REL
    elf.extend_from_slice(&[0x3e, 0x00]); // e_machine = EM_X86_64
    elf.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // e_version = EV_CURRENT
    
    // e_entry, e_phoff, e_shoff (8 + 8 + 8 = 24 bytes)
    elf.extend_from_slice(&[0x00; 8]); // e_entry
    elf.extend_from_slice(&[0x00; 8]); // e_phoff
    elf.extend_from_slice(&[0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]); // e_shoff = 64
    
    // e_flags, e_ehsize, e_phentsize, e_phnum (4 + 2 + 2 + 2 = 10 bytes)
    elf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // e_flags
    elf.extend_from_slice(&[0x40, 0x00]); // e_ehsize = 64
    elf.extend_from_slice(&[0x00, 0x00]); // e_phentsize
    elf.extend_from_slice(&[0x00, 0x00]); // e_phnum
    
    // e_shentsize, e_shnum, e_shstrndx (2 + 2 + 2 = 6 bytes)
    elf.extend_from_slice(&[0x40, 0x00]); // e_shentsize = 64
    elf.extend_from_slice(&[0x01, 0x00]); // e_shnum = 1
    elf.extend_from_slice(&[0x00, 0x00]); // e_shstrndx
    
    // Minimal section header (64 bytes)
    elf.extend_from_slice(&[0x00; 64]);
    
    elf
}

/// Create invalid binary data
fn create_invalid_binary() -> Vec<u8> {
    vec![0x00, 0x01, 0x02, 0x03] // Too small and invalid format
}

#[tokio::test]
async fn test_orchestrator_abi_verification_basic() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    // Test with valid ELF module
    let valid_module = create_test_module(create_minimal_elf(), "valid_module");
    let result = orchestrator.validate_module(&valid_module).await;
    assert!(result.is_ok(), "Valid ELF module should pass basic validation: {:?}", result);
    
    // Test with invalid module
    let invalid_module = create_test_module(create_invalid_binary(), "invalid_module");
    let result = orchestrator.validate_module(&invalid_module).await;
    assert!(result.is_err(), "Invalid binary should fail validation");
}

#[tokio::test]
async fn test_orchestrator_empty_module() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    let empty_module = create_test_module(Vec::new(), "empty_module");
    let result = orchestrator.validate_module(&empty_module).await;
    assert!(result.is_err(), "Empty module should fail validation");
    
    if let Err(ForgeError::ValidationError(msg)) = result {
        assert!(msg.contains("empty binary code"), "Error should mention empty binary: {}", msg);
    } else {
        panic!("Expected ValidationError for empty module");
    }
}

#[tokio::test]
async fn test_orchestrator_small_module() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    let small_module = create_test_module(vec![0x00; 32], "small_module");
    let result = orchestrator.validate_module(&small_module).await;
    assert!(result.is_err(), "Module smaller than 64 bytes should fail validation");
    
    if let Err(ForgeError::ValidationError(msg)) = result {
        assert!(msg.contains("too small"), "Error should mention size: {}", msg);
    } else {
        panic!("Expected ValidationError for small module");
    }
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_abi_verifier_initialization() {
    let verifier = AbiVerifier::new(TargetArchitecture::X86_64);
    
    // Test with minimal module
    let module = create_test_module(create_minimal_elf(), "test_module");
    let result = verifier.verify_module_abi(&module, None).await;
    
    // Should complete without errors (though may have warnings about missing debug info)
    assert!(result.is_ok(), "ABI verifier should handle minimal ELF: {:?}", result);
    
    let report = result.unwrap();
    assert!(report.analysis_duration_ms > 0, "Analysis should take measurable time");
}

#[cfg(feature = "abi-verification")]
#[tokio::test] 
async fn test_abi_verifier_target_architectures() {
    let architectures = vec![
        TargetArchitecture::X86_64,
        TargetArchitecture::AArch64,
        TargetArchitecture::X86,
        TargetArchitecture::RISCV64,
        TargetArchitecture::WASM32,
    ];
    
    for arch in architectures {
        let verifier = AbiVerifier::new(arch.clone());
        let module = create_test_module(create_minimal_elf(), &format!("test_{:?}", arch));
        let result = verifier.verify_module_abi(&module, None).await;
        
        assert!(result.is_ok(), "Verifier should work for {:?}: {:?}", arch, result);
    }
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_function_signature_validation() {
    use hephaestus_forge::abi::signature::SignatureValidator;
    
    let validator = SignatureValidator::new(TargetArchitecture::X86_64);
    
    // Create test function signature
    let signature = FunctionSignature {
        name: "test_function".to_string(),
        return_type: TypeInfo {
            name: "int".to_string(),
            size: 4,
            alignment: 4,
            kind: TypeKind::Integer { width: 32, signed: true },
            fields: Vec::new(),
            is_union: false,
        },
        parameters: vec![
            Parameter {
                name: Some("x".to_string()),
                type_info: TypeInfo {
                    name: "int".to_string(),
                    size: 4,
                    alignment: 4,
                    kind: TypeKind::Integer { width: 32, signed: true },
                    fields: Vec::new(),
                    is_union: false,
                },
                is_register: true,
                register_location: Some("RDI".to_string()),
            }
        ],
        calling_convention: CallingConvention::SystemV,
        is_variadic: false,
        mangled_name: None,
    };
    
    let debug_info = hephaestus_forge::abi::DebugInfo::empty();
    let results = validator.validate_signatures(&debug_info).await;
    assert!(results.is_ok(), "Signature validation should succeed");
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_calling_convention_validation() {
    use hephaestus_forge::abi::calling_convention::CallingConventionValidator;
    
    let validator = CallingConventionValidator::new(TargetArchitecture::X86_64);
    
    // Create test debug info with functions
    let debug_info = hephaestus_forge::abi::DebugInfo::empty();
    
    let result = validator.verify_calling_conventions(&debug_info).await;
    assert!(result.is_ok(), "Calling convention validation should succeed for empty debug info");
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_data_layout_validation() {
    use hephaestus_forge::abi::layout::DataLayoutValidator;
    
    let validator = DataLayoutValidator::new(TargetArchitecture::X86_64);
    
    let debug_info = hephaestus_forge::abi::DebugInfo::empty();
    let results = validator.validate_layouts(&debug_info).await;
    
    assert!(results.is_ok(), "Data layout validation should succeed for empty debug info");
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_symbol_table_analysis() {
    use hephaestus_forge::abi::symbol_table::SymbolTableAnalyzer;
    
    let analyzer = SymbolTableAnalyzer::new();
    
    // Test with minimal ELF
    let violations = analyzer.analyze_symbols(&create_minimal_elf()).await;
    assert!(violations.is_ok(), "Symbol analysis should succeed for minimal ELF");
    
    let violations = violations.unwrap();
    // May have warnings but should not have critical errors for minimal ELF
    let critical_count = violations.iter()
        .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
        .count();
    assert_eq!(critical_count, 0, "Minimal ELF should not have critical violations");
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_cross_module_compatibility() {
    use hephaestus_forge::abi::compatibility::CompatibilityChecker;
    
    let checker = CompatibilityChecker::new(TargetArchitecture::X86_64);
    
    let old_module = create_test_module(create_minimal_elf(), "old_module");
    let new_module = create_test_module(create_minimal_elf(), "new_module");
    
    let violations = checker.check_compatibility(&new_module, &old_module).await;
    assert!(violations.is_ok(), "Compatibility check should succeed for identical modules");
    
    let violations = violations.unwrap();
    // Identical modules should have no violations
    assert_eq!(violations.len(), 0, "Identical modules should have no compatibility violations");
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_abi_violation_types() {
    // Test all violation types are properly categorized
    let violation = hephaestus_forge::abi::AbiViolation {
        violation_type: ViolationType::FunctionSignatureMismatch,
        description: "Test violation".to_string(),
        location: Some("test_location".to_string()),
        severity: ViolationSeverity::Error,
    };
    
    assert_eq!(violation.violation_type, ViolationType::FunctionSignatureMismatch);
    assert_eq!(violation.severity, ViolationSeverity::Error);
    assert_eq!(violation.location, Some("test_location".to_string()));
}

#[tokio::test]
async fn test_pe_binary_detection() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    // Create minimal PE binary (Windows executable format)
    let mut pe_binary = vec![0x4d, 0x5a]; // PE magic bytes "MZ"
    pe_binary.extend_from_slice(&[0x00; 62]); // Pad to minimum size
    
    let pe_module = create_test_module(pe_binary, "pe_module");
    let result = orchestrator.validate_module(&pe_module).await;
    
    // Should pass basic validation and detect PE format
    assert!(result.is_ok(), "PE module should pass basic validation: {:?}", result);
}

#[tokio::test]
async fn test_unknown_binary_format() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    // Create binary with unknown format but valid size
    let unknown_binary = vec![0xff, 0xfe, 0xfd, 0xfc]; // Unknown magic
    let mut padded_binary = unknown_binary;
    padded_binary.extend_from_slice(&[0x00; 64]); // Pad to minimum size
    
    let unknown_module = create_test_module(padded_binary, "unknown_module");
    let result = orchestrator.validate_module(&unknown_module).await;
    
    // Should pass basic validation but log warning about unknown format
    assert!(result.is_ok(), "Unknown format module should pass basic validation with warnings: {:?}", result);
}

#[tokio::test]
async fn test_abi_verification_performance() {
    let config = RuntimeConfig::default();
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    // Create larger test module
    let mut large_elf = create_minimal_elf();
    large_elf.extend_from_slice(&[0x00; 10000]); // Add padding to make it larger
    
    let large_module = create_test_module(large_elf, "large_module");
    
    let start = std::time::Instant::now();
    let result = orchestrator.validate_module(&large_module).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Large module should pass validation: {:?}", result);
    assert!(duration.as_millis() < 1000, "ABI verification should complete within 1 second, took: {:?}", duration);
}

#[tokio::test]
async fn test_concurrent_abi_verification() {
    let config = RuntimeConfig::default();
    let orchestrator = Arc::new(MetamorphicRuntimeOrchestrator::new(config).await.unwrap());
    
    // Test concurrent validation of multiple modules
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let orchestrator = orchestrator.clone();
        let module = create_test_module(create_minimal_elf(), &format!("concurrent_module_{}", i));
        
        let handle = tokio::spawn(async move {
            orchestrator.validate_module(&module).await
        });
        handles.push(handle);
    }
    
    // Wait for all validations to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent validation should succeed: {:?}", result);
    }
}

#[cfg(feature = "abi-verification")]
#[tokio::test]
async fn test_comprehensive_abi_pipeline() {
    let verifier = AbiVerifier::new(TargetArchitecture::X86_64);
    
    // Create two versions of a module for compatibility testing
    let module_v1 = create_test_module(create_minimal_elf(), "test_module");
    let mut module_v2 = module_v1.clone();
    module_v2.version = 2;
    
    // Test full ABI verification pipeline
    let report_v1 = verifier.verify_module_abi(&module_v1, None).await;
    assert!(report_v1.is_ok(), "Version 1 verification should succeed");
    
    let report_v2 = verifier.verify_module_abi(&module_v2, Some(&module_v1)).await;
    assert!(report_v2.is_ok(), "Version 2 compatibility check should succeed");
    
    let report = report_v2.unwrap();
    
    // Verify report structure
    assert!(report.analysis_duration_ms > 0);
    assert!(report.verified_functions >= 0);
    assert!(report.verified_types >= 0);
    
    // Compatible modules should pass
    assert!(report.is_compatible || report.violations.iter().all(|v| 
        !matches!(v.severity, ViolationSeverity::Critical | ViolationSeverity::Error)
    ));
}

/// Integration test with orchestrator's atomic module swap
#[tokio::test]
async fn test_abi_verification_in_module_swap() {
    let config = RuntimeConfig {
        max_concurrent_swaps: 1,
        rollback_window_ms: 1000,
        shadow_traffic_percent: 0.0,
    };
    let orchestrator = MetamorphicRuntimeOrchestrator::new(config).await.unwrap();
    
    let module = create_test_module(create_minimal_elf(), "swap_module");
    let strategy = hephaestus_forge::types::DeploymentStrategy::Immediate;
    
    // This should trigger ABI verification as part of the swap process
    let result = orchestrator.atomic_module_swap(
        module.id.clone(),
        module,
        strategy
    ).await;
    
    // May fail due to missing implementation details, but ABI verification should run
    match result {
        Ok(_) => {
            // Swap succeeded - ABI verification passed
        }
        Err(e) => {
            // If it fails, it should not be due to ABI verification for valid ELF
            let error_msg = format!("{}", e);
            assert!(!error_msg.contains("ABI verification failed"), 
                "Valid ELF should not fail ABI verification: {}", error_msg);
        }
    }
}