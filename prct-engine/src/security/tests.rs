// Comprehensive security tests for industrial-grade validation
#[cfg(test)]
mod security_tests {
    // Specific imports instead of wildcard
    use crate::security::{SecurityValidator, SecurityError, SecurityConfig, RiskLevel, SecurityAuditLog};
    use std::time::Duration;

    #[test]
    fn test_security_validator_creation() {
        let validator = SecurityValidator::new();
        assert!(validator.is_ok(), "SecurityValidator creation should succeed");
    }

    #[test]
    fn test_file_size_validation_success() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid size
        assert!(validator.validate_file_size(1000, "test.pdb").is_ok());
        assert!(validator.validate_file_size(50_000_000, "large.pdb").is_ok()); // 50MB is OK
    }

    #[test]
    fn test_file_size_validation_failure() {
        let validator = SecurityValidator::new().unwrap();
        
        // Excessive size (200MB > 100MB limit)
        let result = validator.validate_file_size(200_000_000, "huge.pdb");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SecurityError::FileSizeExceeded { actual_size, max_allowed, source } => {
                assert_eq!(actual_size, 200_000_000);
                assert_eq!(max_allowed, 100_000_000);
                assert_eq!(source, "huge.pdb");
            },
            _ => panic!("Wrong error type returned"),
        }
    }

    #[test]
    fn test_line_validation_success() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid PDB lines
        let valid_lines = [
            "ATOM      1  N   MET A   1      20.154  16.967  27.462  1.00 45.23           N  ",
            "HEADER    HYDROLASE/DNA                           20-JUL-95   1ABC              ",
            "TITLE     HUMAN LYSOZYME                                                      ",
            "SEQRES   1 A   99  MET ALA SER VAL GLU THR ALA ALA ALA ALA ALA ALA ALA",
        ];

        for line in &valid_lines {
            assert!(validator.validate_line(line, 1).is_ok(), 
                   "Valid line should pass validation: {}", line);
        }
    }

    #[test]
    fn test_line_validation_too_long() {
        let validator = SecurityValidator::new().unwrap();
        
        // Line exceeding maximum length
        let long_line = "A".repeat(2000);
        let result = validator.validate_line(&long_line, 1);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::LineTooLong { line_number, actual_length, max_allowed } => {
                assert_eq!(line_number, 1);
                assert_eq!(actual_length, 2000);
                assert_eq!(max_allowed, 1000);
            },
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_line_validation_sql_injection() {
        let validator = SecurityValidator::new().unwrap();
        
        let malicious_lines = [
            "ATOM  1  N  MET A  1 DROP TABLE proteins; --",
            "HEADER SELECT * FROM users WHERE 1=1",
            "TITLE ' OR 1=1; INSERT INTO malicious",
            "ATOM UNION SELECT password FROM admin",
        ];

        for line in &malicious_lines {
            let result = validator.validate_line(line, 1);
            assert!(result.is_err(), "SQL injection pattern should be detected: {}", line);
            
            match result.unwrap_err() {
                SecurityError::SuspiciousInput { pattern, risk_level, .. } => {
                    assert_eq!(pattern, "SQL injection pattern");
                    assert_eq!(risk_level, RiskLevel::Critical);
                },
                _ => panic!("Wrong error type for SQL injection"),
            }
        }
    }

    #[test]
    fn test_line_validation_xss_patterns() {
        let validator = SecurityValidator::new().unwrap();
        
        let xss_lines = [
            "HEADER <script>alert('xss')</script>",
            "TITLE javascript:alert(document.cookie)",
            "AUTHOR <img src=x onerror=alert('xss')>",
            "REMARK vbscript:msgbox('malicious')",
        ];

        for line in &xss_lines {
            let result = validator.validate_line(line, 1);
            assert!(result.is_err(), "XSS pattern should be detected: {}", line);
            
            match result.unwrap_err() {
                SecurityError::SuspiciousInput { pattern, risk_level, .. } => {
                    assert_eq!(pattern, "XSS pattern");
                    assert_eq!(risk_level, RiskLevel::High);
                },
                _ => panic!("Wrong error type for XSS"),
            }
        }
    }

    #[test]
    fn test_line_validation_path_traversal() {
        let validator = SecurityValidator::new().unwrap();
        
        let path_traversal_lines = [
            "HEADER ../../../etc/passwd",
            "TITLE ..\\windows\\system32\\config",
            "AUTHOR ../../home/user/.ssh/id_rsa",
            "REMARK ../../../var/log/auth.log",
        ];

        for line in &path_traversal_lines {
            let result = validator.validate_line(line, 1);
            assert!(result.is_err(), "Path traversal should be detected: {}", line);
            
            match result.unwrap_err() {
                SecurityError::SuspiciousInput { pattern, risk_level, .. } => {
                    assert_eq!(pattern, "Path traversal");
                    assert_eq!(risk_level, RiskLevel::High);
                },
                _ => panic!("Wrong error type for path traversal"),
            }
        }
    }

    #[test]
    fn test_line_validation_control_characters() {
        let validator = SecurityValidator::new().unwrap();
        
        let control_char_lines = [
            "ATOM\x00\x01test",
            "HEADER\x08\x0Bmalicious",
            "TITLE\x7Fdelete_character",
            "AUTHOR\x1F\x1Econtrol_chars",
        ];

        for line in &control_char_lines {
            let result = validator.validate_line(line, 1);
            assert!(result.is_err(), "Control characters should be detected");
            
            match result.unwrap_err() {
                SecurityError::SuspiciousInput { pattern, risk_level, .. } => {
                    assert_eq!(pattern, "Control characters");
                    assert_eq!(risk_level, RiskLevel::High);
                },
                _ => panic!("Wrong error type for control characters"),
            }
        }
    }

    #[test]
    fn test_coordinate_validation_success() {
        let validator = SecurityValidator::new().unwrap();
        
        let valid_coordinates = [
            ("123.456", 123.456),
            ("-123.456", -123.456),
            ("0.0", 0.0),
            ("999.999", 999.999),
            ("-999.999", -999.999),
            ("1.23e-4", 1.23e-4),
            ("1.23E+3", 1.23E+3),
        ];

        for (coord_str, expected) in &valid_coordinates {
            let result = validator.validate_coordinate(coord_str, "test_coord");
            assert!(result.is_ok(), "Valid coordinate should pass: {}", coord_str);
            assert!((result.unwrap() - expected).abs() < 1e-10, "Coordinate value should match");
        }
    }

    #[test]
    fn test_coordinate_validation_invalid_format() {
        let validator = SecurityValidator::new().unwrap();
        
        let invalid_coordinates = [
            "123.abc",
            "'; DROP TABLE",
            "NaN",
            "infinity",
            "123..456",
            "12.34.56",
            "abc123",
            "<script>",
        ];

        for coord_str in &invalid_coordinates {
            let result = validator.validate_coordinate(coord_str, "test_coord");
            assert!(result.is_err(), "Invalid coordinate should fail: {}", coord_str);
        }
    }

    #[test]
    fn test_coordinate_validation_out_of_bounds() {
        let validator = SecurityValidator::new().unwrap();
        
        let out_of_bounds_coordinates = [
            "99999.999",   // Too large
            "-99999.999",  // Too small  
            "10000.0",     // Just over limit
            "-10000.0",    // Just under limit
        ];

        for coord_str in &out_of_bounds_coordinates {
            let result = validator.validate_coordinate(coord_str, "test_coord");
            assert!(result.is_err(), "Out of bounds coordinate should fail: {}", coord_str);
            
            match result.unwrap_err() {
                SecurityError::NumericalBounds { field_name, min_allowed, max_allowed, .. } => {
                    assert_eq!(field_name, "test_coord");
                    assert_eq!(min_allowed, -9999.999);
                    assert_eq!(max_allowed, 9999.999);
                },
                _ => panic!("Wrong error type for out of bounds coordinate"),
            }
        }
    }

    #[test]
    fn test_string_field_validation() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid strings
        let valid_strings = [
            ("CA", "CA"),
            ("MET", "MET"),
            ("CHAIN_A", "CHAIN_A"),
            ("Protein-1", "Protein-1"),
            ("Test (2021)", "Test (2021)"),
        ];

        for (input, expected) in &valid_strings {
            let result = validator.validate_string_field(input, "test_field", 20);
            assert!(result.is_ok(), "Valid string should pass: {}", input);
            assert_eq!(result.unwrap(), *expected);
        }
    }

    #[test]
    fn test_string_field_validation_too_long() {
        let validator = SecurityValidator::new().unwrap();
        
        let long_string = "A".repeat(100);
        let result = validator.validate_string_field(&long_string, "test_field", 10);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::ValidationFailed { field, reason, .. } => {
                assert_eq!(field, "test_field");
                assert!(reason.contains("Length"));
                assert!(reason.contains("exceeds maximum"));
            },
            _ => panic!("Wrong error type for long string"),
        }
    }

    #[test]
    fn test_string_field_validation_control_chars() {
        let validator = SecurityValidator::new().unwrap();
        
        let control_string = "MET\x00\x01";
        let result = validator.validate_string_field(control_string, "residue_name", 10);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::ValidationFailed { field, reason, .. } => {
                assert_eq!(field, "residue_name");
                assert_eq!(reason, "Contains control characters");
            },
            _ => panic!("Wrong error type for control chars"),
        }
    }

    #[test]
    fn test_resource_tracking() {
        let mut validator = SecurityValidator::new().unwrap();
        
        // Test atom counting within limits
        for _ in 0..1000 {
            assert!(validator.increment_atom_count().is_ok());
        }
        
        // Test chain counting within limits
        for _ in 0..10 {
            assert!(validator.increment_chain_count().is_ok());
        }
        
        // Test residue counting within limits
        for _ in 0..100 {
            assert!(validator.increment_residue_count().is_ok());
        }
    }

    #[test]
    fn test_resource_exhaustion_atoms() {
        let mut validator = SecurityValidator::new().unwrap();
        
        // Set a lower limit for testing by creating a custom config
        // For now, test with default limits but verify error type
        let default_limit = 1_000_000;
        
        // This would take too long to test with real limits, so we verify the error type
        // by manually triggering the condition
        validator.atom_count = default_limit;
        let result = validator.increment_atom_count();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::ResourceExhaustion { resource_type, current_count, max_allowed } => {
                assert_eq!(resource_type, "atoms");
                assert_eq!(current_count, default_limit + 1);
                assert_eq!(max_allowed, default_limit);
            },
            _ => panic!("Wrong error type for atom exhaustion"),
        }
    }

    #[test]
    fn test_timeout_checking() {
        let mut validator = SecurityValidator::new().unwrap();
        
        // Start operation
        validator.start_operation("test_operation");
        
        // Check timeout immediately (should pass)
        assert!(validator.check_timeout("test_operation").is_ok());
        
        // Simulate timeout by manually setting start time to past
        validator.operation_start_time = Some(std::time::Instant::now() - Duration::from_secs(400));
        
        let result = validator.check_timeout("test_operation");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SecurityError::OperationTimeout { operation, max_allowed, .. } => {
                assert_eq!(operation, "test_operation");
                assert_eq!(max_allowed, Duration::from_secs(300));
            },
            _ => panic!("Wrong error type for timeout"),
        }
    }

    #[test]
    fn test_input_hash_generation() {
        let validator = SecurityValidator::new().unwrap();
        
        let input1 = b"ATOM      1  N   MET A   1      20.154  16.967  27.462";
        let input2 = b"ATOM      2  CA  MET A   1      21.618  16.507  -8.620";
        
        let hash1 = validator.generate_input_hash(input1);
        let hash2 = validator.generate_input_hash(input2);
        
        // Hashes should be different for different inputs
        assert_ne!(hash1, hash2);
        
        // Same input should produce same hash
        let hash1_repeat = validator.generate_input_hash(input1);
        assert_eq!(hash1, hash1_repeat);
        
        // Hash should be valid hex string
        assert!(hash1.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash1.len(), 64); // SHA256 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_reset_functionality() {
        let mut validator = SecurityValidator::new().unwrap();
        
        // Set some state
        validator.atom_count = 100;
        validator.chain_count = 5;
        validator.residue_counts = vec![10, 20, 30];
        validator.start_operation("test");
        
        // Reset
        validator.reset_for_new_structure();
        
        // Verify reset
        assert_eq!(validator.atom_count, 0);
        assert_eq!(validator.chain_count, 0);
        assert!(validator.residue_counts.is_empty());
        assert!(validator.operation_start_time.is_none());
    }

    #[test]
    fn test_audit_log() {
        let mut audit_log = SecurityAuditLog::new();
        
        // Log some violations
        audit_log.log_violation("test_violation", RiskLevel::Medium, "Test details", None);
        audit_log.log_violation("critical_issue", RiskLevel::Critical, "Critical details", Some("abc123".to_string()));
        audit_log.log_violation("low_priority", RiskLevel::Low, "Low priority details", None);
        
        // Check violation count
        assert_eq!(audit_log.get_violations().len(), 3);
        
        // Check critical violation detection
        assert!(audit_log.has_critical_violations());
        
        // Generate report
        let report = audit_log.generate_report();
        assert!(report.contains("Security Audit Report"));
        assert!(report.contains("Total Violations: 3"));
        assert!(report.contains("test_violation"));
        assert!(report.contains("critical_issue"));
        assert!(report.contains("low_priority"));
        assert!(report.contains("Input Hash: abc123"));
    }

    #[test]
    fn test_security_error_display() {
        let errors = [
            SecurityError::FileSizeExceeded { 
                actual_size: 200_000_000, 
                max_allowed: 100_000_000, 
                source: "test.pdb".to_string() 
            },
            SecurityError::LineTooLong { 
                line_number: 42, 
                actual_length: 2000, 
                max_allowed: 1000 
            },
            SecurityError::SuspiciousInput { 
                pattern: "SQL injection".to_string(), 
                location: "line 10".to_string(), 
                risk_level: RiskLevel::Critical 
            },
            SecurityError::NumericalBounds { 
                value: 99999.0, 
                field_name: "coordinate".to_string(), 
                min_allowed: -9999.0, 
                max_allowed: 9999.0 
            },
            SecurityError::OperationTimeout { 
                operation: "parsing".to_string(), 
                duration: Duration::from_secs(400), 
                max_allowed: Duration::from_secs(300) 
            },
        ];

        for error in &errors {
            let display_string = format!("{}", error);
            assert!(!display_string.is_empty(), "Error should have display string");
            // Each error type should include key information in display
            match error {
                SecurityError::FileSizeExceeded { actual_size, max_allowed, .. } => {
                    assert!(display_string.contains(&actual_size.to_string()));
                    assert!(display_string.contains(&max_allowed.to_string()));
                },
                SecurityError::LineTooLong { line_number, actual_length, .. } => {
                    assert!(display_string.contains(&line_number.to_string()));
                    assert!(display_string.contains(&actual_length.to_string()));
                },
                SecurityError::SuspiciousInput { pattern, location, .. } => {
                    assert!(display_string.contains(pattern));
                    assert!(display_string.contains(location));
                },
                SecurityError::NumericalBounds { field_name, .. } => {
                    assert!(display_string.contains(field_name));
                },
                SecurityError::OperationTimeout { operation, .. } => {
                    assert!(display_string.contains(operation));
                },
                _ => {}
            }
        }
    }

    #[test]
    fn test_security_config_defaults() {
        let config = SecurityConfig::default();
        
        // Verify sensible defaults
        assert_eq!(config.max_file_size_bytes, 100_000_000); // 100MB
        assert_eq!(config.max_line_length, 1000);
        assert_eq!(config.max_atoms_per_structure, 1_000_000);
        assert_eq!(config.max_coordinate_value, 9999.999);
        assert_eq!(config.parse_timeout_seconds, 300); // 5 minutes
        assert_eq!(config.max_chains_per_structure, 100);
        assert_eq!(config.max_residues_per_chain, 10_000);
    }
}