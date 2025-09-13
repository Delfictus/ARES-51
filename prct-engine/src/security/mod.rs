// Industrial-grade security and validation framework for PRCT engine
// Implements defense-in-depth security principles for molecular data processing

use std::fmt;
use std::time::{Duration, Instant};
use regex::Regex;
use sha2::{Sha256, Digest};

/// Security configuration constants based on industry best practices
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Maximum allowed file size (100MB for large protein structures)
    pub max_file_size_bytes: usize,
    /// Maximum allowed line length in PDB files
    pub max_line_length: usize,
    /// Maximum number of atoms per structure (prevents memory exhaustion)
    pub max_atoms_per_structure: usize,
    /// Maximum coordinate value (prevents numerical overflow)
    pub max_coordinate_value: f64,
    /// Timeout for parsing operations (prevents DoS via slow parsing)
    pub parse_timeout_seconds: u64,
    /// Maximum allowed chains per structure
    pub max_chains_per_structure: usize,
    /// Maximum residues per chain
    pub max_residues_per_chain: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 100_000_000,  // 100MB
            max_line_length: 1000,             // PDB spec allows 80, we're generous
            max_atoms_per_structure: 1_000_000, // Large protein complexes
            max_coordinate_value: 9999.999,     // Beyond reasonable protein dimensions
            parse_timeout_seconds: 300,         // 5 minutes max parsing time
            max_chains_per_structure: 100,      // Multi-chain complexes
            max_residues_per_chain: 10_000,     // Very large proteins
        }
    }
}

/// Security-specific error types with detailed context
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityError {
    /// Input exceeds maximum allowed size
    FileSizeExceeded {
        actual_size: usize,
        max_allowed: usize,
        source: String,
    },
    /// Line length exceeds security limits
    LineTooLong {
        line_number: usize,
        actual_length: usize,
        max_allowed: usize,
    },
    /// Suspicious character patterns detected
    SuspiciousInput {
        pattern: String,
        location: String,
        risk_level: RiskLevel,
    },
    /// Numerical values outside safe ranges
    NumericalBounds {
        value: f64,
        field_name: String,
        min_allowed: f64,
        max_allowed: f64,
    },
    /// Operation timeout (prevents DoS attacks)
    OperationTimeout {
        operation: String,
        duration: Duration,
        max_allowed: Duration,
    },
    /// Resource exhaustion protection
    ResourceExhaustion {
        resource_type: String,
        current_count: usize,
        max_allowed: usize,
    },
    /// Input validation failure
    ValidationFailed {
        field: String,
        reason: String,
        input: String,
    },
    /// Generic invalid input error
    InvalidInput(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::FileSizeExceeded { actual_size, max_allowed, source } => {
                write!(f, "File size {} bytes exceeds maximum {} bytes from source: {}", 
                       actual_size, max_allowed, source)
            },
            SecurityError::LineTooLong { line_number, actual_length, max_allowed } => {
                write!(f, "Line {} length {} exceeds maximum {} characters", 
                       line_number, actual_length, max_allowed)
            },
            SecurityError::SuspiciousInput { pattern, location, risk_level } => {
                write!(f, "Suspicious pattern '{}' detected at {} (Risk: {:?})", 
                       pattern, location, risk_level)
            },
            SecurityError::NumericalBounds { value, field_name, min_allowed, max_allowed } => {
                write!(f, "Value {} in field '{}' outside bounds [{}, {}]", 
                       value, field_name, min_allowed, max_allowed)
            },
            SecurityError::OperationTimeout { operation, duration, max_allowed } => {
                write!(f, "Operation '{}' timed out after {:?} (max: {:?})", 
                       operation, duration, max_allowed)
            },
            SecurityError::ResourceExhaustion { resource_type, current_count, max_allowed } => {
                write!(f, "{} count {} exceeds maximum {}", 
                       resource_type, current_count, max_allowed)
            },
            SecurityError::ValidationFailed { field, reason, input } => {
                write!(f, "Validation failed for field '{}': {} (input: '{}')", 
                       field, reason, input)
            },
            SecurityError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            },
        }
    }
}

impl std::error::Error for SecurityError {}

impl From<Box<dyn std::error::Error>> for SecurityError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        SecurityError::InvalidInput(err.to_string())
    }
}

/// Comprehensive input validator with security focus
#[derive(Debug, Clone)]
pub struct SecurityValidator {
    config: SecurityConfig,
    // Compiled regex patterns for efficiency
    sql_injection_pattern: Regex,
    xss_pattern: Regex,
    path_traversal_pattern: Regex,
    control_char_pattern: Regex,
    // Tracking for rate limiting and resource monitoring
    operation_start_time: Option<Instant>,
    atom_count: usize,
    chain_count: usize,
    residue_counts: Vec<usize>,  // Per chain
}

impl SecurityValidator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            config: SecurityConfig::default(),
            sql_injection_pattern: Regex::new(r"(?i)(union|select|insert|update|delete|drop|exec|script)")?,
            xss_pattern: Regex::new(r"(?i)(<script|javascript:|vbscript:|data:)")?,
            path_traversal_pattern: Regex::new(r"(\.\./|\.\.\\|\.\./\\|\.\.\\//)")?,
            control_char_pattern: Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")?,
            operation_start_time: None,
            atom_count: 0,
            chain_count: 0,
            residue_counts: Vec::new(),
        })
    }

    /// Start timing an operation for timeout protection
    pub fn start_operation(&mut self, operation_name: &str) {
        self.operation_start_time = Some(Instant::now());
        log::info!("Started security-monitored operation: {}", operation_name);
    }

    /// Check if operation has exceeded timeout
    pub fn check_timeout(&self, operation_name: &str) -> Result<(), SecurityError> {
        if let Some(start_time) = self.operation_start_time {
            let elapsed = start_time.elapsed();
            let max_duration = Duration::from_secs(self.config.parse_timeout_seconds);
            
            if elapsed > max_duration {
                return Err(SecurityError::OperationTimeout {
                    operation: operation_name.to_string(),
                    duration: elapsed,
                    max_allowed: max_duration,
                });
            }
        }
        Ok(())
    }

    /// Validate file size before processing
    pub fn validate_file_size(&self, size: usize, source: &str) -> Result<(), SecurityError> {
        if size > self.config.max_file_size_bytes {
            return Err(SecurityError::FileSizeExceeded {
                actual_size: size,
                max_allowed: self.config.max_file_size_bytes,
                source: source.to_string(),
            });
        }
        Ok(())
    }

    /// Validate individual line for security threats
    pub fn validate_line(&self, line: &str, line_number: usize) -> Result<(), SecurityError> {
        // Check line length
        if line.len() > self.config.max_line_length {
            return Err(SecurityError::LineTooLong {
                line_number,
                actual_length: line.len(),
                max_allowed: self.config.max_line_length,
            });
        }

        // Check for control characters
        if self.control_char_pattern.is_match(line) {
            return Err(SecurityError::SuspiciousInput {
                pattern: "Control characters".to_string(),
                location: format!("line {}", line_number),
                risk_level: RiskLevel::High,
            });
        }

        // Check for SQL injection patterns
        if self.sql_injection_pattern.is_match(line) {
            return Err(SecurityError::SuspiciousInput {
                pattern: "SQL injection pattern".to_string(),
                location: format!("line {}", line_number),
                risk_level: RiskLevel::Critical,
            });
        }

        // Check for XSS patterns
        if self.xss_pattern.is_match(line) {
            return Err(SecurityError::SuspiciousInput {
                pattern: "XSS pattern".to_string(),
                location: format!("line {}", line_number),
                risk_level: RiskLevel::High,
            });
        }

        // Check for path traversal
        if self.path_traversal_pattern.is_match(line) {
            return Err(SecurityError::SuspiciousInput {
                pattern: "Path traversal".to_string(),
                location: format!("line {}", line_number),
                risk_level: RiskLevel::High,
            });
        }

        Ok(())
    }

    /// Validate numerical coordinate with bounds checking
    pub fn validate_coordinate(&self, value_str: &str, field_name: &str) -> Result<f64, SecurityError> {
        // First check for non-numeric injection attempts
        if value_str.contains(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E') {
            return Err(SecurityError::ValidationFailed {
                field: field_name.to_string(),
                reason: "Contains non-numeric characters".to_string(),
                input: value_str.to_string(),
            });
        }

        // Parse the value
        let value = value_str.parse::<f64>().map_err(|_| SecurityError::ValidationFailed {
            field: field_name.to_string(),
            reason: "Invalid floating point format".to_string(),
            input: value_str.to_string(),
        })?;

        // Check for special values
        if !value.is_finite() {
            return Err(SecurityError::ValidationFailed {
                field: field_name.to_string(),
                reason: "Non-finite value (NaN or infinity)".to_string(),
                input: value_str.to_string(),
            });
        }

        // Check bounds
        let max_coord = self.config.max_coordinate_value;
        if value.abs() > max_coord {
            return Err(SecurityError::NumericalBounds {
                value,
                field_name: field_name.to_string(),
                min_allowed: -max_coord,
                max_allowed: max_coord,
            });
        }

        Ok(value)
    }

    /// Validate string field with length and content checks
    pub fn validate_string_field(&self, value: &str, field_name: &str, max_length: usize) -> Result<String, SecurityError> {
        if value.len() > max_length {
            return Err(SecurityError::ValidationFailed {
                field: field_name.to_string(),
                reason: format!("Length {} exceeds maximum {}", value.len(), max_length),
                input: value.to_string(),
            });
        }

        // Check for control characters
        if self.control_char_pattern.is_match(value) {
            return Err(SecurityError::ValidationFailed {
                field: field_name.to_string(),
                reason: "Contains control characters".to_string(),
                input: value.to_string(),
            });
        }

        // Sanitize by removing any potentially dangerous characters
        let sanitized = value.chars()
            .filter(|c| c.is_ascii_alphanumeric() || " -._+()[]{}".contains(*c))
            .collect();

        Ok(sanitized)
    }

    /// Track atom count for resource exhaustion protection
    pub fn increment_atom_count(&mut self) -> Result<(), SecurityError> {
        self.atom_count += 1;
        if self.atom_count > self.config.max_atoms_per_structure {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "atoms".to_string(),
                current_count: self.atom_count,
                max_allowed: self.config.max_atoms_per_structure,
            });
        }
        Ok(())
    }

    /// Track chain count
    pub fn increment_chain_count(&mut self) -> Result<(), SecurityError> {
        self.chain_count += 1;
        if self.chain_count > self.config.max_chains_per_structure {
            return Err(SecurityError::ResourceExhaustion {
                resource_type: "chains".to_string(),
                current_count: self.chain_count,
                max_allowed: self.config.max_chains_per_structure,
            });
        }
        self.residue_counts.push(0);  // Initialize residue count for this chain
        Ok(())
    }

    /// Track residue count for current chain
    pub fn increment_residue_count(&mut self) -> Result<(), SecurityError> {
        if let Some(last_count) = self.residue_counts.last_mut() {
            *last_count += 1;
            if *last_count > self.config.max_residues_per_chain {
                return Err(SecurityError::ResourceExhaustion {
                    resource_type: "residues_per_chain".to_string(),
                    current_count: *last_count,
                    max_allowed: self.config.max_residues_per_chain,
                });
            }
        }
        Ok(())
    }

    /// Generate security hash of input for audit trail
    pub fn generate_input_hash(&self, input: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input);
        format!("{:x}", hasher.finalize())
    }

    /// Reset counters for new structure
    pub fn reset_for_new_structure(&mut self) {
        self.atom_count = 0;
        self.chain_count = 0;
        self.residue_counts.clear();
        self.operation_start_time = None;
    }
}

impl Default for SecurityValidator {
    fn default() -> Self {
        Self::new().expect("Failed to create SecurityValidator with default regex patterns")
    }
}

/// Security audit logger for tracking validation events
pub struct SecurityAuditLog {
    violations: Vec<SecurityViolation>,
    start_time: Instant,
}

#[derive(Debug, Clone)]
pub struct SecurityViolation {
    timestamp: Instant,
    violation_type: String,
    severity: RiskLevel,
    details: String,
    input_hash: Option<String>,
}

impl SecurityAuditLog {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn log_violation(&mut self, violation_type: &str, severity: RiskLevel, details: &str, input_hash: Option<String>) {
        self.violations.push(SecurityViolation {
            timestamp: Instant::now(),
            violation_type: violation_type.to_string(),
            severity,
            details: details.to_string(),
            input_hash,
        });
        
        log::warn!("Security violation: {} - {} (severity: {:?})", violation_type, details, severity);
    }

    pub fn get_violations(&self) -> &[SecurityViolation] {
        &self.violations
    }

    pub fn has_critical_violations(&self) -> bool {
        self.violations.iter().any(|v| matches!(v.severity, RiskLevel::Critical))
    }

    pub fn generate_report(&self) -> String {
        let mut report = format!("Security Audit Report\n");
        report.push_str(&format!("Session Duration: {:?}\n", self.start_time.elapsed()));
        report.push_str(&format!("Total Violations: {}\n\n", self.violations.len()));

        for (i, violation) in self.violations.iter().enumerate() {
            report.push_str(&format!("Violation {}: {} (Severity: {:?})\n", i + 1, violation.violation_type, violation.severity));
            report.push_str(&format!("  Time: {:?}\n", violation.timestamp.duration_since(self.start_time)));
            report.push_str(&format!("  Details: {}\n", violation.details));
            if let Some(hash) = &violation.input_hash {
                report.push_str(&format!("  Input Hash: {}\n", hash));
            }
            report.push_str("\n");
        }

        report
    }
}

impl Default for SecurityAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// Include comprehensive test module
#[cfg(test)]
mod tests;

#[cfg(test)]
mod basic_tests {
    use super::*;

    #[test]
    fn test_security_validator_creation() {
        let validator = SecurityValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn test_file_size_validation() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid size
        assert!(validator.validate_file_size(1000, "test").is_ok());
        
        // Excessive size
        let result = validator.validate_file_size(200_000_000, "test");
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::FileSizeExceeded { actual_size, max_allowed, .. } => {
                assert_eq!(actual_size, 200_000_000);
                assert_eq!(max_allowed, 100_000_000);
            },
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_line_validation() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid line
        assert!(validator.validate_line("ATOM  1  N  MET A  1  20.154  16.967  27.462", 1).is_ok());
        
        // Too long
        let long_line = "A".repeat(2000);
        assert!(validator.validate_line(&long_line, 1).is_err());
        
        // SQL injection attempt
        assert!(validator.validate_line("ATOM DROP TABLE users", 1).is_err());
        
        // Control characters
        assert!(validator.validate_line("ATOM\x00\x01test", 1).is_err());
    }

    #[test]
    fn test_coordinate_validation() {
        let validator = SecurityValidator::new().unwrap();
        
        // Valid coordinates
        assert_eq!(validator.validate_coordinate("123.456", "x").unwrap(), 123.456);
        assert_eq!(validator.validate_coordinate("-123.456", "x").unwrap(), -123.456);
        assert_eq!(validator.validate_coordinate("1.23e-4", "x").unwrap(), 1.23e-4);
        
        // Invalid formats
        assert!(validator.validate_coordinate("123.abc", "x").is_err());
        assert!(validator.validate_coordinate("'; DROP", "x").is_err());
        assert!(validator.validate_coordinate("NaN", "x").is_err());
        
        // Out of bounds
        assert!(validator.validate_coordinate("99999.999", "x").is_err());
    }

    #[test]
    fn test_resource_tracking() {
        let mut validator = SecurityValidator::new().unwrap();
        
        // Test atom counting
        for _ in 0..1000 {
            assert!(validator.increment_atom_count().is_ok());
        }
        
        // Test chain counting
        for _ in 0..50 {
            assert!(validator.increment_chain_count().is_ok());
        }
    }

    #[test]
    fn test_audit_log() {
        let mut audit_log = SecurityAuditLog::new();
        
        audit_log.log_violation("test_violation", RiskLevel::Medium, "Test details", None);
        assert_eq!(audit_log.get_violations().len(), 1);
        assert!(!audit_log.has_critical_violations());
        
        audit_log.log_violation("critical_violation", RiskLevel::Critical, "Critical details", Some("hash123".to_string()));
        assert_eq!(audit_log.get_violations().len(), 2);
        assert!(audit_log.has_critical_violations());
        
        let report = audit_log.generate_report();
        assert!(report.contains("Security Audit Report"));
        assert!(report.contains("Total Violations: 2"));
    }
}