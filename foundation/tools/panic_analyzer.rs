#!/usr/bin/env cargo +nightly -Zscript

//! # Panic Pattern Static Analysis Tool
//! 
//! Systematically identifies and categorizes all panic-inducing patterns
//! in the ARES NovaCore ChronoSynclastic Fabric codebase.
//!
//! This tool supports the Phase 1 hardening effort by providing:
//! - Comprehensive inventory of all .unwrap(), .expect(), and panic! calls
//! - Risk assessment and prioritization
//! - Automated replacement suggestions
//! - Progress tracking during elimination efforts

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use regex::Regex;

/// Types of panic-inducing patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanicPattern {
    /// Direct .unwrap() call
    Unwrap,
    /// .expect() call with message
    Expect(String),
    /// Direct panic!() macro
    PanicMacro(String),
    /// Index operations that can panic
    IndexAccess,
    /// Division by zero potential
    Division,
}

/// Context and location of a panic pattern
#[derive(Debug, Clone)]
pub struct PanicOccurrence {
    /// File path where pattern occurs
    pub file_path: PathBuf,
    /// Line number in file
    pub line_number: usize,
    /// Column position
    pub column: usize,
    /// Type of panic pattern
    pub pattern: PanicPattern,
    /// Full line of code containing the pattern
    pub code_line: String,
    /// Surrounding context (previous/next lines)
    pub context: Vec<String>,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Suggested replacement
    pub suggestion: String,
}

/// Risk assessment for panic patterns
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Low risk (test code, benign operations)
    Low,
    /// Medium risk (could affect functionality)
    Medium,
    /// High risk (critical path, library code)
    High,
    /// Critical risk (security, core infrastructure)
    Critical,
}

/// Comprehensive panic analysis report
#[derive(Debug, Default)]
pub struct PanicAnalysisReport {
    /// All panic occurrences found
    pub occurrences: Vec<PanicOccurrence>,
    /// Summary by crate
    pub by_crate: HashMap<String, Vec<PanicOccurrence>>,
    /// Summary by pattern type
    pub by_pattern: HashMap<PanicPattern, usize>,
    /// Risk level distribution
    pub by_risk: HashMap<RiskLevel, usize>,
    /// Total count
    pub total_count: usize,
}

/// Main panic analyzer
pub struct PanicAnalyzer {
    /// Root directory to analyze
    root_dir: PathBuf,
    /// Compiled regex patterns
    unwrap_regex: Regex,
    expect_regex: Regex,
    panic_regex: Regex,
    index_regex: Regex,
    /// Current analysis report
    report: PanicAnalysisReport,
}

impl PanicAnalyzer {
    /// Create a new panic analyzer for the given directory
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let unwrap_regex = Regex::new(r"\.unwrap\(\)")?;
        let expect_regex = Regex::new(r#"\.expect\("([^"]+)"\)"#)?;
        let panic_regex = Regex::new(r#"panic!\("([^"]+)"\)"#)?;
        let index_regex = Regex::new(r"\[[^\]]+\](?!\s*=)")?; // Array access, not assignment
        
        Ok(Self {
            root_dir: root_dir.as_ref().to_path_buf(),
            unwrap_regex,
            expect_regex,
            panic_regex,
            index_regex,
            report: PanicAnalysisReport::default(),
        })
    }
    
    /// Run comprehensive panic analysis
    pub fn analyze(&mut self) -> Result<&PanicAnalysisReport, Box<dyn std::error::Error>> {
        println!("🔍 Starting comprehensive panic pattern analysis...");
        println!("📁 Scanning directory: {}", self.root_dir.display());
        
        // Find all Rust source files
        let rust_files = self.find_rust_files()?;
        println!("📄 Found {} Rust source files", rust_files.len());
        
        // Analyze each file
        for file_path in rust_files {
            if let Err(e) = self.analyze_file(&file_path) {
                eprintln!("⚠️  Warning: Failed to analyze {}: {}", file_path.display(), e);
            }
        }
        
        // Generate summary statistics
        self.generate_summary();
        
        println!("✅ Analysis complete!");
        println!("📊 Total panic patterns found: {}", self.report.total_count);
        
        Ok(&self.report)
    }
    
    /// Find all Rust source files in the directory tree
    fn find_rust_files(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut rust_files = Vec::new();
        
        fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    // Skip target and hidden directories
                    if let Some(name) = path.file_name() {
                        let name = name.to_string_lossy();
                        if !name.starts_with('.') && name != "target" {
                            visit_dir(&path, files)?;
                        }
                    }
                } else if path.extension() == Some("rs".as_ref()) {
                    files.push(path);
                }
            }
            Ok(())
        }
        
        visit_dir(&self.root_dir, &mut rust_files)?;
        Ok(rust_files)
    }
    
    /// Analyze a single Rust source file
    fn analyze_file(&mut self, file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();
        
        for (line_idx, line) in lines.iter().enumerate() {
            let line_number = line_idx + 1;
            
            // Check for .unwrap() patterns
            for unwrap_match in self.unwrap_regex.find_iter(line) {
                let occurrence = PanicOccurrence {
                    file_path: file_path.to_path_buf(),
                    line_number,
                    column: unwrap_match.start(),
                    pattern: PanicPattern::Unwrap,
                    code_line: line.to_string(),
                    context: self.get_context(&lines, line_idx),
                    risk_level: self.assess_risk(file_path, line),
                    suggestion: self.suggest_unwrap_replacement(line),
                };
                self.report.occurrences.push(occurrence);
            }
            
            // Check for .expect() patterns
            for expect_match in self.expect_regex.captures_iter(line) {
                let message = expect_match.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let occurrence = PanicOccurrence {
                    file_path: file_path.to_path_buf(),
                    line_number,
                    column: expect_match.get(0).unwrap().start(),
                    pattern: PanicPattern::Expect(message.clone()),
                    code_line: line.to_string(),
                    context: self.get_context(&lines, line_idx),
                    risk_level: self.assess_risk(file_path, line),
                    suggestion: self.suggest_expect_replacement(line, &message),
                };
                self.report.occurrences.push(occurrence);
            }
            
            // Check for panic!() patterns
            for panic_match in self.panic_regex.captures_iter(line) {
                let message = panic_match.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let occurrence = PanicOccurrence {
                    file_path: file_path.to_path_buf(),
                    line_number,
                    column: panic_match.get(0).unwrap().start(),
                    pattern: PanicPattern::PanicMacro(message.clone()),
                    code_line: line.to_string(),
                    context: self.get_context(&lines, line_idx),
                    risk_level: RiskLevel::Critical, // panic!() is always critical
                    suggestion: self.suggest_panic_replacement(line, &message),
                };
                self.report.occurrences.push(occurrence);
            }
        }
        
        Ok(())
    }
    
    /// Get surrounding context lines for better understanding
    fn get_context(&self, lines: &[&str], line_idx: usize) -> Vec<String> {
        let mut context = Vec::new();
        
        // Add previous line if available
        if line_idx > 0 {
            context.push(format!("  {} | {}", line_idx, lines[line_idx - 1]));
        }
        
        // Add current line
        context.push(format!("> {} | {}", line_idx + 1, lines[line_idx]));
        
        // Add next line if available
        if line_idx + 1 < lines.len() {
            context.push(format!("  {} | {}", line_idx + 2, lines[line_idx + 1]));
        }
        
        context
    }
    
    /// Assess risk level of a panic pattern
    fn assess_risk(&self, file_path: &Path, line: &str) -> RiskLevel {
        let path_str = file_path.to_string_lossy().to_lowercase();
        
        // Test code is lower risk
        if path_str.contains("test") || path_str.contains("example") {
            return RiskLevel::Low;
        }
        
        // Core runtime and security code is critical
        if path_str.contains("csf-runtime") || 
           path_str.contains("csf-sil") || 
           path_str.contains("security") ||
           path_str.contains("crypto") {
            return RiskLevel::Critical;
        }
        
        // Performance-critical paths are high risk
        if path_str.contains("csf-bus") || 
           path_str.contains("csf-time") || 
           path_str.contains("scheduler") ||
           path_str.contains("kernel") {
            return RiskLevel::High;
        }
        
        // Check line content for risk indicators
        let line_lower = line.to_lowercase();
        if line_lower.contains("unsafe") || 
           line_lower.contains("lock") || 
           line_lower.contains("atomic") {
            return RiskLevel::High;
        }
        
        RiskLevel::Medium
    }
    
    /// Suggest replacement for .unwrap() calls
    fn suggest_unwrap_replacement(&self, line: &str) -> String {
        if line.contains("Ok(") {
            return "Replace with proper error propagation using `?` operator".to_string();
        } else if line.contains("Some(") {
            return "Replace with pattern matching or `ok_or_else()`".to_string();
        } else if line.contains("lock()") {
            return "Use `try_lock()` or handle poisoned mutex".to_string();
        } else {
            return "Replace with proper error handling and Result propagation".to_string();
        }
    }
    
    /// Suggest replacement for .expect() calls
    fn suggest_expect_replacement(&self, line: &str, message: &str) -> String {
        format!("Convert expect(\"{}\") to proper error handling with custom error type", message)
    }
    
    /// Suggest replacement for panic!() calls
    fn suggest_panic_replacement(&self, line: &str, message: &str) -> String {
        format!("Replace panic!(\"{}\") with proper error return and logging", message)
    }
    
    /// Generate summary statistics
    fn generate_summary(&mut self) {
        self.report.total_count = self.report.occurrences.len();
        
        // Group by crate
        for occurrence in &self.report.occurrences {
            let crate_name = self.extract_crate_name(&occurrence.file_path);
            self.report.by_crate
                .entry(crate_name)
                .or_insert_with(Vec::new)
                .push(occurrence.clone());
        }
        
        // Group by pattern type
        for occurrence in &self.report.occurrences {
            *self.report.by_pattern
                .entry(occurrence.pattern.clone())
                .or_insert(0) += 1;
        }
        
        // Group by risk level
        for occurrence in &self.report.occurrences {
            *self.report.by_risk
                .entry(occurrence.risk_level.clone())
                .or_insert(0) += 1;
        }
    }
    
    /// Extract crate name from file path
    fn extract_crate_name(&self, path: &Path) -> String {
        let path_str = path.to_string_lossy();
        if let Some(start) = path_str.find("crates/") {
            let after_crates = &path_str[start + 7..];
            if let Some(end) = after_crates.find('/') {
                return after_crates[..end].to_string();
            }
        }
        "unknown".to_string()
    }
    
    /// Print detailed analysis report
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(80));
        println!("📊 PANIC PATTERN ANALYSIS REPORT");
        println!("{}", "=".repeat(80));
        
        println!("\n🎯 SUMMARY:");
        println!("   Total panic patterns found: {}", self.report.total_count);
        
        // Risk level distribution
        println!("\n⚠️  RISK LEVEL DISTRIBUTION:");
        let mut risk_levels: Vec<_> = self.report.by_risk.iter().collect();
        risk_levels.sort_by(|a, b| b.0.cmp(a.0)); // Sort by risk level (high to low)
        for (risk, count) in risk_levels {
            let icon = match risk {
                RiskLevel::Critical => "🚨",
                RiskLevel::High => "⚡",
                RiskLevel::Medium => "⚠️",
                RiskLevel::Low => "ℹ️",
            };
            println!("   {} {:?}: {} occurrences", icon, risk, count);
        }
        
        // Pattern type distribution
        println!("\n📝 PATTERN TYPE DISTRIBUTION:");
        for (pattern, count) in &self.report.by_pattern {
            let description = match pattern {
                PanicPattern::Unwrap => "Direct .unwrap() calls",
                PanicPattern::Expect(_) => ".expect() calls with messages",
                PanicPattern::PanicMacro(_) => "Direct panic!() macro calls",
                PanicPattern::IndexAccess => "Potentially panicking array access",
                PanicPattern::Division => "Potential division by zero",
            };
            println!("   {}: {} ({})", pattern_icon(pattern), count, description);
        }
        
        // Crate breakdown
        println!("\n📦 CRATE BREAKDOWN:");
        let mut crate_counts: Vec<_> = self.report.by_crate.iter().collect();
        crate_counts.sort_by(|a, b| b.1.len().cmp(&a.1.len())); // Sort by count
        for (crate_name, occurrences) in crate_counts.iter().take(10) {
            let critical_count = occurrences.iter()
                .filter(|o| o.risk_level == RiskLevel::Critical)
                .count();
            let high_count = occurrences.iter()
                .filter(|o| o.risk_level == RiskLevel::High)
                .count();
            
            println!("   {}: {} total (🚨{} critical, ⚡{} high)", 
                     crate_name, occurrences.len(), critical_count, high_count);
        }
        
        println!("\n🎯 PRIORITIZATION RECOMMENDATIONS:");
        println!("   1. 🚨 Address CRITICAL risk patterns first (security/core infrastructure)");
        println!("   2. ⚡ Then HIGH risk patterns (performance-critical paths)");
        println!("   3. ⚠️  MEDIUM risk patterns (general functionality)");
        println!("   4. ℹ️  LOW risk patterns (test code) can be addressed last");
        
        println!("\n{}", "=".repeat(80));
    }
    
    /// Generate detailed occurrence report for highest priority items
    pub fn print_critical_occurrences(&self) {
        let mut critical_occurrences: Vec<_> = self.report.occurrences.iter()
            .filter(|o| o.risk_level == RiskLevel::Critical)
            .collect();
        critical_occurrences.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        
        if critical_occurrences.is_empty() {
            println!("🎉 No critical risk panic patterns found!");
            return;
        }
        
        println!("\n🚨 CRITICAL RISK OCCURRENCES (IMMEDIATE ACTION REQUIRED):");
        println!("{}", "=".repeat(80));
        
        for (idx, occurrence) in critical_occurrences.iter().enumerate() {
            println!("\n{} CRITICAL ISSUE #{}", "🚨", idx + 1);
            println!("📁 File: {}", occurrence.file_path.display());
            println!("📍 Line: {}, Column: {}", occurrence.line_number, occurrence.column);
            println!("🔍 Pattern: {:?}", occurrence.pattern);
            println!("💡 Suggestion: {}", occurrence.suggestion);
            println!("📋 Context:");
            for context_line in &occurrence.context {
                println!("     {}", context_line);
            }
        }
        
        println!("\n{}", "=".repeat(80));
        println!("⚡ ACTION PLAN: Address these {} critical issues before proceeding!", critical_occurrences.len());
    }
}

/// Get icon for pattern type
fn pattern_icon(pattern: &PanicPattern) -> &'static str {
    match pattern {
        PanicPattern::Unwrap => "🔓",
        PanicPattern::Expect(_) => "❗",
        PanicPattern::PanicMacro(_) => "💥",
        PanicPattern::IndexAccess => "📇",
        PanicPattern::Division => "➗",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let root_dir = if args.len() > 1 {
        &args[1]
    } else {
        "crates/"
    };
    
    let mut analyzer = PanicAnalyzer::new(root_dir)?;
    let _report = analyzer.analyze()?;
    
    analyzer.print_report();
    analyzer.print_critical_occurrences();
    
    println!("\n🚀 Ready to begin systematic panic elimination!");
    println!("💻 Use this analysis to prioritize and track progress.");
    
    Ok(())
}