use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ares")]
#[command(about = "ARES Neuromorphic Command Line Interface - Natural Language Processing for Quantum Systems")]
#[command(long_about = "
ARES Neuromorphic CLI provides natural language command processing for the 
ARES ChronoSynclastic Fabric quantum temporal correlation system.

Features:
• Natural language command interpretation
• Neuromorphic pattern recognition (Brian2/Lava)
• Dynamic resource allocation based on context
• Always-on learning mode with STDP
• Hardware acceleration (CPU/GPU/Neuromorphic chips)
• Integration with C-LOGIC modules (DRPP, EMS, ADP, EGC)
")]
#[command(version, author)]
pub struct Cli {
    /// Enable verbose output (-v, -vv, -vvv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
    
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    
    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
    
    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
    pub format: OutputFormat,
    
    /// Neuromorphic backend preference
    #[arg(long, value_enum)]
    pub backend: Option<NeuromorphicBackend>,
    
    /// Force CPU simulation (disable hardware detection)
    #[arg(long)]
    pub force_cpu: bool,
    
    /// Enable learn mode by default
    #[arg(long)]
    pub learn: bool,
    
    #[command(subcommand)]
    pub command: crate::commands::Commands,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Human,
    Json,
    Yaml,
}

#[derive(clap::ValueEnum, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum NeuromorphicBackend {
    /// Auto-detect best available backend
    Auto,
    /// Brian2 with CPU simulation
    Brian2Cpu,
    /// Brian2 with GPU acceleration
    Brian2Gpu,
    /// Lava SDK simulation
    LavaSim,
    /// Lava SDK with neuromorphic hardware
    LavaHardware,
    /// Native C-LOGIC modules only
    Native,
}

impl Default for NeuromorphicBackend {
    fn default() -> Self {
        Self::Auto
    }
}