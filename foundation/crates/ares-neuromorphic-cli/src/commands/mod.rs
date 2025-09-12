use clap::Subcommand;
use anyhow::Result;

#[cfg(not(feature = "status-only"))]
pub mod interactive;
#[cfg(not(feature = "status-only"))]
pub mod enhanced_interactive;
pub mod status;
#[cfg(not(feature = "status-only"))]
pub mod learn;
#[cfg(not(feature = "status-only"))]
pub mod query;

#[cfg(not(feature = "status-only"))]
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start interactive neuromorphic CLI mode
    #[command(alias = "i")]
    Interactive,
    /// Start enhanced interactive mode with always-on NLP
    #[command(alias = "e")]
    Enhanced,
    /// Show neuromorphic system status and metrics
    Status(status::StatusArgs),
    /// Manage learning mode and training data
    Learn(learn::LearnArgs),
    /// Execute natural language query
    #[command(alias = "q")]
    Query { #[arg(value_name = "INPUT")] input: String },
}

#[cfg(feature = "status-only")]
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Show neuromorphic system status and metrics
    Status(status::StatusArgs),
}

pub type CommandResult = Result<()>;

/// Shared command utilities for consistent UX
pub mod utils {
    use colored::*;
    use indicatif::{ProgressBar, ProgressStyle};
    
    pub fn success_message(msg: &str) {
        println!("{} {}", "✓".green().bold(), msg);
    }
    
    pub fn error_message(msg: &str) {
        eprintln!("{} {}", "✗".red().bold(), msg);
    }
    
    pub fn info_message(msg: &str) {
        println!("{} {}", "ℹ".blue().bold(), msg);
    }
    
    pub fn neural_message(msg: &str) {
        println!("{} {}", "🧠".cyan(), msg);
    }
    
    pub fn quantum_message(msg: &str) {
        println!("{} {}", "⚛️".magenta(), msg);
    }
    
    pub fn debug_message(msg: &str) {
        println!("{} {}", "🔍".bright_black(), msg);
    }
    
    pub fn warn_message(msg: &str) {
        println!("{} {}", "⚠️".yellow().bold(), msg);
    }
    
    pub fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        pb.set_message(message.to_string());
        pb
    }
    
    pub fn create_spinner(message: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    }
}
