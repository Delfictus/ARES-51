//! Error types for csf-time crate

use thiserror::Error;

/// Result alias for csf-time operations
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for csf-time
#[derive(Debug, Error)]
pub enum Error {
	/// Global or component already initialized
	#[error("Already initialized: {0}")]
	AlreadyInitialized(String),

	/// Global or component not initialized
	#[error("Not initialized: {0}")]
	NotInitialized(String),

	/// Invalid parameter provided
	#[error("Invalid parameter: {0}")]
	InvalidParameter(String),

	/// Arithmetic overflow/underflow
	#[error("Arithmetic overflow/underflow")]
	ArithmeticOverflow,

	/// Time source failures
	#[error("Time error: {0}")]
	Time(String),

	/// Deadline scheduling failures
	#[error("Deadline failure for task {task_id}: {reason}")]
	DeadlineFailure { task_id: String, reason: String },

	/// Generic error wrapper
	#[error("{0}")]
	Generic(String),
}

