//! FFI error handling

use std::fmt;

pub type FFIResult<T> = Result<T, FFIError>;

/// FFI error types
#[derive(Debug, Clone)]
pub enum FFIError {
    /// Runtime not initialized
    NotInitialized,

    /// Runtime already initialized
    AlreadyInitialized,

    /// Invalid argument
    InvalidArgument(String),

    /// Null pointer
    NullPointer,

    /// UTF-8 conversion error
    Utf8Error,

    /// Serialization error
    SerializationError(String),

    /// Runtime error
    RuntimeError(String),

    /// Conversion error
    ConversionError(String),

    /// Unsupported operation
    UnsupportedOperation(String),

    /// Unknown error
    Unknown(String),
}

impl fmt::Display for FFIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FFIError::NotInitialized => write!(f, "CSF runtime not initialized"),
            FFIError::AlreadyInitialized => write!(f, "CSF runtime already initialized"),
            FFIError::InvalidArgument(s) => write!(f, "Invalid argument: {}", s),
            FFIError::NullPointer => write!(f, "Null pointer"),
            FFIError::Utf8Error => write!(f, "UTF-8 conversion error"),
            FFIError::SerializationError(s) => write!(f, "Serialization error: {}", s),
            FFIError::RuntimeError(s) => write!(f, "Runtime error: {}", s),
            FFIError::ConversionError(s) => write!(f, "Conversion error: {}", s),
            FFIError::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
            FFIError::Unknown(s) => write!(f, "Unknown error: {}", s),
        }
    }
}

impl std::error::Error for FFIError {}

impl From<std::str::Utf8Error> for FFIError {
    fn from(_: std::str::Utf8Error) -> Self {
        FFIError::Utf8Error
    }
}

impl From<std::ffi::NulError> for FFIError {
    fn from(_: std::ffi::NulError) -> Self {
        FFIError::Utf8Error
    }
}

impl From<serde_json::Error> for FFIError {
    fn from(e: serde_json::Error) -> Self {
        FFIError::SerializationError(e.to_string())
    }
}

impl From<anyhow::Error> for FFIError {
    fn from(e: anyhow::Error) -> Self {
        FFIError::RuntimeError(e.to_string())
    }
}

impl From<csf_core::error::Error> for FFIError {
    fn from(e: csf_core::error::Error) -> Self {
        FFIError::RuntimeError(e.to_string())
    }
}

impl From<csf_kernel::Error> for FFIError {
    fn from(e: csf_kernel::Error) -> Self {
        FFIError::RuntimeError(e.to_string())
    }
}

// BusError conversion handled manually in code since BusError is private

/// Convert error to FFI error code
pub fn error_to_code(error: &FFIError) -> i32 {
    match error {
        FFIError::NotInitialized => -1,
        FFIError::AlreadyInitialized => -2,
        FFIError::InvalidArgument(_) => -3,
        FFIError::NullPointer => -4,
        FFIError::Utf8Error => -5,
        FFIError::SerializationError(_) => -6,
        FFIError::RuntimeError(_) => -7,
        FFIError::ConversionError(_) => -8,
        FFIError::UnsupportedOperation(_) => -9,
        FFIError::Unknown(_) => -99,
    }
}

/// Get error message for error code
pub fn error_message(code: i32) -> &'static str {
    match code {
        -1 => "CSF runtime not initialized",
        -2 => "CSF runtime already initialized",
        -3 => "Invalid argument",
        -4 => "Null pointer",
        -5 => "UTF-8 conversion error",
        -6 => "Serialization error",
        -7 => "Runtime error",
        -8 => "Conversion error",
        -9 => "Unsupported operation",
        -99 => "Unknown error",
        _ => "Success",
    }
}

/// Get last error message
static mut LAST_ERROR: Option<String> = None;

pub fn set_last_error(error: FFIError) {
    unsafe {
        LAST_ERROR = Some(error.to_string());
    }
}

/// Get the last error message
#[no_mangle]
pub extern "C" fn csf_get_last_error() -> *const std::os::raw::c_char {
    unsafe {
        match &LAST_ERROR {
            Some(msg) => match std::ffi::CString::new(msg.as_str()) {
                Ok(c_str) => c_str.into_raw(),
                Err(_) => std::ptr::null(),
            },
            None => std::ptr::null(),
        }
    }
}
