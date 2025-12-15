/// Common types used across spectrum-governance crates.
///
/// This module will contain shared types like:
/// - Model trait definitions
/// - Result types
/// - Common error types
/// - Shared data structures

use thiserror::Error;

/// Common error type for spectrum operations
#[derive(Debug, Error)]
pub enum SpectrumError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type SpectrumResult<T> = Result<T, SpectrumError>;
