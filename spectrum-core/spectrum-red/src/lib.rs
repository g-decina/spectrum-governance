/// spectrum-red: Adversarial attacks for model robustness testing
///
/// This crate implements pure Rust adversarial attacks with parallelism support.
/// All attacks are designed for mathematical equivalence with IBM ART.

pub mod attacks;
pub mod metrics;
pub mod model;
pub mod scenarios;

#[cfg(feature = "onnx")]
pub mod onnx_model;

// Re-export key types for convenience
pub use metrics::AdversarialMetrics;
pub use model::Model;

#[cfg(feature = "onnx")]
pub use onnx_model::OnnxModel;

// Python bindings module (compiled conditionally when building Python extension)
#[cfg(feature = "python")]
mod python_bindings;

// Re-export the PyO3 module initialization when building for Python
#[cfg(feature = "python")]
pub use python_bindings::*;
