/// ONNX Model wrapper using onnxruntime-rs
///
/// This module provides ONNX model inference using Microsoft's ONNX Runtime,
/// which supports ALL ONNX operators including sklearn's ML operators
/// (LinearClassifier, TreeEnsembleClassifier, etc.).
///
/// KEY FEATURES:
/// - Full ONNX operator support (ai.onnx and ai.onnx.ml domains)
/// - Pure Rust inference (no Python calls, no GIL!)
/// - Optimized execution with graph optimization
/// - Compatible with sklearn, PyTorch, TensorFlow models

#[cfg(feature = "onnx")]
use crate::model::Model;
#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use std::error::Error;
#[cfg(feature = "onnx")]
use std::sync::Mutex;
#[cfg(feature = "onnx")]
use thiserror::Error;

#[cfg(feature = "onnx")]
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("Failed to load ONNX model: {0}")]
    LoadError(String),

    #[error("Model inference failed: {0}")]
    InferenceError(String),

    #[error("Output shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

#[cfg(feature = "onnx")]
pub struct OnnxModel {
    session: Mutex<Session>,  // Mutex for interior mutability (session.run() needs &mut)
    input_name: String,
    output_name: String,
    input_shape: usize,
    num_classes: usize,
}

#[cfg(feature = "onnx")]
impl OnnxModel {
    /// Create ONNX model from serialized bytes
    ///
    /// # Arguments
    /// * `onnx_bytes` - Serialized ONNX model (from skl2onnx, torch.onnx, etc.)
    /// * `input_shape` - Number of input features
    /// * `num_classes` - Number of output classes
    ///
    /// # Returns
    /// * `Ok(OnnxModel)` - Ready for inference
    /// * `Err(OnnxError)` - If model loading fails
    pub fn from_bytes(
        onnx_bytes: &[u8],
        input_shape: usize,
        num_classes: usize,
    ) -> Result<Self, OnnxError> {
        // Create ONNX Runtime session with optimizations
        let session = Session::builder()
            .map_err(|e| OnnxError::LoadError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxError::LoadError(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(1) // Single thread per session (parallelism at attack level)
            .map_err(|e| OnnxError::LoadError(format!("Failed to set thread count: {}", e)))?
            .commit_from_memory(onnx_bytes)
            .map_err(|e| OnnxError::LoadError(format!("Failed to load ONNX model from bytes: {}", e)))?;

        // Get input/output names from the model
        let input_name = session
            .inputs
            .get(0)
            .ok_or_else(|| OnnxError::LoadError("Model has no inputs".to_string()))?
            .name
            .clone();

        // For sklearn models, output index 1 is probabilities (index 0 is labels)
        // For other models, there may only be one output
        let output_idx = if session.outputs.len() > 1 { 1 } else { 0 };
        let output_name = session
            .outputs
            .get(output_idx)
            .ok_or_else(|| OnnxError::LoadError("Model has no outputs".to_string()))?
            .name
            .clone();

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            input_shape,
            num_classes,
        })
    }
}

#[cfg(feature = "onnx")]
impl Model for OnnxModel {
    fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let batch_size = inputs.nrows();

        // Convert f64 to f32 (ONNX standard precision)
        let inputs_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();

        // Create input tensor
        let input_shape = vec![batch_size, self.input_shape];
        let input_tensor = ndarray::Array2::from_shape_vec(
            (batch_size, self.input_shape),
            inputs_f32,
        )?;

        // Run inference
        // Note: ort 2.0 requires creating Values from owned arrays
        use ort::value::Value;
        let input_value = Value::from_array((input_tensor.shape().to_vec(), input_tensor.into_raw_vec()))
            .map_err(|e| OnnxError::InferenceError(format!("Failed to create input tensor: {}", e)))?;

        // Lock the session mutex to get mutable access
        let mut session = self.session.lock()
            .map_err(|e| OnnxError::InferenceError(format!("Failed to lock session: {}", e)))?;

        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => input_value])
            .map_err(|e| OnnxError::InferenceError(format!("ONNX inference failed: {}", e)))?;

        // Extract output tensor
        let output_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| {
                OnnxError::InferenceError(format!("Output '{}' not found", self.output_name))
            })?;

        // Convert to f64 Array2
        // ort 2.0 returns (&Shape, &[T]) from try_extract_tensor
        let (output_shape_ref, output_data) = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| OnnxError::InferenceError(format!("Failed to extract output tensor: {}", e)))?;

        // Get the shape dimensions as slice
        let output_shape = output_shape_ref.as_ref();

        // Validate output shape
        if output_shape.len() != 2 {
            return Err(Box::new(OnnxError::ShapeMismatch {
                expected: format!("2D array [{}, {}]", batch_size, self.num_classes),
                actual: format!("{}D array {:?}", output_shape.len(), output_shape),
            }));
        }

        if output_shape[0] as usize != batch_size {
            return Err(Box::new(OnnxError::ShapeMismatch {
                expected: format!("batch_size={}", batch_size),
                actual: format!("batch_size={}", output_shape[0]),
            }));
        }

        // Convert f32 output to f64
        let output_f64: Vec<f64> = output_data.iter().map(|&x| x as f64).collect();
        let result = Array2::from_shape_vec(
            (output_shape[0] as usize, output_shape[1] as usize),
            output_f64
        )?;

        Ok(result)
    }

    fn input_shape(&self) -> usize {
        self.input_shape
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[cfg(test)]
#[cfg(feature = "onnx")]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_model_creation() {
        // This is a placeholder test - real ONNX bytes would be needed
        // for a proper test. Integration tests should use actual sklearn models.
        assert!(true, "ONNX model module compiles");
    }
}
