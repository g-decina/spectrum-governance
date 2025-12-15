use ndarray::Array2;
use std::error::Error;

/// Defines a model that outputs probability distributions over classes.
/// 
/// This trait abstracts the backend (e.g., ONNX, PyTorch, pure Rust) from the attack logic.
pub trait Model: Sync + Send {
    /// Predict class probabilities for a batch of inputs.
    /// 
    /// # Arguments
    /// * `inputs` - A 2D array of shape (n_samples, n_features).
    /// 
    /// # Returns
    /// * `Result<Array2<f64>>` - A 2D array of shape (n_samples, n_classes) containing probabilities.
    fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;

    /// Returns the number of input features expected by the model.
    fn input_shape(&self) -> usize;
    
    /// Returns the number of output classes.
    fn num_classes(&self) -> usize;
}