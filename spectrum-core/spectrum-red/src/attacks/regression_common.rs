/// Common types and traits for regression adversarial attacks
///
/// This module provides shared infrastructure for all regression attacks.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::error::Error;
use std::fmt;

// ============================================================================
// Confidence Level
// ============================================================================

/// Supported confidence levels for bootstrap CI computation.
///
/// Regression attacks report mean attempts to success with a confidence interval.
/// Users select one of these levels based on their regulatory requirements.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConfidenceLevel {
    /// 98% confidence interval
    Confidence98,
    /// 99% confidence interval
    Confidence99,
    /// 99.5% confidence interval
    Confidence995,
    /// 99.9% confidence interval
    Confidence999,
    /// 99.99% confidence interval
    Confidence9999,
}

impl ConfidenceLevel {
    /// Returns the confidence level as a float (0.98, 0.99, etc.)
    pub fn as_f64(&self) -> f64 {
        match self {
            ConfidenceLevel::Confidence98 => 0.98,
            ConfidenceLevel::Confidence99 => 0.99,
            ConfidenceLevel::Confidence995 => 0.995,
            ConfidenceLevel::Confidence999 => 0.999,
            ConfidenceLevel::Confidence9999 => 0.9999,
        }
    }

    /// Returns the alpha level (1 - confidence) for percentile computation.
    pub fn alpha(&self) -> f64 {
        1.0 - self.as_f64()
    }
}

impl Default for ConfidenceLevel {
    fn default() -> Self {
        ConfidenceLevel::Confidence99
    }
}

// ============================================================================
// Bootstrap CI Computation
// ============================================================================

/// Compute bootstrap confidence interval for the mean of attempt counts.
///
/// Uses the percentile bootstrap method, which is robust and makes no
/// distributional assumptions about the data.
///
/// # Arguments
/// * `attempts` - Slice of attempt counts (one per successful sample)
/// * `confidence` - Confidence level for the interval
/// * `n_bootstrap` - Number of bootstrap resamples (default recommendation: 10000)
///
/// # Returns
/// `(lower_bound, upper_bound)` for the mean attempts. Returns `(NaN, NaN)` if
/// attempts is empty.
///
/// # Algorithm
/// 1. Resample `attempts` with replacement `n_bootstrap` times
/// 2. Compute mean of each resample
/// 3. Return the (alpha/2, 1-alpha/2) percentiles of bootstrap means
pub fn bootstrap_ci_mean(
    attempts: &[usize],
    confidence: ConfidenceLevel,
    n_bootstrap: usize,
) -> (f64, f64) {
    if attempts.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let n = attempts.len();
    let mut rng = rand::thread_rng();

    // Generate bootstrap means
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let sum: usize = (0..n)
            .map(|_| attempts[rng.gen_range(0..n)])
            .sum();
        bootstrap_means.push(sum as f64 / n as f64);
    }

    // Sort for percentile computation
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile indices
    let alpha = confidence.alpha();
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    // Clamp indices to valid range
    let lower_idx = lower_idx.min(n_bootstrap - 1);
    let upper_idx = upper_idx.min(n_bootstrap - 1);

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

// ============================================================================
// Regression Model Trait
// ============================================================================

/// Trait for regression models that output continuous values.
///
/// This extends the classifier Model trait for regression-specific operations.
pub trait RegressionModel: Sync + Send {
    /// Predict continuous values for a batch of inputs.
    ///
    /// # Arguments
    /// * `inputs` - A 2D array of shape (n_samples, n_features).
    ///
    /// # Returns
    /// * `Result<Array1<f64>>` - A 1D array of shape (n_samples,) containing predictions.
    fn predict(&self, inputs: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

    /// Returns the number of input features expected by the model.
    fn input_shape(&self) -> usize;

    /// Optionally return prediction intervals (for conformal prediction models).
    /// Returns (lower_bounds, upper_bounds) for each sample.
    fn predict_interval(
        &self,
        _inputs: &Array2<f64>,
    ) -> Result<Option<(Array1<f64>, Array1<f64>)>, Box<dyn Error>> {
        // Default: no interval prediction
        Ok(None)
    }
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum RegressionAttackError {
    ModelError(String),
    InvalidConfiguration(String),
    ConvergenceFailed(String),
    InsufficientSamples(String),
}

impl fmt::Display for RegressionAttackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionAttackError::ModelError(msg) => write!(f, "Model error: {}", msg),
            RegressionAttackError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            RegressionAttackError::ConvergenceFailed(msg) => {
                write!(f, "Convergence failed: {}", msg)
            }
            RegressionAttackError::InsufficientSamples(msg) => {
                write!(f, "Insufficient samples: {}", msg)
            }
        }
    }
}

impl Error for RegressionAttackError {}

// ============================================================================
// Regression Metrics
// ============================================================================

/// Metrics specific to regression adversarial attacks.
///
/// For OutputManipulation and PredictionShift attacks, the primary metric is
/// **mean attempts to success** rather than attack success rate. This reflects
/// the reality that regression models are inherently vulnerable - the question
/// is not *if* an attack succeeds, but *how easily*.
///
/// Samples that never succeed (within query/iteration budget) are excluded from
/// the mean and reported separately in `samples_never_succeeded`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegressionMetrics {
    // Attack info
    pub attack_type: String,
    pub samples_tested: usize,

    // Attempts-based metrics (primary for OutputManipulation/PredictionShift)
    /// Number of samples that succeeded within the budget
    pub samples_succeeded: usize,
    /// Number of samples that never succeeded within the budget
    pub samples_never_succeeded: usize,
    /// Mean number of attempts (queries or directions) to achieve success.
    /// Computed only over samples that succeeded.
    pub mean_attempts_to_success: f64,
    /// Median attempts to success (over successful samples)
    pub median_attempts_to_success: f64,
    /// Minimum attempts to success
    pub min_attempts: Option<usize>,
    /// Maximum attempts to success
    pub max_attempts: Option<usize>,
    /// Bootstrap confidence interval on mean attempts: (lower, upper)
    pub attempts_ci: (f64, f64),
    /// Confidence level used for the CI (0.98, 0.99, 0.995, 0.999, 0.9999)
    pub confidence_level: f64,

    // Legacy field for backward compatibility / QuantileAttack
    /// Attack success rate (deprecated for OutputManipulation/PredictionShift,
    /// still used by QuantileAttack)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attack_success_rate: Option<f64>,

    // Perturbation metrics
    pub mean_perturbation_l2: f64,
    pub mean_perturbation_linf: f64,
    pub min_perturbation_l2: f64,
    pub max_perturbation_l2: f64,
    pub median_perturbation_l2: f64,

    // Prediction shift metrics
    pub mean_prediction_shift: f64,
    pub max_prediction_shift: f64,
    pub mean_absolute_shift: f64,

    // Query metrics
    pub queries_used: usize,
    pub avg_queries_per_sample: f64,

    // For output manipulation
    pub bin_flip_rate: Option<f64>,
    pub mean_bin_distance: Option<f64>,

    // For quantile attack
    pub coverage_break_rate: Option<f64>,
    pub mean_interval_width_increase: Option<f64>,
    pub original_coverage: Option<f64>,
    pub adversarial_coverage: Option<f64>,
}

impl RegressionMetrics {
    /// Compute basic perturbation statistics from original and adversarial inputs.
    pub fn compute_perturbation_stats(
        x_orig: &Array2<f64>,
        x_adv: &Array2<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        use ndarray::Axis;

        let perturbations = x_adv - x_orig;
        let l2_norms: Vec<f64> = perturbations
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|v| v.powi(2)).sum().sqrt())
            .collect();
        let linf_norms: Vec<f64> = perturbations
            .axis_iter(Axis(0))
            .map(|row| row.mapv(|v| v.abs()).fold(0.0f64, |a, &b| a.max(b)))
            .collect();

        (l2_norms, linf_norms)
    }

    /// Helper to sort and extract statistics from L2 norms.
    pub fn l2_stats(l2_norms: &[f64]) -> (f64, f64, f64, f64) {
        let n = l2_norms.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mean = l2_norms.iter().sum::<f64>() / n as f64;
        let mut sorted = l2_norms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        (
            mean,
            sorted.first().copied().unwrap_or(0.0),
            sorted.last().copied().unwrap_or(0.0),
            sorted.get(n / 2).copied().unwrap_or(0.0),
        )
    }
}
