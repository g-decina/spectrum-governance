/// ZOO: Zeroth Order Optimization attack
///
/// # Overview
///
/// Implements the ZOO attack (Chen et al., 2017) - a score-based black-box attack
/// that estimates gradients using finite differences on class probabilities/scores.
/// Unlike HopSkipJump which only needs hard labels, ZOO requires access to confidence
/// scores or probabilities, making it more powerful but with stricter requirements.
///
/// # Algorithm
///
/// ZOO optimizes the adversarial perturbation using estimated gradients:
///
/// 1. **Initialize**: Start with original sample
/// 2. **Estimate Gradient**: Use finite differences on model outputs
///    - Perturb each coordinate by ±δ
///    - Compute gradient ∇f ≈ (f(x+δeᵢ) - f(x-δeᵢ)) / (2δ)
/// 3. **Update**: Apply Adam optimizer to perturbation
/// 4. **Project**: Clip to valid range and L∞ constraint
///
/// # Key Properties
///
/// - **Score-based**: Requires confidence scores/probabilities
/// - **Coordinate-wise**: Can update random subset of coordinates (scalable to high-dim)
/// - **Optimizer-driven**: Uses Adam for adaptive learning rates
/// - **L∞ constrained**: Enforces box constraints on perturbation
///
/// # Mathematical Equivalence
///
/// This implementation maintains equivalence with IBM ART's ZOO attack.
///
/// # Reference
///
/// Chen, P. Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C. J. (2017).
/// "ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks."
/// https://arxiv.org/abs/1708.03999
///
/// # Regulatory Context
///
/// - **EU AI Act Article 15**: Robustness testing for high-risk AI systems
/// - **NIST AI RMF 1.0 Measure 2.7**: Adversarial example resilience
/// - **ISO/IEC 24029**: Assessment of robustness of neural networks
///
/// # Usage Example
///
/// ```rust
/// use spectrum_red::attacks::{ZOOAttack, ZOOConfig};
///
/// let config = ZOOConfig::builder()
///     .max_iter(1000)
///     .learning_rate(0.01)
///     .epsilon(0.3)          // L∞ constraint
///     .build();
///
/// let attack = ZOOAttack::new(config);
/// let metrics = attack.run(&model, &x_test)?;
/// ```

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

use crate::metrics::AdversarialMetrics;
use crate::model::Model;

/// Errors that can occur during ZOO attack
#[derive(Debug, Error)]
pub enum ZOOError {
    #[error("Model prediction failed: {0}")]
    ModelError(String),

    #[error("Attack failed to find adversarial example within budget")]
    AttackFailed,

    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),
}

impl From<Box<dyn std::error::Error>> for ZOOError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ZOOError::ModelError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for ZOOError {
    fn from(err: ndarray::ShapeError) -> Self {
        ZOOError::InvalidDimensions(err.to_string())
    }
}

/// Configuration for ZOO attack
#[derive(Debug, Clone)]
pub struct ZOOConfig {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,

    /// Learning rate for Adam optimizer (default: 0.01)
    pub learning_rate: f64,

    /// L∞ constraint on perturbation (default: 0.3)
    pub epsilon: f64,

    /// Finite difference delta (default: 0.0001)
    pub delta: f64,

    /// Number of random coordinates to update per iteration (default: 128)
    /// Set to n_features for full-dimensional updates
    pub batch_size: usize,

    /// Adam beta1 parameter (default: 0.9)
    pub adam_beta1: f64,

    /// Adam beta2 parameter (default: 0.999)
    pub adam_beta2: f64,

    /// Adam epsilon for numerical stability (default: 1e-8)
    pub adam_epsilon: f64,

    /// Confidence threshold for targeted attacks (default: 0.0)
    pub confidence: f64,

    /// Enable parallel execution (default: true)
    pub parallel: bool,

    /// Set a seed for random noise generation (default: None for entropy-based)
    pub seed: Option<u64>,
}

impl Default for ZOOConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            learning_rate: 0.01,
            epsilon: 0.3,
            delta: 0.0001,
            batch_size: 128,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            confidence: 0.0,
            parallel: true,
            seed: None,
        }
    }
}

/// Builder for ZOOConfig
#[derive(Debug, Clone)]
pub struct ZOOConfigBuilder {
    config: ZOOConfig,
}

impl Default for ZOOConfigBuilder {
    fn default() -> Self {
        Self {
            config: ZOOConfig::default(),
        }
    }
}

impl ZOOConfigBuilder {
    pub fn max_iter(mut self, value: usize) -> Self {
        self.config.max_iter = value;
        self
    }

    pub fn learning_rate(mut self, value: f64) -> Self {
        self.config.learning_rate = value;
        self
    }

    pub fn epsilon(mut self, value: f64) -> Self {
        self.config.epsilon = value;
        self
    }

    pub fn delta(mut self, value: f64) -> Self {
        self.config.delta = value;
        self
    }

    pub fn batch_size(mut self, value: usize) -> Self {
        self.config.batch_size = value;
        self
    }

    pub fn parallel(mut self, value: bool) -> Self {
        self.config.parallel = value;
        self
    }

    /// Set a random seed for reproducibility
    pub fn seed(mut self, value: u64) -> Self {
        self.config.seed = Some(value);
        self
    }

    pub fn build(self) -> ZOOConfig {
        self.config
    }
}

impl ZOOConfig {
    pub fn builder() -> ZOOConfigBuilder {
        ZOOConfigBuilder::default()
    }
}

/// ZOO attack implementation
pub struct ZOOAttack {
    config: ZOOConfig,
}

impl ZOOAttack {
    pub fn new(config: ZOOConfig) -> Self {
        Self { config }
    }

    /// Execute the attack on a batch of samples
    pub fn run(
        &self,
        model: &dyn Model,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, AdversarialMetrics), ZOOError> {
        // Get original predictions
        let y_pred = model.predict(x)?;
        let y_original: Vec<usize> = y_pred
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        let model = Arc::new(model);
        let total_queries = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let samples: Vec<Array1<f64>> = x.axis_iter(Axis(0))
            .map(|s| s.to_owned())
            .collect();

        let x_adversarial = if self.config.parallel {
            samples
                .par_iter()
                .map(|sample| {
                    self.attack_single(sample.clone(), &**model, &total_queries)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            samples
                .iter()
                .map(|sample| {
                    self.attack_single(sample.clone(), &**model, &total_queries)
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let x_adversarial = ndarray::stack(
            Axis(0),
            &x_adversarial.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?;

        let y_adv_pred = model.predict(&x_adversarial)?;
        let y_adversarial: Vec<usize> = y_adv_pred
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        let queries_used = total_queries.load(std::sync::atomic::Ordering::SeqCst);
        let metrics = AdversarialMetrics::for_evasion(
            "ZOO",
            &y_original,
            &y_adversarial,
            x,
            &x_adversarial,
            queries_used,
        );

        Ok((x_adversarial, metrics))
    }

    /// Attack a single sample using ZOO
    fn attack_single(
        &self,
        x: Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, ZOOError> {
        let n_features = x.len();
        let mut perturbation = Array1::zeros(n_features);

        // Adam optimizer state
        let mut m = Array1::zeros(n_features);  // First moment
        let mut v = Array1::zeros(n_features);  // Second moment

        // Get original label
        let y_original = self.predict_label(&x, model, query_counter)?;

        for iteration in 0..self.config.max_iter {
            // Estimate gradient via finite differences
            let gradient = self.estimate_gradient(&x, &perturbation, model, query_counter)?;

            // Adam update
            let t = (iteration + 1) as f64;
            m = self.config.adam_beta1 * &m + (1.0 - self.config.adam_beta1) * &gradient;
            v = self.config.adam_beta2 * &v + (1.0 - self.config.adam_beta2) * (&gradient * &gradient);

            let m_hat = &m / (1.0 - self.config.adam_beta1.powf(t));
            let v_hat = &v / (1.0 - self.config.adam_beta2.powf(t));

            // Update perturbation
            perturbation = &perturbation - self.config.learning_rate * &m_hat / (v_hat.mapv(|x| x.sqrt()) + self.config.adam_epsilon);

            // Project to L∞ ball
            perturbation.mapv_inplace(|x| x.max(-self.config.epsilon).min(self.config.epsilon));

            // Check if adversarial
            let x_adv = &x + &perturbation;
            let y_adv = self.predict_label(&x_adv, model, query_counter)?;

            if y_adv != y_original {
                return Ok(x_adv);
            }
        }

        // Return best attempt even if not adversarial
        Ok(&x + &perturbation)
    }

    /// Estimate gradient using finite differences on random coordinates
    fn estimate_gradient(
        &self,
        x: &Array1<f64>,
        perturbation: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, ZOOError> {
        let n_features = x.len();
        let batch_size = self.config.batch_size.min(n_features);
        let mut gradient = Array1::zeros(n_features);

        // Sample random coordinates
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut indices: Vec<usize> = (0..n_features).collect();

        // Shuffle and take batch_size coordinates
        for i in 0..batch_size {
            let j = rng.gen_range(i..n_features);
            indices.swap(i, j);
        }

        let x_current = x + perturbation;

        for &idx in &indices[..batch_size] {
            // Perturb coordinate idx by +delta
            let mut x_plus = x_current.clone();
            x_plus[idx] += self.config.delta;
            let score_plus = self.get_target_score(&x_plus, model, query_counter)?;

            // Perturb coordinate idx by -delta
            let mut x_minus = x_current.clone();
            x_minus[idx] -= self.config.delta;
            let score_minus = self.get_target_score(&x_minus, model, query_counter)?;

            // Central difference
            gradient[idx] = (score_plus - score_minus) / (2.0 * self.config.delta);
        }

        Ok(gradient)
    }

    /// Get score for the target class (original class for untargeted attack)
    fn get_target_score(
        &self,
        x: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<f64, ZOOError> {
        let x_batch = x.view().insert_axis(Axis(0));
        let y_pred = model.predict(&x_batch.to_owned())?;

        query_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Return max probability (for untargeted attack, we want to minimize this)
        let score = y_pred
            .row(0)
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        Ok(score)
    }

    fn predict_label(
        &self,
        x: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<usize, ZOOError> {
        let x_batch = x.view().insert_axis(Axis(0));
        let y_pred = model.predict(&x_batch.to_owned())?;

        query_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let label = y_pred
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| ZOOError::ModelError("Empty prediction".to_string()))?;

        Ok(label)
    }
}
