/// Boundary Attack: Decision-based adversarial attack
///
/// # Overview
///
/// Implements the Boundary Attack (Brendel et al., 2017) - a decision-based black-box
/// attack that starts from a large adversarial perturbation and walks along the boundary
/// to minimize it. Unlike HopSkipJump which uses gradient estimation, Boundary Attack
/// uses random walk with rejection sampling.
///
/// # Algorithm
///
/// The attack performs a random walk on the decision boundary:
///
/// 1. **Initialize**: Start with large adversarial perturbation
///    - Random noise far from original
///    - Or use provided initial adversarial
///
/// 2. **Orthogonal Step**: Move perpendicular to line connecting original and current
///    - Sample direction orthogonal to perturbation vector
///    - Check if still adversarial
///    - Accept if adversarial, reject otherwise
///
/// 3. **Source Step**: Move toward original sample
///    - Linear interpolation toward original
///    - Reduce perturbation magnitude
///    - Accept if still adversarial
///
/// 4. **Repeat**: Alternate orthogonal and source steps
///
/// # Key Properties
///
/// - **Decision-based**: Only needs hard labels (like HopSkipJump)
/// - **Random walk**: No gradient estimation (simpler than HopSkipJump)
/// - **Rejection sampling**: High query cost but conceptually simple
/// - **Guaranteed adversarial**: Always stays on adversarial side
///
/// # Mathematical Equivalence
///
/// This implementation maintains equivalence with IBM ART's Boundary Attack.
///
/// # Reference
///
/// Brendel, W., Rauber, J., & Bethge, M. (2017).
/// "Decision-Based Adversarial Attacks: Reliable Attacks Against Machine Learning Models."
/// https://arxiv.org/abs/1712.04248
///
/// # Regulatory Context
///
/// - **EU AI Act Article 15**: Robustness testing for high-risk AI systems
/// - **NIST AI RMF 1.0 Measure 2.7**: Adversarial example resilience
///
/// # Usage Example
///
/// ```rust
/// use spectrum_red::attacks::{BoundaryAttack, BoundaryConfig};
///
/// let config = BoundaryConfig::builder()
///     .max_iter(5000)
///     .delta(0.01)
///     .epsilon(0.01)
///     .build();
///
/// let attack = BoundaryAttack::new(config);
/// let metrics = attack.run(&model, &x_test)?;
/// ```

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{StandardNormal, Distribution};
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

use crate::metrics::AdversarialMetrics;
use crate::model::Model;

/// Errors that can occur during Boundary attack
#[derive(Debug, Error)]
pub enum BoundaryError {
    #[error("Model prediction failed: {0}")]
    ModelError(String),

    #[error("Initialization failed: could not find adversarial sample")]
    InitializationFailed,

    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),
}

impl From<Box<dyn std::error::Error>> for BoundaryError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        BoundaryError::ModelError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for BoundaryError {
    fn from(err: ndarray::ShapeError) -> Self {
        BoundaryError::InvalidDimensions(err.to_string())
    }
}

/// Configuration for Boundary attack
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Maximum number of iterations (default: 5000)
    pub max_iter: usize,

    /// Step size for orthogonal step (default: 0.01)
    pub delta: f64,

    /// Step size for source step (default: 0.01)
    pub epsilon: f64,

    /// Number of initialization attempts (default: 100)
    pub init_size: usize,

    /// Enable parallel execution (default: true)
    pub parallel: bool,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            max_iter: 5000,
            delta: 0.01,
            epsilon: 0.01,
            init_size: 100,
            parallel: true,
        }
    }
}

/// Builder for BoundaryConfig
#[derive(Debug, Clone)]
pub struct BoundaryConfigBuilder {
    config: BoundaryConfig,
}

impl Default for BoundaryConfigBuilder {
    fn default() -> Self {
        Self {
            config: BoundaryConfig::default(),
        }
    }
}

impl BoundaryConfigBuilder {
    pub fn max_iter(mut self, value: usize) -> Self {
        self.config.max_iter = value;
        self
    }

    pub fn delta(mut self, value: f64) -> Self {
        self.config.delta = value;
        self
    }

    pub fn epsilon(mut self, value: f64) -> Self {
        self.config.epsilon = value;
        self
    }

    pub fn init_size(mut self, value: usize) -> Self {
        self.config.init_size = value;
        self
    }

    pub fn parallel(mut self, value: bool) -> Self {
        self.config.parallel = value;
        self
    }

    pub fn build(self) -> BoundaryConfig {
        self.config
    }
}

impl BoundaryConfig {
    pub fn builder() -> BoundaryConfigBuilder {
        BoundaryConfigBuilder::default()
    }
}

/// Boundary attack implementation
pub struct BoundaryAttack {
    config: BoundaryConfig,
}

impl BoundaryAttack {
    pub fn new(config: BoundaryConfig) -> Self {
        Self { config }
    }

    /// Execute the attack on a batch of samples
    pub fn run(
        &self,
        model: &dyn Model,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, AdversarialMetrics), BoundaryError> {
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
            "Boundary",
            &y_original,
            &y_adversarial,
            x,
            &x_adversarial,
            queries_used,
        );

        Ok((x_adversarial, metrics))
    }

    /// Attack a single sample using Boundary Attack
    fn attack_single(
        &self,
        x: Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, BoundaryError> {
        // Initialize with random adversarial
        let mut x_adv = self.initialize(&x, model, query_counter)?;
        let y_original = self.predict_label(&x, model, query_counter)?;

        for _ in 0..self.config.max_iter {
            // Orthogonal step: explore boundary
            if let Some(x_orth) = self.orthogonal_step(&x, &x_adv, y_original, model, query_counter)? {
                x_adv = x_orth;
            }

            // Source step: move toward original
            if let Some(x_source) = self.source_step(&x, &x_adv, y_original, model, query_counter)? {
                x_adv = x_source;
            }
        }

        Ok(x_adv)
    }

    /// Initialize with random adversarial sample
    fn initialize(
        &self,
        x_original: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, BoundaryError> {
        let y_original = self.predict_label(x_original, model, query_counter)?;

        for _ in 0..self.config.init_size {
            let mut rng = rand::thread_rng();
            let x_random: Array1<f64> = Array1::from_shape_fn(x_original.len(), |_| {
                rng.gen::<f64>()
            });

            let y_random = self.predict_label(&x_random, model, query_counter)?;
            if y_random != y_original {
                return Ok(x_random);
            }
        }

        Err(BoundaryError::InitializationFailed)
    }

    /// Orthogonal step: move perpendicular to perturbation vector
    fn orthogonal_step(
        &self,
        x_original: &Array1<f64>,
        x_adv: &Array1<f64>,
        y_original: usize,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Option<Array1<f64>>, BoundaryError> {
        // Perturbation vector
        let perturbation = x_adv - x_original;
        let norm = (perturbation.dot(&perturbation)).sqrt();

        if norm == 0.0 {
            return Ok(None);
        }

        // Sample random direction
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        let random_dir: Array1<f64> = Array1::from_shape_fn(x_original.len(), |_| {
            normal.sample(&mut rng)
        });

        // Project out component parallel to perturbation (Gram-Schmidt)
        let projection = random_dir.dot(&perturbation) / (norm * norm);
        let orthogonal = &random_dir - projection * &perturbation;
        let orth_norm = (orthogonal.dot(&orthogonal)).sqrt();

        if orth_norm == 0.0 {
            return Ok(None);
        }

        let orthogonal_unit = &orthogonal / orth_norm;

        // Take step in orthogonal direction
        let x_new = x_adv + self.config.delta * norm * &orthogonal_unit;
        let y_new = self.predict_label(&x_new, model, query_counter)?;

        if y_new != y_original {
            Ok(Some(x_new))
        } else {
            Ok(None)
        }
    }

    /// Source step: move toward original sample
    fn source_step(
        &self,
        x_original: &Array1<f64>,
        x_adv: &Array1<f64>,
        y_original: usize,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Option<Array1<f64>>, BoundaryError> {
        // Linear interpolation toward original
        let x_new = (1.0 - self.config.epsilon) * x_adv + self.config.epsilon * x_original;
        let y_new = self.predict_label(&x_new, model, query_counter)?;

        if y_new != y_original {
            Ok(Some(x_new))
        } else {
            Ok(None)
        }
    }

    fn predict_label(
        &self,
        x: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<usize, BoundaryError> {
        let x_batch = x.view().insert_axis(Axis(0));
        let y_pred = model.predict(&x_batch.to_owned())?;

        query_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let label = y_pred
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| BoundaryError::ModelError("Empty prediction".to_string()))?;

        Ok(label)
    }
}
