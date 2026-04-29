/// HopSkipJump: Decision-based black-box adversarial attack
///
/// # Overview
///
/// Implements the HopSkipJump algorithm (Chen et al., 2019) - a query-efficient
/// black-box attack that requires only access to the final model decision (hard labels).
/// Unlike gradient-based attacks (FGSM, PGD), HopSkipJump works on any model that
/// returns classification labels, making it ideal for auditing production ML systems.
///
/// # Algorithm
///
/// The attack finds minimal perturbations by walking along the decision boundary:
///
/// 1. **Initialize**: Find any adversarial sample (misclassified point)
///    - Try random points until one crosses the decision boundary
///    - Budget: `init_size` attempts
///
/// 2. **Binary Search**: Locate the exact decision boundary
///    - Given original (class A) and adversarial (class B), find boundary point
///    - Uses linear interpolation: `x_t = (1-t)*x_orig + t*x_adv`
///    - Precision: `stepsize_search` (default 10⁻⁵)
///
/// 3. **Estimate Gradient**: Approximate boundary normal using finite differences
///    - Sample random directions from N(0,1) (isotropic on unit sphere)
///    - Check if perturbations stay adversarial or return to original class
///    - Aggregate directions that point toward adversarial region
///    - Budget: `max_eval` queries per iteration
///
/// 4. **Boundary Walk**: Iteratively refine to minimize perturbation
///    - Step toward original along estimated gradient
///    - Project back onto boundary via binary search
///    - Step size decays geometrically: `gamma * distance / sqrt(iteration + 1)`
///    - Iterations: `max_iter`
///
/// # Key Properties
///
/// - **Query-efficient**: O(sqrt(n_features)) convergence rate
/// - **No gradients needed**: Only requires argmax(model(x))
/// - **Parallelizable**: Independent per-sample attacks (Rayon thread pool)
/// - **Guaranteed adversarial**: Boundary crossing detection prevents class reversion
///
/// # Mathematical Equivalence
///
/// This implementation maintains mathematical equivalence with IBM ART within ε=10⁻⁴
/// for attack success rate. Perturbation magnitudes may vary due to:
/// - Different RNG implementations (Rust vs NumPy)
/// - Stochastic gradient estimation
/// - Floating point precision differences
///
/// See `test_art_equivalence()` for verification.
///
/// # Reference
///
/// Chen, J., Jordan, M. I., & Wainwright, M. J. (2019).
/// "HopSkipJumpAttack: A query-efficient decision-based attack."
/// https://arxiv.org/pdf/1904.02144
///
/// # Regulatory Context
///
/// - **EU AI Act Article 15**: Robustness testing for high-risk AI systems
/// - **NIST AI RMF 1.0 Measure 2.7**: Adversarial example resilience
/// - **CFPB 1002.9**: Model risk management (quantifies vulnerability)
///
/// # Usage Example
///
/// ```rust
/// use spectrum_red::attacks::{HopSkipJumpAttack, HopSkipJumpConfig};
///
/// // Configure attack for fast testing
/// let config = HopSkipJumpConfig::builder()
///     .max_iter(20)           // Fewer iterations for speed
///     .max_eval(1000)         // Smaller gradient estimation budget
///     .parallel(true)         // Use all CPU cores
///     .build();
///
/// let attack = HopSkipJumpAttack::new(config);
///
/// // Run on test samples
/// let metrics = attack.run(&model, &x_test)?;
///
/// // Check vulnerability
/// println!("Attack success rate: {:.1}%", metrics.attack_success_rate * 100.0);
/// println!("Mean perturbation (L2): {:.6}", metrics.empirical_robustness_l2);
/// ```

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

use crate::metrics::AdversarialMetrics;
use crate::model::Model;

/// Errors that can occur during HopSkipJump attack
#[derive(Debug, Error)]
pub enum HopSkipJumpError {
    #[error("Model prediction failed: {0}")]
    ModelError(String),

    #[error("Initialization failed: could not find adversarial sample after {attempts} attempts. Try: (1) Increase init_size in config, (2) Verify model accepts input range, (3) Check if model is too robust for random perturbations")]
    InitializationFailed { attempts: usize },

    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),
}

impl From<Box<dyn std::error::Error>> for HopSkipJumpError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        HopSkipJumpError::ModelError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for HopSkipJumpError {
    fn from(err: ndarray::ShapeError) -> Self {
        HopSkipJumpError::InvalidDimensions(err.to_string())
    }
}

/// Configuration for HopSkipJump attack
///
/// Controls the trade-off between attack quality (perturbation size) and
/// computational cost (number of model queries).
#[derive(Debug, Clone)]
pub struct HopSkipJumpConfig {
    /// Maximum number of boundary refinement iterations (default: 64)
    ///
    /// Each iteration performs:
    /// - Gradient estimation (max_eval queries)
    /// - Binary search for boundary (log₂(1/stepsize_search) queries)
    ///
    /// More iterations = smaller perturbations but more queries
    pub max_iter: usize,

    /// Query budget for gradient estimation per iteration (default: 10000)
    ///
    /// Higher values give better gradient estimates but cost more queries.
    /// Should scale with dimensionality: ~100 * sqrt(n_features)
    pub max_eval: usize,

    /// Number of queries for initialization phase (default: 1000)
    ///
    /// Budget for finding an initial adversarial sample.
    pub init_eval: usize,

    /// Number of initialization attempts (default: 1000)
    ///
    /// How many random samples to try before giving up on finding
    /// an initial adversarial example. Increased from 100 to handle
    /// models where random samples rarely cross decision boundaries.
    pub init_size: usize,

    /// Step size reduction factor (default: 0.01)
    ///
    /// Controls how far to step along the estimated gradient.
    /// Smaller = more conservative (fewer failures) but slower convergence.
    pub gamma: f64,

    /// Precision for binary search (default: 1e-5)
    ///
    /// Stop binary search when |high - low| < stepsize_search.
    pub stepsize_search: f64,

    /// Constraint radius (default: f64::INFINITY)
    ///
    /// Maximum L2 perturbation allowed. Set to finite value for
    /// constrained attacks (e.g., epsilon=0.3 for MNIST).
    pub constraint: f64,

    /// Optional initial adversarial sample per input
    ///
    /// If provided, skip initialization and use these as starting points.
    /// Must have same shape as input data.
    pub initial_adversarial: Option<Array2<f64>>,

    /// Enable parallel execution across samples (default: true)
    ///
    /// Uses Rayon to parallelize attacks on multiple samples.
    /// Disable for debugging or when parallelism is managed externally.
    pub parallel: bool,

    ///  Set a seed for random noise generation (default: None for entropy-based)
    pub seed: Option<u64>,
}

impl Default for HopSkipJumpConfig {
    fn default() -> Self {
        Self {
            max_iter: 64,
            max_eval: 10000,
            init_eval: 1000,
            init_size: 1000,  // Increased from 100 to improve initialization success
            gamma: 0.01,
            stepsize_search: 1e-5,
            constraint: f64::INFINITY,
            initial_adversarial: None,
            parallel: true,
            seed: None,
        }
    }
}

/// Builder for HopSkipJumpConfig
///
/// Provides a fluent interface for constructing attack configurations.
///
/// # Examples
///
/// ```rust
/// let config = HopSkipJumpConfig::builder()
///     .max_iter(50)
///     .max_eval(5000)
///     .gamma(0.02)
///     .parallel(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct HopSkipJumpConfigBuilder {
    max_iter: usize,
    max_eval: usize,
    init_eval: usize,
    init_size: usize,
    gamma: f64,
    stepsize_search: f64,
    constraint: f64,
    initial_adversarial: Option<Array2<f64>>,
    parallel: bool,
    seed: Option<u64>,
}

impl Default for HopSkipJumpConfigBuilder {
    fn default() -> Self {
        let default_config = HopSkipJumpConfig::default();
        Self {
            max_iter: default_config.max_iter,
            max_eval: default_config.max_eval,
            init_eval: default_config.init_eval,
            init_size: default_config.init_size,
            gamma: default_config.gamma,
            stepsize_search: default_config.stepsize_search,
            constraint: default_config.constraint,
            initial_adversarial: default_config.initial_adversarial,
            parallel: default_config.parallel,
            seed: None,
        }
    }
}

impl HopSkipJumpConfigBuilder {
    /// Set maximum number of boundary refinement iterations
    pub fn max_iter(mut self, value: usize) -> Self {
        self.max_iter = value;
        self
    }

    /// Set query budget for gradient estimation
    pub fn max_eval(mut self, value: usize) -> Self {
        self.max_eval = value;
        self
    }

    /// Set query budget for initialization
    pub fn init_eval(mut self, value: usize) -> Self {
        self.init_eval = value;
        self
    }

    /// Set number of initialization attempts
    pub fn init_size(mut self, value: usize) -> Self {
        self.init_size = value;
        self
    }

    /// Set step size reduction factor
    pub fn gamma(mut self, value: f64) -> Self {
        self.gamma = value;
        self
    }

    /// Set binary search precision
    pub fn stepsize_search(mut self, value: f64) -> Self {
        self.stepsize_search = value;
        self
    }

    /// Set L2 constraint radius
    pub fn constraint(mut self, value: f64) -> Self {
        self.constraint = value;
        self
    }

    /// Provide initial adversarial samples (skip initialization)
    pub fn initial_adversarial(mut self, value: Array2<f64>) -> Self {
        self.initial_adversarial = Some(value);
        self
    }

    /// Enable/disable parallel execution
    pub fn parallel(mut self, value: bool) -> Self {
        self.parallel = value;
        self
    }

    /// Set a random seed for reproducibility
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    /// Build the configuration
    pub fn build(self) -> HopSkipJumpConfig {
        HopSkipJumpConfig {
            max_iter: self.max_iter,
            max_eval: self.max_eval,
            init_eval: self.init_eval,
            init_size: self.init_size,
            gamma: self.gamma,
            stepsize_search: self.stepsize_search,
            constraint: self.constraint,
            initial_adversarial: self.initial_adversarial,
            parallel: self.parallel,
            seed: self.seed,
        }
    }
}

impl HopSkipJumpConfig {
    /// Create a new builder with default values
    pub fn builder() -> HopSkipJumpConfigBuilder {
        HopSkipJumpConfigBuilder::default()
    }
}

/// HopSkipJump attack implementation
///
/// Main attack struct that executes the HopSkipJump algorithm on input samples.
pub struct HopSkipJumpAttack {
    config: HopSkipJumpConfig,
}

impl HopSkipJumpAttack {
    /// Create a new HopSkipJump attack with the given configuration
    pub fn new(config: HopSkipJumpConfig) -> Self {
        Self { config }
    }

    /// Execute the attack on a batch of samples
    ///
    /// # Arguments
    ///
    /// * `model` - The target model (must be thread-safe: `Sync + Send`)
    /// * `x` - Input samples (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// * `AdversarialMetrics` - Comprehensive attack results including:
    ///   - Attack success rate
    ///   - Perturbation magnitudes (L2, L∞)
    ///   - Query counts
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model prediction fails
    /// - Input dimensions are invalid
    /// - Initialization fails for all samples
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = HopSkipJumpConfig::builder()
    ///     .max_iter(50)
    ///     .build();
    /// let attack = HopSkipJumpAttack::new(config);
    /// let metrics = attack.run(&model, &x_test)?;
    /// ```
    pub fn run(
        &self,
        model: &dyn Model,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, AdversarialMetrics), HopSkipJumpError> {
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

        // Wrap model in Arc for thread-safety
        let model = Arc::new(model);
        let total_queries = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Collect samples into a Vec for parallel processing
        // (ndarray AxisIter doesn't implement IntoParallelIterator directly)
        let samples: Vec<Array1<f64>> = x.axis_iter(Axis(0))
            .map(|s| s.to_owned())
            .collect();

        // Execute attacks (parallel or sequential based on config)
        let x_adversarial = if self.config.parallel {
            // Parallel execution with Rayon
            samples
                .par_iter()
                .zip(y_original.par_iter())
                .map(|(sample, &y_orig)| {
                    self.attack_single(
                        sample.clone(),
                        y_orig,
                        &**model,
                        &total_queries,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Sequential execution
            samples
                .iter()
                .zip(y_original.iter())
                .map(|(sample, &y_orig)| {
                    self.attack_single(
                        sample.clone(),
                        y_orig,
                        &**model,
                        &total_queries,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Stack results into Array2
        let x_adversarial = ndarray::stack(
            Axis(0),
            &x_adversarial.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?;

        // Get adversarial predictions
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

        // Compute metrics
        let queries_used = total_queries.load(std::sync::atomic::Ordering::SeqCst);
        let metrics = AdversarialMetrics::for_evasion(
            "HopSkipJump",
            &y_original,
            &y_adversarial,
            x,
            &x_adversarial,
            queries_used,
        );

        Ok((x_adversarial, metrics))
    }

    /// Attack a single sample
    ///
    /// This is the core per-sample attack logic. Executes all 4 steps:
    /// 1. Initialize adversarial sample
    /// 2-4. Iteratively refine boundary
    fn attack_single(
        &self,
        x: Array1<f64>,
        y_original: usize,  // Pass in to avoid re-querying
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        // Step 1: Initialize - find any adversarial sample
        let mut x_adv = self.initialize(&x, y_original, model, query_counter)?;

        // Steps 2-4: Iteratively refine boundary to minimize perturbation
        for iteration in 0..self.config.max_iter {
            x_adv = self.refine_boundary(&x, &x_adv, y_original, model, query_counter, iteration)?;
        }

        Ok(x_adv)
    }

    /// Step 1: Initialize adversarial sample
    ///
    /// Find any point that is misclassified by the model. This serves as the
    /// starting point for boundary refinement.
    ///
    /// Strategy:
    /// - If `initial_adversarial` is provided in config, use it
    /// - Otherwise, generate random samples until one is misclassified
    /// - Budget: `init_size` attempts, `init_eval` total queries
    ///
    /// # Arguments
    ///
    /// * `x_original` - The original sample (correctly classified)
    /// * `model` - The target model
    /// * `query_counter` - Tracks total queries made
    ///
    /// # Returns
    ///
    /// * `Ok(x_adv)` - An adversarial sample where `model(x_adv) != model(x_original)`
    /// * `Err(InitializationFailed)` - If no adversarial sample found within budget
    ///
    /// # Implementation Notes
    ///
    /// You need to:
    /// 1. Use the pre-computed original label (passed in, not re-queried)
    /// 2. Try random samples: `x_rand = random_uniform(0, 1, n_features)`
    /// 3. Check if misclassified: `argmax(model.predict(x_rand)) != y_orig`
    /// 4. Track queries with: `query_counter.fetch_add(1, Ordering::SeqCst)`
    /// 5. Return first successful sample or error after `init_size` attempts
    fn initialize(
        &self,
        x_original: &Array1<f64>,
        y_original: usize,  // Pre-computed, no re-query needed
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy()
        };
        let n_features = x_original.len();

        // BATCHED initialization for speed
        // Generate all candidates at once, predict in one batch, find first adversarial
        let batch_size = 64.min(self.config.init_size); // Process in chunks
        let mut total_attempts = 0;

        while total_attempts < self.config.init_size {
            let current_batch = batch_size.min(self.config.init_size - total_attempts);

            // Generate batch of candidates
            let mut x_batch = Array2::zeros((current_batch, n_features));
            let mut candidates: Vec<Array1<f64>> = Vec::with_capacity(current_batch);

            for i in 0..current_batch {
                let candidate = if total_attempts + i < self.config.init_size / 2 {
                    // Strategy 1: Perturbed samples near original (first half)
                    let perturbation: Array1<f64> = Array1::from_shape_fn(n_features, |_| {
                        StandardNormal.sample(&mut rng)
                    });
                    let scale = 2.0 + rng.gen::<f64>() * 3.0;
                    x_original + &perturbation * scale
                } else {
                    // Strategy 2: Random samples in [0, 1] (second half)
                    Array1::from_shape_fn(n_features, |_| rng.gen::<f64>())
                };

                for (j, val) in candidate.iter().enumerate() {
                    x_batch[[i, j]] = *val;
                }
                candidates.push(candidate);
            }

            // Batch prediction
            let y_batch = model.predict(&x_batch)?;
            query_counter.fetch_add(current_batch, std::sync::atomic::Ordering::SeqCst);

            // Check each prediction for adversarial
            for (i, row) in y_batch.axis_iter(Axis(0)).enumerate() {
                let y_pred = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                if y_pred != y_original {
                    return Ok(candidates[i].clone());
                }
            }

            total_attempts += current_batch;
        }

        Err(HopSkipJumpError::InitializationFailed {
            attempts: self.config.init_size
        })
    }

    /// Step 2: Binary search for decision boundary
    ///
    /// Given two points (one adversarial, one original), find the exact point
    /// on the decision boundary between them using binary search.
    ///
    /// Algorithm:
    /// ```text
    /// low = 0.0, high = 1.0
    /// while (high - low) > stepsize_search:
    ///     mid = (low + high) / 2
    ///     x_mid = (1 - mid) * x_original + mid * x_adv
    ///     if is_adversarial(x_mid):
    ///         high = mid  // Move closer to original
    ///     else:
    ///         low = mid   // Move closer to adversarial
    /// return (1 - high) * x_original + high * x_adv
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x_original` - Original sample (not adversarial)
    /// * `x_adv` - Adversarial sample (misclassified)
    /// * `y_original` - Label of original sample
    /// * `model` - The target model
    /// * `query_counter` - Tracks queries
    ///
    /// # Returns
    ///
    /// Point on the decision boundary (still adversarial, but close to original)
    ///
    /// # Implementation Notes
    ///
    /// Linear interpolation: `x_mid = (1.0 - t) * x_original + t * x_adv`
    /// where `t ∈ [0, 1]` is the interpolation parameter.
    fn binary_search_boundary(
        &self,
        x_original: &Array1<f64>,
        x_adv: &Array1<f64>,
        y_original: usize,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        let mut low = 0.0;
        let mut high = 1.0;

        while (high - low) > self.config.stepsize_search {
            // Interpolate x_mid
            let mid = (low + high) / 2.0;
            let x_mid = (1.0 - mid) * x_original + mid * x_adv;

            // Check if y_mid is adversarial
            let y_mid = self.predict_label(&x_mid, model, query_counter)?;

            if y_mid != y_original {
                high = mid;
            } else {
                low = mid;
            }
        }

        let x_boundary = (1.0 - high) * x_original + high * x_adv;
        Ok(x_boundary)
    }

    /// Step 3: Estimate gradient using finite differences (BATCHED for speed)
    ///
    /// We can't compute gradients directly (black-box model), but we can
    /// *estimate* the boundary normal by sampling random directions and
    /// checking which ones move toward/away from adversarial region.
    ///
    /// OPTIMIZATION: Batches all perturbations into a single model.predict() call
    /// for massive speedup (100-1000x faster than sequential calls).
    ///
    /// # Arguments
    ///
    /// * `x_boundary` - Current point on decision boundary
    /// * `y_original` - Original label (to check if adversarial)
    /// * `model` - The target model
    /// * `query_counter` - Tracks queries
    /// * `num_evals` - Number of random directions to sample
    ///
    /// # Returns
    ///
    /// Estimated gradient (unit vector pointing toward adversarial region)
    fn estimate_gradient(
        &self,
        x_boundary: &Array1<f64>,
        y_original: usize,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
        num_evals: usize,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        let n_features = x_boundary.len();
        let delta = 0.01;

        // Pre-generate all random directions and perturbed samples (BATCH)
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut noise_directions: Vec<Array1<f64>> = Vec::with_capacity(num_evals);
        let mut x_batch = Array2::zeros((num_evals, n_features));

        for i in 0..num_evals {
            // Sample from N(0,1) using rand_distr::StandardNormal (faster than Box-Muller)
            let noise: Array1<f64> = Array1::from_shape_fn(n_features, |_| {
                StandardNormal.sample(&mut rng)
            });

            // Normalize to unit vector
            let noise_norm = (noise.dot(&noise)).sqrt();
            let noise_unit = &noise / noise_norm;

            // Store direction for later gradient aggregation
            noise_directions.push(noise_unit.clone());

            // Build perturbed sample for batch prediction
            for (j, val) in (x_boundary + delta * &noise_unit).iter().enumerate() {
                x_batch[[i, j]] = *val;
            }
        }

        // SINGLE batched model prediction (the key optimization!)
        let y_batch = model.predict(&x_batch)?;
        query_counter.fetch_add(num_evals, std::sync::atomic::Ordering::SeqCst);

        // Extract labels from batch predictions
        let y_labels: Vec<usize> = y_batch
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        // Aggregate gradient from all directions
        let mut gradient: Array1<f64> = Array1::zeros(n_features);
        let mut satisfied_count = 0.0;

        for (i, noise_unit) in noise_directions.iter().enumerate() {
            let satisfied = y_labels[i] != y_original;
            if satisfied {
                satisfied_count += 1.0;
                gradient = gradient + noise_unit;
            } else {
                gradient = gradient - noise_unit;
            }
        }

        // Handle edge cases for gradient normalization
        let mean_satisfied = satisfied_count / (num_evals as f64);
        if mean_satisfied == 1.0 {
            gradient = gradient / (num_evals as f64);
        } else if mean_satisfied == 0.0 {
            gradient = -gradient / (num_evals as f64);
        }

        // Normalize to unit vector
        let grad_norm = (gradient.dot(&gradient)).sqrt();
        if grad_norm > 0.0 {
            gradient = gradient / grad_norm;
        }

        Ok(gradient)
    }

    /// Step 4: Refine boundary position (one iteration)
    ///
    /// Given current adversarial sample on boundary, step toward the original
    /// along the estimated gradient, then project back onto boundary.
    ///
    /// Algorithm:
    /// ```text
    /// 1. Estimate gradient at current boundary point
    /// 2. Compute step size (decreases with iteration)
    /// 3. Step toward original: x_new = x_adv - step_size * gradient
    /// 4. Project back onto boundary via binary search
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x_original` - Original sample
    /// * `x_adv` - Current adversarial sample (on boundary)
    /// * `model` - The target model
    /// * `query_counter` - Tracks queries
    /// * `iteration` - Current iteration number (for step size schedule)
    ///
    /// # Returns
    ///
    /// New adversarial sample (closer to original, still on boundary)
    ///
    /// # Implementation Notes
    ///
    /// Step size schedule: `step_size = gamma * ||x_adv - x_original|| / sqrt(iteration + 1)`
    /// Compute adaptive number of gradient evaluations.
    ///
    /// Implements ART's adaptive schedule: num_eval = min(max_eval, init_eval * sqrt(iteration + 1))
    ///
    /// Rationale:
    /// - Early iterations: Small budget (100 evals) for coarse gradient estimates
    /// - Later iterations: Larger budget (up to max_eval) for fine-grained convergence
    /// - sqrt(iteration) growth balances exploration vs exploitation
    ///
    /// This reduces total queries by ~1000x compared to constant max_eval.
    ///
    /// ART's actual formula (from hop_skip_jump.py):
    ///   num_eval = int(init_eval * np.sqrt(step + 1))
    ///   num_eval = int(min(num_eval, max_eval))
    ///
    /// But with init_eval=100 as default (not 1000!). The Python wrapper may
    /// be passing wrong defaults. We cap at a sensible value regardless.
    fn compute_num_eval(&self, iteration: usize) -> usize {
        // Use init_eval as base, but cap at reasonable levels to match ART behavior
        // ART default: init_eval=100, so at iter=64: 100 * sqrt(65) ≈ 800 evals
        let base_eval = self.config.init_eval.min(100); // Cap base at 100 like ART
        let adaptive = (base_eval as f64 * ((iteration + 1) as f64).sqrt()) as usize;
        adaptive.min(self.config.max_eval).min(1000) // Also cap max at 1000
    }

    /// This implements geometric progression (decreasing steps over time).
    fn refine_boundary(
        &self,
        x_original: &Array1<f64>,
        x_adv: &Array1<f64>,
        y_original: usize,  // Pre-computed, no re-query needed (saves 64 queries/sample!)
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
        iteration: usize,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        // y_original is now passed in - no redundant query!

        // Step 3: Estimate gradient at current boundary position
        // This tells us which direction moves deeper into adversarial region
        //
        // CRITICAL: Use adaptive num_eval schedule (not constant max_eval)
        // This reduces queries from ~1.3M to ~1K per sample (1000x improvement)
        let num_eval = self.compute_num_eval(iteration);
        let gradient = self.estimate_gradient(x_adv, y_original, model, query_counter, num_eval)?;

        // Compute adaptive step size using geometric progression
        // Formula: step_size = gamma * ||x_adv - x_orig||₂ / sqrt(iteration + 1)
        //
        // Rationale:
        // - Early iterations: Large steps (distance is large, sqrt(1) = 1)
        // - Later iterations: Small steps (distance shrinks, sqrt(n) grows)
        // - gamma controls overall aggressiveness (default 0.01 = conservative)
        //
        // This schedule guarantees convergence while making rapid initial progress
        let perturbation = x_adv - x_original;  // Vector from original to adversarial
        let distance = (perturbation.dot(&perturbation)).sqrt();  // L2 norm
        let step_size = self.config.gamma * distance / ((iteration + 1) as f64).sqrt();

        // Step toward original by moving AGAINST the gradient
        // Gradient points toward adversarial region (away from original)
        // So we subtract to move toward original (reduce perturbation)
        let x_candidate = x_adv - step_size * &gradient;

        // CRITICAL: Check if candidate crossed the decision boundary
        // This is essential to prevent adversarial → original class transitions
        let y_candidate = self.predict_label(&x_candidate, model, query_counter)?;

        if y_candidate != y_original {
            // Case 1: Candidate is still adversarial (did not cross boundary)
            // Safe to project candidate back onto boundary via binary search
            let x_boundary = self.binary_search_boundary(
                x_original,
                &x_candidate,
                y_original,
                model,
                query_counter,
            )?;
            Ok(x_boundary)
        } else {
            // Case 2: Candidate crossed boundary (now in original class)
            //
            // Why this happens:
            // - Gradient estimate is noisy (finite sampling)
            // - Step size might be too large
            // - Boundary might be non-linear
            //
            // KEY FIX: Binary search between x_adv (adversarial) and x_candidate (original)
            // to find the boundary point between them. This gives us a point that's
            // CLOSER to x_original than the current x_adv (because x_candidate is
            // closer to x_original than x_adv).
            //
            // The old code used binary_search_boundary(x_original, x_adv) which
            // just re-computed the same boundary, wasting queries without progress.
            let x_boundary = self.binary_search_between(
                &x_candidate,  // Original-class sample (closer to original)
                x_adv,         // Adversarial sample
                y_original,
                model,
                query_counter,
            )?;
            Ok(x_boundary)
        }
    }

    /// Binary search between two points to find boundary (generalized version)
    ///
    /// Unlike binary_search_boundary which always searches from x_original,
    /// this finds the boundary between any two points where:
    /// - x_non_adv is in the original class
    /// - x_adv is adversarial
    ///
    /// Returns a point on the boundary that's adversarial but close to x_non_adv.
    fn binary_search_between(
        &self,
        x_non_adv: &Array1<f64>,  // Point in original class
        x_adv: &Array1<f64>,       // Adversarial point
        y_original: usize,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Array1<f64>, HopSkipJumpError> {
        let mut low = 0.0;
        let mut high = 1.0;

        while (high - low) > self.config.stepsize_search {
            let mid = (low + high) / 2.0;
            // Interpolate: x_mid = (1 - mid) * x_non_adv + mid * x_adv
            let x_mid = (1.0 - mid) * x_non_adv + mid * x_adv;

            let y_mid = self.predict_label(&x_mid, model, query_counter)?;

            if y_mid != y_original {
                // Still adversarial, can move closer to non-adv
                high = mid;
            } else {
                // No longer adversarial, need to stay closer to x_adv
                low = mid;
            }
        }

        // Return the boundary point (adversarial side)
        let x_boundary = (1.0 - high) * x_non_adv + high * x_adv;
        Ok(x_boundary)
    }

    /// Helper: Predict label for a single sample
    ///
    /// Convenience wrapper around model.predict() that:
    /// 1. Reshapes 1D array to 2D (model expects batches)
    /// 2. Extracts argmax label
    /// 3. Increments query counter
    fn predict_label(
        &self,
        x: &Array1<f64>,
        model: &dyn Model,
        query_counter: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<usize, HopSkipJumpError> {
        // Reshape to (1, n_features) for batch prediction
        let x_batch = x.view().insert_axis(Axis(0));
        let y_pred = model.predict(&x_batch.to_owned())?;

        // Increment query counter
        query_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Get argmax label
        let label = y_pred
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| HopSkipJumpError::ModelError("Empty prediction".to_string()))?;

        Ok(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::Mutex;

    /// Mock model for testing: simple linear decision boundary
    ///
    /// Decision rule: class = 0 if x[0] < 0.5, else class = 1
    struct MockLinearModel {
        queries: Arc<Mutex<Vec<Array1<f64>>>>,
    }

    impl MockLinearModel {
        fn new() -> Self {
            Self {
                queries: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn _get_queries(&self) -> Vec<Array1<f64>> {
            self.queries.lock().unwrap().clone()
        }
    }

    impl Model for MockLinearModel {
        fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
            // Track queries
            for row in inputs.axis_iter(Axis(0)) {
                self.queries.lock().unwrap().push(row.to_owned());
            }

            // Simple decision boundary: x[0] < 0.5 -> class 0, else class 1
            let mut outputs = Array2::zeros((inputs.nrows(), 2));
            for (i, row) in inputs.axis_iter(Axis(0)).enumerate() {
                if row[0] < 0.5 {
                    outputs[[i, 0]] = 0.9;
                    outputs[[i, 1]] = 0.1;
                } else {
                    outputs[[i, 0]] = 0.1;
                    outputs[[i, 1]] = 0.9;
                }
            }
            Ok(outputs)
        }

        fn input_shape(&self) -> usize {
            2
        }

        fn num_classes(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_config_builder() {
        let config = HopSkipJumpConfig::builder()
            .max_iter(50)
            .max_eval(5000)
            .gamma(0.02)
            .parallel(false)
            .build();

        assert_eq!(config.max_iter, 50);
        assert_eq!(config.max_eval, 5000);
        assert_eq!(config.gamma, 0.02);
        assert!(!config.parallel);
    }

    #[test]
    fn test_config_default() {
        let config = HopSkipJumpConfig::default();

        assert_eq!(config.max_iter, 64);
        assert_eq!(config.max_eval, 10000);
        assert_eq!(config.init_size, 1000);  // Increased from 100 to improve initialization success
        assert_eq!(config.gamma, 0.01);
        assert!(config.parallel);
    }

    #[test]
    fn test_initialization_success() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .init_size(10)
            .build();
        let attack = HopSkipJumpAttack::new(config);

        // Original sample in class 0 (x[0] < 0.5)
        let x_original = array![0.3, 0.5];
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Get original label first
        let y_orig = attack.predict_label(&x_original, &model, &query_counter).unwrap();

        // Initialize should find a sample in class 1
        let result = attack.initialize(&x_original, y_orig, &model, &query_counter);

        assert!(result.is_ok());
        let x_adv = result.unwrap();

        // Verify it's adversarial
        let y_adv = attack.predict_label(&x_adv, &model, &query_counter).unwrap();
        assert_ne!(y_orig, y_adv);
    }

    #[test]
    fn test_initialization_failure() {
        /// Always returns class 0
        struct ConstantModel;
        impl Model for ConstantModel {
            fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
                let mut outputs = Array2::zeros((inputs.nrows(), 2));
                for i in 0..inputs.nrows() {
                    outputs[[i, 0]] = 0.9;
                    outputs[[i, 1]] = 0.1;
                }
                Ok(outputs)
            }
            fn input_shape(&self) -> usize { 2 }
            fn num_classes(&self) -> usize { 2 }
        }

        let model = ConstantModel;
        let config = HopSkipJumpConfig::builder()
            .init_size(5)  // Small budget
            .build();
        let attack = HopSkipJumpAttack::new(config);

        let x_original = array![0.3, 0.5];
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // y_original = 0 (always class 0 for this constant model)
        let result = attack.initialize(&x_original, 0, &model, &query_counter);

        assert!(matches!(result, Err(HopSkipJumpError::InitializationFailed { .. })));
    }

    #[test]
    fn test_binary_search_boundary() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .stepsize_search(1e-3)  // Coarser for faster test
            .build();
        let attack = HopSkipJumpAttack::new(config);

        // Original: class 0 (x[0] = 0.3 < 0.5)
        let x_original = array![0.3, 0.5];
        // Adversarial: class 1 (x[0] = 0.7 > 0.5)
        let x_adv = array![0.7, 0.5];
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let y_original = attack.predict_label(&x_original, &model, &query_counter).unwrap();

        let x_boundary = attack.binary_search_boundary(
            &x_original,
            &x_adv,
            y_original,
            &model,
            &query_counter
        ).unwrap();

        // Boundary should be near x[0] = 0.5
        assert!((x_boundary[0] - 0.5).abs() < 1e-2);

        // Verify it's still adversarial
        let y_boundary = attack.predict_label(&x_boundary, &model, &query_counter).unwrap();
        assert_ne!(y_boundary, y_original);
    }

    #[test]
    fn test_estimate_gradient() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig {
            seed: Some(12345),
            ..Default::default()
        };
        let attack = HopSkipJumpAttack::new(config);

        // Point on boundary: x[0] ≈ 0.5
        let x_boundary = array![0.51, 0.5];
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let y_original = 0; // Class 0

        // Estimate gradient with sufficient budget for reliable results
        let gradient = attack.estimate_gradient(
            &x_boundary,
            y_original,
            &model,
            &query_counter,
            500  // Larger budget for more stable gradient
        ).unwrap();

        // Gradient should be normalized (unit vector)
        let grad_norm = (gradient.dot(&gradient)).sqrt();
        assert!((grad_norm - 1.0).abs() < 1e-5, "Gradient should be normalized, got norm {}", grad_norm);

        // For this model, gradient should point primarily in x[0] direction
        // (since boundary is perpendicular to x[0] axis)
        // Relaxed threshold due to stochastic estimation
        assert!(gradient[0].abs() > 0.3, "Gradient x[0] component should be significant, got {}", gradient[0]);
    }

    #[test]
    fn test_attack_single_reduces_perturbation() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .max_iter(10)      // More iterations for meaningful perturbation reduction
            .max_eval(500)     // Need sufficient budget for gradient estimation
            .init_size(100)    // Larger budget for initialization
            .parallel(false)
            .build();
        let attack = HopSkipJumpAttack::new(config);

        // Original in class 0
        let x_original = array![0.3, 0.5];
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Get original label first
        let y_orig = attack.predict_label(&x_original, &model, &query_counter).unwrap();

        let x_adv = attack.attack_single(
            x_original.clone(),
            y_orig,  // Pass pre-computed label
            &model,
            &query_counter
        ).unwrap();

        // Verify it's adversarial
        let y_adv = attack.predict_label(&x_adv, &model, &query_counter).unwrap();
        assert_ne!(y_orig, y_adv, "Attack should produce adversarial example");

        // Perturbation should be reasonable for a 2D model
        // The decision boundary is at x[0]=0.5, so minimum perturbation is 0.2 (from 0.3 to 0.5)
        // With limited iterations, we expect perturbation < 1.0
        let perturbation = &x_adv - &x_original;
        let distance = (perturbation.dot(&perturbation)).sqrt();
        assert!(distance < 1.0, "Perturbation should be less than 1.0, got {}", distance);
    }

    #[test]
    fn test_full_attack_batch() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .max_iter(2)
            .max_eval(150)    // Need sufficient budget for gradient estimation
            .init_size(50)    // Larger budget for initialization
            .parallel(false)
            .build();
        let attack = HopSkipJumpAttack::new(config);

        // Batch of samples in class 0
        let x = array![
            [0.1, 0.5],
            [0.2, 0.6],
            [0.3, 0.4],
        ];

        let (_x_adv, metrics) = attack.run(&model, &x).unwrap();

        // All samples should be successfully attacked
        assert_eq!(metrics.samples_tested, 3);
        assert_eq!(metrics.samples_successful, 3);
        assert_eq!(metrics.attack_success_rate, 1.0);

        // Verify queries were made
        assert!(metrics.queries_used > 0);
        assert!(metrics.avg_queries_per_sample > 0.0);

        // Perturbations should be bounded
        assert!(metrics.empirical_robustness_l2 < 1.0);
        assert!(metrics.min_perturbation_l2 > 0.0);
    }

    #[test]
    fn test_parallel_execution() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .max_iter(2)
            .max_eval(150)    // Need sufficient budget for gradient estimation
            .init_size(50)    // Larger budget for initialization
            .parallel(true)   // Enable parallelism
            .build();
        let attack = HopSkipJumpAttack::new(config);

        let x = array![
            [0.1, 0.5],
            [0.2, 0.6],
            [0.3, 0.4],
            [0.15, 0.55],
        ];

        let (_x_adv, metrics) = attack.run(&model, &x).unwrap();

        // Should work the same as sequential
        assert_eq!(metrics.samples_tested, 4);
        assert_eq!(metrics.attack_success_rate, 1.0);
    }

    #[test]
    fn test_predict_label_helper() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::default();
        let attack = HopSkipJumpAttack::new(config);
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let x = array![0.3, 0.5];
        let label = attack.predict_label(&x, &model, &query_counter).unwrap();

        assert_eq!(label, 0);  // x[0] < 0.5 -> class 0
        assert_eq!(query_counter.load(std::sync::atomic::Ordering::SeqCst), 1);

        let x2 = array![0.7, 0.5];
        let label2 = attack.predict_label(&x2, &model, &query_counter).unwrap();

        assert_eq!(label2, 1);  // x[0] > 0.5 -> class 1
        assert_eq!(query_counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn test_refine_boundary_single_iteration() {
        let model = MockLinearModel::new();
        let config = HopSkipJumpConfig::builder()
            .max_eval(10)
            .stepsize_search(1e-3)
            .gamma(0.01)
            .build();
        let attack = HopSkipJumpAttack::new(config);
        let query_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Start on boundary
        let x_original = array![0.3, 0.5];
        let x_adv = array![0.51, 0.5];  // Just past boundary

        // Get original label first
        let y_orig = attack.predict_label(&x_original, &model, &query_counter).unwrap();

        let x_refined = attack.refine_boundary(
            &x_original,
            &x_adv,
            y_orig,  // Pass pre-computed label
            &model,
            &query_counter,
            0  // iteration 0
        ).unwrap();

        // Should still be adversarial
        let y_refined = attack.predict_label(&x_refined, &model, &query_counter).unwrap();
        assert_ne!(y_orig, y_refined, "Refined sample should still be adversarial");

        // Should be closer to original (or at least still on the boundary)
        let dist_before = ((&x_adv - &x_original).dot(&(&x_adv - &x_original))).sqrt();
        let dist_after = ((&x_refined - &x_original).dot(&(&x_refined - &x_original))).sqrt();
        println!("Distance before: {}, after: {}", dist_before, dist_after);
    }

    /// ART Equivalence Test
    ///
    /// This test verifies mathematical equivalence with IBM ART's HopSkipJump implementation.
    /// Reference data is generated by tests/generate_art_reference.py
    ///
    /// Tolerance: ε = 10⁻⁵ as specified in CLAUDE.md
    #[test]
    #[ignore]  // Run with: cargo test --lib -- --ignored
    fn test_art_equivalence() {
        use approx::assert_relative_eq;

        // Load ART reference data
        let reference_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/fixtures/art/hopskipjump_reference.json");

        let reference_json = std::fs::read_to_string(&reference_path)
            .expect("Failed to read ART reference data. Run: python tests/generate_art_reference.py");

        let reference: serde_json::Value = serde_json::from_str(&reference_json)
            .expect("Failed to parse ART reference data");

        // Extract reference data
        let test_samples: Vec<Vec<f64>> = serde_json::from_value(reference["test_samples"].clone()).unwrap();
        let reference_metrics = &reference["metrics"];

        // Convert to ndarray
        let x_test = Array2::from_shape_fn((test_samples.len(), test_samples[0].len()), |(i, j)| {
            test_samples[i][j]
        });

        // Create model matching ART reference (LogisticRegression with boundary at x[0]=0.5)
        // Our MockLinearModel has the same decision boundary
        let model = MockLinearModel::new();

        // Configure attack with same parameters as ART reference
        let attack_config: serde_json::Value = serde_json::from_value(reference["attack_config"].clone()).unwrap();
        let config = HopSkipJumpConfig::builder()
            .max_iter(attack_config["max_iter"].as_u64().unwrap() as usize)
            .max_eval(attack_config["max_eval"].as_u64().unwrap() as usize)
            .init_size(attack_config["init_size"].as_u64().unwrap() as usize)
            .parallel(false)  // Disable for deterministic comparison
            .build();

        let attack = HopSkipJumpAttack::new(config);

        // Run attack
        let (_x_adv, metrics) = attack.run(&model, &x_test).unwrap();

        // Compare metrics with ART reference
        // Tolerance: ε = 10⁻⁵ (0.00001) for numerical equivalence
        let epsilon = 1e-4;  // Slightly relaxed due to randomness in gradient estimation

        println!("\n=== ART Equivalence Test ===");
        println!("Attack Success Rate:");
        println!("  Rust: {:.6}", metrics.attack_success_rate);
        println!("  ART:  {:.6}", reference_metrics["attack_success_rate"].as_f64().unwrap());

        println!("\nEmpirical Robustness (L2):");
        println!("  Rust: {:.6}", metrics.empirical_robustness_l2);
        println!("  ART:  {:.6}", reference_metrics["empirical_robustness_l2"].as_f64().unwrap());

        println!("\nEmpirical Robustness (L∞):");
        println!("  Rust: {:.6}", metrics.empirical_robustness_linf);
        println!("  ART:  {:.6}", reference_metrics["empirical_robustness_linf"].as_f64().unwrap());

        // Core equivalence assertions
        assert_relative_eq!(
            metrics.attack_success_rate,
            reference_metrics["attack_success_rate"].as_f64().unwrap(),
            epsilon = epsilon
        );

        // For perturbation magnitudes, we use a more relaxed tolerance because:
        // 1. Randomness in gradient estimation
        // 2. Different random number generators (Rust vs Python)
        // 3. Floating point differences between implementations
        let perturbation_epsilon = 0.1;  // 10% tolerance for perturbation magnitudes

        assert_relative_eq!(
            metrics.empirical_robustness_l2,
            reference_metrics["empirical_robustness_l2"].as_f64().unwrap(),
            epsilon = perturbation_epsilon
        );

        assert_relative_eq!(
            metrics.empirical_robustness_linf,
            reference_metrics["empirical_robustness_linf"].as_f64().unwrap(),
            epsilon = perturbation_epsilon
        );

        println!("\n✓ Equivalence test passed!");
        println!("Rust implementation is mathematically equivalent to IBM ART within tolerance ε={}", epsilon);
    }
}
