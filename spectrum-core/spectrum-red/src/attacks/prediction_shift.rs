/// Prediction Shift Attack for regression models
///
/// This attack maximizes the absolute change in model output using
/// gradient-free optimization. It tests output stability under adversarial
/// perturbations.
///
/// # Algorithm
///
/// For each sample:
/// 1. Sample random directions in feature space
/// 2. Evaluate model at perturbed points within epsilon ball
/// 3. Track perturbation that causes maximum output change
/// 4. Can target specific directions (increase, decrease, or any)
///
/// # Use Cases
///
/// - Testing price stability in pricing models
/// - Evaluating risk score sensitivity
/// - Assessing credit limit stability

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::Normal;
use rayon::prelude::*;

use super::regression_common::{
    bootstrap_ci_mean, ConfidenceLevel, RegressionAttackError, RegressionMetrics, RegressionModel,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Prediction Shift Attack.
#[derive(Clone, Debug)]
pub struct PredictionShiftConfig {
    /// Maximum iterations per sample
    pub max_iter: usize,
    /// Maximum perturbation magnitude (L2 norm)
    pub epsilon: f64,
    /// Number of random directions to sample per iteration
    pub n_directions: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Feature clip bounds (min, max)
    pub clip_bounds: Option<(f64, f64)>,
    /// Target direction: "increase", "decrease", or "any"
    pub target_direction: String,
    /// Success threshold: shift must exceed this fraction of original value.
    /// For example, 0.10 means a 10% change in prediction counts as success.
    pub success_threshold: f64,
    /// Confidence level for bootstrap CI on mean attempts
    pub confidence_level: ConfidenceLevel,
    /// Number of bootstrap resamples for CI computation
    pub n_bootstrap: usize,
    /// Set a seed for random noise generation (default: None for entropy-based)
    pub seed: Option<u64>,
}

impl Default for PredictionShiftConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            epsilon: 1.0,
            n_directions: 50,
            parallel: true,
            clip_bounds: None,
            target_direction: "any".to_string(),
            success_threshold: 0.10,
            confidence_level: ConfidenceLevel::default(),
            n_bootstrap: 10_000,
            seed: None,
        }
    }
}

impl PredictionShiftConfig {
    pub fn builder() -> PredictionShiftConfigBuilder {
        PredictionShiftConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct PredictionShiftConfigBuilder {
    config: PredictionShiftConfig,
}

impl PredictionShiftConfigBuilder {
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn n_directions(mut self, n_directions: usize) -> Self {
        self.config.n_directions = n_directions;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn clip_bounds(mut self, min: f64, max: f64) -> Self {
        self.config.clip_bounds = Some((min, max));
        self
    }

    pub fn target_direction(mut self, direction: &str) -> Self {
        self.config.target_direction = direction.to_string();
        self
    }

    pub fn success_threshold(mut self, threshold: f64) -> Self {
        self.config.success_threshold = threshold;
        self
    }

    pub fn confidence_level(mut self, level: ConfidenceLevel) -> Self {
        self.config.confidence_level = level;
        self
    }

    pub fn n_bootstrap(mut self, n: usize) -> Self {
        self.config.n_bootstrap = n;
        self
    }

    /// Set a random seed for reproducibility
    pub fn seed(mut self, value: u64) -> Self {
        self.config.seed = Some(value);
        self
    }

    pub fn build(self) -> PredictionShiftConfig {
        self.config
    }
}

// ============================================================================
// Attack Implementation
// ============================================================================

/// Prediction Shift Attack for regression models.
///
/// Finds perturbations that maximize the change in predicted output.
pub struct PredictionShiftAttack {
    config: PredictionShiftConfig,
}

impl PredictionShiftAttack {
    pub fn new(config: PredictionShiftConfig) -> Self {
        Self { config }
    }

    fn clip(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.clip_bounds {
            Some((min, max)) => x.mapv(|v| v.max(min).min(max)),
            None => x.clone(),
        }
    }

    /// Attack a single sample.
    ///
    /// Returns:
    /// - Adversarial example
    /// - Best shift achieved
    /// - Attempts to first success (None if never succeeded within budget)
    fn attack_single(
        &self,
        x_original: &Array1<f64>,
        y_original: f64,
        model: &dyn RegressionModel,
    ) -> Result<(Array1<f64>, f64, Option<usize>), RegressionAttackError> {
        let n_features = x_original.len();
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut best_adv = x_original.clone();
        let mut best_shift = 0.0f64;
        let mut attempts_to_success: Option<usize> = None;
        let mut total_attempts: usize = 0;

        // Success threshold for this sample
        let threshold = y_original.abs() * self.config.success_threshold;

        // Random search optimization
        for _iter in 0..self.config.max_iter {
            // Sample random directions
            for _ in 0..self.config.n_directions {
                total_attempts += 1;

                // Random unit direction
                let direction: Array1<f64> =
                    Array1::from_iter((0..n_features).map(|_| normal.sample(&mut rng)));
                let dir_norm = direction.mapv(|v| v.powi(2)).sum().sqrt();

                if dir_norm < 1e-10 {
                    continue;
                }

                let unit_dir = &direction / dir_norm;

                // Random step size within epsilon ball
                let step_size = rng.gen::<f64>() * self.config.epsilon;
                let perturbation = &unit_dir * step_size;

                let x_candidate = x_original + &perturbation;
                let x_clipped = self.clip(&x_candidate);

                // Evaluate
                let x_2d = x_clipped.clone().insert_axis(Axis(0));
                let pred = model
                    .predict(&x_2d)
                    .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

                let shift = pred[0] - y_original;

                // Check if this attempt counts as a success (first time)
                let is_success = match self.config.target_direction.as_str() {
                    "increase" => shift > threshold,
                    "decrease" => shift < -threshold,
                    _ => shift.abs() > threshold, // "any"
                };

                if is_success && attempts_to_success.is_none() {
                    attempts_to_success = Some(total_attempts);
                }

                // Check if better according to target direction
                let is_better = match self.config.target_direction.as_str() {
                    "increase" => shift > best_shift,
                    "decrease" => shift < best_shift,
                    _ => shift.abs() > best_shift.abs(), // "any"
                };

                if is_better {
                    best_adv = x_clipped;
                    best_shift = shift;
                }
            }
        }

        Ok((best_adv, best_shift, attempts_to_success))
    }

    /// Run the attack on a batch of samples.
    pub fn run(
        &self,
        model: &dyn RegressionModel,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, RegressionMetrics), RegressionAttackError> {
        let n_samples = x.nrows();

        // Get original predictions
        let y_original = model
            .predict(x)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        // Run attacks - now returns (adv_example, shift, attempts_to_success)
        let results: Vec<(Array1<f64>, f64, Option<usize>)> = if self.config.parallel {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let x_i = x.row(i).to_owned();
                    let y_i = y_original[i];
                    self.attack_single(&x_i, y_i, model)
                        .unwrap_or_else(|_| (x_i.clone(), 0.0, None))
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let x_i = x.row(i).to_owned();
                    let y_i = y_original[i];
                    self.attack_single(&x_i, y_i, model)
                        .unwrap_or_else(|_| (x_i.clone(), 0.0, None))
                })
                .collect()
        };

        // Aggregate
        let mut x_adv = Array2::zeros(x.dim());
        let mut shifts = Vec::new();
        let mut attempts_list: Vec<Option<usize>> = Vec::new();

        for (i, (adv, shift, attempts)) in results.into_iter().enumerate() {
            x_adv.row_mut(i).assign(&adv);
            shifts.push(shift);
            attempts_list.push(attempts);
        }

        let y_adv = model
            .predict(&x_adv)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        // Compute metrics
        let metrics = self.compute_metrics(x, &x_adv, &y_original, &y_adv, &shifts, &attempts_list);

        Ok((x_adv, metrics))
    }

    fn compute_metrics(
        &self,
        x_orig: &Array2<f64>,
        x_adv: &Array2<f64>,
        _y_orig: &Array1<f64>,
        _y_adv: &Array1<f64>,
        shifts: &[f64],
        attempts_list: &[Option<usize>],
    ) -> RegressionMetrics {
        let n = shifts.len();
        let abs_shifts: Vec<f64> = shifts.iter().map(|s| s.abs()).collect();

        // Extract successful attempts (those with Some value)
        let successful_attempts: Vec<usize> = attempts_list
            .iter()
            .filter_map(|&a| a)
            .collect();

        let samples_succeeded = successful_attempts.len();
        let samples_never_succeeded = n - samples_succeeded;

        // Compute attempts statistics (only over successful samples)
        let (mean_attempts, median_attempts, min_attempts, max_attempts) = if successful_attempts.is_empty() {
            (f64::NAN, f64::NAN, None, None)
        } else {
            let sum: usize = successful_attempts.iter().sum();
            let mean = sum as f64 / successful_attempts.len() as f64;

            let mut sorted = successful_attempts.clone();
            sorted.sort();
            let median = sorted[sorted.len() / 2] as f64;
            let min = *sorted.first().unwrap();
            let max = *sorted.last().unwrap();

            (mean, median, Some(min), Some(max))
        };

        // Bootstrap CI on mean attempts
        let attempts_ci = bootstrap_ci_mean(
            &successful_attempts,
            self.config.confidence_level,
            self.config.n_bootstrap,
        );

        let (l2_norms, linf_norms) = RegressionMetrics::compute_perturbation_stats(x_orig, x_adv);
        let (mean_l2, min_l2, max_l2, median_l2) = RegressionMetrics::l2_stats(&l2_norms);

        // Total directions sampled across all samples
        let total_directions = n * self.config.max_iter * self.config.n_directions;

        RegressionMetrics {
            attack_type: "PredictionShift".to_string(),
            samples_tested: n,
            samples_succeeded,
            samples_never_succeeded,
            mean_attempts_to_success: mean_attempts,
            median_attempts_to_success: median_attempts,
            min_attempts,
            max_attempts,
            attempts_ci,
            confidence_level: self.config.confidence_level.as_f64(),
            attack_success_rate: None, // Not used for PredictionShift
            mean_perturbation_l2: mean_l2,
            mean_perturbation_linf: linf_norms.iter().sum::<f64>() / n as f64,
            min_perturbation_l2: min_l2,
            max_perturbation_l2: max_l2,
            median_perturbation_l2: median_l2,
            mean_prediction_shift: shifts.iter().sum::<f64>() / n as f64,
            max_prediction_shift: shifts.iter().fold(0.0f64, |a, &b| a.max(b.abs())),
            mean_absolute_shift: abs_shifts.iter().sum::<f64>() / n as f64,
            queries_used: total_directions,
            avg_queries_per_sample: total_directions as f64 / n as f64,
            bin_flip_rate: None,
            mean_bin_distance: None,
            coverage_break_rate: None,
            mean_interval_width_increase: None,
            original_coverage: None,
            adversarial_coverage: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRegressionModel {
        n_features: usize,
    }

    impl RegressionModel for MockRegressionModel {
        fn predict(&self, inputs: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
            Ok(inputs.sum_axis(Axis(1)))
        }

        fn input_shape(&self) -> usize {
            self.n_features
        }
    }

    #[test]
    fn test_config_builder() {
        let config = PredictionShiftConfig::builder()
            .epsilon(2.0)
            .target_direction("increase")
            .build();

        assert_eq!(config.epsilon, 2.0);
        assert_eq!(config.target_direction, "increase");
    }

    #[test]
    fn test_attack_single() {
        let attack = PredictionShiftAttack::new(PredictionShiftConfig {
            max_iter: 10,
            n_directions: 10,
            epsilon: 0.5,
            parallel: false,
            ..Default::default()
        });

        let model = MockRegressionModel { n_features: 4 };
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = 10.0;

        let (x_adv, shift, _queries) = attack.attack_single(&x, y, &model).unwrap();
        assert_ne!(shift, 0.0);
        assert_eq!(x_adv.len(), 4);
    }
}
