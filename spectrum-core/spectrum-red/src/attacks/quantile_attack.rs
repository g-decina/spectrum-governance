/// Quantile Attack for Conformal Prediction models
///
/// This attack targets conformal prediction models by finding perturbations that:
/// 1. Push predictions outside calibrated intervals (break coverage guarantee)
/// 2. Shift predictions to interval boundaries (maximize uncertainty)
///
/// # Regulatory Context
///
/// This attack is especially important for testing CP-compliant models because:
/// - Conformal prediction provides coverage guarantees (e.g., 95%)
/// - Regulators may require these guarantees for risk assessment
/// - This attack tests whether guarantees hold under adversarial input
///
/// # Attack Modes
///
/// - **break_coverage**: Find perturbations where true value falls outside interval
/// - **maximize_width**: Find perturbations that increase prediction uncertainty

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

use super::regression_common::{RegressionAttackError, RegressionMetrics, RegressionModel};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Quantile Attack.
#[derive(Clone, Debug)]
pub struct QuantileAttackConfig {
    /// Maximum iterations per sample
    pub max_iter: usize,
    /// Maximum perturbation magnitude (L2 norm)
    pub epsilon: f64,
    /// Number of random directions to sample per iteration
    pub n_directions: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Feature clip bounds
    pub clip_bounds: Option<(f64, f64)>,
    /// Attack mode: "break_coverage" or "maximize_width"
    pub attack_mode: String,
    /// For models without native interval prediction, provide calibrated half-width
    pub calibration_width: Option<f64>,
}

impl Default for QuantileAttackConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            epsilon: 1.0,
            n_directions: 50,
            parallel: true,
            clip_bounds: None,
            attack_mode: "break_coverage".to_string(),
            calibration_width: None,
        }
    }
}

impl QuantileAttackConfig {
    pub fn builder() -> QuantileAttackConfigBuilder {
        QuantileAttackConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct QuantileAttackConfigBuilder {
    config: QuantileAttackConfig,
}

impl QuantileAttackConfigBuilder {
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

    pub fn attack_mode(mut self, mode: &str) -> Self {
        self.config.attack_mode = mode.to_string();
        self
    }

    pub fn calibration_width(mut self, width: f64) -> Self {
        self.config.calibration_width = Some(width);
        self
    }

    pub fn build(self) -> QuantileAttackConfig {
        self.config
    }
}

// ============================================================================
// Attack Implementation
// ============================================================================

/// Quantile Attack for conformal prediction models.
///
/// Targets the coverage guarantee of conformal prediction by finding
/// perturbations that push predictions outside calibrated intervals.
pub struct QuantileAttack {
    config: QuantileAttackConfig,
}

impl QuantileAttack {
    pub fn new(config: QuantileAttackConfig) -> Self {
        Self { config }
    }

    fn clip(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.clip_bounds {
            Some((min, max)) => x.mapv(|v| v.max(min).min(max)),
            None => x.clone(),
        }
    }

    /// Get prediction interval for a single sample.
    fn get_interval(
        &self,
        x: &Array1<f64>,
        model: &dyn RegressionModel,
    ) -> Result<(f64, f64, f64), RegressionAttackError> {
        let x_2d = x.clone().insert_axis(Axis(0));

        let pred = model
            .predict(&x_2d)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        let (lower, upper) = match model.predict_interval(&x_2d) {
            Ok(Some((l, u))) => (l[0], u[0]),
            _ => {
                let half_width = self.config.calibration_width.unwrap_or(0.5);
                (pred[0] - half_width, pred[0] + half_width)
            }
        };

        Ok((pred[0], lower, upper))
    }

    /// Attack a single sample to break coverage.
    fn attack_single_break_coverage(
        &self,
        x_original: &Array1<f64>,
        y_true: f64,
        _lower_bound: f64,
        _upper_bound: f64,
        model: &dyn RegressionModel,
    ) -> Result<(Array1<f64>, bool), RegressionAttackError> {
        let n_features = x_original.len();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Goal: find perturbation where predicted interval no longer covers y_true
        let mut best_adv = x_original.clone();
        let mut found_attack = false;
        let mut best_margin = f64::MAX; // Distance from y_true to interval boundary

        for _iter in 0..self.config.max_iter {
            if found_attack {
                break; // Early exit once we find a successful attack
            }

            for _ in 0..self.config.n_directions {
                // Random direction
                let direction: Array1<f64> =
                    Array1::from_iter((0..n_features).map(|_| normal.sample(&mut rng)));
                let dir_norm = direction.mapv(|v| v.powi(2)).sum().sqrt();

                if dir_norm < 1e-10 {
                    continue;
                }

                let unit_dir = &direction / dir_norm;
                let step_size = rng.gen::<f64>() * self.config.epsilon;
                let perturbation = &unit_dir * step_size;

                let x_candidate = x_original + &perturbation;
                let x_clipped = self.clip(&x_candidate);

                // Get prediction and interval
                let (_, new_lower, new_upper) = self.get_interval(&x_clipped, model)?;

                // Check if y_true is outside the new interval
                let outside = y_true < new_lower || y_true > new_upper;

                if outside {
                    best_adv = x_clipped;
                    found_attack = true;
                    break;
                }

                // Track how close we are to breaking coverage
                let margin_lower = y_true - new_lower;
                let margin_upper = new_upper - y_true;
                let min_margin = margin_lower.min(margin_upper);

                // We want to minimize margin (push y_true towards boundary)
                if min_margin < best_margin {
                    best_margin = min_margin;
                    best_adv = x_clipped;
                }
            }
        }

        Ok((best_adv, found_attack))
    }

    /// Attack a single sample to maximize interval width.
    fn attack_single_maximize_width(
        &self,
        x_original: &Array1<f64>,
        model: &dyn RegressionModel,
    ) -> Result<(Array1<f64>, f64), RegressionAttackError> {
        let n_features = x_original.len();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Get original width
        let (_, orig_lower, orig_upper) = self.get_interval(x_original, model)?;
        let orig_width = orig_upper - orig_lower;

        let mut best_adv = x_original.clone();
        let mut best_width = orig_width;

        for _iter in 0..self.config.max_iter {
            for _ in 0..self.config.n_directions {
                let direction: Array1<f64> =
                    Array1::from_iter((0..n_features).map(|_| normal.sample(&mut rng)));
                let dir_norm = direction.mapv(|v| v.powi(2)).sum().sqrt();

                if dir_norm < 1e-10 {
                    continue;
                }

                let unit_dir = &direction / dir_norm;
                let step_size = rng.gen::<f64>() * self.config.epsilon;
                let perturbation = &unit_dir * step_size;

                let x_candidate = x_original + &perturbation;
                let x_clipped = self.clip(&x_candidate);

                let (_, new_lower, new_upper) = self.get_interval(&x_clipped, model)?;
                let new_width = new_upper - new_lower;

                if new_width > best_width {
                    best_width = new_width;
                    best_adv = x_clipped;
                }
            }
        }

        let width_increase = best_width - orig_width;
        Ok((best_adv, width_increase))
    }

    /// Run the attack on a batch of samples.
    pub fn run(
        &self,
        model: &dyn RegressionModel,
        x: &Array2<f64>,
        y_true: Option<&Array1<f64>>,
    ) -> Result<(Array2<f64>, RegressionMetrics), RegressionAttackError> {
        let n_samples = x.nrows();

        // Get original predictions and intervals
        let y_pred = model
            .predict(x)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        let (lower_bounds, upper_bounds) = match model.predict_interval(x) {
            Ok(Some((l, u))) => (l, u),
            _ => {
                let half_width = self.config.calibration_width.unwrap_or(0.5);
                let lower: Array1<f64> = y_pred.mapv(|v| v - half_width);
                let upper: Array1<f64> = y_pred.mapv(|v| v + half_width);
                (lower, upper)
            }
        };

        // Use y_true if provided, otherwise use predictions
        let y_targets = match y_true {
            Some(y) => y.clone(),
            None => y_pred.clone(),
        };

        // Run attacks based on mode
        if self.config.attack_mode == "break_coverage" {
            let results: Vec<(Array1<f64>, bool)> = if self.config.parallel {
                (0..n_samples)
                    .into_par_iter()
                    .map(|i| {
                        let x_i = x.row(i).to_owned();
                        self.attack_single_break_coverage(
                            &x_i,
                            y_targets[i],
                            lower_bounds[i],
                            upper_bounds[i],
                            model,
                        )
                        .unwrap_or_else(|_| (x_i.clone(), false))
                    })
                    .collect()
            } else {
                (0..n_samples)
                    .map(|i| {
                        let x_i = x.row(i).to_owned();
                        self.attack_single_break_coverage(
                            &x_i,
                            y_targets[i],
                            lower_bounds[i],
                            upper_bounds[i],
                            model,
                        )
                        .unwrap_or_else(|_| (x_i.clone(), false))
                    })
                    .collect()
            };

            let mut x_adv = Array2::zeros(x.dim());
            let mut successes = Vec::new();

            for (i, (adv, success)) in results.into_iter().enumerate() {
                x_adv.row_mut(i).assign(&adv);
                successes.push(success);
            }

            let y_adv = model
                .predict(&x_adv)
                .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

            let (adv_lower, adv_upper) = match model.predict_interval(&x_adv) {
                Ok(Some((l, u))) => (l, u),
                _ => {
                    let half_width = self.config.calibration_width.unwrap_or(0.5);
                    (
                        y_adv.mapv(|v| v - half_width),
                        y_adv.mapv(|v| v + half_width),
                    )
                }
            };

            let metrics = self.compute_metrics(
                x,
                &x_adv,
                &y_pred,
                &y_adv,
                &successes,
                &y_targets,
                &lower_bounds,
                &upper_bounds,
                &adv_lower,
                &adv_upper,
            );

            Ok((x_adv, metrics))
        } else {
            // maximize_width mode
            let results: Vec<(Array1<f64>, f64)> = if self.config.parallel {
                (0..n_samples)
                    .into_par_iter()
                    .map(|i| {
                        let x_i = x.row(i).to_owned();
                        self.attack_single_maximize_width(&x_i, model)
                            .unwrap_or_else(|_| (x_i.clone(), 0.0))
                    })
                    .collect()
            } else {
                (0..n_samples)
                    .map(|i| {
                        let x_i = x.row(i).to_owned();
                        self.attack_single_maximize_width(&x_i, model)
                            .unwrap_or_else(|_| (x_i.clone(), 0.0))
                    })
                    .collect()
            };

            let mut x_adv = Array2::zeros(x.dim());
            let mut width_increases = Vec::new();

            for (i, (adv, width_inc)) in results.into_iter().enumerate() {
                x_adv.row_mut(i).assign(&adv);
                width_increases.push(width_inc);
            }

            let y_adv = model
                .predict(&x_adv)
                .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

            let (adv_lower, adv_upper) = match model.predict_interval(&x_adv) {
                Ok(Some((l, u))) => (l, u),
                _ => {
                    let half_width = self.config.calibration_width.unwrap_or(0.5);
                    (
                        y_adv.mapv(|v| v - half_width),
                        y_adv.mapv(|v| v + half_width),
                    )
                }
            };

            // For width maximization, success means we increased the width
            let successes: Vec<bool> = width_increases.iter().map(|&w| w > 0.0).collect();

            let metrics = self.compute_metrics(
                x,
                &x_adv,
                &y_pred,
                &y_adv,
                &successes,
                &y_targets,
                &lower_bounds,
                &upper_bounds,
                &adv_lower,
                &adv_upper,
            );

            Ok((x_adv, metrics))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_metrics(
        &self,
        x_orig: &Array2<f64>,
        x_adv: &Array2<f64>,
        y_orig: &Array1<f64>,
        y_adv: &Array1<f64>,
        successes: &[bool],
        y_targets: &Array1<f64>,
        orig_lower: &Array1<f64>,
        orig_upper: &Array1<f64>,
        adv_lower: &Array1<f64>,
        adv_upper: &Array1<f64>,
    ) -> RegressionMetrics {
        let n = successes.len();
        let successful = successes.iter().filter(|&&s| s).count();

        let (l2_norms, linf_norms) = RegressionMetrics::compute_perturbation_stats(x_orig, x_adv);
        let (mean_l2, min_l2, max_l2, median_l2) = RegressionMetrics::l2_stats(&l2_norms);

        let shifts: Vec<f64> = y_adv
            .iter()
            .zip(y_orig.iter())
            .map(|(&a, &o)| a - o)
            .collect();
        let abs_shifts: Vec<f64> = shifts.iter().map(|s| s.abs()).collect();

        // Interval width changes
        let orig_widths: Vec<f64> = orig_upper
            .iter()
            .zip(orig_lower.iter())
            .map(|(&u, &l)| u - l)
            .collect();
        let adv_widths: Vec<f64> = adv_upper
            .iter()
            .zip(adv_lower.iter())
            .map(|(&u, &l)| u - l)
            .collect();
        let width_increases: Vec<f64> = adv_widths
            .iter()
            .zip(orig_widths.iter())
            .map(|(&a, &o)| a - o)
            .collect();

        // Compute original and adversarial coverage (using y_targets as ground truth)
        let orig_coverage = y_targets
            .iter()
            .zip(orig_lower.iter())
            .zip(orig_upper.iter())
            .filter(|((&y, &l), &u)| y >= l && y <= u)
            .count() as f64
            / n as f64;

        let adv_coverage = y_targets
            .iter()
            .zip(adv_lower.iter())
            .zip(adv_upper.iter())
            .filter(|((&y, &l), &u)| y >= l && y <= u)
            .count() as f64
            / n as f64;

        RegressionMetrics {
            attack_type: "QuantileAttack".to_string(),
            samples_tested: n,
            // QuantileAttack uses attack_success_rate (not attempts-based metrics)
            samples_succeeded: successful,
            samples_never_succeeded: n - successful,
            mean_attempts_to_success: f64::NAN, // Not applicable for QuantileAttack
            median_attempts_to_success: f64::NAN,
            min_attempts: None,
            max_attempts: None,
            attempts_ci: (f64::NAN, f64::NAN),
            confidence_level: 0.0, // Not used
            attack_success_rate: Some(successful as f64 / n as f64),
            mean_perturbation_l2: mean_l2,
            mean_perturbation_linf: linf_norms.iter().sum::<f64>() / n as f64,
            min_perturbation_l2: min_l2,
            max_perturbation_l2: max_l2,
            median_perturbation_l2: median_l2,
            mean_prediction_shift: shifts.iter().sum::<f64>() / n as f64,
            max_prediction_shift: shifts.iter().fold(0.0f64, |a, &b| a.max(b.abs())),
            mean_absolute_shift: abs_shifts.iter().sum::<f64>() / n as f64,
            queries_used: 0,
            avg_queries_per_sample: 0.0,
            bin_flip_rate: None,
            mean_bin_distance: None,
            coverage_break_rate: Some(successful as f64 / n as f64),
            mean_interval_width_increase: Some(width_increases.iter().sum::<f64>() / n as f64),
            original_coverage: Some(orig_coverage),
            adversarial_coverage: Some(adv_coverage),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRegressionModel {
        n_features: usize,
        interval_width: f64,
    }

    impl RegressionModel for MockRegressionModel {
        fn predict(&self, inputs: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
            Ok(inputs.sum_axis(Axis(1)))
        }

        fn input_shape(&self) -> usize {
            self.n_features
        }

        fn predict_interval(
            &self,
            inputs: &Array2<f64>,
        ) -> Result<Option<(Array1<f64>, Array1<f64>)>, Box<dyn std::error::Error>> {
            let pred = self.predict(inputs)?;
            let lower = pred.mapv(|v| v - self.interval_width);
            let upper = pred.mapv(|v| v + self.interval_width);
            Ok(Some((lower, upper)))
        }
    }

    #[test]
    fn test_config_builder() {
        let config = QuantileAttackConfig::builder()
            .attack_mode("break_coverage")
            .calibration_width(0.5)
            .build();

        assert_eq!(config.attack_mode, "break_coverage");
        assert_eq!(config.calibration_width, Some(0.5));
    }

    #[test]
    fn test_get_interval() {
        let attack = QuantileAttack::new(QuantileAttackConfig::default());
        let model = MockRegressionModel {
            n_features: 4,
            interval_width: 0.5,
        };

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let (pred, lower, upper) = attack.get_interval(&x, &model).unwrap();

        assert_eq!(pred, 10.0);
        assert_eq!(lower, 9.5);
        assert_eq!(upper, 10.5);
    }
}
