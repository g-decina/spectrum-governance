/// Output Manipulation Attack for regression models
///
/// This attack discretizes continuous regression outputs into K bins (pseudo-classes)
/// and uses a HopSkipJump-like approach to find perturbations that shift predictions
/// to a different bin.
///
/// # Algorithm
///
/// 1. Discretize output range into K bins based on training data
/// 2. For each sample, find its current bin
/// 3. Use random search to find a point in a different bin (initialization)
/// 4. Binary search to find the bin boundary
/// 5. Iteratively refine to minimize perturbation while staying in different bin
///
/// # Regulatory Context
///
/// Useful for testing robustness of pricing models, risk scores, and any
/// regression model where output changes beyond a threshold have regulatory
/// implications.

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::Normal;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::regression_common::{
    bootstrap_ci_mean, ConfidenceLevel, RegressionAttackError, RegressionMetrics, RegressionModel,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Output Manipulation Attack.
#[derive(Clone, Debug)]
pub struct OutputManipulationConfig {
    /// Number of bins to discretize output range
    pub n_bins: usize,
    /// Maximum iterations per sample
    pub max_iter: usize,
    /// Maximum model queries per sample
    pub max_eval: usize,
    /// Initial geometric progression for boundary search
    pub stepsize_search: f64,
    /// Gradient estimation samples
    pub num_grad_samples: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Feature clip bounds (min, max) - None for no clipping
    pub clip_bounds: Option<(f64, f64)>,
    /// Bin edge mode: "uniform" or "quantile"
    pub bin_mode: String,
    /// Confidence level for bootstrap CI on mean attempts
    pub confidence_level: ConfidenceLevel,
    /// Number of bootstrap resamples for CI computation
    pub n_bootstrap: usize,
    /// Set a seed for random noise generation (default: None for entropy-based)
    pub seed: Option<u64>,
}

impl Default for OutputManipulationConfig {
    fn default() -> Self {
        Self {
            n_bins: 5,
            max_iter: 50,
            max_eval: 5000,
            stepsize_search: 0.01,
            num_grad_samples: 50,
            parallel: true,
            clip_bounds: None,
            bin_mode: "uniform".to_string(),
            confidence_level: ConfidenceLevel::default(),
            n_bootstrap: 10_000,
            seed: None,
        }
    }
}

impl OutputManipulationConfig {
    pub fn builder() -> OutputManipulationConfigBuilder {
        OutputManipulationConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct OutputManipulationConfigBuilder {
    config: OutputManipulationConfig,
}

impl OutputManipulationConfigBuilder {
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.config.n_bins = n_bins;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.config.max_eval = max_eval;
        self
    }

    pub fn stepsize_search(mut self, stepsize_search: f64) -> Self {
        self.config.stepsize_search = stepsize_search;
        self
    }

    pub fn num_grad_samples(mut self, num_grad_samples: usize) -> Self {
        self.config.num_grad_samples = num_grad_samples;
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

    pub fn bin_mode(mut self, mode: &str) -> Self {
        self.config.bin_mode = mode.to_string();
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

    pub fn build(self) -> OutputManipulationConfig {
        self.config
    }
}

// ============================================================================
// Attack Implementation
// ============================================================================

/// Output Manipulation Attack for regression models.
///
/// Discretizes continuous outputs into bins and finds minimal perturbations
/// that shift the predicted bin.
pub struct OutputManipulationAttack {
    config: OutputManipulationConfig,
}

impl OutputManipulationAttack {
    pub fn new(config: OutputManipulationConfig) -> Self {
        Self { config }
    }

    /// Compute bin edges from training data or predictions.
    fn compute_bin_edges(&self, values: &[f64]) -> Vec<f64> {
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mut edges = Vec::with_capacity(self.config.n_bins + 1);

        if self.config.bin_mode == "quantile" {
            // Quantile-based bins
            for i in 0..=self.config.n_bins {
                let idx = (i * (n - 1)) / self.config.n_bins;
                edges.push(sorted[idx]);
            }
        } else {
            // Uniform bins
            let min_val = sorted[0];
            let max_val = sorted[n - 1];
            let step = (max_val - min_val) / self.config.n_bins as f64;

            for i in 0..=self.config.n_bins {
                edges.push(min_val + step * i as f64);
            }
        }

        edges
    }

    /// Assign a value to a bin index.
    fn value_to_bin(&self, value: f64, bin_edges: &[f64]) -> usize {
        for i in 1..bin_edges.len() {
            if value <= bin_edges[i] {
                return i - 1;
            }
        }
        bin_edges.len() - 2 // Last bin
    }

    /// Predict and get bin index for a single sample.
    fn predict_bin(
        &self,
        x: &Array1<f64>,
        model: &dyn RegressionModel,
        bin_edges: &[f64],
        query_counter: &Arc<AtomicUsize>,
    ) -> Result<usize, RegressionAttackError> {
        let x_2d = x.clone().insert_axis(Axis(0));
        query_counter.fetch_add(1, Ordering::Relaxed);

        let pred = model
            .predict(&x_2d)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        Ok(self.value_to_bin(pred[0], bin_edges))
    }

    /// Initialize adversarial example by random search.
    fn initialize_adversarial(
        &self,
        x_original: &Array1<f64>,
        original_bin: usize,
        model: &dyn RegressionModel,
        bin_edges: &[f64],
        query_counter: &Arc<AtomicUsize>,
    ) -> Result<Option<Array1<f64>>, RegressionAttackError> {
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Try random perturbations to find different bin
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0] {
            for _ in 0..100 {
                if query_counter.load(Ordering::Relaxed) >= self.config.max_eval {
                    return Ok(None);
                }

                let noise: Array1<f64> = Array1::from_iter(
                    (0..x_original.len()).map(|_| normal.sample(&mut rng) * scale),
                );

                let x_candidate = x_original + &noise;
                let x_clipped = self.clip(&x_candidate);

                let candidate_bin =
                    self.predict_bin(&x_clipped, model, bin_edges, query_counter)?;

                if candidate_bin != original_bin {
                    return Ok(Some(x_clipped));
                }
            }
        }

        Ok(None)
    }

    /// Clip perturbation to bounds.
    fn clip(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.clip_bounds {
            Some((min, max)) => x.mapv(|v| v.max(min).min(max)),
            None => x.clone(),
        }
    }

    /// Binary search to find bin boundary.
    fn binary_search_boundary(
        &self,
        x_original: &Array1<f64>,
        x_adv: &Array1<f64>,
        original_bin: usize,
        model: &dyn RegressionModel,
        bin_edges: &[f64],
        query_counter: &Arc<AtomicUsize>,
    ) -> Result<Array1<f64>, RegressionAttackError> {
        let mut low = 0.0;
        let mut high = 1.0;

        while (high - low) > self.config.stepsize_search {
            if query_counter.load(Ordering::Relaxed) >= self.config.max_eval {
                break;
            }

            let mid = (low + high) / 2.0;
            let x_mid = (1.0 - mid) * x_original + mid * x_adv;
            let x_mid_clipped = self.clip(&x_mid);

            let mid_bin = self.predict_bin(&x_mid_clipped, model, bin_edges, query_counter)?;

            if mid_bin != original_bin {
                high = mid;
            } else {
                low = mid;
            }
        }

        let x_boundary = (1.0 - high) * x_original + high * x_adv;
        Ok(self.clip(&x_boundary))
    }

    /// Estimate gradient using finite differences (zeroth-order).
    fn estimate_gradient(
        &self,
        x_boundary: &Array1<f64>,
        x_original: &Array1<f64>,
        model: &dyn RegressionModel,
        query_counter: &Arc<AtomicUsize>,
    ) -> Result<Array1<f64>, RegressionAttackError> {
        let n_features = x_boundary.len();
        let mut rng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Current distance
        let d_current = (x_boundary - x_original).mapv(|v| v.powi(2)).sum().sqrt();

        // Collect gradient estimates
        let mut gradients: Vec<Array1<f64>> = Vec::new();

        let delta = d_current / 100.0; // Step size proportional to distance

        for _ in 0..self.config.num_grad_samples {
            if query_counter.load(Ordering::Relaxed) >= self.config.max_eval {
                break;
            }

            // Random direction
            let u: Array1<f64> =
                Array1::from_iter((0..n_features).map(|_| normal.sample(&mut rng)));
            let u_norm = u.mapv(|v| v.powi(2)).sum().sqrt();
            let u_normalized = &u / u_norm;

            // Evaluate at perturbed point
            let x_plus = x_boundary + &(&u_normalized * delta);
            let x_plus_clipped = self.clip(&x_plus);

            let x_plus_2d = x_plus_clipped.clone().insert_axis(Axis(0));
            query_counter.fetch_add(1, Ordering::Relaxed);

            let pred_plus = model
                .predict(&x_plus_2d)
                .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

            // Gradient in random direction
            let grad_est = u_normalized * pred_plus[0];
            gradients.push(grad_est);
        }

        if gradients.is_empty() {
            return Ok(Array1::zeros(n_features));
        }

        // Average gradient
        let mut avg_grad = Array1::zeros(n_features);
        for g in &gradients {
            avg_grad = avg_grad + g;
        }
        avg_grad = avg_grad / gradients.len() as f64;

        // Normalize
        let norm = avg_grad.mapv(|v: f64| v.powi(2)).sum().sqrt();
        if norm > 1e-10 {
            avg_grad = avg_grad / norm;
        }

        Ok(avg_grad)
    }

    /// Attack a single sample.
    ///
    /// Returns:
    /// - Adversarial example
    /// - Whether attack succeeded
    /// - Total queries used
    /// - Queries to first success (None if never succeeded)
    fn attack_single(
        &self,
        x_original: &Array1<f64>,
        model: &dyn RegressionModel,
        bin_edges: &[f64],
    ) -> Result<(Array1<f64>, bool, usize, Option<usize>), RegressionAttackError> {
        let query_counter = Arc::new(AtomicUsize::new(0));
        let mut queries_to_first_success: Option<usize> = None;

        // Get original bin
        let original_bin = self.predict_bin(x_original, model, bin_edges, &query_counter)?;

        // Initialize adversarial example
        let x_adv_init =
            self.initialize_adversarial(x_original, original_bin, model, bin_edges, &query_counter)?;

        let mut x_adv = match x_adv_init {
            Some(x) => {
                // Record queries to first success (when we found a different bin)
                queries_to_first_success = Some(query_counter.load(Ordering::Relaxed));
                x
            }
            None => {
                // Failed to find any adversarial
                return Ok((
                    x_original.clone(),
                    false,
                    query_counter.load(Ordering::Relaxed),
                    None,
                ));
            }
        };

        // Binary search to boundary
        x_adv =
            self.binary_search_boundary(x_original, &x_adv, original_bin, model, bin_edges, &query_counter)?;

        // Iterative refinement
        for _iter in 0..self.config.max_iter {
            if query_counter.load(Ordering::Relaxed) >= self.config.max_eval {
                break;
            }

            // Estimate gradient
            let gradient =
                self.estimate_gradient(&x_adv, x_original, model, &query_counter)?;

            // Step along gradient (towards original)
            let d_current = (&x_adv - x_original).mapv(|v| v.powi(2)).sum().sqrt();
            let step_size = d_current * 0.1;

            // Move towards original along boundary
            let direction_to_original = x_original - &x_adv;
            let dir_norm = direction_to_original.mapv(|v| v.powi(2)).sum().sqrt();

            if dir_norm < 1e-10 {
                break;
            }

            let step_direction = &direction_to_original / dir_norm;

            // Combine gradient and direction to original
            let combined = &step_direction - &gradient * 0.5;
            let combined_norm = combined.mapv(|v| v.powi(2)).sum().sqrt();
            let combined_normalized = if combined_norm > 1e-10 {
                &combined / combined_norm
            } else {
                step_direction.clone()
            };

            let x_candidate = &x_adv + &combined_normalized * step_size;
            let x_candidate_clipped = self.clip(&x_candidate);

            // Check if still adversarial
            let candidate_bin =
                self.predict_bin(&x_candidate_clipped, model, bin_edges, &query_counter)?;

            if candidate_bin != original_bin {
                // Still adversarial, search for new boundary
                x_adv = self.binary_search_boundary(
                    x_original,
                    &x_candidate_clipped,
                    original_bin,
                    model,
                    bin_edges,
                    &query_counter,
                )?;
            }
        }

        // Verify final result
        let final_bin = self.predict_bin(&x_adv, model, bin_edges, &query_counter)?;
        let success = final_bin != original_bin;

        Ok((x_adv, success, query_counter.load(Ordering::Relaxed), queries_to_first_success))
    }

    /// Run the attack on a batch of samples.
    pub fn run(
        &self,
        model: &dyn RegressionModel,
        x: &Array2<f64>,
        y_reference: Option<&Array1<f64>>,
    ) -> Result<(Array2<f64>, RegressionMetrics), RegressionAttackError> {
        let n_samples = x.nrows();

        // Get predictions for bin computation
        let predictions = model
            .predict(x)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        // Use reference values or predictions for bin edges
        let bin_values: Vec<f64> = match y_reference {
            Some(y) => y.iter().copied().collect(),
            None => predictions.iter().copied().collect(),
        };

        let bin_edges = self.compute_bin_edges(&bin_values);

        // Run attacks - now returns (adv, success, total_queries, queries_to_first_success)
        let results: Vec<(Array1<f64>, bool, usize, Option<usize>)> = if self.config.parallel {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let x_i = x.row(i).to_owned();
                    self.attack_single(&x_i, model, &bin_edges)
                        .unwrap_or_else(|_| (x_i.clone(), false, 0, None))
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let x_i = x.row(i).to_owned();
                    self.attack_single(&x_i, model, &bin_edges)
                        .unwrap_or_else(|_| (x_i.clone(), false, 0, None))
                })
                .collect()
        };

        // Aggregate results
        let mut x_adv = Array2::zeros(x.dim());
        let mut successes = Vec::new();
        let mut total_queries = 0;
        let mut attempts_list: Vec<Option<usize>> = Vec::new();

        for (i, (adv, success, queries, attempts_to_success)) in results.into_iter().enumerate() {
            x_adv.row_mut(i).assign(&adv);
            successes.push(success);
            total_queries += queries;
            attempts_list.push(attempts_to_success);
        }

        // Get adversarial predictions
        let y_original = predictions;
        let y_adv = model
            .predict(&x_adv)
            .map_err(|e| RegressionAttackError::ModelError(e.to_string()))?;

        // Compute metrics
        let metrics = self.compute_metrics(
            x,
            &x_adv,
            &y_original,
            &y_adv,
            &successes,
            total_queries,
            &bin_edges,
            &attempts_list,
        );

        Ok((x_adv, metrics))
    }

    fn compute_metrics(
        &self,
        x_orig: &Array2<f64>,
        x_adv: &Array2<f64>,
        y_orig: &Array1<f64>,
        y_adv: &Array1<f64>,
        successes: &[bool],
        queries: usize,
        bin_edges: &[f64],
        attempts_list: &[Option<usize>],
    ) -> RegressionMetrics {
        let n = successes.len();

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

        // Compute perturbation norms
        let (l2_norms, linf_norms) = RegressionMetrics::compute_perturbation_stats(x_orig, x_adv);
        let (mean_l2, min_l2, max_l2, median_l2) = RegressionMetrics::l2_stats(&l2_norms);

        // Compute prediction shifts
        let shifts: Vec<f64> = y_adv
            .iter()
            .zip(y_orig.iter())
            .map(|(&a, &o)| a - o)
            .collect();
        let abs_shifts: Vec<f64> = shifts.iter().map(|s| s.abs()).collect();

        // Compute bin distances
        let bin_distances: Vec<i32> = y_orig
            .iter()
            .zip(y_adv.iter())
            .map(|(&o, &a)| {
                let bin_o = self.value_to_bin(o, bin_edges) as i32;
                let bin_a = self.value_to_bin(a, bin_edges) as i32;
                (bin_a - bin_o).abs()
            })
            .collect();

        RegressionMetrics {
            attack_type: "OutputManipulation".to_string(),
            samples_tested: n,
            samples_succeeded,
            samples_never_succeeded,
            mean_attempts_to_success: mean_attempts,
            median_attempts_to_success: median_attempts,
            min_attempts,
            max_attempts,
            attempts_ci,
            confidence_level: self.config.confidence_level.as_f64(),
            attack_success_rate: None, // Not used for OutputManipulation
            mean_perturbation_l2: mean_l2,
            mean_perturbation_linf: linf_norms.iter().sum::<f64>() / n as f64,
            min_perturbation_l2: min_l2,
            max_perturbation_l2: max_l2,
            median_perturbation_l2: median_l2,
            mean_prediction_shift: shifts.iter().sum::<f64>() / n as f64,
            max_prediction_shift: shifts.iter().fold(0.0f64, |a, &b| a.max(b.abs())),
            mean_absolute_shift: abs_shifts.iter().sum::<f64>() / n as f64,
            queries_used: queries,
            avg_queries_per_sample: queries as f64 / n as f64,
            bin_flip_rate: Some(samples_succeeded as f64 / n as f64),
            mean_bin_distance: Some(bin_distances.iter().sum::<i32>() as f64 / n as f64),
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
        let config = OutputManipulationConfig::builder()
            .n_bins(10)
            .max_iter(100)
            .parallel(false)
            .build();

        assert_eq!(config.n_bins, 10);
        assert_eq!(config.max_iter, 100);
        assert!(!config.parallel);
    }

    #[test]
    fn test_bin_edges_uniform() {
        let attack = OutputManipulationAttack::new(OutputManipulationConfig {
            n_bins: 4,
            bin_mode: "uniform".to_string(),
            ..Default::default()
        });

        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let edges = attack.compute_bin_edges(&values);

        assert_eq!(edges.len(), 5);
        assert_eq!(edges[0], 0.0);
        assert_eq!(edges[4], 4.0);
    }
}
