use ndarray::{Array1, Array2, Axis};
use serde::{Serialize, Deserialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialMetrics {
    // Core robustness metrics
    pub attack_success_rate: f64,
    pub samples_tested: usize,
    pub samples_successful: usize,

    // Perturbation magnitudes (Evasion)
    pub empirical_robustness_l2: f64,
    pub empirical_robustness_linf: f64,
    pub min_perturbation_l2: f64,
    pub max_perturbation_l2: f64,
    pub median_perturbation_l2: f64,

    // Attack metadata
    pub attack_type: String,
    pub queries_used: usize,
    pub avg_queries_per_sample: f64,

    // Statistical measures
    pub perturbation_std_l2: Option<f64>,
    pub perturbation_std_linf: Option<f64>,
    pub confidence_interval_95: Option<(f64, f64)>,

    // Contextual/Optional metrics
    pub feature_sensitivity: Option<HashMap<String, f64>>,
    pub privacy_leakage_score: Option<f64>,
    pub true_positive_rate: Option<f64>,
    pub false_positive_rate: Option<f64>,
    pub extraction_fidelity: Option<f64>,
    pub extraction_queries: Option<usize>,
    pub poisoning_success_rate: Option<f64>,
    pub clean_accuracy_impact: Option<f64>,
}

impl AdversarialMetrics {
    /// Calculate Wilson score interval
    fn compute_wilson_ci(successes: usize, trials: usize, confidence: f64) -> (f64, f64) {
        if trials == 0 {
            return (0.0, 0.0);
        }
        
        let p = successes as f64 / trials as f64;
        let n = trials as f64;

        let norm = Normal::new(0.0, 1.0).unwrap();
        let z = norm.inverse_cdf((1.0 + confidence) / 2.0);
        let z2 = z.powi(2);

        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n) / denominator).sqrt();
        
        let lower = (center - margin).max(0.0);
        let upper = (center + margin).min(1.0);

        (lower, upper)
    }

    /// constructor for evasion attacks
    pub fn for_evasion(
        attack_type: &str,
        y_original: &[usize],
        y_adversarial: &[usize],
        x_original: &ndarray::Array2<f64>,
        x_adversarial: &ndarray::Array2<f64>,
        queries_used: usize,
    ) -> Self {
        // Implementation logic placeholder...
        let samples_tested = y_original.len();

        if samples_tested == 0 {
            return Self::empty(attack_type);
        }

        let success_mask:Vec<bool> = y_original.iter()
                                .zip(y_adversarial.iter())
                                .map(|(orig, adv)| orig != adv)
                                .collect();
        
        let samples_successful = success_mask.iter()
                                            .filter(|&&x| x).count();
        let attack_success_rate = samples_successful as f64 / samples_tested as f64;
        
        let perturbations = x_adversarial - x_original;
        
        let successful_l2 = if samples_successful > 0 {
            let l2_norms = row_wise_l2(&perturbations);
            apply_boolean_mask(&l2_norms, &success_mask)
        } else {
            vec![0.0]
        };
        
        let successful_linf = if samples_successful > 0 {
            let linf_norms = row_wise_linf(&perturbations);
            apply_boolean_mask(&linf_norms, &success_mask)
        } else {
            vec![0.0]
        };

        let (empirical_robustness_l2, min_perturbation_l2, max_perturbation_l2, median_perturbation_l2, perturbation_std_l2) = compute_stats(&successful_l2);
        let (empirical_robustness_inf, _, _, _, perturbation_std_linf) = compute_stats(&successful_linf);

        Self {
            attack_success_rate: attack_success_rate,
            samples_tested: samples_tested,
            samples_successful: samples_successful,
            empirical_robustness_l2: empirical_robustness_l2,
            empirical_robustness_linf: empirical_robustness_inf,
            min_perturbation_l2: min_perturbation_l2,
            max_perturbation_l2: max_perturbation_l2,
            median_perturbation_l2: median_perturbation_l2,
            attack_type: attack_type.to_string(),
            queries_used: queries_used,
            avg_queries_per_sample: if samples_tested > 0 {
                queries_used as f64 / samples_tested as f64
            } else { 0.0 },
            perturbation_std_l2: Some(perturbation_std_l2),
            perturbation_std_linf: Some(perturbation_std_linf),
            confidence_interval_95: None,  
            feature_sensitivity: None,     
            privacy_leakage_score: None,   
            true_positive_rate: None,      
            false_positive_rate: None,     
            extraction_fidelity: None,     
            extraction_queries: None,      
            poisoning_success_rate: None,  
            clean_accuracy_impact: None,   
        }
    }
    
    fn empty(attack_type: &str) -> Self {
        Self {
            attack_success_rate: 0.0,
            samples_tested: 0,
            samples_successful: 0,
            empirical_robustness_l2: 0.0,
            empirical_robustness_linf: 0.0,
            min_perturbation_l2: 0.0,
            max_perturbation_l2: 0.0,
            median_perturbation_l2: 0.0,
            attack_type: attack_type.to_string(),
            queries_used: 0,
            avg_queries_per_sample: 0.0,
            perturbation_std_l2: None,
            perturbation_std_linf: None,
            confidence_interval_95: None,
            feature_sensitivity: None,
            privacy_leakage_score: None,
            true_positive_rate: None,
            false_positive_rate: None,
            extraction_fidelity: None,
            extraction_queries: None,
            poisoning_success_rate: None,
            clean_accuracy_impact: None,
        }
    // Additional constructors (for_inference, etc.) would follow...
    }
}
// --- Helpers ----

fn compute_stats(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0); // (mean, min, max, median)
    }
    
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    
    // Sort for median/min/max
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let median = sorted[sorted.len() / 2];

    let variance = values.iter().map(|val| {
        let diff = mean - *val;
        diff * diff
    }).sum::<f64>() / values.len() as f64;

    (mean, min, max, median, variance.sqrt())
}

fn row_wise_l2(arr: &Array2<f64>) -> Array1<f64> {
    arr.map_axis(Axis(1), |row| row.dot(&row).sqrt())
}

fn row_wise_linf(arr: &Array2<f64>) -> Array1<f64> {
    arr.map_axis(Axis(1), |row| {
        row.fold(0.0f64, |max, &x| max.max(x.abs()))
    })
}

fn apply_boolean_mask(values: &Array1<f64>, mask: &[bool]) -> Vec<f64> {
    assert_eq!(values.len(), mask.len(), "Values and mask lengths must be equal");

    values.iter()
        .zip(mask.iter())
        .filter_map(|(value, &is_selected)| {
            if is_selected { Some(*value) } else { None }
        })
        .collect()
}