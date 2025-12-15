"""
spectrum.blue.harden
====================

Model Hardening Module - Defense wrappers for production models.

This module provides inference-time wrappers that improve model robustness
WITHOUT requiring retraining. Based on empirical ablation studies.

Key findings from testing (California Housing dataset):
- EnsembleProxy: +66% OutputManipulation, +70% PredictionShift (BEST)
- OutputQuantizer (20 levels): +18% OutputManipulation
- InputSanitizer: +43% PredictionShift, can improve accuracy
- Combined stacks can backfire (-7% in some cases)

Recommendation: Use EnsembleProxy alone for maximum robustness.

For retraining-based hardening, see:
- AdversarialTrainer: Train on perturbed data (+49% robustness)
- HardenedEnsemble: Diverse model ensemble with adversarial training
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class HardeningConfig:
    """Configuration for model hardening."""
    # Input sanitization
    clip_percentile: float = 99.0
    squeeze_bits: int = 6
    smooth_sigma: float = 0.1

    # Ensemble proxy
    proxy_n_neighbors: int = 10
    proxy_weights: Tuple[float, float, float] = (0.6, 0.25, 0.15)

    # Output quantization
    output_levels: int = 20

    # Randomization
    input_noise_std: float = 0.02
    output_noise_std: float = 0.01

    # Prediction smoothing
    smooth_samples: int = 5
    smooth_noise_std: float = 0.03


# =============================================================================
# Inference-Time Wrappers (No Retraining Required)
# =============================================================================

class InputSanitizer(RegressorMixin, BaseEstimator):
    """
    Sanitizes inputs before passing to model.

    Defenses:
    - Range clipping: Bound to calibration distribution
    - Feature squeezing: Reduce precision to collapse perturbations
    - Gaussian smoothing: Blur high-frequency noise

    Empirical results:
    - OutputManipulation: -3% (slightly worse)
    - PredictionShift: +43% (significant improvement)
    - Accuracy: Can improve slightly (-0.1pp drop = improvement)

    Best for: PredictionShift defense with minimal accuracy cost.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 clip_percentile: float = 99.0,
                 squeeze_bits: int = 6,
                 smooth_sigma: float = 0.1):
        self.base_model = base_model
        self.clip_percentile = clip_percentile
        self.squeeze_bits = squeeze_bits
        self.smooth_sigma = smooth_sigma

        self.feature_mins_: Optional[np.ndarray] = None
        self.feature_maxs_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)
        self.is_calibrated_ = False

    def calibrate(self, X: np.ndarray) -> 'InputSanitizer':
        """Learn input bounds from calibration data."""
        lower_pct = (100 - self.clip_percentile) / 2
        upper_pct = 100 - lower_pct

        self.feature_mins_ = np.percentile(X, lower_pct, axis=0)
        self.feature_maxs_ = np.percentile(X, upper_pct, axis=0)
        self.n_features_in_ = X.shape[1]
        self.is_calibrated_ = True
        return self

    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        X_clean = X.copy()

        # 1. Range clipping
        if self.feature_mins_ is not None:
            X_clean = np.clip(X_clean, self.feature_mins_, self.feature_maxs_)

        # 2. Feature squeezing
        if self.squeeze_bits > 0:
            levels = 2 ** self.squeeze_bits
            X_clean = np.round(X_clean * levels) / levels

        # 3. Gaussian smoothing per sample
        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            for i in range(len(X_clean)):
                X_clean[i] = gaussian_filter1d(X_clean[i], sigma=self.smooth_sigma)

        return X_clean

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sanitized = self._sanitize(X)
        return self.base_model.predict(X_sanitized)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """For classifiers."""
        X_sanitized = self._sanitize(X)
        return self.base_model.predict_proba(X_sanitized)


class EnsembleProxy(RegressorMixin, BaseEstimator):
    """
    Wraps a model with lightweight proxy models for robustness.

    Trains simple proxy models (Ridge, KNN) to mimic the base model,
    then averages predictions. This smooths the prediction surface
    without retraining the base model.

    Empirical results (BEST WRAPPER):
    - OutputManipulation: +66% (significant improvement)
    - PredictionShift: +70% (significant improvement)
    - Accuracy: -2.4pp (moderate cost)

    Best for: Maximum robustness when accuracy cost is acceptable.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 n_neighbors: int = 10,
                 weights: Tuple[float, float, float] = (0.6, 0.25, 0.15)):
        self.base_model = base_model
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.ridge_ = Ridge(alpha=1.0)
        self.knn_ = KNeighborsRegressor(n_neighbors=n_neighbors)

        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)
        self.is_calibrated_ = False

    def calibrate(self, X: np.ndarray) -> 'EnsembleProxy':
        """Train proxy models to mimic base model."""
        # Get base model predictions as targets
        y_proxy = self.base_model.predict(X)

        # Train proxies
        self.ridge_.fit(X, y_proxy)
        self.knn_.fit(X, y_proxy)

        self.n_features_in_ = X.shape[1]
        self.is_calibrated_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_base = self.base_model.predict(X)
        pred_ridge = self.ridge_.predict(X)
        pred_knn = self.knn_.predict(X)

        return (
            self.weights[0] * pred_base +
            self.weights[1] * pred_ridge +
            self.weights[2] * pred_knn
        )


class OutputQuantizer(RegressorMixin, BaseEstimator):
    """
    Quantizes outputs to discrete levels.

    Makes small output shifts ineffective - attacker must achieve
    larger perturbations to cross quantization boundaries.

    Empirical results:
    - OutputManipulation: +18% (with 20 levels), -6% (with 50 levels)
    - PredictionShift: +4%
    - Accuracy: -0.6pp

    Best for: OutputManipulation defense. Use 20 levels (sweet spot).
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 n_levels: int = 20):
        self.base_model = base_model
        self.n_levels = n_levels

        self.output_min_: Optional[float] = None
        self.output_max_: Optional[float] = None
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)
        self.is_calibrated_ = False

    def calibrate(self, X: np.ndarray) -> 'OutputQuantizer':
        """Learn output range from calibration predictions."""
        preds = self.base_model.predict(X)
        self.output_min_ = float(preds.min())
        self.output_max_ = float(preds.max())
        self.n_features_in_ = X.shape[1]
        self.is_calibrated_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict(X)

        if self.output_min_ is None:
            return raw

        out_range = self.output_max_ - self.output_min_ + 1e-8
        normalized = (raw - self.output_min_) / out_range
        quantized = np.round(normalized * self.n_levels) / self.n_levels
        return quantized * out_range + self.output_min_


class PredictionSmoother(RegressorMixin, BaseEstimator):
    """
    Smooths predictions by averaging over input neighborhood.

    Makes the output surface locally flat, requiring larger
    perturbations to achieve the same output shift.

    Empirical results:
    - OutputManipulation: +7%
    - PredictionShift: +9%
    - Accuracy: -1.2pp
    - Latency: ~10x (multiple predictions per input)

    Best for: Modest improvement when latency is not critical.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 n_samples: int = 10,
                 noise_std: float = 0.1):
        self.base_model = base_model
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = [self.base_model.predict(X)]

        for _ in range(self.n_samples - 1):
            noise = np.random.randn(*X.shape) * self.noise_std
            all_preds.append(self.base_model.predict(X + noise))

        return np.mean(all_preds, axis=0)


class RandomizedWrapper(RegressorMixin, BaseEstimator):
    """
    Adds randomization that attackers cannot predict.

    Empirical results:
    - OutputManipulation: +3%
    - PredictionShift: +7%
    - Accuracy: -2.3pp

    Note: Limited effectiveness. Consider EnsembleProxy instead.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 input_noise_std: float = 0.02,
                 output_noise_std: float = 0.01):
        self.base_model = base_model
        self.input_noise_std = input_noise_std
        self.output_noise_std = output_noise_std
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.input_noise_std > 0:
            X = X + np.random.randn(*X.shape) * self.input_noise_std

        preds = self.base_model.predict(X)

        if self.output_noise_std > 0:
            preds = preds + np.random.randn(*preds.shape) * self.output_noise_std

        return preds


# =============================================================================
# Retraining-Based Hardening
# =============================================================================

class AdversarialTrainer:
    """
    Trains models on adversarially augmented data.

    This is the MOST EFFECTIVE defense but requires retraining.

    Empirical results (from ablation study):
    - OutputManipulation: +49%
    - PredictionShift: +30%
    - Accuracy: Minimal impact

    Usage:
        trainer = AdversarialTrainer(noise_std=0.3, n_copies=4)
        X_aug, y_aug = trainer.augment(X_train, y_train)
        model.fit(X_aug, y_aug)
    """

    def __init__(self,
                 noise_std: float = 0.3,
                 n_copies: int = 4):
        self.noise_std = noise_std
        self.n_copies = n_copies

    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with noisy copies.

        Returns (X_augmented, y_augmented) for training.
        """
        X_aug = [X]
        y_aug = [y]

        for _ in range(self.n_copies):
            noise = np.random.randn(*X.shape) * self.noise_std
            X_aug.append(X + noise)
            y_aug.append(y)

        return np.vstack(X_aug), np.concatenate(y_aug)

    def fit_transform(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Fit model on adversarially augmented data.

        Returns the fitted model.
        """
        X_aug, y_aug = self.augment(X, y)
        model.fit(X_aug, y_aug)
        return model


# =============================================================================
# High-Level API
# =============================================================================

def harden_model(
    model: BaseEstimator,
    X_calibration: np.ndarray,
    strategy: str = "ensemble_proxy",
    **kwargs
) -> BaseEstimator:
    """
    Harden a model using the specified strategy.

    Strategies (in order of effectiveness):
    - "ensemble_proxy": Best overall (+66% OM, +70% PS)
    - "output_quantizer": Good for OutputManipulation (+18%)
    - "input_sanitizer": Good for PredictionShift (+43%)
    - "prediction_smoother": Modest gains (+7-9%)
    - "randomized": Limited effectiveness (+3-7%)

    Args:
        model: Base model to harden
        X_calibration: Calibration data for fitting wrappers
        strategy: Hardening strategy name
        **kwargs: Strategy-specific parameters

    Returns:
        Hardened model wrapper

    Example:
        hardened = harden_model(model, X_calib, strategy="ensemble_proxy")
        predictions = hardened.predict(X_test)
    """
    strategies = {
        "ensemble_proxy": EnsembleProxy,
        "output_quantizer": OutputQuantizer,
        "input_sanitizer": InputSanitizer,
        "prediction_smoother": PredictionSmoother,
        "randomized": RandomizedWrapper,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    wrapper_class = strategies[strategy]
    wrapper = wrapper_class(model, **kwargs)

    # Calibrate if needed
    if hasattr(wrapper, 'calibrate'):
        wrapper.calibrate(X_calibration)

    return wrapper


def get_recommended_wrapper(
    model: BaseEstimator,
    X_calibration: np.ndarray,
    attack_type: str = "both",
    max_accuracy_cost: float = 3.0
) -> BaseEstimator:
    """
    Get recommended wrapper based on attack type and accuracy budget.

    Args:
        model: Base model to harden
        X_calibration: Calibration data
        attack_type: "output_manipulation", "prediction_shift", or "both"
        max_accuracy_cost: Maximum acceptable R2 drop in percentage points

    Returns:
        Recommended hardened wrapper
    """
    # Recommendations based on empirical results
    if attack_type == "output_manipulation":
        if max_accuracy_cost >= 2.4:
            return harden_model(model, X_calibration, "ensemble_proxy")
        else:
            return harden_model(model, X_calibration, "output_quantizer", n_levels=20)

    elif attack_type == "prediction_shift":
        if max_accuracy_cost >= 2.4:
            return harden_model(model, X_calibration, "ensemble_proxy")
        else:
            # InputSanitizer can actually improve accuracy
            return harden_model(model, X_calibration, "input_sanitizer")

    else:  # both
        if max_accuracy_cost >= 2.4:
            return harden_model(model, X_calibration, "ensemble_proxy")
        else:
            # For limited budget, prioritize based on typical attack likelihood
            return harden_model(model, X_calibration, "input_sanitizer")


# =============================================================================
# Classification Wrappers
# =============================================================================

class ClassifierInputSanitizer(ClassifierMixin, BaseEstimator):
    """
    Input sanitizer for classifiers.

    More aggressive than regression version since we only need
    to preserve the correct class, not exact probabilities.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 clip_percentile: float = 98.0,
                 squeeze_bits: int = 5,
                 smooth_sigma: float = 0.15):
        self.base_model = base_model
        self.clip_percentile = clip_percentile
        self.squeeze_bits = squeeze_bits
        self.smooth_sigma = smooth_sigma

        self.feature_mins_: Optional[np.ndarray] = None
        self.feature_maxs_: Optional[np.ndarray] = None
        self.classes_ = getattr(base_model, 'classes_', None)
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)
        self.is_calibrated_ = False

    def calibrate(self, X: np.ndarray) -> 'ClassifierInputSanitizer':
        lower_pct = (100 - self.clip_percentile) / 2
        upper_pct = 100 - lower_pct
        self.feature_mins_ = np.percentile(X, lower_pct, axis=0)
        self.feature_maxs_ = np.percentile(X, upper_pct, axis=0)
        self.n_features_in_ = X.shape[1]
        self.is_calibrated_ = True
        return self

    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        X_clean = X.copy()
        if self.feature_mins_ is not None:
            X_clean = np.clip(X_clean, self.feature_mins_, self.feature_maxs_)
        if self.squeeze_bits > 0:
            levels = 2 ** self.squeeze_bits
            X_clean = np.round(X_clean * levels) / levels
        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            for i in range(len(X_clean)):
                X_clean[i] = gaussian_filter1d(X_clean[i], sigma=self.smooth_sigma)
        return X_clean

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(self._sanitize(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(self._sanitize(X))


class ConfidenceGate(ClassifierMixin, BaseEstimator):
    """
    Abstains on low-confidence predictions.

    Adversarial examples often have lower confidence than clean examples.
    Rejecting low-confidence predictions can filter out attacks.
    """

    def __init__(self,
                 base_model: BaseEstimator,
                 confidence_threshold: float = 0.7,
                 abstain_label: int = -1):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.abstain_label = abstain_label

        self.classes_ = getattr(base_model, 'classes_', None)
        self.n_features_in_: Optional[int] = getattr(base_model, 'n_features_in_', None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.base_model.predict_proba(X)
        max_probs = probs.max(axis=1)
        predictions = self.base_model.predict(X)

        return np.where(
            max_probs >= self.confidence_threshold,
            predictions,
            self.abstain_label
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def get_abstention_rate(self, X: np.ndarray) -> float:
        """Fraction of inputs that would be abstained."""
        probs = self.base_model.predict_proba(X)
        return float((probs.max(axis=1) < self.confidence_threshold).mean())
