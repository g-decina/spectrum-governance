#!/usr/bin/env python3
"""
Wrapper Effectiveness Test

Empirically measures which inference-time wrappers improve robustness
for regression models WITHOUT retraining.

Tests:
1. Input Sanitizer (clip, squeeze, smooth)
2. Prediction Smoother (neighborhood averaging)
3. Output Quantizer
4. Randomized Wrapper
5. Ensemble Proxy
6. Combined Stack

Measures:
- OutputManipulation: Mean attempts to flip output bin
- PredictionShift: Mean attempts to achieve 10% shift
- Clean accuracy (R2 score)
"""

import os
import sys
import warnings
import numpy as np
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d

from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper


# =============================================================================
# Wrapper Implementations
# =============================================================================

class BaselineModel:
    """Unwrapped model for comparison."""
    def __init__(self, model):
        self.model = model
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = "Baseline (no wrapper)"

    def predict(self, X):
        return self.model.predict(X)


class InputSanitizer:
    """Clips, squeezes, and smooths inputs."""
    def __init__(self, model, clip_percentile=99, squeeze_bits=6, smooth_sigma=0.1):
        self.model = model
        self.clip_percentile = clip_percentile
        self.squeeze_bits = squeeze_bits
        self.smooth_sigma = smooth_sigma
        self.feature_mins = None
        self.feature_maxs = None
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = f"InputSanitizer(clip={clip_percentile}%, bits={squeeze_bits}, σ={smooth_sigma})"

    def calibrate(self, X):
        lower_pct = (100 - self.clip_percentile) / 2
        upper_pct = 100 - lower_pct
        self.feature_mins = np.percentile(X, lower_pct, axis=0)
        self.feature_maxs = np.percentile(X, upper_pct, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X_clean = X.copy()
        if self.feature_mins is not None:
            X_clean = np.clip(X_clean, self.feature_mins, self.feature_maxs)
        if self.squeeze_bits:
            levels = 2 ** self.squeeze_bits
            X_clean = np.round(X_clean * levels) / levels
        if self.smooth_sigma > 0:
            for i in range(len(X_clean)):
                X_clean[i] = gaussian_filter1d(X_clean[i], sigma=self.smooth_sigma)
        return self.model.predict(X_clean)


class PredictionSmoother:
    """Averages predictions over noisy input neighborhood."""
    def __init__(self, model, n_samples=10, noise_std=0.05):
        self.model = model
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = f"PredictionSmoother(n={n_samples}, σ={noise_std})"

    def predict(self, X):
        all_preds = [self.model.predict(X)]
        for _ in range(self.n_samples - 1):
            noise = np.random.randn(*X.shape) * self.noise_std
            all_preds.append(self.model.predict(X + noise))
        return np.mean(all_preds, axis=0)


class OutputQuantizer:
    """Quantizes outputs to discrete levels."""
    def __init__(self, model, n_levels=20):
        self.model = model
        self.n_levels = n_levels
        self.output_range = None
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = f"OutputQuantizer(levels={n_levels})"

    def calibrate(self, X):
        preds = self.model.predict(X)
        self.output_range = (preds.min(), preds.max())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        raw = self.model.predict(X)
        if self.output_range is None:
            return raw
        out_min, out_max = self.output_range
        normalized = (raw - out_min) / (out_max - out_min + 1e-8)
        quantized = np.round(normalized * self.n_levels) / self.n_levels
        return quantized * (out_max - out_min) + out_min


class RandomizedWrapper:
    """Adds random noise to inputs and outputs."""
    def __init__(self, model, input_noise=0.02, output_noise=0.01):
        self.model = model
        self.input_noise = input_noise
        self.output_noise = output_noise
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = f"Randomized(in={input_noise}, out={output_noise})"

    def predict(self, X):
        X_noisy = X + np.random.randn(*X.shape) * self.input_noise
        preds = self.model.predict(X_noisy)
        return preds + np.random.randn(*preds.shape) * self.output_noise


class EnsembleProxy:
    """Adds lightweight proxy models."""
    def __init__(self, model, n_neighbors=10):
        self.model = model
        self.n_neighbors = n_neighbors
        self.ridge = Ridge(alpha=1.0)
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.weights = [0.6, 0.25, 0.15]
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = f"EnsembleProxy(knn={n_neighbors})"

    def calibrate(self, X):
        y_proxy = self.model.predict(X)
        self.ridge.fit(X, y_proxy)
        self.knn.fit(X, y_proxy)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        pred_base = self.model.predict(X)
        pred_ridge = self.ridge.predict(X)
        pred_knn = self.knn.predict(X)
        return (self.weights[0] * pred_base +
                self.weights[1] * pred_ridge +
                self.weights[2] * pred_knn)


class CombinedStack:
    """All wrappers combined."""
    def __init__(self, model,
                 clip_percentile=99, squeeze_bits=6,
                 input_noise=0.02,
                 smooth_samples=5, smooth_noise=0.03,
                 output_levels=50):
        self.model = model
        self.clip_percentile = clip_percentile
        self.squeeze_bits = squeeze_bits
        self.input_noise = input_noise
        self.smooth_samples = smooth_samples
        self.smooth_noise = smooth_noise
        self.output_levels = output_levels

        self.feature_mins = None
        self.feature_maxs = None
        self.output_range = None
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.name = "CombinedStack(sanitize+random+smooth+quantize)"

    def calibrate(self, X):
        lower_pct = (100 - self.clip_percentile) / 2
        upper_pct = 100 - lower_pct
        self.feature_mins = np.percentile(X, lower_pct, axis=0)
        self.feature_maxs = np.percentile(X, upper_pct, axis=0)
        preds = self.model.predict(X)
        self.output_range = (preds.min(), preds.max())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        # 1. Sanitize
        X_clean = np.clip(X, self.feature_mins, self.feature_maxs)
        if self.squeeze_bits:
            levels = 2 ** self.squeeze_bits
            X_clean = np.round(X_clean * levels) / levels

        # 2. Randomize
        X_clean = X_clean + np.random.randn(*X_clean.shape) * self.input_noise

        # 3. Smooth predictions
        all_preds = [self.model.predict(X_clean)]
        for _ in range(self.smooth_samples - 1):
            noise = np.random.randn(*X_clean.shape) * self.smooth_noise
            all_preds.append(self.model.predict(X_clean + noise))
        raw = np.mean(all_preds, axis=0)

        # 4. Quantize
        if self.output_range:
            out_min, out_max = self.output_range
            normalized = (raw - out_min) / (out_max - out_min + 1e-8)
            quantized = np.round(normalized * self.output_levels) / self.output_levels
            return quantized * (out_max - out_min) + out_min
        return raw


# =============================================================================
# Test Runner
# =============================================================================

def run_attacks(model, X_test, n_samples=50):
    """Run attacks and return metrics."""
    X = X_test[:n_samples]
    results = {}

    # OutputManipulation
    try:
        attack = OutputManipulationWrapper(
            base_model=model,
            n_bins=5,
            max_iter=30,
            max_eval=2000,
            parallel=True,
            confidence_level=0.99,
        )
        metrics = attack.run(X)
        results["om_attempts"] = metrics.mean_attempts_to_success
        results["om_ci"] = metrics.attempts_ci
        results["om_l2"] = metrics.mean_perturbation_l2
    except Exception as e:
        results["om_error"] = str(e)

    # PredictionShift
    try:
        attack = PredictionShiftWrapper(
            base_model=model,
            epsilon=1.0,
            max_iter=50,
            n_directions=30,
            parallel=True,
            success_threshold=0.10,
            confidence_level=0.99,
        )
        metrics = attack.run(X)
        results["ps_attempts"] = metrics.mean_attempts_to_success
        results["ps_ci"] = metrics.attempts_ci
        results["ps_l2"] = metrics.mean_perturbation_l2
    except Exception as e:
        results["ps_error"] = str(e)

    return results


def main():
    print("=" * 75)
    print("  WRAPPER EFFECTIVENESS TEST")
    print("  Measuring inference-time defenses WITHOUT retraining")
    print("=" * 75)
    print()

    # Load data
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data.values, housing.target.values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    # Train base model (this is our "production model" we can't retrain)
    print("Training base MLP model (simulating production model)...")
    base_model = MLPRegressor(
        hidden_layer_sizes=(128,),
        activation='tanh',
        alpha=0.5,
        max_iter=500,
        early_stopping=True,
        random_state=42,
        verbose=False
    )
    base_model.fit(X_train_scaled, y_train)
    base_r2 = r2_score(y_test, base_model.predict(X_test_scaled))
    print(f"Base model R2: {base_r2:.4f}")
    print()

    # Define wrappers to test
    wrappers = [
        BaselineModel(base_model),

        # Individual wrappers
        InputSanitizer(base_model, clip_percentile=99, squeeze_bits=6, smooth_sigma=0.1),
        InputSanitizer(base_model, clip_percentile=95, squeeze_bits=5, smooth_sigma=0.2),
        PredictionSmoother(base_model, n_samples=5, noise_std=0.05),
        PredictionSmoother(base_model, n_samples=10, noise_std=0.10),
        OutputQuantizer(base_model, n_levels=20),
        OutputQuantizer(base_model, n_levels=50),
        RandomizedWrapper(base_model, input_noise=0.02, output_noise=0.01),
        RandomizedWrapper(base_model, input_noise=0.05, output_noise=0.02),
        EnsembleProxy(base_model, n_neighbors=10),

        # Combined
        CombinedStack(base_model),
    ]

    # Calibrate wrappers that need it
    for wrapper in wrappers:
        if hasattr(wrapper, 'calibrate'):
            wrapper.calibrate(X_calib_scaled)

    # Test each wrapper
    results = []
    n_samples = 50  # Samples to attack

    print(f"Testing {len(wrappers)} wrapper configurations on {n_samples} samples...")
    print("-" * 75)

    for wrapper in wrappers:
        print(f"\nTesting: {wrapper.name}")

        # Measure clean accuracy
        preds = wrapper.predict(X_test_scaled)
        r2 = r2_score(y_test, preds)

        # Measure latency
        start = time.time()
        for _ in range(10):
            _ = wrapper.predict(X_test_scaled[:100])
        latency = (time.time() - start) / 10 * 1000  # ms per 100 samples

        # Run attacks
        attack_results = run_attacks(wrapper, X_test_scaled, n_samples=n_samples)

        result = {
            "name": wrapper.name,
            "r2": r2,
            "r2_drop": (base_r2 - r2) * 100,  # percentage points
            "latency_ms": latency,
            **attack_results
        }
        results.append(result)

        # Print progress
        om = result.get('om_attempts', float('nan'))
        ps = result.get('ps_attempts', float('nan'))
        print(f"  R2: {r2:.4f} ({result['r2_drop']:+.1f}pp) | "
              f"OM: {om:.1f} attempts | PS: {ps:.1f} attempts | "
              f"Latency: {latency:.1f}ms")

    # Summary table
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    # Get baseline for comparison
    baseline_om = results[0].get('om_attempts', 1)
    baseline_ps = results[0].get('ps_attempts', 1)

    print()
    print(f"{'Wrapper':<50} {'R2 Drop':<10} {'OM Δ':<12} {'PS Δ':<12} {'Latency':<10}")
    print("-" * 94)

    for r in results:
        om = r.get('om_attempts', float('nan'))
        ps = r.get('ps_attempts', float('nan'))

        if not np.isnan(om) and baseline_om > 0:
            om_delta = (om / baseline_om - 1) * 100
            om_str = f"{om_delta:+.0f}%"
        else:
            om_str = "N/A"

        if not np.isnan(ps) and baseline_ps > 0:
            ps_delta = (ps / baseline_ps - 1) * 100
            ps_str = f"{ps_delta:+.0f}%"
        else:
            ps_str = "N/A"

        print(f"{r['name']:<50} {r['r2_drop']:+.1f}pp     {om_str:<12} {ps_str:<12} {r['latency_ms']:.0f}ms")

    # Recommendations
    print("\n" + "=" * 75)
    print("  RECOMMENDATIONS")
    print("=" * 75)

    # Find best wrappers
    best_om = max(results[1:], key=lambda r: r.get('om_attempts', 0))
    best_ps = max(results[1:], key=lambda r: r.get('ps_attempts', 0))
    best_balanced = max(results[1:], key=lambda r: (
        r.get('om_attempts', 0) / baseline_om +
        r.get('ps_attempts', 0) / baseline_ps -
        r['r2_drop'] / 2
    ))

    print()
    print(f"Best for OutputManipulation: {best_om['name']}")
    print(f"  {best_om.get('om_attempts', 0):.1f} attempts "
          f"({(best_om.get('om_attempts', 0) / baseline_om - 1) * 100:+.0f}% vs baseline)")

    print()
    print(f"Best for PredictionShift: {best_ps['name']}")
    print(f"  {best_ps.get('ps_attempts', 0):.1f} attempts "
          f"({(best_ps.get('ps_attempts', 0) / baseline_ps - 1) * 100:+.0f}% vs baseline)")

    print()
    print(f"Best balanced (robustness vs accuracy): {best_balanced['name']}")
    print(f"  OM: {best_balanced.get('om_attempts', 0):.1f}, "
          f"PS: {best_balanced.get('ps_attempts', 0):.1f}, "
          f"R2 drop: {best_balanced['r2_drop']:.1f}pp")

    print()


if __name__ == "__main__":
    main()
