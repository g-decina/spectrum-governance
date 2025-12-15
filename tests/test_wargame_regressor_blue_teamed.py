#!/usr/bin/env python3
"""
Wargame Runner Test Script - Blue-Teamed Regressor

This script demonstrates a hardened MLP regressor with multiple defense layers:

1. Input Validation: Range clipping, outlier detection, feature bounds
2. Model Hardening: Ensemble of diverse models, adversarial training
3. Output Wrapping: Prediction smoothing, confidence gating, conformal intervals
4. Runtime Monitoring: Query tracking, anomaly detection, audit logging

Compare results with test_wargame_regressor.py to see defense effectiveness.

Usage:
    python tests/test_wargame_regressor_blue_teamed.py

Output:
    - audit_report_regressor_hardened.docx (compliance report)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from collections import deque
import hashlib
import json
from datetime import datetime

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add spectrum to path if running from tests directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Terminal Colors and Formatting
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, '')


if not sys.stdout.isatty():
    Colors.disable()


def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"

def success(text: str) -> str:
    return f"{Colors.GREEN}{Colors.BOLD}{text}{Colors.RESET}"

def warning(text: str) -> str:
    return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"

def error(text: str) -> str:
    return f"{Colors.RED}{Colors.BOLD}{text}{Colors.RESET}"

def dim(text: str) -> str:
    return f"{Colors.DIM}{text}{Colors.RESET}"

def metric(label: str, value: str, unit: str = "") -> str:
    unit_str = f" {dim(unit)}" if unit else ""
    return f"  {Colors.WHITE}{label}:{Colors.RESET} {Colors.BOLD}{value}{Colors.RESET}{unit_str}"

def bullet(text: str, indent: int = 2) -> str:
    return f"{' ' * indent}{Colors.CYAN}>{Colors.RESET} {text}"

def print_header(title: str, color: str = Colors.CYAN):
    width = 60
    print()
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print()

def print_subheader(title: str):
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}--- {title} ---{Colors.RESET}")


# =============================================================================
# Defense Layer 1: Input Validator
# =============================================================================

@dataclass
class InputValidatorConfig:
    """Configuration for input validation."""
    clip_to_range: bool = True
    reject_outliers: bool = True
    outlier_threshold: float = 4.0  # Standard deviations
    feature_bounds: Optional[dict] = None  # {feature_idx: (min, max)}


class InputValidator:
    """
    Validates and sanitizes inputs before model inference.

    Defense mechanisms:
    - Range clipping: Bound inputs to training distribution
    - Outlier rejection: Flag inputs far from training data
    - Feature bounds: Enforce domain-specific constraints
    """

    def __init__(self, config: InputValidatorConfig = None):
        self.config = config or InputValidatorConfig()
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.feature_mins: Optional[np.ndarray] = None
        self.feature_maxs: Optional[np.ndarray] = None
        self.n_features: int = 0

    def fit(self, X: np.ndarray):
        """Learn input statistics from training data."""
        self.n_features = X.shape[1]
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        self.feature_mins = np.min(X, axis=0)
        self.feature_maxs = np.max(X, axis=0)

    def validate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and sanitize inputs.

        Returns:
            X_clean: Sanitized inputs
            flags: Boolean array, True if input was modified/flagged
        """
        X_clean = X.copy()
        flags = np.zeros(len(X), dtype=bool)

        # Check for outliers (before clipping)
        if self.config.reject_outliers:
            z_scores = np.abs((X - self.feature_means) / self.feature_stds)
            max_z = np.max(z_scores, axis=1)
            outlier_mask = max_z > self.config.outlier_threshold
            flags |= outlier_mask

        # Clip to training range
        if self.config.clip_to_range:
            X_before = X_clean.copy()
            X_clean = np.clip(X_clean, self.feature_mins, self.feature_maxs)
            modified = np.any(X_clean != X_before, axis=1)
            flags |= modified

        # Apply custom feature bounds
        if self.config.feature_bounds:
            for idx, (lo, hi) in self.config.feature_bounds.items():
                X_before = X_clean[:, idx].copy()
                X_clean[:, idx] = np.clip(X_clean[:, idx], lo, hi)
                modified = X_clean[:, idx] != X_before
                flags |= modified

        return X_clean, flags

    def get_stats(self) -> dict:
        return {
            "n_features": self.n_features,
            "feature_ranges": list(zip(self.feature_mins.tolist(), self.feature_maxs.tolist())),
        }


# =============================================================================
# Defense Layer 2: Hardened Ensemble Model
# =============================================================================

class HardenedEnsemble:
    """
    Ensemble of diverse models for improved robustness.

    Defense mechanisms:
    - Model diversity: MLP + GBM + Ridge have different vulnerabilities
    - Adversarial training: MLP trained on perturbed examples
    - Prediction averaging: Reduces impact of attacking any single model
    - Disagreement detection: Flag when models disagree significantly

    Key insight from ablation study: Adversarial training is the primary
    defense mechanism. Ensemble averaging helps for gradient-based attacks
    but output smoothing provides no benefit.
    """

    def __init__(
        self,
        adversarial_training: bool = True,
        adversarial_noise_std: float = 0.3,  # Increased from 0.1 - more aggressive perturbation
        n_adversarial_copies: int = 4,       # Increased from 2 - more training diversity
    ):
        self.adversarial_training = adversarial_training
        self.adversarial_noise_std = adversarial_noise_std
        self.n_adversarial_copies = n_adversarial_copies

        # Diverse model architectures
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128,),
            activation='tanh',
            solver='adam',
            alpha=0.5,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )

        self.gbm = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42
        )

        self.ridge = Ridge(alpha=1.0, random_state=42)

        # Weights for ensemble (can be tuned)
        self.weights = np.array([0.5, 0.35, 0.15])  # MLP, GBM, Ridge

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models, with optional adversarial training for MLP."""

        # Adversarial training: augment data with noisy copies
        if self.adversarial_training:
            X_aug = [X]
            y_aug = [y]
            for _ in range(self.n_adversarial_copies):
                noise = np.random.randn(*X.shape) * self.adversarial_noise_std
                X_noisy = X + noise
                X_aug.append(X_noisy)
                y_aug.append(y)
            X_mlp = np.vstack(X_aug)
            y_mlp = np.concatenate(y_aug)
        else:
            X_mlp = X
            y_mlp = y

        # Fit all models
        self.mlp.fit(X_mlp, y_mlp)
        self.gbm.fit(X, y)
        self.ridge.fit(X, y)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with disagreement detection.

        Returns:
            predictions: Weighted ensemble predictions
            disagreement: Per-sample disagreement score (std of model predictions)
        """
        pred_mlp = self.mlp.predict(X)
        pred_gbm = self.gbm.predict(X)
        pred_ridge = self.ridge.predict(X)

        # Stack predictions: (n_samples, n_models)
        all_preds = np.column_stack([pred_mlp, pred_gbm, pred_ridge])

        # Weighted average
        predictions = np.average(all_preds, axis=1, weights=self.weights)

        # Disagreement score (standard deviation across models)
        disagreement = np.std(all_preds, axis=1)

        return predictions, disagreement

    def get_individual_predictions(self, X: np.ndarray) -> dict:
        """Get predictions from each model separately."""
        return {
            "mlp": self.mlp.predict(X),
            "gbm": self.gbm.predict(X),
            "ridge": self.ridge.predict(X),
        }


# =============================================================================
# Defense Layer 3: Output Post-Processor
# =============================================================================

@dataclass
class OutputProcessorConfig:
    """Configuration for output processing.

    NOTE: Output smoothing was found to provide NO robustness benefit in ablation
    testing and may actually make attacks easier by creating predictable output
    bins. It is disabled by default.
    """
    enable_smoothing: bool = False     # DISABLED - provides no robustness benefit
    smoothing_resolution: float = 0.1  # Round to nearest 0.1 (if enabled)
    enable_confidence_gate: bool = True
    disagreement_threshold: float = 0.3  # Lowered from 0.5 - more sensitive flagging
    enable_bounds: bool = True
    output_min: float = 0.0
    output_max: float = 6.0  # Slightly above max house price in dataset


class OutputPostProcessor:
    """
    Post-processes model outputs for robustness.

    Defense mechanisms:
    - Smoothing: Discretize outputs to reduce precision attacks
    - Confidence gating: Flag low-confidence predictions
    - Output bounds: Clamp to valid range
    """

    def __init__(self, config: OutputProcessorConfig = None):
        self.config = config or OutputProcessorConfig()

    def process(
        self,
        predictions: np.ndarray,
        disagreement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process model outputs.

        Returns:
            processed: Processed predictions
            low_confidence: Boolean mask for low-confidence predictions
        """
        processed = predictions.copy()
        low_confidence = np.zeros(len(predictions), dtype=bool)

        # Bound outputs
        if self.config.enable_bounds:
            processed = np.clip(
                processed,
                self.config.output_min,
                self.config.output_max
            )

        # Smooth outputs (discretize)
        if self.config.enable_smoothing:
            resolution = self.config.smoothing_resolution
            processed = np.round(processed / resolution) * resolution

        # Flag high-disagreement predictions
        if self.config.enable_confidence_gate:
            low_confidence = disagreement > self.config.disagreement_threshold

        return processed, low_confidence


# =============================================================================
# Defense Layer 4: Runtime Monitor
# =============================================================================

@dataclass
class MonitorConfig:
    """Configuration for runtime monitoring."""
    window_size: int = 1000  # Track last N predictions
    anomaly_threshold: float = 3.0  # Z-score threshold
    enable_query_tracking: bool = True
    enable_distribution_monitoring: bool = True
    log_file: Optional[str] = "hardened_model_audit.jsonl"


class RuntimeMonitor:
    """
    Monitors model usage for anomalies.

    Defense mechanisms:
    - Query tracking: Detect repeated/similar queries (attack signature)
    - Distribution monitoring: Alert if input/output distributions shift
    - Audit logging: Record all predictions for forensics
    """

    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()

        # Rolling windows for statistics
        self.recent_inputs: deque = deque(maxlen=self.config.window_size)
        self.recent_outputs: deque = deque(maxlen=self.config.window_size)
        self.recent_hashes: deque = deque(maxlen=self.config.window_size)

        # Baseline statistics (set during calibration)
        self.baseline_input_mean: Optional[np.ndarray] = None
        self.baseline_input_std: Optional[np.ndarray] = None
        self.baseline_output_mean: Optional[float] = None
        self.baseline_output_std: Optional[float] = None

        # Counters
        self.total_queries = 0
        self.flagged_queries = 0
        self.duplicate_queries = 0

    def calibrate(self, X: np.ndarray, y_pred: np.ndarray):
        """Set baseline statistics from calibration data."""
        self.baseline_input_mean = np.mean(X, axis=0)
        self.baseline_input_std = np.std(X, axis=0) + 1e-8
        self.baseline_output_mean = np.mean(y_pred)
        self.baseline_output_std = np.std(y_pred) + 1e-8

    def _hash_input(self, x: np.ndarray) -> str:
        """Create hash of input for duplicate detection."""
        # Round to reduce near-duplicates
        x_rounded = np.round(x, decimals=3)
        return hashlib.md5(x_rounded.tobytes()).hexdigest()[:16]

    def record(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        flags: dict
    ) -> dict:
        """
        Record predictions and check for anomalies.

        Returns:
            alerts: Dict of alert types and counts
        """
        alerts = {
            "duplicate_inputs": 0,
            "distribution_shift_input": False,
            "distribution_shift_output": False,
        }

        for i in range(len(X)):
            self.total_queries += 1

            # Check for duplicate/similar inputs
            if self.config.enable_query_tracking:
                input_hash = self._hash_input(X[i])
                if input_hash in self.recent_hashes:
                    alerts["duplicate_inputs"] += 1
                    self.duplicate_queries += 1
                self.recent_hashes.append(input_hash)

            # Track for distribution monitoring
            self.recent_inputs.append(X[i])
            self.recent_outputs.append(predictions[i])

        # Check for distribution shift
        if (self.config.enable_distribution_monitoring and
            len(self.recent_outputs) >= 100 and
            self.baseline_output_mean is not None):

            recent_output_mean = np.mean(list(self.recent_outputs))
            z_score = abs(recent_output_mean - self.baseline_output_mean) / self.baseline_output_std
            if z_score > self.config.anomaly_threshold:
                alerts["distribution_shift_output"] = True

        # Log to file
        if self.config.log_file and flags.get("log", True):
            self._write_log(X, predictions, flags, alerts)

        if any(v for v in alerts.values() if v):
            self.flagged_queries += len(X)

        return alerts

    def _write_log(self, X, predictions, flags, alerts):
        """Append to audit log."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(X),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
            "flags": {k: int(v) if isinstance(v, (bool, np.bool_)) else v
                     for k, v in flags.items()},
            "alerts": alerts,
        }
        with open(self.config.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_stats(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "flagged_queries": self.flagged_queries,
            "duplicate_queries": self.duplicate_queries,
            "flag_rate": self.flagged_queries / max(1, self.total_queries),
        }


# =============================================================================
# Blue-Teamed Model Wrapper
# =============================================================================

class BlueTeamedRegressor:
    """
    Production-hardened regressor with multiple defense layers.

    Wraps a model with:
    1. Input validation and sanitization
    2. Hardened ensemble prediction
    3. Output post-processing
    4. Runtime monitoring
    """

    def __init__(
        self,
        input_config: InputValidatorConfig = None,
        output_config: OutputProcessorConfig = None,
        monitor_config: MonitorConfig = None,
        adversarial_training: bool = True,
    ):
        self.input_validator = InputValidator(input_config)
        self.ensemble = HardenedEnsemble(adversarial_training=adversarial_training)
        self.output_processor = OutputPostProcessor(output_config)
        self.monitor = RuntimeMonitor(monitor_config)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all components."""
        print(f"  {dim('Fitting input validator...')}")
        self.input_validator.fit(X)

        print(f"  {dim('Fitting hardened ensemble (with adversarial training)...')}")
        self.ensemble.fit(X, y)

        # Calibrate monitor with training predictions
        print(f"  {dim('Calibrating runtime monitor...')}")
        preds, _ = self.ensemble.predict(X)
        self.monitor.calibrate(X, preds)

        self.is_fitted = True

    def predict(self, X: np.ndarray, return_details: bool = False) -> dict:
        """
        Make hardened predictions.

        Returns:
            Dict with predictions and defense metadata
        """
        # Layer 1: Input validation
        X_clean, input_flags = self.input_validator.validate(X)

        # Layer 2: Ensemble prediction
        raw_predictions, disagreement = self.ensemble.predict(X_clean)

        # Layer 3: Output processing
        predictions, low_confidence = self.output_processor.process(
            raw_predictions, disagreement
        )

        # Layer 4: Runtime monitoring
        flags = {
            "inputs_modified": int(np.sum(input_flags)),
            "low_confidence": int(np.sum(low_confidence)),
            "log": True,
        }
        alerts = self.monitor.record(X_clean, predictions, flags)

        result = {
            "prediction": predictions,
            "low_confidence": low_confidence,
            "disagreement": disagreement,
            "inputs_modified": input_flags,
            "alerts": alerts,
        }

        if return_details:
            result["raw_predictions"] = raw_predictions
            result["individual_predictions"] = self.ensemble.get_individual_predictions(X_clean)

        return result

    def get_defense_stats(self) -> dict:
        """Get statistics from all defense layers."""
        return {
            "input_validator": self.input_validator.get_stats(),
            "monitor": self.monitor.get_stats(),
        }


# =============================================================================
# Spectrum Integration
# =============================================================================

from spectrum.infra.types import RiskProfile, RiskLevel
from spectrum.blue.trust import SpectrumRegressor
from spectrum.blue.monitor import DriftCheck
from spectrum.blue.explain import generate_shap_explanations
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder

try:
    from spectrum.red.attack import (
        OutputManipulationWrapper,
        PredictionShiftWrapper,
        QuantileAttackWrapper,
    )
    RED_TEAM_AVAILABLE = True
except ImportError:
    RED_TEAM_AVAILABLE = False
    print("NOTE: Red team regression attacks not available.")


# =============================================================================
# Sklearn-compatible wrapper for spectrum integration
# =============================================================================

class BlueTeamedRegressorSklearn:
    """Sklearn-compatible wrapper for BlueTeamedRegressor."""

    def __init__(self, hardened_model: BlueTeamedRegressor):
        self.hardened_model = hardened_model
        # Set attributes that sklearn's check_is_fitted looks for
        self.n_features_in_ = hardened_model.input_validator.n_features
        self.is_fitted_ = True

    def fit(self, X, y):
        # Already fitted
        return self

    def predict(self, X):
        result = self.hardened_model.predict(X)
        return result["prediction"]

    def score(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)


# =============================================================================
# Test Functions
# =============================================================================

def load_housing_data():
    print_header("Loading California Housing Dataset", Colors.BLUE)
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    print(metric("Dataset shape", f"{X.shape[0]:,} samples x {X.shape[1]} features"))
    print(metric("Target range", f"[{y.min():.2f}, {y.max():.2f}]"))
    return X, y


def prepare_data_splits(X, y, random_state=42):
    print_header("Preparing Data Splits", Colors.BLUE)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    print(metric("Training set", f"{len(X_train):,} samples", "(60%)"))
    print(metric("Calibration set", f"{len(X_calib):,} samples", "(20%)"))
    print(metric("Test set", f"{len(X_test):,} samples", "(20%)"))

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_calib_scaled = pd.DataFrame(
        scaler.transform(X_calib), columns=X_calib.columns, index=X_calib.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test


def train_hardened_model(X_train, y_train):
    print_header("Training Blue-Teamed Model", Colors.BLUE)

    print(f"  {Colors.CYAN}Defense layers:{Colors.RESET}")
    print(bullet("Input validation (range clipping, outlier detection)"))
    print(bullet("Hardened ensemble (MLP + GBM + Ridge)"))
    print(bullet("Adversarial training (σ=0.3, 4x augmentation)"))
    print(bullet("Confidence gating (disagreement threshold=0.3)"))
    print(bullet("Runtime monitoring (query tracking, audit log)"))
    print(bullet(f"{Colors.DIM}Output smoothing: DISABLED (no robustness benefit){Colors.RESET}"))
    print()

    model = BlueTeamedRegressor(
        input_config=InputValidatorConfig(
            clip_to_range=True,
            reject_outliers=True,
            outlier_threshold=4.0,
        ),
        output_config=OutputProcessorConfig(
            enable_smoothing=False,        # Disabled - no robustness benefit
            smoothing_resolution=0.1,
            enable_confidence_gate=True,
            disagreement_threshold=0.3,    # More sensitive
        ),
        monitor_config=MonitorConfig(
            window_size=1000,
            enable_query_tracking=True,
            enable_distribution_monitoring=True,
            log_file="hardened_model_audit.jsonl",
        ),
        adversarial_training=True,
    )

    model.fit(X_train.values, y_train.values)

    # Wrap for sklearn compatibility
    sklearn_wrapper = BlueTeamedRegressorSklearn(model)

    train_score = sklearn_wrapper.score(X_train.values, y_train.values)
    print()
    print(metric("Model", "BlueTeamedRegressor"))
    print(bullet(f"ensemble: {bold('MLP(128) + GBM(100) + Ridge')}"))
    print(bullet(f"adversarial training: {bold('σ=0.3, 4x copies')}"))
    print(bullet(f"output smoothing: {bold('disabled')}"))
    print()
    print(metric("Training R2", f"{train_score:.4f}"))

    return model, sklearn_wrapper


def run_blue_team(sklearn_wrapper, X_calib, y_calib, X_test, y_test, risk_profile):
    print_header("Blue Team: Uncertainty Quantification", Colors.BLUE)

    blue_model = SpectrumRegressor(
        base_model=sklearn_wrapper,
        risk_profile=risk_profile
    )
    blue_model.fit(X_calib.values, y_calib.values)
    print(f"  {dim(f'Conformal prediction calibrated on {len(X_calib):,} samples')}")

    result = blue_model.predict(X_test.values)
    predictions = result["prediction"]
    lower_bounds = result["lower_bound"]
    upper_bounds = result["upper_bound"]
    confidence = result["confidence"]

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    y_test_arr = y_test.values
    in_interval = (y_test_arr >= lower_bounds) & (y_test_arr <= upper_bounds)
    empirical_coverage = np.mean(in_interval)
    avg_interval_width = np.mean(upper_bounds - lower_bounds)

    print()
    print(metric("Confidence level", f"{confidence:.1%}"))
    print(metric("Empirical coverage", f"{empirical_coverage:.1%}"))
    print(metric("R2 score", f"{r2:.4f}"))
    print(metric("RMSE", f"{rmse:.4f}"))
    print(metric("Avg interval width", f"{avg_interval_width:.4f}"))

    coverage_ok = empirical_coverage >= confidence
    status = success("PASS") if coverage_ok else warning("BELOW TARGET")
    print(f"\n  {Colors.WHITE}Coverage achieved:{Colors.RESET} {status}")

    return {
        "predictions": predictions,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "empirical_coverage": empirical_coverage,
        "confidence": confidence,
        "rmse": rmse,
        "r2": r2,
        "interval_width": avg_interval_width,
    }


def run_drift_check(X_train, X_test):
    print_header("Blue Team: Drift Monitoring", Colors.BLUE)
    drift_result = DriftCheck(reference_data=X_train, current_data=X_test)
    drift_detected = drift_result['drift_detected']
    drift_status = error("YES") if drift_detected else success("NO")
    print(metric("Drift detected", drift_status))
    print(metric("Max PSI", f"{drift_result['max_psi']:.4f}"))
    return drift_result


def run_explainability(sklearn_wrapper, X_test):
    print_header("Blue Team: Explainability (SHAP)", Colors.BLUE)
    print(f"  {dim('Computing SHAP values...')}")
    explanations = generate_shap_explanations(
        model=sklearn_wrapper,
        X_test=X_test,
        max_samples=50
    )
    print(f"\n  {Colors.WHITE}Top contributing features:{Colors.RESET}")
    for i, feature in enumerate(explanations.get('top_features', [])[:5], 1):
        if i == 1:
            feat_display = f"{Colors.GREEN}{Colors.BOLD}{feature}{Colors.RESET}"
        elif i <= 3:
            feat_display = f"{Colors.CYAN}{feature}{Colors.RESET}"
        else:
            feat_display = feature
        print(f"    {Colors.DIM}{i}.{Colors.RESET} {feat_display}")
    return explanations


def run_red_team(sklearn_wrapper, X_test, y_test, blue_results, n_samples=100):
    print_header("Red Team: Adversarial Testing", Colors.RED)

    if not RED_TEAM_AVAILABLE:
        print(f"  {warning('Red team components not available.')}")
        return None

    X_attack = X_test.iloc[:n_samples].values
    y_attack = y_test.iloc[:n_samples].values

    print(f"  {dim(f'Testing {n_samples} samples against hardened model...')}")

    results = {}

    # Output Manipulation Attack
    try:
        print_subheader("Output Manipulation Attack")
        attack = OutputManipulationWrapper(
            base_model=sklearn_wrapper,
            n_bins=5,
            max_iter=30,
            max_eval=2000,
            parallel=True,
            bin_mode="quantile",
            confidence_level=0.99,  # 99% CI on mean attempts
        )
        metrics = attack.run(X_attack)

        # Report attempts-based metrics
        print(metric("Samples succeeded", f"{metrics.samples_succeeded}/{metrics.samples_tested}"))
        if not np.isnan(metrics.mean_attempts_to_success):
            # Fewer attempts = more vulnerable (red), more = robust (green)
            attempts_color = Colors.RED if metrics.mean_attempts_to_success < 100 else Colors.GREEN
            print(metric("Mean attempts to success", f"{attempts_color}{metrics.mean_attempts_to_success:.1f}{Colors.RESET}", "queries"))
            print(metric("99% CI", f"[{metrics.attempts_ci[0]:.1f}, {metrics.attempts_ci[1]:.1f}]"))
        print(metric("Mean L2 perturbation", f"{metrics.mean_perturbation_l2:.4f}"))
        results["output_manipulation"] = metrics
    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")

    # Prediction Shift Attack
    try:
        print_subheader("Prediction Shift Attack")
        attack = PredictionShiftWrapper(
            base_model=sklearn_wrapper,
            epsilon=1.0,
            max_iter=50,
            n_directions=30,
            parallel=True,
            target_direction="any",
            success_threshold=0.10,  # 10% shift = success
            confidence_level=0.99,   # 99% CI on mean attempts
        )
        metrics = attack.run(X_attack)

        # Report attempts-based metrics
        print(metric("Samples succeeded", f"{metrics.samples_succeeded}/{metrics.samples_tested}"))
        if not np.isnan(metrics.mean_attempts_to_success):
            attempts_color = Colors.RED if metrics.mean_attempts_to_success < 50 else Colors.GREEN
            print(metric("Mean attempts to success", f"{attempts_color}{metrics.mean_attempts_to_success:.1f}{Colors.RESET}", "directions"))
            print(metric("99% CI", f"[{metrics.attempts_ci[0]:.1f}, {metrics.attempts_ci[1]:.1f}]"))
        print(metric("Mean L2 perturbation", f"{metrics.mean_perturbation_l2:.4f}"))
        print(metric("Mean prediction shift", f"{metrics.mean_absolute_shift:.4f}"))
        print(metric("Max prediction shift", f"{metrics.max_prediction_shift:.4f}"))
        results["prediction_shift"] = metrics
    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")

    # Quantile Attack
    try:
        print_subheader("Quantile Attack (Break Coverage)")
        calibration_width = blue_results.get("interval_width", 1.0) / 2
        attack = QuantileAttackWrapper(
            base_model=sklearn_wrapper,
            epsilon=0.5,
            max_iter=50,
            n_directions=30,
            parallel=True,
            attack_mode="break_coverage",
            calibration_width=calibration_width,
        )
        metrics = attack.run(X_attack, y_attack)
        break_rate_color = Colors.RED if metrics.coverage_break_rate > 0.3 else Colors.GREEN
        print(metric("Coverage break rate", f"{break_rate_color}{metrics.coverage_break_rate:.1%}{Colors.RESET}"))
        print(metric("Original coverage", f"{metrics.original_coverage:.1%}"))
        print(metric("Adversarial coverage", f"{metrics.adversarial_coverage:.1%}"))
        results["quantile_attack"] = metrics
    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")

    if results:
        return results.get("output_manipulation") or results.get("prediction_shift")
    return None


def print_defense_comparison(hardened_model):
    """Print defense layer statistics."""
    print_header("Defense Layer Statistics", Colors.CYAN)

    stats = hardened_model.get_defense_stats()

    print(f"  {Colors.WHITE}Input Validator:{Colors.RESET}")
    print(bullet(f"Features monitored: {stats['input_validator']['n_features']}"))

    print(f"\n  {Colors.WHITE}Runtime Monitor:{Colors.RESET}")
    monitor_stats = stats['monitor']
    print(bullet(f"Total queries: {monitor_stats['total_queries']:,}"))
    print(bullet(f"Flagged queries: {monitor_stats['flagged_queries']:,}"))
    print(bullet(f"Duplicate queries: {monitor_stats['duplicate_queries']:,}"))
    print(bullet(f"Flag rate: {monitor_stats['flag_rate']:.1%}"))


def generate_compliance_report(
    model_name: str,
    risk_profile: RiskProfile,
    blue_results: dict,
    drift_result: dict,
    explanations: dict,
    red_metrics,
    output_path: str
):
    print_header("Generating Compliance Report", Colors.MAGENTA)
    print(f"  {dim('Building DOCX report...')}")

    if red_metrics is not None:
        adversarial_metrics = red_metrics.to_dict()
    else:
        adversarial_metrics = {
            "attack_type": "Not Performed",
            "attack_success_rate": None,
            "samples_tested": 0,
            "samples_successful": 0,
            "empirical_robustness_l2": 0.0,
            "empirical_robustness_linf": 0.0,
            "min_perturbation_l2": 0.0,
            "max_perturbation_l2": 0.0,
            "median_perturbation_l2": 0.0,
            "queries_used": 0,
            "avg_queries_per_sample": 0.0
        }

    report_data = {
        "model_name": model_name,
        "model_type": "Tabular (Regression) - Blue-Teamed",
        "risk_level": risk_profile.level.value,
        "confidence_required": 1.0 - risk_profile.alpha,
        "empirical_coverage": blue_results["empirical_coverage"],
        "adversarial_metrics": adversarial_metrics,
        "sample_adverse_reasons": explanations.get("top_features", [])[:5],
        "data_drift_status": drift_result["status"],
        "data_drift_alert": drift_result["alert_required"],
        "lineage_run_id": "regressor-blue-teamed-001",
        "audit_log_path": "hardened_model_audit.jsonl"
    }

    report = ComplianceReport(**report_data)
    builder = ReportBuilder()
    builder.generate_docx(
        data=report.model_dump(),
        output_path=output_path,
        template_type="tier1_forensic_audit"
    )
    print(f"\n  {success('Report saved:')} {Colors.UNDERLINE}{output_path}{Colors.RESET}")
    return report


def main():
    # Banner
    print()
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}   SPECTRUM GOVERNANCE - BLUE-TEAMED REGRESSOR{Colors.RESET}")
    print(f"{Colors.WHITE}   Hardened Model with Defense Layers{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    risk_profile = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.10)
    print()
    print(f"  {Colors.WHITE}Risk Profile:{Colors.RESET}       {Colors.YELLOW}{Colors.BOLD}{risk_profile.level.value}{Colors.RESET}")
    print(f"  {Colors.WHITE}Required Confidence:{Colors.RESET} {bold(f'{1 - risk_profile.alpha:.0%}')}")

    # Load and prepare data
    X, y = load_housing_data()
    X_train, X_calib, X_test, y_train, y_calib, y_test = prepare_data_splits(X, y)

    # Train hardened model
    hardened_model, sklearn_wrapper = train_hardened_model(X_train, y_train)

    test_score = sklearn_wrapper.score(X_test.values, y_test.values)
    print(metric("Test R2", f"{test_score:.4f}"))

    # Blue team analysis
    blue_results = run_blue_team(
        sklearn_wrapper, X_calib, y_calib, X_test, y_test, risk_profile
    )
    drift_result = run_drift_check(X_train, X_test)
    explanations = run_explainability(sklearn_wrapper, X_test)

    # Red team attacks (100 samples for statistically meaningful comparison)
    red_metrics = run_red_team(
        sklearn_wrapper, X_test, y_test, blue_results, n_samples=100
    )

    # Defense statistics
    print_defense_comparison(hardened_model)

    # Generate report
    report = generate_compliance_report(
        model_name="BlueTeamedRegressor(Ensemble+Hardened)",
        risk_profile=risk_profile,
        blue_results=blue_results,
        drift_result=drift_result,
        explanations=explanations,
        red_metrics=red_metrics,
        output_path="audit_report_regressor_hardened.docx"
    )

    # Summary
    print_header("Audit Summary", Colors.GREEN)
    print(metric("Model", report.model_name))
    print(metric("Task", "Regression (Blue-Teamed)"))
    print(metric("Risk Level", f"{Colors.YELLOW}{report.risk_level}{Colors.RESET}"))
    print(metric("Required Confidence", f"{report.confidence_required:.0%}"))
    print(metric("Empirical Coverage", f"{report.empirical_coverage:.1%}"))
    print(metric("R2 Score", f"{blue_results['r2']:.4f}"))
    print(metric("RMSE", f"{blue_results['rmse']:.4f}"))

    drift_alert = error("YES") if report.data_drift_alert else success("NO")
    print(metric("Drift Alert", drift_alert))

    if red_metrics:
        print(f"\n  {Colors.RED}{Colors.BOLD}Adversarial Testing:{Colors.RESET}")
        print(metric("  Attack", red_metrics.attack_type))
        if not np.isnan(red_metrics.mean_attempts_to_success):
            attempts = red_metrics.mean_attempts_to_success
            # Fewer attempts = more vulnerable
            attempts_color = Colors.RED if attempts < 100 else Colors.GREEN
            print(metric("  Mean Attempts", f"{attempts_color}{attempts:.1f}{Colors.RESET}"))
            ci_pct = f"{red_metrics.confidence_level*100:.0f}%"
            print(metric(f"  {ci_pct} CI", f"[{red_metrics.attempts_ci[0]:.1f}, {red_metrics.attempts_ci[1]:.1f}]"))
        print(metric("  Mean L2", f"{red_metrics.mean_perturbation_l2:.4f}"))

    coverage_pass = report.empirical_coverage >= report.confidence_required
    drift_pass = not report.data_drift_alert

    print()
    print(f"  {'─' * 40}")

    if coverage_pass and drift_pass:
        status_text = f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD}  COMPLIANT  {Colors.RESET}"
    else:
        status_text = f"{Colors.BG_MAGENTA}{Colors.WHITE}{Colors.BOLD}  REQUIRES ATTENTION  {Colors.RESET}"

    print(f"\n  Overall Status: {status_text}")
    print(f"\n  {Colors.WHITE}Generated artifacts:{Colors.RESET}")
    print(f"    {Colors.CYAN}>{Colors.RESET} {Colors.UNDERLINE}audit_report_regressor_hardened.docx{Colors.RESET}")
    print(f"    {Colors.CYAN}>{Colors.RESET} {Colors.UNDERLINE}hardened_model_audit.jsonl{Colors.RESET}")
    print()

    return report


if __name__ == "__main__":
    main()
