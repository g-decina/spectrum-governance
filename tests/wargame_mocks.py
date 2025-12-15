import numpy as np
from typing import Any, Dict, List, Union
from sklearn.base import BaseEstimator
from dataclasses import dataclass

# --- Mock AdversarialMetrics (compatible with real AdversarialMetrics interface) ---
@dataclass
class MockAdversarialMetrics:
    """Mock version of AdversarialMetrics for testing."""
    attack_success_rate: float
    samples_tested: int = 100
    samples_successful: int = 35
    empirical_robustness_l2: float = 2.34
    empirical_robustness_linf: float = 0.45
    attack_type: str = "HopSkipJump"
    queries_used: int = 10000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'attack_success_rate': self.attack_success_rate,
            'samples_tested': self.samples_tested,
            'samples_successful': self.samples_successful,
            'empirical_robustness_l2': self.empirical_robustness_l2,
            'empirical_robustness_linf': self.empirical_robustness_linf,
            'attack_type': self.attack_type,
            'queries_used': self.queries_used,
            'avg_queries_per_sample': self.queries_used / self.samples_tested if self.samples_tested > 0 else 0.0,
        }


# --- Mock Logger (Confirms logging occurred) ---
class MockRCIALogger:
    def __init__(self, log_path="mock_log.jsonl"):
        self.logs = []
    def log_event(self, event: Any, context: str = "INFERENCE"):
        self.logs.append({"context": context, "data": event.output_payload})


# --- Mock Blue Team (Confirms calibration and returns test bounds) ---
class MockUncertaintyWrapper:
    def __init__(self, base_model: BaseEstimator, risk_profile: Any):
        self.base_model = base_model
        self.risk_profile = risk_profile
        self.is_calibrated = False
    def fit(self, X_calib, y_calib):
        self.is_calibrated = True
        return self
    def predict(self, X):
        # Always return mock bounds for the first sample
        n_samples = len(X)
        return {
            "prediction": np.zeros(n_samples),
            "lower_bound": np.array([0.1] * n_samples),
            "upper_bound": np.array([0.9] * n_samples),
            "confidence": 1.0 - self.risk_profile.alpha
        }


# --- Mock Red Team (Confirms execution and returns AdversarialMetrics) ---
class MockHopSkipJumpWrapper:
    def __init__(self, base_model: BaseEstimator):
        self.base_model = base_model

    def run(self, X_input: np.ndarray) -> MockAdversarialMetrics:
        # Fixed attack success rate for verification
        return MockAdversarialMetrics(
            attack_success_rate=0.35,
            samples_tested=len(X_input),
            samples_successful=int(0.35 * len(X_input)),
            attack_type="MockHopSkipJump"
        )


# --- Mock LLM Red Team ---
class MockInjectionScanner:
    def __init__(self, target_api_url: str):
        self.target_api_url = target_api_url

    def run(self, X_input: Union[np.ndarray, List[str]] = None) -> MockAdversarialMetrics:
        # Fixed attack success rate for verification
        n_samples = len(X_input) if X_input is not None else 10
        return MockAdversarialMetrics(
            attack_success_rate=0.50,
            samples_tested=n_samples,
            samples_successful=int(0.50 * n_samples),
            attack_type="MockInjectionScanner"
        )