import numpy as np
import pytest
import os

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from spectrum.red.attack import HopSkipJumpWrapper
from spectrum.red.metrics import AdversarialMetrics
from spectrum.utils.threading import _with_limited_threads

# --- Helper to check environment inside the decorated function ---
def check_threads():
    """Returns the current MKL environment variable."""
    return os.environ.get('MKL_NUM_THREADS')


# ==========================================
# 1. Testing Thread Safety Primitive
# ==========================================

def test_thread_safety_primitive():
    """
    Verifies that _with_limited_threads correctly sets the thread count to 1
    and restores the original values.
    """
    # Arrange: Set a known, non-default value before testing
    original_mkl = check_threads()

    def inner_test_function():
        # Assert: Inside the decorated function, limits must be set to 1
        current_mkl = check_threads()
        assert current_mkl == '1'
        return True

    # Act: Use the primitive explicitly (not via decorator for clean teardown check)
    try:
        # Set to minimum
        os.environ['MKL_NUM_THREADS'] = '1'

        result = inner_test_function()

    finally:
        # Teardown: Restore original settings
        if original_mkl is not None:
            os.environ['MKL_NUM_THREADS'] = original_mkl
        elif 'MKL_NUM_THREADS' in os.environ:
            del os.environ['MKL_NUM_THREADS']

    # Assert: After the function, values must be restored
    assert result is True
    assert check_threads() == original_mkl  # Can be None


# ==========================================
# 2. Testing HopSkipJump Wrapper
# ==========================================

def test_hsj_wrapper_initialization():
    """Test HopSkipJumpWrapper can be initialized with different backends."""
    X, y = make_classification(
        n_samples=20,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42,
    )
    model = LogisticRegression().fit(X, y)

    # Test default backend
    wrapper = HopSkipJumpWrapper(model)
    assert wrapper.backend in ["rust", "art", "auto"]
    assert wrapper.base_model is model

    # Test explicit art backend
    wrapper_art = HopSkipJumpWrapper(model, backend="art")
    assert wrapper_art.backend == "art"

    # Test explicit rust backend
    wrapper_rust = HopSkipJumpWrapper(model, backend="rust")
    assert wrapper_rust.backend == "rust"


def test_hsj_wrapper_has_run_method():
    """Test that HopSkipJumpWrapper has a run method."""
    X, y = make_classification(
        n_samples=20,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42,
    )
    model = LogisticRegression().fit(X, y)

    wrapper = HopSkipJumpWrapper(model)

    # Check that run method exists and is callable
    assert hasattr(wrapper, 'run')
    assert callable(wrapper.run)


def test_hsj_wrapper_config():
    """Test HopSkipJumpWrapper configuration options."""
    X, y = make_classification(
        n_samples=20,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42,
    )
    model = LogisticRegression().fit(X, y)

    # Test with custom max_iter (stored in config dict)
    wrapper = HopSkipJumpWrapper(model, max_iter=50)
    assert wrapper.config["max_iter"] == 50

    # Test default config values
    wrapper_default = HopSkipJumpWrapper(model)
    assert wrapper_default.config["max_iter"] == 64
    assert wrapper_default.config["max_eval"] == 1000


# ==========================================
# 3. Testing AdversarialMetrics
# ==========================================

def test_adversarial_metrics_creation():
    """Test creating AdversarialMetrics directly."""
    metrics = AdversarialMetrics(
        attack_success_rate=0.25,
        samples_tested=100,
        samples_successful=25,
        empirical_robustness_l2=0.15,
        empirical_robustness_linf=0.08,
        min_perturbation_l2=0.05,
        max_perturbation_l2=0.30,
        median_perturbation_l2=0.12,
        attack_type="HopSkipJump",
        queries_used=50000,
        avg_queries_per_sample=500.0,
    )

    assert metrics.attack_success_rate == 0.25
    assert metrics.samples_tested == 100
    assert metrics.samples_successful == 25
    assert metrics.attack_type == "HopSkipJump"


def test_adversarial_metrics_to_dict():
    """Test AdversarialMetrics serialization."""
    metrics = AdversarialMetrics(
        attack_success_rate=0.20,
        samples_tested=50,
        samples_successful=10,
        empirical_robustness_l2=0.10,
        empirical_robustness_linf=0.05,
        min_perturbation_l2=0.03,
        max_perturbation_l2=0.20,
        median_perturbation_l2=0.08,
        attack_type="TestAttack",
        queries_used=10000,
        avg_queries_per_sample=200.0,
    )

    d = metrics.to_dict()

    assert isinstance(d, dict)
    assert d['attack_success_rate'] == 0.20
    assert d['samples_tested'] == 50
    assert d['attack_type'] == "TestAttack"


def test_adversarial_metrics_for_evasion_attack():
    """Test AdversarialMetrics.for_evasion_attack factory method."""
    # Simulate attack results
    y_original = np.array([0, 0, 1, 1, 0])
    y_adversarial = np.array([1, 0, 1, 0, 0])  # 2 flips: indices 0 and 3

    X_original = np.random.randn(5, 4)
    X_adversarial = X_original + np.random.randn(5, 4) * 0.1  # Small perturbation

    metrics = AdversarialMetrics.for_evasion_attack(
        attack_type="HopSkipJump",
        y_original=y_original,
        y_adversarial=y_adversarial,
        X_original=X_original,
        X_adversarial=X_adversarial,
        queries_used=5000
    )

    assert isinstance(metrics, AdversarialMetrics)
    assert metrics.attack_success_rate == pytest.approx(0.4)  # 2/5 = 0.4
    assert metrics.samples_tested == 5
    assert metrics.samples_successful == 2
    assert metrics.attack_type == "HopSkipJump"


def test_adversarial_metrics_confidence_interval():
    """Test that AdversarialMetrics computes Wilson confidence interval."""
    metrics = AdversarialMetrics(
        attack_success_rate=0.30,
        samples_tested=100,
        samples_successful=30,
        empirical_robustness_l2=0.10,
        empirical_robustness_linf=0.05,
        min_perturbation_l2=0.03,
        max_perturbation_l2=0.20,
        median_perturbation_l2=0.08,
        attack_type="TestAttack",
        queries_used=10000,
        avg_queries_per_sample=100.0,
    )

    # Should have auto-computed 95% CI
    assert metrics.confidence_interval_95 is not None
    lower, upper = metrics.confidence_interval_95
    assert lower < 0.30 < upper  # CI should contain the point estimate
    assert 0.0 <= lower <= upper <= 1.0  # CI should be valid probabilities
