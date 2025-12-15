"""
Tests for WargameRunner Helper Methods

Tests the refactored helper methods for drift checking, explanations, and coverage calculation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from spectrum.lens.wargame_runner import WargameRunner
from spectrum.infra.types import RiskProfile, TargetModel, RiskLevel


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = Mock()
    model.__class__.__name__ = "MockClassifier"
    model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
    return model


@pytest.fixture
def risk_profile():
    """Create a test risk profile."""
    return RiskProfile(level=RiskLevel.HIGH, alpha=0.05)


@pytest.fixture
def target_model(mock_model):
    """Create a test target model."""
    return TargetModel(
        model_name="test_model",
        is_tabular=True,
        model_object=mock_model
    )


@pytest.fixture
def mock_logger():
    """Create a mock RCIA logger."""
    logger = Mock()
    logger.log_path = "/tmp/test_audit.jsonl"
    logger.log_event = Mock()
    return logger


@pytest.fixture
def wargame_runner(target_model, risk_profile, mock_logger):
    """Create a WargameRunner instance for testing."""
    attack_map = {"tabular": Mock, "llm": Mock}
    return WargameRunner(
        target_model=target_model,
        risk_profile=risk_profile,
        logger=mock_logger,
        attack_wrapper_map=attack_map
    )


def test_check_drift_with_valid_data(wargame_runner):
    """Test drift checking with valid reference and current data."""
    print("\n=== Test: Drift Check with Valid Data ===")

    np.random.seed(42)
    X_ref = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })

    X_current = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1, 100),  # Slight shift
        'feature2': np.random.normal(0, 1, 100)
    })

    with patch('spectrum.lens.wargame_runner.DriftCheck') as mock_drift:
        mock_drift.return_value = {
            "status": "✓ No Drift Detected",
            "alert_required": False,
            "max_psi": 0.05
        }

        result = wargame_runner._check_drift(X_ref, X_current)

        print(f"Drift status: {result['status']}")
        print(f"Alert required: {result['alert_required']}")
        print(f"Max PSI: {result['max_psi']}")

        assert result["status"] == "✓ No Drift Detected"
        assert result["alert_required"] is False
        assert "max_psi" in result
        mock_drift.assert_called_once_with(X_ref, X_current)


def test_check_drift_with_none_data(wargame_runner):
    """Test drift checking when reference data is None."""
    print("\n=== Test: Drift Check with None Data ===")

    result = wargame_runner._check_drift(None, None)

    print(f"Drift status: {result['status']}")
    print(f"Alert required: {result['alert_required']}")

    assert result["status"] == "Not Performed"
    assert result["alert_required"] is False
    assert result["max_psi"] == 0.0


def test_check_drift_with_exception(wargame_runner):
    """Test drift checking handles exceptions gracefully."""
    print("\n=== Test: Drift Check with Exception ===")

    X_ref = pd.DataFrame({'a': [1, 2, 3]})
    X_current = pd.DataFrame({'a': [4, 5, 6]})

    with patch('spectrum.lens.wargame_runner.DriftCheck') as mock_drift:
        mock_drift.side_effect = ValueError("Test error")

        result = wargame_runner._check_drift(X_ref, X_current)

        print(f"Drift status: {result['status']}")
        print(f"Error handled: {result['status'] == 'Error'}")

        assert result["status"] == "Error"
        assert result["alert_required"] is False
        assert result["max_psi"] == 0.0


def test_generate_explanations_success(wargame_runner, mock_model):
    """Test successful SHAP explanation generation."""
    print("\n=== Test: Generate Explanations Success ===")

    X_test = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    })

    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([0, 1, 0])

    with patch('spectrum.lens.wargame_runner.generate_shap_explanations') as mock_shap:
        mock_shap.return_value = {
            "top_features": [
                "feature1: high value",
                "feature2: low value"
            ]
        }

        reasons = wargame_runner._generate_explanations(
            base_model=mock_model,
            X_test=X_test,
            defense_results=mock_defense_results
        )

        print(f"Generated reasons: {reasons}")
        print(f"Number of reasons: {len(reasons)}")

        assert len(reasons) == 2
        assert "feature1" in reasons[0]
        assert "feature2" in reasons[1]
        mock_shap.assert_called_once()


def test_generate_explanations_no_predictions(wargame_runner, mock_model):
    """Test explanation generation when no predictions available."""
    print("\n=== Test: Generate Explanations - No Predictions ===")

    X_test = pd.DataFrame({'a': [1, 2, 3]})
    mock_defense_results = Mock(spec=[])  # No y_preds attribute

    reasons = wargame_runner._generate_explanations(
        base_model=mock_model,
        X_test=X_test,
        defense_results=mock_defense_results
    )

    print(f"Fallback reason: {reasons}")

    assert len(reasons) == 1
    assert reasons[0] == "Explanations not available"


def test_generate_explanations_exception(wargame_runner, mock_model):
    """Test explanation generation handles exceptions."""
    print("\n=== Test: Generate Explanations - Exception Handling ===")

    X_test = pd.DataFrame({'a': [1, 2, 3]})
    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([0, 1])

    with patch('spectrum.lens.wargame_runner.generate_shap_explanations') as mock_shap:
        mock_shap.side_effect = RuntimeError("SHAP failed")

        reasons = wargame_runner._generate_explanations(
            base_model=mock_model,
            X_test=X_test,
            defense_results=mock_defense_results
        )

        print(f"Fallback reason: {reasons}")

        assert len(reasons) == 1
        assert "unavailable" in reasons[0].lower()


def test_calculate_empirical_coverage_success(wargame_runner):
    """Test successful empirical coverage calculation."""
    print("\n=== Test: Calculate Empirical Coverage - Success ===")

    # Create mock prediction intervals
    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Shape: (n_samples, 2, 1) for lower and upper bounds
    mock_defense_results.y_pis = np.array([
        [[0.5], [1.5]],  # y_test[0] = 1.0, in interval [0.5, 1.5] ✓
        [[1.5], [2.5]],  # y_test[1] = 2.0, in interval [1.5, 2.5] ✓
        [[2.5], [3.5]],  # y_test[2] = 3.0, in interval [2.5, 3.5] ✓
        [[3.5], [4.5]],  # y_test[3] = 4.0, in interval [3.5, 4.5] ✓
        [[4.5], [5.5]]   # y_test[4] = 10.0, NOT in interval [4.5, 5.5] ✗
    ])

    y_test = np.array([1.0, 2.0, 3.0, 4.0, 10.0])

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=y_test
    )

    print(f"Empirical coverage: {coverage}")
    print(f"Expected coverage: 0.8 (4/5)")

    assert coverage is not None
    assert coverage == 0.8  # 4 out of 5 within intervals


def test_calculate_empirical_coverage_perfect(wargame_runner):
    """Test coverage calculation with perfect coverage."""
    print("\n=== Test: Calculate Empirical Coverage - Perfect ===")

    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([1.0, 2.0, 3.0])
    mock_defense_results.y_pis = np.array([
        [[0.5], [1.5]],
        [[1.5], [2.5]],
        [[2.5], [3.5]]
    ])

    y_test = np.array([1.0, 2.0, 3.0])  # All within intervals

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=y_test
    )

    print(f"Perfect coverage: {coverage}")

    assert coverage == 1.0


def test_calculate_empirical_coverage_zero(wargame_runner):
    """Test coverage calculation with zero coverage."""
    print("\n=== Test: Calculate Empirical Coverage - Zero ===")

    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([1.0, 2.0, 3.0])
    mock_defense_results.y_pis = np.array([
        [[0.0], [0.5]],  # y_test outside
        [[0.0], [0.5]],  # y_test outside
        [[0.0], [0.5]]   # y_test outside
    ])

    y_test = np.array([10.0, 20.0, 30.0])  # All outside intervals

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=y_test
    )

    print(f"Zero coverage: {coverage}")

    assert coverage == 0.0


def test_calculate_empirical_coverage_no_y_test(wargame_runner):
    """Test coverage calculation when y_test is None."""
    print("\n=== Test: Calculate Empirical Coverage - No y_test ===")

    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([1.0, 2.0])
    mock_defense_results.y_pis = np.array([[[0.5], [1.5]], [[1.5], [2.5]]])

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=None
    )

    print(f"Coverage (no y_test): {coverage}")

    assert coverage is None


def test_calculate_empirical_coverage_missing_attributes(wargame_runner):
    """Test coverage calculation when predictions missing."""
    print("\n=== Test: Calculate Empirical Coverage - Missing Attributes ===")

    mock_defense_results = Mock(spec=[])  # No y_preds or y_pis

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=np.array([1, 2, 3])
    )

    print(f"Coverage (missing attrs): {coverage}")

    assert coverage is None


def test_calculate_empirical_coverage_exception(wargame_runner):
    """Test coverage calculation handles exceptions."""
    print("\n=== Test: Calculate Empirical Coverage - Exception ===")

    mock_defense_results = Mock()
    mock_defense_results.y_preds = np.array([1.0])
    # Intentionally wrong shape to cause error
    mock_defense_results.y_pis = np.array([1.0])

    coverage = wargame_runner._calculate_empirical_coverage(
        defense_results=mock_defense_results,
        y_test=np.array([1.0])
    )

    print(f"Coverage (exception): {coverage}")

    assert coverage is None


def test_run_tabular_wargame_validation_error(wargame_runner):
    """Test tabular wargame with missing model_object."""
    print("\n=== Test: Tabular Wargame - Validation Error ===")

    # Set model_object to None
    wargame_runner.target.model_object = None

    X_test = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(ValueError, match="Tabular Wargame requires a 'model_object'"):
        wargame_runner._run_tabular_wargame(
            X_test=X_test,
            y_test=None,
            X_calib=None,
            y_calib=None,
            X_ref_drift=None,
            X_current_drift=None
        )

    print("ValueError raised as expected")


def test_run_tabular_wargame_missing_calibration(wargame_runner, mock_model):
    """Test tabular wargame with missing calibration data."""
    print("\n=== Test: Tabular Wargame - Missing Calibration ===")

    wargame_runner.target.model_object = mock_model
    X_test = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(ValueError, match="calibration data"):
        wargame_runner._run_tabular_wargame(
            X_test=X_test,
            y_test=None,
            X_calib=None,  # Missing
            y_calib=None,  # Missing
            X_ref_drift=None,
            X_current_drift=None
        )

    print("ValueError raised for missing calibration")


def test_run_llm_wargame_missing_wrapper(wargame_runner):
    """Test LLM wargame with missing attack wrapper."""
    print("\n=== Test: LLM Wargame - Missing Wrapper ===")

    # Remove LLM wrapper from attack map
    wargame_runner.attack_map = {"tabular": Mock}
    wargame_runner.target.is_tabular = False
    wargame_runner.target.target_api_url = "https://api.example.com"

    with pytest.raises(NotImplementedError, match="LLM attack wrapper not configured"):
        wargame_runner._run_llm_wargame(X_test=["test prompt"])

    print("NotImplementedError raised as expected")


def test_wargame_runner_initialization(target_model, risk_profile, mock_logger):
    """Test WargameRunner initialization."""
    print("\n=== Test: WargameRunner Initialization ===")

    attack_map = {"tabular": Mock, "llm": Mock}
    runner = WargameRunner(
        target_model=target_model,
        risk_profile=risk_profile,
        logger=mock_logger,
        attack_wrapper_map=attack_map
    )

    print(f"Target model: {runner.target.model_name}")
    print(f"Risk level: {runner.risk_profile.level}")
    print(f"Attack map keys: {list(runner.attack_map.keys())}")

    assert runner.target == target_model
    assert runner.risk_profile == risk_profile
    assert runner.logger == mock_logger
    assert "tabular" in runner.attack_map
    assert "llm" in runner.attack_map


def test_check_drift_only_ref_data(wargame_runner):
    """Test drift check with only reference data (no current)."""
    print("\n=== Test: Drift Check - Only Reference Data ===")

    X_ref = pd.DataFrame({'a': [1, 2, 3]})

    result = wargame_runner._check_drift(X_ref, None)

    print(f"Status: {result['status']}")

    assert result["status"] == "Not Performed"
    assert result["alert_required"] is False


def test_check_drift_only_current_data(wargame_runner):
    """Test drift check with only current data (no reference)."""
    print("\n=== Test: Drift Check - Only Current Data ===")

    X_current = pd.DataFrame({'a': [1, 2, 3]})

    result = wargame_runner._check_drift(None, X_current)

    print(f"Status: {result['status']}")

    assert result["status"] == "Not Performed"
    assert result["alert_required"] is False
