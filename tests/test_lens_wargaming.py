import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Import the WargameRunner class definition (as submitted by the user)
# NOTE: The user must ensure all their imports (like from spectrum.infra._types) are correct.
from spectrum.lens.wargame_runner import WargameRunner
from spectrum.infra.types import RiskProfile, RiskLevel, TargetModel

from tests.wargame_mocks import MockRCIALogger, MockUncertaintyWrapper, MockHopSkipJumpWrapper, MockInjectionScanner

# Prepare mock data for tabular test
X_tab, y_tab = make_regression(n_samples=5, n_features=3, random_state=42)
MODEL = LinearRegression().fit(X_tab, y_tab)


def test_wargame_tabular_flow():
    """
    Test 1: Verify Tabular Wargame initializes Red and Blue teams correctly,
    executes them, and returns the combined results in the audit summary.
    """

    # 1. Arrange
    # Define the contract and target
    risk = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
    target = TargetModel(model_object=MODEL)

    logger = MockRCIALogger()

    # Map the mock wrappers to the runner
    attack_map = {"tabular": MockHopSkipJumpWrapper}

    runner = WargameRunner(
        target_model=target,
        risk_profile=risk,
        logger=logger,
        attack_wrapper_map=attack_map,
        defense_wrapper=MockUncertaintyWrapper # Pass the Mock Blue Team
    )

    # 2. Act
    # Use split data for X_calib/X_test as required by the runner logic
    X_calib = X_tab[:3]
    y_calib = y_tab[:3]
    X_test = X_tab[3:]
    y_test = y_tab[3:]

    audit_summary = runner.run_wargame(
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib
    )

    # 3. Assert on Output Data
    assert audit_summary["model_type"] == "Tabular"

    # Check adversarial_metrics structure
    assert "adversarial_metrics" in audit_summary
    adversarial_metrics = audit_summary["adversarial_metrics"]
    assert adversarial_metrics["attack_success_rate"] == pytest.approx(0.35)  # From MockHopSkipJumpWrapper

    assert audit_summary["confidence_required"] == pytest.approx(0.95)

    # Assert on Defense Artifacts (Bounds from MockUncertaintyWrapper)
    # The first sample bound should be the mock 0.1
    assert audit_summary["prediction_bounds_first_sample"] == pytest.approx(0.1)

    # 4. Assert on Logging (CRITICAL: Check the RCIA log)
    assert len(logger.logs) == 1
    logged_data = logger.logs[0]
    assert logged_data["context"] == "WARGAME_END"
    # Ensure the logged output contains the adversarial metrics
    assert logged_data["data"]["adversarial_metrics"]["attack_success_rate"] == pytest.approx(0.35)


def test_wargame_tabular_requires_calibration():
    """
    Test 2: Verify the runner correctly enforces the statistical constraint
    that calibration data must be provided.
    """
    risk = RiskProfile(level=RiskLevel.LOW, alpha=0.2)
    target = TargetModel(model_object=MODEL)
    logger = MockRCIALogger()
    attack_map = {"tabular": MockHopSkipJumpWrapper}

    runner = WargameRunner(
        target_model=target,
        risk_profile=risk,
        logger=logger,
        attack_wrapper_map=attack_map,
        defense_wrapper=MockUncertaintyWrapper
    )

    # Act & Assert: Should raise ValueError because X_calib is None
    with pytest.raises(ValueError, match="requires separate calibration data"):
        runner.run_wargame(X_test=X_tab, y_test=y_tab, X_calib=None)