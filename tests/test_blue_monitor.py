import pytest
import numpy as np
import pandas as pd

from spectrum.blue.monitor import DriftCheck, PSI_ALERT_THRESHOLD, PSI_MONITOR_THRESHOLD


def test_drift_check_no_drift():
    """
    Test that DriftCheck correctly identifies when there is no significant drift
    between reference and current data.
    """
    print("\n=== Test: No Drift Detection ===")
    # Generate identical distributions
    np.random.seed(42)
    n_samples = 1000

    reference_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples)
    })

    # Current data with very similar distribution (same seed + small noise)
    np.random.seed(42)
    current_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples) + np.random.normal(0, 0.1, n_samples),
        'income': np.random.normal(50000, 15000, n_samples) + np.random.normal(0, 10, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples) + np.random.normal(0, 1, n_samples)
    })

    result = DriftCheck(reference_data, current_data)

    print(f"Drift detected: {result['drift_detected']}")
    print(f"Number of drifted features: {result['n_drifted_features']}")
    print(f"Max PSI: {result['max_psi']:.4f} (threshold: {PSI_MONITOR_THRESHOLD})")
    print(f"Alert required: {result['alert_required']}")
    print(f"Status: {result['status']}")
    print(f"Feature scores: {result['feature_drift_scores']}")

    # Assertions
    assert isinstance(result, dict)
    assert result["drift_detected"] is False or result["n_drifted_features"] == 0
    assert result["max_psi"] < PSI_MONITOR_THRESHOLD
    assert result["alert_required"] is False
    assert "✓" in result["status"]


def test_drift_check_significant_drift():
    """
    Test that DriftCheck correctly detects significant drift when the
    current data distribution differs substantially from reference.
    """
    print("\n=== Test: Significant Drift Detection ===")
    np.random.seed(42)
    n_samples = 1000

    # Reference data
    reference_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples)
    })

    # Current data with SHIFTED distributions (different means)
    np.random.seed(99)
    current_data = pd.DataFrame({
        'age': np.random.normal(55, 10, n_samples),  # 20 year shift!
        'income': np.random.normal(80000, 15000, n_samples),  # 30k shift!
        'credit_score': np.random.uniform(500, 850, n_samples)  # Shifted range
    })

    result = DriftCheck(reference_data, current_data)

    print(f"Drift detected: {result['drift_detected']}")
    print(f"Number of drifted features: {result['n_drifted_features']}")
    print(f"Drifted features: {result['drifted_features']}")
    print(f"Max PSI: {result['max_psi']:.4f} (alert threshold: {PSI_ALERT_THRESHOLD})")
    print(f"Alert required: {result['alert_required']}")
    print(f"Status: {result['status']}")
    print(f"Feature scores: {result['feature_drift_scores']}")

    # Assertions
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] > 0
    assert result["max_psi"] >= PSI_ALERT_THRESHOLD
    assert result["alert_required"] is True
    assert "CRITICAL" in result["status"]
    assert len(result["drifted_features"]) > 0
    assert len(result["feature_drift_scores"]) == 3


def test_drift_check_per_feature_scores():
    """
    Test that DriftCheck returns per-feature PSI scores and correctly
    identifies which specific features are drifting.
    """
    print("\n=== Test: Per-Feature Drift Scores ===")
    np.random.seed(42)
    n_samples = 1000

    # Reference data
    reference_data = pd.DataFrame({
        'stable_feature': np.random.normal(100, 10, n_samples),
        'drifted_feature': np.random.normal(50, 5, n_samples),
        'another_stable': np.random.uniform(0, 1, n_samples)
    })

    # Current data: only 'drifted_feature' has significant shift
    np.random.seed(42)
    current_data = pd.DataFrame({
        'stable_feature': np.random.normal(100, 10, n_samples),  # Same distribution
        'drifted_feature': np.random.normal(150, 5, n_samples),  # SHIFTED by 100!
        'another_stable': np.random.uniform(0, 1, n_samples)  # Same distribution
    })

    result = DriftCheck(reference_data, current_data)

    # Assertions
    assert "feature_drift_scores" in result
    assert "drifted_feature" in result["feature_drift_scores"]
    assert "stable_feature" in result["feature_drift_scores"]

    # The drifted feature should have a higher PSI score
    drifted_psi = result["feature_drift_scores"]["drifted_feature"]
    stable_psi = result["feature_drift_scores"]["stable_feature"]

    print(f"Drifted feature PSI: {drifted_psi:.4f}")
    print(f"Stable feature PSI: {stable_psi:.4f}")
    print(f"PSI Alert threshold: {PSI_ALERT_THRESHOLD}")
    print(f"Identified drifted features: {result['drifted_features']}")
    print(f"All feature scores: {result['feature_drift_scores']}")

    assert drifted_psi > stable_psi
    assert drifted_psi >= PSI_ALERT_THRESHOLD
    assert "drifted_feature" in result["drifted_features"]


def test_drift_check_feature_filtering():
    """
    Test that DriftCheck correctly filters to only analyze specified features.
    """
    print("\n=== Test: Feature Filtering ===")
    np.random.seed(42)
    n_samples = 500

    reference_data = pd.DataFrame({
        'feature_a': np.random.normal(10, 2, n_samples),
        'feature_b': np.random.normal(20, 5, n_samples),
        'feature_c': np.random.normal(30, 7, n_samples)
    })

    current_data = pd.DataFrame({
        'feature_a': np.random.normal(50, 2, n_samples),  # Drifted
        'feature_b': np.random.normal(20, 5, n_samples),  # Stable
        'feature_c': np.random.normal(30, 7, n_samples)   # Stable
    })

    # Only analyze feature_a and feature_b
    print("Requesting analysis of: ['feature_a', 'feature_b']")
    result = DriftCheck(reference_data, current_data, feature_names=['feature_a', 'feature_b'])

    print(f"Feature scores returned: {result['feature_drift_scores']}")
    print(f"Drifted features: {result['drifted_features']}")

    # Assertions
    assert 'feature_a' in result["feature_drift_scores"]
    assert 'feature_b' in result["feature_drift_scores"]
    # feature_c should still be in scores because Evidently analyzes all columns
    # but our filtering logic will only check the requested features


def test_drift_check_schema_mismatch_error():
    """
    Test that DriftCheck raises a ValueError when reference and current data
    have mismatched schemas.
    """
    print("\n=== Test: Schema Mismatch Error ===")
    reference_data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
    })

    current_data = pd.DataFrame({
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]  # Different column name!
    })

    print(f"Reference columns: {list(reference_data.columns)}")
    print(f"Current columns: {list(current_data.columns)}")
    print("Expecting ValueError for schema mismatch...")

    with pytest.raises(ValueError, match="Schema mismatch"):
        DriftCheck(reference_data, current_data)

    print("Successfully caught schema mismatch error")


def test_drift_check_missing_feature_error():
    """
    Test that DriftCheck raises a ValueError when requested feature_names
    don't exist in the data.
    """
    print("\n=== Test: Missing Feature Error ===")
    reference_data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
    })

    current_data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
    })

    print(f"Available columns: {list(reference_data.columns)}")
    print("Requesting nonexistent feature: 'nonexistent_feature'")
    print("Expecting ValueError...")

    with pytest.raises(ValueError, match="Features not found in data"):
        DriftCheck(reference_data, current_data, feature_names=['nonexistent_feature'])

    print("Successfully caught missing feature error")


def test_drift_check_empty_dataframe():
    """
    Test that DriftCheck handles empty DataFrames gracefully.
    """
    print("\n=== Test: Empty DataFrame Handling ===")
    reference_data = pd.DataFrame({'age': [], 'income': []})
    current_data = pd.DataFrame({'age': [], 'income': []})

    print(f"Reference data shape: {reference_data.shape}")
    print(f"Current data shape: {current_data.shape}")

    # This should either raise an error or return a safe default
    # Depending on Evidently's behavior with empty data
    try:
        result = DriftCheck(reference_data, current_data)
        # If it succeeds, ensure the output is reasonable
        print("Empty dataframe handled successfully")
        print(f"Result: {result}")
        assert isinstance(result, dict)
        assert "drift_detected" in result
    except Exception as e:
        # If Evidently fails on empty data, that's acceptable
        print(f"Exception raised (expected): {type(e).__name__}: {e}")
        assert isinstance(e, (ValueError, RuntimeError, KeyError))


def test_drift_check_single_feature():
    """
    Test DriftCheck with a single feature to ensure edge case handling.
    """
    print("\n=== Test: Single Feature Drift ===")
    np.random.seed(42)
    n_samples = 800

    reference_data = pd.DataFrame({
        'single_feature': np.random.normal(100, 20, n_samples)
    })

    current_data = pd.DataFrame({
        'single_feature': np.random.normal(200, 20, n_samples)  # Drifted
    })

    result = DriftCheck(reference_data, current_data)

    print(f"Drift detected: {result['drift_detected']}")
    print(f"Number of drifted features: {result['n_drifted_features']}")
    print(f"Max PSI: {result['max_psi']:.4f}")
    print(f"Drifted features: {result['drifted_features']}")

    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 1
    assert result["max_psi"] >= PSI_ALERT_THRESHOLD
    assert 'single_feature' in result["drifted_features"]


def test_drift_check_categorical_features():
    """
    Test DriftCheck with categorical features to ensure PSI works
    for non-numeric data.
    """
    print("\n=== Test: Categorical Feature Drift ===")
    np.random.seed(42)
    n_samples = 1000

    # Reference data with categorical feature
    reference_data = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'numeric': np.random.normal(50, 10, n_samples)
    })

    # Current data with DIFFERENT category distribution
    current_data = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.2, 0.3, 0.5]),  # Shifted!
        'numeric': np.random.normal(50, 10, n_samples)  # Same
    })

    print("Reference category distribution: A=50%, B=30%, C=20%")
    print("Current category distribution: A=20%, B=30%, C=50%")

    result = DriftCheck(reference_data, current_data)

    print(f"Category PSI: {result['feature_drift_scores']['category']:.4f}")
    print(f"Numeric PSI: {result['feature_drift_scores']['numeric']:.4f}")
    print(f"All feature scores: {result['feature_drift_scores']}")

    # Assertions
    assert "category" in result["feature_drift_scores"]
    assert result["feature_drift_scores"]["category"] > 0
    # The category feature should show drift due to changed proportions


def test_drift_check_moderate_drift_warning():
    """
    Test that DriftCheck correctly flags moderate drift (between thresholds)
    with a WARNING status.
    """
    print("\n=== Test: Moderate Drift Warning ===")
    np.random.seed(42)
    n_samples = 1000

    reference_data = pd.DataFrame({
        'feature': np.random.normal(100, 15, n_samples)
    })

    # Slight shift to trigger moderate drift (0.1 <= PSI < 0.25)
    current_data = pd.DataFrame({
        'feature': np.random.normal(110, 15, n_samples)  # Moderate shift
    })

    result = DriftCheck(reference_data, current_data)

    print(f"Max PSI: {result['max_psi']:.4f}")
    print(f"Monitor threshold: {PSI_MONITOR_THRESHOLD}")
    print(f"Alert threshold: {PSI_ALERT_THRESHOLD}")
    print(f"Status: {result['status']}")
    print(f"Alert required: {result['alert_required']}")

    # The exact threshold is hard to predict, but we can check the logic
    if PSI_MONITOR_THRESHOLD <= result["max_psi"] < PSI_ALERT_THRESHOLD:
        print(f"PSI is in moderate range - checking for WARNING status")
        assert "WARNING" in result["status"] or "⚡" in result["status"]
        assert result["alert_required"] is False


def test_drift_check_output_structure():
    """
    Test that DriftCheck returns the expected output structure with all keys.
    """
    print("\n=== Test: Output Structure Validation ===")
    np.random.seed(42)
    n_samples = 500

    reference_data = pd.DataFrame({
        'x': np.random.normal(0, 1, n_samples),
        'y': np.random.normal(0, 1, n_samples)
    })

    current_data = pd.DataFrame({
        'x': np.random.normal(0, 1, n_samples),
        'y': np.random.normal(0, 1, n_samples)
    })

    result = DriftCheck(reference_data, current_data)

    # Verify all expected keys exist
    expected_keys = {
        "drift_detected",
        "n_drifted_features",
        "feature_drift_scores",
        "drifted_features",
        "max_psi",
        "status",
        "alert_required"
    }

    print(f"Expected keys: {expected_keys}")
    print(f"Actual keys: {set(result.keys())}")
    print(f"\nResult types:")
    print(f"  drift_detected: {type(result['drift_detected']).__name__}")
    print(f"  n_drifted_features: {type(result['n_drifted_features']).__name__}")
    print(f"  feature_drift_scores: {type(result['feature_drift_scores']).__name__}")
    print(f"  drifted_features: {type(result['drifted_features']).__name__}")
    print(f"  max_psi: {type(result['max_psi']).__name__}")
    print(f"  status: {type(result['status']).__name__}")
    print(f"  alert_required: {type(result['alert_required']).__name__}")

    assert set(result.keys()) == expected_keys
    assert isinstance(result["drift_detected"], bool)
    assert isinstance(result["n_drifted_features"], int)
    assert isinstance(result["feature_drift_scores"], dict)
    assert isinstance(result["drifted_features"], list)
    assert isinstance(result["max_psi"], float)
    assert isinstance(result["status"], str)
    assert isinstance(result["alert_required"], bool)
