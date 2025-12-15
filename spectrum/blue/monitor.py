import numpy as np
import pandas as pd
import warnings

from typing import Dict, List, Any, Tuple

r"""
spectrum.blue.monitor
=====================

Defensive Shield for Production Data Quality.

This module detects distribution shifts in production data streams using Population
Stability Index (PSI) to ensure the model's input distribution remains statistically
consistent with the training distribution. Calculations are done using NumPy to keep
this implementation verifiable and white-box.

CORE CONCEPT: POPULATION STABILITY INDEX (PSI)
----------------------------------------------
PSI quantifies the change in a feature's distribution between two datasets
(reference vs. current). It is calculated as:

$$
PSI = \sum_{i=1}^{n} (\%_{current,i} - \%_{reference,i}) \times \ln\left(\frac{\%_{current,i}}{\%_{reference,i}}\right)
$$

Where each feature is binned into n buckets, and percentages are computed per bucket.

PSI THRESHOLDS:
--------------
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.25: Slight shift [MONITOR]
- PSI ≥ 0.25: Significant shift [ALERT - Model may be unsafe]

KEY COMPONENT:
--------------
DriftCheck: Custom PSI implementation providing granular, per-feature drift
analysis with actionable alerts for governance enforcement.

USAGE FLOW:
-----------
1. Collect reference_data (training or validation set)
2. Stream current_data (live production batch)
3. Call DriftCheck(reference_data, current_data)
4. The function returns drift status, per-feature PSI scores, and alert flags
5. The RCIALogger should log this output as a critical governance event
"""

# Default PSI Thresholds
PSI_MONITOR_THRESHOLD = 0.1   # Warning: Distribution shift detected
PSI_ALERT_THRESHOLD = 0.25    # Critical: Model may be unsafe

def _calculate_psi_numeric(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) for a numerical feature.

    PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

    Parameters
    ----------
    reference_values : np.ndarray
        Reference distribution (baseline/training data).
    current_values : np.ndarray
        Current distribution (production data).
    n_bins : int, default=10
        Number of bins to use for binning continuous variables.

    Returns
    -------
    float
        PSI score for the feature.
    """
    # Remove NaN values
    reference_values = reference_values[~np.isnan(reference_values)]
    current_values = current_values[~np.isnan(current_values)]

    if len(reference_values) == 0 or len(current_values) == 0:
        return 0.0

    # Bin the reference data using quantiles to get equal-frequency bins
    try:
        breakpoints = np.percentile(reference_values, np.linspace(0, 100, n_bins + 1))
        # Remove duplicate breakpoints (can happen with discrete data)
        breakpoints = np.unique(breakpoints)
    except Exception:
        # Fallback to linspace if percentile fails
        min_val, max_val = reference_values.min(), reference_values.max()
        breakpoints = np.linspace(min_val, max_val, n_bins + 1)

    # Ensure bins cover full range
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count samples in each bin for both distributions
    reference_counts = np.histogram(reference_values, bins=breakpoints)[0]
    current_counts = np.histogram(current_values, bins=breakpoints)[0]

    # Convert to percentages
    reference_percents = reference_counts / len(reference_values)
    current_percents = current_counts / len(current_values)

    # Avoid division by zero and log(0) by adding small epsilon
    epsilon = 1e-10
    reference_percents = np.maximum(reference_percents, epsilon)
    current_percents = np.maximum(current_percents, epsilon)

    # Calculate PSI
    psi = np.sum((current_percents - reference_percents) * np.log(current_percents / reference_percents))

    return float(psi)


def _calculate_psi_categorical(
    reference_values: np.ndarray,
    current_values: np.ndarray
) -> float:
    """
    Calculate Population Stability Index (PSI) for a categorical feature.

    Parameters
    ----------
    reference_values : np.ndarray
        Reference distribution (baseline/training data).
    current_values : np.ndarray
        Current distribution (production data).

    Returns
    -------
    float
        PSI score for the feature.
    """
    # Remove NaN values
    reference_values = reference_values[pd.notna(reference_values)]
    current_values = current_values[pd.notna(current_values)]

    if len(reference_values) == 0 or len(current_values) == 0:
        return 0.0

    # Get all unique categories from both distributions
    all_categories = np.union1d(
        np.unique(reference_values),
        np.unique(current_values)
    )

    # Calculate proportions for each category
    reference_counts = pd.Series(reference_values).value_counts()
    current_counts = pd.Series(current_values).value_counts()

    # Convert to percentages with epsilon to avoid log(0)
    epsilon = 1e-10
    reference_percents = {}
    current_percents = {}

    for category in all_categories:
        ref_pct = reference_counts.get(category, 0) / len(reference_values)
        curr_pct = current_counts.get(category, 0) / len(current_values)

        reference_percents[category] = max(ref_pct, epsilon)
        current_percents[category] = max(curr_pct, epsilon)

    # Calculate PSI
    psi = 0.0
    for category in all_categories:
        curr_pct = current_percents[category]
        ref_pct = reference_percents[category]
        psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)

    return float(psi)


def DriftCheck(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_names: List[str] = None,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Performs comprehensive data drift detection using custom PSI implementation.

    This function calculates PSI (Population Stability Index) for each feature:
    1. Overall dataset drift status (Boolean)
    2. Per-feature PSI scores (Dict[str, float])
    3. List of drifted features exceeding the alert threshold
    4. Maximum PSI score across all features
    5. Actionable status message

    Parameters
    ----------
    reference_data : pd.DataFrame
        The baseline (training/validation) dataset. Must contain the same
        columns as current_data.
    current_data : pd.DataFrame
        The live production data batch. Must match reference_data schema.
    feature_names : List[str], optional
        Specific features to monitor. If None, all shared columns are analyzed.
    n_bins : int, default=10
        Number of bins for numerical features (ignored for categorical).

    Returns
    -------
    Dict[str, Any]
        {
            "drift_detected": bool,
            "n_drifted_features": int,
            "feature_drift_scores": Dict[str, float],  # PSI per feature
            "drifted_features": List[str],             # Features exceeding threshold
            "max_psi": float,                          # Highest PSI score
            "status": str,                             # Human-readable summary
            "alert_required": bool                     # True if max_psi >= 0.25
        }

    Raises
    ------
    ValueError
        If reference_data and current_data do not share the same columns.

    Examples
    --------
    >>> ref = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
    >>> curr = pd.DataFrame({'age': [45, 50, 55], 'income': [50000, 60000, 70000]})
    >>> result = DriftCheck(ref, curr)
    >>> if result["alert_required"]:
    ...     print(f"ALERT: {result['status']}")
    """
    # 1. VALIDATION: Ensure schema consistency
    if not reference_data.columns.equals(current_data.columns):
        missing_in_current = set(reference_data.columns) - set(current_data.columns)
        missing_in_ref = set(current_data.columns) - set(reference_data.columns)

        error_msg = "Schema mismatch between reference and current data.\n"
        if missing_in_current:
            error_msg += f"Missing in current: {missing_in_current}\n"
        if missing_in_ref:
            error_msg += f"Missing in reference: {missing_in_ref}"

        raise ValueError(error_msg)

    # 2. FEATURE FILTERING: Select features to monitor
    if feature_names is None:
        feature_names = reference_data.columns.tolist()
    else:
        # Validate that requested features exist
        invalid_features = set(feature_names) - set(reference_data.columns)
        if invalid_features:
            raise ValueError(f"Features not found in data: {invalid_features}")

    # 3. PSI CALCULATION: Calculate PSI for each feature
    feature_drift_scores = {}
    drifted_features = []

    for feature_name in feature_names:
        ref_values = reference_data[feature_name].values
        curr_values = current_data[feature_name].values

        # Determine if feature is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(reference_data[feature_name])

        # Calculate PSI based on feature type
        if is_numeric:
            psi_score = _calculate_psi_numeric(ref_values, curr_values, n_bins)
        else:
            psi_score = _calculate_psi_categorical(ref_values, curr_values)

        feature_drift_scores[feature_name] = psi_score

        # Check if this feature exceeds alert threshold
        if psi_score >= PSI_ALERT_THRESHOLD:
            drifted_features.append(feature_name)

    # 4. AGGREGATION: Calculate overall metrics
    if feature_drift_scores:
        max_psi = max(feature_drift_scores.values())
    else:
        max_psi = 0.0

    # Count features that exceed threshold
    n_drifted_features = len(drifted_features)
    drift_detected = (n_drifted_features > 0)

    # 5. STATUS MESSAGE: Generate actionable summary
    if not drift_detected:
        status = "No significant drift detected. Model inputs are stable."
    elif max_psi >= PSI_ALERT_THRESHOLD:
        status = (
            f"CRITICAL: {n_drifted_features} feature(s) exceeded PSI threshold. "
            f"Features: {', '.join(drifted_features)}. Max PSI: {max_psi:.3f}."
        )
    elif max_psi >= PSI_MONITOR_THRESHOLD:
        status = (
            f"WARNING: {n_drifted_features} feature(s) showing drift. "
            f"Monitor closely. Max PSI: {max_psi:.3f}."
        )
    else:
        status = "✓ Minor drift detected but below monitoring threshold."

    # 6. RETURN AUDIT OUTPUT
    return {
        "drift_detected": drift_detected,
        "n_drifted_features": n_drifted_features,
        "feature_drift_scores": feature_drift_scores,
        "drifted_features": drifted_features,
        "max_psi": max_psi,
        "status": status,
        "alert_required": (max_psi >= PSI_ALERT_THRESHOLD)
    }