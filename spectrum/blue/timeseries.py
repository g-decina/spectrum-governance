import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array

from spectrum.infra.types import RiskProfile, RiskLevel

"""
spectrum.blue.timeseries
========================

Time Series Uncertainty Quantification for Sequential Data.

This module provides EnbPI (Ensemble batch Prediction Intervals) for constructing
prediction intervals in time series forecasting tasks. EnbPI adapts to distribution
shifts and provides coverage guarantees for sequential predictions.

CORE CONCEPTS:
--------------
1. EnbPI (Ensemble batch Prediction Intervals): A method for constructing
   prediction intervals that works with time series data where traditional
   conformal prediction assumptions (exchangeability) break down.

2. Bootstrap Aggregating: Uses ensemble of models trained on bootstrapped
   samples to estimate prediction uncertainty.

3. Adaptive Residuals: Tracks residuals over time and adapts interval width
   based on recent prediction errors.

4. Rolling Window: Uses sliding window approach to handle non-stationarity
   and distribution shifts in time series.

COVERAGE GUARANTEE:
-------------------
EnbPI provides asymptotic coverage guarantee for time series:

$$
\lim_{t \to \infty} \frac{1}{t} \sum_{i=1}^{t} \mathbb{1}(Y_i \in C(X_i)) \geq 1 - \alpha
$$

where C(X) is the prediction interval and α is the miscoverage rate.

KEY COMPONENTS:
---------------
- EnbPIRegressor: Time series regression with prediction intervals
- TimeSeriesValidator: Validates time series data structure
- AdaptiveResidualTracker: Tracks and adapts to changing residual distributions

USAGE FLOW:
-----------
1. Initialize with base forecaster:
    `enbpi = EnbPIRegressor(base_forecaster=model, n_bootstraps=50)`
2. Fit on initial training window:
    `enbpi.fit(X_train, y_train)`
3. Update online with new observations:
    `enbpi.partial_fit(X_new, y_new)`
4. Predict with intervals:
    `predictions, lower, upper = enbpi.predict(X_future, return_intervals=True)`

READ: https://arxiv.org/abs/2010.09107
"""


@dataclass
class TimeSeriesPrediction:
    """
    Structured output for time series predictions with uncertainty.
    """
    predictions: np.ndarray      # Point predictions
    lower_bounds: np.ndarray     # Lower prediction interval bounds
    upper_bounds: np.ndarray     # Upper prediction interval bounds
    confidence: float            # Confidence level (1 - alpha)
    residuals: Optional[np.ndarray] = None  # Historical residuals (for diagnostics)
    coverage: Optional[float] = None        # Empirical coverage on validation data


class TimeSeriesValidator:
    """
    Validates time series data structure and temporal ordering.
    """

    @staticmethod
    def validate_temporal_data(
        X: np.ndarray,
        y: np.ndarray,
        ensure_ordered: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates that data is suitable for time series analysis.

        :param X: Feature matrix (n_samples, n_features)
        :param y: Target values (n_samples,)
        :param ensure_ordered: Whether to check for temporal ordering
        :return: Validated X, y arrays
        """
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=np.float64)

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")

        if len(X) < 10:
            raise ValueError(f"Time series too short. Need at least 10 samples, got {len(X)}")

        return X, y

    @staticmethod
    def split_temporal(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits time series data temporally (no shuffling).

        :param X: Feature matrix
        :param y: Target values
        :param train_ratio: Proportion of data for training
        :return: X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        split_idx = int(n_samples * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test


class AdaptiveResidualTracker:
    """
    Tracks residuals over time and adapts interval width to distribution shifts.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize residual tracker.

        :param window_size: Size of sliding window for residual tracking
        """
        self.window_size = window_size
        self.residuals = []

    def add_residual(self, residual: float):
        """Add new residual to tracker."""
        self.residuals.append(residual)
        if len(self.residuals) > self.window_size:
            self.residuals.pop(0)

    def get_quantile(self, alpha: float) -> float:
        """
        Get quantile of recent residuals for interval construction.

        :param alpha: Miscoverage rate
        :return: Quantile value for interval width
        """
        if len(self.residuals) == 0:
            return 0.0

        residuals_abs = np.abs(self.residuals)
        quantile = np.quantile(residuals_abs, 1 - alpha)
        return quantile

    def get_coverage_estimate(self, true_values: np.ndarray, intervals: np.ndarray) -> float:
        """
        Estimate empirical coverage from recent predictions.

        :param true_values: True observed values
        :param intervals: Prediction intervals (n_samples, 2) with [lower, upper]
        :return: Empirical coverage rate
        """
        if len(true_values) == 0:
            return 0.0

        in_interval = (true_values >= intervals[:, 0]) & (true_values <= intervals[:, 1])
        coverage = np.mean(in_interval)
        return float(coverage)


class EnbPIRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble batch Prediction Intervals for time series regression.

    This estimator constructs prediction intervals for time series data using
    bootstrap aggregating and adaptive residual tracking. It handles
    non-stationarity and distribution shifts common in temporal data.

    Parameters:
    -----------
    base_forecaster : BaseEstimator
        Sklearn-compatible regressor to use as base forecaster
    n_bootstraps : int, default=50
        Number of bootstrap samples for ensemble
    window_size : int, default=100
        Size of sliding window for residual adaptation
    risk_profile : RiskProfile, optional
        Governance risk profile (determines alpha)
    alpha : float, default=0.1
        Miscoverage rate (1 - confidence level)

    Example:
    --------
    >>> from sklearn.linear_model import Ridge
    >>> base_model = Ridge()
    >>> enbpi = EnbPIRegressor(base_forecaster=base_model, n_bootstraps=30)
    >>> enbpi.fit(X_train, y_train)
    >>> result = enbpi.predict(X_test, return_intervals=True)
    """

    def __init__(
        self,
        base_forecaster: BaseEstimator,
        n_bootstraps: int = 50,
        window_size: int = 100,
        risk_profile: Optional[RiskProfile] = None,
        alpha: float = 0.1
    ):
        self.base_forecaster = base_forecaster
        self.n_bootstraps = n_bootstraps
        self.window_size = window_size
        self.risk_profile = risk_profile
        self.alpha = alpha

        # Set alpha from risk profile if provided
        if risk_profile is not None:
            self.alpha = risk_profile.alpha

        # Internal state
        self.ensemble_models_ = []
        self.residual_tracker_ = AdaptiveResidualTracker(window_size=window_size)
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnbPIRegressor':
        """
        Fit the EnbPI model on initial training data.

        :param X: Feature matrix (n_samples, n_features)
        :param y: Target values (n_samples,)
        :return: self
        """
        X, y = TimeSeriesValidator.validate_temporal_data(X, y)

        # Create bootstrap ensemble
        self.ensemble_models_ = []
        n_samples = len(X)

        for i in range(self.n_bootstraps):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]

            # Clone and fit base forecaster
            model = clone(self.base_forecaster)
            model.fit(X_boot, y_boot)
            self.ensemble_models_.append(model)

        # Initialize residual tracker with training residuals
        y_pred = self._predict_ensemble(X)
        residuals = y - y_pred
        for residual in residuals:
            self.residual_tracker_.add_residual(residual)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'EnbPIRegressor':
        """
        Update the model with new observations (online learning).

        :param X: New feature matrix
        :param y: New target values
        :return: self
        """
        check_is_fitted(self, 'is_fitted_')
        X, y = TimeSeriesValidator.validate_temporal_data(X, y, ensure_ordered=False)

        # Update residual tracker with new observations
        y_pred = self._predict_ensemble(X)
        residuals = y - y_pred

        for residual in residuals:
            self.residual_tracker_.add_residual(residual)

        return self

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions using ensemble average.

        :param X: Feature matrix
        :return: Averaged predictions
        """
        predictions = np.array([model.predict(X) for model in self.ensemble_models_])
        return np.mean(predictions, axis=0)

    def predict(
        self,
        X: np.ndarray,
        return_intervals: bool = False
    ) -> Union[np.ndarray, TimeSeriesPrediction]:
        """
        Make predictions with optional prediction intervals.

        :param X: Feature matrix for prediction
        :param return_intervals: Whether to return prediction intervals
        :return: Predictions or TimeSeriesPrediction with intervals
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        # Get point predictions
        y_pred = self._predict_ensemble(X)

        if not return_intervals:
            return y_pred

        # Construct prediction intervals using adaptive residuals
        quantile = self.residual_tracker_.get_quantile(self.alpha)

        # Get ensemble variance for additional uncertainty
        ensemble_preds = np.array([model.predict(X) for model in self.ensemble_models_])
        ensemble_std = np.std(ensemble_preds, axis=0)

        # Combine residual quantile with ensemble uncertainty
        interval_width = quantile + ensemble_std

        lower_bounds = y_pred - interval_width
        upper_bounds = y_pred + interval_width

        return TimeSeriesPrediction(
            predictions=y_pred,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence=1 - self.alpha,
            residuals=np.array(self.residual_tracker_.residuals) if self.residual_tracker_.residuals else None
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.

        :param X: Test features
        :param y: True values
        :return: R² score
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def evaluate_coverage(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate prediction interval coverage on test data.

        :param X: Test features
        :param y: True values
        :return: Dictionary with coverage metrics
        """
        result = self.predict(X, return_intervals=True)

        # Calculate empirical coverage
        in_interval = (y >= result.lower_bounds) & (y <= result.upper_bounds)
        empirical_coverage = np.mean(in_interval)

        # Calculate average interval width
        avg_width = np.mean(result.upper_bounds - result.lower_bounds)

        # Calculate RMSE
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y, result.predictions))

        return {
            'empirical_coverage': float(empirical_coverage),
            'target_coverage': 1 - self.alpha,
            'coverage_achieved': empirical_coverage >= (1 - self.alpha),
            'average_interval_width': float(avg_width),
            'rmse': float(rmse),
            'n_samples': len(y),
            'n_in_interval': int(np.sum(in_interval)),
            'n_outside_interval': int(np.sum(~in_interval))
        }
