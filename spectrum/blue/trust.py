import warnings
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
# MAPIE 1.0+ renamed MapieRegressor -> SplitConformalRegressor
try:
    from mapie.regression import MapieRegressor
except ImportError:
    from mapie.regression import SplitConformalRegressor as MapieRegressor

try:
    from mapie.classification import MapieClassifier
except ImportError:
    from mapie.classification import SplitConformalClassifier as MapieClassifier

from spectrum.infra.types import RiskProfile

"""
Solves the translation problem between stochastic
models and deterministic regulation.

1. Guarantees the validity of the predictions
on a finite sample: P(Y in C(X)) ≥ 1 - a
regardless of the underlying model's compexity
or the true data distribution.

2. Allows for defensible artifact generation.
The RiskProfile dictates the required confidence.
spectrum.lens logs the confidence and calculated
bounds alongside the decision.
"""

"""
spectrum.blue.trust
===================

Defensive Shield for Uncertainty Quantification.

This module resolves the translation problem between the stochastic nature of ML 
models and the deterministic requirements of regulation by enforcing a statistical 
guarantee on every model prediction.

CORE CONCEPT: CONFORMAL PREDICTION (CP)
--------------------------------------
Instead of outputting a single point estimate (e.g., P(X) = 0.8), this module 
produces a prediction set (classification) or interval (regression). The key is the 
**Validity Guarantee** on a finite sample:

$$
P(Y \in \hat{C}(X)) \geq 1 - \alpha
$$

Where:
- $Y$: The true outcome.
- $\hat{C}(X)$: The prediction set/interval (the output).
- $1 - \alpha$: The target confidence, determined by the `RiskProfile`.

This guarantee holds irrespective of the model's complexity or data distribution, 
provided the calibration data is exchangeable. This provides a robust, low-latency 
alternative to complex Bayesian methods.

KEY COMPONENT:
--------------
SpectrumUncertaintyWrapper: A unified wrapper that dynamically composes 
MapieClassifier or MapieRegressor to enforce CP calibration. It consumes the 
governance layer's `RiskProfile` to set the confidence level ($\alpha$).

USAGE FLOW:
-----------
1. Define Risk: `profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.01)`
2. Wrap Model: `trusted_model = SpectrumUncertaintyWrapper(base_model, profile)`
3. Calibrate: `trusted_model.fit(X_calib, y_calib)` (using hold-out data)
4. Predict: `result = trusted_model.predict(X_test)` (returns bounds/sets)
"""

class SpectrumRegressor(RegressorMixin, BaseEstimator):
    """
    A unified wrapper for Regressors and Classifiers that adds
    uncertainty quantification via Conformal Prediction (CP)
    to any sklearn-compatible Regressor.
    
    This class composes MapieRegressor based on the input model's type, 
    guaranteeing finite-sample coverage (1 - alpha) for the 
    prediction output (either an interval or a set).
    
    The wrapper assumes the base model is already trained and uses the .fit() 
    method *only* for calibrating the CP non-conformity scores using a 
    separate hold-out dataset.

    Parameters
    ----------
    base_model : sklearn.base.BaseEstimator
        The pre-fitted (trained) model instance (e.g., RandomForestClassifier).
        Must have a defined '_estimator_type' attribute (either 'classifier' or 'regressor').
    risk_profile : spectrum.infra.types.RiskProfile
        The governance contract specifying the required error tolerance (alpha) 
        which dictates the width of the prediction interval/set.

    Attributes
    ----------
    mapie_ : MapieRegressor or MapieClassifier
        The internal Mapie instance used for conformal calibration.
    is_calibrated_ : bool
        Flag indicating if the model has been fitted on calibration data.

    Methods
    -------
    fit(X_calib, y_calib):
        Calibrates the uncertainty using the provided calibration set.
    predict(X):
        Returns prediction output with uncertainty bounds (interval or set) 
        based on the model type.
    """
    def __init__(self, base_model: BaseEstimator, risk_profile: RiskProfile):
        """
        :param base_model: A fitted sklearn estimator
        :param risk_profile: Governance config defining the alpha (~error tolerance)
        """
        self.base_model = base_model
        self.risk_profile = risk_profile
        self._use_new_api = False

        # MAPIE 1.0+ uses prefit=True instead of cv="prefit"
        # and confidence_level instead of alpha in predict
        try:
            # Try new API first (MAPIE >= 1.0)
            self.mapie_ = MapieRegressor(
                estimator=self.base_model,
                prefit=True,
                confidence_level=1.0 - self.risk_profile.alpha
            )
            self._use_new_api = True
        except TypeError:
            # Fall back to old API (MAPIE < 1.0)
            self.mapie_ = MapieRegressor(estimator=self.base_model, cv="prefit")
        self.is_calibrated_ = False
        
    def fit(self, X_calib, y_calib):
        """
        Calibrates the conformal intervals using a hold-out dataset.
        DO NOT pass the training set here, or coverage guarantees are void.
        """
        try:
            check_is_fitted(self.base_model)
        except Exception:
            raise RuntimeError(
                """
                The base_model must be fitted on training data
                before being passed to SpectrumRegressor.
                """
            )
        # MAPIE 1.0+ uses conformalize() instead of fit() in prefit mode
        if hasattr(self.mapie_, 'conformalize'):
            self.mapie_.conformalize(X_calib, y_calib)
        else:
            self.mapie_.fit(X_calib, y_calib)
        self.is_calibrated_ = True
        return self

    def predict(self, X) -> Dict[str, Union[np.ndarray, float]]:
        """
        Returns a dictionary containing:
        - "prediction": The point estimate
        - "lower_bound": The lower estimate
        - "upper_bound": The upper
        - "confidence": 1 - alpha
        """
        if not self.is_calibrated_:
            raise RuntimeError("Model is not calibrated. Call .fit() with calibration data first.")

        # MAPIE 1.0+ uses predict_interval() instead of predict(alpha=...)
        if self._use_new_api:
            y_pred, y_pis = self.mapie_.predict_interval(X)
        else:
            y_pred, y_pis = self.mapie_.predict(X, alpha=self.risk_profile.alpha)

        lb, ub = y_pis[:, 0, 0].flatten(), y_pis[:, 1, 0].flatten()

        return {
            "prediction": y_pred,
            "lower_bound": lb,
            "upper_bound": ub,
            "confidence": 1 - self.risk_profile.alpha
        }

class SpectrumClassifier(ClassifierMixin, BaseEstimator):
    """
    A unified wrapper for Regressors and Classifiers that adds
    uncertainty quantification via Conformal Prediction (CP)
    to any sklearn-compatible Classifier.
    
    This class composes MapieRegressor based on the input model's type, 
    guaranteeing finite-sample coverage (1 - alpha) for the 
    prediction output (either an interval or a set).
    
    The wrapper assumes the base model is already trained and uses the .fit() 
    method *only* for calibrating the CP non-conformity scores using a 
    separate hold-out dataset.

    Parameters
    ----------
    base_model : sklearn.base.BaseEstimator
        The pre-fitted (trained) model instance (e.g., RandomForestClassifier).
        Must have a defined '_estimator_type' attribute (either 'classifier' or 'regressor').
    risk_profile : spectrum.infra.types.RiskProfile
        The governance contract specifying the required error tolerance (alpha) 
        which dictates the width of the prediction interval/set.

    Attributes
    ----------
    mapie_ : MapieRegressor or MapieClassifier
        The internal Mapie instance used for conformal calibration.
    is_calibrated_ : bool
        Flag indicating if the model has been fitted on calibration data.

    Methods
    -------
    fit(X_calib, y_calib):
        Calibrates the uncertainty using the provided calibration set.
    predict(X):
        Returns prediction output with uncertainty bounds (interval or set) 
        based on the model type.
    """
    
    def __init__(self, base_model: BaseEstimator, risk_profile: RiskProfile):
        """
        :param base_model: A fitted sklearn estimator
        :param risk_profile: Governance config defining the alpha (~error tolerance)
        """
        self.base_model = base_model
        self.risk_profile = risk_profile
        self._use_new_api = False

        # MAPIE 1.0+ uses prefit=True, conformity_score instead of method, and confidence_level
        try:
            # Try new API first (MAPIE >= 1.0)
            self.mapie_ = MapieClassifier(
                estimator=self.base_model,
                conformity_score="lac",
                prefit=True,
                confidence_level=1.0 - self.risk_profile.alpha
            )
            self._use_new_api = True
        except TypeError:
            # Fall back to old API (MAPIE < 1.0)
            self.mapie_ = MapieClassifier(
                estimator=self.base_model,
                method="lac",
                cv="prefit"
            )
        self.is_calibrated_ = False

    def fit(self, X_calib, y_calib):
        """
        Calibrates the conformal intervals using a hold-out dataset.
        DO NOT pass the training set here, or coverage guarantees are void.
        """
        try:
            check_is_fitted(self.base_model)
        except Exception:
            raise RuntimeError(
                """
                The base_model must be fitted on training data
                before being passed to SpectrumClassifier.
                """
            )
        # MAPIE 1.0+ uses conformalize() instead of fit() in prefit mode
        if hasattr(self.mapie_, 'conformalize'):
            self.mapie_.conformalize(X_calib, y_calib)
        else:
            self.mapie_.fit(X_calib, y_calib)
        self.is_calibrated_ = True
        return self

    def predict(self, X) -> Dict[str, Union[np.ndarray, float]]:
        """
        Returns a dictionary containing:
        - "prediction": The point estimate
        - "prediction_set": The set of predicted labels
        - "classes": The predicted classes
        - "confidence": 1 - alpha
        """
        if not self.is_calibrated_:
            raise RuntimeError("Model is not calibrated. Call .fit() with calibration data first.")

        # MAPIE 1.0+ uses predict_set() instead of predict(alpha=...)
        if self._use_new_api:
            y_pred, y_set = self.mapie_.predict_set(X)
        else:
            y_pred, y_set = self.mapie_.predict(X, alpha=self.risk_profile.alpha)

        if hasattr(self.mapie_, "classes_"):
            classes = self.mapie_.classes_
        else:
            classes = np.arange(y_set.shape[1])

        return {
            "prediction": y_pred,
            "prediction_set": y_set,  # (N, K)
            "classes": classes.tolist(),
            "confidence": 1 - self.risk_profile.alpha
        }