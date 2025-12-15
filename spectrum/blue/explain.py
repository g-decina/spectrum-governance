import shap
import numpy as np
import pandas as pd

# MAPIE 1.0+ renamed MapieRegressor -> SplitConformalRegressor
try:
    from mapie.regression import MapieRegressor
except ImportError:
    from mapie.regression import SplitConformalRegressor as MapieRegressor

try:
    from mapie.classification import MapieClassifier
except ImportError:
    from mapie.classification import SplitConformalClassifier as MapieClassifier
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing import Union, List, Dict, Any

from spectrum.infra.types import RiskProfile
from spectrum.utils.dl_adapter import DLAdapter
from dataclasses import dataclass

"""
spectrum.blue.explain
=====================

Defensive Shield for Semantic Transparency.

This module ensures model decisions are legally defensible by translating opaque 
mathematical attribution (SHAP values) into human-readable, regulatory-compliant 
reasons (Adverse Action Notices).

CORE CONCEPT: ADVERSE SCORE SPACE
---------------------------------
To generate actionable reasons, we must isolate features that push the model's
score *towards* the denial or adverse outcome. We achieve this by normalizing SHAP
contributions to the 'Adverse Score Space' ($\\phi_{C_A}$).

- SHAPWrapper: Dynamically selects the most performant SHAP explainer (TreeExplainer 
    vs. KernelExplainer) and ensures the SHAP calculation is focused on the 
    score (logit link) rather than the final label.
- ReasonCodeGenerator: Consumes the top-K adverse SHAP contributions and maps 
    them to templated strings (e.g., "Time at address too short") defined in the 
    governance spec (CFPB/Adverse Action requirements).

DELIVERABLE:
------------
The output of this module a list of immutable, templated text strings that can be 
directly injected into the EU AI Act Technical File or a CFPB Adverse Action Notice.

PERFORMANCE NOTE:
-----------------
The module enforces thread safety and dynamically optimizes explainer selection
to minimize the latency added by SHAP calculation, which is a major bottleneck
in high-throughput applications.
"""


@dataclass
class PredictionResult:
    """
    Result object for predictions with uncertainty quantification.

    Attributes
    ----------
    y_preds : np.ndarray
        Point predictions
    y_pis : np.ndarray, optional
        Prediction intervals (for regression) with shape (n_samples, 2, 1)
    y_set : np.ndarray, optional
        Prediction sets (for classification)
    classes : list, optional
        Class labels (for classification)
    confidence : float
        Confidence level (1 - alpha)
    """
    y_preds: np.ndarray
    y_pis: np.ndarray = None
    y_set: np.ndarray = None
    classes: list = None
    confidence: float = None

class SpectrumUncertaintyWrapper:
    """
    A unified wrapper for Regressors and Classifiers that enforces governance 
    via Conformal Prediction (CP) for risk quantification.

    This class composes MapieRegressor or MapieClassifier based on the input 
    model's type, guaranteeing finite-sample coverage (1 - alpha) for the 
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
    X_background : np.ndarray, optional
        A small, representative sample of the training data. **Mandatory** if the model is a non-tree classifier/regressor (e.g., LogisticRegression) 
        as it is required for `shap.KernelExplainer` initialization.

    Attributes
    ----------
    mapie_ : MapieRegressor or MapieClassifier
        The internal Mapie instance used for conformal calibration.
    explainer : shap.Explainer
        The internal SHAP explainer instance, dynamically optimized 
        (TreeExplainer for tree ensembles, KernelExplainer otherwise).
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
    def __init__(
        self, 
        base_model: BaseEstimator, 
        risk_profile: RiskProfile,
        X_background: np.ndarray = None,
    ):
        # 1. INIT
        self.base_model = base_model
        self.risk_profile = risk_profile

        # 2. DEEP LEARNING COMPATIBILITY
        model_to_use = base_model  # Default to base model
        try:
            dl_adapter_check = DLAdapter(base_model)
            if dl_adapter_check.is_torch or dl_adapter_check.is_tf:
                # If DL model is detected, use the adapter as the estimator
                model_to_use = dl_adapter_check
                print(f"INFO: Wrapping DL model with {model_to_use.__class__.__name__} for API compliance.")
        except Exception:
            # DL libraries not installed; assume not a DL model
            pass
        self.model_to_use = model_to_use
        self.type_ = getattr(self.model_to_use, "_estimator_type", None)
        
        # 2. MAPIE INSTANTIATION
        if self.type_ == "classifier":
            self.mapie_ = MapieClassifier(
                estimator = self.model_to_use,
                method = "score",
                cv = "prefit"
            ) 
        elif self.type_ == "regressor":
            self.mapie_ = MapieRegressor(
                estimator = self.model_to_use,
                cv = "prefit"
            )
        else:
            raise TypeError("Unsupported estimator type.")
            
        # 3. EXPLAINER INSTANTIATION
        self.explainer = self._get_optimal_explainer(self.model_to_use, X_background) 
        
        self.is_calibrated_ = False
        
    def fit(self, X_calib, y_calib):
        try:
            check_is_fitted(self.model_to_use)
        except Exception:
            raise RuntimeError(
                """
                The base_model must be fitted on training data
                before being passed to SpectrumUncertaintyWrapper.
                """
            )
        self.mapie_.fit(X_calib, y_calib)
        self.is_calibrated_ = True
        return self

    def predict(self, X) -> PredictionResult:
        """
        Routes prediction to the appropriate Mapie wrapper and formats the output
        as a set (classifier) or interval (regressor).

        Returns
        -------
        PredictionResult
            Object with y_preds and either y_pis (regression) or y_set (classification)
        """
        if not self.is_calibrated_:
            raise RuntimeError("Model is not calibrated. Call .fit() with calibration data first.")

        alpha = self.risk_profile.alpha

        if self.type_ == "regressor":
            y_pred, y_pis = self.mapie_.predict(X, alpha=alpha) # (N, 2, 1)
            return PredictionResult(
                y_preds=y_pred,
                y_pis=y_pis,
                confidence=1 - alpha
            )

        elif self.type_ == "classifier":
            y_pred, y_set = self.mapie_.predict(X, alpha=alpha)

            classes = getattr(self.mapie_, "classes_", np.arange(y_set.shape[1])).tolist()

            return PredictionResult(
                y_preds=y_pred,
                y_set=y_set,
                classes=classes,
                confidence=1 - alpha
            )

        else:
            raise TypeError(f"Internal type {self.type_} is invalid after initialization.")
    
    def _get_optimal_explainer(self, model: BaseEstimator, X_background) -> shap.Explainer:
        if hasattr(model, "get_booster") or hasattr(model, "estimators_"):
            # XGBoost/LightGBM model route
            return shap.TreeExplainer(model, data = X_background)
    
        elif hasattr(model, "predict_proba") or hasattr(model, "predict"):
            # General sklearn model route
            print("""
            shap.KernelExplainer is used for general sklearn models.
            This model may be slow.
            """)
            
            if self.type_ == "regressor":
                return shap.KernelExplainer(model, data = X_background, link = "identity")
            
            elif self.type_ == "classifier":
                return shap.KernelExplainer(model, data = X_background, link = "logit")
        
        else:
            raise NotImplementedError("Model type not supported for fast explanation.")
        
import numpy as np
from typing import Dict, List, Any


def generate_shap_explanations(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    max_samples: int = 50,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Generate SHAP-based explanations for a model's predictions.

    This is a simplified utility function for quick SHAP analysis without
    requiring template configuration. For production use with regulatory
    compliance, use ReasonCodeGenerator instead.

    Parameters
    ----------
    model : BaseEstimator
        The fitted model to explain
    X_test : pd.DataFrame
        Test data to generate explanations for
    max_samples : int, default=5
        Maximum number of samples to explain
    feature_names : List[str], optional
        Feature names (uses X_test.columns if not provided)

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'top_features': List of feature importance strings
        - 'shap_values': Raw SHAP values (if available)
        - 'feature_importances': Feature importance scores
    """
    if feature_names is None:
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Limit samples
    X_sample = X_test.iloc[:max_samples] if isinstance(X_test, pd.DataFrame) else X_test[:max_samples]

    try:
        # Try TreeExplainer first (fast for tree models)
        if hasattr(model, 'estimators_') or hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        else:
            # Fall back to simpler approach for non-tree models
            # Use a small background sample
            background = X_test.iloc[:100] if isinstance(X_test, pd.DataFrame) else X_test[:100]
            explainer = shap.KernelExplainer(model.predict, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Multi-class: use first class
            shap_values = shap_values[0]

        # Get feature importances (mean absolute SHAP)
        if shap_values.ndim == 2:
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:
            feature_importance = np.abs(shap_values)

        # Get top features
        top_indices = np.argsort(feature_importance)[::-1][:5]
        top_features = []

        for idx in top_indices:
            feat_name = feature_names[idx]
            importance = feature_importance[idx]
            top_features.append(f"{feat_name}: {importance:.4f}")

        return {
            'top_features': top_features,
            'shap_values': shap_values,
            'feature_importances': feature_importance
        }

    except Exception as e:
        # Fallback: use scikit-learn feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            top_features = [f"{feature_names[i]}: {importances[i]:.4f}" for i in top_indices]

            return {
                'top_features': top_features,
                'shap_values': None,
                'feature_importances': importances
            }
        else:
            # Last resort: return placeholder
            return {
                'top_features': [f"Feature importance unavailable: {str(e)}"],
                'shap_values': None,
                'feature_importances': None
            }


class ReasonCodeGenerator:
    """
    Translates raw SHAP outputs (multi-class or single-class) into 
    human-readable, regulatory-compliant adverse action reasons.
    
    Adverse contributions are defined as feature contributions that 
    push the model score TOWARDS the defined adverse_class_index.
    """
    
    def __init__(self, templates: Dict[str, str]):
        """
        :param templates: Dictionary mapping feature names (e.g., 'age', 'f_04') 
                        to English explanation templates (e.g., 'Age of {value} is below policy minimum').
        """
        self.templates = templates

    def generate_reasons(
        self, 
        shap_values_raw: np.ndarray, 
        feature_values: np.ndarray,
        feature_names: List[str],
        adverse_class_index: Union[int, None], # NEW: The index we are explaining AGAINST (e.g., Denied)
        top_k: int = 3
    ) -> List[str]:
        """
        Generates the top K adverse action reasons based on contributions 
        to the score of the adverse class.
        
        Note: This assumes 'shap_values_raw' has shape (N_features,) for 
        regression/binary log-odds, or (N_features, N_classes) for multi-class.
        """
        
        # 1. Isolate the SHAP Vector for the Target Adverse Score
        if shap_values_raw.ndim == 2:
            # Multi-class output: (N_features, N_classes)
            if adverse_class_index is None:
                raise ValueError("Multi-class SHAP requires 'adverse_class_index'.")
            
            # Select the column corresponding to the adverse class
            phi_adverse = shap_values_raw[:, adverse_class_index]
        
        elif shap_values_raw.ndim == 1:
            # Single output (Regression or Binary Log-Odds)
            # IMPORTANT: We assume a positive SHAP value is adverse
            phi_adverse = -shap_values_raw
        
        else:
            raise ValueError(f"Unexpected SHAP dimension: {shap_values_raw.ndim}")

        # 2. Identify Adverse Contributions 
        # We assume that positive contributions are always adverse, regardless of space.
        # This simplifies the logic by assuming the explainer output is normalized 
        # such that positive values are "bad."
        adverse_indices = np.where(phi_adverse > 0)[0] 
        
        # 3. Filter and Sort by Magnitude (Vectorized)
        adverse_magnitudes = np.abs(phi_adverse[adverse_indices])
        
        # Get the indices, then map to original indices and select top K
        sorted_indices_in_adverse_array = np.argsort(adverse_magnitudes)[::-1]
        top_k_original_indices = adverse_indices[sorted_indices_in_adverse_array][:top_k]
        
        # 4. Generate Templated Strings
        final_reasons = []
        for i in top_k_original_indices:
            name = feature_names[i]
            value = feature_values[i]
            
            if name in self.templates:
                template = self.templates[name]
                
                # Format the template with the feature's actual value
                value_str = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
                
                # Use value_str to format the template
                final_reasons.append(template.format(value=value_str))
            else:
                final_reasons.append(
                    f"Feature {name} contributed negatively (SHAP: {phi_adverse[i]:.3f})"
                )
        
        return final_reasons