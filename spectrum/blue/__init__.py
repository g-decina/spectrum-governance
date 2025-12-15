"""
spectrum.blue - Blue Team Defense Module

Defensive capabilities for AI governance including:
- Uncertainty quantification (conformal prediction)
- Time series forecasting with prediction intervals (EnbPI)
- Drift monitoring (PSI-based)
- Explainability (SHAP)
- Model hardening (inference-time wrappers)
- Trust and fairness metrics
"""

from spectrum.blue.explain import (
    SpectrumUncertaintyWrapper,
    PredictionResult,
    ReasonCodeGenerator,
    generate_shap_explanations
)
from spectrum.blue.monitor import DriftCheck, PSI_MONITOR_THRESHOLD, PSI_ALERT_THRESHOLD
from spectrum.blue.timeseries import (
    EnbPIRegressor,
    TimeSeriesPrediction,
    TimeSeriesValidator,
    AdaptiveResidualTracker
)
from spectrum.blue.harden import (
    # Configuration
    HardeningConfig,
    # Regression wrappers
    InputSanitizer,
    EnsembleProxy,
    OutputQuantizer,
    PredictionSmoother,
    RandomizedWrapper,
    # Classification wrappers
    ClassifierInputSanitizer,
    ConfidenceGate,
    # Retraining utilities
    AdversarialTrainer,
    # High-level API
    harden_model,
    get_recommended_wrapper,
)

__all__ = [
    # Uncertainty quantification
    'SpectrumUncertaintyWrapper',
    'PredictionResult',

    # Time series forecasting
    'EnbPIRegressor',
    'TimeSeriesPrediction',
    'TimeSeriesValidator',
    'AdaptiveResidualTracker',

    # Explainability
    'ReasonCodeGenerator',
    'generate_shap_explanations',

    # Monitoring
    'DriftCheck',
    'PSI_MONITOR_THRESHOLD',
    'PSI_ALERT_THRESHOLD',

    # Model hardening
    'HardeningConfig',
    'InputSanitizer',
    'EnsembleProxy',
    'OutputQuantizer',
    'PredictionSmoother',
    'RandomizedWrapper',
    'ClassifierInputSanitizer',
    'ConfidenceGate',
    'AdversarialTrainer',
    'harden_model',
    'get_recommended_wrapper',
]
