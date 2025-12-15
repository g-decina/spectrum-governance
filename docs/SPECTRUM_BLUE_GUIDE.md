# Spectrum Blue Guide

## Blue Team Defense Module

**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

`spectrum.blue` provides defensive capabilities for ML governance, transforming opaque model predictions into transparent, legally defensible outputs with statistical guarantees. The module addresses three critical regulatory requirements:

1. **Uncertainty Quantification** - Finite-sample coverage guarantees via Conformal Prediction
2. **Explainability** - SHAP-based feature attribution with CFPB-compliant adverse action reasons
3. **Drift Monitoring** - Population Stability Index (PSI) for detecting distribution shifts

```
spectrum.blue/
├── trust.py       # Conformal Prediction wrappers (SpectrumRegressor, SpectrumClassifier)
├── explain.py     # SHAP explanations & adverse action reason generation
├── monitor.py     # PSI-based drift detection (DriftCheck)
├── timeseries.py  # EnbPI time series forecasting with prediction intervals
└── thresholds.py  # Regulatory threshold definitions
```

---

## Quick Start

### Uncertainty Quantification

```python
from spectrum.blue import SpectrumUncertaintyWrapper
from spectrum.infra.types import RiskProfile, RiskLevel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Train your model
model = RandomForestClassifier(n_estimators=100)
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)

# 2. Define risk profile (HIGH = 95% confidence)
risk_profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)

# 3. Wrap with uncertainty quantification
trusted_model = SpectrumUncertaintyWrapper(
    base_model=model,
    risk_profile=risk_profile,
    X_background=X_train[:100]  # Required for KernelExplainer fallback
)

# 4. Calibrate on hold-out data
trusted_model.fit(X_calib, y_calib)

# 5. Predict with guaranteed coverage
result = trusted_model.predict(X_test)
print(f"Predictions: {result.y_preds}")
print(f"Prediction Sets: {result.y_set}")  # For classifiers
print(f"Confidence: {result.confidence:.1%}")
```

### Drift Detection

```python
from spectrum.blue import DriftCheck
import pandas as pd

# Reference data (training distribution)
reference_data = pd.read_csv("training_data.csv")

# Current production batch
current_data = pd.read_csv("production_batch.csv")

# Run drift check
drift_result = DriftCheck(reference_data, current_data)

print(f"Drift Detected: {drift_result['drift_detected']}")
print(f"Max PSI: {drift_result['max_psi']:.3f}")
print(f"Status: {drift_result['status']}")

if drift_result['alert_required']:
    print(f"ALERT: Features with drift: {drift_result['drifted_features']}")
```

### Explainability

```python
from spectrum.blue import ReasonCodeGenerator, generate_shap_explanations

# Quick SHAP analysis
explanations = generate_shap_explanations(model, X_test, max_samples=50)
print("Top contributing features:", explanations['top_features'])

# Regulatory-compliant adverse action reasons
templates = {
    "credit_score": "Credit score of {value} is below approval threshold",
    "income": "Annual income of ${value} does not meet minimum requirement",
    "dti": "Debt-to-income ratio of {value}% exceeds maximum allowed"
}

reason_gen = ReasonCodeGenerator(templates=templates)
reasons = reason_gen.generate_reasons(
    shap_values_raw=shap_values,
    feature_values=X_test[0],
    feature_names=feature_names,
    adverse_class_index=1,  # Class 1 = denial
    top_k=3
)

# Output: ["Credit score of 580 is below approval threshold", ...]
```

---

## Mathematical Foundations

### 1. Conformal Prediction

Conformal Prediction provides **distribution-free** coverage guarantees on finite samples.

#### Core Guarantee

For any prediction set $\hat{C}(X)$ constructed via conformal prediction:

$$P(Y \in \hat{C}(X)) \geq 1 - \alpha$$

Where:
- $Y$ is the true outcome
- $\hat{C}(X)$ is the prediction set (classification) or interval (regression)
- $\alpha$ is the user-specified miscoverage rate
- $1 - \alpha$ is the **confidence level**

#### Non-Conformity Score

The method works by computing a **non-conformity score** that measures how unusual a prediction is:

$$s(x, y) = |y - \hat{f}(x)|$$

For regression, or for classification:

$$s(x, y) = 1 - \hat{p}_y(x)$$

where $\hat{p}_y(x)$ is the predicted probability for class $y$.

#### Coverage Calibration

Given calibration data $(X_1, Y_1), \ldots, (X_n, Y_n)$:

1. Compute non-conformity scores: $s_i = s(X_i, Y_i)$
2. Find quantile: $\hat{q} = \text{Quantile}_{(1-\alpha)(1 + 1/n)}(\{s_1, \ldots, s_n\})$
3. Prediction set: $\hat{C}(X_{new}) = \{y : s(X_{new}, y) \leq \hat{q}\}$

#### Key Property: Exchangeability

The guarantee holds if calibration data and test data are **exchangeable** (i.i.d. is sufficient but not necessary). This makes conformal prediction robust to model misspecification.

### 2. Population Stability Index (PSI)

PSI measures distribution shift between reference and current data:

$$\text{PSI} = \sum_{i=1}^{n} (p_i^{curr} - p_i^{ref}) \times \ln\left(\frac{p_i^{curr}}{p_i^{ref}}\right)$$

Where:
- $p_i^{ref}$ = proportion of reference data in bin $i$
- $p_i^{curr}$ = proportion of current data in bin $i$
- $n$ = number of bins (default: 10 quantile-based bins)

#### Industry Thresholds

| PSI Range | Interpretation | Action |
|-----------|----------------|--------|
| PSI < 0.10 | No significant change | Continue monitoring |
| 0.10 ≤ PSI < 0.25 | Moderate shift | Investigate cause |
| PSI ≥ 0.25 | Significant shift | Model retraining required |

#### Mathematical Properties

PSI is related to **Kullback-Leibler divergence** but is symmetric:

$$\text{PSI} = D_{KL}(P_{curr} \| P_{ref}) + D_{KL}(P_{ref} \| P_{curr})$$

This symmetry makes it more interpretable than one-sided divergence measures.

### 3. SHAP (SHapley Additive exPlanations)

SHAP values decompose predictions into feature contributions:

$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i(x)$$

Where:
- $f(x)$ is the model prediction
- $\phi_0$ is the expected prediction (base value)
- $\phi_i(x)$ is the contribution of feature $i$

#### Shapley Value Formula

Each SHAP value is computed as:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [f(S \cup \{i\}) - f(S)]$$

Where:
- $N$ is the set of all features
- $S$ is a subset of features
- $f(S)$ is the model prediction using only features in $S$

#### Adverse Score Space

For adverse action notices, we need contributions toward the **adverse outcome**:

For binary classification with log-odds link:
$$\phi_{adverse} = -\phi$$ (positive = pushes toward denial)

For multi-class:
$$\phi_{adverse} = \phi[:, \text{adverse\_class\_index}]$$

### 4. EnbPI (Ensemble batch Prediction Intervals)

EnbPI extends conformal prediction to **time series** where exchangeability breaks down.

#### Algorithm

1. **Bootstrap Ensemble**: Train $B$ models on bootstrap samples of training data
2. **Aggregated Prediction**: $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)$
3. **Adaptive Residuals**: Track residuals $r_t = y_t - \hat{y}_t$ in rolling window
4. **Interval Construction**:
   $$[\hat{y} - q_{1-\alpha/2}(|r|), \hat{y} + q_{1-\alpha/2}(|r|)]$$

#### Coverage Guarantee (Asymptotic)

$$\lim_{t \to \infty} \frac{1}{t} \sum_{i=1}^{t} \mathbb{1}(Y_i \in C(X_i)) \geq 1 - \alpha$$

---

## API Reference

### SpectrumUncertaintyWrapper

Unified wrapper for classifiers and regressors with conformal prediction.

```python
class SpectrumUncertaintyWrapper:
    def __init__(
        self,
        base_model: BaseEstimator,     # Pre-fitted sklearn model
        risk_profile: RiskProfile,      # Governance risk configuration
        X_background: np.ndarray = None # Required for KernelExplainer
    )

    def fit(self, X_calib, y_calib) -> 'SpectrumUncertaintyWrapper':
        """Calibrate uncertainty using hold-out data."""

    def predict(self, X) -> PredictionResult:
        """Return predictions with uncertainty bounds."""
```

**PredictionResult Fields:**
- `y_preds`: Point predictions
- `y_pis`: Prediction intervals (regression) with shape `(n_samples, 2, 1)`
- `y_set`: Prediction sets (classification) with shape `(n_samples, n_classes)`
- `confidence`: Coverage guarantee (1 - alpha)

### SpectrumRegressor / SpectrumClassifier

Specialized wrappers for regression and classification.

```python
# Regression
regressor = SpectrumRegressor(base_model, risk_profile)
regressor.fit(X_calib, y_calib)
result = regressor.predict(X_test)
# result = {"prediction": y_pred, "lower_bound": lb, "upper_bound": ub, "confidence": 0.95}

# Classification
classifier = SpectrumClassifier(base_model, risk_profile)
classifier.fit(X_calib, y_calib)
result = classifier.predict(X_test)
# result = {"prediction": y_pred, "prediction_set": y_set, "classes": [...], "confidence": 0.95}
```

### DriftCheck

PSI-based distribution shift detection.

```python
def DriftCheck(
    reference_data: pd.DataFrame,   # Training/baseline distribution
    current_data: pd.DataFrame,     # Production data batch
    feature_names: List[str] = None, # Features to monitor (all if None)
    n_bins: int = 10                 # Bins for numerical features
) -> Dict[str, Any]:
    """
    Returns:
        drift_detected: bool
        n_drifted_features: int
        feature_drift_scores: Dict[str, float]  # PSI per feature
        drifted_features: List[str]              # Features exceeding threshold
        max_psi: float
        status: str                              # Human-readable summary
        alert_required: bool                     # True if max_psi >= 0.25
    """
```

### ReasonCodeGenerator

Translates SHAP values to regulatory-compliant adverse action reasons.

```python
class ReasonCodeGenerator:
    def __init__(self, templates: Dict[str, str]):
        """
        templates: Mapping from feature names to explanation templates.
                   Use {value} placeholder for feature value.
        """

    def generate_reasons(
        self,
        shap_values_raw: np.ndarray,    # Shape: (n_features,) or (n_features, n_classes)
        feature_values: np.ndarray,     # Actual feature values for this sample
        feature_names: List[str],       # Feature names
        adverse_class_index: int,       # Which class is "adverse" (e.g., denial)
        top_k: int = 3                  # Number of reasons to return
    ) -> List[str]:
        """Returns templated adverse action reasons."""
```

### EnbPIRegressor

Time series regression with prediction intervals.

```python
class EnbPIRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        base_forecaster: BaseEstimator,  # sklearn regressor
        n_bootstraps: int = 50,           # Bootstrap ensemble size
        window_size: int = 100,           # Residual tracking window
        risk_profile: RiskProfile = None, # Optional governance config
        alpha: float = 0.1                # Miscoverage rate
    )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnbPIRegressor':
        """Fit bootstrap ensemble on training data."""

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'EnbPIRegressor':
        """Online update with new observations."""

    def predict(
        self,
        X: np.ndarray,
        return_intervals: bool = False
    ) -> Union[np.ndarray, TimeSeriesPrediction]:
        """Predict with optional prediction intervals."""

    def evaluate_coverage(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate empirical coverage on test data."""
```

---

## Regulatory Compliance Mapping

### EU AI Act Article 15

**Requirement:** High-risk AI systems must achieve appropriate levels of accuracy.

**Spectrum Blue Implementation:**
- Conformal prediction provides **finite-sample accuracy guarantees**
- Coverage metrics are logged for Technical Documentation (Annex IV)
- Prediction sets make uncertainty explicit and auditable

### CFPB Adverse Action Requirements (Regulation B)

**Requirement:** Creditors must provide specific reasons for adverse actions.

**Spectrum Blue Implementation:**
- `ReasonCodeGenerator` maps SHAP attributions to templated reasons
- Reasons are sorted by contribution magnitude (most impactful first)
- Output format suitable for Adverse Action Notices

### OCC SR 11-7 (Model Risk Management)

**Requirement:** Models must be monitored for performance degradation.

**Spectrum Blue Implementation:**
- `DriftCheck` detects distribution shifts in production data
- PSI thresholds align with industry standards
- Alerts trigger before model accuracy degrades

---

## Risk Profile Configuration

| Risk Level | Alpha | Confidence | Use Case | PII Policy |
|------------|-------|------------|----------|------------|
| LOW | 0.20 | 80% | Marketing segmentation | DETECT_ONLY |
| MEDIUM | 0.10 | 90% | Fraud alerts | DETECT_ONLY |
| HIGH | 0.05 | 95% | Credit decisions | PSEUDONYMIZE |
| CRITICAL | 0.01 | 99% | Autonomous systems | REDACT |

```python
from spectrum.infra.types import RiskProfile, RiskLevel

# High-risk credit model
profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)

# Critical autonomous system
profile = RiskProfile(level=RiskLevel.CRITICAL, alpha=0.01)
```

---

## Best Practices

### Calibration Data Selection

1. **Never use training data** - Coverage guarantees are void
2. **Use held-out data** - Temporally or randomly split
3. **Match production distribution** - Calibration should reflect deployment
4. **Size matters** - Larger calibration sets give tighter intervals

### Drift Monitoring Strategy

1. **Establish baseline** - Use training data as reference
2. **Batch processing** - Check drift on production batches, not single samples
3. **Feature-level analysis** - Investigate which features are drifting
4. **Root cause analysis** - Drift may indicate data pipeline issues

### Explainability Trade-offs

| Explainer | Speed | Model Support | Accuracy |
|-----------|-------|---------------|----------|
| TreeExplainer | Fast | Tree ensembles only | Exact |
| KernelExplainer | Slow | Any model | Approximate |

Spectrum Blue automatically selects the optimal explainer based on model type.

---

## Example: Complete Credit Scoring Workflow

```python
from spectrum.blue import (
    SpectrumUncertaintyWrapper,
    DriftCheck,
    ReasonCodeGenerator,
    generate_shap_explanations
)
from spectrum.infra.types import RiskProfile, RiskLevel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load and split data
df = pd.read_csv("credit_applications.csv")
X = df.drop("approved", axis=1)
y = df["approved"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# 2. Train model
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. Configure governance
risk_profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)

# 4. Add uncertainty quantification
trusted_model = SpectrumUncertaintyWrapper(
    base_model=model,
    risk_profile=risk_profile,
    X_background=X_train[:100]
)
trusted_model.fit(X_calib, y_calib)

# 5. Production prediction with confidence
result = trusted_model.predict(X_test)
print(f"Empirical coverage: {(result.y_preds == y_test).mean():.1%}")

# 6. Monitor for drift
reference_df = pd.DataFrame(X_train, columns=X.columns)
current_df = pd.DataFrame(X_test, columns=X.columns)
drift_result = DriftCheck(reference_df, current_df)

if drift_result['alert_required']:
    print(f"ALERT: Drift detected in {drift_result['drifted_features']}")

# 7. Generate adverse action reasons for denied applications
denied_mask = result.y_preds == 0
if denied_mask.any():
    explanations = generate_shap_explanations(model, X_test[denied_mask])
    print("Top factors for denials:", explanations['top_features'])
```

---

## Troubleshooting

### "Model is not calibrated"

```python
# ERROR: RuntimeError: Model is not calibrated. Call .fit() with calibration data first.
# FIX: Call fit() before predict()
trusted_model.fit(X_calib, y_calib)
result = trusted_model.predict(X_test)
```

### "Base model must be fitted"

```python
# ERROR: RuntimeError: The base_model must be fitted on training data
# FIX: Train the model before wrapping
model.fit(X_train, y_train)  # Required!
trusted_model = SpectrumUncertaintyWrapper(model, risk_profile)
```

### PSI Calculation Issues

```python
# ERROR: Schema mismatch between reference and current data
# FIX: Ensure both DataFrames have identical columns
assert set(reference_data.columns) == set(current_data.columns)
```

### Slow SHAP Calculations

```python
# Issue: KernelExplainer is slow for non-tree models
# Solution: Use a smaller background sample
trusted_model = SpectrumUncertaintyWrapper(
    base_model=model,
    risk_profile=risk_profile,
    X_background=X_train[:50]  # Smaller = faster
)
```

---

## References

1. **Conformal Prediction**: Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.

2. **MAPIE Library**: Taquet, M., et al. (2022). MAPIE: Model Agnostic Prediction Interval Estimator.

3. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.

4. **PSI**: Siddiqi, N. (2012). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.

5. **EnbPI**: Xu, C., & Xie, Y. (2021). Conformal Prediction Interval for Dynamic Time-Series.

6. **CFPB Regulation B**: 12 CFR 1002.9 - Adverse Action Notice Requirements.

7. **OCC SR 11-7**: Guidance on Model Risk Management.

---

*Last updated: 2024*
