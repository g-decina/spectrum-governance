import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from spectrum.infra.types import RiskProfile, RiskLevel
from spectrum.blue.trust import SpectrumRegressor, SpectrumClassifier

# --------------------------
# 1. SpectrumRegressor
# --------------------------

def test_regressor_conformal_coverage():
    """
    Mathematical Proof:
    Does the model actually cover (1-alpha)% of the data?
    """
    # 1. Create synthetic linear data
    X, y = make_regression(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    
    # Split: Train (50%) / Calib (0.3%) / Test (0.2%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.5, random_state = 42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size = 0.4, random_state = 42)

    # 2. Train Base -> Wrap -> Calibrate
    base = LinearRegression().fit(X_train, y_train)
    
    # Alpha = 0.1 (90% Coverage)
    profile = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.1)
    
    spectrum_model = SpectrumRegressor(base_model=base, risk_profile=profile)
    spectrum_model.fit(X_calib, y_calib)
    
    # Predict on unseen test data
    results = spectrum_model.predict(X_test)
    
    # 3. Assert: Calculate empirical coverage
    # Check if y_true is between lower and upper
    in_bounds = (y_test >= results["lower_bound"]) & (y_test <= results["upper_bound"])
    coverage = np.mean(in_bounds)
    
    print(f"\nTarget Coverage: 0.90 | Actual Coverage: {coverage:.4f}")
    
    # Tolerance: It won't be exactly 0.90 due to sample size, but should be close (e.g. > 0.85)
    assert coverage >= 0.85
    assert coverage <= 0.98 # Shouldn't be 100% (that implies intervals are too wide)

def test_regressor_uncalibrated_error():
    """Ensure we block predictions if fit() wasn't called."""
    X, y = make_regression(n_samples=10, n_features=2)
    base = LinearRegression().fit(X, y)
    profile = RiskProfile(level=RiskLevel.LOW, alpha=0.2)
    
    model = SpectrumRegressor(base, profile)
    
    with pytest.raises(RuntimeError):
        model.predict(X)

# --------------------------
# 2. SpectrumClassifier
# --------------------------

def test_classifier_conformal_coverage():
    """
    Mathematical Proof:
    Does the model actually cover (1-alpha)% of the data?
    """
    # 1. Create synthetic linear data
    X, y = make_classification(n_samples=5_000, n_features=20, n_classes=4, n_informative=8, random_state=42)
    
    # Split: Train (50%) / Calib (0.3%) / Test (0.2%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.5, random_state = 42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size = 0.4, random_state = 42)

    # 2. Train Base -> Wrap -> Calibrate
    base = LogisticRegression().fit(X_train, y_train)
    
    # Alpha = 0.1 (90% Coverage)
    profile = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.1)
    
    spectrum_model = SpectrumClassifier(base_model=base, risk_profile=profile)
    spectrum_model.fit(X_calib, y_calib)
    
    # Predict on unseen test data
    results = spectrum_model.predict(X_test)
    
    # 3. Assert: Calculate empirical coverage
    # Check if y_true is in the prediction set
    true_class_indices = np.searchsorted(results["classes"], y_test)
    n_samples = len(y_test)
    
    in_bounds = results["prediction_set"][np.arange(n_samples), true_class_indices]
    coverage = np.mean(in_bounds)
    
    print(f"\nTarget Coverage: 0.90 | Actual Coverage: {coverage:.4f}")
    
    # Tolerance: It won't be exactly 0.90 due to sample size, but should be close (e.g. > 0.85)
    assert coverage >= 0.85

def test_classifier_uncalibrated_error():
    """Ensure we block predictions if fit() wasn't called."""
    X, y = make_classification(n_samples=10, n_features=5)
    base = LogisticRegression().fit(X, y)
    profile = RiskProfile(level=RiskLevel.LOW, alpha=0.2)
    
    model = SpectrumClassifier(base, profile)
    
    with pytest.raises(RuntimeError):
        model.predict(X)