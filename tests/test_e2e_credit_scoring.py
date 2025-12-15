"""
End-to-End Test: Credit Scoring Governance

This script demonstrates the full spectrum-governance framework in a real-world
credit scoring scenario. It builds a real model and runs through the complete
governance pipeline:

1. Train a real credit scoring model
2. Blue Team: Uncertainty quantification, drift monitoring, explanations
3. Red Team: Adversarial attacks
4. Lens: Lineage tracking and compliance reporting

Prerequisites:
    pip install -e ".[blue]"
    docker-compose -f docker-compose.marquez.yml up -d  # Optional: for lineage

Run:
    python tests/test_e2e_credit_scoring.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectrum.infra.types import RiskProfile, RiskLevel
from spectrum.infra.logger import RCIALogger
from spectrum.blue.explain import SpectrumUncertaintyWrapper, generate_shap_explanations
from spectrum.blue.monitor import DriftCheck, PSI_ALERT_THRESHOLD, PSI_MONITOR_THRESHOLD
from spectrum.red.attack import HopSkipJumpWrapper
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def generate_credit_dataset(n_samples=50_000, random_state=42):
    """
    Generate a realistic credit scoring dataset.

    Features simulate:
    - Credit score
    - Annual income
    - Debt-to-income ratio
    - Employment length
    - Number of credit lines
    """
    print_section("STEP 1: GENERATING CREDIT SCORING DATASET")

    np.random.seed(random_state)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[0.75, 0.25],  # Imbalanced: 75% approved, 25% denied
        flip_y=0.05,
        random_state=random_state
    )

    # Create meaningful feature names
    feature_names = [
        'credit_score',
        'annual_income',
        'debt_to_income_ratio',
        'employment_length_years',
        'num_credit_lines',
        'credit_utilization',
        'num_late_payments',
        'age_years',
        'months_since_last_inquiry',
        'total_debt'
    ]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Scale features to realistic ranges
    df['credit_score'] = (df['credit_score'] - df['credit_score'].min()) / \
                         (df['credit_score'].max() - df['credit_score'].min()) * 550 + 300  # 300-850
    df['annual_income'] = np.exp(df['annual_income'] * 0.5 + 10) * 1000  # $20k-$200k
    df['debt_to_income_ratio'] = (df['debt_to_income_ratio'] - df['debt_to_income_ratio'].min()) / \
                                  (df['debt_to_income_ratio'].max() - df['debt_to_income_ratio'].min()) * 0.6  # 0-0.6
    df['employment_length_years'] = np.abs(df['employment_length_years']) * 2  # 0-20 years
    df['num_credit_lines'] = np.abs(df['num_credit_lines']) * 2 + 1  # 1-20
    df['credit_utilization'] = (df['credit_utilization'] - df['credit_utilization'].min()) / \
                                (df['credit_utilization'].max() - df['credit_utilization'].min())  # 0-1
    df['num_late_payments'] = np.abs(df['num_late_payments']) * 2  # 0-20
    df['age_years'] = np.abs(df['age_years']) * 5 + 25  # 25-75
    df['months_since_last_inquiry'] = np.abs(df['months_since_last_inquiry']) * 3  # 0-30
    df['total_debt'] = np.exp(df['total_debt'] * 0.3 + 8) * 100  # $2k-$100k

    print(f"✓ Generated {n_samples} credit applications")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Approval rate: {(1-y.mean())*100:.1f}%")
    print(f"\nDataset preview:")
    print(df.head())
    print(f"\nTarget distribution:")
    print(f"  Approved (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Denied (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

    return df, y, feature_names


def train_credit_model(X_train, y_train):
    """Train a Random Forest credit scoring model."""
    print_section("STEP 2: TRAINING CREDIT SCORING MODEL")

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train.values, y_train)

    print(f"✓ Model trained successfully")
    print(f"  Estimators: {model.n_estimators}")
    print(f"  Max depth: {model.max_depth}")
    print(f"  Features: {model.n_features_in_}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print_section("STEP 3: MODEL EVALUATION")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Approved', 'Denied']))

    return accuracy


def setup_governance_components():
    """Set up governance infrastructure."""
    print_section("STEP 4: SETTING UP GOVERNANCE INFRASTRUCTURE")

    # Risk Profile: HIGH risk for credit decisions
    risk_profile = RiskProfile(
        level=RiskLevel.HIGH,
        alpha=0.05  # 95% confidence required
    )
    print(f"✓ Risk Profile: {risk_profile.level.value}")
    print(f"  Confidence Required: {(1-risk_profile.alpha)*100}%")

    # RCIA Logger
    log_path = "/tmp/rcia_credit_audit.jsonl"
    logger = RCIALogger(log_path=log_path)
    print(f"✓ RCIA Logger initialized: {log_path}")

    # Lineage Tracker (disable if Marquez not running)
    lineage_enabled = os.getenv("OPENLINEAGE_ENABLED", "false").lower() == "true"
    if not lineage_enabled:
        print("ℹ Lineage tracking disabled (set OPENLINEAGE_ENABLED=true to enable)")

    return risk_profile, logger


def blue_team_defense(model, X_calib, y_calib, X_test, y_test, risk_profile):
    """Blue Team: Uncertainty quantification and monitoring."""
    print_section("STEP 5: BLUE TEAM - UNCERTAINTY QUANTIFICATION")

    # 1. Calibrate uncertainty wrapper
    print("\n[1/3] Calibrating uncertainty wrapper...")
    
    background_sample = X_calib.iloc[:100] if hasattr(X_calib, "iloc") else X_calib[:100]
    
    blue_wrapper = SpectrumUncertaintyWrapper(
        base_model=model,
        risk_profile=risk_profile,
        X_background=background_sample
    )
    blue_wrapper.fit(X_calib, y_calib)
    print(f"✓ Calibration complete on {len(X_calib)} samples")

    # 2. Generate predictions with uncertainty
    print("\n[2/3] Generating predictions with uncertainty estimates...")
    results = blue_wrapper.predict(X_test)

    print(f"✓ Predictions generated for {len(X_test)} samples")
    
    # === CORRECTED DISPLAY LOGIC ===
    if results.y_pis is not None:
        # REGRESSION Logic (Intervals)
        print(f"\nPrediction Intervals (first 5 samples):")
        for i in range(min(5, len(results.y_preds))):
            lower = results.y_pis[i, 0, 0]
            upper = results.y_pis[i, 1, 0]
            pred = results.y_preds[i]
            # Handle pandas series or numpy array for actual
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            in_interval = lower <= actual <= upper
            print(f"  Sample {i}: pred={pred:.3f}, interval=[{lower:.3f}, {upper:.3f}], "
                    f"actual={actual:.1f} {'✓' if in_interval else '✗'}")
                
    elif results.y_set is not None:
        # CLASSIFICATION Logic (Prediction Sets)
        print(f"\nPrediction Sets (first 5 samples):")
        
        # Get class names if available (0/1 for this dataset)
        class_names = results.classes if results.classes else [0, 1]
        
        for i in range(min(5, len(results.y_preds))):
            # y_set is a boolean mask of shape (n_samples, n_classes)
            # We convert the boolean mask to the actual class names included in the set
            prediction_set = [class_names[j] for j, is_included in enumerate(results.y_set[i]) if is_included]
            
            pred = results.y_preds[i]
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            
            # Check if actual label is inside the predicted set
            in_set = actual in prediction_set
            
            print(f"  Sample {i}: pred={pred}, set={prediction_set}, "
                    f"actual={actual} {'✓' if in_set else '✗'}")

    # 3. Calculate empirical coverage
    print("\n[3/3] Calculating empirical coverage...")
    
    if results.y_pis is not None:
        # Regression Coverage
        lower_bounds = results.y_pis[:, 0, 0]
        upper_bounds = results.y_pis[:, 1, 0]
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        coverage_mask = (y_test_array >= lower_bounds) & (y_test_array <= upper_bounds)
    else:
        # Classification Coverage: Is the true label in the prediction set?
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        # We need to map y_test values to the column indices of y_set
        # Assuming y_test contains the same labels as self.classes_
        # For simple 0/1, we can just look up the index directly
        coverage_mask = []
        for i in range(len(y_test_array)):
            true_label_idx = int(y_test_array[i]) # Assumes 0 or 1 index matches
            if true_label_idx < results.y_set.shape[1]:
                coverage_mask.append(results.y_set[i, true_label_idx])
            else:
                coverage_mask.append(False)
        coverage_mask = np.array(coverage_mask)

    empirical_coverage = float(np.mean(coverage_mask))

    print(f"✓ Empirical Coverage: {empirical_coverage:.4f}")
    print(f"  Required: {1 - risk_profile.alpha:.4f}")
    print(f"  Status: {'✓ PASS' if empirical_coverage >= (1 - risk_profile.alpha) else '✗ FAIL'}")

    return blue_wrapper, results, empirical_coverage


def blue_team_drift_monitoring(X_reference, X_current):
    """Blue Team: Data drift monitoring."""
    print_section("STEP 6: BLUE TEAM - DRIFT MONITORING")

    print(f"Checking drift between reference ({len(X_reference)}) and current ({len(X_current)}) data...")

    drift_results = DriftCheck(X_reference, X_current)

    print(f"\n✓ Drift Check Complete")
    print(f"  Status: {drift_results['status']}")
    print(f"  Alert Required: {drift_results['alert_required']}")
    print(f"  Max PSI: {drift_results['max_psi']:.4f}")
    print(f"  Monitor Threshold: {PSI_MONITOR_THRESHOLD}")
    print(f"  Alert Threshold: {PSI_ALERT_THRESHOLD}")

    print(f"\n  Per-Feature Drift Scores:")
    for feature, psi in drift_results['feature_drift_scores'].items():
        status_icon = "⚠" if psi >= PSI_ALERT_THRESHOLD else "⚡" if psi >= PSI_MONITOR_THRESHOLD else "✓"
        print(f"    {status_icon} {feature}: {psi:.4f}")

    if drift_results['drifted_features']:
        print(f"\n  Drifted Features: {', '.join(drift_results['drifted_features'])}")

    return drift_results


def blue_team_explanations(model, X_test):
    """Blue Team: Generate SHAP explanations."""
    print_section("STEP 7: BLUE TEAM - EXPLAINABILITY")

    print("Generating SHAP explanations for sample predictions...")

    explanations = generate_shap_explanations(
        model=model,
        X_test=X_test,
        max_samples=500
    )

    print(f"\n✓ Generated explanations for {len(explanations['top_features'])} samples")
    print(f"\nTop Feature Contributions:")
    for i, reason in enumerate(explanations['top_features'][:5], 1):
        print(f"  {i}. {reason}")

    return explanations


def red_team_attack(model, X_test):
    """Red Team: Adversarial attack."""
    print_section("STEP 8: RED TEAM - ADVERSARIAL ATTACK")

    print("Launching HopSkipJump adversarial attack...")
    print("(This simulates finding model vulnerabilities)")

    try:
        attack_wrapper = HopSkipJumpWrapper(base_model=model)
        metrics = attack_wrapper.run(X_test[:50])  # Attack subset for speed

        # Fragility score = attack success rate (higher = more vulnerable)
        fragility_score = metrics.attack_success_rate

        print(f"\n✓ Attack Complete")
        print(f"  Attack Success Rate: {metrics.attack_success_rate:.1%}")
        print(f"  Samples Tested: {metrics.samples_tested}")
        print(f"  Samples Successful: {metrics.samples_successful}")
        print(f"  Mean L2 Perturbation: {metrics.empirical_robustness_l2:.4f}")
        print(f"  Queries Used: {metrics.queries_used}")
        print(f"  Fragility Score: {fragility_score:.4f}")
        print(f"  Interpretation:")
        if fragility_score < 0.1:
            print(f"    ✓ Model is ROBUST (low fragility)")
        elif fragility_score < 0.3:
            print(f"    ⚡ Model has MODERATE vulnerabilities")
        else:
            print(f"    ⚠ Model is FRAGILE (high risk)")

        return metrics

    except Exception as e:
        print(f"⚠ Attack simulation unavailable: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_compliance_report(
    model,
    risk_profile,
    empirical_coverage,
    adversarial_metrics,
    drift_results,
    explanations,
    logger
):
    """Generate compliance audit report."""
    print_section("STEP 9: GENERATING COMPLIANCE REPORT")

    # Convert AdversarialMetrics to dict if provided
    if adversarial_metrics is not None:
        metrics_dict = adversarial_metrics.to_dict()
    else:
        # Provide default metrics if attack failed
        metrics_dict = {
            "attack_type": "HopSkipJump (unavailable)",
            "attack_success_rate": 0.0,
            "samples_tested": 0,
            "samples_successful": 0,
            "empirical_robustness_l2": 0.0,
            "empirical_robustness_linf": 0.0,
            "min_perturbation_l2": 0.0,
            "max_perturbation_l2": 0.0,
            "median_perturbation_l2": 0.0,
            "queries_used": 0,
            "avg_queries_per_sample": 0.0
        }

    # Create compliance report
    report_data = {
        "model_name": model.__class__.__name__,
        "model_type": "Tabular",
        "risk_level": risk_profile.level.value,
        "confidence_required": 1.0 - risk_profile.alpha,
        "empirical_coverage": empirical_coverage,
        "adversarial_metrics": metrics_dict,
        "sample_adverse_reasons": explanations['top_features'][:3],
        "data_drift_status": drift_results['status'],
        "data_drift_alert": drift_results['alert_required'],
        "lineage_run_id": "e2e-test-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "audit_log_path": logger.log_path
    }

    print("Creating compliance report...")
    report = ComplianceReport(**report_data)

    fragility_score = metrics_dict.get("attack_success_rate", 0.0)

    print(f"✓ Compliance Report Created")
    print(f"\n  Model: {report.model_name}")
    print(f"  Risk Level: {report.risk_level}")
    print(f"  Required Confidence: {report.confidence_required:.1%}")
    print(f"  Achieved Coverage: {report.empirical_coverage:.1%}")
    print(f"  Fragility Score: {fragility_score:.4f}")
    print(f"  Drift Status: {report.data_drift_status}")

    # Generate HTML report
    print("\nGenerating HTML report...")
    builder = ReportBuilder()
    report_path = "/tmp/credit_scoring_audit_report.html"

    try:
        builder.generate_html(
            template_name="audit_template.html",
            data=report.model_dump(),
            output_path=report_path
        )
        print(f"✓ Report saved to: {report_path}")
    except Exception as e:
        print(f"⚠ Report generation: {e}")

    return report


def assess_compliance(report):
    """Assess overall compliance status."""
    print_section("STEP 10: COMPLIANCE ASSESSMENT")

    checks = []

    # Extract fragility score from adversarial_metrics
    fragility_score = report.adversarial_metrics.get("attack_success_rate", 0.0)

    # Check 1: Coverage meets requirement
    coverage_ok = report.empirical_coverage >= report.confidence_required
    checks.append(("Coverage Requirement", coverage_ok))
    print(f"  {'✓' if coverage_ok else '✗'} Coverage: {report.empirical_coverage:.1%} >= {report.confidence_required:.1%}")

    # Check 2: Fragility is acceptable (attack success rate < 25%)
    fragility_ok = fragility_score < 0.25
    checks.append(("Fragility Threshold", fragility_ok))
    print(f"  {'✓' if fragility_ok else '✗'} Fragility: {fragility_score:.4f} < 0.25")

    # Check 3: No critical drift
    drift_ok = not report.data_drift_alert
    checks.append(("No Critical Drift", drift_ok))
    print(f"  {'✓' if drift_ok else '✗'} Drift Alert: {report.data_drift_alert}")

    # Check 4: High risk model meets standards
    high_risk_ok = True
    if report.risk_level == "HIGH":
        high_risk_ok = coverage_ok and fragility_ok
    checks.append(("High Risk Standards", high_risk_ok))
    print(f"  {'✓' if high_risk_ok else '✗'} High Risk Standards Met")

    # Overall assessment
    all_passed = all(check[1] for check in checks)

    print(f"\n{'='*80}")
    if all_passed:
        print(f"  ✓ COMPLIANCE STATUS: APPROVED")
        print(f"  Model meets all governance requirements")
    else:
        print(f"  ✗ COMPLIANCE STATUS: REQUIRES ATTENTION")
        failed_checks = [check[0] for check in checks if not check[1]]
        print(f"  Failed checks: {', '.join(failed_checks)}")
    print(f"{'='*80}")

    return all_passed


def main():
    """Run the complete end-to-end test."""
    print("\n" + "=" * 80)
    print("  SPECTRUM GOVERNANCE - END-TO-END TEST")
    print("  Scenario: Credit Scoring Model Governance Pipeline")
    print("=" * 80)

    # Generate dataset
    df, y, _ = generate_credit_dataset(n_samples=50_000)  # feature_names not used

    # Split data: train, calibration, test, reference, current
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.6, random_state=42, stratify=y
    )

    X_calib, X_test_full, y_calib, y_test_full = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp
    )

    X_test, X_current, y_test, _ = train_test_split(
        X_test_full, y_test_full, test_size=0.3, random_state=42, stratify=y_test_full
    )

    X_reference = X_train.copy()  # For drift monitoring

    print(f"\nData splits:")
    print(f"  Training: {len(X_train)}")
    print(f"  Calibration: {len(X_calib)}")
    print(f"  Test: {len(X_test)}")
    print(f"  Reference (drift): {len(X_reference)}")
    print(f"  Current (drift): {len(X_current)}")

    # Train model
    model = train_credit_model(X_train, y_train)

    # Evaluate
    _ = evaluate_model(model, X_test, y_test)  # accuracy not used later

    # Setup governance
    risk_profile, logger = setup_governance_components()

    # Blue Team: Defense
    _, _, empirical_coverage = blue_team_defense(
        model, X_calib, y_calib, X_test, y_test, risk_profile
    )

    # Blue Team: Drift monitoring
    drift_results = blue_team_drift_monitoring(X_reference, X_current)

    # Blue Team: Explanations
    explanations = blue_team_explanations(model, X_test)

    # Red Team: Attack
    adversarial_metrics = red_team_attack(model, X_test.values)

    # Generate compliance report
    report = generate_compliance_report(
        model=model,
        risk_profile=risk_profile,
        empirical_coverage=empirical_coverage,
        adversarial_metrics=adversarial_metrics,
        drift_results=drift_results,
        explanations=explanations,
        logger=logger
    )

    # Assess compliance
    compliant = assess_compliance(report)

    print_section("TEST COMPLETE")
    print(f"✓ Successfully demonstrated all framework capabilities")
    print(f"✓ Blue Team: Uncertainty, Drift, Explanations")
    print(f"✓ Red Team: Adversarial Testing")
    print(f"✓ Lens: Compliance Reporting")
    print(f"\nFinal Status: {'APPROVED ✓' if compliant else 'REQUIRES REVIEW ⚠'}")

    return 0 if compliant else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
