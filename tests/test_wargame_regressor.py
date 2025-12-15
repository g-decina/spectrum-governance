#!/usr/bin/env python3
"""
Wargame Runner Test Script - Regression (House Price Prediction)

This script demonstrates a complete spectrum-governance audit workflow using
a regression task (house price prediction).

It exercises:
1. Red Team: Output Manipulation, Prediction Shift, and Quantile Attacks
2. Blue Team: Conformal prediction (prediction intervals), SHAP explanations
3. Drift Monitoring: PSI-based distribution shift detection
4. Report Generation: DOCX compliance report

Usage:
    python tests/test_wargame_regressor.py

Output:
    - audit_report_regressor.docx (compliance report)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add spectrum to path if running from tests directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Terminal Colors and Formatting
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = "\033[0m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    YELLOW = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_YELLOW = "\033[45m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, '')


# Disable colors if not running in a terminal
if not sys.stdout.isatty():
    Colors.disable()


def bold(text: str) -> str:
    """Make text bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def success(text: str) -> str:
    """Green success text."""
    return f"{Colors.GREEN}{Colors.BOLD}{text}{Colors.RESET}"


def warning(text: str) -> str:
    """Yellow warning text."""
    return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"


def error(text: str) -> str:
    """Red error text."""
    return f"{Colors.RED}{Colors.BOLD}{text}{Colors.RESET}"


def info(text: str) -> str:
    """Cyan info text."""
    return f"{Colors.CYAN}{text}{Colors.RESET}"


def dim(text: str) -> str:
    """Dimmed text for secondary information."""
    return f"{Colors.DIM}{text}{Colors.RESET}"


def metric(label: str, value: str, unit: str = "") -> str:
    """Format a metric with label and value."""
    unit_str = f" {dim(unit)}" if unit else ""
    return f"  {Colors.WHITE}{label}:{Colors.RESET} {Colors.BOLD}{value}{Colors.RESET}{unit_str}"


def bullet(text: str, indent: int = 2) -> str:
    """Format a bullet point."""
    return f"{' ' * indent}{Colors.CYAN}>{Colors.RESET} {text}"

from spectrum.infra.types import RiskProfile, RiskLevel
from spectrum.blue.trust import SpectrumRegressor
from spectrum.blue.monitor import DriftCheck
from spectrum.blue.explain import generate_shap_explanations
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder

# Try to import red team regression attacks
try:
    from spectrum.red.attack import (
        OutputManipulationWrapper,
        PredictionShiftWrapper,
        QuantileAttackWrapper,
    )
    RED_TEAM_AVAILABLE = True
except ImportError:
    RED_TEAM_AVAILABLE = False
    print("NOTE: Red team regression attacks not available. Skipping adversarial testing.")


def print_header(title: str, color: str = Colors.CYAN):
    """Print a formatted section header."""
    width = 60
    print()
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print()


def print_subheader(title: str):
    """Print a sub-section header."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}--- {title} ---{Colors.RESET}")


def print_step(step_num: int, description: str):
    """Print a numbered step indicator."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}[{step_num}]{Colors.RESET} {Colors.WHITE}{description}{Colors.RESET}")


def load_housing_data():
    """Load California Housing dataset for regression."""
    print_header("Loading California Housing Dataset", Colors.YELLOW)

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    print(metric("Dataset shape", f"{X.shape[0]:,} samples x {X.shape[1]} features"))
    print(metric("Features", ", ".join(X.columns[:4]) + f" ... ({len(X.columns)} total)"))
    print(metric("Target", "Median house value", "($100,000s)"))
    print(metric("Target range", f"[{y.min():.2f}, {y.max():.2f}]"))
    print(metric("Target mean", f"{y.mean():.2f}"))

    return X, y


def prepare_data_splits(X, y, random_state=42):
    """Split data into train, calibration, and test sets."""
    print_header("Preparing Data Splits", Colors.YELLOW)

    # Split: 60% train, 20% calibration, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    print(metric("Training set", f"{len(X_train):,} samples", "(60%)"))
    print(metric("Calibration set", f"{len(X_calib):,} samples", "(20%)"))
    print(metric("Test set", f"{len(X_test):,} samples", "(20%)"))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_calib_scaled = pd.DataFrame(
        scaler.transform(X_calib),
        columns=X_calib.columns,
        index=X_calib.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test


def train_model(X_train, y_train, robust: bool = True):
    """Train an MLP regressor.

    Uses a Multi-Layer Perceptron for:
    - More interesting adversarial attack case study
    - Neural networks are often more vulnerable to perturbations
    - Common model type in production systems

    Args:
        X_train: Training features
        y_train: Training targets
        robust: If True, use robustness-enhancing settings:
                - Stronger L2 regularization (alpha=0.1 vs 0.001)
                - Tanh activation (smoother gradients than ReLU)
                - Wider but shallower network (less overfitting)
    """
    print_header("Training Model", Colors.YELLOW)

    if robust:
        print(f"  {dim('Building robust MLPRegressor...')}")
        # Robustness-focused configuration:
        # - Tanh: Bounded outputs, smoother gradients than ReLU
        # - Higher alpha: Stronger weight decay limits extreme weights
        # - Wider single layer: Less depth = smoother function approximation
        model = MLPRegressor(
            hidden_layer_sizes=(128,),    # Single wider layer
            activation='tanh',            # Bounded, smooth gradients
            solver='adam',
            alpha=0.5,                    # Strong L2 regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        arch_str = "8 -> 128 -> 1"
        activation_str = "Tanh"
        alpha_str = "0.5"
    else:
        print(f"  {dim('Building standard MLPRegressor...')}")
        # Standard configuration (more vulnerable)
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),  # Two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.001,                  # Weak L2 regularization
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        arch_str = "8 -> 64 -> 32 -> 1"
        activation_str = "ReLU"
        alpha_str = "0.001"

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)

    mode_label = f"{Colors.GREEN}Robust{Colors.RESET}" if robust else f"{Colors.YELLOW}Standard{Colors.RESET}"
    print(metric("Model", f"MLPRegressor ({mode_label})"))
    print(bullet(f"architecture: {bold(arch_str)}"))
    print(bullet(f"activation: {bold(activation_str)}"))
    print(bullet(f"optimizer: {bold('Adam')}"))
    print(bullet(f"L2 regularization (alpha): {bold(alpha_str)}"))
    print()
    print(metric("Training R2", f"{train_score:.4f}"))

    return model


def run_blue_team(model, X_calib, y_calib, X_test, y_test, risk_profile):
    """Run Blue Team defensive analysis."""
    print_header("Blue Team: Uncertainty Quantification", Colors.BLUE)

    # Wrap model with conformal prediction
    blue_model = SpectrumRegressor(
        base_model=model,
        risk_profile=risk_profile
    )

    # Calibrate on held-out data
    blue_model.fit(X_calib.values, y_calib.values)
    print(f"  {dim(f'Conformal prediction calibrated on {len(X_calib):,} samples')}")

    # Predict with prediction intervals
    result = blue_model.predict(X_test.values)

    predictions = result["prediction"]
    lower_bounds = result["lower_bound"]
    upper_bounds = result["upper_bound"]
    confidence = result["confidence"]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Calculate R2
    r2 = r2_score(y_test, predictions)

    # Calculate empirical coverage
    y_test_arr = y_test.values
    in_interval = (y_test_arr >= lower_bounds) & (y_test_arr <= upper_bounds)
    empirical_coverage = np.mean(in_interval)

    # Calculate average interval width
    avg_interval_width = np.mean(upper_bounds - lower_bounds)

    print()
    print(metric("Confidence level", f"{confidence:.1%}"))
    print(metric("Empirical coverage", f"{empirical_coverage:.1%}"))
    print(metric("R2 score", f"{r2:.4f}"))
    print(metric("RMSE", f"{rmse:.4f}"))
    print(metric("Avg interval width", f"{avg_interval_width:.4f}"))

    coverage_ok = empirical_coverage >= confidence
    status = success("PASS") if coverage_ok else warning("BELOW TARGET")
    print(f"\n  {Colors.WHITE}Coverage achieved:{Colors.RESET} {status}")

    return {
        "predictions": predictions,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "empirical_coverage": empirical_coverage,
        "confidence": confidence,
        "rmse": rmse,
        "r2": r2,
        "interval_width": avg_interval_width,
    }


def run_drift_check(X_train, X_test):
    """Run PSI-based drift detection."""
    print_header("Blue Team: Drift Monitoring", Colors.BLUE)

    drift_result = DriftCheck(
        reference_data=X_train,
        current_data=X_test
    )

    drift_detected = drift_result['drift_detected']
    drift_status = error("YES") if drift_detected else success("NO")

    print(metric("Drift detected", drift_status))
    print(metric("Max PSI", f"{drift_result['max_psi']:.4f}"))
    print(metric("Status", drift_result['status']))

    if drift_result['drifted_features']:
        print(f"\n  {warning('Drifted features:')}")
        for feat in drift_result['drifted_features']:
            print(bullet(feat, indent=4))

    return drift_result


def run_explainability(model, X_test):
    """Generate SHAP-based explanations."""
    print_header("Blue Team: Explainability (SHAP)", Colors.BLUE)

    print(f"  {dim('Computing SHAP values with TreeExplainer...')}")

    explanations = generate_shap_explanations(
        model=model,
        X_test=X_test,
        max_samples=50
    )

    print(f"\n  {Colors.WHITE}Top contributing features:{Colors.RESET}")
    top_features = explanations.get('top_features', [])[:5]
    for i, feature in enumerate(top_features, 1):
        # Color code by rank
        if i == 1:
            feat_display = f"{Colors.GREEN}{Colors.BOLD}{feature}{Colors.RESET}"
        elif i <= 3:
            feat_display = f"{Colors.CYAN}{feature}{Colors.RESET}"
        else:
            feat_display = feature
        print(f"    {Colors.DIM}{i}.{Colors.RESET} {feat_display}")

    return explanations


def run_red_team(model, X_test, y_test, blue_results, n_samples=20):
    """Run Red Team adversarial testing for regression.

    Runs three regression attacks:
    1. Output Manipulation: Discretize outputs into bins and attack
    2. Prediction Shift: Maximize output change
    3. Quantile Attack: Break conformal prediction coverage

    Args:
        model: The trained regressor
        X_test: Test data
        y_test: True target values
        blue_results: Results from blue team (for interval width)
        n_samples: Number of samples to attack
    """
    print_header("Red Team: Adversarial Testing", Colors.RED)

    if not RED_TEAM_AVAILABLE:
        print(f"  {warning('Red team components not available.')}")
        print(f"  {dim('Build with: maturin develop --release')}")
        return None

    X_attack = X_test.iloc[:n_samples].values
    y_attack = y_test.iloc[:n_samples].values

    print(f"  {dim(f'Testing {n_samples} samples...')}")

    results = {}

    # 1. Output Manipulation Attack
    try:
        print_subheader("Output Manipulation Attack")
        print(f"  {dim('Discretizing outputs into 5 bins and finding bin-flipping perturbations...')}")

        attack = OutputManipulationWrapper(
            base_model=model,
            n_bins=5,
            max_iter=30,
            max_eval=2000,
            parallel=True,
            bin_mode="quantile",
            confidence_level=0.99,  # 99% CI on mean attempts
        )
        metrics = attack.run(X_attack)

        # Report attempts-based metrics (not success rate - always ~100% for regression)
        print(metric("Samples succeeded", f"{metrics.samples_succeeded}/{metrics.samples_tested}"))
        if metrics.samples_never_succeeded > 0:
            print(metric("Never succeeded", f"{warning(str(metrics.samples_never_succeeded))}"))

        if not np.isnan(metrics.mean_attempts_to_success):
            # Color code: fewer attempts = more vulnerable (red), more attempts = more robust (green)
            attempts_color = Colors.RED if metrics.mean_attempts_to_success < 100 else Colors.GREEN
            print(metric("Mean attempts to success", f"{attempts_color}{metrics.mean_attempts_to_success:.1f}{Colors.RESET}", "queries"))
            print(metric("99% CI", f"[{metrics.attempts_ci[0]:.1f}, {metrics.attempts_ci[1]:.1f}]"))
        print(metric("Mean L2 perturbation", f"{metrics.mean_perturbation_l2:.4f}"))
        if metrics.bin_flip_rate is not None:
            print(metric("Bin flip rate", f"{metrics.bin_flip_rate:.1%}"))
        if metrics.mean_bin_distance:
            print(metric("Mean bin distance", f"{metrics.mean_bin_distance:.2f}"))

        results["output_manipulation"] = metrics

    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")
        import traceback
        traceback.print_exc()

    # 2. Prediction Shift Attack
    try:
        print_subheader("Prediction Shift Attack")
        print(f"  {dim('Finding perturbations that maximize output change...')}")

        attack = PredictionShiftWrapper(
            base_model=model,
            epsilon=1.0,  # Max L2 perturbation
            max_iter=50,
            n_directions=30,
            parallel=True,
            target_direction="any",
            success_threshold=0.10,  # 10% shift = success
            confidence_level=0.99,   # 99% CI on mean attempts
        )
        metrics = attack.run(X_attack)

        # Report attempts-based metrics
        print(metric("Samples succeeded", f"{metrics.samples_succeeded}/{metrics.samples_tested}"))
        if metrics.samples_never_succeeded > 0:
            print(metric("Never succeeded", f"{warning(str(metrics.samples_never_succeeded))}"))

        if not np.isnan(metrics.mean_attempts_to_success):
            attempts_color = Colors.RED if metrics.mean_attempts_to_success < 50 else Colors.GREEN
            print(metric("Mean attempts to success", f"{attempts_color}{metrics.mean_attempts_to_success:.1f}{Colors.RESET}", "directions"))
            print(metric("99% CI", f"[{metrics.attempts_ci[0]:.1f}, {metrics.attempts_ci[1]:.1f}]"))
        print(metric("Mean L2 perturbation", f"{metrics.mean_perturbation_l2:.4f}"))
        print(metric("Mean prediction shift", f"{metrics.mean_absolute_shift:.4f}"))
        print(metric("Max prediction shift", f"{metrics.max_prediction_shift:.4f}"))

        results["prediction_shift"] = metrics

    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")
        import traceback
        traceback.print_exc()

    # 3. Quantile Attack (for CP models)
    try:
        print_subheader("Quantile Attack (Break Coverage)")
        print(f"  {dim('Finding perturbations that push values outside prediction intervals...')}")

        # Use interval width from blue team results
        calibration_width = blue_results.get("interval_width", 1.0) / 2

        attack = QuantileAttackWrapper(
            base_model=model,
            epsilon=0.5,  # Smaller epsilon for CP attack
            max_iter=50,
            n_directions=30,
            parallel=True,
            attack_mode="break_coverage",
            calibration_width=calibration_width,
        )
        metrics = attack.run(X_attack, y_attack)

        break_rate_color = Colors.RED if metrics.coverage_break_rate > 0.3 else Colors.GREEN
        print(metric("Coverage break rate", f"{break_rate_color}{metrics.coverage_break_rate:.1%}{Colors.RESET}"))
        print(metric("Original coverage", f"{metrics.original_coverage:.1%}"))
        print(metric("Adversarial coverage", f"{metrics.adversarial_coverage:.1%}"))
        print(metric("Mean L2 perturbation", f"{metrics.mean_perturbation_l2:.4f}"))

        results["quantile_attack"] = metrics

    except Exception as e:
        print(f"  {error(f'Attack failed: {e}')}")
        import traceback
        traceback.print_exc()

    # Return combined results
    if results:
        # Use output manipulation as primary metric
        primary = results.get("output_manipulation") or results.get("prediction_shift")
        return primary

    return None


def generate_compliance_report(
    model_name: str,
    risk_profile: RiskProfile,
    blue_results: dict,
    drift_result: dict,
    explanations: dict,
    red_metrics,
    output_path: str
):
    """Generate DOCX compliance report."""
    print_header("Generating Compliance Report", Colors.YELLOW)

    print(f"  {dim('Building DOCX report...')}")

    # Prepare adversarial metrics
    if red_metrics is not None:
        adversarial_metrics = red_metrics.to_dict()
    else:
        adversarial_metrics = {
            "attack_type": "Not Performed",
            "attack_success_rate": None,
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

    # Create ComplianceReport
    report_data = {
        "model_name": model_name,
        "model_type": "Tabular (Regression)",
        "risk_level": risk_profile.level.value,
        "confidence_required": 1.0 - risk_profile.alpha,
        "empirical_coverage": blue_results["empirical_coverage"],
        "adversarial_metrics": adversarial_metrics,
        "sample_adverse_reasons": explanations.get("top_features", [])[:5],
        "data_drift_status": drift_result["status"],
        "data_drift_alert": drift_result["alert_required"],
        "lineage_run_id": "regressor-test-001",
        "audit_log_path": "wargame_audit.jsonl"
    }

    # Validate with Pydantic
    report = ComplianceReport(**report_data)

    # Generate DOCX
    builder = ReportBuilder()
    builder.generate_docx(
        data=report.model_dump(),
        output_path=output_path,
        template_type="tier1_forensic_audit"
    )

    print(f"\n  {success('Report saved:')} {Colors.UNDERLINE}{output_path}{Colors.RESET}")

    return report


def main():
    """Run the complete wargame audit workflow."""
    # Banner
    print()
    print(f"{Colors.YELLOW}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.YELLOW}{Colors.BOLD}   SPECTRUM GOVERNANCE - WARGAME RUNNER{Colors.RESET}")
    print(f"{Colors.WHITE}   Regression Task: House Price Prediction{Colors.RESET}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    # 1. Setup
    risk_profile = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.10)
    print()
    print(f"  {Colors.WHITE}Risk Profile:{Colors.RESET}       {Colors.YELLOW}{Colors.BOLD}{risk_profile.level.value}{Colors.RESET}")
    print(f"  {Colors.WHITE}Required Confidence:{Colors.RESET} {bold(f'{1 - risk_profile.alpha:.0%}')}")
    print(f"  {Colors.WHITE}PII Policy:{Colors.RESET}         {risk_profile.pii_policy.value}")

    # 2. Load data
    X, y = load_housing_data()

    # 3. Prepare splits
    X_train, X_calib, X_test, y_train, y_calib, y_test = prepare_data_splits(X, y)

    # 4. Train model
    model = train_model(X_train, y_train)

    # Test score
    test_score = model.score(X_test, y_test)
    print(metric("Test R2", f"{test_score:.4f}"))

    # 5. Blue Team: Uncertainty Quantification
    blue_results = run_blue_team(
        model, X_calib, y_calib, X_test, y_test, risk_profile
    )

    # 6. Blue Team: Drift Monitoring
    drift_result = run_drift_check(X_train, X_test)

    # 7. Blue Team: Explainability
    explanations = run_explainability(model, X_test)

    # 8. Red Team: Adversarial Testing
    red_metrics = run_red_team(model, X_test, y_test, blue_results, n_samples=1_000)

    # 9. Generate Compliance Report
    report = generate_compliance_report(
        model_name="MLPRegressor(128)-Robust",
        risk_profile=risk_profile,
        blue_results=blue_results,
        drift_result=drift_result,
        explanations=explanations,
        red_metrics=red_metrics,
        output_path="audit_report_regressor.docx"
    )

    # 10. Summary
    print_header("Audit Summary", Colors.GREEN)

    print(metric("Model", report.model_name))
    print(metric("Task", "Regression"))
    print(metric("Risk Level", f"{Colors.YELLOW}{report.risk_level}{Colors.RESET}"))
    print(metric("Required Confidence", f"{report.confidence_required:.0%}"))
    print(metric("Empirical Coverage", f"{report.empirical_coverage:.1%}"))
    print(metric("R2 Score", f"{blue_results['r2']:.4f}"))
    print(metric("RMSE", f"{blue_results['rmse']:.4f}"))

    drift_alert = error("YES") if report.data_drift_alert else success("NO")
    print(metric("Drift Alert", drift_alert))

    if red_metrics:
        print(f"\n  {Colors.RED}{Colors.BOLD}Adversarial Testing:{Colors.RESET}")
        print(metric("  Attack", red_metrics.attack_type))
        if not np.isnan(red_metrics.mean_attempts_to_success):
            attempts = red_metrics.mean_attempts_to_success
            # Fewer attempts = more vulnerable
            attempts_color = Colors.RED if attempts < 100 else Colors.GREEN
            print(metric("  Mean Attempts", f"{attempts_color}{attempts:.1f}{Colors.RESET}"))
            ci_pct = f"{red_metrics.confidence_level*100:.0f}%"
            print(metric(f"  {ci_pct} CI", f"[{red_metrics.attempts_ci[0]:.1f}, {red_metrics.attempts_ci[1]:.1f}]"))
        print(metric("  Mean L2", f"{red_metrics.mean_perturbation_l2:.4f}"))

    # Overall compliance status
    coverage_pass = report.empirical_coverage >= report.confidence_required
    drift_pass = not report.data_drift_alert

    print()
    print(f"  {'─' * 40}")

    if coverage_pass and drift_pass:
        status_text = f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD}  NO MAJOR VULNERABILITY DETECTED  {Colors.RESET}"
    else:
        status_text = f"{Colors.BG_YELLOW}{Colors.WHITE}{Colors.BOLD}  REQUIRES ATTENTION  {Colors.RESET}"

    print(f"\n  Overall Status: {status_text}")

    print(f"\n  {Colors.WHITE}Generated artifacts:{Colors.RESET}")
    print(f"    {Colors.CYAN}>{Colors.RESET} {Colors.UNDERLINE}audit_report_regressor.docx{Colors.RESET}")

    print()

    return report


if __name__ == "__main__":
    main()
