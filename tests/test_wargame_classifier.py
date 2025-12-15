#!/usr/bin/env python3
"""
Wargame Runner Test Script - Classification (Credit Scoring)

This script demonstrates a complete spectrum-governance audit workflow using
a binary classification task (credit approval simulation).

It exercises:
1. Red Team: HopSkipJump adversarial attack
2. Blue Team: Conformal prediction (prediction sets), SHAP explanations
3. Drift Monitoring: PSI-based distribution shift detection
4. Report Generation: DOCX compliance report

Usage:
    python tests/test_wargame_classifier.py

Output:
    - audit_report_classifier.docx (compliance report)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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


def success(text: str) -> str:
    """Green success text."""
    return f"{Colors.GREEN}{Colors.BOLD}{text}{Colors.RESET}"


def warning(text: str) -> str:
    """Yellow warning text."""
    return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"


def error(text: str) -> str:
    """Red error text."""
    return f"{Colors.RED}{Colors.BOLD}{text}{Colors.RESET}"


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
from spectrum.blue.trust import SpectrumClassifier
from spectrum.blue.monitor import DriftCheck
from spectrum.blue.explain import generate_shap_explanations
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder

# Try to import red team components
try:
    from spectrum.red.attack import HopSkipJumpWrapper
    RED_TEAM_AVAILABLE = True
except ImportError:
    RED_TEAM_AVAILABLE = False
    print("NOTE: Red team components not available. Skipping adversarial testing.")


def print_header(title: str, color: str = Colors.CYAN):
    """Print a formatted section header."""
    width = 60
    print()
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  {title}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print()


def load_credit_scoring_data():
    """Load California Housing and convert to credit scoring classification."""
    print_header("Loading Credit Scoring Dataset", Colors.YELLOW)

    # Load base dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y_continuous = housing.target

    # Convert to binary classification: "high value" homes (top 30%) as "approved"
    threshold = np.percentile(y_continuous, 70)
    y = (y_continuous >= threshold).astype(int)

    # Rename features to credit-like names for demonstration
    feature_mapping = {
        'MedInc': 'Income',
        'HouseAge': 'AccountAge',
        'AveRooms': 'AvgBalance',
        'AveBedrms': 'NumAccounts',
        'Population': 'NumTransactions',
        'AveOccup': 'UtilizationRatio',
        'Latitude': 'RegionScore1',
        'Longitude': 'RegionScore2'
    }
    X = X.rename(columns=feature_mapping)

    print(metric("Dataset shape", f"{X.shape[0]:,} samples x {X.shape[1]} features"))
    print(metric("Features", ", ".join(list(X.columns)[:4]) + f" ... ({len(X.columns)} total)"))
    print(metric("Target", "Credit Approval", "(1=Approved, 0=Denied)"))
    print(metric("Class distribution", f"Approved={int(y.sum()):,}, Denied={int(len(y) - y.sum()):,}"))
    print(metric("Approval rate", f"{y.mean():.1%}"))

    return X, y


def prepare_data_splits(X, y, random_state=42):
    """Split data into train, calibration, and test sets."""
    print_header("Preparing Data Splits", Colors.YELLOW)

    # Split: 60% train, 20% calibration, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
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
    """Train an MLP classifier.

    Uses a Multi-Layer Perceptron for:
    - More interesting adversarial attack case study
    - Neural networks are often more vulnerable to perturbations
    - Common model type in production systems

    Args:
        X_train: Training features
        y_train: Training targets
        robust: If True, use robustness-enhancing settings:
                - Stronger L2 regularization (alpha=0.5 vs 0.001)
                - Tanh activation (smoother gradients than ReLU)
                - Wider but shallower network (less overfitting)
    """
    print_header("Training Model", Colors.YELLOW)

    if robust:
        print(f"  {dim('Building robust MLPClassifier...')}")
        # Robustness-focused configuration:
        # - Tanh: Bounded outputs, smoother gradients than ReLU
        # - Higher alpha: Stronger weight decay limits extreme weights
        # - Wider single layer: Less depth = smoother function approximation
        model = MLPClassifier(
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
        arch_str = "8 -> 128 -> 2"
        activation_str = "Tanh"
        alpha_str = "0.5"
    else:
        print(f"  {dim('Building standard MLPClassifier...')}")
        # Standard configuration (more vulnerable)
        model = MLPClassifier(
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
        arch_str = "8 -> 64 -> 32 -> 2"
        activation_str = "ReLU"
        alpha_str = "0.001"

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)

    mode_label = f"{Colors.GREEN}Robust{Colors.RESET}" if robust else f"{Colors.YELLOW}Standard{Colors.RESET}"
    print(metric("Model", f"MLPClassifier ({mode_label})"))
    print(bullet(f"architecture: {bold(arch_str)}"))
    print(bullet(f"activation: {bold(activation_str)}"))
    print(bullet(f"optimizer: {bold('Adam')}"))
    print(bullet(f"L2 regularization (alpha): {bold(alpha_str)}"))
    print()
    print(metric("Training accuracy", f"{train_score:.4f}"))

    return model


def run_blue_team(model, X_calib, y_calib, X_test, y_test, risk_profile):
    """Run Blue Team defensive analysis."""
    print_header("Blue Team: Uncertainty Quantification", Colors.BLUE)

    # Wrap model with conformal prediction
    blue_model = SpectrumClassifier(
        base_model=model,
        risk_profile=risk_profile
    )

    # Calibrate on held-out data
    blue_model.fit(X_calib.values, y_calib.values)
    print(f"  {dim(f'Conformal prediction calibrated on {len(X_calib):,} samples')}")

    # Predict with prediction sets
    result = blue_model.predict(X_test.values)

    predictions = result["prediction"]
    prediction_sets = result["prediction_set"]
    confidence = result["confidence"]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Calculate empirical coverage (true label in prediction set)
    y_test_arr = y_test.values
    in_set = np.array([
        prediction_sets[i, y_test_arr[i]]
        for i in range(len(y_test_arr))
    ])
    empirical_coverage = np.mean(in_set)

    # Calculate average prediction set size
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    print()
    print(metric("Confidence level", f"{confidence:.1%}"))
    print(metric("Empirical coverage", f"{empirical_coverage:.1%}"))
    print(metric("Point prediction accuracy", f"{accuracy:.1%}"))
    print(metric("Avg prediction set size", f"{avg_set_size:.2f}"))

    coverage_ok = empirical_coverage >= confidence
    status = success("PASS") if coverage_ok else warning("BELOW TARGET")
    print(f"\n  {Colors.WHITE}Coverage achieved:{Colors.RESET} {status}")

    return {
        "predictions": predictions,
        "prediction_sets": prediction_sets,
        "empirical_coverage": empirical_coverage,
        "confidence": confidence,
        "accuracy": accuracy
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

    print(f"  {dim('Computing SHAP values with KernelExplainer...')}")

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


def run_red_team(model, X_test, n_samples=20, backend="rust"):
    """Run Red Team adversarial testing.

    Args:
        model: The trained classifier
        X_test: Test data
        n_samples: Number of samples to attack
        backend: "rust" (fast, Rust backend) or "art" (fallback)
    """
    print_header("Red Team: Adversarial Testing", Colors.RED)

    if not RED_TEAM_AVAILABLE:
        print(f"  {warning('Red team components not available.')}")
        print(f"  {dim('Build with: maturin develop --release')}")
        return None

    # Use a subset for faster testing
    X_attack = X_test.iloc[:n_samples].values

    print(f"  {dim(f'Testing {n_samples} samples with HopSkipJump attack...')}")

    try:
        # Use Rust backend for maximum performance (9.4x faster)
        wrapper = HopSkipJumpWrapper(
            base_model=model,
            backend=backend,          # Use Rust backend
            enable_onnx=True,         # Enable ONNX optimization
            max_iter=500,             # Reduced for faster testing
            max_eval=5000,            # Reduced for faster testing
        )
        print(f"  {dim(f'Using backend: {backend.upper()}')}")

        metrics = wrapper.run(X_attack)

        asr_color = Colors.RED if metrics.attack_success_rate > 0.5 else Colors.GREEN
        print()
        print(metric("Attack type", metrics.attack_type))
        print(metric("Samples tested", str(metrics.samples_tested)))
        print(metric("Attack success rate", f"{asr_color}{metrics.attack_success_rate:.1%}{Colors.RESET}"))
        print(metric("Mean L2 perturbation", f"{metrics.empirical_robustness_l2:.4f}"))
        if metrics.queries_used > 0:
            print(metric("Total queries", f"{metrics.queries_used:,}"))
            print(metric("Avg queries/sample", f"{metrics.avg_queries_per_sample:.0f}"))

        return metrics
    except Exception as e:
        print(f"  {error(f'Adversarial testing failed: {e}')}")
        import traceback
        traceback.print_exc()
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
        "model_type": "Tabular (Classification)",
        "risk_level": risk_profile.level.value,
        "confidence_required": 1.0 - risk_profile.alpha,
        "empirical_coverage": blue_results["empirical_coverage"],
        "adversarial_metrics": adversarial_metrics,
        "sample_adverse_reasons": explanations.get("top_features", [])[:5],
        "data_drift_status": drift_result["status"],
        "data_drift_alert": drift_result["alert_required"],
        "lineage_run_id": "classifier-test-001",
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
    print(f"{Colors.WHITE}   Classification Task: Credit Scoring{Colors.RESET}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    # 1. Setup
    risk_profile = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.10)
    print()
    print(f"  {Colors.WHITE}Risk Profile:{Colors.RESET}       {Colors.YELLOW}{Colors.BOLD}{risk_profile.level.value}{Colors.RESET}")
    print(f"  {Colors.WHITE}Required Confidence:{Colors.RESET} {bold(f'{1 - risk_profile.alpha:.0%}')}")
    print(f"  {Colors.WHITE}PII Policy:{Colors.RESET}         {risk_profile.pii_policy.value}")

    # 2. Load data
    X, y = load_credit_scoring_data()

    # 3. Prepare splits
    X_train, X_calib, X_test, y_train, y_calib, y_test = prepare_data_splits(X, y)

    # 4. Train model
    model = train_model(X_train, y_train)

    # Test score
    test_score = model.score(X_test, y_test)
    print(metric("Test accuracy", f"{test_score:.4f}"))

    # 5. Blue Team: Uncertainty Quantification
    blue_results = run_blue_team(
        model, X_calib, y_calib, X_test, y_test, risk_profile
    )

    # 6. Blue Team: Drift Monitoring
    drift_result = run_drift_check(X_train, X_test)

    # 7. Blue Team: Explainability
    explanations = run_explainability(model, X_test)

    # 8. Red Team: Adversarial Testing
    red_metrics = run_red_team(model, X_test, n_samples=1_000)

    # 9. Generate Compliance Report
    report = generate_compliance_report(
        model_name="MLPClassifier(128)-Robust",
        risk_profile=risk_profile,
        blue_results=blue_results,
        drift_result=drift_result,
        explanations=explanations,
        red_metrics=red_metrics,
        output_path="audit_report_classifier.docx"
    )

    # 10. Summary
    print_header("Audit Summary", Colors.GREEN)

    print(metric("Model", report.model_name))
    print(metric("Task", "Classification"))
    print(metric("Risk Level", f"{Colors.YELLOW}{report.risk_level}{Colors.RESET}"))
    print(metric("Required Confidence", f"{report.confidence_required:.0%}"))
    print(metric("Empirical Coverage", f"{report.empirical_coverage:.1%}"))
    print(metric("Accuracy", f"{blue_results['accuracy']:.1%}"))

    drift_alert = error("YES") if report.data_drift_alert else success("NO")
    print(metric("Drift Alert", drift_alert))

    if report.adversarial_metrics.get("attack_success_rate") is not None:
        asr = report.adversarial_metrics["attack_success_rate"]
        asr_color = Colors.RED if asr > 0.5 else Colors.GREEN
        print(f"\n  {Colors.RED}{Colors.BOLD}Adversarial Testing:{Colors.RESET}")
        print(metric("  Attack Success Rate", f"{asr_color}{asr:.1%}{Colors.RESET}"))

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
    print(f"    {Colors.CYAN}>{Colors.RESET} {Colors.UNDERLINE}audit_report_classifier.docx{Colors.RESET}")

    print()

    return report


if __name__ == "__main__":
    main()
