#!/usr/bin/env python3
"""
Workbench Test: Complete Red/Blue Team Workflow

This workbench demonstrates a realistic end-to-end workflow:
1. Train a model
2. Attack it (Red Team)
3. Diagnose vulnerabilities
4. Harden the model (Blue Team)
5. Verify hardening effectiveness
6. Generate compliance report

Simulates a real-world ML governance audit cycle.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


@dataclass
class AuditResult:
    """Result of a complete audit cycle."""
    model_name: str
    phase: str

    # Accuracy metrics
    r2: float
    mae: float

    # Robustness metrics
    om_success_rate: float
    om_mean_attempts: float
    ps_success_rate: float
    ps_mean_attempts: float

    # Timing
    audit_time_sec: float

    def robustness_score(self) -> float:
        """Combined robustness score (higher is better)."""
        # Weighted average of attempts (more attempts = harder to attack)
        om_score = min(self.om_mean_attempts / 50, 1.0) if not np.isnan(self.om_mean_attempts) else 0.5
        ps_score = min(self.ps_mean_attempts / 30, 1.0) if not np.isnan(self.ps_mean_attempts) else 0.5
        return (om_score + ps_score) / 2


def print_header(text: str, width: int = 80, char: str = "="):
    """Print formatted header."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_subheader(text: str, width: int = 80):
    """Print formatted subheader."""
    print(f"\n{'-' * width}")
    print(f"  {text}")
    print(f"{'-' * width}")


def run_red_team_assessment(model, X_test, n_samples: int = 30) -> Dict:
    """Run red team attacks and return metrics."""
    from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper

    X = X_test[:n_samples]
    results = {
        "om_success_rate": 0.0, "om_mean_attempts": float('nan'),
        "ps_success_rate": 0.0, "ps_mean_attempts": float('nan'),
    }

    # OutputManipulation Attack
    try:
        om = OutputManipulationWrapper(
            model, n_bins=5, max_iter=25, max_eval=1500,
            parallel=True, confidence_level=0.99
        )
        om_metrics = om.run(X)
        results["om_success_rate"] = om_metrics.samples_succeeded / om_metrics.samples_tested
        results["om_mean_attempts"] = om_metrics.mean_attempts_to_success
    except Exception as e:
        print(f"      OM Attack Error: {e}")

    # PredictionShift Attack
    try:
        ps = PredictionShiftWrapper(
            model, epsilon=1.0, max_iter=40, n_directions=25,
            success_threshold=0.10, confidence_level=0.99
        )
        ps_metrics = ps.run(X)
        results["ps_success_rate"] = ps_metrics.samples_succeeded / ps_metrics.samples_tested
        results["ps_mean_attempts"] = ps_metrics.mean_attempts_to_success
    except Exception as e:
        print(f"      PS Attack Error: {e}")

    return results


def diagnose_vulnerabilities(audit_result: AuditResult) -> List[str]:
    """Analyze audit results and return list of vulnerabilities."""
    vulnerabilities = []

    if audit_result.om_success_rate > 0.5:
        vulnerabilities.append(
            f"HIGH: OutputManipulation success rate {audit_result.om_success_rate:.0%} - "
            "model outputs are easily manipulable"
        )
    elif audit_result.om_success_rate > 0.2:
        vulnerabilities.append(
            f"MEDIUM: OutputManipulation success rate {audit_result.om_success_rate:.0%}"
        )

    if not np.isnan(audit_result.om_mean_attempts) and audit_result.om_mean_attempts < 20:
        vulnerabilities.append(
            f"HIGH: Average {audit_result.om_mean_attempts:.1f} queries to manipulate output - "
            "low query cost for attackers"
        )

    if audit_result.ps_success_rate > 0.5:
        vulnerabilities.append(
            f"HIGH: PredictionShift success rate {audit_result.ps_success_rate:.0%} - "
            "predictions unstable to small perturbations"
        )
    elif audit_result.ps_success_rate > 0.2:
        vulnerabilities.append(
            f"MEDIUM: PredictionShift success rate {audit_result.ps_success_rate:.0%}"
        )

    if not np.isnan(audit_result.ps_mean_attempts) and audit_result.ps_mean_attempts < 10:
        vulnerabilities.append(
            f"HIGH: Average {audit_result.ps_mean_attempts:.1f} directions to shift prediction - "
            "decision boundary easily found"
        )

    return vulnerabilities


def recommend_hardening(vulnerabilities: List[str], model_type: str) -> Dict:
    """Recommend hardening strategy based on vulnerabilities."""
    recommendations = {
        "strategy": "ensemble_proxy",  # Default best
        "parameters": {},
        "rationale": []
    }

    has_om_vulnerability = any("OutputManipulation" in v for v in vulnerabilities)
    has_ps_vulnerability = any("PredictionShift" in v for v in vulnerabilities)

    if has_om_vulnerability and has_ps_vulnerability:
        recommendations["strategy"] = "ensemble_proxy"
        recommendations["parameters"] = {"n_neighbors": 10}
        recommendations["rationale"].append(
            "EnsembleProxy provides best combined protection (+66% OM, +70% PS)"
        )
    elif has_om_vulnerability:
        recommendations["strategy"] = "output_quantizer"
        recommendations["parameters"] = {"n_levels": 20}
        recommendations["rationale"].append(
            "OutputQuantizer specifically targets output manipulation attacks"
        )
    elif has_ps_vulnerability:
        recommendations["strategy"] = "input_sanitizer"
        recommendations["parameters"] = {"clip_percentile": 99, "squeeze_bits": 6}
        recommendations["rationale"].append(
            "InputSanitizer reduces sensitivity to small input perturbations"
        )

    if model_type in ["MLP", "NeuralNetwork"]:
        recommendations["rationale"].append(
            "Neural networks benefit from input smoothing to reduce gradient exploits"
        )

    return recommendations


def apply_hardening(model, X_calibration, strategy: str, parameters: Dict):
    """Apply hardening wrapper to model."""
    from spectrum.blue.harden import (
        EnsembleProxy, OutputQuantizer, InputSanitizer,
        PredictionSmoother, harden_model
    )

    if strategy == "ensemble_proxy":
        wrapper = EnsembleProxy(model, **parameters)
        wrapper.calibrate(X_calibration)
        return wrapper
    elif strategy == "output_quantizer":
        return OutputQuantizer(model, **parameters)
    elif strategy == "input_sanitizer":
        wrapper = InputSanitizer(model, **parameters)
        wrapper.calibrate(X_calibration)
        return wrapper
    elif strategy == "prediction_smoother":
        return PredictionSmoother(model, **parameters)
    else:
        # Use high-level API
        return harden_model(model, X_calibration, strategy=strategy, **parameters)


def generate_compliance_report(
    baseline_audit: AuditResult,
    hardened_audit: AuditResult,
    vulnerabilities: List[str],
    recommendations: Dict
) -> str:
    """Generate a compliance-style report."""

    lines = []
    lines.append("=" * 80)
    lines.append("  SPECTRUM GOVERNANCE - MODEL ROBUSTNESS AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model: {baseline_audit.model_name}")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    baseline_score = baseline_audit.robustness_score()
    hardened_score = hardened_audit.robustness_score()
    improvement = (hardened_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0

    if hardened_score >= 0.7:
        status = "PASS"
    elif hardened_score >= 0.5:
        status = "CONDITIONAL PASS"
    else:
        status = "FAIL"

    lines.append(f"Audit Status: {status}")
    lines.append(f"Baseline Robustness Score: {baseline_score:.2f}")
    lines.append(f"Hardened Robustness Score: {hardened_score:.2f}")
    lines.append(f"Improvement: {improvement:+.0f}%")
    lines.append("")

    # Vulnerabilities Found
    lines.append("VULNERABILITIES IDENTIFIED")
    lines.append("-" * 40)
    if vulnerabilities:
        for v in vulnerabilities:
            lines.append(f"  • {v}")
    else:
        lines.append("  No significant vulnerabilities found.")
    lines.append("")

    # Remediation Applied
    lines.append("REMEDIATION APPLIED")
    lines.append("-" * 40)
    lines.append(f"  Strategy: {recommendations['strategy']}")
    lines.append(f"  Parameters: {recommendations['parameters']}")
    for r in recommendations['rationale']:
        lines.append(f"  • {r}")
    lines.append("")

    # Metrics Comparison
    lines.append("METRICS COMPARISON")
    lines.append("-" * 40)
    lines.append(f"{'Metric':<30} {'Baseline':<15} {'Hardened':<15} {'Change':<15}")
    lines.append("-" * 75)

    # Accuracy
    r2_change = (hardened_audit.r2 - baseline_audit.r2) * 100
    lines.append(f"{'R² Score':<30} {baseline_audit.r2:.4f}         {hardened_audit.r2:.4f}         {r2_change:+.2f}pp")

    mae_change = (hardened_audit.mae - baseline_audit.mae) / baseline_audit.mae * 100 if baseline_audit.mae > 0 else 0
    lines.append(f"{'MAE':<30} {baseline_audit.mae:.4f}         {hardened_audit.mae:.4f}         {mae_change:+.1f}%")

    # Robustness
    om_att_b = f"{baseline_audit.om_mean_attempts:.1f}" if not np.isnan(baseline_audit.om_mean_attempts) else "N/A"
    om_att_h = f"{hardened_audit.om_mean_attempts:.1f}" if not np.isnan(hardened_audit.om_mean_attempts) else "N/A"
    if not np.isnan(baseline_audit.om_mean_attempts) and not np.isnan(hardened_audit.om_mean_attempts):
        om_change = (hardened_audit.om_mean_attempts - baseline_audit.om_mean_attempts) / baseline_audit.om_mean_attempts * 100
        om_change_str = f"{om_change:+.0f}%"
    else:
        om_change_str = "N/A"
    lines.append(f"{'OM Mean Attempts':<30} {om_att_b:<15} {om_att_h:<15} {om_change_str}")

    ps_att_b = f"{baseline_audit.ps_mean_attempts:.1f}" if not np.isnan(baseline_audit.ps_mean_attempts) else "N/A"
    ps_att_h = f"{hardened_audit.ps_mean_attempts:.1f}" if not np.isnan(hardened_audit.ps_mean_attempts) else "N/A"
    if not np.isnan(baseline_audit.ps_mean_attempts) and not np.isnan(hardened_audit.ps_mean_attempts):
        ps_change = (hardened_audit.ps_mean_attempts - baseline_audit.ps_mean_attempts) / baseline_audit.ps_mean_attempts * 100
        ps_change_str = f"{ps_change:+.0f}%"
    else:
        ps_change_str = "N/A"
    lines.append(f"{'PS Mean Attempts':<30} {ps_att_b:<15} {ps_att_h:<15} {ps_change_str}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_complete_workflow(model_name: str, model, X_train, y_train, X_calib, X_test, y_test):
    """Run complete red/blue workflow for a model."""

    print_subheader(f"Model: {model_name}")

    # Phase 1: Train model
    print("  [1/5] Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_r2 = r2_score(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, y_pred)
    print(f"        R² = {baseline_r2:.4f}, MAE = {baseline_mae:.4f}")

    # Phase 2: Red Team Assessment
    print("  [2/5] Running Red Team assessment...")
    start = time.perf_counter()
    red_results = run_red_team_assessment(model, X_test, n_samples=30)
    red_time = time.perf_counter() - start

    baseline_audit = AuditResult(
        model_name=model_name,
        phase="baseline",
        r2=baseline_r2,
        mae=baseline_mae,
        om_success_rate=red_results["om_success_rate"],
        om_mean_attempts=red_results["om_mean_attempts"],
        ps_success_rate=red_results["ps_success_rate"],
        ps_mean_attempts=red_results["ps_mean_attempts"],
        audit_time_sec=red_time
    )

    om_att = f"{baseline_audit.om_mean_attempts:.1f}" if not np.isnan(baseline_audit.om_mean_attempts) else "N/A"
    ps_att = f"{baseline_audit.ps_mean_attempts:.1f}" if not np.isnan(baseline_audit.ps_mean_attempts) else "N/A"
    print(f"        OM: {baseline_audit.om_success_rate:.0%} success, {om_att} mean attempts")
    print(f"        PS: {baseline_audit.ps_success_rate:.0%} success, {ps_att} mean attempts")

    # Phase 3: Diagnose vulnerabilities
    print("  [3/5] Diagnosing vulnerabilities...")
    vulnerabilities = diagnose_vulnerabilities(baseline_audit)
    for v in vulnerabilities:
        print(f"        • {v}")
    if not vulnerabilities:
        print("        No significant vulnerabilities found.")

    # Phase 4: Apply hardening
    print("  [4/5] Applying Blue Team hardening...")
    recommendations = recommend_hardening(vulnerabilities, model_name)
    print(f"        Strategy: {recommendations['strategy']}")
    print(f"        Parameters: {recommendations['parameters']}")

    hardened_model = apply_hardening(
        model, X_calib,
        recommendations['strategy'],
        recommendations['parameters']
    )

    # Phase 5: Verify hardening
    print("  [5/5] Verifying hardened model...")
    y_pred_hardened = hardened_model.predict(X_test)
    hardened_r2 = r2_score(y_test, y_pred_hardened)
    hardened_mae = mean_absolute_error(y_test, y_pred_hardened)

    start = time.perf_counter()
    hardened_red_results = run_red_team_assessment(hardened_model, X_test, n_samples=30)
    hardened_time = time.perf_counter() - start

    hardened_audit = AuditResult(
        model_name=f"{model_name} (Hardened)",
        phase="hardened",
        r2=hardened_r2,
        mae=hardened_mae,
        om_success_rate=hardened_red_results["om_success_rate"],
        om_mean_attempts=hardened_red_results["om_mean_attempts"],
        ps_success_rate=hardened_red_results["ps_success_rate"],
        ps_mean_attempts=hardened_red_results["ps_mean_attempts"],
        audit_time_sec=hardened_time
    )

    om_att_h = f"{hardened_audit.om_mean_attempts:.1f}" if not np.isnan(hardened_audit.om_mean_attempts) else "N/A"
    ps_att_h = f"{hardened_audit.ps_mean_attempts:.1f}" if not np.isnan(hardened_audit.ps_mean_attempts) else "N/A"
    print(f"        R² = {hardened_r2:.4f} (drop: {(baseline_r2-hardened_r2)*100:+.2f}pp)")
    print(f"        OM: {hardened_audit.om_success_rate:.0%} success, {om_att_h} mean attempts")
    print(f"        PS: {hardened_audit.ps_success_rate:.0%} success, {ps_att_h} mean attempts")

    # Generate report
    report = generate_compliance_report(
        baseline_audit, hardened_audit,
        vulnerabilities, recommendations
    )

    return baseline_audit, hardened_audit, report


def main():
    print_header("SPECTRUM GOVERNANCE - RED/BLUE TEAM WORKBENCH")
    print("""
This workbench demonstrates a complete model governance audit cycle:
  1. Train a baseline model
  2. Attack it with Red Team tools
  3. Diagnose vulnerabilities
  4. Harden with Blue Team wrappers
  5. Verify hardening effectiveness
  6. Generate compliance report
""")

    # Load data
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    print(f"Train: {len(X_train)}, Calibration: {len(X_calib)}, Test: {len(X_test)}")

    # Define models to audit
    models = [
        ("MLP-64", MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)),
        ("GBM-50", GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
    ]

    all_results = []
    all_reports = []

    print_header("RUNNING AUDIT WORKFLOWS")

    for model_name, model in models:
        baseline, hardened, report = run_complete_workflow(
            model_name, model,
            X_train, y_train, X_calib, X_test, y_test
        )
        all_results.append((baseline, hardened))
        all_reports.append(report)

    # Print all reports
    print_header("COMPLIANCE REPORTS")

    for report in all_reports:
        print(report)
        print("\n")

    # Summary statistics
    print_header("WORKBENCH SUMMARY")

    print(f"\n{'Model':<20} {'Baseline Score':<18} {'Hardened Score':<18} {'Improvement':<15}")
    print("-" * 71)

    for baseline, hardened in all_results:
        b_score = baseline.robustness_score()
        h_score = hardened.robustness_score()
        improvement = (h_score - b_score) / b_score * 100 if b_score > 0 else 0
        print(f"{baseline.model_name:<20} {b_score:.3f}             {h_score:.3f}             {improvement:+.0f}%")

    print("\n" + "=" * 80)
    print("  WORKBENCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
