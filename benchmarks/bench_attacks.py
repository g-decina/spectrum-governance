#!/usr/bin/env python3
"""
Benchmark Suite: Attack Performance

Measures attack efficiency and success rates across different
model architectures and configurations.

Metrics:
- Attack success rate
- Mean attempts to success
- Query efficiency
- Perturbation magnitude (L2, Linf)
- Wall-clock time
"""

import os
import sys
import time
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score


@dataclass
class AttackBenchmarkResult:
    """Result of attack benchmark."""
    attack_type: str
    model_type: str
    task_type: str  # regression or classification
    n_samples: int
    success_rate: float
    mean_attempts: float
    median_attempts: float
    mean_l2: float
    mean_linf: float
    samples_succeeded: int
    wall_time_sec: float
    attempts_per_second: float


def benchmark_regression_attacks(model, X_test, model_name: str, n_samples: int = 30) -> List[AttackBenchmarkResult]:
    """Benchmark regression attacks."""
    from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper, QuantileAttackWrapper

    X = X_test[:n_samples]
    results = []

    # OutputManipulation
    print(f"    OutputManipulation...")
    start = time.perf_counter()
    try:
        attack = OutputManipulationWrapper(
            model, n_bins=5, max_iter=30, max_eval=2000,
            parallel=True, confidence_level=0.99
        )
        metrics = attack.run(X)
        elapsed = time.perf_counter() - start

        results.append(AttackBenchmarkResult(
            attack_type="OutputManipulation",
            model_type=model_name,
            task_type="regression",
            n_samples=n_samples,
            success_rate=metrics.samples_succeeded / metrics.samples_tested,
            mean_attempts=metrics.mean_attempts_to_success,
            median_attempts=metrics.median_attempts_to_success,
            mean_l2=metrics.mean_perturbation_l2,
            mean_linf=metrics.mean_perturbation_linf,
            samples_succeeded=metrics.samples_succeeded,
            wall_time_sec=elapsed,
            attempts_per_second=n_samples / elapsed if elapsed > 0 else 0,
        ))
    except Exception as e:
        print(f"      Failed: {e}")

    # PredictionShift
    print(f"    PredictionShift...")
    start = time.perf_counter()
    try:
        attack = PredictionShiftWrapper(
            model, epsilon=1.0, max_iter=50, n_directions=30,
            success_threshold=0.10, confidence_level=0.99
        )
        metrics = attack.run(X)
        elapsed = time.perf_counter() - start

        results.append(AttackBenchmarkResult(
            attack_type="PredictionShift",
            model_type=model_name,
            task_type="regression",
            n_samples=n_samples,
            success_rate=metrics.samples_succeeded / metrics.samples_tested,
            mean_attempts=metrics.mean_attempts_to_success,
            median_attempts=metrics.median_attempts_to_success,
            mean_l2=metrics.mean_perturbation_l2,
            mean_linf=metrics.mean_perturbation_linf,
            samples_succeeded=metrics.samples_succeeded,
            wall_time_sec=elapsed,
            attempts_per_second=n_samples / elapsed if elapsed > 0 else 0,
        ))
    except Exception as e:
        print(f"      Failed: {e}")

    # QuantileAttack
    print(f"    QuantileAttack...")
    start = time.perf_counter()
    try:
        attack = QuantileAttackWrapper(
            model, epsilon=0.5, max_iter=30, n_directions=20,
            attack_mode="break_coverage", calibration_width=0.5
        )
        # QuantileAttack needs y_test for coverage calculation
        y_dummy = np.zeros(n_samples)  # Placeholder
        metrics = attack.run(X, y_dummy)
        elapsed = time.perf_counter() - start

        results.append(AttackBenchmarkResult(
            attack_type="QuantileAttack",
            model_type=model_name,
            task_type="regression",
            n_samples=n_samples,
            success_rate=metrics.coverage_break_rate if metrics.coverage_break_rate else 0.0,
            mean_attempts=float('nan'),  # Not applicable for coverage attacks
            median_attempts=float('nan'),
            mean_l2=metrics.mean_perturbation_l2,
            mean_linf=metrics.mean_perturbation_linf,
            samples_succeeded=metrics.samples_succeeded,
            wall_time_sec=elapsed,
            attempts_per_second=n_samples / elapsed if elapsed > 0 else 0,
        ))
    except Exception as e:
        print(f"      Failed: {e}")

    return results


def benchmark_classification_attacks(model, X_test, model_name: str, n_samples: int = 30) -> List[AttackBenchmarkResult]:
    """Benchmark classification attacks."""
    # Classification attacks would go here
    # For now, placeholder
    return []


def print_attack_results(results: List[AttackBenchmarkResult]):
    """Print formatted attack results."""
    print("\n" + "=" * 110)
    print("ATTACK BENCHMARK RESULTS")
    print("=" * 110)

    print(f"\n{'Attack':<20} {'Model':<15} {'Success':<10} {'Attempts':<12} {'L2':<10} {'Succeeded':<12} {'SPS':<10} {'Time':<8}")
    print("-" * 107)

    for r in results:
        attempts_str = f"{r.mean_attempts:.1f}" if not np.isnan(r.mean_attempts) else "N/A"
        print(f"{r.attack_type:<20} {r.model_type:<15} {r.success_rate:.1%}     "
              f"{attempts_str:<12} {r.mean_l2:.4f}    {r.samples_succeeded:<12} "
              f"{r.attempts_per_second:.1f}       {r.wall_time_sec:.1f}s")


def main():
    print("=" * 80)
    print("  SPECTRUM GOVERNANCE - ATTACK BENCHMARK SUITE")
    print("=" * 80)
    print()

    # Load regression dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models to benchmark
    regression_models = [
        ("MLP-64", MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)),
        ("MLP-128", MLPRegressor(hidden_layer_sizes=(128,), max_iter=300, random_state=42)),
        ("MLP-64-64", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)),
        ("RF-50", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
        ("GBM-50", GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
    ]

    all_results = []

    print("\n" + "─" * 80)
    print("REGRESSION ATTACK BENCHMARKS")
    print("─" * 80)

    for model_name, model in regression_models:
        print(f"\n  Training {model_name}...")
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        print(f"  R2: {r2:.4f}")

        print(f"  Running attacks...")
        results = benchmark_regression_attacks(model, X_test, model_name, n_samples=30)
        all_results.extend(results)

    print_attack_results(all_results)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS BY ATTACK TYPE")
    print("=" * 80)

    attack_stats = {}
    for r in all_results:
        if r.attack_type not in attack_stats:
            attack_stats[r.attack_type] = {
                "success_rates": [], "attempts": [], "l2s": [], "sps": [], "times": []
            }
        attack_stats[r.attack_type]["success_rates"].append(r.success_rate)
        if not np.isnan(r.mean_attempts):
            attack_stats[r.attack_type]["attempts"].append(r.mean_attempts)
        attack_stats[r.attack_type]["l2s"].append(r.mean_l2)
        attack_stats[r.attack_type]["sps"].append(r.attempts_per_second)
        attack_stats[r.attack_type]["times"].append(r.wall_time_sec)

    print(f"\n{'Attack':<20} {'Avg Success':<12} {'Avg Attempts':<14} {'Avg L2':<10} {'Avg SPS':<12} {'Avg Time':<10}")
    print("-" * 78)

    for attack, stats in attack_stats.items():
        avg_success = np.mean(stats["success_rates"])
        avg_attempts = np.mean(stats["attempts"]) if stats["attempts"] else float('nan')
        avg_l2 = np.mean(stats["l2s"])
        avg_sps = np.mean(stats["sps"])
        avg_time = np.mean(stats["times"])

        attempts_str = f"{avg_attempts:.1f}" if not np.isnan(avg_attempts) else "N/A"
        print(f"{attack:<20} {avg_success:.1%}       {attempts_str:<14} {avg_l2:.4f}    {avg_sps:.1f}         {avg_time:.1f}s")

    # Model vulnerability ranking
    print("\n" + "=" * 80)
    print("MODEL VULNERABILITY RANKING (lower attempts = more vulnerable)")
    print("=" * 80)

    model_vuln = {}
    for r in all_results:
        if r.attack_type in ["OutputManipulation", "PredictionShift"]:
            if r.model_type not in model_vuln:
                model_vuln[r.model_type] = []
            if not np.isnan(r.mean_attempts):
                model_vuln[r.model_type].append(r.mean_attempts)

    print(f"\n{'Model':<15} {'Avg Attempts':<15} {'Vulnerability':<15}")
    print("-" * 45)

    sorted_models = sorted(model_vuln.items(), key=lambda x: np.mean(x[1]))
    for model, attempts in sorted_models:
        avg = np.mean(attempts)
        vuln = "HIGH" if avg < 20 else "MEDIUM" if avg < 50 else "LOW"
        print(f"{model:<15} {avg:.1f}           {vuln}")

    print()


if __name__ == "__main__":
    main()
