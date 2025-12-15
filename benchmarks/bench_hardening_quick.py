#!/usr/bin/env python3
"""
Quick Hardening Benchmark - Runs faster subset of model/wrapper combos.

For full benchmarks, use bench_hardening.py
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

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from spectrum.blue.harden import (
    InputSanitizer, EnsembleProxy, OutputQuantizer,
    PredictionSmoother, RandomizedWrapper
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_type: str
    wrapper_type: str
    base_r2: float
    hardened_r2: float
    r2_drop_pp: float
    om_attempts_base: float
    om_attempts_hardened: float
    om_improvement_pct: float
    ps_attempts_base: float
    ps_attempts_hardened: float
    ps_improvement_pct: float
    latency_overhead_pct: float


def measure_latency(model, X, n_runs=20):
    """Measure prediction latency in milliseconds."""
    _ = model.predict(X[:10])  # Warmup
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000


def run_attacks(model, X, n_samples=20):
    """Run attacks and return mean attempts."""
    try:
        from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper

        X_attack = X[:n_samples]

        om = OutputManipulationWrapper(
            model, n_bins=5, max_iter=20, max_eval=1000,
            parallel=True, confidence_level=0.99
        )
        om_metrics = om.run(X_attack)
        om_attempts = om_metrics.mean_attempts_to_success

        ps = PredictionShiftWrapper(
            model, epsilon=1.0, max_iter=30, n_directions=20,
            success_threshold=0.10, confidence_level=0.99
        )
        ps_metrics = ps.run(X_attack)
        ps_attempts = ps_metrics.mean_attempts_to_success

        return om_attempts, ps_attempts
    except Exception as e:
        print(f"      Attack error: {e}")
        return float('nan'), float('nan')


def benchmark_wrapper(
    base_model, wrapper_class, X_train, y_train, X_calib, X_test, y_test,
    model_name: str, attack_samples: int = 20, **wrapper_kwargs
) -> BenchmarkResult:
    """Benchmark a single wrapper configuration."""

    # Train base model
    base_model.fit(X_train, y_train)
    base_r2 = r2_score(y_test, base_model.predict(X_test))

    # Create wrapper
    if wrapper_class is None:
        wrapper = base_model
        wrapper_name = "Baseline"
    else:
        wrapper = wrapper_class(base_model, **wrapper_kwargs)
        wrapper_name = wrapper_class.__name__
        if hasattr(wrapper, 'calibrate'):
            wrapper.calibrate(X_calib)

    hardened_r2 = r2_score(y_test, wrapper.predict(X_test))
    r2_drop = (base_r2 - hardened_r2) * 100

    # Measure latency
    latency_base = measure_latency(base_model, X_test[:100])
    latency_hardened = measure_latency(wrapper, X_test[:100])
    latency_overhead = ((latency_hardened / latency_base) - 1) * 100

    # Run attacks
    print(f"    Running attacks on {wrapper_name}...")
    om_base, ps_base = run_attacks(base_model, X_test, attack_samples)
    om_hardened, ps_hardened = run_attacks(wrapper, X_test, attack_samples)

    om_improvement = ((om_hardened / om_base) - 1) * 100 if om_base > 0 else 0
    ps_improvement = ((ps_hardened / ps_base) - 1) * 100 if ps_base > 0 else 0

    return BenchmarkResult(
        model_type=model_name,
        wrapper_type=wrapper_name,
        base_r2=base_r2,
        hardened_r2=hardened_r2,
        r2_drop_pp=r2_drop,
        om_attempts_base=om_base,
        om_attempts_hardened=om_hardened,
        om_improvement_pct=om_improvement,
        ps_attempts_base=ps_base,
        ps_attempts_hardened=ps_hardened,
        ps_improvement_pct=ps_improvement,
        latency_overhead_pct=latency_overhead,
    )


def main():
    print("=" * 80)
    print("  SPECTRUM GOVERNANCE - QUICK HARDENING BENCHMARK")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    print(f"Data: {len(X_train)} train, {len(X_calib)} calibration, {len(X_test)} test\n")

    # Quick model set (skip RF which is slow)
    models = [
        ("MLP-64", lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)),
        ("GBM-50", lambda: GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
    ]

    # Key wrappers to benchmark
    wrappers = [
        (None, {}),  # Baseline
        (EnsembleProxy, {"n_neighbors": 10}),
        (OutputQuantizer, {"n_levels": 20}),
        (InputSanitizer, {"clip_percentile": 99, "squeeze_bits": 6}),
    ]

    all_results = []

    for model_name, model_factory in models:
        print(f"{'─' * 80}")
        print(f"Model: {model_name}")
        print(f"{'─' * 80}")

        for wrapper_class, wrapper_kwargs in wrappers:
            wrapper_name = wrapper_class.__name__ if wrapper_class else "Baseline"
            print(f"\n  Wrapper: {wrapper_name}")

            try:
                result = benchmark_wrapper(
                    base_model=model_factory(),
                    wrapper_class=wrapper_class,
                    X_train=X_train, y_train=y_train,
                    X_calib=X_calib, X_test=X_test, y_test=y_test,
                    model_name=model_name,
                    attack_samples=20,
                    **wrapper_kwargs
                )
                all_results.append(result)

                print(f"      R2: {result.hardened_r2:.4f} (drop: {result.r2_drop_pp:+.2f}pp)")
                om_str = f"{result.om_improvement_pct:+.0f}%" if not np.isnan(result.om_improvement_pct) else "N/A"
                ps_str = f"{result.ps_improvement_pct:+.0f}%" if not np.isnan(result.ps_improvement_pct) else "N/A"
                print(f"      OM: {om_str}, PS: {ps_str}, Latency: {result.latency_overhead_pct:+.0f}%")

            except Exception as e:
                print(f"      ERROR: {e}")

    # Print summary
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    print(f"\n{'Model':<12} {'Wrapper':<20} {'R2 Drop':<10} {'OM Δ':<12} {'PS Δ':<12} {'Latency':<12}")
    print("-" * 78)

    for r in all_results:
        om_str = f"{r.om_improvement_pct:+.0f}%" if not np.isnan(r.om_improvement_pct) else "N/A"
        ps_str = f"{r.ps_improvement_pct:+.0f}%" if not np.isnan(r.ps_improvement_pct) else "N/A"
        print(f"{r.model_type:<12} {r.wrapper_type:<20} {r.r2_drop_pp:+.2f}pp    "
              f"{om_str:<12} {ps_str:<12} {r.latency_overhead_pct:+.0f}%")

    # Aggregate by wrapper
    print("\n" + "=" * 80)
    print("AGGREGATE BY WRAPPER")
    print("=" * 80)

    wrapper_stats = {}
    for r in all_results:
        if r.wrapper_type not in wrapper_stats:
            wrapper_stats[r.wrapper_type] = {
                "r2_drops": [], "om_improvements": [], "ps_improvements": [], "latency_overheads": []
            }
        wrapper_stats[r.wrapper_type]["r2_drops"].append(r.r2_drop_pp)
        if not np.isnan(r.om_improvement_pct):
            wrapper_stats[r.wrapper_type]["om_improvements"].append(r.om_improvement_pct)
        if not np.isnan(r.ps_improvement_pct):
            wrapper_stats[r.wrapper_type]["ps_improvements"].append(r.ps_improvement_pct)
        wrapper_stats[r.wrapper_type]["latency_overheads"].append(r.latency_overhead_pct)

    print(f"\n{'Wrapper':<20} {'Avg R2 Drop':<14} {'Avg OM Δ':<14} {'Avg PS Δ':<14} {'Avg Latency':<14}")
    print("-" * 76)

    for wrapper, stats in wrapper_stats.items():
        avg_r2 = np.mean(stats["r2_drops"])
        avg_om = np.mean(stats["om_improvements"]) if stats["om_improvements"] else float('nan')
        avg_ps = np.mean(stats["ps_improvements"]) if stats["ps_improvements"] else float('nan')
        avg_lat = np.mean(stats["latency_overheads"])

        om_str = f"{avg_om:+.0f}%" if not np.isnan(avg_om) else "N/A"
        ps_str = f"{avg_ps:+.0f}%" if not np.isnan(avg_ps) else "N/A"
        print(f"{wrapper:<20} {avg_r2:+.2f}pp        {om_str:<14} {ps_str:<14} {avg_lat:+.0f}%")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_robustness = max(
        [(w, np.mean(s["om_improvements"]) + np.mean(s["ps_improvements"]))
         for w, s in wrapper_stats.items()
         if w != "Baseline" and s["om_improvements"] and s["ps_improvements"]],
        key=lambda x: x[1]
    )

    best_efficiency = min(
        [(w, np.mean(s["latency_overheads"]))
         for w, s in wrapper_stats.items()
         if w != "Baseline"],
        key=lambda x: x[1]
    )

    print(f"\n  Best Robustness:  {best_robustness[0]} ({best_robustness[1]:+.0f}% combined improvement)")
    print(f"  Best Efficiency:  {best_efficiency[0]} ({best_efficiency[1]:+.0f}% latency overhead)")
    print()


if __name__ == "__main__":
    main()
