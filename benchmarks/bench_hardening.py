#!/usr/bin/env python3
"""
Benchmark Suite: Model Hardening

Measures performance and effectiveness of hardening wrappers across
different model types and dataset sizes.

Metrics:
- Robustness improvement (OutputManipulation, PredictionShift)
- Accuracy cost (R2 drop)
- Latency overhead
- Memory footprint
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
import tracemalloc

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from spectrum.blue.harden import (
    InputSanitizer, EnsembleProxy, OutputQuantizer,
    PredictionSmoother, RandomizedWrapper, AdversarialTrainer,
    harden_model
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_type: str
    wrapper_type: str
    dataset: str
    n_samples: int
    n_features: int
    base_r2: float
    hardened_r2: float
    r2_drop_pp: float
    om_attempts_base: float
    om_attempts_hardened: float
    om_improvement_pct: float
    ps_attempts_base: float
    ps_attempts_hardened: float
    ps_improvement_pct: float
    latency_base_ms: float
    latency_hardened_ms: float
    latency_overhead_pct: float
    memory_overhead_mb: float
    calibration_time_ms: float


def measure_latency(model, X, n_runs=50):
    """Measure prediction latency in milliseconds."""
    # Warmup
    _ = model.predict(X[:10])

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000  # ms per prediction batch


def measure_memory(model, X):
    """Measure memory usage during prediction in MB."""
    tracemalloc.start()
    _ = model.predict(X)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024  # MB


def run_attacks(model, X, n_samples=30):
    """Run OutputManipulation and PredictionShift attacks."""
    try:
        from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper

        X_attack = X[:n_samples]

        # OutputManipulation
        om = OutputManipulationWrapper(
            model, n_bins=5, max_iter=20, max_eval=1000,
            parallel=True, confidence_level=0.99
        )
        om_metrics = om.run(X_attack)
        om_attempts = om_metrics.mean_attempts_to_success

        # PredictionShift
        ps = PredictionShiftWrapper(
            model, epsilon=1.0, max_iter=30, n_directions=20,
            success_threshold=0.10, confidence_level=0.99
        )
        ps_metrics = ps.run(X_attack)
        ps_attempts = ps_metrics.mean_attempts_to_success

        return om_attempts, ps_attempts
    except Exception as e:
        print(f"    Attack failed: {e}")
        return float('nan'), float('nan')


def benchmark_wrapper(
    base_model,
    wrapper_class,
    X_train, y_train,
    X_calib, X_test, y_test,
    model_name: str,
    dataset_name: str,
    attack_samples: int = 30,
    **wrapper_kwargs
) -> BenchmarkResult:
    """Benchmark a single wrapper configuration."""

    # Train base model
    base_model.fit(X_train, y_train)
    base_r2 = r2_score(y_test, base_model.predict(X_test))

    # Create and calibrate wrapper
    start_calib = time.perf_counter()
    if wrapper_class is None:
        wrapper = base_model
        wrapper_name = "Baseline"
    else:
        wrapper = wrapper_class(base_model, **wrapper_kwargs)
        wrapper_name = wrapper_class.__name__
        if hasattr(wrapper, 'calibrate'):
            wrapper.calibrate(X_calib)
    calib_time = (time.perf_counter() - start_calib) * 1000

    # Measure hardened accuracy
    hardened_r2 = r2_score(y_test, wrapper.predict(X_test))
    r2_drop = (base_r2 - hardened_r2) * 100

    # Measure latency
    latency_base = measure_latency(base_model, X_test[:100])
    latency_hardened = measure_latency(wrapper, X_test[:100])
    latency_overhead = ((latency_hardened / latency_base) - 1) * 100

    # Measure memory
    mem_base = measure_memory(base_model, X_test[:100])
    mem_hardened = measure_memory(wrapper, X_test[:100])
    mem_overhead = mem_hardened - mem_base

    # Run attacks
    print(f"    Running attacks on {wrapper_name}...")
    om_base, ps_base = run_attacks(base_model, X_test, attack_samples)
    om_hardened, ps_hardened = run_attacks(wrapper, X_test, attack_samples)

    om_improvement = ((om_hardened / om_base) - 1) * 100 if om_base > 0 else 0
    ps_improvement = ((ps_hardened / ps_base) - 1) * 100 if ps_base > 0 else 0

    return BenchmarkResult(
        model_type=model_name,
        wrapper_type=wrapper_name,
        dataset=dataset_name,
        n_samples=len(X_train),
        n_features=X_train.shape[1],
        base_r2=base_r2,
        hardened_r2=hardened_r2,
        r2_drop_pp=r2_drop,
        om_attempts_base=om_base,
        om_attempts_hardened=om_hardened,
        om_improvement_pct=om_improvement,
        ps_attempts_base=ps_base,
        ps_attempts_hardened=ps_hardened,
        ps_improvement_pct=ps_improvement,
        latency_base_ms=latency_base,
        latency_hardened_ms=latency_hardened,
        latency_overhead_pct=latency_overhead,
        memory_overhead_mb=mem_overhead,
        calibration_time_ms=calib_time,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    print(f"\n{'Model':<15} {'Wrapper':<20} {'R2 Drop':<10} {'OM Δ':<10} {'PS Δ':<10} {'Latency':<12} {'Memory':<10}")
    print("-" * 97)

    for r in results:
        om_str = f"{r.om_improvement_pct:+.0f}%" if not np.isnan(r.om_improvement_pct) else "N/A"
        ps_str = f"{r.ps_improvement_pct:+.0f}%" if not np.isnan(r.ps_improvement_pct) else "N/A"

        print(f"{r.model_type:<15} {r.wrapper_type:<20} {r.r2_drop_pp:+.2f}pp    "
              f"{om_str:<10} {ps_str:<10} {r.latency_overhead_pct:+.0f}%        "
              f"{r.memory_overhead_mb:+.2f}MB")


def main():
    print("=" * 80)
    print("  SPECTRUM GOVERNANCE - HARDENING BENCHMARK SUITE")
    print("=" * 80)
    print()

    # Load datasets
    print("Loading datasets...")

    # California Housing (larger)
    housing = fetch_california_housing()
    X_housing, y_housing = housing.data, housing.target

    # Diabetes (smaller, different characteristics)
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target

    datasets = [
        ("California Housing", X_housing, y_housing),
        ("Diabetes", X_diabetes, y_diabetes),
    ]

    # Model architectures to test
    models = [
        ("MLP", lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)),
        ("RandomForest", lambda: RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
        ("GBM", lambda: GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
        ("Ridge", lambda: Ridge(alpha=1.0)),
    ]

    # Wrappers to benchmark
    wrappers = [
        (None, {}),  # Baseline
        (InputSanitizer, {"clip_percentile": 99, "squeeze_bits": 6}),
        (EnsembleProxy, {"n_neighbors": 10}),
        (OutputQuantizer, {"n_levels": 20}),
        (PredictionSmoother, {"n_samples": 5, "noise_std": 0.05}),
        (RandomizedWrapper, {"input_noise_std": 0.02}),
    ]

    all_results = []

    for dataset_name, X, y in datasets:
        print(f"\n{'─' * 80}")
        print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
        print(f"{'─' * 80}")

        # Prepare data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_calib = scaler.transform(X_calib)
        X_test = scaler.transform(X_test)

        for model_name, model_factory in models:
            print(f"\n  Model: {model_name}")

            for wrapper_class, wrapper_kwargs in wrappers:
                wrapper_name = wrapper_class.__name__ if wrapper_class else "Baseline"
                print(f"    Wrapper: {wrapper_name}")

                try:
                    result = benchmark_wrapper(
                        base_model=model_factory(),
                        wrapper_class=wrapper_class,
                        X_train=X_train, y_train=y_train,
                        X_calib=X_calib, X_test=X_test, y_test=y_test,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        attack_samples=20,
                        **wrapper_kwargs
                    )
                    all_results.append(result)

                    print(f"      R2: {result.hardened_r2:.4f} (drop: {result.r2_drop_pp:+.2f}pp)")
                    print(f"      OM: {result.om_improvement_pct:+.0f}%, PS: {result.ps_improvement_pct:+.0f}%")

                except Exception as e:
                    print(f"      ERROR: {e}")

    # Print summary
    print_results_table(all_results)

    # Aggregate by wrapper type
    print("\n" + "=" * 80)
    print("AGGREGATE BY WRAPPER (across all models and datasets)")
    print("=" * 80)

    wrapper_stats = {}
    for r in all_results:
        if r.wrapper_type not in wrapper_stats:
            wrapper_stats[r.wrapper_type] = {
                "r2_drops": [], "om_improvements": [], "ps_improvements": [],
                "latency_overheads": []
            }
        wrapper_stats[r.wrapper_type]["r2_drops"].append(r.r2_drop_pp)
        if not np.isnan(r.om_improvement_pct):
            wrapper_stats[r.wrapper_type]["om_improvements"].append(r.om_improvement_pct)
        if not np.isnan(r.ps_improvement_pct):
            wrapper_stats[r.wrapper_type]["ps_improvements"].append(r.ps_improvement_pct)
        wrapper_stats[r.wrapper_type]["latency_overheads"].append(r.latency_overhead_pct)

    print(f"\n{'Wrapper':<20} {'Avg R2 Drop':<12} {'Avg OM Δ':<12} {'Avg PS Δ':<12} {'Avg Latency':<12}")
    print("-" * 68)

    for wrapper, stats in wrapper_stats.items():
        avg_r2 = np.mean(stats["r2_drops"])
        avg_om = np.mean(stats["om_improvements"]) if stats["om_improvements"] else float('nan')
        avg_ps = np.mean(stats["ps_improvements"]) if stats["ps_improvements"] else float('nan')
        avg_lat = np.mean(stats["latency_overheads"])

        om_str = f"{avg_om:+.0f}%" if not np.isnan(avg_om) else "N/A"
        ps_str = f"{avg_ps:+.0f}%" if not np.isnan(avg_ps) else "N/A"

        print(f"{wrapper:<20} {avg_r2:+.2f}pp      {om_str:<12} {ps_str:<12} {avg_lat:+.0f}%")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find best wrapper
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

    print(f"\n  Best Robustness:  {best_robustness[0]}")
    print(f"  Best Efficiency:  {best_efficiency[0]}")
    print()


if __name__ == "__main__":
    main()
