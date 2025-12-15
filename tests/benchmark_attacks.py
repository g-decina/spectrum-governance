"""
Performance Benchmarking Suite for Adversarial Attacks
========================================================

Compares performance across backends:
- ART (1 core) - Baseline
- Rust (1 core) - Single-threaded Rust
- Rust (16 cores) - Parallel Rust

Tests all 4 attacks:
- HopSkipJump
- ZOO
- Boundary
- Square

Metrics tracked:
- Wall-clock time
- Queries used
- Success rate
- L2/L∞ perturbation norms
- Memory usage

Usage:
    pytest tests/benchmark_attacks.py -v -s
    python tests/benchmark_attacks.py --save-results results.json
"""

import os
import sys
import time
import json
import psutil
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import attack wrappers
from spectrum.red.attack import (
    HopSkipJumpWrapper,
    ZooAttackWrapper,
    BoundaryAttackWrapper,
    SquareAttackWrapper,
    RUST_AVAILABLE,
    ART_AVAILABLE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence ART's verbose logging (it spams "Found initial adversarial image" messages)
logging.getLogger("art").setLevel(logging.WARNING)
logging.getLogger("art.attacks").setLevel(logging.WARNING)
logging.getLogger("art.attacks.evasion").setLevel(logging.WARNING)
logging.getLogger("art.attacks.evasion.hop_skip_jump").setLevel(logging.ERROR)


# =============================================================================
# Terminal Colors - Professional palette
# =============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = "\033[0m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLUE = "\033[38;5;69m"      # Soft blue for headers
    CYAN = "\033[38;5;80m"      # Cyan for info
    GREEN = "\033[38;5;114m"    # Soft green for success/good values
    YELLOW = "\033[38;5;221m"   # Yellow for warnings
    RED = "\033[38;5;203m"      # Soft red for errors/bad values
    GRAY = "\033[38;5;245m"     # Gray for secondary info
    WHITE = "\033[38;5;255m"    # Bright white for emphasis
    ORANGE = "\033[38;5;215m"   # Orange for Rust-related items

    @classmethod
    def header(cls, text: str) -> str:
        """Format header text."""
        return f"{cls.BOLD}{cls.BLUE}{text}{cls.RESET}"

    @classmethod
    def success(cls, text: str) -> str:
        """Format success text."""
        return f"{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def warning(cls, text: str) -> str:
        """Format warning text."""
        return f"{cls.YELLOW}{text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        """Format error text."""
        return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        """Format info text."""
        return f"{cls.CYAN}{text}{cls.RESET}"

    @classmethod
    def dim(cls, text: str) -> str:
        """Format dimmed text."""
        return f"{cls.DIM}{cls.GRAY}{text}{cls.RESET}"

    @classmethod
    def rust(cls, text: str) -> str:
        """Format Rust-related text (orange like the Rust logo)."""
        return f"{cls.ORANGE}{text}{cls.RESET}"

    @classmethod
    def speedup(cls, value: float) -> str:
        """Color-code speedup values."""
        text = f"{value:.2f}x"
        if value >= 5.0:
            return f"{cls.BOLD}{cls.GREEN}{text}{cls.RESET}"
        elif value >= 2.0:
            return f"{cls.GREEN}{text}{cls.RESET}"
        elif value >= 1.0:
            return f"{cls.CYAN}{text}{cls.RESET}"
        else:
            return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def time(cls, seconds: float) -> str:
        """Color-code time values (green=fast, red=slow)."""
        text = f"{seconds:.2f}s"
        if seconds < 1.0:
            return f"{cls.GREEN}{text}{cls.RESET}"
        elif seconds < 10.0:
            return f"{cls.CYAN}{text}{cls.RESET}"
        elif seconds < 60.0:
            return f"{cls.YELLOW}{text}{cls.RESET}"
        else:
            return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def percentage(cls, value: float) -> str:
        """Color-code percentage values (success rates)."""
        text = f"{value:.1%}"
        if value >= 0.9:
            return f"{cls.GREEN}{text}{cls.RESET}"
        elif value >= 0.7:
            return f"{cls.CYAN}{text}{cls.RESET}"
        elif value >= 0.5:
            return f"{cls.YELLOW}{text}{cls.RESET}"
        else:
            return f"{cls.RED}{text}{cls.RESET}"

    @staticmethod
    def pad(text: str, width: int) -> str:
        """Pad a string with ANSI codes to a fixed visible width."""
        import re
        # Strip ANSI codes to get visible length
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        visible_text = ansi_escape.sub('', text)
        visible_len = len(visible_text)
        # Add padding spaces
        padding = max(0, width - visible_len)
        return text + ' ' * padding


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    attack_name: str
    backend: str
    n_cores: int
    onnx_enabled: bool
    wall_time: float
    success_rate: float
    queries_used: int
    avg_queries_per_sample: float
    l2_perturbation: float
    linf_perturbation: float
    memory_mb: float
    samples_tested: int
    samples_successful: int

    @property
    def queries_per_second(self) -> float:
        """Compute queries per second."""
        return self.queries_used / self.wall_time if self.wall_time > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['queries_per_second'] = self.queries_per_second
        return result


@dataclass
class BenchmarkComparison:
    """Comparison between backends for a single attack."""
    attack_name: str
    art_1core: BenchmarkResult
    art_1core_onnx: Optional[BenchmarkResult]
    rust_1core: Optional[BenchmarkResult]
    rust_1core_onnx: Optional[BenchmarkResult]
    rust_16core: Optional[BenchmarkResult]
    rust_16core_onnx: Optional[BenchmarkResult]

    @property
    def onnx_speedup_art(self) -> Optional[float]:
        """Speedup from ONNX on ART backend."""
        if self.art_1core_onnx:
            return self.art_1core.wall_time / self.art_1core_onnx.wall_time
        return None

    @property
    def onnx_speedup_rust(self) -> Optional[float]:
        """Speedup from ONNX on Rust backend (1 core)."""
        if self.rust_1core and self.rust_1core_onnx:
            return self.rust_1core.wall_time / self.rust_1core_onnx.wall_time
        return None

    @property
    def rust_1core_speedup(self) -> Optional[float]:
        """Speedup of Rust (1 core) vs ART (both without ONNX)."""
        if self.rust_1core:
            return self.art_1core.wall_time / self.rust_1core.wall_time
        return None

    @property
    def rust_16core_speedup(self) -> Optional[float]:
        """Speedup of Rust (16 cores) vs ART (both with ONNX)."""
        if self.rust_16core_onnx and self.art_1core_onnx:
            return self.art_1core_onnx.wall_time / self.rust_16core_onnx.wall_time
        return None

    @property
    def best_speedup(self) -> Optional[float]:
        """Best speedup: Rust (16 cores) + ONNX vs ART (1 core) no ONNX."""
        if self.rust_16core_onnx:
            return self.art_1core.wall_time / self.rust_16core_onnx.wall_time
        return None

    @property
    def parallel_efficiency(self) -> Optional[float]:
        """Parallel efficiency (speedup / cores)."""
        if self.rust_16core_speedup:
            return self.rust_16core_speedup / 16.0
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'attack_name': self.attack_name,
            'art_1core': self.art_1core.to_dict(),
            'art_1core_onnx': self.art_1core_onnx.to_dict() if self.art_1core_onnx else None,
            'rust_1core': self.rust_1core.to_dict() if self.rust_1core else None,
            'rust_1core_onnx': self.rust_1core_onnx.to_dict() if self.rust_1core_onnx else None,
            'rust_16core': self.rust_16core.to_dict() if self.rust_16core else None,
            'rust_16core_onnx': self.rust_16core_onnx.to_dict() if self.rust_16core_onnx else None,
            'onnx_speedup_art': self.onnx_speedup_art,
            'onnx_speedup_rust': self.onnx_speedup_rust,
            'rust_1core_speedup': self.rust_1core_speedup,
            'rust_16core_speedup': self.rust_16core_speedup,
            'best_speedup': self.best_speedup,
            'parallel_efficiency': self.parallel_efficiency,
        }


class BenchmarkDataset:
    """Standard dataset for benchmarking attacks."""

    def __init__(self, n_features: int = 20, n_samples: int = 100, n_test: int = 10):
        """
        Create synthetic dataset for benchmarking.

        Args:
            n_features: Number of features (default: 20)
            n_samples: Total samples for training (default: 100)
            n_test: Number of test samples to attack (default: 10)
        """
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_test = n_test

        # Generate synthetic data with clear decision boundary
        # Use sklearn's make_classification to ensure balanced classes
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples + n_test,
            n_features=n_features,
            n_informative=min(n_features, 10),  # At most 10 informative features
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.0,  # No label noise
            class_sep=1.0,  # Clear separation
            random_state=42
        )

        # Split train/test
        self.X_train = X[:n_samples]
        self.y_train = y[:n_samples]
        self.X_test = X[n_samples:n_samples + n_test]
        self.y_test = y[n_samples:n_samples + n_test]

        logger.info(f"Dataset: {n_features} features, {n_samples} train, {n_test} test")

    def get_trained_model(self) -> LogisticRegression:
        """Train and return logistic regression model."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)

        # Filter test set to only include correctly classified samples
        # (attacks should only be run on samples the model gets right)
        y_pred = model.predict(self.X_test)
        correct_indices = (y_pred == self.y_test)

        self.X_test = self.X_test[correct_indices]
        self.y_test = self.y_test[correct_indices]

        if len(self.X_test) == 0:
            raise ValueError("Model has 0% accuracy on test set - cannot benchmark attacks")

        accuracy = len(self.X_test) / (self.n_test) * 100
        logger.info(f"Model accuracy: {accuracy:.1%} ({len(self.X_test)}/{self.n_test} samples correctly classified)")
        logger.info(f"Using {len(self.X_test)} correctly classified samples for attack benchmarking")

        return model


class AttackBenchmark:
    """Benchmark runner for adversarial attacks."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model: Any,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ):
        """
        Initialize benchmark runner.

        Args:
            dataset: Dataset to use for benchmarking
            model: Trained model to attack
            warmup_runs: Number of warmup runs (default: 1)
            benchmark_runs: Number of benchmark runs for averaging (default: 3)
        """
        self.dataset = dataset
        self.model = model
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def run_attack(
        self,
        attack_class: type,
        backend: str,
        config: Dict[str, Any],
        parallel: Optional[bool] = None,
        enable_onnx: bool = True,
    ) -> BenchmarkResult:
        """
        Run a single attack benchmark.

        Args:
            attack_class: Attack class (e.g., HopSkipJumpWrapper)
            backend: Backend to use ("art" or "rust")
            config: Attack configuration parameters
            parallel: Whether to enable parallel execution (for Rust)
            enable_onnx: Whether to enable ONNX conversion

        Returns:
            BenchmarkResult with performance metrics
        """
        attack_name = attack_class.__name__.replace('Wrapper', '')

        # Determine n_cores
        if backend == "rust" and parallel:
            n_cores = os.cpu_count() or 16
        else:
            n_cores = 1

        # Create attack instance
        attack = attack_class(
            base_model=self.model,
            backend=backend,
            enable_onnx=enable_onnx,
            **config
        )

        # Warmup runs
        logger.info(f"Warmup: {attack_name} ({backend}, {n_cores} cores)")
        for i in range(self.warmup_runs):
            _ = attack.run(self.dataset.X_test[:2])  # Small sample

        # Benchmark runs
        logger.info(f"Benchmarking: {attack_name} ({backend}, {n_cores} cores)")
        times = []
        memory_usages = []
        metrics_list = []

        for run in range(self.benchmark_runs):
            # Get memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run attack
            start_time = time.time()
            metrics = attack.run(self.dataset.X_test)
            wall_time = time.time() - start_time

            # Get memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = mem_after - mem_before

            times.append(wall_time)
            memory_usages.append(memory_used)
            metrics_list.append(metrics)

            logger.info(f"  Run {run + 1}/{self.benchmark_runs}: {wall_time:.2f}s")

        # Average results
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_memory = np.mean(memory_usages)

        # Use metrics from last run (should be consistent)
        metrics = metrics_list[-1]

        logger.info(f"  Result: {avg_time:.2f}s ± {std_time:.2f}s")

        return BenchmarkResult(
            attack_name=attack_name,
            backend=backend,
            n_cores=n_cores,
            onnx_enabled=enable_onnx,
            wall_time=avg_time,
            success_rate=metrics.attack_success_rate,
            queries_used=metrics.queries_used,
            avg_queries_per_sample=metrics.avg_queries_per_sample,
            l2_perturbation=metrics.empirical_robustness_l2,
            linf_perturbation=metrics.empirical_robustness_linf,
            memory_mb=avg_memory,
            samples_tested=metrics.samples_tested,
            samples_successful=metrics.samples_successful,
        )

    def benchmark_attack(
        self,
        attack_class: type,
        config: Dict[str, Any],
    ) -> BenchmarkComparison:
        """
        Benchmark an attack across all backends and ONNX configurations.

        Tests 6 configurations:
        1. ART (1 core) - no ONNX
        2. ART (1 core) - with ONNX
        3. Rust (1 core) - no ONNX
        4. Rust (1 core) - with ONNX
        5. Rust (16 cores) - no ONNX
        6. Rust (16 cores) - with ONNX

        Args:
            attack_class: Attack class to benchmark
            config: Attack configuration

        Returns:
            BenchmarkComparison with results from all configurations
        """
        attack_name = attack_class.__name__.replace('Wrapper', '')
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking: {attack_name}")
        logger.info(f"{'=' * 60}")

        # Always run ART (baseline)
        if not ART_AVAILABLE:
            raise RuntimeError("ART not available, cannot establish baseline")

        # ART (1 core) - no ONNX
        logger.info("Running: ART (1 core, no ONNX)")
        art_result = self.run_attack(
            attack_class,
            backend="art",
            config=config,
            enable_onnx=False
        )

        # ART (1 core) - with ONNX
        logger.info("Running: ART (1 core, with ONNX)")
        art_onnx_result = self.run_attack(
            attack_class,
            backend="art",
            config=config,
            enable_onnx=True
        )

        # Run Rust if available
        rust_1core_result = None
        rust_1core_onnx_result = None
        rust_16core_result = None
        rust_16core_onnx_result = None

        if RUST_AVAILABLE:
            # Rust (1 core) - no ONNX
            logger.info("Running: Rust (1 core, no ONNX)")
            rust_1core_result = self.run_attack(
                attack_class,
                backend="rust",
                config=config,
                parallel=False,
                enable_onnx=False
            )

            # Rust (1 core) - with ONNX
            logger.info("Running: Rust (1 core, with ONNX)")
            rust_1core_onnx_result = self.run_attack(
                attack_class,
                backend="rust",
                config=config,
                parallel=False,
                enable_onnx=True
            )

            # Rust (16 cores) - no ONNX
            logger.info("Running: Rust (16 cores, no ONNX)")
            rust_16core_result = self.run_attack(
                attack_class,
                backend="rust",
                config=config,
                parallel=True,
                enable_onnx=False
            )

            # Rust (16 cores) - with ONNX
            logger.info("Running: Rust (16 cores, with ONNX)")
            rust_16core_onnx_result = self.run_attack(
                attack_class,
                backend="rust",
                config=config,
                parallel=True,
                enable_onnx=True
            )
        else:
            logger.warning("Rust backend not available, skipping Rust benchmarks")

        return BenchmarkComparison(
            attack_name=attack_name,
            art_1core=art_result,
            art_1core_onnx=art_onnx_result,
            rust_1core=rust_1core_result,
            rust_1core_onnx=rust_1core_onnx_result,
            rust_16core=rust_16core_result,
            rust_16core_onnx=rust_16core_onnx_result,
        )


class BenchmarkSuite:
    """Complete benchmark suite for all attacks."""

    # Fast configurations for quick smoke tests (NOT for real benchmarking)
    FAST_CONFIGS = {
        HopSkipJumpWrapper: {
            'max_iter': 5,
            'max_eval': 500,
            'init_eval': 50,
            'init_size': 1000,
        },
        ZooAttackWrapper: {
            'max_iter': 50,
            'learning_rate': 0.01,
            'epsilon': 0.3,
            'batch_size': 32,
            'nb_parallel': 5,
        },
        BoundaryAttackWrapper: {
            'max_iter': 100,
            'delta': 0.01,
            'epsilon': 0.01,
            'init_size': 100,
        },
    }

    # Benchmark configurations for meaningful performance comparison (~30s+ per attack)
    # These are tuned to give statistically meaningful results on 50+ test samples
    BENCHMARK_CONFIGS = {
        HopSkipJumpWrapper: {
            'max_iter': 64,          # Full boundary refinement (ART default)
            'max_eval': 10000,       # Full gradient estimation budget
            'init_eval': 100,
            'init_size': 1000,
        },
        ZooAttackWrapper: {
            'max_iter': 1000,        # Enough iterations for convergence
            'learning_rate': 0.01,
            'epsilon': 0.3,
            'batch_size': 128,
            'nb_parallel': 16,
        },
        BoundaryAttackWrapper: {
            'max_iter': 5000,        # Full boundary walking
            'delta': 0.01,
            'epsilon': 0.01,
            'init_size': 100,
        },
    }

    # Production configurations (thorough, for audit reports)
    PRODUCTION_CONFIGS = {
        HopSkipJumpWrapper: {
            'max_iter': 64,
            'max_eval': 10000,
            'init_eval': 100,
            'init_size': 1000,
        },
        ZooAttackWrapper: {
            'max_iter': 1000,
            'learning_rate': 0.01,
            'epsilon': 0.3,
            'batch_size': 128,
            'nb_parallel': 16,
        },
        BoundaryAttackWrapper: {
            'max_iter': 5000,
            'delta': 0.01,
            'epsilon': 0.01,
            'init_size': 100,
        },
    }

    def __init__(
        self,
        n_features: int = 20,
        n_samples: int = 100,
        n_test: int = 10,
        config_mode: str = "benchmark",
    ):
        """
        Initialize benchmark suite.

        Args:
            n_features: Number of features in dataset
            n_samples: Number of training samples
            n_test: Number of test samples to attack
            config_mode: "fast" (smoke test), "benchmark" (meaningful 30s+), "production" (thorough)
        """
        self.dataset = BenchmarkDataset(n_features, n_samples, n_test)
        self.model = self.dataset.get_trained_model()

        if config_mode == "fast":
            self.configs = self.FAST_CONFIGS
        elif config_mode == "benchmark":
            self.configs = self.BENCHMARK_CONFIGS
        elif config_mode == "production":
            self.configs = self.PRODUCTION_CONFIGS
        else:
            raise ValueError(f"Unknown config_mode: {config_mode}. Use 'fast', 'benchmark', or 'production'")

        self.benchmark_runner = AttackBenchmark(
            dataset=self.dataset,
            model=self.model,
            warmup_runs=1,
            benchmark_runs=3,
        )

    def run_all(self) -> List[BenchmarkComparison]:
        """
        Run all benchmarks.

        Returns:
            List of BenchmarkComparison objects
        """
        results = []

        for attack_class, config in self.configs.items():
            try:
                comparison = self.benchmark_runner.benchmark_attack(attack_class, config)
                results.append(comparison)
            except Exception as e:
                logger.error(f"Benchmark failed for {attack_class.__name__}: {e}")

        return results

    def print_summary(self, results: List[BenchmarkComparison]):
        """Print comprehensive summary table including ONNX comparisons."""
        C = Colors  # Shorthand

        # Column widths
        W_NAME = 16
        W_TIME = 10
        W_GAIN = 8

        # Header
        print()
        print(C.header("═" * 110))
        print(C.header("  BENCHMARK SUMMARY"))
        print(C.header("═" * 110))

        # Column headers - use pad() for colored strings
        print(
            f"{C.pad(f'{C.BOLD}{C.WHITE}Attack{C.RESET}', W_NAME)}"
            f"{C.pad(f'{C.CYAN}ART{C.RESET}', W_TIME)}"
            f"{C.pad(f'{C.CYAN}ART+ONNX{C.RESET}', W_TIME)}"
            f"{C.pad(C.rust('Rust'), W_TIME)}"
            f"{C.pad(C.rust('Rust+ONNX'), W_TIME)}"
            f"{C.pad(f'{C.GREEN}ONNX{C.RESET}', W_GAIN)}"
            f"{C.pad(f'{C.GREEN}Rust{C.RESET}', W_GAIN)}"
            f"{C.pad(f'{C.BOLD}{C.GREEN}Best{C.RESET}', W_GAIN)}"
            f"{C.DIM}Success{C.RESET}"
        )
        # Subheaders
        print(
            f"{'':<{W_NAME}}"
            f"{C.pad(C.dim('(1c)'), W_TIME)}"
            f"{C.pad(C.dim('(1c)'), W_TIME)}"
            f"{C.pad(C.dim('(16c)'), W_TIME)}"
            f"{C.pad(C.dim('(16c)'), W_TIME)}"
            f"{C.pad(C.dim('Gain'), W_GAIN)}"
            f"{C.pad(C.dim('Gain'), W_GAIN)}"
            f"{C.pad(C.dim('Total'), W_GAIN)}"
            f"{C.dim('Rate')}"
        )
        print(C.dim("─" * 110))

        for comp in results:
            # Format times with colors
            art_time = C.time(comp.art_1core.wall_time)
            art_onnx_time = C.time(comp.art_1core_onnx.wall_time) if comp.art_1core_onnx else C.dim("N/A")
            rust_16c_time = C.time(comp.rust_16core.wall_time) if comp.rust_16core else C.dim("N/A")
            rust_16c_onnx_time = C.time(comp.rust_16core_onnx.wall_time) if comp.rust_16core_onnx else C.dim("N/A")

            # Format speedups with colors
            onnx_gain = C.speedup(comp.onnx_speedup_art) if comp.onnx_speedup_art else C.dim("N/A")
            rust_gain = C.speedup(comp.rust_1core_speedup) if comp.rust_1core_speedup else C.dim("N/A")
            best_speedup = C.speedup(comp.best_speedup) if comp.best_speedup else C.dim("N/A")

            # Success rate
            if comp.rust_16core_onnx:
                success = C.percentage(comp.rust_16core_onnx.success_rate)
            else:
                success = C.percentage(comp.art_1core.success_rate)

            # Print row using pad() for proper alignment
            print(
                f"{C.pad(f'{C.BOLD}{C.WHITE}{comp.attack_name}{C.RESET}', W_NAME)}"
                f"{C.pad(art_time, W_TIME)}"
                f"{C.pad(art_onnx_time, W_TIME)}"
                f"{C.pad(rust_16c_time, W_TIME)}"
                f"{C.pad(rust_16c_onnx_time, W_TIME)}"
                f"{C.pad(onnx_gain, W_GAIN)}"
                f"{C.pad(rust_gain, W_GAIN)}"
                f"{C.pad(best_speedup, W_GAIN)}"
                f"{success}"
            )

        print(C.header("═" * 110))

        # Print detailed metrics
        print()
        print(C.header("═" * 100))
        print(C.header("  DETAILED METRICS"))
        print(C.header("═" * 100))

        for comp in results:
            print(f"\n{C.BOLD}{C.WHITE}{comp.attack_name}{C.RESET}")

            # Success rates
            art_rate = C.percentage(comp.art_1core.success_rate)
            print(f"  {C.dim('Success Rate:')} ART={art_rate}", end="")
            if comp.art_1core_onnx:
                print(f", ART+ONNX={C.percentage(comp.art_1core_onnx.success_rate)}", end="")
            if comp.rust_16core_onnx:
                print(f", {C.rust('Rust+ONNX')}={C.percentage(comp.rust_16core_onnx.success_rate)}")
            else:
                print()

            # Queries
            print(f"  {C.dim('Queries:')} ART={C.info(f'{comp.art_1core.queries_used:,}')}", end="")
            if comp.rust_16core_onnx:
                print(f", {C.rust('Rust+ONNX')}={C.info(f'{comp.rust_16core_onnx.queries_used:,}')}")
            else:
                print()

            # L2 Perturbation
            print(f"  {C.dim('L2 Perturbation:')} ART={comp.art_1core.l2_perturbation:.4f}", end="")
            if comp.rust_16core_onnx:
                print(f", {C.rust('Rust+ONNX')}={comp.rust_16core_onnx.l2_perturbation:.4f}")
            else:
                print()

            # Memory
            print(f"  {C.dim('Memory:')} ART={comp.art_1core.memory_mb:.1f}MB", end="")
            if comp.rust_16core_onnx:
                print(f", {C.rust('Rust+ONNX')}={comp.rust_16core_onnx.memory_mb:.1f}MB")
            else:
                print()

            # Optimization breakdown
            print(f"  {C.dim('Optimization Breakdown:')}")
            if comp.onnx_speedup_art:
                print(f"    {C.info('ONNX:')} {C.speedup(comp.onnx_speedup_art)} speedup on ART")
            if comp.rust_1core_speedup:
                print(f"    {C.rust('Rust:')} {C.speedup(comp.rust_1core_speedup)} speedup (1 core)")
            if comp.best_speedup:
                print(f"    {C.BOLD}{C.GREEN}Combined:{C.RESET} {C.speedup(comp.best_speedup)} total speedup (Rust 16c + ONNX)")

    def save_results(self, results: List[BenchmarkComparison], output_path: str):
        """Save results to JSON file."""
        output = {
            'dataset': {
                'n_features': self.dataset.n_features,
                'n_samples': self.dataset.n_samples,
                'n_test': self.dataset.n_test,
            },
            'system': {
                'cpu_count': os.cpu_count(),
                'rust_available': RUST_AVAILABLE,
                'art_available': ART_AVAILABLE,
            },
            'results': [comp.to_dict() for comp in results],
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark adversarial attacks")
    parser.add_argument(
        '--features',
        type=int,
        default=20,
        help='Number of features in dataset (default: 20)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of training samples (default: 100)'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=50,
        help='Number of test samples to attack (default: 50 for meaningful benchmarks)'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=['fast', 'benchmark', 'production'],
        default='benchmark',
        help='Configuration mode: fast (smoke test), benchmark (30s+, default), production (thorough)'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        help='Path to save JSON results'
    )

    args = parser.parse_args()

    # Handle legacy --production flag

    # Run benchmark suite
    suite = BenchmarkSuite(
        n_features=args.features,
        n_samples=args.samples,
        n_test=args.test,
        config_mode=args.config,
    )

    config_labels = {
        'fast': 'Fast (smoke test)',
        'benchmark': 'Benchmark (30s+ per attack)',
        'production': 'Production (thorough)'
    }

    C = Colors  # Shorthand

    # Professional header
    print()
    print(C.header("═" * 100))
    print(C.header("  SPECTRUM-RED ATTACK PERFORMANCE BENCHMARK"))
    print(C.header("═" * 100))
    print()
    print(f"  {C.dim('Dataset:')}    {C.WHITE}{args.features}{C.RESET} features, {C.WHITE}{args.samples}{C.RESET} train, {C.WHITE}{args.test}{C.RESET} test")
    print(f"  {C.dim('Config:')}     {C.info(config_labels[args.config])}")

    # Backend availability with colored checkmarks
    art_status = f"{C.GREEN}✓{C.RESET}" if ART_AVAILABLE else f"{C.RED}✗{C.RESET}"
    rust_status = f"{C.GREEN}✓{C.RESET}" if RUST_AVAILABLE else f"{C.RED}✗{C.RESET}"
    print(f"  {C.dim('Backends:')}   {C.CYAN}ART{C.RESET}={art_status}  {C.rust('Rust')}={rust_status}")
    print()
    print(C.dim("─" * 100))

    results = suite.run_all()
    suite.print_summary(results)

    if args.save_results:
        suite.save_results(results, args.save_results)


if __name__ == '__main__':
    main()
