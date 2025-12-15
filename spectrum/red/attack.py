import numpy as np
import logging
from typing import Optional

from art.estimators.classification import SklearnClassifier
from sklearn.base import BaseEstimator
from tqdm import tqdm

# ANSI colors for progress bars
class _Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    ORANGE = "\033[38;5;215m"  # Rust orange
    CYAN = "\033[38;5;80m"     # Info
    GREEN = "\033[38;5;114m"   # Success
    DIM = "\033[2m"
    GRAY = "\033[38;5;245m"

# Colored progress bar format for Rust attacks
RUST_BAR_FORMAT = (
    f"{_Colors.ORANGE}{{desc}}{_Colors.RESET} "
    f"{_Colors.DIM}|{_Colors.RESET}{_Colors.ORANGE}{{bar}}{_Colors.RESET}{_Colors.DIM}|{_Colors.RESET} "
    f"{_Colors.CYAN}{{n_fmt}}/{{total_fmt}}{_Colors.RESET} "
    f"{_Colors.DIM}[{{elapsed}}<{{remaining}}]{_Colors.RESET}"
)

from spectrum.red.scenario import AttackScenario
from spectrum.red.metrics import AdversarialMetrics
from spectrum.red.model_adapter import ModelAdapter
from spectrum.utils.threading import _with_limited_threads
from spectrum.infra.types import RiskProfile

logger = logging.getLogger(__name__)

"""
spectrum.red.attack
===================

Adversarial Engine for Model Risk Quantification with Unified Rust/ART Backend.

This module provides wrappers for adversarial attacks with automatic backend selection:
- **Rust backend** (15x faster, parallel): Used when spectrum_red is installed
- **ART backend** (fallback): Used when Rust unavailable

CORE ARCHITECTURE:
------------------
1. **Lazy Model Binding**: Model stored in constructor, converted to ONNX on first run()
2. **Automatic Backend Selection**:
   - backend="auto" → Prefer Rust if available, fallback to ART
   - backend="rust" → Force Rust (raises error if unavailable)
   - backend="art" → Force ART
3. **ONNX Optimization**: ModelAdapter automatically converts sklearn/pytorch/tf to ONNX
4. **Unified API**: Same interface regardless of backend

ADVERSARIAL METRICS:
--------------------
All attacks return comprehensive AdversarialMetrics containing:
- Attack success rate (primary robustness metric)
- Perturbation magnitudes (L2, L∞ norms)
- Query efficiency metrics
- Statistical confidence intervals

KEY COMPONENTS:
---------------
- HopSkipJumpWrapper: Black-box, decision-based evasion attack
- ZooAttackWrapper: Black-box, zeroth-order optimization attack
- BoundaryAttackWrapper: Decision-based boundary walking attack
- SquareAttackWrapper: Query-efficient score-based attack

USAGE FLOW:
-----------
1. Initialize with model:
    `hsj = HopSkipJumpWrapper(base_model=my_model, backend="auto")`
2. Execute attack:
    `metrics = hsj.run(X_test_data)`
3. Metrics captured by Wargame Runner and logged as evidence

PERFORMANCE:
------------
Rust backend provides ~15x speedup with parallel execution:
- HopSkipJump: 120s (ART) → 8s (Rust, 16 cores)
- ZOO: 180s (ART) → 12s (Rust, 16 cores)
"""

# Try to import Rust implementations
try:
    from spectrum_red_core import HopSkipJump as RustHopSkipJump
    from spectrum_red_core import ZOO as RustZOO
    from spectrum_red_core import Boundary as RustBoundary
    from spectrum_red_core import Square as RustSquare
    # Regression attacks
    from spectrum_red_core import OutputManipulation as RustOutputManipulation
    from spectrum_red_core import PredictionShift as RustPredictionShift
    from spectrum_red_core import QuantileAttack as RustQuantileAttack
    RUST_AVAILABLE = True
    RUST_REGRESSION_AVAILABLE = True
    logger.info("✓ Rust backend available with ONNX support for optimal performance")
except ImportError:
    RUST_AVAILABLE = False
    RUST_REGRESSION_AVAILABLE = False
    logger.info("ℹ Rust attacks not installed, using ART backend (install with: maturin develop --features python)")

# Try to import ART
try:
    from art.attacks.evasion import HopSkipJump as ARTHopSkipJump
    from art.attacks.evasion import ZooAttack as ARTZooAttack
    from art.attacks.evasion import BoundaryAttack as ARTBoundaryAttack
    from art.attacks.evasion import SquareAttack as ARTSquareAttack
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    logger.warning("⚠ ART not installed, adversarial attacks unavailable")


class HopSkipJumpWrapper(AttackScenario):
    """
    Black-box decision-based attack using HopSkipJump algorithm.

    Finds minimum perturbation to flip model predictions using only
    hard labels (no probability scores needed).

    Backend options:
    - Rust: 15x faster, parallel execution, ONNX-optimized
    - ART: Fallback, single-threaded

    READ: https://arxiv.org/pdf/1904.02144
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        max_iter: int = 64,
        max_eval: int = 1000,
        init_eval: int = 25,
        init_size: int = 100,
        backend: str = "auto",
        enable_onnx: bool = True,
    ):
        """
        Initialize HopSkipJump attack with unified API.

        Args:
            base_model: Model to attack (sklearn, pytorch, tensorflow, or any with .predict())
            max_iter: Maximum boundary refinement iterations (default: 64)
            max_eval: Query budget for gradient estimation per iteration (default: 1000)
            init_eval: Base queries for gradient estimation, scales with sqrt(iter) (default: 25)
            init_size: Number of random initializations (default: 100)
            backend: "auto" (prefer Rust), "rust" (force Rust), "art" (force ART)
            enable_onnx: Enable automatic ONNX conversion (default: True)
        """
        self.base_model = base_model
        self.config = {
            "max_iter": max_iter,
            "max_eval": max_eval,
            "init_eval": init_eval,
            "init_size": init_size,
        }
        self.enable_onnx = enable_onnx

        # Select backend
        self.backend = self._select_backend(backend)
        logger.info(f"HopSkipJump using backend: {self.backend}")

        # Lazy initialization
        self._model_adapter: Optional[ModelAdapter] = None
        self._rust_attack = None
        self._art_attack = None
        self._art_estimator = None

    def _select_backend(self, requested: str) -> str:
        """Select backend based on availability and user preference."""
        if requested == "rust":
            if not RUST_AVAILABLE:
                raise RuntimeError("Rust backend requested but spectrum_red not installed")
            return "rust"
        elif requested == "art":
            if not ART_AVAILABLE:
                raise RuntimeError("ART backend requested but adversarial-robustness-toolbox not installed")
            return "art"
        elif requested == "auto":
            # Prefer Rust if available
            if RUST_AVAILABLE:
                return "rust"
            elif ART_AVAILABLE:
                return "art"
            else:
                raise RuntimeError("No attack backend available (install spectrum_red or adversarial-robustness-toolbox)")
        else:
            raise ValueError(f"Unknown backend: {requested}. Use 'auto', 'rust', or 'art'")

    def _get_model_adapter(self) -> ModelAdapter:
        """Lazy initialization of model adapter with ONNX conversion."""
        if self._model_adapter is None:
            self._model_adapter = ModelAdapter(self.base_model, enable_onnx=self.enable_onnx)
            logger.info(f"Model adapter initialized (ONNX: {self.enable_onnx})")
        return self._model_adapter

    def _adapt_to_art(self, model_adapter: ModelAdapter) -> SklearnClassifier:
        """Convert model adapter to ART-compatible estimator."""
        if self._art_estimator is None:
            # Wrap the adapter in ART's SklearnClassifier
            # Must inherit from BaseEstimator for ART compatibility
            from sklearn.base import BaseEstimator as SklearnBaseEstimator

            class AdapterWrapper(SklearnBaseEstimator):
                def __init__(self, adapter):
                    self.adapter = adapter
                    # Infer classes from original model
                    self.classes_ = getattr(adapter.original_model, 'classes_', np.array([0, 1]))
                    # Infer n_features_in_ for ART input shape detection
                    self.n_features_in_ = getattr(adapter.original_model, 'n_features_in_', None)

                def predict_proba(self, X):
                    return self.adapter.predict(X)

                def predict(self, X):
                    proba = self.predict_proba(X)
                    return np.argmax(proba, axis=1)

            # Trick ART into accepting our wrapper by setting module to sklearn
            AdapterWrapper.__module__ = "sklearn.base"
            wrapper = AdapterWrapper(model_adapter)
            self._art_estimator = SklearnClassifier(
                model=wrapper,
                clip_values=(0, 1),
                preprocessing_defences=[]
            )

        return self._art_estimator

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """
        Execute HopSkipJump attack with automatic backend selection.

        Args:
            X_input: Input samples to attack (2D array)

        Returns:
            AdversarialMetrics with attack results
        """
        # Get model adapter (with lazy ONNX conversion)
        model_adapter = self._get_model_adapter()

        if self.backend == "rust":
            return self._run_rust(X_input, model_adapter)
        else:
            return self._run_art(X_input, model_adapter)

    def _run_rust(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute attack using Rust backend."""
        # Lazy initialization of Rust attack
        if self._rust_attack is None:
            # Get model metadata
            n_features = getattr(self.base_model, 'n_features_in_', 0)
            n_classes = len(getattr(self.base_model, 'classes_', [0, 1]))

            # Try to use ONNX model for 15x speedup, fallback to Python model
            onnx_bytes = model_adapter.onnx_bytes
            if onnx_bytes:
                logger.info(f"✓ Using ONNX model ({len(onnx_bytes)} bytes) for Rust backend (15x speedup!)")
                self._rust_attack = RustHopSkipJump(
                    onnx_bytes=onnx_bytes,
                    n_features=n_features,
                    n_classes=n_classes,
                    max_iter=self.config["max_iter"],
                    max_eval=self.config["max_eval"],
                    init_eval=self.config["init_eval"],
                    init_size=self.config["init_size"],
                    parallel=True,
                )
            else:
                logger.warning(
                    "⚠ ONNX conversion failed, using Python model (16x SLOWER due to GIL)\n"
                    "   Install skl2onnx for optimal performance: pip install skl2onnx"
                )
                self._rust_attack = RustHopSkipJump(
                    python_model=model_adapter.original_model,
                    max_iter=self.config["max_iter"],
                    max_eval=self.config["max_eval"],
                    init_eval=self.config["init_eval"],
                    init_size=self.config["init_size"],
                    parallel=True,
                )
            logger.debug("Rust HopSkipJump attack initialized")

        # Process in mini-batches to balance parallelism with progress feedback
        # Rust uses Rayon for parallel processing within each batch
        n_samples = X_input.shape[0]
        batch_size = max(1, min(16, n_samples))  # 16 samples per batch for good parallelism
        n_batches = (n_samples + batch_size - 1) // batch_size

        X_adv_list = []
        total_queries = 0

        with tqdm(total=n_samples, desc="HopSkipJump", bar_format=RUST_BAR_FORMAT, leave=True) as pbar:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = X_input[start:end]

                x_adv_batch, metrics = self._rust_attack.run(batch)
                X_adv_list.append(x_adv_batch)
                total_queries += metrics.queries_used
                pbar.update(end - start)

        X_adv = np.vstack(X_adv_list)

        # Compute aggregate metrics
        return AdversarialMetrics.for_evasion_attack(
            attack_type="HopSkipJump (Rust)",
            y_original=model_adapter.predict(X_input).argmax(axis=1),
            y_adversarial=model_adapter.predict(X_adv).argmax(axis=1),
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=total_queries,
        )

    def _run_art(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute attack using ART backend."""
        # Lazy initialization of ART attack
        if self._art_attack is None:
            art_estimator = self._adapt_to_art(model_adapter)
            self._art_attack = ARTHopSkipJump(
                classifier=art_estimator,
                targeted=False,
                max_iter=self.config["max_iter"],
                max_eval=self.config["max_eval"],
                init_size=self.config["init_size"],
            )
            logger.debug("ART HopSkipJump attack initialized")

        # Get original predictions
        y_pred_original = model_adapter.predict(X_input)
        y_label_original = np.argmax(y_pred_original, axis=1)

        # Run attack
        X_adv = self._art_attack.generate(x=X_input)

        # Get adversarial predictions
        y_pred_adv = model_adapter.predict(X_adv)
        y_label_adv = np.argmax(y_pred_adv, axis=1)

        # Compute metrics
        return AdversarialMetrics.for_evasion_attack(
            attack_type="HopSkipJump (ART)",
            y_original=y_label_original,
            y_adversarial=y_label_adv,
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=self.config["max_eval"] * len(X_input),  # Approximate
        )


class ZooAttackWrapper(AttackScenario):
    """
    Black-box score-based attack using zeroth-order optimization.

    Estimates gradients using finite differences and Adam optimizer.
    Requires model probability scores.

    READ: https://arxiv.org/abs/1708.03999
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        epsilon: float = 0.3,
        batch_size: int = 128,
        nb_parallel: int = 128,
        backend: str = "auto",
        enable_onnx: bool = True,
        risk_profile: Optional[RiskProfile] = None,
    ):
        """Initialize ZOO attack with unified API."""
        self.base_model = base_model
        self.config = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "nb_parallel": nb_parallel,
        }
        self.enable_onnx = enable_onnx
        self.risk_profile = risk_profile

        # Select backend
        self.backend = self._select_backend(backend)
        logger.info(f"ZOO using backend: {self.backend}")

        # Lazy initialization
        self._model_adapter: Optional[ModelAdapter] = None
        self._rust_attack = None
        self._art_attack = None
        self._art_estimator = None

    def _select_backend(self, requested: str) -> str:
        """Select backend based on availability."""
        if requested == "rust":
            if not RUST_AVAILABLE:
                raise RuntimeError("Rust backend requested but spectrum_red not installed")
            return "rust"
        elif requested == "art":
            if not ART_AVAILABLE:
                raise RuntimeError("ART backend requested but adversarial-robustness-toolbox not installed")
            return "art"
        elif requested == "auto":
            # Prefer ART until ONNX-in-Rust is implemented (Rust is currently slower!)
            return "art" if ART_AVAILABLE else "rust" if RUST_AVAILABLE else None
        else:
            raise ValueError(f"Unknown backend: {requested}")

    def _get_model_adapter(self) -> ModelAdapter:
        """Lazy initialization of model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelAdapter(self.base_model, enable_onnx=self.enable_onnx)
        return self._model_adapter

    def _adapt_to_art(self, model_adapter: ModelAdapter) -> SklearnClassifier:
        """Convert model adapter to ART estimator."""
        if self._art_estimator is None:
            from sklearn.base import BaseEstimator as SklearnBaseEstimator

            class AdapterWrapper(SklearnBaseEstimator):
                def __init__(self, adapter):
                    self.adapter = adapter
                    self.classes_ = getattr(adapter.original_model, 'classes_', np.array([0, 1]))
                    self.n_features_in_ = getattr(adapter.original_model, 'n_features_in_', None)

                def predict_proba(self, X):
                    return self.adapter.predict(X)

                def predict(self, X):
                    return np.argmax(self.predict_proba(X), axis=1)

            # Trick ART into accepting our wrapper by setting module to sklearn
            AdapterWrapper.__module__ = "sklearn.base"
            wrapper = AdapterWrapper(model_adapter)
            self._art_estimator = SklearnClassifier(
                model=wrapper,
                clip_values=(0, 1),
                preprocessing_defences=[]
            )
        return self._art_estimator

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """Execute ZOO attack."""
        model_adapter = self._get_model_adapter()

        if self.backend == "rust":
            return self._run_rust(X_input, model_adapter)
        else:
            return self._run_art(X_input, model_adapter)

    def _run_rust(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using Rust backend."""
        if self._rust_attack is None:
            # Get model metadata
            n_features = getattr(self.base_model, 'n_features_in_', 0)
            n_classes = len(getattr(self.base_model, 'classes_', [0, 1]))

            # Try to use ONNX model for 15x speedup, fallback to Python model
            onnx_bytes = model_adapter.onnx_bytes
            if onnx_bytes:
                logger.info(f"✓ Using ONNX model ({len(onnx_bytes)} bytes) for Rust backend")
                self._rust_attack = RustZOO(
                    onnx_bytes=onnx_bytes,
                    n_features=n_features,
                    n_classes=n_classes,
                    max_iter=self.config["max_iter"],
                    learning_rate=self.config["learning_rate"],
                    epsilon=self.config["epsilon"],
                    batch_size=self.config["batch_size"],
                    parallel=True,
                )
            else:
                logger.warning("⚠ ONNX conversion failed, using Python model (16x SLOWER due to GIL)")
                self._rust_attack = RustZOO(
                    python_model=model_adapter.original_model,
                    max_iter=self.config["max_iter"],
                    learning_rate=self.config["learning_rate"],
                    epsilon=self.config["epsilon"],
                    batch_size=self.config["batch_size"],
                    parallel=True,
                )

        # Process in mini-batches to balance parallelism with progress feedback
        n_samples = X_input.shape[0]
        batch_size = max(1, min(16, n_samples))
        n_batches = (n_samples + batch_size - 1) // batch_size

        X_adv_list = []
        total_queries = 0

        with tqdm(total=n_samples, desc="ZOO", bar_format=RUST_BAR_FORMAT, leave=True) as pbar:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = X_input[start:end]

                x_adv_batch, metrics = self._rust_attack.run(batch)
                X_adv_list.append(x_adv_batch)
                total_queries += metrics.queries_used
                pbar.update(end - start)

        X_adv = np.vstack(X_adv_list)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="ZOO (Rust)",
            y_original=model_adapter.predict(X_input).argmax(axis=1),
            y_adversarial=model_adapter.predict(X_adv).argmax(axis=1),
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=total_queries,
        )

    def _run_art(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using ART backend."""
        if self._art_attack is None:
            art_estimator = self._adapt_to_art(model_adapter)
            confidence = self.risk_profile.alpha if self.risk_profile else 0.0
            self._art_attack = ARTZooAttack(
                classifier=art_estimator,
                targeted=False,
                max_iter=self.config["max_iter"],
                learning_rate=self.config["learning_rate"],
                confidence=confidence,
                nb_parallel=self.config["nb_parallel"],
            )

        y_pred_original = model_adapter.predict(X_input)
        y_label_original = np.argmax(y_pred_original, axis=1)

        X_adv = self._art_attack.generate(x=X_input)

        y_pred_adv = model_adapter.predict(X_adv)
        y_label_adv = np.argmax(y_pred_adv, axis=1)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="ZOO (ART)",
            y_original=y_label_original,
            y_adversarial=y_label_adv,
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=self.config["max_iter"] * self.config["batch_size"] * 2 * len(X_input),
        )


class BoundaryAttackWrapper(AttackScenario):
    """
    Decision-based boundary walking attack.

    Starts from large perturbation and walks along decision boundary
    to find minimal adversarial example.

    READ: https://arxiv.org/abs/1712.04248
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        max_iter: int = 5000,
        delta: float = 0.01,
        epsilon: float = 0.01,
        init_size: int = 100,
        backend: str = "auto",
        enable_onnx: bool = True,
        risk_profile: Optional[RiskProfile] = None,
    ):
        """Initialize Boundary attack."""
        self.base_model = base_model
        self.config = {
            "max_iter": max_iter,
            "delta": delta,
            "epsilon": epsilon,
            "init_size": init_size,
        }
        self.enable_onnx = enable_onnx
        self.risk_profile = risk_profile
        self.backend = self._select_backend(backend)
        logger.info(f"Boundary using backend: {self.backend}")

        self._model_adapter: Optional[ModelAdapter] = None
        self._rust_attack = None
        self._art_attack = None
        self._art_estimator = None

    def _select_backend(self, requested: str) -> str:
        """Select backend."""
        if requested == "rust":
            if not RUST_AVAILABLE:
                raise RuntimeError("Rust backend not available")
            return "rust"
        elif requested == "art":
            if not ART_AVAILABLE:
                raise RuntimeError("ART backend not available")
            return "art"
        elif requested == "auto":
            # Prefer ART until ONNX-in-Rust is implemented (Rust is currently slower!)
            return "art" if ART_AVAILABLE else "rust" if RUST_AVAILABLE else None
        raise ValueError(f"Unknown backend: {requested}")

    def _get_model_adapter(self) -> ModelAdapter:
        """Lazy initialization."""
        if self._model_adapter is None:
            self._model_adapter = ModelAdapter(self.base_model, enable_onnx=self.enable_onnx)
        return self._model_adapter

    def _adapt_to_art(self, model_adapter: ModelAdapter) -> SklearnClassifier:
        """Convert to ART estimator."""
        if self._art_estimator is None:
            from sklearn.base import BaseEstimator as SklearnBaseEstimator

            class AdapterWrapper(SklearnBaseEstimator):
                def __init__(self, adapter):
                    self.adapter = adapter
                    self.classes_ = getattr(adapter.original_model, 'classes_', np.array([0, 1]))
                    self.n_features_in_ = getattr(adapter.original_model, 'n_features_in_', None)

                def predict_proba(self, X):
                    return self.adapter.predict(X)

                def predict(self, X):
                    return np.argmax(self.predict_proba(X), axis=1)

            # Trick ART into accepting our wrapper by setting module to sklearn
            AdapterWrapper.__module__ = "sklearn.base"
            wrapper = AdapterWrapper(model_adapter)
            self._art_estimator = SklearnClassifier(
                model=wrapper,
                clip_values=(0, 1),
                preprocessing_defences=[]
            )
        return self._art_estimator

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """Execute Boundary attack."""
        model_adapter = self._get_model_adapter()

        if self.backend == "rust":
            return self._run_rust(X_input, model_adapter)
        else:
            return self._run_art(X_input, model_adapter)

    def _run_rust(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using Rust."""
        if self._rust_attack is None:
            # Get model metadata
            n_features = getattr(self.base_model, 'n_features_in_', 0)
            n_classes = len(getattr(self.base_model, 'classes_', [0, 1]))

            # Try to use ONNX model for 15x speedup, fallback to Python model
            onnx_bytes = model_adapter.onnx_bytes
            if onnx_bytes:
                logger.info(f"✓ Using ONNX model ({len(onnx_bytes)} bytes) for Rust backend")
                self._rust_attack = RustBoundary(
                    onnx_bytes=onnx_bytes,
                    n_features=n_features,
                    n_classes=n_classes,
                    max_iter=self.config["max_iter"],
                    delta=self.config["delta"],
                    epsilon=self.config["epsilon"],
                    init_size=self.config["init_size"],
                    parallel=True,
                )
            else:
                logger.warning("⚠ ONNX conversion failed, using Python model (slower)")
                self._rust_attack = RustBoundary(
                    python_model=model_adapter.original_model,
                    max_iter=self.config["max_iter"],
                    delta=self.config["delta"],
                    epsilon=self.config["epsilon"],
                    init_size=self.config["init_size"],
                    parallel=True,
                )

        # Process in mini-batches to balance parallelism with progress feedback
        n_samples = X_input.shape[0]
        batch_size = max(1, min(16, n_samples))
        n_batches = (n_samples + batch_size - 1) // batch_size

        X_adv_list = []
        total_queries = 0

        with tqdm(total=n_samples, desc="Boundary", bar_format=RUST_BAR_FORMAT, leave=True) as pbar:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = X_input[start:end]

                x_adv_batch, metrics = self._rust_attack.run(batch)
                X_adv_list.append(x_adv_batch)
                total_queries += metrics.queries_used
                pbar.update(end - start)

        X_adv = np.vstack(X_adv_list)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="Boundary (Rust)",
            y_original=model_adapter.predict(X_input).argmax(axis=1),
            y_adversarial=model_adapter.predict(X_adv).argmax(axis=1),
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=total_queries,
        )

    def _run_art(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using ART."""
        if self._art_attack is None:
            art_estimator = self._adapt_to_art(model_adapter)
            self._art_attack = ARTBoundaryAttack(
                estimator=art_estimator,
                targeted=False,
                max_iter=self.config["max_iter"],
                delta=self.config["delta"],
                epsilon=self.config["epsilon"],
            )

        y_pred_original = model_adapter.predict(X_input)
        y_label_original = np.argmax(y_pred_original, axis=1)

        X_adv = self._art_attack.generate(x=X_input)

        y_pred_adv = model_adapter.predict(X_adv)
        y_label_adv = np.argmax(y_pred_adv, axis=1)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="Boundary (ART)",
            y_original=y_label_original,
            y_adversarial=y_label_adv,
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=self.config["max_iter"] * len(X_input),
        )


class SquareAttackWrapper(AttackScenario):
    """
    Query-efficient score-based attack using random search.

    Particularly effective for high-dimensional inputs and images.
    Requires model probability scores.

    READ: https://arxiv.org/abs/1912.00049
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        max_iter: int = 10000,
        epsilon: float = 0.05,
        p_init: float = 0.8,
        n_restarts: int = 100,
        backend: str = "auto",
        enable_onnx: bool = True,
        risk_profile: Optional[RiskProfile] = None,
    ):
        """Initialize Square attack."""
        self.base_model = base_model
        self.config = {
            "max_iter": max_iter,
            "epsilon": epsilon,
            "p_init": p_init,
            "n_restarts": n_restarts,
        }
        self.enable_onnx = enable_onnx
        self.risk_profile = risk_profile
        self.backend = self._select_backend(backend)
        logger.info(f"Square using backend: {self.backend}")

        self._model_adapter: Optional[ModelAdapter] = None
        self._rust_attack = None
        self._art_attack = None
        self._art_estimator = None

    def _select_backend(self, requested: str) -> str:
        """Select backend."""
        if requested == "rust":
            if not RUST_AVAILABLE:
                raise RuntimeError("Rust backend not available")
            return "rust"
        elif requested == "art":
            if not ART_AVAILABLE:
                raise RuntimeError("ART backend not available")
            return "art"
        elif requested == "auto":
            # Prefer ART until ONNX-in-Rust is implemented (Rust is currently slower!)
            return "art" if ART_AVAILABLE else "rust" if RUST_AVAILABLE else None
        raise ValueError(f"Unknown backend: {requested}")

    def _get_model_adapter(self) -> ModelAdapter:
        """Lazy initialization."""
        if self._model_adapter is None:
            self._model_adapter = ModelAdapter(self.base_model, enable_onnx=self.enable_onnx)
        return self._model_adapter

    def _adapt_to_art(self, model_adapter: ModelAdapter) -> SklearnClassifier:
        """Convert to ART estimator."""
        if self._art_estimator is None:
            from sklearn.base import BaseEstimator as SklearnBaseEstimator

            class AdapterWrapper(SklearnBaseEstimator):
                def __init__(self, adapter):
                    self.adapter = adapter
                    self.classes_ = getattr(adapter.original_model, 'classes_', np.array([0, 1]))
                    self.n_features_in_ = getattr(adapter.original_model, 'n_features_in_', None)

                def predict_proba(self, X):
                    return self.adapter.predict(X)

                def predict(self, X):
                    return np.argmax(self.predict_proba(X), axis=1)

            # Trick ART into accepting our wrapper by setting module to sklearn
            AdapterWrapper.__module__ = "sklearn.base"
            wrapper = AdapterWrapper(model_adapter)
            self._art_estimator = SklearnClassifier(
                model=wrapper,
                clip_values=(0, 1),
                preprocessing_defences=[]
            )
        return self._art_estimator

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """Execute Square attack."""
        model_adapter = self._get_model_adapter()

        if self.backend == "rust":
            return self._run_rust(X_input, model_adapter)
        else:
            return self._run_art(X_input, model_adapter)

    def _run_rust(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using Rust."""
        if self._rust_attack is None:
            # Get model metadata
            n_features = getattr(self.base_model, 'n_features_in_', 0)
            n_classes = len(getattr(self.base_model, 'classes_', [0, 1]))

            # Try to use ONNX model for 15x speedup, fallback to Python model
            onnx_bytes = model_adapter.onnx_bytes
            if onnx_bytes:
                logger.info(f"✓ Using ONNX model ({len(onnx_bytes)} bytes) for Rust backend")
                self._rust_attack = RustSquare(
                    onnx_bytes=onnx_bytes,
                    n_features=n_features,
                    n_classes=n_classes,
                    max_iter=self.config["max_iter"],
                    epsilon=self.config["epsilon"],
                    p_init=self.config["p_init"],
                    n_restarts=self.config["n_restarts"],
                    parallel=True,
                )
            else:
                logger.warning("⚠ ONNX conversion failed, using Python model (slower)")
                self._rust_attack = RustSquare(
                    python_model=model_adapter.original_model,
                    max_iter=self.config["max_iter"],
                    epsilon=self.config["epsilon"],
                    p_init=self.config["p_init"],
                    n_restarts=self.config["n_restarts"],
                    parallel=True,
                )

        # Process in mini-batches to balance parallelism with progress feedback
        n_samples = X_input.shape[0]
        batch_size = max(1, min(16, n_samples))
        n_batches = (n_samples + batch_size - 1) // batch_size

        X_adv_list = []
        total_queries = 0

        with tqdm(total=n_samples, desc="Square", bar_format=RUST_BAR_FORMAT, leave=True) as pbar:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch = X_input[start:end]

                x_adv_batch, metrics = self._rust_attack.run(batch)
                X_adv_list.append(x_adv_batch)
                total_queries += metrics.queries_used
                pbar.update(end - start)

        X_adv = np.vstack(X_adv_list)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="Square (Rust)",
            y_original=model_adapter.predict(X_input).argmax(axis=1),
            y_adversarial=model_adapter.predict(X_adv).argmax(axis=1),
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=total_queries,
        )

    def _run_art(self, X_input: np.ndarray, model_adapter: ModelAdapter) -> AdversarialMetrics:
        """Execute using ART."""
        if self._art_attack is None:
            art_estimator = self._adapt_to_art(model_adapter)
            self._art_attack = ARTSquareAttack(
                estimator=art_estimator,
                norm=np.inf,
                max_iter=self.config["max_iter"],
                eps=self.config["epsilon"],
                p_init=self.config["p_init"],
            )

        y_pred_original = model_adapter.predict(X_input)
        y_label_original = np.argmax(y_pred_original, axis=1)

        X_adv = self._art_attack.generate(x=X_input)

        y_pred_adv = model_adapter.predict(X_adv)
        y_label_adv = np.argmax(y_pred_adv, axis=1)

        return AdversarialMetrics.for_evasion_attack(
            attack_type="Square (ART)",
            y_original=y_label_original,
            y_adversarial=y_label_adv,
            X_original=X_input,
            X_adversarial=X_adv,
            queries_used=self.config["max_iter"] * len(X_input),
        )


# ============================================================================
# Regression Attacks
# ============================================================================

from typing import Tuple

# Valid confidence levels for regression attacks
CONFIDENCE_LEVELS = {
    0.98: "98%",
    0.99: "99%",
    0.995: "99.5%",
    0.999: "99.9%",
    0.9999: "99.99%",
}


class RegressionMetrics:
    """
    Metrics for regression adversarial attacks.

    For OutputManipulation and PredictionShift attacks, the primary metric is
    **mean attempts to success** rather than attack success rate. This reflects
    the reality that regression models are inherently vulnerable - the question
    is not *if* an attack succeeds, but *how easily*.

    Samples that never succeed within the query/iteration budget are excluded
    from the mean and reported separately in `samples_never_succeeded`.
    """

    def __init__(
        self,
        attack_type: str,
        samples_tested: int,
        # Attempts-based metrics (primary for OutputManipulation/PredictionShift)
        samples_succeeded: int,
        samples_never_succeeded: int,
        mean_attempts_to_success: float,
        median_attempts_to_success: float,
        min_attempts: Optional[int],
        max_attempts: Optional[int],
        attempts_ci: Tuple[float, float],
        confidence_level: float,
        # Perturbation metrics
        mean_perturbation_l2: float,
        mean_perturbation_linf: float,
        mean_prediction_shift: float,
        max_prediction_shift: float,
        mean_absolute_shift: float,
        # Legacy/QuantileAttack specific
        attack_success_rate: Optional[float] = None,
        bin_flip_rate: Optional[float] = None,
        mean_bin_distance: Optional[float] = None,
        coverage_break_rate: Optional[float] = None,
        original_coverage: Optional[float] = None,
        adversarial_coverage: Optional[float] = None,
    ):
        self.attack_type = attack_type
        self.samples_tested = samples_tested
        # Attempts-based metrics
        self.samples_succeeded = samples_succeeded
        self.samples_never_succeeded = samples_never_succeeded
        self.mean_attempts_to_success = mean_attempts_to_success
        self.median_attempts_to_success = median_attempts_to_success
        self.min_attempts = min_attempts
        self.max_attempts = max_attempts
        self.attempts_ci = attempts_ci
        self.confidence_level = confidence_level
        # Perturbation metrics
        self.mean_perturbation_l2 = mean_perturbation_l2
        self.mean_perturbation_linf = mean_perturbation_linf
        self.mean_prediction_shift = mean_prediction_shift
        self.max_prediction_shift = max_prediction_shift
        self.mean_absolute_shift = mean_absolute_shift
        # Legacy/QuantileAttack specific
        self.attack_success_rate = attack_success_rate
        self.bin_flip_rate = bin_flip_rate
        self.mean_bin_distance = mean_bin_distance
        self.coverage_break_rate = coverage_break_rate
        self.original_coverage = original_coverage
        self.adversarial_coverage = adversarial_coverage

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "attack_type": self.attack_type,
            "samples_tested": self.samples_tested,
            "samples_succeeded": self.samples_succeeded,
            "samples_never_succeeded": self.samples_never_succeeded,
            "mean_attempts_to_success": self.mean_attempts_to_success,
            "median_attempts_to_success": self.median_attempts_to_success,
            "min_attempts": self.min_attempts,
            "max_attempts": self.max_attempts,
            "attempts_ci": self.attempts_ci,
            "confidence_level": self.confidence_level,
            "mean_perturbation_l2": self.mean_perturbation_l2,
            "mean_perturbation_linf": self.mean_perturbation_linf,
            "mean_prediction_shift": self.mean_prediction_shift,
            "max_prediction_shift": self.max_prediction_shift,
            "mean_absolute_shift": self.mean_absolute_shift,
            "attack_success_rate": self.attack_success_rate,
            "bin_flip_rate": self.bin_flip_rate,
            "mean_bin_distance": self.mean_bin_distance,
            "coverage_break_rate": self.coverage_break_rate,
            "original_coverage": self.original_coverage,
            "adversarial_coverage": self.adversarial_coverage,
        }

    def __repr__(self):
        # Format based on attack type
        if self.attack_type == "QuantileAttack":
            return (
                f"RegressionMetrics(attack='{self.attack_type}', "
                f"success_rate={self.attack_success_rate:.1%}, "
                f"L2={self.mean_perturbation_l2:.4f})"
            )
        else:
            ci_str = f"[{self.attempts_ci[0]:.1f}, {self.attempts_ci[1]:.1f}]"
            return (
                f"RegressionMetrics(attack='{self.attack_type}', "
                f"mean_attempts={self.mean_attempts_to_success:.1f}, "
                f"{self.confidence_level*100:.1f}% CI={ci_str}, "
                f"L2={self.mean_perturbation_l2:.4f})"
            )

    def summary(self) -> str:
        """Generate human-readable summary of attack results."""
        lines = [
            f"=== {self.attack_type} Results ===",
            f"Samples tested: {self.samples_tested}",
            f"Samples succeeded: {self.samples_succeeded}",
            f"Samples never succeeded: {self.samples_never_succeeded}",
        ]

        if self.attack_type != "QuantileAttack":
            if not np.isnan(self.mean_attempts_to_success):
                lines.append(f"Mean attempts to success: {self.mean_attempts_to_success:.2f}")
                lines.append(f"Median attempts: {self.median_attempts_to_success:.2f}")
                if self.min_attempts is not None:
                    lines.append(f"Min attempts: {self.min_attempts}")
                if self.max_attempts is not None:
                    lines.append(f"Max attempts: {self.max_attempts}")
                ci_pct = CONFIDENCE_LEVELS.get(self.confidence_level, f"{self.confidence_level*100}%")
                lines.append(f"{ci_pct} CI: [{self.attempts_ci[0]:.2f}, {self.attempts_ci[1]:.2f}]")
            else:
                lines.append("Mean attempts: N/A (no samples succeeded)")
        else:
            if self.attack_success_rate is not None:
                lines.append(f"Attack success rate: {self.attack_success_rate:.1%}")

        lines.extend([
            f"Mean L2 perturbation: {self.mean_perturbation_l2:.4f}",
            f"Mean L∞ perturbation: {self.mean_perturbation_linf:.4f}",
            f"Mean prediction shift: {self.mean_prediction_shift:.4f}",
            f"Max prediction shift: {self.max_prediction_shift:.4f}",
        ])

        if self.bin_flip_rate is not None:
            lines.append(f"Bin flip rate: {self.bin_flip_rate:.1%}")
        if self.mean_bin_distance is not None:
            lines.append(f"Mean bin distance: {self.mean_bin_distance:.2f}")
        if self.coverage_break_rate is not None:
            lines.append(f"Coverage break rate: {self.coverage_break_rate:.1%}")
        if self.original_coverage is not None:
            lines.append(f"Original coverage: {self.original_coverage:.1%}")
        if self.adversarial_coverage is not None:
            lines.append(f"Adversarial coverage: {self.adversarial_coverage:.1%}")

        return "\n".join(lines)


class OutputManipulationWrapper:
    """
    Output Manipulation Attack for regression models.

    Discretizes continuous outputs into bins and uses HopSkipJump-like
    boundary walking to find perturbations that shift predictions to
    a different bin.

    Reports **mean attempts (queries) to success** with bootstrap confidence
    interval, not attack success rate. Since regression models are inherently
    vulnerable, the meaningful metric is how many queries are required to
    achieve a successful bin flip.

    Use cases:
    - Testing robustness of pricing models
    - Evaluating stability of risk score systems
    - Assessing credit limit manipulation resistance
    """

    def __init__(
        self,
        base_model,
        n_bins: int = 5,
        max_iter: int = 50,
        max_eval: int = 5000,
        parallel: bool = True,
        bin_mode: str = "uniform",
        confidence_level: float = 0.99,
    ):
        """
        Initialize Output Manipulation attack.

        Args:
            base_model: Regression model to attack (sklearn-compatible with .predict())
            n_bins: Number of bins to discretize output (default: 5)
            max_iter: Maximum iterations per sample (default: 50)
            max_eval: Maximum model queries per sample (default: 5000)
            parallel: Enable parallel processing (default: True)
            bin_mode: "uniform" (equal width) or "quantile" (equal count)
            confidence_level: Confidence level for bootstrap CI on mean attempts.
                             One of: 0.98, 0.99, 0.995, 0.999, 0.9999 (default: 0.99)
        """
        if not RUST_REGRESSION_AVAILABLE:
            raise RuntimeError(
                "Regression attacks require Rust backend. "
                "Build with: maturin develop --release"
            )

        if confidence_level not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence_level must be one of {list(CONFIDENCE_LEVELS.keys())}, "
                f"got {confidence_level}"
            )

        self.base_model = base_model
        self.config = {
            "n_bins": n_bins,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "parallel": parallel,
            "bin_mode": bin_mode,
            "confidence_level": confidence_level,
        }
        self._rust_attack = None

    def run(self, X_input: np.ndarray) -> RegressionMetrics:
        """
        Execute Output Manipulation attack.

        Args:
            X_input: Input samples to attack (2D array)

        Returns:
            RegressionMetrics with attack results including mean attempts to
            success with bootstrap confidence interval.
        """
        if self._rust_attack is None:
            self._rust_attack = RustOutputManipulation(
                python_model=self.base_model,
                n_bins=self.config["n_bins"],
                max_iter=self.config["max_iter"],
                max_eval=self.config["max_eval"],
                parallel=self.config["parallel"],
                bin_mode=self.config["bin_mode"],
                confidence_level=self.config["confidence_level"],
            )

        X_adv, metrics = self._rust_attack.run(X_input)

        return RegressionMetrics(
            attack_type=metrics.attack_type,
            samples_tested=metrics.samples_tested,
            samples_succeeded=metrics.samples_succeeded,
            samples_never_succeeded=metrics.samples_never_succeeded,
            mean_attempts_to_success=metrics.mean_attempts_to_success,
            median_attempts_to_success=metrics.median_attempts_to_success,
            min_attempts=metrics.min_attempts,
            max_attempts=metrics.max_attempts,
            attempts_ci=metrics.attempts_ci,
            confidence_level=metrics.confidence_level,
            mean_perturbation_l2=metrics.mean_perturbation_l2,
            mean_perturbation_linf=metrics.mean_perturbation_linf,
            mean_prediction_shift=metrics.mean_prediction_shift,
            max_prediction_shift=metrics.max_prediction_shift,
            mean_absolute_shift=metrics.mean_absolute_shift,
            bin_flip_rate=metrics.bin_flip_rate,
            mean_bin_distance=metrics.mean_bin_distance,
        )


class PredictionShiftWrapper:
    """
    Prediction Shift Attack for regression models.

    Finds perturbations that maximize the change in model output.
    Useful for testing output stability under adversarial conditions.

    Reports **mean attempts (directions sampled) to success** with bootstrap
    confidence interval. Success is defined as shifting the prediction by
    more than `success_threshold` fraction of the original value.

    Use cases:
    - Testing price stability in pricing models
    - Evaluating risk score sensitivity
    - Assessing credit limit stability
    """

    def __init__(
        self,
        base_model,
        epsilon: float = 1.0,
        max_iter: int = 100,
        n_directions: int = 50,
        parallel: bool = True,
        target_direction: str = "any",
        success_threshold: float = 0.10,
        confidence_level: float = 0.99,
    ):
        """
        Initialize Prediction Shift attack.

        Args:
            base_model: Regression model to attack
            epsilon: Maximum L2 perturbation magnitude (default: 1.0)
            max_iter: Maximum iterations (default: 100)
            n_directions: Random directions per iteration (default: 50)
            parallel: Enable parallel processing (default: True)
            target_direction: "any" (maximize |shift|), "increase", or "decrease"
            success_threshold: Shift must exceed this fraction of original value
                              to count as success (default: 0.10 = 10%)
            confidence_level: Confidence level for bootstrap CI on mean attempts.
                             One of: 0.98, 0.99, 0.995, 0.999, 0.9999 (default: 0.99)
        """
        if not RUST_REGRESSION_AVAILABLE:
            raise RuntimeError(
                "Regression attacks require Rust backend. "
                "Build with: maturin develop --release"
            )

        if confidence_level not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence_level must be one of {list(CONFIDENCE_LEVELS.keys())}, "
                f"got {confidence_level}"
            )

        self.base_model = base_model
        self.config = {
            "epsilon": epsilon,
            "max_iter": max_iter,
            "n_directions": n_directions,
            "parallel": parallel,
            "target_direction": target_direction,
            "success_threshold": success_threshold,
            "confidence_level": confidence_level,
        }
        self._rust_attack = None

    def run(self, X_input: np.ndarray) -> RegressionMetrics:
        """
        Execute Prediction Shift attack.

        Args:
            X_input: Input samples to attack (2D array)

        Returns:
            RegressionMetrics with attack results including mean attempts to
            success with bootstrap confidence interval.
        """
        if self._rust_attack is None:
            self._rust_attack = RustPredictionShift(
                python_model=self.base_model,
                epsilon=self.config["epsilon"],
                max_iter=self.config["max_iter"],
                n_directions=self.config["n_directions"],
                parallel=self.config["parallel"],
                target_direction=self.config["target_direction"],
                success_threshold=self.config["success_threshold"],
                confidence_level=self.config["confidence_level"],
            )

        X_adv, metrics = self._rust_attack.run(X_input)

        return RegressionMetrics(
            attack_type=metrics.attack_type,
            samples_tested=metrics.samples_tested,
            samples_succeeded=metrics.samples_succeeded,
            samples_never_succeeded=metrics.samples_never_succeeded,
            mean_attempts_to_success=metrics.mean_attempts_to_success,
            median_attempts_to_success=metrics.median_attempts_to_success,
            min_attempts=metrics.min_attempts,
            max_attempts=metrics.max_attempts,
            attempts_ci=metrics.attempts_ci,
            confidence_level=metrics.confidence_level,
            mean_perturbation_l2=metrics.mean_perturbation_l2,
            mean_perturbation_linf=metrics.mean_perturbation_linf,
            mean_prediction_shift=metrics.mean_prediction_shift,
            max_prediction_shift=metrics.max_prediction_shift,
            mean_absolute_shift=metrics.mean_absolute_shift,
        )


class QuantileAttackWrapper:
    """
    Quantile Attack for Conformal Prediction models.

    Targets the coverage guarantee of conformal prediction by finding
    perturbations that push true values outside predicted intervals.

    This attack is especially useful for testing CP-compliant models
    where regulators require coverage guarantees.

    Use cases:
    - Testing robustness of CP-wrapped models
    - Evaluating coverage guarantee stability
    - Assessing interval prediction reliability
    """

    def __init__(
        self,
        base_model,
        epsilon: float = 1.0,
        max_iter: int = 100,
        n_directions: int = 50,
        parallel: bool = True,
        attack_mode: str = "break_coverage",
        calibration_width: Optional[float] = None,
    ):
        """
        Initialize Quantile Attack.

        Args:
            base_model: Regression model (optionally with .predict_interval())
            epsilon: Maximum L2 perturbation magnitude (default: 1.0)
            max_iter: Maximum iterations (default: 100)
            n_directions: Random directions per iteration (default: 50)
            parallel: Enable parallel processing (default: True)
            attack_mode: "break_coverage" or "maximize_width"
            calibration_width: Half-width for synthetic intervals if model
                              doesn't provide predict_interval()
        """
        if not RUST_REGRESSION_AVAILABLE:
            raise RuntimeError(
                "Regression attacks require Rust backend. "
                "Build with: maturin develop --release"
            )

        self.base_model = base_model
        self.config = {
            "epsilon": epsilon,
            "max_iter": max_iter,
            "n_directions": n_directions,
            "parallel": parallel,
            "attack_mode": attack_mode,
            "calibration_width": calibration_width,
        }
        self._rust_attack = None

    def run(
        self,
        X_input: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> RegressionMetrics:
        """
        Execute Quantile Attack.

        Args:
            X_input: Input samples to attack (2D array)
            y_true: True target values (optional, for coverage calculation)

        Returns:
            RegressionMetrics with attack results
        """
        if self._rust_attack is None:
            self._rust_attack = RustQuantileAttack(
                python_model=self.base_model,
                epsilon=self.config["epsilon"],
                max_iter=self.config["max_iter"],
                n_directions=self.config["n_directions"],
                parallel=self.config["parallel"],
                attack_mode=self.config["attack_mode"],
                calibration_width=self.config["calibration_width"],
            )

        X_adv, metrics = self._rust_attack.run(X_input, y_true)

        return RegressionMetrics(
            attack_type=metrics.attack_type,
            samples_tested=metrics.samples_tested,
            # QuantileAttack uses attack_success_rate, not attempts-based metrics
            samples_succeeded=metrics.samples_succeeded,
            samples_never_succeeded=metrics.samples_never_succeeded,
            mean_attempts_to_success=float('nan'),  # Not applicable
            median_attempts_to_success=float('nan'),
            min_attempts=None,
            max_attempts=None,
            attempts_ci=(float('nan'), float('nan')),
            confidence_level=0.0,  # Not used
            mean_perturbation_l2=metrics.mean_perturbation_l2,
            mean_perturbation_linf=metrics.mean_perturbation_linf,
            mean_prediction_shift=metrics.mean_prediction_shift,
            max_prediction_shift=metrics.max_prediction_shift,
            mean_absolute_shift=metrics.mean_absolute_shift,
            attack_success_rate=metrics.attack_success_rate,
            coverage_break_rate=metrics.coverage_break_rate,
            original_coverage=metrics.original_coverage,
            adversarial_coverage=metrics.adversarial_coverage,
        )
