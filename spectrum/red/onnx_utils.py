"""
ONNX Conversion Utilities for spectrum-red

Provides sklearn model to ONNX conversion to enable high-performance
Rust inference without Python GIL overhead.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def sklearn_to_onnx(model, n_features: int) -> Optional[bytes]:
    """
    Convert sklearn model to ONNX format.

    This enables pure-Rust inference using onnxruntime-rs, eliminating the
    Python-Rust boundary crossing overhead that causes 16x slowdown.

    Args:
        model: sklearn model (LogisticRegression, RandomForest, etc.)
        n_features: Number of input features

    Returns:
        ONNX model serialized as bytes, or None if conversion fails

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>> onnx_bytes = sklearn_to_onnx(model, n_features=10)
        >>> # Pass onnx_bytes to Rust backend for 15x speedup!
    """
    try:
        from skl2onnx import to_onnx

        # Use to_onnx helper which properly handles zipmap option
        # Without zipmap=False, output is Sequence<Map<i64, f32>> which is incompatible with Rust
        X_dummy = np.zeros((1, n_features), dtype=np.float32)

        onnx_model = to_onnx(
            model,
            X_dummy,
            target_opset=12,
            options={'zipmap': False}  # Critical: returns tensor instead of sequence
        )

        # Serialize to bytes
        onnx_bytes = onnx_model.SerializeToString()

        logger.info(f"✓ ONNX conversion successful: {len(onnx_bytes)} bytes")
        return onnx_bytes

    except ImportError:
        logger.warning(
            "⚠ skl2onnx not installed. Install with: pip install skl2onnx\n"
            "   Falling back to Python model (16x slower)"
        )
        return None

    except Exception as e:
        logger.warning(
            f"⚠ ONNX conversion failed: {e}\n"
            f"   Model type: {type(model).__name__}\n"
            f"   Falling back to Python model (16x slower)"
        )
        return None


def validate_onnx_model(onnx_bytes: bytes, model, X_sample: np.ndarray) -> bool:
    """
    Validate that ONNX model produces same results as original sklearn model.

    Args:
        onnx_bytes: Serialized ONNX model
        model: Original sklearn model
        X_sample: Sample input data for validation

    Returns:
        True if predictions match within tolerance, False otherwise
    """
    try:
        import onnxruntime as ort

        # Load ONNX model
        sess = ort.InferenceSession(onnx_bytes)

        # Get predictions from sklearn
        sklearn_pred = model.predict_proba(X_sample)

        # Get predictions from ONNX
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name  # probabilities output
        onnx_pred = sess.run([label_name], {input_name: X_sample.astype(np.float32)})[0]

        # Compare predictions
        max_diff = np.max(np.abs(sklearn_pred - onnx_pred))

        if max_diff < 1e-5:
            logger.info(f"✓ ONNX validation passed (max diff: {max_diff:.2e})")
            return True
        else:
            logger.warning(f"⚠ ONNX validation failed (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        logger.warning(f"⚠ ONNX validation error: {e}")
        return False
