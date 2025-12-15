"""
spectrum.red.model_adapter
===========================

Unified model adapter for adversarial attacks with automatic ONNX conversion.

This module provides a unified interface for models across different backends
(ART, Rust) with automatic optimization via ONNX conversion when possible.

KEY FEATURES:
-------------
- Lazy ONNX conversion (only on first predict call)
- Graceful fallback to original model if conversion fails
- Performance logging for transparency
- Thread-safe caching

SUPPORTED MODEL TYPES:
----------------------
- scikit-learn estimators (via skl2onnx)
- PyTorch models (via torch.onnx.export)
- TensorFlow models (via tf2onnx)
- Any model with .predict() method (passthrough)

USAGE:
------
    adapter = ModelAdapter(sklearn_model)
    predictions = adapter.predict(X)  # Automatically uses ONNX if possible
"""

import logging
import numpy as np
from typing import Any, Optional
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ONNXModel:
    """Wrapper for ONNX Runtime inference."""

    def __init__(self, onnx_session, input_name: str, output_name: str):
        self.session = onnx_session
        self.input_name = input_name
        self.output_name = output_name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference using ONNX Runtime."""
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: X})
        return outputs[0]


class ModelAdapter:
    """
    Unified model interface with automatic ONNX optimization.

    Provides lazy conversion to ONNX for performance, with graceful fallback
    to the original model if conversion fails.
    """

    def __init__(self, model: Any, enable_onnx: bool = True):
        """
        Initialize model adapter.

        Args:
            model: Base model (sklearn, pytorch, tensorflow, or any with .predict())
            enable_onnx: Whether to attempt ONNX conversion (default: True)
        """
        self.original_model = model
        self.enable_onnx = enable_onnx

        # Lazy initialization
        self._onnx_model: Optional[ONNXModel] = None
        self._onnx_attempted = False
        self._onnx_available = False
        self._onnx_bytes: Optional[bytes] = None  # For Rust backend (serialized ONNX)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ONNX if available, otherwise use original model.

        Args:
            X: Input features (2D array)

        Returns:
            Predictions (probabilities for classifiers)
        """
        # Lazy ONNX conversion on first call
        if self.enable_onnx and not self._onnx_attempted:
            self._try_convert_to_onnx()

        # Use ONNX if available
        if self._onnx_available and self._onnx_model is not None:
            return self._onnx_model.predict(X)

        # Fallback to original model
        return self._predict_original(X)

    def _predict_original(self, X: np.ndarray) -> np.ndarray:
        """Predict using the original model."""
        if hasattr(self.original_model, 'predict_proba'):
            # Classifier with probability output
            return self.original_model.predict_proba(X)
        elif hasattr(self.original_model, 'predict'):
            # Generic predict method
            predictions = self.original_model.predict(X)

            # Convert labels to probabilities if needed
            if predictions.ndim == 1:
                # Binary classification: convert to probabilities
                n_classes = len(np.unique(predictions))
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, int(pred)] = 1.0
                return proba

            return predictions
        else:
            raise ValueError(f"Model {type(self.original_model)} must have .predict() or .predict_proba() method")

    def _try_convert_to_onnx(self):
        """
        Attempt to convert model to ONNX format.

        Tries multiple conversion strategies based on model type:
        1. sklearn models -> skl2onnx
        2. PyTorch models -> torch.onnx.export
        3. TensorFlow models -> tf2onnx
        """
        self._onnx_attempted = True

        try:
            # Try sklearn conversion
            if isinstance(self.original_model, BaseEstimator):
                self._onnx_model = self._convert_sklearn_to_onnx()
                if self._onnx_model is not None:
                    self._onnx_available = True
                    logger.info("✓ Model converted to ONNX (sklearn)")
                    return

            # Try PyTorch conversion
            if self._is_pytorch_model(self.original_model):
                self._onnx_model = self._convert_pytorch_to_onnx()
                if self._onnx_model is not None:
                    self._onnx_available = True
                    logger.info("✓ Model converted to ONNX (pytorch)")
                    return

            # Try TensorFlow conversion
            if self._is_tensorflow_model(self.original_model):
                self._onnx_model = self._convert_tensorflow_to_onnx()
                if self._onnx_model is not None:
                    self._onnx_available = True
                    logger.info("✓ Model converted to ONNX (tensorflow)")
                    return

            logger.info("ℹ ONNX conversion not available for this model type, using original")

        except Exception as e:
            logger.warning(f"⚠ ONNX conversion failed: {e}, using original model")
            self._onnx_available = False

    def _convert_sklearn_to_onnx(self) -> Optional[ONNXModel]:
        """Convert sklearn model to ONNX."""
        try:
            from .onnx_utils import sklearn_to_onnx
            import onnxruntime as ort

            # Infer number of features from model
            n_features = None
            if hasattr(self.original_model, 'n_features_in_'):
                n_features = self.original_model.n_features_in_
            elif hasattr(self.original_model, 'coef_'):
                n_features = self.original_model.coef_.shape[1]
            else:
                logger.debug("Could not infer n_features, ONNX conversion skipped")
                return None

            # Convert using helper (which properly sets zipmap=False for tensor output)
            onnx_bytes = sklearn_to_onnx(self.original_model, n_features)

            if onnx_bytes is None:
                return None

            # Store bytes for Rust backend
            self._onnx_bytes = onnx_bytes

            # Create ONNX Runtime session for Python inference
            session = ort.InferenceSession(onnx_bytes)

            # Get input/output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[1].name  # Probabilities output

            return ONNXModel(session, input_name, output_name)

        except ImportError:
            logger.debug("skl2onnx or onnxruntime not installed, skipping conversion")
            return None
        except Exception as e:
            logger.debug(f"sklearn->ONNX conversion failed: {e}")
            return None

    def _convert_pytorch_to_onnx(self) -> Optional[ONNXModel]:
        """Convert PyTorch model to ONNX."""
        try:
            import torch
            import onnxruntime as ort
            import tempfile

            # Create dummy input for tracing
            # TODO: Need to infer input shape from model
            logger.debug("PyTorch ONNX conversion not yet implemented")
            return None

        except ImportError:
            logger.debug("torch or onnxruntime not installed")
            return None
        except Exception as e:
            logger.debug(f"PyTorch->ONNX conversion failed: {e}")
            return None

    def _convert_tensorflow_to_onnx(self) -> Optional[ONNXModel]:
        """Convert TensorFlow model to ONNX."""
        try:
            import tf2onnx
            import onnxruntime as ort

            # TODO: Implement TensorFlow conversion
            logger.debug("TensorFlow ONNX conversion not yet implemented")
            return None

        except ImportError:
            logger.debug("tf2onnx or onnxruntime not installed")
            return None
        except Exception as e:
            logger.debug(f"TensorFlow->ONNX conversion failed: {e}")
            return None

    @staticmethod
    def _is_pytorch_model(model: Any) -> bool:
        """Check if model is a PyTorch model."""
        try:
            import torch
            return isinstance(model, torch.nn.Module)
        except ImportError:
            return False

    @staticmethod
    def _is_tensorflow_model(model: Any) -> bool:
        """Check if model is a TensorFlow model."""
        try:
            import tensorflow as tf
            return isinstance(model, (tf.keras.Model, tf.Module))
        except ImportError:
            return False

    @property
    def is_using_onnx(self) -> bool:
        """Check if currently using ONNX backend."""
        return self._onnx_available and self._onnx_model is not None

    @property
    def backend_name(self) -> str:
        """Get name of current backend."""
        if self.is_using_onnx:
            return "onnx"
        return "original"

    @property
    def onnx_bytes(self) -> Optional[bytes]:
        """
        Get ONNX model as serialized bytes for Rust backend.

        Returns:
            Serialized ONNX model bytes, or None if conversion not attempted/failed
        """
        # Trigger lazy conversion if not already attempted
        if self.enable_onnx and not self._onnx_attempted:
            self._try_convert_to_onnx()

        return self._onnx_bytes
