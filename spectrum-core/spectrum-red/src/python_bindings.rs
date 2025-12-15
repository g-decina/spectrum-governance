/// Python bindings for spectrum-red attacks
///
/// This module provides PyO3 bindings to expose Rust attacks to Python.
/// All attacks maintain the same interface as IBM ART for drop-in compatibility.
///
/// Supports two model backends:
/// 1. Python Model: Calls back to Python (slow, GIL overhead)
/// 2. ONNX Model: Pure Rust inference (fast, no GIL!)

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use crate::attacks::{
    BoundaryAttack, BoundaryConfig, HopSkipJumpAttack, HopSkipJumpConfig, SquareAttack,
    SquareConfig, ZOOAttack, ZOOConfig,
    // Regression attacks
    OutputManipulationAttack, OutputManipulationConfig,
    PredictionShiftAttack, PredictionShiftConfig,
    QuantileAttack, QuantileAttackConfig,
    RegressionModel, RegressionMetrics,
    ConfidenceLevel,
};
use crate::metrics::AdversarialMetrics;
use crate::model::Model;

#[cfg(feature = "onnx")]
use crate::onnx_model::OnnxModel;

// ============================================================================
// Python Model Wrapper
// ============================================================================

/// Wrapper that allows Python model objects to implement the Rust Model trait
struct PythonModel {
    model: Py<PyAny>,
}

impl Model for PythonModel {
    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);

            // Convert ndarray to numpy array
            let x_py = PyArray2::from_array(py, x);

            // Try predict_proba first (for classifiers), fall back to predict
            let result = if model.hasattr("predict_proba").unwrap_or(false) {
                // Use predict_proba for probability output
                model
                    .call_method1("predict_proba", (x_py,))
                    .map_err(|e| format!("Python model.predict_proba() failed: {}", e))?
            } else {
                // Fall back to predict (less ideal, but works for some models)
                model
                    .call_method1("predict", (x_py,))
                    .map_err(|e| format!("Python model.predict() failed: {}", e))?
            };

            // Convert result back to ndarray
            let result_array: &PyArray2<f64> = result
                .extract()
                .map_err(|e| format!("Failed to extract prediction result: {}", e))?;

            let array = result_array
                .readonly()
                .as_array()
                .to_owned();

            Ok(array)
        })
    }

    fn input_shape(&self) -> usize {
        // Try to get n_features_in_ from the model
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);
            model
                .getattr("n_features_in_")
                .and_then(|attr| attr.extract::<usize>())
                .unwrap_or(0)  // Default to 0 if not available
        })
    }

    fn num_classes(&self) -> usize {
        // Try to get classes_ length from the model
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);
            model
                .getattr("classes_")
                .and_then(|classes| classes.len())
                .unwrap_or(2)  // Default to binary classification
        })
    }
}

// ============================================================================
// Python Metrics Wrapper
// ============================================================================

/// Python-facing metrics object
#[pyclass(name = "AdversarialMetrics")]
#[derive(Clone)]
struct PyAdversarialMetrics {
    inner: AdversarialMetrics,
}

#[pymethods]
impl PyAdversarialMetrics {
    #[getter]
    fn attack_success_rate(&self) -> f64 {
        self.inner.attack_success_rate
    }

    #[getter]
    fn samples_tested(&self) -> usize {
        self.inner.samples_tested
    }

    #[getter]
    fn samples_successful(&self) -> usize {
        self.inner.samples_successful
    }

    #[getter]
    fn empirical_robustness_l2(&self) -> f64 {
        self.inner.empirical_robustness_l2
    }

    #[getter]
    fn empirical_robustness_linf(&self) -> f64 {
        self.inner.empirical_robustness_linf
    }

    #[getter]
    fn min_perturbation_l2(&self) -> f64 {
        self.inner.min_perturbation_l2
    }

    #[getter]
    fn max_perturbation_l2(&self) -> f64 {
        self.inner.max_perturbation_l2
    }

    #[getter]
    fn median_perturbation_l2(&self) -> f64 {
        self.inner.median_perturbation_l2
    }

    #[getter]
    fn attack_type(&self) -> String {
        self.inner.attack_type.clone()
    }

    #[getter]
    fn queries_used(&self) -> usize {
        self.inner.queries_used
    }

    #[getter]
    fn avg_queries_per_sample(&self) -> f64 {
        self.inner.avg_queries_per_sample
    }

    #[getter]
    fn perturbation_std_l2(&self) -> Option<f64> {
        self.inner.perturbation_std_l2
    }

    #[getter]
    fn perturbation_std_linf(&self) -> Option<f64> {
        self.inner.perturbation_std_linf
    }

    fn __repr__(&self) -> String {
        format!(
            "AdversarialMetrics(attack='{}', success_rate={:.2}%, samples={}/{}, L2={:.4}, queries={})",
            self.inner.attack_type,
            self.inner.attack_success_rate * 100.0,
            self.inner.samples_successful,
            self.inner.samples_tested,
            self.inner.empirical_robustness_l2,
            self.inner.queries_used
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("attack_success_rate", self.inner.attack_success_rate)?;
        dict.set_item("samples_tested", self.inner.samples_tested)?;
        dict.set_item("samples_successful", self.inner.samples_successful)?;
        dict.set_item("empirical_robustness_l2", self.inner.empirical_robustness_l2)?;
        dict.set_item("empirical_robustness_linf", self.inner.empirical_robustness_linf)?;
        dict.set_item("min_perturbation_l2", self.inner.min_perturbation_l2)?;
        dict.set_item("max_perturbation_l2", self.inner.max_perturbation_l2)?;
        dict.set_item("median_perturbation_l2", self.inner.median_perturbation_l2)?;
        dict.set_item("attack_type", self.inner.attack_type.clone())?;
        dict.set_item("queries_used", self.inner.queries_used)?;
        dict.set_item("avg_queries_per_sample", self.inner.avg_queries_per_sample)?;
        Ok(dict.into())
    }
}

// ============================================================================
// HopSkipJump Attack
// ============================================================================

#[pyclass(name = "HopSkipJump")]
struct PyHopSkipJump {
    attack: HopSkipJumpAttack,
    model: Arc<dyn Model>,
}

#[pymethods]
impl PyHopSkipJump {
    #[new]
    #[pyo3(signature = (python_model=None, onnx_bytes=None, n_features=0, n_classes=2, max_iter=64, max_eval=10000, init_eval=100, init_size=1000, parallel=true))]
    fn new(
        python_model: Option<PyObject>,
        onnx_bytes: Option<&[u8]>,
        n_features: usize,
        n_classes: usize,
        max_iter: usize,
        max_eval: usize,
        init_eval: usize,
        init_size: usize,
        parallel: bool,
    ) -> PyResult<Self> {
        // Create model backend (ONNX preferred, fallback to Python)
        let model: Arc<dyn Model> = if let Some(onnx) = onnx_bytes {
            #[cfg(feature = "onnx")]
            {
                let onnx_model = OnnxModel::from_bytes(onnx, n_features, n_classes)
                    .map_err(|e| PyRuntimeError::new_err(format!("ONNX load failed: {}", e)))?;
                Arc::new(onnx_model)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(PyRuntimeError::new_err(
                    "ONNX feature not enabled. Rebuild with --features onnx"
                ));
            }
        } else if let Some(py_model) = python_model {
            Arc::new(PythonModel { model: py_model })
        } else {
            return Err(PyValueError::new_err(
                "Must provide either python_model or onnx_bytes"
            ));
        };

        let config = HopSkipJumpConfig::builder()
            .max_iter(max_iter)
            .max_eval(max_eval)
            .init_eval(init_eval)
            .init_size(init_size)
            .parallel(parallel)
            .build();

        Ok(Self {
            attack: HopSkipJumpAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyAdversarialMetrics)> {
        let x_array = x.as_array().to_owned();

        // Run attack (release GIL for parallel execution)
        // With ONNX model, this is truly parallel (no GIL contention!)
        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyAdversarialMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// ZOO Attack
// ============================================================================

#[pyclass(name = "ZOO")]
struct PyZOO {
    attack: ZOOAttack,
    model: Arc<dyn Model>,
}

#[pymethods]
impl PyZOO {
    #[new]
    #[pyo3(signature = (python_model=None, onnx_bytes=None, n_features=0, n_classes=2, max_iter=1000, learning_rate=0.01, epsilon=0.3, batch_size=128, parallel=true))]
    fn new(
        python_model: Option<PyObject>,
        onnx_bytes: Option<&[u8]>,
        n_features: usize,
        n_classes: usize,
        max_iter: usize,
        learning_rate: f64,
        epsilon: f64,
        batch_size: usize,
        parallel: bool,
    ) -> PyResult<Self> {
        // Create model backend (ONNX preferred, fallback to Python)
        let model: Arc<dyn Model> = if let Some(onnx) = onnx_bytes {
            #[cfg(feature = "onnx")]
            {
                let onnx_model = OnnxModel::from_bytes(onnx, n_features, n_classes)
                    .map_err(|e| PyRuntimeError::new_err(format!("ONNX load failed: {}", e)))?;
                Arc::new(onnx_model)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(PyRuntimeError::new_err(
                    "ONNX feature not enabled. Rebuild with --features onnx"
                ));
            }
        } else if let Some(py_model) = python_model {
            Arc::new(PythonModel { model: py_model })
        } else {
            return Err(PyValueError::new_err(
                "Must provide either python_model or onnx_bytes"
            ));
        };

        let config = ZOOConfig::builder()
            .max_iter(max_iter)
            .learning_rate(learning_rate)
            .epsilon(epsilon)
            .batch_size(batch_size)
            .parallel(parallel)
            .build();

        Ok(Self {
            attack: ZOOAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyAdversarialMetrics)> {
        let x_array = x.as_array().to_owned();

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyAdversarialMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Boundary Attack
// ============================================================================

#[pyclass(name = "Boundary")]
struct PyBoundary {
    attack: BoundaryAttack,
    model: Arc<dyn Model>,
}

#[pymethods]
impl PyBoundary {
    #[new]
    #[pyo3(signature = (python_model=None, onnx_bytes=None, n_features=0, n_classes=2, max_iter=5000, delta=0.01, epsilon=0.01, init_size=100, parallel=true))]
    fn new(
        python_model: Option<PyObject>,
        onnx_bytes: Option<&[u8]>,
        n_features: usize,
        n_classes: usize,
        max_iter: usize,
        delta: f64,
        epsilon: f64,
        init_size: usize,
        parallel: bool,
    ) -> PyResult<Self> {
        // Create model backend (ONNX preferred, fallback to Python)
        let model: Arc<dyn Model> = if let Some(onnx) = onnx_bytes {
            #[cfg(feature = "onnx")]
            {
                let onnx_model = OnnxModel::from_bytes(onnx, n_features, n_classes)
                    .map_err(|e| PyRuntimeError::new_err(format!("ONNX load failed: {}", e)))?;
                Arc::new(onnx_model)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(PyRuntimeError::new_err(
                    "ONNX feature not enabled. Rebuild with --features onnx"
                ));
            }
        } else if let Some(py_model) = python_model {
            Arc::new(PythonModel { model: py_model })
        } else {
            return Err(PyValueError::new_err(
                "Must provide either python_model or onnx_bytes"
            ));
        };

        let config = BoundaryConfig::builder()
            .max_iter(max_iter)
            .delta(delta)
            .epsilon(epsilon)
            .init_size(init_size)
            .parallel(parallel)
            .build();

        Ok(Self {
            attack: BoundaryAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyAdversarialMetrics)> {
        let x_array = x.as_array().to_owned();

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyAdversarialMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Square Attack
// ============================================================================

#[pyclass(name = "Square")]
struct PySquare {
    attack: SquareAttack,
    model: Arc<dyn Model>,
}

#[pymethods]
impl PySquare {
    #[new]
    #[pyo3(signature = (python_model=None, onnx_bytes=None, n_features=0, n_classes=2, max_iter=10000, epsilon=0.05, p_init=0.8, n_restarts=100, parallel=true))]
    fn new(
        python_model: Option<PyObject>,
        onnx_bytes: Option<&[u8]>,
        n_features: usize,
        n_classes: usize,
        max_iter: usize,
        epsilon: f64,
        p_init: f64,
        n_restarts: usize,
        parallel: bool,
    ) -> PyResult<Self> {
        // Create model backend (ONNX preferred, fallback to Python)
        let model: Arc<dyn Model> = if let Some(onnx) = onnx_bytes {
            #[cfg(feature = "onnx")]
            {
                let onnx_model = OnnxModel::from_bytes(onnx, n_features, n_classes)
                    .map_err(|e| PyRuntimeError::new_err(format!("ONNX load failed: {}", e)))?;
                Arc::new(onnx_model)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(PyRuntimeError::new_err(
                    "ONNX feature not enabled. Rebuild with --features onnx"
                ));
            }
        } else if let Some(py_model) = python_model {
            Arc::new(PythonModel { model: py_model })
        } else {
            return Err(PyValueError::new_err(
                "Must provide either python_model or onnx_bytes"
            ));
        };

        let config = SquareConfig::builder()
            .max_iter(max_iter)
            .epsilon(epsilon)
            .p_init(p_init)
            .n_restarts(n_restarts)
            .parallel(parallel)
            .build();

        Ok(Self {
            attack: SquareAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyAdversarialMetrics)> {
        let x_array = x.as_array().to_owned();

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyAdversarialMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Python Regression Model Wrapper
// ============================================================================

/// Wrapper that allows Python regression model objects to implement the Rust RegressionModel trait
struct PythonRegressionModel {
    model: Py<PyAny>,
    /// Optional calibration half-width for conformal prediction
    calibration_width: Option<f64>,
}

impl RegressionModel for PythonRegressionModel {
    fn predict(&self, x: &Array2<f64>) -> Result<ndarray::Array1<f64>, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);
            let x_py = PyArray2::from_array(py, x);

            let result = model
                .call_method1("predict", (x_py,))
                .map_err(|e| format!("Python model.predict() failed: {}", e))?;

            // Handle both 1D and 2D outputs
            if let Ok(arr1) = result.extract::<&numpy::PyArray1<f64>>() {
                Ok(arr1.readonly().as_array().to_owned())
            } else if let Ok(arr2) = result.extract::<&PyArray2<f64>>() {
                // If 2D, take the first column
                let array = arr2.readonly().as_array().to_owned();
                Ok(array.column(0).to_owned())
            } else {
                Err("Failed to extract prediction result".into())
            }
        })
    }

    fn input_shape(&self) -> usize {
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);
            model
                .getattr("n_features_in_")
                .and_then(|attr| attr.extract::<usize>())
                .unwrap_or(0)
        })
    }

    fn predict_interval(
        &self,
        x: &Array2<f64>,
    ) -> Result<Option<(ndarray::Array1<f64>, ndarray::Array1<f64>)>, Box<dyn std::error::Error>> {
        // Check if model has predict_interval method
        Python::with_gil(|py| {
            let model = self.model.as_ref(py);

            if model.hasattr("predict_interval").unwrap_or(false) {
                let x_py = PyArray2::from_array(py, x);
                let result = model
                    .call_method1("predict_interval", (x_py,))
                    .map_err(|e| format!("Python model.predict_interval() failed: {}", e))?;

                // Expect tuple of (lower, upper)
                let (lower_py, upper_py): (&numpy::PyArray1<f64>, &numpy::PyArray1<f64>) = result
                    .extract()
                    .map_err(|e| format!("Failed to extract interval: {}", e))?;

                let lower = lower_py.readonly().as_array().to_owned();
                let upper = upper_py.readonly().as_array().to_owned();

                Ok(Some((lower, upper)))
            } else if let Some(width) = self.calibration_width {
                // Fall back to calibration width
                let pred = self.predict(x)?;
                let lower = pred.mapv(|v| v - width);
                let upper = pred.mapv(|v| v + width);
                Ok(Some((lower, upper)))
            } else {
                Ok(None)
            }
        })
    }
}

// ============================================================================
// Python Regression Metrics Wrapper
// ============================================================================

#[pyclass(name = "RegressionMetrics")]
#[derive(Clone)]
struct PyRegressionMetrics {
    inner: RegressionMetrics,
}

#[pymethods]
impl PyRegressionMetrics {
    #[getter]
    fn attack_type(&self) -> String {
        self.inner.attack_type.clone()
    }

    #[getter]
    fn samples_tested(&self) -> usize {
        self.inner.samples_tested
    }

    // New attempts-based metrics
    #[getter]
    fn samples_succeeded(&self) -> usize {
        self.inner.samples_succeeded
    }

    #[getter]
    fn samples_never_succeeded(&self) -> usize {
        self.inner.samples_never_succeeded
    }

    #[getter]
    fn mean_attempts_to_success(&self) -> f64 {
        self.inner.mean_attempts_to_success
    }

    #[getter]
    fn median_attempts_to_success(&self) -> f64 {
        self.inner.median_attempts_to_success
    }

    #[getter]
    fn min_attempts(&self) -> Option<usize> {
        self.inner.min_attempts
    }

    #[getter]
    fn max_attempts(&self) -> Option<usize> {
        self.inner.max_attempts
    }

    #[getter]
    fn attempts_ci(&self) -> (f64, f64) {
        self.inner.attempts_ci
    }

    #[getter]
    fn confidence_level(&self) -> f64 {
        self.inner.confidence_level
    }

    // Legacy field for QuantileAttack
    #[getter]
    fn attack_success_rate(&self) -> Option<f64> {
        self.inner.attack_success_rate
    }

    #[getter]
    fn mean_perturbation_l2(&self) -> f64 {
        self.inner.mean_perturbation_l2
    }

    #[getter]
    fn mean_perturbation_linf(&self) -> f64 {
        self.inner.mean_perturbation_linf
    }

    #[getter]
    fn mean_prediction_shift(&self) -> f64 {
        self.inner.mean_prediction_shift
    }

    #[getter]
    fn max_prediction_shift(&self) -> f64 {
        self.inner.max_prediction_shift
    }

    #[getter]
    fn mean_absolute_shift(&self) -> f64 {
        self.inner.mean_absolute_shift
    }

    #[getter]
    fn bin_flip_rate(&self) -> Option<f64> {
        self.inner.bin_flip_rate
    }

    #[getter]
    fn mean_bin_distance(&self) -> Option<f64> {
        self.inner.mean_bin_distance
    }

    #[getter]
    fn coverage_break_rate(&self) -> Option<f64> {
        self.inner.coverage_break_rate
    }

    #[getter]
    fn original_coverage(&self) -> Option<f64> {
        self.inner.original_coverage
    }

    #[getter]
    fn adversarial_coverage(&self) -> Option<f64> {
        self.inner.adversarial_coverage
    }

    fn __repr__(&self) -> String {
        // Format based on attack type
        if self.inner.attack_type == "QuantileAttack" {
            format!(
                "RegressionMetrics(attack='{}', success_rate={:.1}%, samples={}/{}, L2={:.4})",
                self.inner.attack_type,
                self.inner.attack_success_rate.unwrap_or(0.0) * 100.0,
                self.inner.samples_succeeded,
                self.inner.samples_tested,
                self.inner.mean_perturbation_l2
            )
        } else {
            format!(
                "RegressionMetrics(attack='{}', mean_attempts={:.1}, CI=[{:.1}, {:.1}], samples={}/{}, L2={:.4})",
                self.inner.attack_type,
                self.inner.mean_attempts_to_success,
                self.inner.attempts_ci.0,
                self.inner.attempts_ci.1,
                self.inner.samples_succeeded,
                self.inner.samples_tested,
                self.inner.mean_perturbation_l2
            )
        }
    }

    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("attack_type", self.inner.attack_type.clone())?;
        dict.set_item("samples_tested", self.inner.samples_tested)?;
        dict.set_item("samples_succeeded", self.inner.samples_succeeded)?;
        dict.set_item("samples_never_succeeded", self.inner.samples_never_succeeded)?;
        dict.set_item("mean_attempts_to_success", self.inner.mean_attempts_to_success)?;
        dict.set_item("median_attempts_to_success", self.inner.median_attempts_to_success)?;
        dict.set_item("min_attempts", self.inner.min_attempts)?;
        dict.set_item("max_attempts", self.inner.max_attempts)?;
        dict.set_item("attempts_ci", self.inner.attempts_ci)?;
        dict.set_item("confidence_level", self.inner.confidence_level)?;
        dict.set_item("attack_success_rate", self.inner.attack_success_rate)?;
        dict.set_item("mean_perturbation_l2", self.inner.mean_perturbation_l2)?;
        dict.set_item("mean_perturbation_linf", self.inner.mean_perturbation_linf)?;
        dict.set_item("mean_prediction_shift", self.inner.mean_prediction_shift)?;
        dict.set_item("max_prediction_shift", self.inner.max_prediction_shift)?;
        dict.set_item("mean_absolute_shift", self.inner.mean_absolute_shift)?;
        dict.set_item("bin_flip_rate", self.inner.bin_flip_rate)?;
        dict.set_item("mean_bin_distance", self.inner.mean_bin_distance)?;
        dict.set_item("coverage_break_rate", self.inner.coverage_break_rate)?;
        dict.set_item("original_coverage", self.inner.original_coverage)?;
        dict.set_item("adversarial_coverage", self.inner.adversarial_coverage)?;
        Ok(dict.into())
    }
}

// ============================================================================
// Confidence Level Conversion
// ============================================================================

/// Convert a Python float confidence level to Rust ConfidenceLevel enum.
fn confidence_level_from_f64(level: f64) -> PyResult<ConfidenceLevel> {
    match level {
        x if (x - 0.98).abs() < 1e-6 => Ok(ConfidenceLevel::Confidence98),
        x if (x - 0.99).abs() < 1e-6 => Ok(ConfidenceLevel::Confidence99),
        x if (x - 0.995).abs() < 1e-6 => Ok(ConfidenceLevel::Confidence995),
        x if (x - 0.999).abs() < 1e-6 => Ok(ConfidenceLevel::Confidence999),
        x if (x - 0.9999).abs() < 1e-6 => Ok(ConfidenceLevel::Confidence9999),
        _ => Err(PyValueError::new_err(format!(
            "Invalid confidence_level: {}. Must be one of: 0.98, 0.99, 0.995, 0.999, 0.9999",
            level
        ))),
    }
}

// ============================================================================
// Output Manipulation Attack
// ============================================================================

#[pyclass(name = "OutputManipulation")]
struct PyOutputManipulation {
    attack: OutputManipulationAttack,
    model: Arc<dyn RegressionModel>,
}

#[pymethods]
impl PyOutputManipulation {
    #[new]
    #[pyo3(signature = (python_model, n_bins=5, max_iter=50, max_eval=5000, parallel=true, bin_mode="uniform", confidence_level=0.99, calibration_width=None))]
    fn new(
        python_model: PyObject,
        n_bins: usize,
        max_iter: usize,
        max_eval: usize,
        parallel: bool,
        bin_mode: &str,
        confidence_level: f64,
        calibration_width: Option<f64>,
    ) -> PyResult<Self> {
        let model: Arc<dyn RegressionModel> = Arc::new(PythonRegressionModel {
            model: python_model,
            calibration_width,
        });

        let conf_level = confidence_level_from_f64(confidence_level)?;

        let config = OutputManipulationConfig::builder()
            .n_bins(n_bins)
            .max_iter(max_iter)
            .max_eval(max_eval)
            .parallel(parallel)
            .bin_mode(bin_mode)
            .confidence_level(conf_level)
            .build();

        Ok(Self {
            attack: OutputManipulationAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyRegressionMetrics)> {
        let x_array = x.as_array().to_owned();

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array, None)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyRegressionMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Prediction Shift Attack
// ============================================================================

#[pyclass(name = "PredictionShift")]
struct PyPredictionShift {
    attack: PredictionShiftAttack,
    model: Arc<dyn RegressionModel>,
}

#[pymethods]
impl PyPredictionShift {
    #[new]
    #[pyo3(signature = (python_model, epsilon=1.0, max_iter=100, n_directions=50, parallel=true, target_direction="any", success_threshold=0.10, confidence_level=0.99, calibration_width=None))]
    fn new(
        python_model: PyObject,
        epsilon: f64,
        max_iter: usize,
        n_directions: usize,
        parallel: bool,
        target_direction: &str,
        success_threshold: f64,
        confidence_level: f64,
        calibration_width: Option<f64>,
    ) -> PyResult<Self> {
        let model: Arc<dyn RegressionModel> = Arc::new(PythonRegressionModel {
            model: python_model,
            calibration_width,
        });

        let conf_level = confidence_level_from_f64(confidence_level)?;

        let config = PredictionShiftConfig::builder()
            .epsilon(epsilon)
            .max_iter(max_iter)
            .n_directions(n_directions)
            .parallel(parallel)
            .target_direction(target_direction)
            .success_threshold(success_threshold)
            .confidence_level(conf_level)
            .build();

        Ok(Self {
            attack: PredictionShiftAttack::new(config),
            model,
        })
    }

    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyRegressionMetrics)> {
        let x_array = x.as_array().to_owned();

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyRegressionMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Quantile Attack
// ============================================================================

#[pyclass(name = "QuantileAttack")]
struct PyQuantileAttack {
    attack: QuantileAttack,
    model: Arc<dyn RegressionModel>,
}

#[pymethods]
impl PyQuantileAttack {
    #[new]
    #[pyo3(signature = (python_model, epsilon=1.0, max_iter=100, n_directions=50, parallel=true, attack_mode="break_coverage", calibration_width=None))]
    fn new(
        python_model: PyObject,
        epsilon: f64,
        max_iter: usize,
        n_directions: usize,
        parallel: bool,
        attack_mode: &str,
        calibration_width: Option<f64>,
    ) -> PyResult<Self> {
        let model: Arc<dyn RegressionModel> = Arc::new(PythonRegressionModel {
            model: python_model,
            calibration_width,
        });

        let mut config_builder = QuantileAttackConfig::builder()
            .epsilon(epsilon)
            .max_iter(max_iter)
            .n_directions(n_directions)
            .parallel(parallel)
            .attack_mode(attack_mode);

        if let Some(width) = calibration_width {
            config_builder = config_builder.calibration_width(width);
        }

        let config = config_builder.build();

        Ok(Self {
            attack: QuantileAttack::new(config),
            model,
        })
    }

    #[pyo3(signature = (x, y_true=None))]
    fn run(
        &self,
        py: Python,
        x: PyReadonlyArray2<f64>,
        y_true: Option<numpy::PyReadonlyArray1<f64>>,
    ) -> PyResult<(Py<PyArray2<f64>>, PyRegressionMetrics)> {
        let x_array = x.as_array().to_owned();
        let y_true_array = y_true.map(|y| y.as_array().to_owned());

        let (x_adv, metrics) = py.allow_threads(|| {
            self.attack
                .run(&*self.model, &x_array, y_true_array.as_ref())
                .map_err(|e| PyRuntimeError::new_err(format!("Attack failed: {}", e)))
        })?;

        let x_adv_py = PyArray2::from_array(py, &x_adv).to_owned();
        let metrics_py = PyRegressionMetrics { inner: metrics };

        Ok((x_adv_py, metrics_py))
    }
}

// ============================================================================
// Python Module
// ============================================================================

#[pymodule]
fn spectrum_red_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classification attacks
    m.add_class::<PyHopSkipJump>()?;
    m.add_class::<PyZOO>()?;
    m.add_class::<PyBoundary>()?;
    m.add_class::<PySquare>()?;
    m.add_class::<PyAdversarialMetrics>()?;

    // Regression attacks
    m.add_class::<PyOutputManipulation>()?;
    m.add_class::<PyPredictionShift>()?;
    m.add_class::<PyQuantileAttack>()?;
    m.add_class::<PyRegressionMetrics>()?;

    Ok(())
}
