# Spectrum SHAP Implementation Roadmap

## Vision

**"SHAP You Can Defend in Court"** - Explainability with audit trails, stability guarantees, and regulatory compliance built-in.

Unlike standard SHAP libraries that focus on data science workflows, spectrum-blue provides:
- Immutable audit trails for every explanation
- Quantified uncertainty (confidence intervals on SHAP values)
- Deterministic reproducibility via seeded RNG
- Automatic adverse action reason code generation
- Explanation drift monitoring

---

## Python API Design

### Design Principles

The Python API follows the **two-step load-then-explain pattern** familiar from scikit-learn and the `shap` library:

1. **Parse once, explain many** - Optimized for deployment middleware where models are loaded once and used for many inference calls
2. **Familiar interface** - Mirrors existing `shap.TreeExplainer` API for easy adoption
3. **Optional audit logging** - Can be enabled for regulatory compliance without impacting performance
4. **Multi-format support** - Single API for ONNX, XGBoost, and LightGBM models

### Basic Usage

```python
import spectrum_blue

# Step 1: Load model (parse once)
model = spectrum_blue.load_tree_model("model.onnx")

# Step 2: Create explainer
explainer = spectrum_blue.TreeSHAPExplainer()

# Step 3: Explain (many times, efficiently)
for sample in inference_stream:
    shap_values = explainer.explain(model, sample)
    # shap_values: numpy array matching sklearn/shap conventions
```

### With Audit Logging

```python
from spectrum_lens import JsonlSink

# Enable audit trail for regulatory compliance
explainer = spectrum_blue.TreeSHAPExplainer(
    audit_sink=JsonlSink("explanations.jsonl")
)

# Every explain() call is now logged with immutable audit trail
shap_values = explainer.explain(model, features)
```

### Deployment Middleware Pattern

```python
# One-time setup
model = spectrum_blue.load_tree_model("credit_model.onnx")
explainer = spectrum_blue.TreeSHAPExplainer()

# At inference time (high-throughput)
@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']

    # Fast: model already parsed, Rust backend
    shap_values = explainer.explain(model, features)

    return {
        'prediction': model.predict(features),
        'shap_values': shap_values.tolist()
    }
```

### Batch Processing

```python
# Explain multiple samples efficiently
import numpy as np

X_batch = np.array([...])  # Shape: (n_samples, n_features)
shap_batch = explainer.explain_batch(model, X_batch)
# shap_batch: Shape (n_samples, n_features)
```

### API Compatibility Matrix

| Use Case | Current Python API | New Rust API | Migration |
|----------|-------------------|--------------|-----------|
| Ad-hoc analysis | `generate_shap_explanations()` | `spectrum_blue.TreeSHAPExplainer` | Simple |
| Deployment middleware | Not optimized | Optimized (load once) | Recommended |
| Adverse action reasons | `ReasonCodeGenerator` | Phase 4 integration | Future |
| Audit compliance | Manual logging | Built-in `audit_sink` | Phase 3 |

### Rust Public API (Internal)

The PyO3 bindings expose these Rust types to Python:

```rust
// spectrum-blue/src/lib.rs (with PyO3)

#[pyfunction]
fn load_tree_model(path: &str) -> PyResult<Tree> {
    // Auto-detect format (ONNX/XGBoost/LightGBM)
    // Returns parsed Tree instance
}

#[pyclass]
struct TreeSHAPExplainer {
    audit_sink: Option<Box<dyn AuditSink>>,
}

#[pymethods]
impl TreeSHAPExplainer {
    #[new]
    fn new(audit_sink: Option<PyObject>) -> Self { ... }

    fn explain(&self, tree: &Tree, features: Vec<f64>) -> PyResult<Vec<f64>> {
        // Efficient TreeSHAP algorithm (15x faster than Python)
        // Optional audit logging if sink configured
    }

    fn explain_batch(&self, tree: &Tree, features: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        // Parallel batch processing via Rayon
    }
}
```

---

## Phase 1: Core TreeSHAP (Foundation)

**Goal:** High-performance TreeSHAP with reproducibility

**Deliverables:**
- [x] Tree model trait and base implementation
  - [x] `Model` trait (base prediction interface)
  - [x] `TreeModel` trait (tree-specific structure access)
  - [x] `Tree` struct with node representation
- [x] Naive SHAP implementation (exponential, test/validation only)
- [ ] Efficient TreeSHAP algorithm (Lundberg's path-dependent method)
- [ ] Model parsers supporting multiple formats
  - [ ] XGBoost JSON
  - [ ] LightGBM model files
  - [ ] ONNX tree ensemble
- [ ] Seeded RNG for deterministic computation
- [ ] Rayon parallelization (per-sample and per-tree)
- [ ] Python bindings via PyO3
- [ ] Equivalence tests vs Python `shap` library (ε = 10⁻⁵)

**Performance Target:** ≥15x faster than Python shap on 16 cores

**Current Status (2026-04-29):**
- ✅ Module structure established (`models/`, `explainability/`)
- ✅ Naive SHAP working (tests passing)
- 🔜 Next: Efficient TreeSHAP algorithm implementation
- 🔜 Next: PyO3 bindings for Python integration

**Key Files:**
```
spectrum-core/spectrum-blue/src/
├── lib.rs                           # Public API + future PyO3 bindings
├── models/
│   ├── mod.rs                       # Model and TreeModel traits
│   └── tree.rs                      # Tree, TreeNode, NodeType impls
├── explainability/
│   ├── mod.rs
│   └── naive_shap.rs                # NaiveSHAPExplainer (O(2^n))
└── (future) parsers/
    ├── xgboost.rs
    ├── lightgbm.rs
    └── onnx.rs
```

---

## Phase 2: Stability Metrics

**Goal:** Quantify explanation uncertainty

**Deliverables:**
- [ ] Bootstrap resampling for SHAP confidence intervals
- [ ] `StabilityMetrics` struct
  ```rust
  pub struct StabilityMetrics {
      pub confidence_intervals: Array2<(f64, f64)>,  // (low, high) per value
      pub stability_score: f64,                       // 0-1 overall stability
      pub unstable_features: Vec<usize>,              // Features with wide CI
  }
  ```
- [ ] Unstable feature detection and flagging
- [ ] Configurable confidence levels (90%, 95%, 99%)
- [ ] Stability-aware reason code generation (don't rank unstable features highly)

**Regulatory Link:** Model risk management requires understanding explanation uncertainty

---

## Phase 3: Audit Integration

**Goal:** Every explanation auditable via spectrum-lens

**Deliverables:**
- [ ] `AuditSink` trait for pluggable logging
  ```rust
  pub trait AuditSink: Send + Sync {
      fn log_explanation(&self, event: ExplanationEvent) -> Result<Uuid>;
  }
  ```
- [ ] `ExplanationEvent` structure
  ```rust
  pub struct ExplanationEvent {
      pub audit_id: Uuid,
      pub timestamp: DateTime<Utc>,
      pub model_hash: String,        // SHA-256 of model bytes
      pub input_hash: String,        // SHA-256 of input features
      pub shap_values: Vec<f64>,
      pub reason_codes: Vec<ReasonCode>,
      pub stability: StabilityMetrics,
      pub config_hash: String,       // SHA-256 of SHAP config
  }
  ```
- [ ] Input/output hashing (SHA-256)
- [ ] Audit ID generation and tracking
- [ ] spectrum-lens integration (JSONL sink)

**Regulatory Link:** CFPB 1002.9 requires ability to reproduce explanations given to applicants

---

## Phase 4: Reason Code Engine

**Goal:** CFPB 1002.9 compliant adverse action reason codes

**Deliverables:**
- [ ] SHAP → reason code ranking algorithm
- [ ] `ReasonCodeEngine` with configurable templates
  ```rust
  pub struct ReasonCodeEngine {
      templates: HashMap<String, ReasonCodeTemplate>,
      feature_groups: HashMap<String, Vec<String>>,  // Group related features
  }

  pub struct ReasonCode {
      pub code: String,           // e.g., "RC001"
      pub description: String,    // e.g., "High debt-to-income ratio"
      pub shap_contribution: f64, // Aggregated SHAP value
      pub confidence: f64,        // From stability metrics
  }
  ```
- [ ] Multi-factor grouping (e.g., "credit history" combines `credit_length`, `num_accounts`, etc.)
- [ ] Regulatory threshold validation
- [ ] Top-N reason code selection (typically 4-5 per CFPB guidance)

**Regulatory Link:** CFPB Regulation B requires specific, accurate adverse action reasons

---

## Phase 5: Explanation Drift

**Goal:** Monitor explanation consistency over time

**Deliverables:**
- [ ] Baseline explanation fingerprinting
  - Store distribution of SHAP values per feature
  - Track reason code frequency distribution
- [ ] Drift detection metrics
  - PSI on SHAP value distributions
  - Jensen-Shannon divergence on reason code rankings
- [ ] `ExplanationDriftMonitor`
  ```rust
  pub struct ExplanationDriftMonitor {
      baseline: ExplanationBaseline,
      thresholds: DriftThresholds,
  }

  pub struct DriftReport {
      pub overall_drift_score: f64,
      pub drifted_features: Vec<(String, f64)>,  // (feature, drift score)
      pub reason_code_drift: f64,
      pub alert_level: AlertLevel,  // None, Warning, Critical
  }
  ```
- [ ] Alerting integration hooks
- [ ] Drift root cause analysis helpers

**Regulatory Link:** OCC 2011-12 requires ongoing model validation including explanation consistency

---

## Phase 6: KernelSHAP (Model-Agnostic)

**Goal:** Extend to any model type (not just trees)

**Deliverables:**
- [ ] KernelSHAP implementation
- [ ] Background sampling strategies
  - K-means summarization
  - Random sampling
  - Stratified sampling
- [ ] Convergence detection (early stopping when values stabilize)
- [ ] Stability metrics for KernelSHAP (inherently noisier than TreeSHAP)
- [ ] Batched model evaluation for efficiency

**Note:** KernelSHAP is slower but works with any model (neural nets, ensembles, etc.)

---

## Dependency Graph

```
Phase 1 (TreeSHAP)
    │
    ├──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼              │
Phase 2        Phase 4        Phase 5           │
(Stability)    (Reason Codes) (Drift)           │
    │                                           │
    ▼                                           │
Phase 3                                         │
(Audit)                                         │
                                                │
Phase 6 (KernelSHAP) ◄──────────────────────────┘
(can start independently)
```

**Parallelizable work:**
- Phases 2, 4, 5 can run in parallel after Phase 1
- Phase 6 is independent (can start anytime)
- Phase 3 depends on Phase 2 (needs stability metrics for audit events)

---

## Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Performance vs Python shap | ≥15x on 16 cores |
| 1 | Equivalence vs Python shap | ε < 10⁻⁵ |
| 2 | CI coverage accuracy | 95% CI covers true value 95% of time |
| 3 | Audit completeness | 100% of explanations logged |
| 4 | Reason code accuracy | Matches Python ReasonCodeGenerator |
| 5 | Drift detection | AUC > 0.9 for synthetic drift |
| 6 | KernelSHAP convergence | Stable within 1000 samples |

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
2. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." Nature Machine Intelligence.
3. CFPB Regulation B (12 CFR 1002.9) - Adverse Action Notices
4. OCC 2011-12 - Sound Practices for Model Risk Management
5. SR 11-7 - Guidance on Model Risk Management

---

## Change Log

**2026-04-29:**
- ✅ Added Python API design section
- ✅ Completed module structure (models/, explainability/)
- ✅ Naive SHAP implementation working (tests passing)
- 📝 Documented design rationale in private/dev-notes/SHAP_API_DESIGN.md

**2026 (initial):**
- 📋 Created roadmap and phase breakdown

---

*Last updated: 2026-04-29*