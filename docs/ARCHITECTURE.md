# Spectrum Governance Architecture

## Library Overview

**Spectrum Governance** is a hybrid Rust/Python ML auditing framework designed for regulatory compliance. It provides a comprehensive toolkit for adversarial testing, defensive analysis, and audit logging—enabling organizations to demonstrate compliance with EU AI Act, CFPB, NIST AI RMF, and other governance frameworks.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SPECTRUM GOVERNANCE FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              │
│   │   spectrum.red  │   │  spectrum.blue  │   │  spectrum.lens  │              │
│   │   ────────────  │   │  ────────────── │   │  ────────────── │              │
│   │   Red Team      │   │  Blue Team      │   │  Governance     │              │
│   │   Adversarial   │   │  Defense        │   │  Audit & Report │              │
│   │   Attacks       │   │  & Monitoring   │   │                 │              │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘              │
│            │                     │                     │                        │
│            └─────────────────────┼─────────────────────┘                        │
│                                  │                                              │
│                    ┌─────────────▼─────────────┐                                │
│                    │     spectrum.infra        │                                │
│                    │     ─────────────────     │                                │
│                    │     Core Types, PII,      │                                │
│                    │     RCIA Logging          │                                │
│                    └───────────────────────────┘                                │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           RUST PERFORMANCE CORE                                  │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      spectrum-core (Rust Workspace)                      │   │
│   ├─────────────────┬─────────────────┬─────────────────┬───────────────────┤   │
│   │ spectrum-red    │ spectrum-blue   │ spectrum-lens   │ spectrum-common   │   │
│   │ (Attacks)       │ (SHAP, CP)      │ (Lineage)       │ (Shared Types)    │   │
│   └─────────────────┴─────────────────┴─────────────────┴───────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Hierarchy

### Python Layer (`spectrum/`)

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `spectrum.red` | Adversarial testing | HopSkipJumpWrapper, AdversarialMetrics |
| `spectrum.blue` | Defensive analysis | SpectrumUncertaintyWrapper, DriftCheck, ReasonCodeGenerator |
| `spectrum.lens` | Governance & audit | WargameRunner, LineageTracker, ComplianceReport |
| `spectrum.infra` | Infrastructure | RiskProfile, RCIALogger, PIIDetector |

### Rust Core (`spectrum-core/`)

| Crate | Purpose | Key Components |
|-------|---------|----------------|
| `spectrum-red` | High-performance attacks | HopSkipJumpAttack, ZOOAttack, BoundaryAttack |
| `spectrum-blue` | Fast SHAP/CP | (Planned: TreeSHAP, streaming CP) |
| `spectrum-lens` | Audit infrastructure | (Planned: fast logging, hash chain) |
| `spectrum-common` | Shared types | AdversarialMetrics, Model trait |

---

## Data Flow

### Wargame Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WARGAME EXECUTION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT                    PROCESSING                       OUTPUT              │
│   ─────                    ──────────                       ──────              │
│                                                                                  │
│   ┌─────────────┐         ┌─────────────────┐             ┌─────────────────┐   │
│   │ Model       │─────────│ WargameRunner   │─────────────│ ComplianceReport│   │
│   │ + Test Data │         │                 │             │ (.docx/.html)   │   │
│   │ + Calib Data│         │                 │             └─────────────────┘   │
│   └─────────────┘         │   ┌─────────┐   │                                   │
│                           │   │ RED     │   │             ┌─────────────────┐   │
│                           │   │ Attack  │◄──┼─────────────│ AdversarialMetrics│ │
│                           │   └─────────┘   │             └─────────────────┘   │
│                           │                 │                                   │
│                           │   ┌─────────┐   │             ┌─────────────────┐   │
│                           │   │ BLUE    │◄──┼─────────────│ PredictionResult │  │
│                           │   │ Defense │   │             │ (Coverage/SHAP)  │  │
│                           │   └─────────┘   │             └─────────────────┘   │
│                           │                 │                                   │
│                           │   ┌─────────┐   │             ┌─────────────────┐   │
│                           │   │ DRIFT   │◄──┼─────────────│ PSI Scores       │  │
│                           │   │ Monitor │   │             └─────────────────┘   │
│                           │   └─────────┘   │                                   │
│                           │                 │             ┌─────────────────┐   │
│                           └─────────────────┘             │ RCIA Audit Log  │   │
│                                    │                      │ (.jsonl)        │   │
│                                    └──────────────────────└─────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Adversarial Attack Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ADVERSARIAL ATTACK FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Python                    Rust Core                      Model                 │
│   ──────                    ─────────                      ─────                 │
│                                                                                  │
│   ┌─────────────────┐      ┌──────────────────┐       ┌─────────────────┐       │
│   │HopSkipJumpWrapper│─────│ HopSkipJumpAttack│───────│  ONNX Runtime   │       │
│   │  (Python API)   │      │    (Rust)        │       │  (for perf)     │       │
│   └─────────────────┘      │                  │       └─────────────────┘       │
│          │                 │  ┌────────────┐  │              │                  │
│          │                 │  │ Initialize │──┼──────────────┤ predict(x)       │
│          │                 │  │ (find adv) │  │              │                  │
│          │                 │  └─────┬──────┘  │              │                  │
│          │                 │        │         │              │                  │
│          │                 │  ┌─────▼──────┐  │              │                  │
│          │                 │  │ Binary     │──┼──────────────┤ predict(x)       │
│          │                 │  │ Search     │  │              │                  │
│          │                 │  └─────┬──────┘  │              │                  │
│          │                 │        │         │              │                  │
│          │                 │  ┌─────▼──────┐  │              │                  │
│          │                 │  │ Gradient   │──┼──────────────┤ predict(batch)   │
│          │                 │  │ Estimation │  │              │                  │
│          │                 │  └─────┬──────┘  │              │                  │
│          │                 │        │         │              │                  │
│          │                 │  ┌─────▼──────┐  │              │                  │
│          │                 │  │ Boundary   │──┼──────────────┤ predict(x)       │
│          │                 │  │ Walk       │  │              │                  │
│          │                 │  └────────────┘  │              │                  │
│          │                 │                  │              │                  │
│          │◄────────────────│ AdversarialMetrics              │                  │
│          │                 └──────────────────┘              │                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### spectrum.red - Red Team Module

**Purpose:** Adversarial testing to assess model robustness

**Key Classes:**

| Class | Description |
|-------|-------------|
| `HopSkipJumpWrapper` | Python wrapper for Rust HopSkipJump attack |
| `AdversarialMetrics` | Comprehensive metrics from attack execution |

**Algorithms Implemented:**

1. **HopSkipJump** (Chen et al., 2019) - Decision-based black-box attack
2. **ZOO** (Zeroth Order Optimization) - Gradient-free optimization attack
3. **Boundary Attack** (Brendel et al., 2018) - Random walk on decision boundary
4. **Square Attack** (Andriushchenko et al., 2020) - Query-efficient L∞ attack

**Performance:**
- 9.42x speedup over IBM ART (Rust+ONNX vs Python)
- Parallel execution across samples via Rayon
- Batched model queries for efficiency

### spectrum.blue - Blue Team Module

**Purpose:** Defensive analysis and uncertainty quantification

**Key Classes:**

| Class | Description |
|-------|-------------|
| `SpectrumUncertaintyWrapper` | Conformal prediction wrapper |
| `SpectrumRegressor` | Regression with prediction intervals |
| `SpectrumClassifier` | Classification with prediction sets |
| `DriftCheck` | PSI-based distribution shift detection |
| `ReasonCodeGenerator` | SHAP-to-adverse-action translation |
| `EnbPIRegressor` | Time series with prediction intervals |

**Mathematical Foundations:**

1. **Conformal Prediction**: P(Y ∈ Ĉ(X)) ≥ 1 - α
2. **Population Stability Index**: Symmetric KL divergence
3. **SHAP Values**: Shapley value-based feature attribution
4. **EnbPI**: Bootstrap-based time series intervals

### spectrum.lens - Governance Module

**Purpose:** Audit logging, lineage tracking, and compliance reporting

**Key Classes:**

| Class | Description |
|-------|-------------|
| `WargameRunner` | Orchestrates red vs blue testing |
| `LineageTracker` | OpenLineage-based data provenance |
| `ComplianceReport` | Pydantic model for audit artifacts |
| `ReportBuilder` | DOCX/HTML/PDF report generation |
| `RMFComplianceEngine` | NIST AI RMF requirement validation |

**Output Formats:**
- DOCX (professional Word documents)
- HTML (web-based reports)
- PDF (via HTML conversion)
- JSONL (RCIA audit logs)

### spectrum.infra - Infrastructure

**Purpose:** Core types, privacy, and logging infrastructure

**Key Classes:**

| Class | Description |
|-------|-------------|
| `RiskProfile` | Immutable governance contract |
| `InferenceEvent` | Audit log atom |
| `TargetModel` | Model specification contract |
| `RCIALogger` | Risk/Compliance/Inference/Audit logging |
| `PIIDetector` | Automatic PII detection |
| `PIISanitizer` | Payload sanitization for logs |
| `PIIDataSanitizer` | DataFrame PII handling |

---

## Rust/Python Interoperability

### PyO3 Bindings

The Rust core exposes Python bindings via PyO3:

```rust
// Rust side (spectrum-red/src/lib.rs)
#[pyfunction]
fn run_hopskipjump(
    py: Python<'_>,
    samples: &PyArray2<f64>,
    model: &PyAny,
    config: HopSkipJumpConfigPy,
) -> PyResult<AdversarialMetricsPy> {
    // ...
}
```

```python
# Python side (spectrum/red/attack.py)
from spectrum_red import run_hopskipjump

metrics = run_hopskipjump(
    samples=X_test,
    model=model_callback,
    config=config
)
```

### ONNX Optimization Path

For maximum performance, models can be converted to ONNX:

```python
# Convert sklearn model to ONNX
import skl2onnx

onnx_model = skl2onnx.convert_sklearn(sklearn_model)
onnx_model.save("model.onnx")

# Use ONNX path in attacks
wrapper = HopSkipJumpWrapper(base_model=model)
wrapper.use_onnx = True  # ~9x speedup
```

---

## Regulatory Mapping

| Regulation | Spectrum Component | Evidence Generated |
|------------|-------------------|-------------------|
| EU AI Act Article 15 | spectrum.red | Attack success rates, perturbation metrics |
| CFPB Regulation B | spectrum.blue | SHAP-based adverse action reasons |
| OCC SR 11-7 | spectrum.blue | Drift monitoring, coverage metrics |
| NIST AI RMF 1.0 | spectrum.lens | RMF compliance gap reports |
| GDPR Article 25 | spectrum.infra | PII detection and handling |

---

## Performance Characteristics

### Benchmark Results (50 features, 200 samples)

| Backend | Time | Speedup | ASR | L2 Perturbation |
|---------|------|---------|-----|-----------------|
| ART (Python) | 0.250s | 1.0x | 100% | 8.05 |
| Rust + ONNX | 0.027s | 9.4x | 100% | 8.47 |

### Scaling Properties

| Component | Complexity | Parallelization |
|-----------|------------|-----------------|
| HopSkipJump | O(n × iter × eval) | Per-sample (Rayon) |
| TreeSHAP | O(n × L × D²) | Per-sample |
| PSI | O(features × bins) | Per-feature |
| RCIA Logging | O(1) per event | Async I/O |

---

## Deployment Architecture

### Standalone Mode

```
┌─────────────────────────────────────────┐
│           Application Server            │
│  ┌─────────────────────────────────────┐│
│  │         ML Model                    ││
│  │         ┌─────────────────────────┐ ││
│  │         │  Spectrum Governance    │ ││
│  │         │  (Python + Rust Core)   │ ││
│  │         └─────────────────────────┘ ││
│  └─────────────────────────────────────┘│
│                     │                    │
│              ┌──────▼──────┐            │
│              │ RCIA Log    │            │
│              │ (.jsonl)    │            │
│              └─────────────┘            │
└─────────────────────────────────────────┘
```

### Enterprise Mode (with OpenLineage)

```
┌─────────────────────────────────────────┐       ┌───────────────────┐
│           Application Server            │       │    Marquez        │
│  ┌─────────────────────────────────────┐│       │  (Lineage UI)     │
│  │  Spectrum Governance                ││       └─────────▲─────────┘
│  │         │                           ││                 │
│  │  ┌──────▼──────┐   ┌──────────────┐ ││   OpenLineage   │
│  │  │LineageTracker│──│ RCIA Logger  │─┼┼────────────────┘
│  │  └─────────────┘   └──────────────┘ ││
│  └─────────────────────────────────────┘│
│                                          │
└──────────────────────────────────────────┘
```

---

## Extension Points

### Custom Attack Implementation

```python
from spectrum.red.attack import BaseAttack
from spectrum.red.metrics import AdversarialMetrics

class MyCustomAttack(BaseAttack):
    def run(self, X_test: np.ndarray) -> AdversarialMetrics:
        # Custom attack logic
        return AdversarialMetrics.for_evasion_attack(...)
```

### Custom Report Template

```python
from spectrum.lens.report_builder import ReportBuilder

builder = ReportBuilder(template_dir="/path/to/custom/templates")
builder.generate_html(
    template_name="my_custom_report.html",
    data=audit_data,
    output_path="report.html"
)
```

### Custom PII Detection

```python
from spectrum.infra.privacy import PIIDetector, PIISchema, PIIFieldType

schema = PIISchema(
    pii_fields={
        "custom_id_field": PIIFieldType.IDENTIFIER,
        "sensitive_field": PIIFieldType.SENSITIVE
    },
    safe_fields={"non_pii_field"}
)

detector = PIIDetector(schema=schema)
```

---

## Installation

### Standard Installation

```bash
pip install spectrum-governance
```

### With All Dependencies

```bash
pip install spectrum-governance[all]
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/org/spectrum-governance.git
cd spectrum-governance

# Install Python package
pip install -e ".[dev]"

# Build Rust core
cd spectrum-core
cargo build --release

# Install Rust bindings
cd spectrum-red
maturin develop --release
```

---

## Directory Structure

```
spectrum-governance/
├── spectrum/                    # Python package
│   ├── red/                     # Adversarial testing
│   │   ├── attack.py            # Attack wrappers
│   │   └── metrics.py           # AdversarialMetrics
│   ├── blue/                    # Defense & monitoring
│   │   ├── trust.py             # Conformal prediction
│   │   ├── explain.py           # SHAP explanations
│   │   ├── monitor.py           # Drift detection
│   │   └── timeseries.py        # EnbPI
│   ├── lens/                    # Governance
│   │   ├── wargame_runner.py    # Orchestration
│   │   ├── lineage.py           # OpenLineage
│   │   ├── compliance_report.py # Report models
│   │   ├── report_builder.py    # DOCX/HTML generation
│   │   └── rmf.py               # NIST AI RMF
│   └── infra/                   # Infrastructure
│       ├── types.py             # Core types
│       ├── privacy.py           # PII handling
│       └── logger.py            # RCIA logging
│
├── spectrum-core/               # Rust workspace
│   ├── spectrum-red/            # Rust attacks
│   │   └── src/
│   │       ├── lib.rs           # PyO3 bindings
│   │       ├── attacks/         # Attack implementations
│   │       │   ├── hop_skip_jump.rs
│   │       │   ├── zoo.rs
│   │       │   ├── boundary.rs
│   │       │   └── square.rs
│   │       ├── model.rs         # Model trait
│   │       └── metrics.rs       # Rust metrics
│   ├── spectrum-blue/           # Rust defense (planned)
│   ├── spectrum-lens/           # Rust logging (planned)
│   └── spectrum-common/         # Shared types
│
├── tests/                       # Test suite
│   ├── fixtures/                # ART/SHAP reference data
│   └── benchmark_attacks.py     # Performance benchmarks
│
└── docs/                        # Documentation
    ├── SPECTRUM_RED_GUIDE.md
    ├── SPECTRUM_BLUE_GUIDE.md
    ├── SPECTRUM_LENS_GUIDE.md
    ├── SPECTRUM_INFRA_TECHNICAL_NOTE.md
    └── ARCHITECTURE.md          # This file
```

---

## Future Roadmap

### Planned Features

1. **spectrum.red**
   - Transfer attack support
   - Ensemble attack strategies
   - Certified defense verification

2. **spectrum.blue (Rust)**
   - High-performance TreeSHAP
   - Streaming conformal prediction
   - GPU-accelerated computations

3. **spectrum.lens**
   - Blockchain-based audit trail
   - Real-time compliance dashboard
   - Automated remediation suggestions

4. **spectrum.infra**
   - Distributed logging
   - Multi-tenant isolation
   - Kubernetes operator

---

## References

1. Chen, J., et al. (2019). "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack."
2. Vovk, V., et al. (2005). "Algorithmic Learning in a Random World."
3. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions."
4. NIST AI Risk Management Framework 1.0 (2023)
5. EU AI Act Regulation (EU) 2024/1689

---

*Last updated: 2024*
