# Spectrum Governance - Quick Reference

**1-Page Cheat Sheet for Developers**

---

## Installation

```bash
# Basic install
pip install -e .

# With Blue Team (recommended)
pip install -e ".[blue]"

# With Red Team
pip install -e ".[red_torch]"  # or red_tensorflow

# Everything
pip install -e ".[blue,red_torch,server]"
```

---

## Core Concepts

### Risk Levels
```python
from spectrum.infra.types import RiskProfile, RiskLevel

# HIGH risk: 95% confidence required
high_risk = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)

# MEDIUM risk: 90% confidence
medium_risk = RiskProfile(level=RiskLevel.MEDIUM, alpha=0.10)

# LOW risk: 80% confidence
low_risk = RiskProfile(level=RiskLevel.LOW, alpha=0.20)
```

### Model Types
```python
from spectrum.infra.types import TargetModel

# Tabular model (sklearn, XGBoost, etc.)
target = TargetModel(
    model_name="credit_scorer",
    is_tabular=True,
    model_object=sklearn_model
)

# LLM model (API endpoint)
target = TargetModel(
    model_name="gpt4_assistant",
    is_tabular=False,
    target_api_url="https://api.openai.com/v1/chat/completions"
)
```

---

## Blue Team - Defense

### 1. Uncertainty Quantification
```python
from spectrum.blue.explain import SpectrumUncertaintyWrapper

# Setup
wrapper = SpectrumUncertaintyWrapper(
    base_model=model,
    risk_profile=RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
)

# Calibrate
wrapper.fit(X_calib, y_calib)

# Predict with uncertainty
results = wrapper.predict(X_test)
print(results.y_preds)  # Point predictions
print(results.y_pis)    # Prediction intervals (n, 2, 1)

# Check coverage
coverage = np.mean((y_test >= results.y_pis[:,0,0]) &
                   (y_test <= results.y_pis[:,1,0]))
```

### 2. Drift Monitoring
```python
from spectrum.blue.monitor import DriftCheck

# Check drift
drift = DriftCheck(X_reference, X_current)

# Results
print(drift['drift_detected'])      # bool
print(drift['n_drifted_features'])  # int
print(drift['max_psi'])             # float
print(drift['alert_required'])      # bool
print(drift['status'])              # str: "✓", "⚡", "⚠"
print(drift['feature_drift_scores'])# dict[str, float]
print(drift['drifted_features'])    # list[str]

# With specific features
drift = DriftCheck(X_ref, X_current,
                   feature_names=['age', 'income'])
```

### 3. Explanations
```python
from spectrum.blue.explain import generate_shap_explanations

# Generate SHAP explanations
exp = generate_shap_explanations(
    model=model,
    X_test=X_test,
    max_samples=10
)

print(exp['top_features'])  # list[str]
```

---

## Red Team - Attacks

### Adversarial Attack
```python
from spectrum.red.attack import HopSkipJumpWrapper

# Run attack
attack = HopSkipJumpWrapper(base_model=model)
fragility_score = attack.run(X_test)

print(f"Fragility: {fragility_score}")
# 0.0-0.1: Robust
# 0.1-0.3: Moderate risk
# 0.3+: Fragile
```

⚠️ **Warning**: Red Team is 40% complete. Most attacks are stubs.

---

## Lens - Governance

### 1. Lineage Tracking
```python
from spectrum.lens.lineage import LineageTracker

# Initialize
tracker = LineageTracker(job_name="credit_audit")

# Start
tracker.log_start(
    input_features=["age", "income", "credit_score"],
    documentation="Monthly credit model audit"
)

# Heartbeat (optional, for long jobs)
tracker.log_running()

# Complete
tracker.log_end(audit_summary={
    "fragility_score": 0.15,
    "confidence_required": 0.95,
    "audit_hash": "abc123"
})

# Or fail
tracker.log_failure(error_message="Validation failed")

# Or abort
tracker.log_abort(abort_reason="User cancelled")
```

**Environment Variables**:
```bash
export OPENLINEAGE_URL="http://localhost:5000"
export OPENLINEAGE_NAMESPACE="production"
export OPENLINEAGE_ENABLED="true"
```

### 2. Compliance Report
```python
from spectrum.lens.compliance_report import ComplianceReport

report = ComplianceReport(
    model_name="RandomForestClassifier",
    model_type="Tabular",
    risk_level="HIGH",
    confidence_required=0.95,
    empirical_coverage=0.94,
    attack_method="HopSkipJump",
    fragility_score=0.15,
    sample_adverse_reasons=["Feature A high", "Feature B low"],
    data_drift_status="✓ No Drift",
    data_drift_alert=False,
    lineage_run_id="run-123",
    audit_log_path="/var/log/audit.jsonl"
)

# Export
report_dict = report.model_dump()
report_json = report.model_dump_json()
```

### 3. Report Builder
```python
from spectrum.lens.report_builder import ReportBuilder

builder = ReportBuilder()

# Generate HTML
html = builder.generate_html(
    template_name="audit.html",
    data=report.model_dump(),
    output_path="/tmp/report.html"
)

# Generate PDF (creates HTML for now)
builder.generate_pdf(
    template_name="audit.html",
    data=report.model_dump(),
    output_path="/tmp/report.pdf"
)
```

---

## Complete Example

```python
# 1. Setup
from spectrum.infra.types import RiskProfile, RiskLevel
from spectrum.blue.explain import SpectrumUncertaintyWrapper
from spectrum.blue.monitor import DriftCheck
from spectrum.red.attack import HopSkipJumpWrapper
from spectrum.lens.lineage import LineageTracker

risk_profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
tracker = LineageTracker(job_name="audit")
tracker.log_start(input_features=X_test.columns.tolist())

# 2. Blue Team: Uncertainty
wrapper = SpectrumUncertaintyWrapper(model, risk_profile)
wrapper.fit(X_calib, y_calib)
results = wrapper.predict(X_test)

# 3. Blue Team: Drift
drift = DriftCheck(X_reference, X_current)

# 4. Red Team: Attack
attack = HopSkipJumpWrapper(model)
fragility = attack.run(X_test)

# 5. Report
from spectrum.lens.compliance_report import ComplianceReport
report = ComplianceReport(
    model_name=model.__class__.__name__,
    model_type="Tabular",
    risk_level="HIGH",
    confidence_required=0.95,
    empirical_coverage=0.94,
    fragility_score=fragility,
    data_drift_status=drift['status'],
    data_drift_alert=drift['alert_required'],
    # ... other fields
)

tracker.log_end(audit_summary=report.model_dump())
```

---

## Testing

```bash
# Unit tests
pytest tests/test_compliance_report.py -v
pytest tests/test_report_builder.py -v
pytest tests/test_wargame_runner_helpers.py -v

# End-to-end test
python tests/test_e2e_credit_scoring.py

# With printouts
pytest tests/test_compliance_report.py -s

# Coverage
pytest tests/ --cov=src/spectrum --cov-report=html
```

---

## Thresholds & Constants

### Drift Monitoring
```python
PSI_MONITOR_THRESHOLD = 0.1   # Warning threshold
PSI_ALERT_THRESHOLD = 0.25    # Critical threshold

# PSI Interpretation:
# 0.0-0.1: No drift
# 0.1-0.25: Moderate drift (monitor)
# 0.25+: Critical drift (alert)
```

### Risk Levels
```python
RiskLevel.HIGH   # alpha=0.05 (95% confidence)
RiskLevel.MEDIUM # alpha=0.10 (90% confidence)
RiskLevel.LOW    # alpha=0.20 (80% confidence)
```

---

## File Locations

```
src/spectrum/
├── infra/types.py           # RiskProfile, TargetModel
├── infra/logger.py          # RCIALogger
├── blue/explain.py          # SpectrumUncertaintyWrapper
├── blue/monitor.py          # DriftCheck
├── red/attack.py            # HopSkipJumpWrapper
├── lens/lineage.py          # LineageTracker
├── lens/compliance_report.py # ComplianceReport
└── lens/report_builder.py  # ReportBuilder

tests/
└── test_e2e_credit_scoring.py  # Complete example

docs/
└── LINEAGE.md              # Full lineage docs
```

---

## Common Issues

### Import Errors
```bash
# Missing dependencies
pip install -e ".[blue]"

# Missing openlineage
pip install openlineage-python
```

### Lineage Connection Errors
```bash
# Start Marquez
docker-compose -f docker-compose.marquez.yml up -d

# Or disable lineage
export OPENLINEAGE_ENABLED=false
```

### SHAP Slow
```python
# Reduce samples
generate_shap_explanations(model, X_test, max_samples=5)
```

---

## Quick Commands

```bash
# Run full test
python tests/test_e2e_credit_scoring.py

# Start Marquez
docker-compose -f docker-compose.marquez.yml up -d

# View Marquez UI
open http://localhost:3000

# Run linter
ruff check src/

# Format code
ruff format src/
```

---

## Status Symbols

- ✅ Feature complete and tested
- ⚠️ Feature partial or has issues
- ❌ Feature missing or not implemented
- 🔴 Critical gap
- 🟡 Important gap
- 🟢 Minor issue

---

**For Full Documentation**: See [README.md](README.md) and [LIBRARY_ASSESSMENT.md](LIBRARY_ASSESSMENT.md)

**Last Updated**: December 2, 2024
