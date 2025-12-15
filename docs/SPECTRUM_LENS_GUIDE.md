# Spectrum Lens Guide

## Governance & Audit Module

**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

`spectrum.lens` provides comprehensive governance, lineage tracking, and compliance reporting for ML systems. The module transforms raw audit data into regulatory-compliant documentation, ensuring complete traceability from model training through production deployment.

### Core Capabilities

1. **Lineage Tracking** - OpenLineage-based data provenance and job tracking
2. **Compliance Reports** - Structured audit artifacts (DOCX, HTML, PDF)
3. **RMF Validation** - NIST AI RMF requirement verification
4. **RCIA Logging** - Risk, Compliance, Inference, and Audit event capture

```
spectrum.lens/
├── lineage.py           # OpenLineage integration (LineageTracker)
├── compliance_report.py # Pydantic models for audit artifacts
├── report_builder.py    # DOCX/HTML/PDF report generation
├── rmf.py               # NIST AI RMF compliance engine
├── wargame_runner.py    # Orchestration for automated audits
├── metrics_aggregator.py # Cross-module metrics collection
└── citations.py         # Regulatory citation database
```

---

## Quick Start

### Lineage Tracking

```python
from spectrum.lens import LineageTracker

# Initialize tracker
tracker = LineageTracker(
    job_name="credit_model_audit",
    namespace="production.credit"
)

# Log start of audit job
tracker.log_start(
    input_features=["income", "credit_score", "dti"],
    input_dataset_name="credit.applications.v2",
    documentation="Monthly model validation audit"
)

# ... perform audit operations ...

# Log completion with audit summary
tracker.log_end(
    audit_summary={
        "adversarial_metrics": {
            "attack_success_rate": 0.12,
            "attack_type": "HopSkipJump"
        },
        "drift_detected": False
    },
    output_dataset_name="credit.audit.rcia_log"
)
```

### Compliance Reports

```python
from spectrum.lens import ComplianceReport, ReportBuilder
from datetime import datetime, timezone

# Create structured report
report = ComplianceReport(
    model_name="CreditScoringModelV2",
    model_type="Tabular",
    risk_level="HIGH",
    confidence_required=0.95,
    empirical_coverage=0.94,
    adversarial_metrics={
        "attack_type": "HopSkipJump",
        "attack_success_rate": 0.12,
        "samples_tested": 100,
        "empirical_robustness_l2": 2.34
    },
    sample_adverse_reasons=["Credit Score below threshold"],
    data_drift_status="No Drift Detected",
    data_drift_alert=False,
    lineage_run_id="abc123",
    audit_log_path="/var/log/rcia/audit.jsonl"
)

# Generate professional DOCX report
builder = ReportBuilder()
builder.generate_docx(
    data=report.model_dump(),
    output_path="audit_report.docx",
    template_type="tier1_forensic_audit"
)
```

### RMF Compliance Analysis

```python
from spectrum.lens import RMFComplianceEngine

# Initialize compliance engine
engine = RMFComplianceEngine(
    artifact_directory="/path/to/governance/artifacts",
    rcia_log_path="/var/log/rcia/audit.jsonl"
)

# Validate artifact against RMF requirement
evidence = engine.validate_artifact(
    requirement_id="MEASURE-2.2-a",
    artifact_path="/artifacts/robustness_assessment.json"
)

# Check if technical control was executed
evidence = engine.check_technical_control(requirement_id="MEASURE-2.2-a")

# Generate comprehensive gap report
gap_report = engine.generate_gap_report()
print(f"Compliance: {gap_report['compliance_summary']}")
```

---

## Architectural Concepts

### OpenLineage Integration

Spectrum Lens uses the [OpenLineage](https://openlineage.io/) standard for data lineage tracking. OpenLineage provides a vendor-neutral specification for capturing dataset and job metadata.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenLineage Event Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   START Event                 RUNNING Event               COMPLETE │
│   ┌─────────┐                 ┌─────────┐               ┌─────────┐│
│   │Job: audit│ ───────────▶  │heartbeat│ ───────────▶  │ success ││
│   │Run: uuid │                │   ...   │               │ outputs ││
│   │Inputs:   │                └─────────┘               │ metrics ││
│   │ - dataset│                                          └─────────┘│
│   └─────────┘                                                     │
│                                                                   │
│   FAIL Event                  ABORT Event                         │
│   ┌─────────┐                 ┌─────────┐                         │
│   │ error   │                 │cancelled│                         │
│   │ message │                 │ reason  │                         │
│   └─────────┘                 └─────────┘                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Event Types

| Event | When to Use | Purpose |
|-------|-------------|---------|
| `START` | Beginning of audit job | Record inputs, configuration |
| `RUNNING` | Periodic heartbeat | Show job is active |
| `COMPLETE` | Successful completion | Record outputs, metrics |
| `FAIL` | Error occurred | Capture error details |
| `ABORT` | Job cancelled | Record cancellation reason |

### RCIA Logging Architecture

RCIA (Risk, Compliance, Inference, Audit) logging captures every model interaction:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RCIA Log Entry Structure                     │
├─────────────────────────────────────────────────────────────────┤
│ {                                                                │
│   "rcia_context": "INFERENCE",                                   │
│   "audit_hash": "sha256:a1b2c3...",  // Tamper detection        │
│   "data": {                                                      │
│     "event_id": "uuid",                                          │
│     "timestamp": "2024-01-15T10:30:00Z",                        │
│     "model_version": "v2.1.0",                                   │
│     "input_payload_sanitized": "<Object: numpy.ndarray>",        │
│     "output_payload_sanitized": {"prediction": 1, "prob": 0.87}, │
│     "input_shape": [1, 15],                                      │
│     "input_bytes": 120                                           │
│   }                                                              │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

**Key Properties:**

1. **Immutability**: SHA-256 hash proves entry hasn't been modified
2. **PII Sanitization**: Personal data replaced with placeholders
3. **Non-blocking**: Async logging doesn't impact inference latency
4. **Rotation**: Automatic log rotation at 10MB with compression

---

## API Reference

### LineageTracker

OpenLineage-based job and dataset lineage tracking.

```python
class LineageTracker:
    def __init__(
        self,
        job_name: str,                    # Unique job identifier
        namespace: str = None,            # OpenLineage namespace (env: OPENLINEAGE_NAMESPACE)
        url: str = None,                  # Backend URL (env: OPENLINEAGE_URL)
        enabled: bool = None              # Enable/disable (env: OPENLINEAGE_ENABLED)
    )

    def log_start(
        self,
        input_features: List[str],        # Features being processed
        input_dataset_name: str = "wargame.input.features",
        documentation: str = None,        # Job description
        source_code_location: str = None  # Git URL
    ) -> None:
        """Emit START event before job execution."""

    def log_end(
        self,
        audit_summary: Dict[str, Any],    # Must include 'adversarial_metrics'
        output_dataset_name: str = "wargame.output.rcia_log",
        output_uri: str = None            # Output storage location
    ) -> None:
        """Emit COMPLETE event after successful job."""

    def log_failure(
        self,
        error_message: str,
        error_type: str = None
    ) -> None:
        """Emit FAIL event when error occurs."""

    def log_running(self) -> None:
        """Emit RUNNING heartbeat for long jobs."""

    def log_abort(self, abort_reason: str = None) -> None:
        """Emit ABORT event when job is cancelled."""
```

### ComplianceReport

Pydantic model for structured audit artifacts.

```python
class ComplianceReport(BaseModel):
    # Model Information
    model_name: str
    model_type: str  # "Tabular", "LLM", etc.

    # Risk Assessment
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    confidence_required: float  # 1 - alpha
    empirical_coverage: Optional[float]

    # Attack Results
    adversarial_metrics: Dict  # From spectrum.red

    # Explainability
    sample_adverse_reasons: List[str]

    # Drift Monitoring
    data_drift_status: str
    data_drift_alert: bool

    # Audit Metadata
    lineage_run_id: str
    audit_log_path: str
    timestamp: datetime
```

### ReportBuilder

Generate professional compliance documents.

```python
class ReportBuilder:
    def __init__(self, template_dir: str = None):
        """Initialize with custom template directory."""

    def generate_docx(
        self,
        data: Dict[str, Any],
        output_path: str,
        template_type: str = "tier1_forensic_audit"
    ) -> str:
        """
        Generate DOCX report.

        template_type options:
        - "tier1_forensic_audit": Standard comprehensive audit
        - "eu_ai_act_annex_iv": EU AI Act Technical Documentation
        - "cfpb_model_validation": CFPB compliance report
        """

    def generate_html(
        self,
        template_name: str,
        data: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """Generate HTML report from Jinja2 template."""

    def generate_pdf(
        self,
        template_name: str,
        data: Dict[str, Any],
        output_path: str
    ) -> None:
        """Generate PDF (requires weasyprint)."""
```

### RMFComplianceEngine

NIST AI RMF requirement validation.

```python
class RMFComplianceEngine:
    def __init__(
        self,
        artifact_directory: str,  # Path to governance artifacts
        rcia_log_path: str        # Path to RCIA log file
    )

    def validate_artifact(
        self,
        requirement_id: str,      # e.g., "MEASURE-2.2-a"
        artifact_path: str        # JSON artifact to validate
    ) -> RMFEvidence:
        """Validate artifact against requirement schema."""

    def check_technical_control(
        self,
        requirement_id: str
    ) -> RMFEvidence:
        """Check RCIA logs for technical control execution."""

    def policy_enforcement_check(
        self,
        policy_path: str,         # Policy document (JSON)
        model_path: str           # Model to verify
    ) -> List[RMFComplianceStatus]:
        """Verify model meets stated policy thresholds."""

    def generate_gap_report(self) -> dict:
        """Generate comprehensive RMF compliance gap report."""
```

### Data Classes

```python
@dataclass
class RMFRequirement:
    function: Literal["GOVERN", "MAP", "MEASURE", "MANAGE"]
    category: str           # e.g., "1.1"
    subcategory: str        # e.g., "a"
    description: str
    artifact_schema: Optional[str]      # JSON schema for documentation
    technical_control: Optional[str]    # spectrum command for evidence
    evidence_query: Optional[str]       # RCIA log query pattern
    staleness_threshold: timedelta      # Evidence validity period

@dataclass
class RMFEvidence:
    requirement_id: str
    evidence_type: Literal["artifact", "technical_test", "log_entry"]
    evidence_date: datetime
    evidence_summary: str
    evidence_location: str
    valid_until: datetime

@dataclass
class RMFComplianceStatus:
    requirement: RMFRequirement
    status: Literal["COMPLIANT", "PARTIAL", "GAP", "STALE"]
    evidence: List[RMFEvidence]
    gap_description: Optional[str]
    remediation_suggestion: Optional[str]
```

---

## NIST AI RMF Mapping

The NIST AI Risk Management Framework organizes requirements into four functions:

### GOVERN
Organizational governance and accountability structures.

| Requirement | Description | Spectrum Evidence |
|-------------|-------------|-------------------|
| GOVERN-1.1 | Risk management policies | Policy artifacts |
| GOVERN-1.2 | Senior leadership oversight | Audit trail |
| GOVERN-2.1 | Clear roles and responsibilities | Documentation |

### MAP
Contextual analysis of AI systems.

| Requirement | Description | Spectrum Evidence |
|-------------|-------------|-------------------|
| MAP-1.1 | System purpose documentation | Model registry |
| MAP-2.1 | Stakeholder identification | Governance docs |
| MAP-3.1 | Impact assessment | Risk profiles |

### MEASURE
Risk assessment and measurement.

| Requirement | Description | Spectrum Evidence |
|-------------|-------------|-------------------|
| MEASURE-2.2-a | Adversarial robustness testing | `spectrum red` output |
| MEASURE-2.3 | Fairness assessment | Disparate impact metrics |
| MEASURE-2.7 | Accuracy and reliability | `spectrum blue` coverage |

### MANAGE
Risk treatment and monitoring.

| Requirement | Description | Spectrum Evidence |
|-------------|-------------|-------------------|
| MANAGE-2.1-a | Drift monitoring | PSI metrics from `DriftCheck` |
| MANAGE-3.1 | Incident response | RCIA logs |
| MANAGE-4.1 | Continuous monitoring | Production metrics |

---

## Report Templates

### Tier 1 Forensic Audit

Comprehensive audit report with sections:

1. **Cover Page** - Model identification, risk level, date
2. **Executive Summary** - Compliance status, key findings
3. **Scope & Methodology** - Testing approach, sample sizes
4. **Robustness Assessment** - Attack success rates, perturbation metrics
5. **Explainability Analysis** - Top features, SHAP analysis
6. **Drift Monitoring** - PSI results, feature-level analysis
7. **Uncertainty Quantification** - Coverage metrics
8. **Regulatory Mapping** - EU AI Act, CFPB requirements
9. **Recommendations** - Prioritized remediation items
10. **Technical Appendix** - Audit metadata, methodology details

### EU AI Act Annex IV

Technical documentation requirements:
- Design specifications
- Validation procedures
- Risk management measures
- Change management logs

### CFPB Model Validation

Consumer protection compliance:
- Model development documentation
- Adverse action compliance
- Ongoing monitoring procedures

---

## Environment Configuration

### OpenLineage Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENLINEAGE_URL` | `http://localhost:5000` | Backend server URL |
| `OPENLINEAGE_NAMESPACE` | `spectrum.governance` | Job namespace |
| `OPENLINEAGE_ENABLED` | `true` | Enable/disable tracking |

### Integration with Marquez

[Marquez](https://marquezproject.ai/) provides visualization for OpenLineage data:

```bash
# Start Marquez server
docker run -p 5000:5000 marquezproject/marquez

# Configure Spectrum to use Marquez
export OPENLINEAGE_URL=http://localhost:5000
```

---

## Best Practices

### Artifact Management

1. **Version artifacts** - Use semantic versioning for policy documents
2. **Schema validation** - All artifacts should pass JSON schema validation
3. **Staleness tracking** - Set appropriate staleness thresholds (typically 90 days)
4. **Central repository** - Store all governance artifacts in version control

### Audit Trail Integrity

1. **Immutable logs** - Never modify RCIA log files directly
2. **Hash verification** - Periodically verify SHA-256 hashes
3. **Backup strategy** - Maintain encrypted backups of audit logs
4. **Retention policy** - Follow regulatory retention requirements (typically 7 years)

### Compliance Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                  Continuous Compliance Workflow                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │ Define  │───▶│ Execute │───▶│ Collect │───▶│ Validate│    │
│   │ Policy  │    │ Tests   │    │Evidence │    │Compliance│   │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│       │                                              │         │
│       │                                              │         │
│       │           ┌─────────────────────────────────┘         │
│       │           │                                            │
│       │           ▼                                            │
│       │      ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│       └─────▶│ Report  │───▶│ Review  │───▶│ Remediate│       │
│              │ Gaps    │    │ Findings│    │ Issues   │       │
│              └─────────┘    └─────────┘    └─────────┘        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Example: Complete Audit Workflow

```python
from spectrum.lens import (
    LineageTracker,
    ComplianceReport,
    ReportBuilder,
    RMFComplianceEngine
)
from spectrum.red import AdversarialTester
from spectrum.blue import DriftCheck, SpectrumUncertaintyWrapper
from spectrum.infra.types import RiskProfile, RiskLevel
import uuid

# 1. Initialize lineage tracking
tracker = LineageTracker(
    job_name=f"monthly_audit_{uuid.uuid4().hex[:8]}",
    namespace="production.credit"
)

try:
    # 2. Log audit start
    tracker.log_start(
        input_features=feature_names,
        documentation="Monthly model validation - Q4 2024"
    )

    # 3. Run adversarial testing (spectrum.red)
    adv_metrics = AdversarialTester(model).run_hopskipjump(X_test)

    # 4. Check drift (spectrum.blue)
    drift_result = DriftCheck(reference_data, current_data)

    # 5. Validate uncertainty coverage
    trusted_model = SpectrumUncertaintyWrapper(model, RiskProfile(RiskLevel.HIGH, 0.05))
    result = trusted_model.predict(X_test)
    coverage = (result.y_preds == y_test).mean()

    # 6. Create compliance report
    report = ComplianceReport(
        model_name="CreditScorer_v2.1",
        model_type="Tabular",
        risk_level="HIGH",
        confidence_required=0.95,
        empirical_coverage=coverage,
        adversarial_metrics=adv_metrics.to_dict(),
        sample_adverse_reasons=["Credit score below threshold"],
        data_drift_status=drift_result['status'],
        data_drift_alert=drift_result['alert_required'],
        lineage_run_id=tracker.run_id,
        audit_log_path="/var/log/rcia/audit.jsonl"
    )

    # 7. Generate professional report
    builder = ReportBuilder()
    builder.generate_docx(
        data=report.model_dump(),
        output_path="audit_report_q4_2024.docx"
    )

    # 8. Validate RMF compliance
    engine = RMFComplianceEngine(
        artifact_directory="/governance/artifacts",
        rcia_log_path="/var/log/rcia/audit.jsonl"
    )
    gap_report = engine.generate_gap_report()

    # 9. Log successful completion
    tracker.log_end(
        audit_summary={
            "adversarial_metrics": adv_metrics.to_dict(),
            "drift_result": drift_result,
            "rmf_gaps": gap_report['compliance_summary']
        }
    )

    print(f"Audit complete. Report: audit_report_q4_2024.docx")
    print(f"RMF Status: {gap_report['compliance_summary']}")

except Exception as e:
    tracker.log_failure(str(e), type(e).__name__)
    raise
```

---

## Troubleshooting

### OpenLineage Connection Issues

```python
# ERROR: Failed to initialize OpenLineage client
# FIX: Check backend is running and accessible
# Or disable if not needed:
tracker = LineageTracker(job_name="audit", enabled=False)
```

### Missing python-docx

```bash
# ERROR: ImportError: python-docx is required for DOCX generation
pip install python-docx
```

### RMF Requirements Not Found

```python
# WARNING: RMF requirements file not found
# FIX: Create rmf_requirements.yaml in spectrum/lens/
# Or use default requirements
```

### Stale Evidence

```python
# STATUS: STALE - Evidence expired
# FIX: Re-run required tests to generate fresh evidence
# Evidence validity is configured per-requirement in RMF YAML
```

---

## References

1. **OpenLineage**: https://openlineage.io/

2. **Marquez**: https://marquezproject.ai/

3. **NIST AI RMF 1.0**: https://www.nist.gov/itl/ai-risk-management-framework

4. **EU AI Act**: Regulation (EU) 2024/1689

5. **CFPB Model Guidance**: Consumer Financial Protection Bureau Supervisory Guidance

6. **python-docx**: https://python-docx.readthedocs.io/

---

*Last updated: 2024*
