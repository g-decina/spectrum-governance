"""
Tests for ComplianceReport Pydantic Model

Tests validation, required fields, and data structure.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from spectrum.lens.compliance_report import ComplianceReport


def _make_adversarial_metrics(attack_success_rate: float = 0.15, **overrides) -> dict:
    """Helper to create adversarial_metrics dict with sensible defaults."""
    base = {
        "attack_type": "HopSkipJump",
        "attack_success_rate": attack_success_rate,
        "samples_tested": 100,
        "samples_successful": int(attack_success_rate * 100),
        "empirical_robustness_l2": 2.34,
        "empirical_robustness_linf": 0.45,
        "queries_used": 100000,
        "avg_queries_per_sample": 1000.0,
    }
    base.update(overrides)
    return base


def test_compliance_report_valid_creation():
    """Test creating a valid compliance report with all required fields."""
    print("\n=== Test: Valid ComplianceReport Creation ===")

    report_data = {
        "model_name": "RandomForestClassifier",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "empirical_coverage": 0.94,
        "adversarial_metrics": _make_adversarial_metrics(attack_success_rate=0.15),
        "sample_adverse_reasons": ["Feature A below threshold", "Feature B too high"],
        "data_drift_status": "No Drift Detected",
        "data_drift_alert": False,
        "lineage_run_id": "test-run-id-123",
        "audit_log_path": "/var/log/rcia/audit.jsonl"
    }

    report = ComplianceReport(**report_data)

    print(f"Model Name: {report.model_name}")
    print(f"Model Type: {report.model_type}")
    print(f"Risk Level: {report.risk_level}")
    print(f"Confidence Required: {report.confidence_required}")
    print(f"Empirical Coverage: {report.empirical_coverage}")
    print(f"Attack Success Rate: {report.adversarial_metrics['attack_success_rate']}")
    print(f"Timestamp: {report.timestamp}")

    assert report.model_name == "RandomForestClassifier"
    assert report.model_type == "Tabular"
    assert report.risk_level == "HIGH"
    assert report.confidence_required == 0.95
    assert report.empirical_coverage == 0.94
    assert report.adversarial_metrics["attack_success_rate"] == 0.15
    assert len(report.sample_adverse_reasons) == 2
    assert report.data_drift_alert is False
    assert isinstance(report.timestamp, datetime)


def test_compliance_report_missing_required_field():
    """Test that missing required fields raise ValidationError."""
    print("\n=== Test: Missing Required Field ===")

    incomplete_data = {
        "model_name": "RandomForestClassifier",
        # Missing model_type (required)
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "adversarial_metrics": _make_adversarial_metrics(),
        "data_drift_status": "No Drift",
        "data_drift_alert": False,
        "lineage_run_id": "test-run-id",
        "audit_log_path": "/var/log/audit.jsonl"
    }

    print("Attempting to create report without model_type...")

    with pytest.raises(ValidationError) as exc_info:
        ComplianceReport(**incomplete_data)

    print(f"ValidationError raised as expected: {exc_info.value}")
    assert "model_type" in str(exc_info.value)


def test_compliance_report_invalid_confidence_range():
    """Test that confidence_required must be between 0 and 1."""
    print("\n=== Test: Invalid Confidence Range ===")

    invalid_data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "MEDIUM",
        "confidence_required": 1.5,  # Invalid: > 1.0
        "empirical_coverage": 0.9,
        "adversarial_metrics": _make_adversarial_metrics(),
        "sample_adverse_reasons": [],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    print("Attempting to create report with confidence_required=1.5...")

    with pytest.raises(ValidationError) as exc_info:
        ComplianceReport(**invalid_data)

    print(f"ValidationError raised: {exc_info.value}")
    assert "confidence_required" in str(exc_info.value)


def test_compliance_report_invalid_coverage_range():
    """Test that empirical_coverage must be between 0 and 1."""
    print("\n=== Test: Invalid Empirical Coverage Range ===")

    invalid_data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "LOW",
        "confidence_required": 0.9,
        "empirical_coverage": -0.1,  # Invalid: < 0.0
        "adversarial_metrics": _make_adversarial_metrics(),
        "sample_adverse_reasons": [],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    print("Attempting to create report with empirical_coverage=-0.1...")

    with pytest.raises(ValidationError) as exc_info:
        ComplianceReport(**invalid_data)

    print(f"ValidationError raised: {exc_info.value}")
    assert "empirical_coverage" in str(exc_info.value)


def test_compliance_report_optional_coverage():
    """Test that empirical_coverage can be None."""
    print("\n=== Test: Optional Empirical Coverage ===")

    data_with_none = {
        "model_name": "LLMModel",
        "model_type": "LLM",
        "risk_level": "HIGH",
        "confidence_required": 1.0,
        "empirical_coverage": None,  # Optional
        "adversarial_metrics": _make_adversarial_metrics(
            attack_type="PromptInjection",
            attack_success_rate=0.30
        ),
        "sample_adverse_reasons": ["Injection successful"],
        "data_drift_status": "Not Applicable",
        "data_drift_alert": False,
        "lineage_run_id": "llm-test-id",
        "audit_log_path": "/tmp/llm.log"
    }

    report = ComplianceReport(**data_with_none)

    print(f"Empirical Coverage: {report.empirical_coverage}")
    assert report.empirical_coverage is None


def test_compliance_report_default_timestamp():
    """Test that timestamp is auto-generated if not provided."""
    print("\n=== Test: Default Timestamp Generation ===")

    data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "MEDIUM",
        "confidence_required": 0.9,
        "empirical_coverage": 0.88,
        "adversarial_metrics": _make_adversarial_metrics(),
        "sample_adverse_reasons": [],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    report = ComplianceReport(**data)

    print(f"Auto-generated timestamp: {report.timestamp}")
    assert isinstance(report.timestamp, datetime)
    # Check it's recent (within last minute)
    now = datetime.now(timezone.utc)
    time_diff = (now - report.timestamp).total_seconds()
    assert time_diff < 60


def test_compliance_report_empty_reasons_list():
    """Test that sample_adverse_reasons can be empty list."""
    print("\n=== Test: Empty Reasons List ===")

    data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "LOW",
        "confidence_required": 0.85,
        "empirical_coverage": 0.9,
        "adversarial_metrics": _make_adversarial_metrics(attack_success_rate=0.05),
        "sample_adverse_reasons": [],  # Empty list is valid
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    report = ComplianceReport(**data)

    print(f"Sample reasons: {report.sample_adverse_reasons}")
    assert report.sample_adverse_reasons == []
    assert len(report.sample_adverse_reasons) == 0


def test_compliance_report_model_dump():
    """Test converting report to dictionary."""
    print("\n=== Test: Model Dump to Dictionary ===")

    data = {
        "model_name": "XGBClassifier",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "empirical_coverage": 0.93,
        "adversarial_metrics": _make_adversarial_metrics(
            attack_type="HopSkipJump",
            attack_success_rate=0.18
        ),
        "sample_adverse_reasons": ["Reason 1", "Reason 2"],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "xgb-run-123",
        "audit_log_path": "/logs/xgb.log"
    }

    report = ComplianceReport(**data)
    report_dict = report.model_dump()

    print(f"Report dictionary keys: {report_dict.keys()}")
    print(f"Model name from dict: {report_dict['model_name']}")

    assert isinstance(report_dict, dict)
    assert report_dict["model_name"] == "XGBClassifier"
    assert report_dict["confidence_required"] == 0.95
    assert "timestamp" in report_dict
    assert "adversarial_metrics" in report_dict


def test_compliance_report_json_serialization():
    """Test JSON serialization of report."""
    print("\n=== Test: JSON Serialization ===")

    data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "MEDIUM",
        "confidence_required": 0.9,
        "empirical_coverage": 0.88,
        "adversarial_metrics": _make_adversarial_metrics(attack_success_rate=0.12),
        "sample_adverse_reasons": ["Test reason"],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    report = ComplianceReport(**data)
    json_str = report.model_dump_json()

    print(f"JSON output (first 200 chars): {json_str[:200]}...")

    assert isinstance(json_str, str)
    assert "TestModel" in json_str
    assert "test-id" in json_str
    assert "adversarial_metrics" in json_str


def test_compliance_report_with_drift_alert():
    """Test report when drift alert is triggered."""
    print("\n=== Test: Drift Alert Triggered ===")

    data = {
        "model_name": "DriftModel",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "empirical_coverage": 0.87,
        "adversarial_metrics": _make_adversarial_metrics(attack_success_rate=0.25),
        "sample_adverse_reasons": ["Drift detected"],
        "data_drift_status": "CRITICAL DRIFT",
        "data_drift_alert": True,  # Alert triggered
        "lineage_run_id": "drift-run",
        "audit_log_path": "/logs/drift.log"
    }

    report = ComplianceReport(**data)

    print(f"Drift status: {report.data_drift_status}")
    print(f"Drift alert: {report.data_drift_alert}")

    assert report.data_drift_alert is True
    assert "CRITICAL" in report.data_drift_status


def test_compliance_report_llm_type():
    """Test report for LLM model type."""
    print("\n=== Test: LLM Model Type ===")

    data = {
        "model_name": "GPT-4-API",
        "model_type": "LLM",
        "risk_level": "HIGH",
        "confidence_required": 1.0,
        "empirical_coverage": None,
        "adversarial_metrics": _make_adversarial_metrics(
            attack_type="PromptInjectionAttack",
            attack_success_rate=0.42
        ),
        "sample_adverse_reasons": [
            "Prompt injection successful",
            "Data exfiltration possible"
        ],
        "data_drift_status": "Not Applicable",
        "data_drift_alert": False,
        "lineage_run_id": "llm-attack-run",
        "audit_log_path": "/logs/llm_attack.log"
    }

    report = ComplianceReport(**data)

    print(f"Model type: {report.model_type}")
    print(f"Attack type: {report.adversarial_metrics['attack_type']}")
    print(f"Attack success rate: {report.adversarial_metrics['attack_success_rate']}")
    print(f"Reasons: {report.sample_adverse_reasons}")

    assert report.model_type == "LLM"
    assert report.empirical_coverage is None
    assert len(report.sample_adverse_reasons) == 2
    assert report.adversarial_metrics["attack_type"] == "PromptInjectionAttack"


def test_compliance_report_high_attack_success_rate():
    """Test report with high attack success rate (vulnerable model)."""
    print("\n=== Test: High Attack Success Rate ===")

    data = {
        "model_name": "VulnerableModel",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "empirical_coverage": 0.75,  # Low coverage
        "adversarial_metrics": _make_adversarial_metrics(
            attack_success_rate=0.85,  # High attack success = vulnerable
            attack_type="AdversarialAttack"
        ),
        "sample_adverse_reasons": [
            "Model easily fooled",
            "Low robustness to perturbations"
        ],
        "data_drift_status": "WARNING",
        "data_drift_alert": True,
        "lineage_run_id": "vulnerable-run",
        "audit_log_path": "/logs/vulnerable.log"
    }

    report = ComplianceReport(**data)

    attack_success_rate = report.adversarial_metrics["attack_success_rate"]
    print(f"Attack success rate: {attack_success_rate}")
    print(f"Empirical coverage: {report.empirical_coverage}")
    print(f"Risk assessment: {'FAILED' if attack_success_rate > 0.5 else 'PASSED'}")

    assert attack_success_rate == 0.85
    assert attack_success_rate > 0.5  # High risk
    assert report.empirical_coverage < 0.8  # Below expected


def test_compliance_report_field_types():
    """Test that all fields have correct types."""
    print("\n=== Test: Field Type Validation ===")

    data = {
        "model_name": "TypeTestModel",
        "model_type": "Tabular",
        "risk_level": "MEDIUM",
        "confidence_required": 0.9,
        "empirical_coverage": 0.88,
        "adversarial_metrics": _make_adversarial_metrics(attack_success_rate=0.15),
        "sample_adverse_reasons": ["Reason 1", "Reason 2"],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "type-test-id",
        "audit_log_path": "/tmp/type_test.log"
    }

    report = ComplianceReport(**data)

    print("Checking field types...")
    print(f"  model_name: {type(report.model_name).__name__}")
    print(f"  confidence_required: {type(report.confidence_required).__name__}")
    print(f"  empirical_coverage: {type(report.empirical_coverage).__name__}")
    print(f"  adversarial_metrics: {type(report.adversarial_metrics).__name__}")
    print(f"  sample_adverse_reasons: {type(report.sample_adverse_reasons).__name__}")
    print(f"  data_drift_alert: {type(report.data_drift_alert).__name__}")
    print(f"  timestamp: {type(report.timestamp).__name__}")

    assert isinstance(report.model_name, str)
    assert isinstance(report.confidence_required, float)
    assert isinstance(report.empirical_coverage, float)
    assert isinstance(report.adversarial_metrics, dict)
    assert isinstance(report.sample_adverse_reasons, list)
    assert isinstance(report.data_drift_alert, bool)
    assert isinstance(report.timestamp, datetime)


def test_compliance_report_adversarial_metrics_structure():
    """Test that adversarial_metrics contains expected fields."""
    print("\n=== Test: Adversarial Metrics Structure ===")

    adversarial_metrics = {
        "attack_type": "HopSkipJump",
        "attack_success_rate": 0.15,
        "samples_tested": 100,
        "samples_successful": 15,
        "empirical_robustness_l2": 2.34,
        "empirical_robustness_linf": 0.45,
        "queries_used": 100000,
        "avg_queries_per_sample": 1000.0,
    }

    data = {
        "model_name": "TestModel",
        "model_type": "Tabular",
        "risk_level": "MEDIUM",
        "confidence_required": 0.95,
        "empirical_coverage": 0.93,
        "adversarial_metrics": adversarial_metrics,
        "sample_adverse_reasons": [],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test-id",
        "audit_log_path": "/tmp/log"
    }

    report = ComplianceReport(**data)

    print(f"adversarial_metrics keys: {report.adversarial_metrics.keys()}")

    # Verify structure
    metrics = report.adversarial_metrics
    assert "attack_type" in metrics
    assert "attack_success_rate" in metrics
    assert "samples_tested" in metrics
    assert metrics["attack_type"] == "HopSkipJump"
    assert metrics["attack_success_rate"] == 0.15
    assert metrics["samples_tested"] == 100
