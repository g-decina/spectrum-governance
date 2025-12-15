"""
Tests for ReportBuilder Class

Tests HTML/PDF report generation from templates.
"""

import pytest
import os
import tempfile
from pathlib import Path

from spectrum.lens.report_builder import ReportBuilder


@pytest.fixture
def temp_template_dir():
    """Create a temporary directory for templates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_audit_data():
    """Sample audit data for report generation."""
    return {
        "model_name": "RandomForestClassifier",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.95,
        "empirical_coverage": 0.93,
        "adversarial_metrics": {
            "attack_type": "HopSkipJump",
            "attack_success_rate": 0.15,
            "samples_tested": 100,
            "samples_successful": 15,
            "empirical_robustness_l2": 2.34,
            "empirical_robustness_linf": 0.45,
            "queries_used": 100000,
            "avg_queries_per_sample": 1000.0,
        },
        "sample_adverse_reasons": [
            "Credit Score below threshold (620)",
            "Debt-to-Income ratio too high (0.45)"
        ],
        "data_drift_status": "No Drift Detected",
        "data_drift_alert": False,
        "lineage_run_id": "test-run-123",
        "audit_log_path": "/var/log/rcia/audit.jsonl",
        "timestamp": "2024-01-15T10:30:00Z"
    }


def test_report_builder_initialization():
    """Test ReportBuilder initialization."""
    print("\n=== Test: ReportBuilder Initialization ===")

    builder = ReportBuilder()

    print(f"Template directory: {builder.template_dir}")
    print(f"Environment configured: {builder.env is not None}")

    assert builder.env is not None
    assert isinstance(builder.template_dir, str)


def test_report_builder_custom_template_dir(temp_template_dir):
    """Test ReportBuilder with custom template directory."""
    print("\n=== Test: Custom Template Directory ===")

    builder = ReportBuilder(template_dir=temp_template_dir)

    print(f"Custom template directory: {builder.template_dir}")
    print(f"Directory exists: {os.path.exists(builder.template_dir)}")

    assert builder.template_dir == temp_template_dir
    assert os.path.exists(builder.template_dir)


def test_fallback_html_generation(sample_audit_data):
    """Test fallback HTML generation when template is missing."""
    print("\n=== Test: Fallback HTML Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)

        # Try to generate with non-existent template
        html = builder.generate_html(
            template_name="nonexistent_template.html",
            data=sample_audit_data
        )

        print(f"Generated HTML length: {len(html)} characters")
        print(f"Contains model name: {'RandomForestClassifier' in html}")
        print(f"Contains attack success rate: {'0.15' in html}")

        assert html is not None
        assert len(html) > 0
        assert "RandomForestClassifier" in html
        assert "Compliance Audit Report" in html
        # Check that adversarial_metrics data is present
        assert "0.15" in html or "attack_success_rate" in html.lower()


def test_fallback_html_save_to_file(sample_audit_data):
    """Test saving fallback HTML to file."""
    print("\n=== Test: Save Fallback HTML to File ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        output_path = os.path.join(tmpdir, "test_report.html")

        html = builder.generate_html(
            template_name="missing_template.html",
            data=sample_audit_data,
            output_path=output_path
        )

        print(f"Output path: {output_path}")
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"File size: {os.path.getsize(output_path)} bytes")

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "RandomForestClassifier" in content
            assert "HIGH" in content


def test_custom_template_rendering(temp_template_dir, sample_audit_data):
    """Test rendering with a custom template."""
    print("\n=== Test: Custom Template Rendering ===")

    # Create a simple test template
    template_path = os.path.join(temp_template_dir, "test_template.html")
    with open(template_path, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head><title>Test Report</title></head>
<body>
    <h1>Model: {{ model_name }}</h1>
    <p>Type: {{ model_type }}</p>
    <p>Attack Success Rate: {{ adversarial_metrics.attack_success_rate }}</p>
    <p>Risk: {{ risk_level }}</p>
</body>
</html>
""")

    builder = ReportBuilder(template_dir=temp_template_dir)
    html = builder.generate_html(
        template_name="test_template.html",
        data=sample_audit_data
    )

    print(f"Template rendered successfully")
    print(f"Contains model name: {'RandomForestClassifier' in html}")
    print(f"Contains attack success rate: {'0.15' in html}")

    assert "RandomForestClassifier" in html
    assert "Tabular" in html
    assert "0.15" in html
    assert "HIGH" in html


def test_template_with_loops(temp_template_dir, sample_audit_data):
    """Test template with Jinja2 loops."""
    print("\n=== Test: Template with Loops ===")

    # Create template with loop
    template_path = os.path.join(temp_template_dir, "loop_template.html")
    with open(template_path, 'w') as f:
        f.write("""
<html>
<body>
    <h1>Adverse Reasons:</h1>
    <ul>
    {% for reason in sample_adverse_reasons %}
        <li>{{ reason }}</li>
    {% endfor %}
    </ul>
</body>
</html>
""")

    builder = ReportBuilder(template_dir=temp_template_dir)
    html = builder.generate_html(
        template_name="loop_template.html",
        data=sample_audit_data
    )

    print(f"Rendered template with loop")
    print(f"Number of reasons: {len(sample_audit_data['sample_adverse_reasons'])}")

    assert "Credit Score below threshold" in html
    assert "Debt-to-Income ratio too high" in html
    assert "<li>" in html


def test_template_with_conditionals(temp_template_dir, sample_audit_data):
    """Test template with Jinja2 conditionals."""
    print("\n=== Test: Template with Conditionals ===")

    template_path = os.path.join(temp_template_dir, "conditional_template.html")
    with open(template_path, 'w') as f:
        f.write("""
<html>
<body>
    {% if data_drift_alert %}
        <p class="alert">DRIFT ALERT: {{ data_drift_status }}</p>
    {% else %}
        <p class="ok">No drift detected</p>
    {% endif %}
</body>
</html>
""")

    builder = ReportBuilder(template_dir=temp_template_dir)

    # Test with no drift alert
    html_no_alert = builder.generate_html(
        template_name="conditional_template.html",
        data=sample_audit_data
    )

    print("Testing conditional: no drift alert")
    assert "No drift detected" in html_no_alert
    assert "DRIFT ALERT" not in html_no_alert

    # Test with drift alert
    drift_data = sample_audit_data.copy()
    drift_data["data_drift_alert"] = True
    drift_data["data_drift_status"] = "CRITICAL DRIFT"

    html_with_alert = builder.generate_html(
        template_name="conditional_template.html",
        data=drift_data
    )

    print("Testing conditional: with drift alert")
    assert "DRIFT ALERT" in html_with_alert
    assert "CRITICAL DRIFT" in html_with_alert


def test_pdf_generation_creates_html(sample_audit_data):
    """Test that PDF generation creates HTML (since PDF not yet implemented)."""
    print("\n=== Test: PDF Generation (HTML Fallback) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        pdf_path = os.path.join(tmpdir, "report.pdf")

        builder.generate_pdf(
            template_name="test_template.html",
            data=sample_audit_data,
            output_path=pdf_path
        )

        # Should create HTML file instead
        html_path = pdf_path.replace('.pdf', '.html')

        print(f"PDF path: {pdf_path}")
        print(f"HTML path: {html_path}")
        print(f"HTML exists: {os.path.exists(html_path)}")

        assert os.path.exists(html_path)


def test_fallback_html_structure(sample_audit_data):
    """Test that fallback HTML has proper structure."""
    print("\n=== Test: Fallback HTML Structure ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        html = builder._generate_fallback_html(sample_audit_data)

        print("Checking HTML structure...")
        print(f"  Has DOCTYPE: {'DOCTYPE' in html}")
        print(f"  Has <html> tag: {'<html>' in html}")
        print(f"  Has <head> tag: {'<head>' in html}")
        print(f"  Has <body> tag: {'<body>' in html}")
        print(f"  Has CSS styles: {'<style>' in html}")

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "<style>" in html
        assert "</html>" in html


def test_fallback_html_all_sections(sample_audit_data):
    """Test that fallback HTML includes all expected sections."""
    print("\n=== Test: Fallback HTML Sections ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        html = builder._generate_fallback_html(sample_audit_data)

        expected_sections = [
            "Model Information",
            "Performance Metrics",
            "Security Assessment",
            "Data Drift Monitoring",
            "Audit Trail"
        ]

        print("Checking for expected sections...")
        for section in expected_sections:
            print(f"  {section}: {section in html}")
            assert section in html, f"Missing section: {section}"


def test_fallback_html_data_values(sample_audit_data):
    """Test that fallback HTML contains actual data values."""
    print("\n=== Test: Fallback HTML Data Values ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        html = builder._generate_fallback_html(sample_audit_data)

        expected_values = [
            "RandomForestClassifier",
            "Tabular",
            "HIGH",
            "0.95",
            "0.15",  # attack_success_rate
            "test-run-123",
            "/var/log/rcia/audit.jsonl"
        ]

        print("Checking for data values...")
        for value in expected_values:
            print(f"  {value}: {value in html}")
            assert value in html, f"Missing value: {value}"


def test_empty_template_directory():
    """Test behavior with empty template directory."""
    print("\n=== Test: Empty Template Directory ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)

        # Should still work, just use fallback
        html = builder.generate_html(
            template_name="any_template.html",
            data={"model_name": "Test", "model_type": "Test"}
        )

        print(f"Generated fallback HTML: {len(html)} characters")
        assert html is not None
        assert len(html) > 0


def test_html_autoescape():
    """Test that HTML content is properly escaped."""
    print("\n=== Test: HTML Autoescape ===")

    dangerous_data = {
        "model_name": "<script>alert('XSS')</script>",
        "model_type": "Tabular",
        "risk_level": "HIGH",
        "confidence_required": 0.9,
        "adversarial_metrics": {
            "attack_type": "Test",
            "attack_success_rate": 0.1,
        },
        "sample_adverse_reasons": ["<b>Bold</b>"],
        "data_drift_status": "OK",
        "data_drift_alert": False,
        "lineage_run_id": "test",
        "audit_log_path": "/tmp/test"
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create template
        template_path = os.path.join(tmpdir, "escape_test.html")
        with open(template_path, 'w') as f:
            f.write("<html><body>{{ model_name }}</body></html>")

        builder = ReportBuilder(template_dir=tmpdir)
        html = builder.generate_html(
            template_name="escape_test.html",
            data=dangerous_data
        )

        print("Checking HTML escaping...")
        print(f"  Script tag escaped: {'&lt;script&gt;' in html}")

        # Should be escaped
        assert "&lt;script&gt;" in html or "alert" not in html
        # Should NOT contain raw script
        assert "<script>alert" not in html


def test_multiple_report_generation(sample_audit_data):
    """Test generating multiple reports sequentially."""
    print("\n=== Test: Multiple Report Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)

        for i in range(3):
            output_path = os.path.join(tmpdir, f"report_{i}.html")
            html = builder.generate_html(
                template_name="test.html",
                data=sample_audit_data,
                output_path=output_path
            )

            print(f"Generated report {i + 1}: {output_path}")
            assert os.path.exists(output_path)

        # Check all files exist
        files = os.listdir(tmpdir)
        html_files = [f for f in files if f.endswith('.html')]

        print(f"Total HTML files generated: {len(html_files)}")
        assert len(html_files) == 3


def test_report_with_none_values():
    """Test report generation with None values."""
    print("\n=== Test: Report with None Values ===")

    data_with_nones = {
        "model_name": "TestModel",
        "model_type": "LLM",
        "risk_level": "HIGH",
        "confidence_required": 1.0,
        "empirical_coverage": None,  # None value
        "adversarial_metrics": {
            "attack_type": "PromptInjection",
            "attack_success_rate": 0.3,
        },
        "sample_adverse_reasons": [],
        "data_drift_status": "N/A",
        "data_drift_alert": False,
        "lineage_run_id": "test",
        "audit_log_path": "/tmp/test"
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        html = builder._generate_fallback_html(data_with_nones)

        print("Checking None value handling...")
        print(f"  Contains 'None': {'None' in html}")
        print(f"  Contains 'N/A': {'N/A' in html}")

        assert html is not None
        assert len(html) > 0


def test_adversarial_metrics_in_fallback_html(sample_audit_data):
    """Test that adversarial metrics are properly included in fallback HTML."""
    print("\n=== Test: Adversarial Metrics in Fallback HTML ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ReportBuilder(template_dir=tmpdir)
        html = builder._generate_fallback_html(sample_audit_data)

        print("Checking adversarial metrics in HTML...")

        # Check for attack_success_rate
        assert "0.15" in html, "Attack success rate should be in HTML"

        # Check for attack type
        assert "HopSkipJump" in html, "Attack type should be in HTML"

        # Check Security Assessment section
        assert "Security Assessment" in html
        assert "Attack Method" in html or "attack_type" in html.lower()