# OpenLineage + Marquez Integration

This document describes how to use the `spectrum.lens.lineage` module to track data lineage for governance workflows using OpenLineage and Marquez.

## Overview

The `LineageTracker` class captures the execution flow of Wargame runs, audit processes, and other governance operations. It emits OpenLineage events that can be visualized in Marquez or other compatible backends.

## Quick Start

### 1. Start Marquez (Local Development)

```bash
# Start Marquez backend + Web UI
docker-compose -f docker-compose.marquez.yml up -d

# Verify services are running
docker-compose -f docker-compose.marquez.yml ps

# View logs
docker-compose -f docker-compose.marquez.yml logs -f
```

**Access Points:**
- **OpenLineage API**: http://localhost:5000
- **Marquez Web UI**: http://localhost:3000
- **Admin API**: http://localhost:5001

### 2. Install Dependencies

```bash
# Install spectrum-governance with blue team extras (includes OpenLineage)
pip install -e ".[blue]"

# Or install OpenLineage separately
pip install openlineage-python
```

### 3. Basic Usage

```python
from spectrum.lens.lineage import LineageTracker

# Initialize tracker
tracker = LineageTracker(
    job_name="wargame_audit",
    namespace="spectrum.governance"
)

# Log job start
tracker.log_start(
    input_features=["age", "income", "credit_score"],
    documentation="Audit compliance check for credit model"
)

# ... perform analysis ...

# Log successful completion
tracker.log_end(
    audit_summary={
        "fragility_score": 0.85,
        "confidence_required": 0.95,
        "audit_hash": "abc123"
    }
)

# Or log failure
# tracker.log_failure("Model failed validation checks")
```

## Configuration

The `LineageTracker` can be configured via environment variables or constructor arguments:

### Environment Variables

```bash
# Backend URL (default: http://localhost:5000)
export OPENLINEAGE_URL="http://localhost:5000"

# Namespace for all jobs (default: spectrum.governance)
export OPENLINEAGE_NAMESPACE="production"

# Enable/disable lineage tracking (default: true)
export OPENLINEAGE_ENABLED="true"
```

### Constructor Arguments

```python
tracker = LineageTracker(
    job_name="my_audit_job",
    namespace="custom_namespace",  # Override default namespace
    url="http://marquez.example.com:5000",  # Override backend URL
    enabled=True  # Override enabled flag
)
```

## Advanced Features

### Job Documentation

Add documentation to jobs for better context:

```python
tracker.log_start(
    input_features=["feature1", "feature2"],
    documentation="This job audits model predictions for bias and fairness",
    source_code_location="https://github.com/org/repo/blob/main/audit.py"
)
```

### Heartbeat for Long-Running Jobs

For long-running jobs, send periodic heartbeat events:

```python
tracker.log_start(input_features=features)

# During long computation
tracker.log_running()  # Send heartbeat

# Later...
tracker.log_running()  # Another heartbeat

tracker.log_end(audit_summary=results)
```

### Aborting Jobs

Gracefully handle job cancellations:

```python
try:
    tracker.log_start(input_features=features)
    # ... work ...
except KeyboardInterrupt:
    tracker.log_abort(abort_reason="User cancelled operation")
    raise
```

### Error Tracking

Capture detailed error information:

```python
try:
    tracker.log_start(input_features=features)
    # ... work that might fail ...
    tracker.log_end(audit_summary=results)
except Exception as e:
    tracker.log_failure(
        error_message=str(e),
        error_type=type(e).__name__
    )
    raise
```

## Integration Examples

### With Wargame Runner

```python
from spectrum.lens.lineage import LineageTracker
from spectrum.red.wargame import WargameRunner

def run_audit_with_lineage(model, X_test, y_test):
    tracker = LineageTracker(job_name="wargame_audit")

    try:
        tracker.log_start(
            input_features=list(X_test.columns),
            documentation="Red team audit using adversarial attacks"
        )

        # Run wargame
        runner = WargameRunner(model=model)
        results = runner.run(X_test, y_test)

        # Log completion
        tracker.log_end(
            audit_summary={
                "fragility_score": results["fragility_score"],
                "confidence_required": results["confidence_required"],
                "audit_hash": results["audit_hash"]
            }
        )

        return results

    except Exception as e:
        tracker.log_failure(error_message=str(e))
        raise
```

### With Drift Monitoring

```python
from spectrum.lens.lineage import LineageTracker
from spectrum.blue.monitor import DriftCheck

def monitor_drift_with_lineage(reference_data, current_data):
    tracker = LineageTracker(job_name="drift_monitoring")

    try:
        tracker.log_start(
            input_features=list(reference_data.columns),
            input_dataset_name="production.reference_data",
            documentation="PSI-based drift detection for production model"
        )

        # Check drift
        result = DriftCheck(reference_data, current_data)

        # Log completion
        tracker.log_end(
            audit_summary={
                "fragility_score": result["max_psi"],
                "confidence_required": 0.9,
                "audit_hash": f"drift_{result['n_drifted_features']}"
            },
            output_dataset_name="production.drift_report"
        )

        return result

    except Exception as e:
        tracker.log_failure(error_message=str(e))
        raise
```

### Pipeline Integration

```python
from spectrum.lens.lineage import LineageTracker

def governance_pipeline(data):
    """
    Complete governance pipeline with lineage tracking.
    """
    tracker = LineageTracker(job_name="governance_pipeline")

    try:
        # Start tracking
        tracker.log_start(
            input_features=list(data.columns),
            documentation="Full governance pipeline: drift + audit + explain"
        )

        # Step 1: Drift check
        drift_result = DriftCheck(reference_data, data)

        # Step 2: Audit (if no drift)
        if not drift_result["drift_detected"]:
            audit_result = run_audit(model, data)

        # Step 3: Generate explanations
        explanations = generate_shap_explanations(model, data)

        # Complete
        tracker.log_end(
            audit_summary={
                "fragility_score": audit_result["fragility_score"],
                "confidence_required": 0.95,
                "audit_hash": audit_result["audit_hash"]
            }
        )

    except Exception as e:
        tracker.log_failure(error_message=str(e))
        raise
```

## Disabling Lineage Tracking

For testing or development, you can disable lineage tracking:

```python
# Method 1: Environment variable
export OPENLINEAGE_ENABLED="false"

# Method 2: Constructor argument
tracker = LineageTracker(job_name="test", enabled=False)

# All log methods will be no-ops
tracker.log_start(input_features=["x"])  # Does nothing
tracker.log_end(audit_summary={})  # Does nothing
```

## Viewing Lineage in Marquez

1. **Start Marquez**: `docker-compose -f docker-compose.marquez.yml up -d`
2. **Run your code**: Execute jobs with `LineageTracker`
3. **Open Web UI**: Navigate to http://localhost:3000
4. **Explore**:
   - View job runs and their status
   - See input/output datasets
   - Visualize lineage graphs
   - Inspect schemas and metadata

## Troubleshooting

### Connection Errors

If you see connection errors:

```
Failed to initialize OpenLineage client: [Errno 61] Connection refused
```

**Solution**: Ensure Marquez is running:
```bash
docker-compose -f docker-compose.marquez.yml ps
docker-compose -f docker-compose.marquez.yml up -d
```

### Events Not Appearing in Marquez

1. Check that `OPENLINEAGE_ENABLED=true`
2. Verify the URL is correct: `http://localhost:5000`
3. Check Marquez logs: `docker-compose -f docker-compose.marquez.yml logs marquez`
4. Verify network connectivity: `curl http://localhost:5000/api/v1/namespaces`

### Import Errors

If you see:
```
ModuleNotFoundError: No module named 'openlineage'
```

**Solution**: Install dependencies:
```bash
pip install openlineage-python
# Or
pip install -e ".[blue]"
```

## Production Deployment

For production use:

1. **Use external Marquez instance**: Set `OPENLINEAGE_URL` to your production Marquez server
2. **Configure authentication**: Add API keys/tokens to OpenLineage client
3. **Set proper namespace**: Use `OPENLINEAGE_NAMESPACE=production`
4. **Monitor performance**: OpenLineage client is async and shouldn't impact performance
5. **Handle failures gracefully**: The tracker catches exceptions and continues execution

## API Reference

See docstrings in `src/spectrum/lens/lineage.py` for detailed API documentation.

### LineageTracker Methods

- `__init__(job_name, namespace, url, enabled)` - Initialize tracker
- `log_start(input_features, input_dataset_name, documentation, source_code_location)` - Start event
- `log_end(audit_summary, output_dataset_name, output_uri)` - Complete event
- `log_failure(error_message, error_type)` - Failure event
- `log_running()` - Running/heartbeat event
- `log_abort(abort_reason)` - Abort event

## Additional Resources

- [OpenLineage Documentation](https://openlineage.io/)
- [Marquez Documentation](https://marquezproject.ai/)
- [OpenLineage Spec](https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md)
