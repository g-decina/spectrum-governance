# Lineage Tracking Quick Start

Get started with OpenLineage + Marquez integration in 5 minutes.

## Step 1: Install Dependencies

```bash
# Install spectrum-governance with blue team extras (includes OpenLineage)
pip install -e ".[blue]"
```

## Step 2: Start Marquez Backend

```bash
# Start Marquez (PostgreSQL + API + Web UI)
docker-compose -f docker-compose.marquez.yml up -d

# Verify it's running
docker-compose -f docker-compose.marquez.yml ps

# Expected output:
# NAME                STATUS              PORTS
# marquez             running             0.0.0.0:5000-5001->5000-5001/tcp
# marquez-postgres    running             0.0.0.0:5432->5432/tcp
# marquez-web         running             0.0.0.0:3000->3000/tcp
```

## Step 3: Run Example

```bash
# Run the example script
python examples/lineage_example.py
```

**Expected output:**
```
======================================================================
OpenLineage + Marquez Integration Examples
======================================================================

Make sure Marquez is running:
  docker-compose -f docker-compose.marquez.yml up -d

View results at: http://localhost:3000
======================================================================

=== Example 1: Basic Lineage Tracking ===
Running audit...
✓ Job completed. Run ID: 3f9c2b4a-...

=== Example 2: Drift Monitoring with Lineage ===
Checking for drift...
  Drift detected: False
  Max PSI: 0.0123
✓ Drift check completed. Run ID: 7e8d3a2f-...

...
```

## Step 4: View Lineage in Marquez

1. Open your browser: **http://localhost:3000**
2. You should see the Marquez Web UI
3. Navigate to **Jobs** to see your runs
4. Click on a job to see:
   - Input/Output datasets
   - Schema information
   - Run history
   - Lineage graph

## Step 5: Use in Your Code

```python
from spectrum.lens.lineage import LineageTracker

# Initialize tracker
tracker = LineageTracker(job_name="my_audit")

# Start tracking
tracker.log_start(
    input_features=["age", "income", "credit_score"],
    documentation="My audit workflow"
)

# Your code here
# ... run analysis ...

# Complete tracking
tracker.log_end(
    audit_summary={
        "fragility_score": 0.85,
        "confidence_required": 0.95,
        "audit_hash": "abc123"
    }
)
```

## Troubleshooting

### Marquez not starting?

```bash
# Check logs
docker-compose -f docker-compose.marquez.yml logs

# Restart services
docker-compose -f docker-compose.marquez.yml down
docker-compose -f docker-compose.marquez.yml up -d
```

### Connection refused error?

Make sure Marquez is running and accessible:
```bash
# Test connection
curl http://localhost:5000/api/v1/namespaces

# Should return JSON response
```

### Import errors?

```bash
# Make sure OpenLineage is installed
pip install openlineage-python

# Or reinstall with extras
pip install -e ".[blue]"
```

## Next Steps

- Read the full documentation: [`docs/LINEAGE.md`](docs/LINEAGE.md)
- Integrate with your workflows
- Deploy Marquez to production
- Customize namespaces and job names

## Stopping Marquez

```bash
# Stop services (keeps data)
docker-compose -f docker-compose.marquez.yml stop

# Stop and remove (deletes data)
docker-compose -f docker-compose.marquez.yml down -v
```

## Resources

- **Marquez Web UI**: http://localhost:3000
- **OpenLineage API**: http://localhost:5000
- **Documentation**: [`docs/LINEAGE.md`](docs/LINEAGE.md)
- **Examples**: [`examples/lineage_example.py`](examples/lineage_example.py)
