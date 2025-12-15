"""
Example: OpenLineage Integration with Spectrum Governance

This example demonstrates how to use LineageTracker to capture
data lineage for governance workflows.

Prerequisites:
    1. Install dependencies: pip install -e ".[blue]"
    2. Start Marquez: docker-compose -f docker-compose.marquez.yml up -d
    3. Open Web UI: http://localhost:3000

Run:
    python examples/lineage_example.py
"""

import time
import numpy as np
import pandas as pd
from spectrum.lens.lineage import LineageTracker


def example_basic_tracking():
    """Example 1: Basic lineage tracking for a simple job."""
    print("\n=== Example 1: Basic Lineage Tracking ===")

    tracker = LineageTracker(job_name="basic_audit_example")

    # Start the job
    tracker.log_start(
        input_features=["age", "income", "credit_score"],
        documentation="Basic example of lineage tracking for model audit"
    )

    # Simulate some work
    print("Running audit...")
    time.sleep(2)

    # Complete the job
    tracker.log_end(
        audit_summary={
            "attack_success_rate": 0.15,  # 15% of samples successfully attacked
            "confidence_required": 0.90,
            "audit_hash": "example_hash_123"
        }
    )

    print(f"Job completed. Run ID: {tracker.run_id}")
    print(f"  View in Marquez: http://localhost:3000")


def example_with_drift_check():
    """Example 2: Lineage tracking with drift monitoring."""
    print("\n=== Example 2: Drift Monitoring with Lineage ===")

    # Generate sample data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.uniform(300, 850, 1000)
    })

    # Current data with some drift
    current_data = pd.DataFrame({
        'age': np.random.normal(40, 10, 1000),  # Slight shift
        'income': np.random.normal(55000, 15000, 1000),  # Slight shift
        'credit_score': np.random.uniform(300, 850, 1000)
    })

    tracker = LineageTracker(job_name="drift_monitoring_example")

    try:
        # Start tracking
        tracker.log_start(
            input_features=list(reference_data.columns),
            input_dataset_name="production.reference_dataset",
            documentation="Monitoring data drift in production"
        )

        # Import and run drift check
        from spectrum.blue.monitor import DriftCheck

        print("Checking for drift...")
        result = DriftCheck(reference_data, current_data)

        print(f"  Drift detected: {result['drift_detected']}")
        print(f"  Max PSI: {result['max_psi']:.4f}")

        # Complete tracking
        tracker.log_end(
            audit_summary={
                "drift_score": result["max_psi"],  # Use PSI for drift monitoring
                "confidence_required": 0.90,
                "audit_hash": f"drift_check_{result['n_drifted_features']}"
            },
            output_dataset_name="production.drift_report"
        )

        print(f"Drift check completed. Run ID: {tracker.run_id}")

    except Exception as e:
        tracker.log_failure(error_message=str(e), error_type=type(e).__name__)
        raise


def example_failure_handling():
    """Example 3: Handling job failures with lineage tracking."""
    print("\n=== Example 3: Failure Handling ===")

    tracker = LineageTracker(job_name="failure_handling_example")

    try:
        tracker.log_start(
            input_features=["feature1", "feature2"],
            documentation="Demonstrating failure tracking"
        )

        print("Simulating failure...")
        time.sleep(1)

        # Simulate a failure
        raise ValueError("Simulated validation error")

    except ValueError as e:
        tracker.log_failure(
            error_message=str(e),
            error_type=type(e).__name__
        )
        print(f"Job failed: {e}")
        print(f"  Failure recorded. Run ID: {tracker.run_id}")


def example_heartbeat():
    """Example 4: Long-running job with heartbeat."""
    print("\n=== Example 4: Long-Running Job with Heartbeat ===")

    tracker = LineageTracker(job_name="long_running_example")

    tracker.log_start(
        input_features=["x", "y", "z"],
        documentation="Long-running job with periodic heartbeats"
    )

    # Simulate long-running job with heartbeats
    for i in range(5):
        print(f"  Working... step {i+1}/5")
        time.sleep(1)

        if i % 2 == 0:
            tracker.log_running()  # Send heartbeat
            print(f"    -> Sent heartbeat")

    tracker.log_end(
        audit_summary={
            "attack_success_rate": 0.12,
            "confidence_required": 0.95,
            "audit_hash": "long_job_complete"
        }
    )

    print(f"Long job completed. Run ID: {tracker.run_id}")


def example_abort():
    """Example 5: Aborting a job."""
    print("\n=== Example 5: Job Abortion ===")

    tracker = LineageTracker(job_name="abort_example")

    tracker.log_start(
        input_features=["a", "b", "c"],
        documentation="Demonstrating job abortion"
    )

    print("Starting job...")
    time.sleep(1)

    # Simulate abortion
    tracker.log_abort(abort_reason="User requested cancellation")

    print(f"Job aborted. Run ID: {tracker.run_id}")


def example_disabled_tracking():
    """Example 6: Tracking disabled."""
    print("\n=== Example 6: Disabled Tracking ===")

    tracker = LineageTracker(job_name="disabled_example", enabled=False)

    # All these calls will be no-ops
    tracker.log_start(input_features=["x"])
    print("Working (lineage disabled)...")
    time.sleep(1)
    tracker.log_end(audit_summary={"attack_success_rate": 0.10})

    print("Completed (no lineage events emitted)")


def main():
    """Run all examples."""
    print("=" * 70)
    print("OpenLineage + Marquez Integration Examples")
    print("=" * 70)
    print("\nMake sure Marquez is running:")
    print("  docker-compose -f docker-compose.marquez.yml up -d")
    print("\nView results at: http://localhost:3000")
    print("=" * 70)

    # Run examples
    example_basic_tracking()
    example_with_drift_check()
    example_failure_handling()
    example_heartbeat()
    example_abort()
    example_disabled_tracking()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("View the lineage graph at: http://localhost:3000")
    print("=" * 70)


if __name__ == "__main__":
    main()