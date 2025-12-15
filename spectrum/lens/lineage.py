import uuid
import os

from datetime import datetime
from typing import Dict, Any, List, Optional
from openlineage.client import OpenLineageClient, RunState
from openlineage.client.facet import (
    SchemaDatasetFacet,
    SchemaField,
    DocumentationJobFacet,
    SourceCodeLocationJobFacet,
    ErrorMessageRunFacet
)
from openlineage.client.run import RunEvent, Job, Run, InputDataset, OutputDataset
from loguru import logger


class LineageTracker:
    """
    Records the Wargame Runner execution flow using OpenLineage standards.
    Logs the start and end of the compliance process for data provenance.

    Integrates with Marquez (OpenLineage backend) for lineage visualization.

    Configuration:
        - OPENLINEAGE_URL: Backend URL (default: http://localhost:5000)
        - OPENLINEAGE_NAMESPACE: Namespace for all jobs (default: spectrum.governance)
        - OPENLINEAGE_ENABLED: Enable/disable lineage tracking (default: true)

    Example:
        tracker = LineageTracker(job_name="wargame_audit", namespace="prod")
        tracker.log_start(input_features=["age", "income"])
        # ... run analysis ...
        tracker.log_end(audit_summary={"adversarial_metrics": {...}})
    """

    def __init__(
        self,
        job_name: str,
        namespace: Optional[str] = None,
        url: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        self.job_name = job_name
        self.namespace = namespace or os.getenv("OPENLINEAGE_NAMESPACE", "spectrum.governance")
        self.enabled = enabled if enabled is not None else os.getenv("OPENLINEAGE_ENABLED", "true").lower() == "true"

        # Configuration
        lineage_url = url or os.getenv("OPENLINEAGE_URL", "http://localhost:5000")

        if self.enabled:
            try:
                self.client = OpenLineageClient(url=lineage_url)
                logger.info(f"OpenLineage client initialized: {lineage_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenLineage client: {e}. Lineage tracking disabled.")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            logger.info("OpenLineage tracking is disabled")

        self.run_id = str(uuid.uuid4())
        
    def _emit_event(self, event: RunEvent) -> None:
        """Safely emit an event to the OpenLineage backend."""
        if not self.enabled or self.client is None:
            logger.debug(f"Lineage tracking disabled, skipping event: {event.eventType}")
            return

        try:
            self.client.emit(event)
            logger.debug(f"Emitted {event.eventType} event for job {self.job_name}")
        except Exception as e:
            logger.error(f"Failed to emit lineage event: {e}")

    def log_start(
        self,
        input_features: List[str],
        input_dataset_name: str = "wargame.input.features",
        documentation: Optional[str] = None,
        source_code_location: Optional[str] = None
    ):
        """
        Logs the START event before the Wargame begins.

        Args:
            input_features: List of feature names being analyzed
            input_dataset_name: Name of the input dataset
            documentation: Optional job documentation
            source_code_location: Optional source code location (e.g., git URL)
        """
        # Define Input Datasets
        input_datasets = [
            InputDataset(
                namespace=self.namespace,
                name=input_dataset_name,
                facets={
                    "schema": SchemaDatasetFacet(
                        fields=[SchemaField(name=f, type="DOUBLE") for f in input_features]
                    )
                }
            )
        ]

        # Build job facets
        job_facets = {}
        if documentation:
            job_facets["documentation"] = DocumentationJobFacet(description=documentation)
        if source_code_location:
            job_facets["sourceCodeLocation"] = SourceCodeLocationJobFacet(
                type="git",
                url=source_code_location
            )

        event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=self.run_id),
            job=Job(
                namespace=self.namespace,
                name=self.job_name,
                facets=job_facets if job_facets else {}
            ),
            inputs=input_datasets,
            outputs=[],
        )

        self._emit_event(event)
        logger.info(f"Lineage START: {self.job_name} with {len(input_features)} features")

    def log_end(
        self,
        audit_summary: Dict[str, Any],
        output_dataset_name: str = "wargame.output.rcia_log",
        output_uri: Optional[str] = None
    ):
        """
        Logs the COMPLETE event after the Wargame finishes.

        Args:
            audit_summary: Dictionary containing audit results (must include adversarial_metrics)
            output_dataset_name: Name of the output dataset
            output_uri: Optional URI where output is stored
        """
        # Define Output Datasets
        output_datasets = [
            OutputDataset(
                namespace=self.namespace,
                name=output_dataset_name,
                facets={
                    "schema": SchemaDatasetFacet(fields=[
                        SchemaField(name="audit_hash", type="STRING"),
                        SchemaField(name="attack_success_rate", type="DOUBLE"),
                        SchemaField(name="attack_type", type="STRING"),
                        SchemaField(name="confidence_required", type="DOUBLE"),
                        SchemaField(name="timestamp", type="TIMESTAMP"),
                    ])
                }
            )
        ]

        # Add custom run facets with audit metrics
        run_facets = {}

        event = RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=self.run_id, facets=run_facets),
            job=Job(namespace=self.namespace, name=self.job_name),
            inputs=[],
            outputs=output_datasets,
        )

        self._emit_event(event)
        adv_metrics = audit_summary.get("adversarial_metrics", {})
        attack_success_rate = adv_metrics.get("attack_success_rate", 0.0)
        logger.info(f"Lineage COMPLETE: {self.job_name} with attack success rate {attack_success_rate:.1%}")

    def log_failure(self, error_message: str, error_type: Optional[str] = None):
        """
        Logs a FAIL event when the Wargame encounters an error.

        Args:
            error_message: Description of the error
            error_type: Optional error type/class name
        """
        # Add error facet to run
        run_facets = {
            "errorMessage": ErrorMessageRunFacet(
                message=error_message,
                programmingLanguage="PYTHON"
            )
        }

        event = RunEvent(
            eventType=RunState.FAIL,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=self.run_id, facets=run_facets),
            job=Job(namespace=self.namespace, name=self.job_name),
            inputs=[],
            outputs=[],
        )

        self._emit_event(event)
        logger.error(f"Lineage FAIL: {self.job_name} - {error_message}")

    def log_running(self):
        """
        Logs a RUNNING event to indicate the job is actively executing.
        Useful for long-running jobs to show heartbeat.
        """
        event = RunEvent(
            eventType=RunState.RUNNING,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=self.run_id),
            job=Job(namespace=self.namespace, name=self.job_name),
            inputs=[],
            outputs=[],
        )

        self._emit_event(event)
        logger.debug(f"Lineage RUNNING: {self.job_name}")

    def log_abort(self, abort_reason: Optional[str] = None):
        """
        Logs an ABORT event when the job is cancelled.

        Args:
            abort_reason: Optional reason for abortion
        """
        run_facets = {}
        if abort_reason:
            run_facets["errorMessage"] = ErrorMessageRunFacet(
                message=f"Aborted: {abort_reason}",
                programmingLanguage="PYTHON"
            )

        event = RunEvent(
            eventType=RunState.ABORT,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=self.run_id, facets=run_facets if run_facets else {}),
            job=Job(namespace=self.namespace, name=self.job_name),
            inputs=[],
            outputs=[],
        )

        self._emit_event(event)
        logger.warning(f"Lineage ABORT: {self.job_name} - {abort_reason or 'No reason provided'}")