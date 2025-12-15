"""
Metrics Aggregator for Audit Sessions

Collects metrics from various CLI command outputs and aggregates them
into a single data structure for report generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class MetricsAggregator:
    """
    Aggregates metrics from CLI test outputs into unified report data.

    Scans the audit session directory for JSON/TXT outputs from:
    - Red team scans (attack metrics)
    - Blue team tests (drift, uncertainty, explainability)
    - RCIA logs

    Produces a consolidated dictionary suitable for ReportBuilder.
    """

    def __init__(self, audit_session_dir: Path):
        """
        Initialize aggregator for an audit session.

        Args:
            audit_session_dir: Path to audit session directory
        """
        self.audit_dir = Path(audit_session_dir)
        self.artifacts_dir = self.audit_dir / "artifacts"
        self.logs_dir = self.audit_dir / "logs"
        self.metadata_path = self.audit_dir / "audit_metadata.json"

    def aggregate(self) -> Dict[str, Any]:
        """
        Aggregate all available metrics from audit session.

        Returns:
            Dictionary containing all aggregated metrics for report generation
        """
        logger.info(f"Aggregating metrics from {self.audit_dir}")

        # Start with metadata
        data = self._load_metadata()

        # Add adversarial metrics (red team)
        data['adversarial_metrics'] = self._load_adversarial_metrics()

        # Add drift metrics (blue team)
        drift_data = self._load_drift_metrics()
        data.update(drift_data)

        # Add uncertainty metrics (blue team)
        uncertainty_data = self._load_uncertainty_metrics()
        data.update(uncertainty_data)

        # Add explainability metrics (blue team)
        data['sample_adverse_reasons'] = self._load_explainability_metrics()

        # Add model information
        data.update(self._extract_model_info(data))

        logger.info(f"Aggregated {len(data)} metric categories")
        return data

    def _load_metadata(self) -> Dict[str, Any]:
        """Load audit session metadata."""
        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return {}

        try:
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            logger.info("Loaded audit metadata")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}

    def _load_adversarial_metrics(self) -> Dict[str, Any]:
        """
        Load adversarial attack metrics from red team scans.

        Searches for files matching:
        - hopskipjump_metrics.json
        - zoo_metrics.json
        - boundary_metrics.json
        - square_metrics.json

        If multiple attacks were run, returns the one with highest attack success rate
        (most concerning result for governance).
        """
        attack_types = ['hopskipjump', 'zoo', 'boundary', 'square']
        all_metrics = []

        for attack_type in attack_types:
            metrics_file = self.artifacts_dir / f"{attack_type}_metrics.json"

            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    logger.info(f"Found {attack_type} metrics (ASR: {metrics.get('attack_success_rate', 0):.1%})")
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to load {metrics_file}: {e}")

        if all_metrics:
            # Return the attack with HIGHEST success rate (most concerning for governance)
            worst_attack = max(all_metrics, key=lambda m: m.get('attack_success_rate', 0) or 0)
            logger.info(f"Using worst-case attack: {worst_attack.get('attack_type')} "
                       f"(ASR: {worst_attack.get('attack_success_rate', 0):.1%})")
            return worst_attack

        logger.warning("No adversarial metrics found")
        return {
            'attack_type': 'N/A',
            'attack_success_rate': None,
            'samples_tested': 0,
            'samples_successful': 0,
            'empirical_robustness_l2': 0,
            'empirical_robustness_linf': 0
        }

    def _load_drift_metrics(self) -> Dict[str, Any]:
        """
        Load drift monitoring metrics.

        Searches for drift_analysis.txt and parses PSI scores.
        """
        drift_file = self.artifacts_dir / "drift_analysis.txt"

        if drift_file.exists():
            try:
                with open(drift_file) as f:
                    content = f.read()

                # Parse status line
                data_drift_status = "Unknown"
                data_drift_alert = False
                max_psi = 0.0

                for line in content.split('\n'):
                    if line.startswith('Status:'):
                        data_drift_status = line.split(':', 1)[1].strip()
                        # Check if it's a significant drift
                        if 'Significant drift' in data_drift_status:
                            data_drift_alert = True
                    elif line.startswith('Max PSI:'):
                        max_psi_str = line.split(':', 1)[1].strip()
                        max_psi = float(max_psi_str)
                    elif line.startswith('Alert Required:'):
                        alert_str = line.split(':', 1)[1].strip()
                        data_drift_alert = (alert_str == 'True')

                logger.info(f"Loaded drift metrics (Max PSI: {max_psi:.4f}, Alert: {data_drift_alert})")
                return {
                    'data_drift_status': data_drift_status,
                    'data_drift_alert': data_drift_alert,
                    'max_psi': max_psi
                }
            except Exception as e:
                logger.error(f"Failed to load drift metrics: {e}")

        logger.warning("No drift metrics found")
        return {
            'data_drift_status': 'Unknown',
            'data_drift_alert': False,
            'max_psi': 0.0
        }

    def _load_uncertainty_metrics(self) -> Dict[str, Any]:
        """
        Load uncertainty quantification metrics.

        Searches for uncertainty_analysis.txt.
        """
        uncertainty_file = self.artifacts_dir / "uncertainty_analysis.txt"

        if uncertainty_file.exists():
            try:
                with open(uncertainty_file) as f:
                    content = f.read()

                # Parse metrics
                risk_level = "HIGH"
                confidence_required = 0.95
                empirical_coverage = 0.0

                # Extract values from text
                for line in content.split('\n'):
                    if "Risk Level:" in line:
                        risk_level = line.split(":")[1].strip()
                    elif "Required Confidence:" in line:
                        # Parse "95.0%" -> 0.95
                        val_str = line.split(":")[1].strip().replace('%', '')
                        confidence_required = float(val_str) / 100.0
                    elif "Empirical Coverage:" in line:
                        val_str = line.split(":")[1].strip().replace('%', '')
                        empirical_coverage = float(val_str) / 100.0

                logger.info("Loaded uncertainty metrics")
                return {
                    'risk_level': risk_level,
                    'confidence_required': confidence_required,
                    'empirical_coverage': empirical_coverage
                }
            except Exception as e:
                logger.error(f"Failed to load uncertainty metrics: {e}")

        logger.warning("No uncertainty metrics found")
        return {
            'risk_level': 'N/A',
            'confidence_required': 0.95,
            'empirical_coverage': 0.0
        }

    def _load_explainability_metrics(self) -> list:
        """
        Load explainability metrics (SHAP feature importance).

        Searches for explanations.txt.
        """
        explain_file = self.artifacts_dir / "explanations.txt"

        if explain_file.exists():
            try:
                with open(explain_file) as f:
                    content = f.read()

                # Extract top features
                features = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and line[0].isdigit() and '.' in line:
                        # Parse "1. feature_name"
                        feature = line.split('.', 1)[1].strip()
                        features.append(feature)

                if features:
                    logger.info(f"Loaded {len(features)} feature importance rankings")
                    return features
            except Exception as e:
                logger.error(f"Failed to load explainability metrics: {e}")

        logger.warning("No explainability metrics found")
        return []

    def _extract_model_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model information from metadata.

        Args:
            data: Current aggregated data

        Returns:
            Dictionary with model_name, model_type fields
        """
        model_path = data.get('model_path', '')

        # Extract model name from path
        if model_path:
            model_name = Path(model_path).stem
        else:
            model_name = data.get('audit_name', 'Unknown Model')

        return {
            'model_name': model_name,
            'model_type': 'Tabular ML Model',  # Could be enhanced to detect from model file
            'timestamp': data.get('created_at', 'N/A')
        }

    def save_aggregated_metrics(self, output_path: Optional[Path] = None) -> Path:
        """
        Aggregate metrics and save to JSON file.

        Args:
            output_path: Optional custom output path.
                        Defaults to audit_dir/aggregated_metrics.json

        Returns:
            Path to saved metrics file
        """
        if output_path is None:
            output_path = self.audit_dir / "aggregated_metrics.json"

        metrics = self.aggregate()

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved aggregated metrics to {output_path}")
        return output_path
