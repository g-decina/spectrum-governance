import numpy as np
import pandas as pd

from datetime import datetime, timezone
from typing import Any, Dict, List, Union, Optional
from loguru import logger

from spectrum.infra.types import RiskProfile, TargetModel
from spectrum.infra.logger import RCIALogger
from spectrum.infra.events import InferenceEvent
from spectrum.blue.explain import SpectrumUncertaintyWrapper, generate_shap_explanations
from spectrum.blue.monitor import DriftCheck
from spectrum.red.attack import HopSkipJumpWrapper
from spectrum.lens.lineage import LineageTracker
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder

class WargameRunner:
    """
    The Orchestrator for the spectrum-governance conflict.

    Routes execution based on the TargetModel type (Tabular vs. LLM)
    and logs all outcomes to the RCIALogger.
    """
    def __init__(self,
                target_model: TargetModel,
                risk_profile: RiskProfile,
                logger: RCIALogger,
                attack_wrapper_map: Dict[str, Any], # e.g., {'tabular': HopSkipJumpWrapper, 'llm': InjectionScanner}
                defense_wrapper: Any = SpectrumUncertaintyWrapper
                ):
        """
        :param target_model: The unified model contract.
        :param risk_profile: The governance policy.
        :param logger: The RCIALogger instance.
        :param attack_wrapper_map: Map of model type to the corresponding AttackScenario class.
        :param defense_wrapper: The SpectrumUncertaintyWrapper class.
        """
        self.target = target_model
        self.risk_profile = risk_profile
        self.logger = logger
        self.attack_map = attack_wrapper_map
        self.defense_wrapper = defense_wrapper

    def run_wargame(self,
                    X_test: Union[np.ndarray, pd.DataFrame, List[str]],
                    y_test: Optional[np.ndarray] = None,
                    X_calib: Optional[np.ndarray] = None,
                    y_calib: Optional[np.ndarray] = None,
                    X_ref_drift: Optional[pd.DataFrame] = None,
                    X_current_drift: Optional[pd.DataFrame] = None,
                    report_path: str = "audit_report.pdf"
                    ) -> Dict[str, Any]:
        """
        Orchestrates the Red vs. Blue conflict based on the TargetModel type.

        :param X_test: Input data (DataFrame/numpy array for tabular, list of prompts for LLM).
        :param y_test: Ground truth labels (optional, for coverage calculation).
        :param X_calib: Calibration data for uncertainty estimation.
        :param y_calib: Calibration labels.
        :param X_ref_drift: Reference data for drift detection.
        :param X_current_drift: Current data for drift detection.
        :param report_path: Path to save the compliance report.
        :return: Final audit summary dictionary.
        """

        # Initialize the Lineage Tracker
        lineage_tracker = LineageTracker(
            job_name=f"Wargame_{self.target.model_name.replace('.', '_')}"
        )

        # Lineage Start Event
        input_features = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else ["input_prompt"]
        lineage_tracker.log_start(
            input_features=input_features,
            documentation=f"Wargame execution for {self.target.model_name}"
        )

        # Prepare Audit Report data structure
        audit_data = {}

        try:
            # 1. ROUTING: Determine the Wargame Type
            if self.target.is_tabular:
                audit_data = self._run_tabular_wargame(
                    X_test=X_test,
                    y_test=y_test,
                    X_calib=X_calib,
                    y_calib=y_calib,
                    X_ref_drift=X_ref_drift,
                    X_current_drift=X_current_drift
                )

            elif self.target.target_api_url:
                audit_data = self._run_llm_wargame(X_test=X_test)

            else:
                # Should be prevented by TargetModel Pydantic validator, but safety first
                raise TypeError("Target model configuration is invalid. Cannot route wargame.")

            # 7. FINAL ARTIFACT GENERATION

            end_time = datetime.now(timezone.utc)

            # A. RCIA Logger (Logs the raw audit data)
            self.logger.log_event(
                event=InferenceEvent(
                    model_version=audit_data["model_name"],
                    input_payload=f"Wargame Run on {len(X_test)} samples.",
                    output_payload=audit_data,
                    timestamp=end_time
                ),
                context="WARGAME_END"
            )

            # B. Compliance Report Pydantic Validation
            report = ComplianceReport(
                **audit_data,
                lineage_run_id=lineage_tracker.run_id,
                audit_log_path=self.logger.log_path
            ).model_dump()

            # C. PDF Report Generation
            report_builder = ReportBuilder()
            report_builder.generate_pdf(
                template_name="eu_ai_act_annex_iv.html",
                data=report,
                output_path=report_path
            )

            # D. Lineage End Event
            lineage_tracker.log_end(audit_summary=report)

            logger.info(f"Wargame Complete. Final Artifact saved to {report_path}")
            return report

        except Exception as e:
            logger.error(f"Wargame failed: {e}")
            lineage_tracker.log_failure(error_message=str(e), error_type=type(e).__name__)
            raise

    def _run_tabular_wargame(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Optional[np.ndarray],
        X_calib: Optional[np.ndarray],
        y_calib: Optional[np.ndarray],
        X_ref_drift: Optional[pd.DataFrame],
        X_current_drift: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Execute tabular model wargame (Red Team vs Blue Team).

        Returns:
            Dictionary containing audit results
        """
        if self.target.model_object is None:
            raise ValueError("Tabular Wargame requires a 'model_object'.")

        # Check for calibration data
        if X_calib is None or y_calib is None:
            raise ValueError("Tabular Wargame requires calibration data (X_calib, y_calib).")

        logger.info("Wargame Start: Tabular Conflict (Red Team vs Blue Team)")

        base_model = self.target.model_object

        # 1. BLUE TEAM: Setup and Calibration
        blue_wrapper = self.defense_wrapper(
            base_model=base_model,
            risk_profile=self.risk_profile
        )
        blue_wrapper.fit(X_calib, y_calib)
        logger.info("Blue Team: Uncertainty wrapper calibrated")

        # 2. BLUE TEAM: Generate Defense Artifacts
        defense_results = blue_wrapper.predict(X_test)
        logger.info("Blue Team: Predictions generated with uncertainty estimates")

        # 3. RED TEAM: Attack Setup & Execution
        RedAttackClass = self.attack_map.get("tabular")
        if RedAttackClass is None:
            raise NotImplementedError("Tabular attack wrapper not configured in attack_map.")

        red_wrapper = RedAttackClass(base_model=base_model)
        adversarial_metrics = red_wrapper.run(X_test)
        logger.info(f"Red Team: Attack complete. Attack Success Rate: {adversarial_metrics.attack_success_rate:.1%}")

        # 4. MONITORING: Data Drift Check
        drift_results = self._check_drift(X_ref_drift, X_current_drift)

        # 5. EXPLAINABILITY: Generate SHAP-based explanations
        sample_reasons = self._generate_explanations(
            base_model=base_model,
            X_test=X_test,
            defense_results=defense_results
        )

        # 6. CALCULATE EMPIRICAL COVERAGE: Evaluate on held-out test set
        empirical_coverage = self._calculate_empirical_coverage(
            defense_results=defense_results,
            y_test=y_test
        )

        return {
            "model_name": self.target.model_object.__class__.__name__,
            "model_type": "Tabular",
            "risk_level": self.risk_profile.level.value,
            "confidence_required": 1.0 - self.risk_profile.alpha,
            "empirical_coverage": empirical_coverage,
            "adversarial_metrics": adversarial_metrics.to_dict(),
            "sample_adverse_reasons": sample_reasons,
            "data_drift_status": drift_results["status"],
            "data_drift_alert": drift_results["alert_required"]
        }

    def _run_llm_wargame(
        self,
        X_test: List[str]
    ) -> Dict[str, Any]:
        """
        Execute LLM wargame (Red Team injection campaign).

        Returns:
            Dictionary containing audit results
        """
        logger.info("Wargame Start: LLM Injection Campaign (Red Team Only)")

        RedAttackClass = self.attack_map.get("llm")
        if RedAttackClass is None:
            raise NotImplementedError("LLM attack wrapper not configured in attack_map.")

        red_wrapper = RedAttackClass(target_api_url=self.target.target_api_url)

        # Execute Attack
        adversarial_metrics = red_wrapper.run(X_test)
        logger.info(f"Red Team: LLM attack complete. Attack Success Rate: {adversarial_metrics.attack_success_rate:.1%}")

        return {
            "model_name": self.target.target_api_url,
            "model_type": "LLM",
            "risk_level": self.risk_profile.level.value,
            "confidence_required": 1.0,
            "empirical_coverage": None,
            "adversarial_metrics": adversarial_metrics.to_dict(),
            "sample_adverse_reasons": ["Prompt injection successful", "Data exfiltration possible"],
            "data_drift_status": "Not Applicable",
            "data_drift_alert": False
        }

    def _check_drift(
        self,
        X_ref: Optional[pd.DataFrame],
        X_current: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Check for data drift between reference and current data.

        Returns:
            Dictionary with drift status and alert flag
        """
        if X_ref is not None and X_current is not None:
            try:
                drift_results = DriftCheck(X_ref, X_current)
                logger.info(f"Drift check complete: {drift_results['status']}")
                return drift_results
            except Exception as e:
                logger.warning(f"Drift check failed: {e}")
                return {
                    "status": "Error",
                    "alert_required": False,
                    "max_psi": 0.0
                }
        else:
            logger.info("Drift check skipped (no reference data provided)")
            return {
                "status": "Not Performed",
                "alert_required": False,
                "max_psi": 0.0
            }

    def _generate_explanations(
        self,
        base_model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        defense_results: Any,
        max_samples: int = 5
    ) -> List[str]:
        """
        Generate SHAP-based explanations for predictions.

        Returns:
            List of explanation strings
        """
        try:
            # Get SHAP explanations for a sample of adverse cases
            if hasattr(defense_results, 'y_preds') and len(defense_results.y_preds) > 0:
                # Generate explanations for first few predictions
                explanations = generate_shap_explanations(
                    model=base_model,
                    X_test=X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test),
                    max_samples=max_samples
                )
                sample_reasons = explanations.get("top_features", [])
                logger.info(f"Generated {len(sample_reasons)} SHAP explanations")
                return sample_reasons
            else:
                logger.warning("Explanations not available: no predictions found")
                return ["Explanations not available"]
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
            return ["Explanation generation unavailable"]

    def _calculate_empirical_coverage(
        self,
        defense_results: Any,
        y_test: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Calculate empirical coverage on held-out test set.

        Returns:
            Coverage value (float) or None if cannot be calculated
        """
        try:
            if hasattr(defense_results, 'y_preds') and hasattr(defense_results, 'y_pis') and y_test is not None:
                # Calculate coverage: proportion of y_test within prediction intervals
                lower_bounds = defense_results.y_pis[:, 0, 0]
                upper_bounds = defense_results.y_pis[:, 1, 0]
                coverage_mask = (y_test >= lower_bounds) & (y_test <= upper_bounds)
                empirical_coverage = float(np.mean(coverage_mask))
                logger.info(f"Empirical coverage: {empirical_coverage:.4f}")
                return empirical_coverage
            else:
                logger.warning("Cannot calculate empirical coverage: missing predictions or labels")
                return None
        except Exception as e:
            logger.error(f"Empirical coverage calculation failed: {e}")
            return None
