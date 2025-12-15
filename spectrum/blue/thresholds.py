"""
Regulatory Thresholds

Defines industry-standard and regulatory thresholds for model governance.
"""

from dataclasses import dataclass
from typing import Literal, Dict, List


@dataclass
class RegulatoryThreshold:
    """
    A specific threshold requirement from regulations or industry standards.
    """
    regulation: str                     # "EU_AI_ACT", "CFPB", "EEOC", etc.
    requirement: str                    # Human-readable requirement description
    metric_name: str                    # Which metric this applies to
    threshold_value: float              # The threshold value
    comparison: Literal["<", "<=", ">", ">="]  # How to compare
    citation: str                       # Legal/regulatory citation

    def evaluate(self, measured_value: float) -> Literal["PASS", "FAIL"]:
        """
        Evaluate whether a measured value passes the threshold.

        Returns:
            "PASS" if the measured value meets the requirement
            "FAIL" if the measured value fails the requirement
        """
        if self.comparison == "<":
            return "PASS" if measured_value < self.threshold_value else "FAIL"
        elif self.comparison == "<=":
            return "PASS" if measured_value <= self.threshold_value else "FAIL"
        elif self.comparison == ">":
            return "PASS" if measured_value > self.threshold_value else "FAIL"
        elif self.comparison == ">=":
            return "PASS" if measured_value >= self.threshold_value else "FAIL"
        else:
            raise ValueError(f"Unknown comparison: {self.comparison}")

    def __repr__(self):
        return f"RegulatoryThreshold({self.regulation}: {self.metric_name} {self.comparison} {self.threshold_value})"


# ============================================================================
# EU AI ACT THRESHOLDS
# ============================================================================

EU_AI_ACT_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="EU_AI_ACT",
        requirement="High-risk AI systems must demonstrate adequate robustness",
        metric_name="attack_success_rate",
        threshold_value=0.25,
        comparison="<",
        citation="EU AI Act Article 15 - Accuracy, robustness and cybersecurity"
    ),
    RegulatoryThreshold(
        regulation="EU_AI_ACT",
        requirement="Model must maintain stability under perturbations",
        metric_name="empirical_robustness_l2",
        threshold_value=0.5,
        comparison="<",
        citation="EU AI Act Annex IV - Technical documentation requirements"
    ),
    RegulatoryThreshold(
        regulation="EU_AI_ACT",
        requirement="Uncertainty quantification for high-risk decisions",
        metric_name="empirical_coverage",
        threshold_value=0.95,
        comparison=">=",
        citation="EU AI Act Article 15 - Accuracy requirements"
    ),
]

# ============================================================================
# CFPB (CONSUMER FINANCIAL PROTECTION BUREAU) THRESHOLDS
# ============================================================================

CFPB_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="CFPB",
        requirement="Fair lending - model stability under adversarial conditions",
        metric_name="attack_success_rate",
        threshold_value=0.20,
        comparison="<",
        citation="CFPB Bulletin 2023-02: Adverse action notification requirements"
    ),
    RegulatoryThreshold(
        regulation="CFPB",
        requirement="Model risk management - adequate calibration",
        metric_name="empirical_coverage",
        threshold_value=0.90,
        comparison=">=",
        citation="SR 11-7: Model Risk Management Guidance"
    ),
    RegulatoryThreshold(
        regulation="CFPB",
        requirement="Data drift monitoring - population stability",
        metric_name="max_psi",
        threshold_value=0.25,
        comparison="<",
        citation="CFPB Circular 2022-03: Adverse action requirements"
    ),
]

# ============================================================================
# EEOC (EQUAL EMPLOYMENT OPPORTUNITY COMMISSION) THRESHOLDS
# ============================================================================

EEOC_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="EEOC",
        requirement="Four-fifths rule for disparate impact",
        metric_name="disparate_impact_ratio",
        threshold_value=0.80,
        comparison=">=",
        citation="29 CFR 1607.4(D) - Uniform Guidelines on Employee Selection"
    ),
    RegulatoryThreshold(
        regulation="EEOC",
        requirement="Model stability - adversarial robustness",
        metric_name="attack_success_rate",
        threshold_value=0.30,
        comparison="<",
        citation="EEOC Guidance on AI and Algorithmic Fairness (2023)"
    ),
]

# ============================================================================
# INDUSTRY BEST PRACTICES (NIST, ISO, etc.)
# ============================================================================

NIST_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="NIST_AI_RMF",
        requirement="AI Risk Management Framework - Robustness",
        metric_name="attack_success_rate",
        threshold_value=0.15,
        comparison="<",
        citation="NIST AI RMF 1.0: MEASURE 2.3 - AI system robustness"
    ),
    RegulatoryThreshold(
        regulation="NIST_AI_RMF",
        requirement="Conformal prediction coverage",
        metric_name="empirical_coverage",
        threshold_value=0.95,
        comparison=">=",
        citation="NIST AI RMF 1.0: MEASURE 2.10 - Uncertainty quantification"
    ),
]

ISO_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="ISO_IEC_42001",
        requirement="AI Management System - Model resilience",
        metric_name="attack_success_rate",
        threshold_value=0.20,
        comparison="<",
        citation="ISO/IEC 42001:2023 - AI Management System"
    ),
]

# ============================================================================
# CONSOLIDATED REGISTRY
# ============================================================================

REGULATORY_THRESHOLDS: Dict[str, List[RegulatoryThreshold]] = {
    "EU_AI_ACT": EU_AI_ACT_THRESHOLDS,
    "CFPB": CFPB_THRESHOLDS,
    "EEOC": EEOC_THRESHOLDS,
    "NIST_AI_RMF": NIST_THRESHOLDS,
    "ISO_IEC_42001": ISO_THRESHOLDS,
}


def get_thresholds_for_context(regulatory_context: str) -> List[RegulatoryThreshold]:
    """
    Get all applicable thresholds for a given regulatory context.

    Args:
        regulatory_context: Comma-separated list of regulations (e.g., "EU_AI_ACT,CFPB")

    Returns:
        List of applicable RegulatoryThreshold objects
    """
    contexts = [ctx.strip() for ctx in regulatory_context.split(",")]
    thresholds = []

    for ctx in contexts:
        if ctx in REGULATORY_THRESHOLDS:
            thresholds.extend(REGULATORY_THRESHOLDS[ctx])

    return thresholds


def get_threshold_by_metric(
    metric_name: str,
    regulatory_context: str
) -> RegulatoryThreshold:
    """
    Find the most stringent threshold for a given metric across contexts.

    Args:
        metric_name: Name of the metric (e.g., "attack_success_rate")
        regulatory_context: Comma-separated list of regulations

    Returns:
        The most stringent applicable threshold, or None if not found
    """
    thresholds = get_thresholds_for_context(regulatory_context)
    applicable = [t for t in thresholds if t.metric_name == metric_name]

    if not applicable:
        return None

    # Return the most stringent threshold
    # For "<" comparisons, smallest threshold is most stringent
    # For ">=" comparisons, largest threshold is most stringent
    if applicable[0].comparison in ["<", "<="]:
        return min(applicable, key=lambda t: t.threshold_value)
    else:
        return max(applicable, key=lambda t: t.threshold_value)
