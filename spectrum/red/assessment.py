from dataclasses import dataclass
from typing import List, Literal
from spectrum.utils.metrics import AdversarialMetrics
from spectrum.utils.thresholds import REGULATORY_THRESHOLDS, RegulatoryThreshold

@dataclass
class RegulatoryFinding:
    threshold: RegulatoryThreshold
    measured_value: float
    status: Literal["PASS", "MARGINAL", "FAIL"]
    evidence: str

@dataclass
class CompositeAssessment:
    """
    High-level assessment derived from raw metrics.
    This is a VIEW, not a score; every field is traceable to source metrics.
    """
    
    overall_robustness: Literal["ROBUST", "MARGINAL", "FRAGILE"]
    findings: List[RegulatoryFinding]
    source_metrics: AdversarialMetrics
    
    @classmethod
    def from_metrics(
        cls, 
        metrics: AdversarialMetrics, 
        regulatory_context: str
    ) -> "CompositeAssessment":
        
        thresholds = REGULATORY_THRESHOLDS.get(regulatory_context, [])
        findings = []
        
        for threshold in thresholds:
            measured = getattr(metrics, threshold.metric_name, None)
            if measured is None:
                continue
                
            status = threshold.evaluate(measured)
            
            # Add marginal zone (within 20% of threshold)
            if status == "PASS":
                margin = abs(measured - threshold.threshold_value) / threshold.threshold_value
                if margin < 0.20:
                    status = "MARGINAL"
            
            findings.append(RegulatoryFinding(
                threshold=threshold,
                measured_value=measured,
                status=status,
                evidence=f"{threshold.metric_name}={measured:.3f} vs threshold {threshold.threshold_value}"
            ))
        
        # Determine overall assessment
        statuses = [f.status for f in findings]
        if "FAIL" in statuses:
            overall = "FRAGILE"
        elif "MARGINAL" in statuses:
            overall = "MARGINAL"
        else:
            overall = "ROBUST"
        
        return cls(
            overall_robustness=overall,
            findings=findings,
            source_metrics=metrics
        )