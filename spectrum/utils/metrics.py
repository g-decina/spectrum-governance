from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import numpy as np

@dataclass
class AdversarialMetrics:
    """Raw metrics from adversarial testing. All values are directly from ART."""
    
    # Core robustness metrics
    attack_success_rate: float
    empirical_robustness_l2: float
    empirical_robustness_linf: float
    
    # Perturbation distribution
    perturbation_mean: float
    perturbation_median: float
    perturbation_min: float
    perturbation_max: float
    perturbation_std: float
    perturbation_percentiles: Dict[int, float]  # {5: 0.02, 25: 0.05, ...}
    
    # Feature analysis
    feature_sensitivity: Dict[str, float]
    
    # Certified bounds (optional, requires specific analysis)
    clever_score_l2: Optional[float] = None
    clever_score_linf: Optional[float] = None
    
    # Attack metadata
    attack_type: str = ""
    samples_tested: int = 0
    samples_attacked_successfully: int = 0
    total_queries: int = 0
    
    @property
    def robustness_l2(self) -> float:
        """Convenience alias for empirical_robustness_l2."""
        return self.empirical_robustness_l2
    
    def top_vulnerable_features(self, n: int = 5) -> List[tuple]:
        """Return the n most sensitive features."""
        sorted_features = sorted(
            self.feature_sensitivity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_features[:n]