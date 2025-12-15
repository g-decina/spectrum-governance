"""
Adversarial Metrics

Raw metrics from adversarial testing, providing source-of-truth measurements
for regulatory assessment.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class AdversarialMetrics:
    """
    Comprehensive metrics from adversarial attack execution.

    These are the source-of-truth measurements that feed into regulatory
    assessments. All metrics are directly observable and auditable.

    Metrics Philosophy:
    -------------------
    - Attack Success Rate: Primary robustness metric (lower is better)
    - Perturbation Magnitudes: Measure attack strength (higher means weaker model)
    - Query Efficiency: Measures attack cost (relevant for deployment security)
    - Statistical Confidence: Provides uncertainty bounds for regulatory reporting
    """

    # Core robustness metrics
    attack_success_rate: float          # Fraction of samples successfully attacked (0.0-1.0)
    samples_tested: int                 # Total samples tested
    samples_successful: int             # Number of successful attacks

    # Perturbation magnitudes (measured on successful attacks)
    empirical_robustness_l2: float      # Mean L2 norm of perturbations
    empirical_robustness_linf: float    # Mean L∞ norm of perturbations
    min_perturbation_l2: float          # Smallest L2 perturbation (best-case attack)
    max_perturbation_l2: float          # Largest L2 perturbation (worst-case attack)
    median_perturbation_l2: float       # Median L2 perturbation

    # Attack metadata
    attack_type: str                    # "HopSkipJump", "ZOO", "Boundary", etc.
    queries_used: int                   # Total queries to model
    avg_queries_per_sample: float       # Average queries per sample

    # Statistical measures
    perturbation_std_l2: Optional[float] = None             # Std dev of L2 perturbations
    perturbation_std_linf: Optional[float] = None           # Std dev of L∞ perturbations
    confidence_interval_95: Optional[Tuple[float, float]] = None  # 95% CI for attack success rate

    # Per-feature sensitivity (optional, for attribution analysis)
    feature_sensitivity: Optional[Dict[str, float]] = None

    # Privacy/inference metrics (for inference attacks)
    privacy_leakage_score: Optional[float] = None           # For membership/attribute inference
    true_positive_rate: Optional[float] = None              # For membership inference
    false_positive_rate: Optional[float] = None             # For membership inference

    # Extraction metrics (for model stealing)
    extraction_fidelity: Optional[float] = None             # Agreement between target and stolen model
    extraction_queries: Optional[int] = None                # Queries used for extraction

    # Poisoning metrics (for data poisoning)
    poisoning_success_rate: Optional[float] = None          # Backdoor trigger success rate
    clean_accuracy_impact: Optional[float] = None           # Impact on clean test accuracy

    def __post_init__(self):
        """Validate metrics after initialization."""
        # Ensure attack_success_rate is consistent with counts
        expected_rate = self.samples_successful / self.samples_tested if self.samples_tested > 0 else 0.0
        if abs(self.attack_success_rate - expected_rate) > 1e-6:
            # Auto-correct if there's a mismatch
            self.attack_success_rate = expected_rate

        # Compute confidence interval if not provided
        if self.confidence_interval_95 is None and self.samples_tested > 0:
            self.confidence_interval_95 = self._compute_wilson_ci(
                self.samples_successful,
                self.samples_tested,
                confidence=0.95
            )

    @staticmethod
    def _compute_wilson_ci(
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute Wilson score confidence interval for binomial proportion.

        This is more accurate than normal approximation for small sample sizes.

        :param successes: Number of successful attacks
        :param trials: Total number of trials
        :param confidence: Confidence level (e.g., 0.95 for 95%)
        :return: (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 0.0)

        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        p = successes / trials

        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    @classmethod
    def for_evasion_attack(
        cls,
        attack_type: str,
        y_original: np.ndarray,
        y_adversarial: np.ndarray,
        X_original: np.ndarray,
        X_adversarial: np.ndarray,
        queries_used: int = 0
    ) -> "AdversarialMetrics":
        """
        Create metrics from evasion attack results.

        :param attack_type: Name of the attack
        :param y_original: Original predictions (labels)
        :param y_adversarial: Adversarial predictions (labels)
        :param X_original: Original samples
        :param X_adversarial: Adversarial samples
        :param queries_used: Total queries to model
        :return: AdversarialMetrics instance
        """
        # Compute success mask
        success_mask = (y_original != y_adversarial)
        samples_tested = len(y_original)
        samples_successful = int(np.sum(success_mask))
        attack_success_rate = float(np.mean(success_mask))

        # Compute perturbations
        perturbations = X_adversarial - X_original

        # L2 norms
        l2_norms = np.linalg.norm(perturbations, ord=2, axis=1)
        successful_l2 = l2_norms[success_mask] if samples_successful > 0 else np.array([0.0])

        # L∞ norms
        linf_norms = np.linalg.norm(perturbations, ord=np.inf, axis=1)
        successful_linf = linf_norms[success_mask] if samples_successful > 0 else np.array([0.0])

        return cls(
            attack_success_rate=attack_success_rate,
            samples_tested=samples_tested,
            samples_successful=samples_successful,
            empirical_robustness_l2=float(np.mean(successful_l2)),
            empirical_robustness_linf=float(np.mean(successful_linf)),
            min_perturbation_l2=float(np.min(successful_l2)),
            max_perturbation_l2=float(np.max(successful_l2)),
            median_perturbation_l2=float(np.median(successful_l2)),
            attack_type=attack_type,
            queries_used=queries_used,
            avg_queries_per_sample=queries_used / samples_tested if samples_tested > 0 else 0.0,
            perturbation_std_l2=float(np.std(successful_l2)) if samples_successful > 0 else 0.0,
            perturbation_std_linf=float(np.std(successful_linf)) if samples_successful > 0 else 0.0
        )

    @classmethod
    def for_inference_attack(
        cls,
        attack_type: str,
        privacy_leakage_score: float,
        samples_tested: int,
        true_positive_rate: Optional[float] = None,
        false_positive_rate: Optional[float] = None
    ) -> "AdversarialMetrics":
        """
        Create metrics for privacy inference attacks.

        :param attack_type: Name of attack (e.g., "MembershipInference")
        :param privacy_leakage_score: Overall privacy leakage (0.0-1.0)
        :param samples_tested: Number of samples tested
        :param true_positive_rate: TPR for membership inference
        :param false_positive_rate: FPR for membership inference
        :return: AdversarialMetrics instance
        """
        # For inference attacks, success means privacy leakage
        # Use privacy_leakage_score as the primary metric
        samples_successful = int(privacy_leakage_score * samples_tested)

        return cls(
            attack_success_rate=privacy_leakage_score,
            samples_tested=samples_tested,
            samples_successful=samples_successful,
            empirical_robustness_l2=0.0,  # Not applicable for inference
            empirical_robustness_linf=0.0,
            min_perturbation_l2=0.0,
            max_perturbation_l2=0.0,
            median_perturbation_l2=0.0,
            attack_type=attack_type,
            queries_used=samples_tested,  # Each inference is one query
            avg_queries_per_sample=1.0,
            privacy_leakage_score=privacy_leakage_score,
            true_positive_rate=true_positive_rate,
            false_positive_rate=false_positive_rate
        )

    @classmethod
    def for_extraction_attack(
        cls,
        attack_type: str,
        extraction_fidelity: float,
        samples_tested: int,
        extraction_queries: int
    ) -> "AdversarialMetrics":
        """
        Create metrics for model extraction attacks.

        :param attack_type: Name of attack (e.g., "FunctionallyEquivalentExtraction")
        :param extraction_fidelity: Agreement between target and stolen model (0.0-1.0)
        :param samples_tested: Number of samples tested
        :param extraction_queries: Queries used for extraction
        :return: AdversarialMetrics instance
        """
        # For extraction, fidelity is the success metric
        samples_successful = int(extraction_fidelity * samples_tested)

        return cls(
            attack_success_rate=extraction_fidelity,
            samples_tested=samples_tested,
            samples_successful=samples_successful,
            empirical_robustness_l2=0.0,  # Not applicable
            empirical_robustness_linf=0.0,
            min_perturbation_l2=0.0,
            max_perturbation_l2=0.0,
            median_perturbation_l2=0.0,
            attack_type=attack_type,
            queries_used=extraction_queries,
            avg_queries_per_sample=extraction_queries / samples_tested if samples_tested > 0 else 0.0,
            extraction_fidelity=extraction_fidelity,
            extraction_queries=extraction_queries
        )

    @classmethod
    def for_poisoning_attack(
        cls,
        attack_type: str,
        poisoning_success_rate: float,
        samples_tested: int,
        clean_accuracy_impact: Optional[float] = None
    ) -> "AdversarialMetrics":
        """
        Create metrics for data poisoning attacks.

        :param attack_type: Name of attack (e.g., "BackdoorPoisoning")
        :param poisoning_success_rate: Backdoor trigger success rate (0.0-1.0)
        :param samples_tested: Number of samples tested
        :param clean_accuracy_impact: Impact on clean accuracy
        :return: AdversarialMetrics instance
        """
        samples_successful = int(poisoning_success_rate * samples_tested)

        return cls(
            attack_success_rate=poisoning_success_rate,
            samples_tested=samples_tested,
            samples_successful=samples_successful,
            empirical_robustness_l2=0.0,  # Not applicable
            empirical_robustness_linf=0.0,
            min_perturbation_l2=0.0,
            max_perturbation_l2=0.0,
            median_perturbation_l2=0.0,
            attack_type=attack_type,
            queries_used=samples_tested,
            avg_queries_per_sample=1.0,
            poisoning_success_rate=poisoning_success_rate,
            clean_accuracy_impact=clean_accuracy_impact
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'attack_success_rate': self.attack_success_rate,
            'samples_tested': self.samples_tested,
            'samples_successful': self.samples_successful,
            'empirical_robustness_l2': self.empirical_robustness_l2,
            'empirical_robustness_linf': self.empirical_robustness_linf,
            'min_perturbation_l2': self.min_perturbation_l2,
            'max_perturbation_l2': self.max_perturbation_l2,
            'median_perturbation_l2': self.median_perturbation_l2,
            'attack_type': self.attack_type,
            'queries_used': self.queries_used,
            'avg_queries_per_sample': self.avg_queries_per_sample,
            'perturbation_std_l2': self.perturbation_std_l2,
            'perturbation_std_linf': self.perturbation_std_linf,
            'confidence_interval_95': self.confidence_interval_95,
            'feature_sensitivity': self.feature_sensitivity,
            'privacy_leakage_score': self.privacy_leakage_score,
            'true_positive_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            'extraction_fidelity': self.extraction_fidelity,
            'extraction_queries': self.extraction_queries,
            'poisoning_success_rate': self.poisoning_success_rate,
            'clean_accuracy_impact': self.clean_accuracy_impact
        }

    def __format__(self, format_spec: str) -> str:
        """
        Format the metrics for string interpolation.

        Supports format specs:
        - '' or 's': Returns summary string
        - '.Nf': Returns attack success rate as float with N decimal places
        """
        if not format_spec or format_spec == 's':
            return self.summary()
        elif format_spec.endswith('f'):
            # Format attack success rate as float
            return format(self.attack_success_rate, format_spec)
        elif format_spec.endswith('%'):
            # Format attack success rate as percentage
            return format(self.attack_success_rate, format_spec)
        else:
            return self.summary()

    def __str__(self) -> str:
        """String representation."""
        return self.summary()

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"AdversarialMetrics(attack_type='{self.attack_type}', "
            f"success_rate={self.attack_success_rate:.1%}, "
            f"samples={self.samples_successful}/{self.samples_tested}, "
            f"L2={self.empirical_robustness_l2:.4f})"
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Adversarial Metrics Report: {self.attack_type}",
            "=" * 60,
            f"Attack Success Rate: {self.attack_success_rate:.1%}",
            f"Samples Tested: {self.samples_tested}",
            f"Samples Successful: {self.samples_successful}",
        ]

        if self.confidence_interval_95:
            lower, upper = self.confidence_interval_95
            lines.append(f"95% Confidence Interval: [{lower:.1%}, {upper:.1%}]")

        # Perturbation metrics (for evasion attacks)
        if self.median_perturbation_l2 > 0:
            lines.extend([
                "",
                "Perturbation Magnitudes:",
                f"  Median L2: {self.median_perturbation_l2:.4f}",
                f"  Mean L2: {self.empirical_robustness_l2:.4f}",
                f"  Std Dev L2: {self.perturbation_std_l2:.4f}" if self.perturbation_std_l2 else "",
                f"  Min L2: {self.min_perturbation_l2:.4f}",
                f"  Max L2: {self.max_perturbation_l2:.4f}",
                f"  Mean L∞: {self.empirical_robustness_linf:.4f}",
            ])

        # Query efficiency
        lines.extend([
            "",
            "Query Efficiency:",
            f"  Total Queries: {self.queries_used}",
            f"  Avg Queries/Sample: {self.avg_queries_per_sample:.1f}",
        ])

        # Privacy metrics
        if self.privacy_leakage_score is not None:
            lines.extend([
                "",
                "Privacy Metrics:",
                f"  Privacy Leakage: {self.privacy_leakage_score:.1%}",
            ])
            if self.true_positive_rate is not None:
                lines.append(f"  True Positive Rate: {self.true_positive_rate:.1%}")
            if self.false_positive_rate is not None:
                lines.append(f"  False Positive Rate: {self.false_positive_rate:.1%}")

        # Extraction metrics
        if self.extraction_fidelity is not None:
            lines.extend([
                "",
                "Extraction Metrics:",
                f"  Model Fidelity: {self.extraction_fidelity:.1%}",
                f"  Extraction Queries: {self.extraction_queries}",
            ])

        # Poisoning metrics
        if self.poisoning_success_rate is not None:
            lines.extend([
                "",
                "Poisoning Metrics:",
                f"  Backdoor Success Rate: {self.poisoning_success_rate:.1%}",
            ])
            if self.clean_accuracy_impact is not None:
                lines.append(f"  Clean Accuracy Impact: {self.clean_accuracy_impact:.1%}")

        return "\n".join(line for line in lines if line)  # Filter empty strings
