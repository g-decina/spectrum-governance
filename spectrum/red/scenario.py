import numpy as np

from abc import ABC, abstractmethod
from spectrum.red.metrics import AdversarialMetrics

class AttackScenario(ABC):
    """
    Contract definition for all adversarial attack wrappers in spectrum.red.
    Ensures that any attack can be executed polymorphically by the Wargame Runner.

    All attacks must return comprehensive AdversarialMetrics containing:
    - Attack success rate
    - Perturbation magnitudes (for evasion attacks)
    - Privacy leakage scores (for inference attacks)
    - Extraction fidelity (for model stealing)
    - Poisoning success rates (for data poisoning)
    - Query counts and efficiency metrics
    - Statistical confidence intervals
    """
    @abstractmethod
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """
        Executes the attack against the base model and returns comprehensive metrics.

        :param X_input: The batch of input samples to attack.
        :return: AdversarialMetrics object with complete attack measurements
        """
        pass