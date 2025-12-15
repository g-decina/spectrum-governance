import numpy as np

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import SklearnClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score

from spectrum.red.scenario import AttackScenario
from spectrum.red.metrics import AdversarialMetrics
from spectrum.utils.threading import _with_limited_threads
from spectrum.infra.types import RiskProfile

"""
spectrum.red.poisoning
======================

Data Poisoning Attacks for Training Data Integrity Assessment.

This module provides wrappers for data poisoning attacks from ART. These attacks
manipulate training data to compromise model behavior, either by degrading overall
performance or by inserting backdoors that trigger specific behaviors.

POISONING ATTACK TYPES:
-----------------------
1. Backdoor Poisoning: Injects poisoned samples with a trigger pattern that
   causes the model to misclassify inputs containing that trigger.

2. Availability Poisoning: Corrupts training data to degrade overall model
   performance and availability.

ADVERSARIAL METRICS:
--------------------
Poisoning attacks return AdversarialMetrics with poisoning-specific fields:
- Poisoning Success Rate: Backdoor trigger success rate (0.0-1.0)
- Clean Accuracy Impact: Impact on clean test accuracy
- Attack Success Rate: Overall poisoning effectiveness

Higher success rates indicate greater vulnerability to training data manipulation.

KEY COMPONENTS:
---------------
- PoisoningAttackWrapper: Tests backdoor insertion and availability attacks.

USAGE FLOW:
-----------
1. Initialize with clean model and training data:
    `poisoner = PoisoningAttackWrapper(base_model=model, X_train=X, y_train=y)`
2. Execute poisoning attack:
    `metrics = poisoner.run(X_test)`
3. Interpret: metrics.poisoning_success_rate > 0.7 indicates high susceptibility.
"""


class PoisoningAttackWrapper(AttackScenario):
    """
    Implements Data Poisoning Attack that injects backdoor triggers into
    training data to compromise model behavior.

    The attack creates a backdoor trigger (a specific pattern) and poisons
    a small fraction of training data. Models trained on poisoned data will
    misclassify inputs containing the trigger.

    READ: https://arxiv.org/abs/2410.13722 (LLMs) ;
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        poison_ratio: float = 0.1,
        risk_profile: RiskProfile = None
    ):
        """
        Initializes the Poisoning Attack.

        :param base_model: The model architecture to poison (will be retrained)
        :param X_train: Clean training data
        :param y_train: Clean training labels
        :param poison_ratio: Fraction of training data to poison (0.0 to 1.0)
        :param risk_profile: Optional risk configuration
        """
        self.base_model = base_model
        self.X_train = X_train
        self.y_train = y_train
        self.poison_ratio = poison_ratio
        self.profile = risk_profile
        self.poisoned_model = None
        self.backdoor_trigger = None

    def _create_backdoor_trigger(self, n_features: int) -> np.ndarray:
        """
        Creates a simple backdoor trigger pattern.

        :param n_features: Number of features in the data
        :return: Trigger pattern as a 1D array
        """
        # Create a simple trigger: set specific features to specific values
        # For example: set 10% of features to 1.0
        trigger = np.zeros(n_features)
        n_trigger_features = max(1, int(n_features * 0.1))
        trigger_indices = np.random.choice(n_features, n_trigger_features, replace=False)
        trigger[trigger_indices] = 1.0
        return trigger

    def _apply_trigger(self, X: np.ndarray, trigger: np.ndarray) -> np.ndarray:
        """
        Applies the backdoor trigger to input samples.

        :param X: Input samples
        :param trigger: Trigger pattern
        :return: Triggered samples
        """
        X_triggered = X.copy()
        # Simple additive trigger (could also be replacement or other patterns)
        X_triggered = np.clip(X_triggered + trigger, 0, 1)
        return X_triggered

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """
        Executes the Data Poisoning Attack and returns comprehensive metrics.

        The attack:
        1. Creates a backdoor trigger pattern
        2. Poisons training data with triggered samples
        3. Trains a new model on poisoned data
        4. Tests backdoor effectiveness on clean test samples

        :param X_input: Clean test samples
        :return: AdversarialMetrics with poisoning effectiveness measurements
        """
        n_samples, n_features = self.X_train.shape
        n_poison = int(n_samples * self.poison_ratio)

        # 1. Create backdoor trigger
        self.backdoor_trigger = self._create_backdoor_trigger(n_features)

        # 2. Select samples to poison (random selection)
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)

        # 3. Create poisoned training data
        X_poisoned = self.X_train.copy()
        y_poisoned = self.y_train.copy()

        # Apply trigger to poisoned samples
        X_poisoned[poison_indices] = self._apply_trigger(
            self.X_train[poison_indices],
            self.backdoor_trigger
        )

        # Flip labels to target class (e.g., always predict class 0)
        target_class = 0
        y_poisoned[poison_indices] = target_class

        # 4. Train a new model on poisoned data
        # Clone the base model to get same architecture
        self.poisoned_model = clone(self.base_model)
        self.poisoned_model.fit(X_poisoned, y_poisoned)

        # 5. Test backdoor effectiveness
        # Apply trigger to test samples
        X_triggered = self._apply_trigger(X_input, self.backdoor_trigger)

        # Predict on triggered samples
        y_triggered_pred = self.poisoned_model.predict(X_triggered)

        # 6. Calculate Attack Success Rate (ASR)
        # Proportion of triggered samples predicted as target class
        backdoor_success = np.sum(y_triggered_pred == target_class) / len(y_triggered_pred)

        # Also measure impact on clean accuracy
        y_clean_pred = self.poisoned_model.predict(X_input)
        # Get original predictions from clean model
        y_original_pred = self.base_model.predict(X_input)
        clean_accuracy_impact = accuracy_score(y_original_pred, y_clean_pred)

        # 7. Create comprehensive metrics
        metrics = AdversarialMetrics.for_poisoning_attack(
            attack_type="BackdoorPoisoning",
            poisoning_success_rate=float(backdoor_success),
            samples_tested=len(X_input),
            clean_accuracy_impact=clean_accuracy_impact
        )

        return metrics

    def get_poisoned_model(self) -> BaseEstimator:
        """
        Returns the model trained on poisoned data.

        :return: The poisoned model
        """
        return self.poisoned_model

    def get_trigger(self) -> np.ndarray:
        """
        Returns the backdoor trigger pattern.

        :return: The trigger pattern
        """
        return self.backdoor_trigger
