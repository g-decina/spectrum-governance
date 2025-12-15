import numpy as np

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.estimators.classification import SklearnClassifier
from sklearn.base import BaseEstimator

from spectrum.red.scenario import AttackScenario
from spectrum.red.metrics import AdversarialMetrics
from spectrum.utils.threading import _with_limited_threads
from spectrum.infra.types import RiskProfile

"""
spectrum.red.inference
======================

Inference Attacks for Privacy Risk Assessment.

This module provides wrappers for privacy inference attacks from ART that attempt
to infer sensitive information about the training data or model. These attacks
help assess privacy vulnerabilities in deployed models.

INFERENCE ATTACK TYPES:
-----------------------
1. Membership Inference: Determines whether a specific data point was part of
   the training dataset. This can reveal sensitive information about individuals.

2. Attribute Inference: Infers the value of a sensitive attribute (feature) that
   was not provided as input, based on other features and model predictions.

ADVERSARIAL METRICS:
--------------------
Inference attacks return AdversarialMetrics with privacy-specific fields:
- Privacy Leakage Score: Primary metric (higher means more privacy risk)
- True Positive Rate: For membership inference (identifying training samples)
- False Positive Rate: For membership inference (false alarms)
- Attack Success Rate: Overall inference accuracy

KEY COMPONENTS:
---------------
- MembershipInferenceWrapper: Tests if specific samples were in the training set.
- AttributeInferenceWrapper: Attempts to infer hidden/sensitive attributes.

USAGE FLOW:
-----------
1. Initialize with model and training data:
    `mia = MembershipInferenceWrapper(base_model=model, X_train=X_train, y_train=y_train)`
2. Execute the test on test data:
    `metrics = mia.run(X_test)`
3. Interpret: metrics.privacy_leakage_score > 0.6 indicates significant privacy risk.
"""


class MembershipInferenceWrapper(AttackScenario):
    """
    Implements a Membership Inference Attack to determine if specific samples
    were part of the model's training dataset.

    This attack trains a separate "attack model" that learns to distinguish
    between the model's behavior on training vs. non-training data.

    READ: https://arxiv.org/abs/1610.05820
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        risk_profile: RiskProfile = None
    ):
        """
        Initializes the Membership Inference Attack.

        :param base_model: The target model to attack
        :param X_train: Training data used to train the target model
        :param y_train: Training labels
        :param risk_profile: Optional risk configuration
        """
        self.art_estimator = self._adapt_to_art(base_model)
        self.X_train = X_train
        self.y_train = y_train
        self.profile = risk_profile

    def _adapt_to_art(self, base_model: BaseEstimator) -> SklearnClassifier:
        """Converts an sklearn estimator into an ART-compatible estimator."""
        nb_classes = getattr(base_model, 'classes_', np.array([0, 1])).shape[0]

        return SklearnClassifier(
            model=base_model,
            clip_values=(0, 1),
            preprocessing_defences=[]
        )

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """
        Executes the Membership Inference Attack and returns comprehensive metrics.

        :param X_input: Test samples (assumed NOT in training set)
        :return: AdversarialMetrics with privacy leakage measurements
        """
        # Ensure we have labels for the test data
        # For MIA, we need both member (training) and non-member (test) samples
        y_test_pred = self.art_estimator.predict(X_input)
        y_test = np.argmax(y_test_pred, axis=1)

        # 1. Initialize the Membership Inference Attack
        # This creates an attack model that will try to distinguish
        # training samples from test samples
        attack = MembershipInferenceBlackBox(
            estimator=self.art_estimator,
            input_type='prediction'  # Use model predictions as features
        )

        # 2. Train the attack model
        # The attack learns patterns that distinguish training vs. test data
        attack.fit(
            x=self.X_train,
            y=self.y_train,
            test_x=X_input,
            test_y=y_test
        )

        # 3. Infer membership on test set
        # Returns probability that each sample was in training set
        # Shape: (n_samples,) with values in [0, 1]
        inferred_train = attack.infer(x=self.X_train, y=self.y_train)
        inferred_test = attack.infer(x=X_input, y=y_test)

        # 4. Calculate Privacy Leakage Metrics
        # True Positive Rate: Correctly identifying training samples
        tpr = float(np.mean(inferred_train))
        # False Positive Rate: Incorrectly identifying test samples as training
        fpr = float(np.mean(inferred_test))

        # Privacy leakage is measured by the attack's ability to distinguish
        # A perfect attack would have TPR=1.0, FPR=0.0
        # Random guessing would have TPR≈0.5, FPR≈0.5
        privacy_leakage_score = (tpr + (1 - fpr)) / 2.0

        # 5. Create comprehensive metrics
        metrics = AdversarialMetrics.for_inference_attack(
            attack_type="MembershipInference",
            privacy_leakage_score=privacy_leakage_score,
            samples_tested=len(X_input),
            true_positive_rate=tpr,
            false_positive_rate=fpr
        )

        return metrics


class AttributeInferenceWrapper(AttackScenario):
    """
    Implements an Attribute Inference Attack that attempts to infer the value
    of a sensitive attribute that was not provided to the model.

    This attack exploits correlations between features to predict hidden
    attributes from the model's predictions on other features.

    READ: https://arxiv.org/abs/2208.09967
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        attack_feature: int,
        risk_profile: RiskProfile = None
    ):
        """
        Initializes the Attribute Inference Attack.

        :param base_model: The target model to attack
        :param attack_feature: Index of the feature to infer (will be removed from input)
        :param risk_profile: Optional risk configuration
        """
        self.art_estimator = self._adapt_to_art(base_model)
        self.attack_feature = attack_feature
        self.profile = risk_profile

    def _adapt_to_art(self, base_model: BaseEstimator) -> SklearnClassifier:
        """Converts an sklearn estimator into an ART-compatible estimator."""
        nb_classes = getattr(base_model, 'classes_', np.array([0, 1])).shape[0]

        return SklearnClassifier(
            model=base_model,
            clip_values=(0, 1),
            preprocessing_defences=[]
        )

    @_with_limited_threads
    def run(self, X_input: np.ndarray) -> AdversarialMetrics:
        """
        Executes the Attribute Inference Attack and returns comprehensive metrics.

        :param X_input: Test samples with all features
        :return: AdversarialMetrics with attribute inference measurements
        """
        # 1. Extract the target attribute values
        true_attribute_values = X_input[:, self.attack_feature].copy()

        # 2. Create input with target attribute removed
        # Stack columns before and after the attack feature
        if self.attack_feature == 0:
            X_missing = X_input[:, 1:]
        elif self.attack_feature == X_input.shape[1] - 1:
            X_missing = X_input[:, :-1]
        else:
            X_missing = np.hstack([
                X_input[:, :self.attack_feature],
                X_input[:, self.attack_feature + 1:]
            ])

        # 3. Get predictions for creating attack model training data
        y_pred = self.art_estimator.predict(X_input)
        y_labels = np.argmax(y_pred, axis=1)

        # 4. Initialize Attribute Inference Attack
        attack = AttributeInferenceBlackBox(
            estimator=self.art_estimator,
            attack_feature=self.attack_feature
        )

        # 5. Train the attack model
        # Split data for training the attack model
        n_train = len(X_input) // 2
        attack.fit(X_input[:n_train])

        # 6. Infer the attribute values on remaining data
        inferred_values = attack.infer(
            x=X_missing[n_train:],
            y=y_labels[n_train:],
            values=[np.unique(true_attribute_values)]
        )

        # 7. Calculate inference accuracy
        # For regression features, use proximity; for categorical, use exact match
        true_values_test = true_attribute_values[n_train:]

        # Check if attribute is categorical (discrete) or continuous
        unique_values = np.unique(true_attribute_values)
        if len(unique_values) <= 10:  # Assume categorical
            # Exact match accuracy
            correct_inferences = np.sum(inferred_values == true_values_test)
            inference_score = correct_inferences / len(true_values_test)
        else:  # Continuous
            # Use normalized error (lower is better, so invert)
            errors = np.abs(inferred_values - true_values_test)
            normalized_error = np.mean(errors) / (np.max(true_attribute_values) - np.min(true_attribute_values))
            inference_score = 1.0 - min(normalized_error, 1.0)

        # 8. Create comprehensive metrics
        metrics = AdversarialMetrics.for_inference_attack(
            attack_type="AttributeInference",
            privacy_leakage_score=float(inference_score),
            samples_tested=len(X_input)
        )

        return metrics
