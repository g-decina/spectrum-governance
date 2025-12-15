import numpy as np

from art.attacks.extraction import FunctionallyEquivalentExtraction
from art.estimators.classification import SklearnClassifier
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from spectrum.red.scenario import AttackScenario
from spectrum.red.metrics import AdversarialMetrics
from spectrum.utils.threading import _with_limited_threads
from spectrum.infra.types import RiskProfile

"""
spectrum.red.extraction
=======================

Model Extraction Attacks for IP Protection Assessment.

This module provides wrappers for model extraction/stealing attacks from ART.
These attacks attempt to create a surrogate model that replicates the behavior
of the target model through strategic queries.

EXTRACTION ATTACK TYPES:
------------------------
1. Functionally Equivalent Extraction: Creates a surrogate model that mimics
   the target model's decision boundaries and predictions through adaptive
   query strategies.

ADVERSARIAL METRICS:
--------------------
Extraction attacks return AdversarialMetrics with extraction-specific fields:
- Extraction Fidelity: Agreement between target and stolen model (0.0-1.0)
- Extraction Queries: Number of queries used for model stealing
- Attack Success Rate: Overall extraction effectiveness

Higher fidelity scores indicate greater IP theft vulnerability.

KEY COMPONENTS:
---------------
- FunctionallyEquivalentExtractionWrapper: Adaptive query-based model stealing.

USAGE FLOW:
-----------
1. Initialize with target model:
    `extractor = FunctionallyEquivalentExtractionWrapper(base_model=model)`
2. Execute extraction attack:
    `metrics = extractor.run(X_test)`
3. Interpret: metrics.extraction_fidelity > 0.8 indicates high model theft risk.
"""


class FunctionallyEquivalentExtractionWrapper(AttackScenario):
    """
    Implements Functionally Equivalent Extraction attack that creates a
    surrogate model mimicking the target model through strategic queries.
    This class only supports classification models.

    The attack uses adaptive sampling to efficiently explore the decision
    space and train a substitute model that behaves similarly to the target.

    READ: https://arxiv.org/abs/1909.01838 / https://arxiv.org/abs/1811.02054v6
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        risk_profile: RiskProfile = None
    ):
        """
        Initializes the Functionally Equivalent Extraction Attack.

        :param base_model: The target model to extract/steal
        :param risk_profile: Optional risk configuration
        """
        self.art_estimator = self._adapt_to_art(base_model)
        self.base_model = base_model
        self.profile = risk_profile
        self.stolen_model = None

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
        Executes the Model Extraction Attack and returns comprehensive metrics.

        :param X_input: Input samples to use for extraction queries
        :return: AdversarialMetrics with extraction fidelity measurements
        """
        # 1. Initialize the extraction attack
        # The stolen model will be a Decision Tree (simple but effective)
        stolen_classifier = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )

        # 2. Create ART estimator for the stolen model
        nb_classes = getattr(self.base_model, 'classes_', np.array([0, 1])).shape[0]
        stolen_art_estimator = SklearnClassifier(
            model=stolen_classifier,
            clip_values=(0, 1)
        )

        # 3. Initialize Functionally Equivalent Extraction Attack
        num_samples_for_extraction = min(1000, len(X_input) * 5)
        attack = FunctionallyEquivalentExtraction(
            classifier=self.art_estimator,
            stolen_classifier=stolen_art_estimator,
            num_samples=num_samples_for_extraction,
            delta=0.1,
        )

        # 4. Execute the extraction attack
        # This queries the target model and trains the surrogate
        X_extracted = attack.extract(x=X_input, y=None)

        # 5. Get predictions from both models on test set
        y_target = self.art_estimator.predict(X_input)
        y_target_labels = np.argmax(y_target, axis=1)

        # Train the stolen model on extracted data if needed
        extraction_queries = num_samples_for_extraction
        if X_extracted is not None:
            y_extracted = self.art_estimator.predict(X_extracted)
            y_extracted_labels = np.argmax(y_extracted, axis=1)
            stolen_classifier.fit(X_extracted, y_extracted_labels)
            extraction_queries = len(X_extracted)

        y_stolen = stolen_art_estimator.predict(X_input)
        y_stolen_labels = np.argmax(y_stolen, axis=1)

        # 6. Calculate Extraction Fidelity (agreement between target and stolen model)
        extraction_fidelity = accuracy_score(y_target_labels, y_stolen_labels)

        # Store the stolen model for analysis
        self.stolen_model = stolen_classifier

        # 7. Create comprehensive metrics
        metrics = AdversarialMetrics.for_extraction_attack(
            attack_type="FunctionallyEquivalentExtraction",
            extraction_fidelity=extraction_fidelity,
            samples_tested=len(X_input),
            extraction_queries=extraction_queries
        )

        return metrics

    def get_stolen_model(self) -> BaseEstimator:
        """
        Returns the stolen/surrogate model created by the extraction attack.

        :return: The surrogate model
        """
        return self.stolen_model
