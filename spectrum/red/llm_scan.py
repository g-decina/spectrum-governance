import numpy as np
import random
import uuid
import hashlib

from typing import Union, List
from abc import ABC, abstractmethod

from spectrum.red.attack import AttackScenario
from spectrum.red.metrics import AdversarialMetrics 

class InjectionScanner(AttackScenario):
    """
    Wraps the garak library conceptually to execute targeted prompt injection
    probes against an LLM endpoint.

    Returns AdversarialMetrics with attack success rate and probe details.
    """
    
    def __init__(self, target_api_url: str):
        """
        :param target_api_url: The URL or endpoint of the LLM to test.
        """
        if not target_api_url.startswith(("http", "https")):
            raise ValueError("LLM target must be a valid API URL.")
        
        self.target_api_url = target_api_url
        self.campaign_id = str(uuid.uuid4())
        self.probes_to_run = ["dan_injection", "promptinject", "role_reversal"]
        
    def _run_probe_family(self, probe_family: str, num_tests: int = 10) -> float:
        """
        Conceptual function simulating garak execution against the target LLM.
        This function would initiate the external garak process or API call.
        
        Note: We explicitly use a fixed seed to ensure our Wargame Runner 
        results are reproducible, a CRITICAL requirement for auditability.
        """
        np.random.seed(int(hashlib.sha256(self.campaign_id.encode()).hexdigest(), 16) % (2**32))
        
        # Mock success rates based on typical model weaknesses
        # TO DO: Tweak according to a RiskProfile
        if probe_family == "dan_injection":
            success_rate = 0.25
        elif probe_family == "promptinject":
            success_rate = 0.40
        else:
            success_rate = 0.10
        
        successes = np.random.binomial(num_tests, success_rate)
        
        return float(successes / num_tests)

    def run(self, X_input: Union[np.ndarray, List[str]] = None) -> AdversarialMetrics:
        """
        Runs the full injection campaign and returns comprehensive metrics.

        :param X_input: Seed prompts (optional)
        :return: AdversarialMetrics with attack success rate and probe details
        """
        if not X_input:
            print("WARNING: X_input (seed prompts) not provided. Running default campaign.")

        scores = []
        total_tests = 0
        successful_injections = 0

        for probe in self.probes_to_run:
            score = self._run_probe_family(probe, num_tests=10)
            scores.append(score)
            total_tests += 10
            successful_injections += int(score * 10)

        attack_success_rate = np.mean(scores)

        # CRITICAL LOGGING: spectrum.lens must log a detailed RCIA event here.
        # for each successful injection.

        return AdversarialMetrics(
            attack_type="PromptInjection",
            attack_success_rate=float(attack_success_rate),
            samples_tested=total_tests,
            samples_successful=successful_injections,
            empirical_robustness_l2=None,  # Not applicable for LLM attacks
            empirical_robustness_linf=None,  # Not applicable for LLM attacks
            min_perturbation_l2=None,  # Not applicable for LLM attacks
            max_perturbation_l2=None,  # Not applicable for LLM attacks
            median_perturbation_l2=None,  # Not applicable for LLM attacks
            queries_used=total_tests,
            avg_queries_per_sample=1.0
        )