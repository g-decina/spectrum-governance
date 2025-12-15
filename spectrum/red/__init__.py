"""
spectrum.red
============

Red Team Module - Adversarial Attack Framework

This module provides comprehensive adversarial attack capabilities for assessing
model robustness, privacy, and security vulnerabilities.

Attack Categories:
------------------
1. Evasion Attacks (attack.py):
   - HopSkipJumpWrapper: Decision-based boundary attack
   - ZooAttackWrapper: Zeroth-order optimization attack
   - BoundaryAttackWrapper: Boundary attack
   - SquareAttackWrapper: Query-efficient square attack

2. Inference Attacks (inference.py):
   - MembershipInferenceWrapper: Detect training data membership
   - AttributeInferenceWrapper: Infer hidden attribute values

3. Extraction Attacks (extraction.py):
   - FunctionallyEquivalentExtractionWrapper: Model stealing via queries

4. Poisoning Attacks (poisoning.py):
   - PoisoningAttackWrapper: Backdoor injection in training data

5. LLM Attacks (llm_scan.py):
   - InjectionScanner: Prompt injection testing

Base Classes:
-------------
- AttackScenario: Abstract base class for all attacks

Utilities:
----------
- metrics.py: Adversarial metrics and measurements
- assessment.py: Regulatory assessment logic
"""

# Evasion attacks
from spectrum.red.attack import (
    HopSkipJumpWrapper,
    ZooAttackWrapper,
    BoundaryAttackWrapper,
    SquareAttackWrapper
)

# Inference attacks
from spectrum.red.inference import (
    MembershipInferenceWrapper,
    AttributeInferenceWrapper
)

# Extraction attacks
from spectrum.red.extraction import (
    FunctionallyEquivalentExtractionWrapper
)

# Poisoning attacks
from spectrum.red.poisoning import (
    PoisoningAttackWrapper
)

# LLM attacks
from spectrum.red.llm_scan import (
    InjectionScanner
)

# Base classes
from spectrum.red.scenario import (
    AttackScenario
)

__all__ = [
    # Evasion
    "HopSkipJumpWrapper",
    "ZooAttackWrapper",
    "BoundaryAttackWrapper",
    "SquareAttackWrapper",

    # Inference
    "MembershipInferenceWrapper",
    "AttributeInferenceWrapper",

    # Extraction
    "FunctionallyEquivalentExtractionWrapper",

    # Poisoning
    "PoisoningAttackWrapper",

    # LLM
    "InjectionScanner",

    # Base
    "AttackScenario",
]
