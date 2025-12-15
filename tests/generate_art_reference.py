#!/usr/bin/env python3
"""
Generate reference data from IBM ART for equivalence testing.

This script runs ART's HopSkipJump attack on a simple model and saves
the results as JSON for comparison with the Rust implementation.

Requirements:
    pip install adversarial-robustness-toolbox scikit-learn numpy

Usage:
    python tests/generate_art_reference.py --attack hopskipjump --output tests/fixtures/art/
"""

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump


def create_simple_model():
    """
    Create a simple linear model with a known decision boundary.

    Decision rule: x[0] < 0.5 -> class 0, else class 1
    This matches the MockLinearModel in the Rust tests.
    """
    # Generate training data with clear boundary at x[0] = 0.5
    np.random.seed(42)
    n_samples = 1000
    n_features = 2

    X = np.random.rand(n_samples, n_features)
    y = (X[:, 0] >= 0.5).astype(int)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    # Verify decision boundary
    test_left = np.array([[0.3, 0.5]])
    test_right = np.array([[0.7, 0.5]])
    assert model.predict(test_left)[0] == 0, "Boundary check failed"
    assert model.predict(test_right)[0] == 1, "Boundary check failed"

    return model


def generate_hopskipjump_reference(output_dir: Path):
    """
    Generate HopSkipJump reference data.

    Runs ART's HopSkipJump attack and saves:
    - Original samples
    - Adversarial samples
    - Original predictions
    - Adversarial predictions
    - Attack parameters
    - Metrics (success rate, L2 norms, query counts)
    """
    print("Generating HopSkipJump reference data...")

    # Create model
    model = create_simple_model()
    art_classifier = SklearnClassifier(model=model)

    # Test samples (class 0, left of boundary)
    np.random.seed(123)  # Different seed for test data
    x_test = np.array([
        [0.1, 0.5],
        [0.2, 0.6],
        [0.3, 0.4],
    ])

    # Get original predictions
    y_original = art_classifier.predict(x_test)
    y_original_labels = np.argmax(y_original, axis=1)

    print(f"Original samples: {x_test.shape}")
    print(f"Original labels: {y_original_labels}")

    # Configure HopSkipJump attack
    # Use same parameters as Rust tests for direct comparison
    attack_config = {
        'max_iter': 2,
        'max_eval': 150,
        'init_eval': 100,
        'init_size': 50,
        'verbose': False,
    }

    attack = HopSkipJump(
        classifier=art_classifier,
        **attack_config
    )

    # Run attack
    print("Running HopSkipJump attack...")
    x_adversarial = attack.generate(x=x_test)

    # Get adversarial predictions
    y_adversarial = art_classifier.predict(x_adversarial)
    y_adversarial_labels = np.argmax(y_adversarial, axis=1)

    print(f"Adversarial labels: {y_adversarial_labels}")

    # Compute metrics
    success_mask = y_original_labels != y_adversarial_labels
    success_rate = np.mean(success_mask)

    perturbations = x_adversarial - x_test
    l2_norms = np.linalg.norm(perturbations, ord=2, axis=1)
    linf_norms = np.linalg.norm(perturbations, ord=np.inf, axis=1)

    # Filter successful attacks for perturbation statistics
    if success_rate > 0:
        successful_l2 = l2_norms[success_mask]
        successful_linf = linf_norms[success_mask]

        empirical_robustness_l2 = np.mean(successful_l2)
        empirical_robustness_linf = np.mean(successful_linf)
        min_l2 = np.min(successful_l2)
        max_l2 = np.max(successful_l2)
        median_l2 = np.median(successful_l2)
        std_l2 = np.std(successful_l2)
        std_linf = np.std(successful_linf)
    else:
        empirical_robustness_l2 = 0.0
        empirical_robustness_linf = 0.0
        min_l2 = 0.0
        max_l2 = 0.0
        median_l2 = 0.0
        std_l2 = 0.0
        std_linf = 0.0

    # Prepare reference data
    reference_data = {
        'attack_type': 'HopSkipJump',
        'attack_config': attack_config,
        'test_samples': x_test.tolist(),
        'adversarial_samples': x_adversarial.tolist(),
        'original_labels': y_original_labels.tolist(),
        'adversarial_labels': y_adversarial_labels.tolist(),
        'metrics': {
            'attack_success_rate': float(success_rate),
            'samples_tested': int(len(x_test)),
            'samples_successful': int(np.sum(success_mask)),
            'empirical_robustness_l2': float(empirical_robustness_l2),
            'empirical_robustness_linf': float(empirical_robustness_linf),
            'min_perturbation_l2': float(min_l2),
            'max_perturbation_l2': float(max_l2),
            'median_perturbation_l2': float(median_l2),
            'perturbation_std_l2': float(std_l2),
            'perturbation_std_linf': float(std_linf),
        },
        'perturbations': {
            'l2_norms': l2_norms.tolist(),
            'linf_norms': linf_norms.tolist(),
        },
        'notes': 'Generated with ART for Rust equivalence testing. Model: LogisticRegression with decision boundary at x[0]=0.5'
    }

    # Save to file
    output_file = output_dir / 'hopskipjump_reference.json'
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"\nReference data saved to: {output_file}")
    print(f"Attack success rate: {success_rate:.2%}")
    print(f"Mean L2 perturbation: {empirical_robustness_l2:.6f}")
    print(f"Mean L∞ perturbation: {empirical_robustness_linf:.6f}")

    return reference_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate ART reference data for equivalence testing'
    )
    parser.add_argument(
        '--attack',
        type=str,
        default='hopskipjump',
        choices=['hopskipjump'],
        help='Attack to generate reference data for'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('tests/fixtures/art/'),
        help='Output directory for reference data'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate reference data
    if args.attack == 'hopskipjump':
        generate_hopskipjump_reference(args.output)
    else:
        raise ValueError(f"Unknown attack: {args.attack}")

    print("\n✓ Reference data generation complete!")


if __name__ == '__main__':
    main()
