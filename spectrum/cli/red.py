"""
spectrum red - Adversarial testing commands
"""

import typer
from pathlib import Path
from typing import Optional
import numpy as np

from spectrum.cli.utils import (
    console, print_header, print_success, print_error, print_info, print_warning,
    load_model, load_data, interpret_attack_success_rate
)
from spectrum.red.attack import (
    HopSkipJumpWrapper, ZooAttackWrapper,
    BoundaryAttackWrapper, SquareAttackWrapper
)
from spectrum.red.inference import MembershipInferenceWrapper, AttributeInferenceWrapper
from spectrum.red.extraction import FunctionallyEquivalentExtractionWrapper
from spectrum.red.poisoning import PoisoningAttackWrapper

app = typer.Typer(help="Adversarial testing and attack campaigns")


@app.command("scan")
def scan(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file (.pkl, .joblib)"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to test data (.csv, .parquet)"),
    attack_type: str = typer.Option("hopskipjump", "--attack", "-a", help="Attack type: hopskipjump, zoo, boundary, square"),
    sample_size: int = typer.Option(100, "--sample-size", "-n", help="Number of samples to test"),
    max_queries: int = typer.Option(10000, "--max-queries", help="Maximum model queries per sample"),
    risk_level: str = typer.Option("HIGH", "--risk-level", help="Risk level (LOW/MEDIUM/HIGH/CRITICAL) - determines PII policy"),
    pii_policy: Optional[str] = typer.Option(None, "--pii-policy", help="Override PII policy (detect/pseudonymize/redact/allow)"),
    pii_safe_fields: Optional[str] = typer.Option(None, "--pii-safe-fields", help="Comma-separated list of fields to mark as safe (non-PII)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute evasion attack scan against a model.

    This command runs various black-box adversarial attacks to assess
    model robustness.

    Available attacks:
    - hopskipjump: Decision-based boundary attack (default)
    - zoo: Zeroth-order optimization attack
    - boundary: Boundary attack
    - square: Square attack (query-efficient)

    Example:
        spectrum red scan --model ./model.pkl --data ./test.csv --attack hopskipjump
    """
    print_header("SPECTRUM RED - EVASION ATTACK TESTING")

    # Build RiskProfile for PII policy determination
    from spectrum.infra.types import RiskLevel, RiskProfile
    from spectrum.infra.privacy import PIIPolicy, PIISchema

    try:
        risk_level_enum = RiskLevel[risk_level.upper()]
    except KeyError:
        print_error(f"Invalid risk level: {risk_level}")
        print_info("Valid options: LOW, MEDIUM, HIGH, CRITICAL")
        raise typer.Exit(1)

    # Build RiskProfile (alpha is required but not used for PII policy)
    alpha = 0.05  # Default significance level
    profile = RiskProfile(level=risk_level_enum, alpha=alpha)

    # Override PII policy if specified
    if pii_policy:
        try:
            policy_override = PIIPolicy[pii_policy.upper()]
            # Need to reconstruct with explicit policy since RiskProfile is frozen
            from spectrum.infra.privacy import get_recommended_policy
            profile = RiskProfile(
                level=risk_level_enum,
                alpha=alpha,
                pii_policy=policy_override
            )
        except KeyError:
            print_error(f"Invalid PII policy: {pii_policy}")
            print_info("Valid options: detect, pseudonymize, redact, allow")
            raise typer.Exit(1)

    # Parse safe fields if provided
    schema = None
    if pii_safe_fields:
        safe_fields = set(field.strip() for field in pii_safe_fields.split(','))
        schema = PIISchema(safe_fields=safe_fields)
        print_info(f"Marked {len(safe_fields)} field(s) as safe (non-PII)")

    # Load model and data with PII handling
    model_obj = load_model(str(model))
    df, pii_results, audit_report = load_data(
        str(data),
        risk_profile=profile,
        pii_schema=schema,
        detect_pii=True
    )

    # Sample data if needed
    if len(df) > sample_size:
        print_info(f"Sampling {sample_size} records from {len(df)} total")
        df = df.sample(n=sample_size, random_state=42)

    # Separate features and target (assume last column is target)
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1].values

    print_info(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    console.print()

    # Select attack based on type
    attack_map = {
        "hopskipjump": (HopSkipJumpWrapper, "HopSkipJump"),
        "zoo": (ZooAttackWrapper, "ZOO"),
        "boundary": (BoundaryAttackWrapper, "Boundary"),
        "square": (SquareAttackWrapper, "Square")
    }

    if attack_type.lower() not in attack_map:
        print_error(f"Unknown attack type: {attack_type}")
        print_info(f"Available: {', '.join(attack_map.keys())}")
        raise typer.Exit(1)

    AttackClass, attack_name = attack_map[attack_type.lower()]

    # Run adversarial attack
    print_header(f"Running {attack_name} Attack")
    print_info("This may take several minutes...")
    console.print()

    try:
        attack = AttackClass(base_model=model_obj)
        metrics = attack.run(X_test.values)

        console.print()
        print_success("Attack complete")
        console.print()

        # Interpret results using attack success rate
        risk_level, interpretation = interpret_attack_success_rate(metrics.attack_success_rate)

        # Display comprehensive metrics summary
        console.print(metrics.summary())
        console.print()

        # Color-coded verdict
        if metrics.attack_success_rate <= 0.30:
            console.print("✓ Model demonstrates strong robustness", style="bold green")
        elif metrics.attack_success_rate <= 0.60:
            console.print("⚠ Model shows moderate vulnerability", style="bold yellow")
        else:
            console.print("✗ Model is highly vulnerable", style="bold red")

        # Save results if output specified
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / f"{attack_type}_scan_results.txt"

            with open(result_file, 'w') as f:
                f.write(metrics.summary())
                f.write("\n\n")
                f.write(f"Risk Level: {risk_level}\n")
                f.write(f"Interpretation: {interpretation}\n")

            # Also save metrics as JSON
            import json
            json_file = output / f"{attack_type}_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            print_success(f"Results saved to {result_file}")
            print_success(f"Metrics saved to {json_file}")

            # Save PII audit report if PII was detected and sanitized
            if audit_report and audit_report.get('action') != 'none':
                pii_audit_file = output / "pii_audit_report.json"
                with open(pii_audit_file, 'w') as f:
                    json.dump(audit_report, f, indent=2)
                print_success(f"PII audit report saved to {pii_audit_file}")

    except Exception as e:
        print_error(f"Attack failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("llm-probe")
def llm_probe(
    endpoint: str = typer.Option(..., "--endpoint", "-e", help="LLM API endpoint URL"),
    probe_families: str = typer.Option("dan,promptinject", "--probes", "-p", help="Comma-separated probe families"),
    max_attempts: int = typer.Option(100, "--max-attempts", help="Maximum probe attempts per family"),
):
    """
    Execute LLM security probes (PLACEHOLDER - Not yet implemented).
    
    This command will run security probes against language model endpoints
    including DAN attacks, prompt injection, and encoding-based attacks.
    """
    print_header("SPECTRUM RED - LLM SECURITY PROBES")
    print_warning("LLM probes not yet implemented")
    print_info("This feature is planned for a future release")
    
    # Placeholder logic
    console.print(f"\nTarget endpoint: {endpoint}")
    console.print(f"Probe families: {probe_families}")
    console.print(f"Max attempts: {max_attempts}")
    
    raise typer.Exit(0)


@app.command("inference")
def inference_attack(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file (.pkl, .joblib)"),
    train_data: Path = typer.Option(..., "--train-data", help="Path to training data (.csv, .parquet)"),
    test_data: Path = typer.Option(..., "--test-data", help="Path to test data (.csv, .parquet)"),
    attack_type: str = typer.Option("membership", "--attack", "-a", help="Attack type: membership, attribute"),
    attribute_index: Optional[int] = typer.Option(None, "--attribute-idx", help="Feature index for attribute inference"),
    sample_size: int = typer.Option(100, "--sample-size", "-n", help="Number of samples to test"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute privacy inference attacks against a model.

    This command runs inference attacks to assess privacy vulnerabilities.

    Available attacks:
    - membership: Membership Inference Attack (requires train + test data)
    - attribute: Attribute Inference Attack (requires --attribute-idx)

    Example:
        spectrum red inference --model ./model.pkl --train-data ./train.csv --test-data ./test.csv --attack membership
    """
    print_header("SPECTRUM RED - PRIVACY INFERENCE TESTING")

    # Load model and data (no PII handling for inference attacks)
    model_obj = load_model(str(model))
    df_train, _, _ = load_data(str(train_data), detect_pii=False)
    df_test, _, _ = load_data(str(test_data), detect_pii=False)

    # Sample data if needed
    if len(df_test) > sample_size:
        print_info(f"Sampling {sample_size} test records from {len(df_test)} total")
        df_test = df_test.sample(n=sample_size, random_state=42)

    # Separate features and target
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    print_info(f"Train set: {len(X_train)} samples")
    print_info(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    console.print()

    try:
        if attack_type.lower() == "membership":
            print_header("Running Membership Inference Attack")
            print_info("Testing if samples were in training set...")
            console.print()

            attack = MembershipInferenceWrapper(
                base_model=model_obj,
                X_train=X_train,
                y_train=y_train
            )
            metrics = attack.run(X_test)

            console.print()
            print_success("Attack complete")
            console.print()

            console.print(metrics.summary())
            console.print()

            privacy_score = metrics.privacy_leakage_score
            if privacy_score <= 0.55:
                console.print("✓ Low privacy risk (near random)", style="bold green")
            elif privacy_score <= 0.70:
                console.print("⚠ Moderate privacy leakage", style="bold yellow")
            else:
                console.print("✗ High privacy risk", style="bold red")

        elif attack_type.lower() == "attribute":
            if attribute_index is None:
                print_error("--attribute-idx required for attribute inference")
                raise typer.Exit(1)

            print_header("Running Attribute Inference Attack")
            print_info(f"Inferring attribute at index {attribute_index}...")
            console.print()

            attack = AttributeInferenceWrapper(
                base_model=model_obj,
                attack_feature=attribute_index
            )
            metrics = attack.run(X_test)

            console.print()
            print_success("Attack complete")
            console.print()

            console.print(metrics.summary())
            console.print()

            inference_score = metrics.privacy_leakage_score
            if inference_score <= 0.30:
                console.print("✓ Low inference risk", style="bold green")
            elif inference_score <= 0.60:
                console.print("⚠ Moderate inference accuracy", style="bold yellow")
            else:
                console.print("✗ High inference accuracy (privacy risk)", style="bold red")

        else:
            print_error(f"Unknown attack type: {attack_type}")
            print_info("Available: membership, attribute")
            raise typer.Exit(1)

        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / f"{attack_type}_inference_results.txt"

            with open(result_file, 'w') as f:
                f.write(metrics.summary())

            # Save metrics as JSON
            import json
            json_file = output / f"{attack_type}_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            print_success(f"Results saved to {result_file}")
            print_success(f"Metrics saved to {json_file}")

    except Exception as e:
        print_error(f"Attack failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("extraction")
def extraction_attack(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file (.pkl, .joblib)"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to test data (.csv, .parquet)"),
    sample_size: int = typer.Option(100, "--sample-size", "-n", help="Number of samples for extraction"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute model extraction attack to steal model behavior.

    This command attempts to create a surrogate model that replicates
    the target model's behavior through strategic queries.

    Example:
        spectrum red extraction --model ./model.pkl --data ./test.csv
    """
    print_header("SPECTRUM RED - MODEL EXTRACTION TESTING")

    # Load model and data (no PII handling for extraction attacks)
    model_obj = load_model(str(model))
    df, _, _ = load_data(str(data), detect_pii=False)

    # Sample data if needed
    if len(df) > sample_size:
        print_info(f"Sampling {sample_size} records from {len(df)} total")
        df = df.sample(n=sample_size, random_state=42)

    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    print_info(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    console.print()

    print_header("Running Functionally Equivalent Extraction Attack")
    print_info("Creating surrogate model through queries...")
    console.print()

    try:
        attack = FunctionallyEquivalentExtractionWrapper(base_model=model_obj)
        metrics = attack.run(X_test)

        console.print()
        print_success("Attack complete")
        console.print()

        console.print(metrics.summary())
        console.print()

        extraction_score = metrics.extraction_fidelity
        if extraction_score <= 0.60:
            console.print("✓ Low extraction risk", style="bold green")
        elif extraction_score <= 0.80:
            console.print("⚠ Moderate model theft risk", style="bold yellow")
        else:
            console.print("✗ High model theft risk (near-perfect clone)", style="bold red")

        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "extraction_results.txt"

            with open(result_file, 'w') as f:
                f.write(metrics.summary())

            # Save metrics as JSON
            import json
            json_file = output / "extraction_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            print_success(f"Results saved to {result_file}")
            print_success(f"Metrics saved to {json_file}")

    except Exception as e:
        print_error(f"Attack failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("poisoning")
def poisoning_attack(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file (.pkl, .joblib)"),
    train_data: Path = typer.Option(..., "--train-data", help="Path to training data (.csv, .parquet)"),
    test_data: Path = typer.Option(..., "--test-data", help="Path to test data (.csv, .parquet)"),
    poison_ratio: float = typer.Option(0.1, "--poison-ratio", help="Fraction of training data to poison (0.0-1.0)"),
    sample_size: int = typer.Option(100, "--sample-size", "-n", help="Number of test samples"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute data poisoning attack to test training data integrity.

    This command injects backdoor triggers into training data and
    measures the attack success rate.

    Example:
        spectrum red poisoning --model ./model.pkl --train-data ./train.csv --test-data ./test.csv
    """
    print_header("SPECTRUM RED - DATA POISONING TESTING")

    # Load model and data (no PII handling for poisoning attacks)
    model_obj = load_model(str(model))
    df_train, _, _ = load_data(str(train_data), detect_pii=False)
    df_test, _, _ = load_data(str(test_data), detect_pii=False)

    # Sample data if needed
    if len(df_test) > sample_size:
        print_info(f"Sampling {sample_size} test records from {len(df_test)} total")
        df_test = df_test.sample(n=sample_size, random_state=42)

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    print_info(f"Train set: {len(X_train)} samples")
    print_info(f"Test set: {len(X_test)} samples")
    print_info(f"Poison ratio: {poison_ratio:.1%}")
    console.print()

    print_header("Running Backdoor Poisoning Attack")
    print_info("Injecting backdoor triggers into training data...")
    console.print()

    try:
        attack = PoisoningAttackWrapper(
            base_model=model_obj,
            X_train=X_train,
            y_train=y_train,
            poison_ratio=poison_ratio
        )
        metrics = attack.run(X_test)

        console.print()
        print_success("Attack complete")
        console.print()

        console.print(metrics.summary())
        console.print()

        poisoning_score = metrics.poisoning_success_rate
        if poisoning_score <= 0.50:
            console.print("✓ Low poisoning risk (backdoor ineffective)", style="bold green")
        elif poisoning_score <= 0.70:
            console.print("⚠ Moderate poisoning vulnerability", style="bold yellow")
        else:
            console.print("✗ High poisoning risk (backdoor effective)", style="bold red")

        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "poisoning_results.txt"

            with open(result_file, 'w') as f:
                f.write(metrics.summary())
                f.write(f"\nPoison Ratio: {poison_ratio:.1%}\n")

            # Save metrics as JSON
            import json
            json_file = output / "poisoning_metrics.json"
            with open(json_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            print_success(f"Results saved to {result_file}")
            print_success(f"Metrics saved to {json_file}")

    except Exception as e:
        print_error(f"Attack failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
