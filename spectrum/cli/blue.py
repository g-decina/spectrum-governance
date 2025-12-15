"""
spectrum blue - Defensive evaluation commands
"""

import typer
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from spectrum.cli.utils import (
    console, print_header, print_success, print_error, print_info, print_warning,
    load_model, load_data, format_percentage, interpret_drift_status, create_summary_table
)
from spectrum.blue.explain import SpectrumUncertaintyWrapper, generate_shap_explanations
from spectrum.blue.monitor import DriftCheck
from spectrum.blue.timeseries import EnbPIRegressor, TimeSeriesValidator, TimeSeriesPrediction
from spectrum.infra.types import RiskProfile, RiskLevel

app = typer.Typer(help="Defensive hardening and evaluation")


@app.command("explain")
def explain(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to data for explanation"),
    max_samples: int = typer.Option(10, "--max-samples", "-n", help="Number of samples to explain"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """
    Generate SHAP-based model explanations.
    
    Analyzes which features contribute most to model predictions
    using SHAP (SHapley Additive exPlanations) values.
    
    Example:
        spectrum blue explain --model ./model.pkl --data ./test.csv --max-samples 10
    """
    print_header("SPECTRUM BLUE - MODEL EXPLAINABILITY")
    
    # Load model and data
    model_obj = load_model(str(model))
    df, _, _ = load_data(str(data), detect_pii=False)  # No PII handling for explain

    # Sample data
    X_test = df.iloc[:, :-1]
    if len(X_test) > max_samples:
        X_test = X_test.sample(n=max_samples, random_state=42)
    
    print_info(f"Generating explanations for {len(X_test)} samples...")
    console.print()
    
    try:
        # Generate SHAP explanations
        explanations = generate_shap_explanations(
            model=model_obj,
            X_test=X_test,
            max_samples=max_samples
        )
        
        print_success("Explanations generated")
        console.print()
        
        # Display top features
        console.print("[bold]Top Feature Contributions:[/bold]")
        for i, feature in enumerate(explanations['top_features'], 1):
            console.print(f"  {i}. {feature}")
        
        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "explanations.txt"
            
            with open(result_file, 'w') as f:
                f.write("Top Feature Contributions:\n")
                for i, feature in enumerate(explanations['top_features'], 1):
                    f.write(f"  {i}. {feature}\n")
            
            print_success(f"Results saved to {result_file}")
    
    except Exception as e:
        print_error(f"Explanation generation failed: {e}")
        raise typer.Exit(1)


@app.command("drift")
def drift(
    reference: Path = typer.Option(..., "--reference", "-r", help="Path to reference data (e.g., training data)"),
    current: Path = typer.Option(..., "--current", "-c", help="Path to current/production data"),
    psi_threshold: float = typer.Option(0.20, "--psi-threshold", help="PSI alert threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """
    Analyze distribution drift between reference and current data.
    
    Uses Population Stability Index (PSI) to detect feature drift.
    
    Thresholds:
      - PSI < 0.10: No significant drift
      - PSI 0.10-0.25: Moderate drift (monitor)
      - PSI > 0.25: Significant drift (alert)
    
    Example:
        spectrum blue drift --reference ./train.csv --current ./production.csv
    """
    print_header("SPECTRUM BLUE - DRIFT MONITORING")
    
    # Load data (no PII handling for drift detection)
    df_reference, _, _ = load_data(str(reference), detect_pii=False)
    df_current, _, _ = load_data(str(current), detect_pii=False)
    
    print_info(f"Reference set: {len(df_reference)} samples")
    print_info(f"Current set: {len(df_current)} samples")
    console.print()
    
    try:
        # Run drift check
        drift_results = DriftCheck(df_reference, df_current)
        
        print_success("Drift analysis complete")
        console.print()
        
        # Display summary
        status_symbol, status_msg = interpret_drift_status(
            drift_results['max_psi'],
            alert_threshold=drift_results.get('alert_threshold', 0.25),
            monitor_threshold=drift_results.get('monitor_threshold', 0.10)
        )
        
        console.print(f"[bold]Status:[/bold] {status_symbol} {status_msg}")
        console.print(f"[bold]Max PSI:[/bold] {drift_results['max_psi']:.4f}")
        console.print(f"[bold]Alert Required:[/bold] {drift_results['alert_required']}")
        console.print(f"[bold]Drifted Features:[/bold] {drift_results['n_drifted_features']}")
        console.print()
        
        # Display per-feature drift
        if drift_results.get('feature_drift_scores'):
            console.print("[bold]Per-Feature Drift Scores:[/bold]")
            for feature, psi in sorted(
                drift_results['feature_drift_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                symbol = "⚠" if psi >= 0.25 else "⚡" if psi >= 0.10 else "✓"
                console.print(f"  {symbol} {feature}: {psi:.4f}")
        
        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "drift_analysis.txt"
            
            with open(result_file, 'w') as f:
                f.write(f"Status: {status_msg}\n")
                f.write(f"Max PSI: {drift_results['max_psi']:.4f}\n")
                f.write(f"Alert Required: {drift_results['alert_required']}\n")
                f.write(f"Drifted Features: {drift_results['n_drifted_features']}\n\n")
                f.write("Per-Feature Drift Scores:\n")
                for feature, psi in drift_results['feature_drift_scores'].items():
                    f.write(f"  {feature}: {psi:.4f}\n")
            
            print_success(f"Results saved to {result_file}")
    
    except Exception as e:
        print_error(f"Drift analysis failed: {e}")
        raise typer.Exit(1)


@app.command("uncertainty")
def uncertainty(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file"),
    calibration_data: Path = typer.Option(..., "--calibration-data", "-c", help="Path to calibration data"),
    test_data: Path = typer.Option(..., "--test-data", "-t", help="Path to test data"),
    risk_level: str = typer.Option("HIGH", "--risk-level", "-r", help="Risk level: HIGH, MEDIUM, LOW"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """
    Evaluate uncertainty quantification and coverage.
    
    Uses conformal prediction (MAPIE) to generate prediction intervals
    with guaranteed coverage.
    
    Risk Levels:
      - HIGH: 95% confidence (alpha=0.05)
      - MEDIUM: 90% confidence (alpha=0.10)  
      - LOW: 80% confidence (alpha=0.20)
    
    Example:
        spectrum blue uncertainty --model ./model.pkl \\
            --calibration-data ./calib.csv --test-data ./test.csv \\
            --risk-level HIGH
    """
    print_header("SPECTRUM BLUE - UNCERTAINTY QUANTIFICATION")
    
    # Load model and data (no PII handling for uncertainty quantification)
    model_obj = load_model(str(model))
    df_calib, _, _ = load_data(str(calibration_data), detect_pii=False)
    df_test, _, _ = load_data(str(test_data), detect_pii=False)
    
    # Parse risk level
    try:
        risk_enum = RiskLevel[risk_level.upper()]
        # Map risk level to alpha value
        alpha_mapping = {
            RiskLevel.HIGH: 0.05,      # 95% confidence
            RiskLevel.MEDIUM: 0.10,    # 90% confidence
            RiskLevel.LOW: 0.20,       # 80% confidence
            RiskLevel.CRITICAL: 0.01,  # 99% confidence
        }
        alpha = alpha_mapping[risk_enum]
        risk_profile = RiskProfile(level=risk_enum, alpha=alpha)
    except KeyError:
        print_error(f"Invalid risk level: {risk_level}")
        print_info("Valid options: HIGH, MEDIUM, LOW")
        raise typer.Exit(1)
    
    print_info(f"Risk Profile: {risk_profile.level.value}")
    print_info(f"Required Confidence: {(1 - risk_profile.alpha) * 100:.1f}%")
    console.print()
    
    # Separate features and target
    X_calib = df_calib.iloc[:, :-1]
    y_calib = df_calib.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1].values
    
    try:
        # Create uncertainty wrapper
        print_info("Calibrating uncertainty wrapper...")
        
        background_sample = X_calib.iloc[:100] if hasattr(X_calib, "iloc") else X_calib[:100]
        
        wrapper = SpectrumUncertaintyWrapper(
            base_model=model_obj,
            risk_profile=risk_profile,
            X_background=background_sample
        )
        wrapper.fit(X_calib, y_calib)
        
        print_success("Calibration complete")
        console.print()
        
        # Generate predictions
        print_info("Generating predictions with uncertainty...")
        results = wrapper.predict(X_test)
        
        print_success(f"Predictions generated for {len(X_test)} samples")
        console.print()
        
        # Calculate empirical coverage
        if results.y_pis is not None:
            # Regression
            lower_bounds = results.y_pis[:, 0, 0]
            upper_bounds = results.y_pis[:, 1, 0]
            coverage_mask = (y_test >= lower_bounds) & (y_test <= upper_bounds)
            coverage = float(np.mean(coverage_mask))
            
            console.print(f"[bold]Empirical Coverage:[/bold] {format_percentage(coverage)}")
            console.print(f"[bold]Required Coverage:[/bold] {format_percentage(1 - risk_profile.alpha)}")
            
            if coverage >= (1 - risk_profile.alpha):
                print_success("Coverage requirement MET")
            else:
                print_error("Coverage requirement NOT MET")
        
        elif results.y_set is not None:
            # Classification
            set_sizes = results.y_set.sum(axis=1)
            avg_set_size = float(np.mean(set_sizes))
            
            # Check if true class is in prediction set
            coverage_mask = results.y_set[np.arange(len(y_test)), y_test.astype(int)]
            coverage = float(np.mean(coverage_mask))
            
            console.print(f"[bold]Empirical Coverage:[/bold] {format_percentage(coverage)}")
            console.print(f"[bold]Required Coverage:[/bold] {format_percentage(1 - risk_profile.alpha)}")
            console.print(f"[bold]Average Set Size:[/bold] {avg_set_size:.2f}")
            
            if coverage >= (1 - risk_profile.alpha):
                print_success("Coverage requirement MET")
            else:
                print_error("Coverage requirement NOT MET")
        
        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "uncertainty_analysis.txt"
            
            with open(result_file, 'w') as f:
                f.write(f"Risk Level: {risk_profile.level.value}\n")
                f.write(f"Required Confidence: {(1 - risk_profile.alpha) * 100:.1f}%\n")
                f.write(f"Empirical Coverage: {format_percentage(coverage)}\n")
                f.write(f"Status: {'PASS' if coverage >= (1 - risk_profile.alpha) else 'FAIL'}\n")
            
            print_success(f"Results saved to {result_file}")
    
    except Exception as e:
        print_error(f"Uncertainty analysis failed: {e}")
        raise typer.Exit(1)


@app.command("forecast")
def forecast(
    model: Path = typer.Option(..., "--model", "-m", help="Path to base forecaster model"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to time series data (.csv, .parquet)"),
    n_bootstraps: int = typer.Option(50, "--bootstraps", "-b", help="Number of bootstrap ensembles"),
    window_size: int = typer.Option(100, "--window-size", "-w", help="Size of residual tracking window"),
    train_ratio: float = typer.Option(0.7, "--train-ratio", help="Proportion of data for training (0.0-1.0)"),
    risk_level: str = typer.Option("HIGH", "--risk-level", "-r", help="Risk level: HIGH, MEDIUM, LOW, CRITICAL"),
    pii_policy: Optional[str] = typer.Option(None, "--pii-policy", help="Override PII policy (detect/pseudonymize/redact/allow)"),
    pii_safe_fields: Optional[str] = typer.Option(None, "--pii-safe-fields", help="Comma-separated list of fields to mark as safe (non-PII)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Time series forecasting with EnbPI prediction intervals.

    Uses Ensemble batch Prediction Intervals (EnbPI) to provide
    adaptive prediction intervals for time series data.

    EnbPI handles:
    - Non-stationary distributions
    - Distribution shifts over time
    - Sequential/temporal dependencies

    Risk Levels:
      - HIGH: 95% confidence (alpha=0.05)
      - MEDIUM: 90% confidence (alpha=0.10)
      - LOW: 80% confidence (alpha=0.20)
      - CRITICAL: 99% confidence (alpha=0.01)

    Example:
        spectrum blue forecast --model ./forecaster.pkl --data ./timeseries.csv \\
            --bootstraps 50 --risk-level HIGH
    """
    print_header("SPECTRUM BLUE - TIME SERIES FORECASTING (EnbPI)")

    # Parse risk level and build RiskProfile
    from spectrum.infra.privacy import PIIPolicy, PIISchema

    try:
        risk_enum = RiskLevel[risk_level.upper()]
        alpha_mapping = {
            RiskLevel.HIGH: 0.05,
            RiskLevel.MEDIUM: 0.10,
            RiskLevel.LOW: 0.20,
            RiskLevel.CRITICAL: 0.01,
        }
        alpha = alpha_mapping[risk_enum]
        risk_profile = RiskProfile(level=risk_enum, alpha=alpha)
    except KeyError:
        print_error(f"Invalid risk level: {risk_level}")
        print_info("Valid options: HIGH, MEDIUM, LOW, CRITICAL")
        raise typer.Exit(1)

    # Override PII policy if specified
    if pii_policy:
        try:
            policy_override = PIIPolicy[pii_policy.upper()]
            risk_profile = RiskProfile(
                level=risk_enum,
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
    base_forecaster = load_model(str(model))
    df, pii_results, audit_report = load_data(
        str(data),
        risk_profile=risk_profile,
        pii_schema=schema,
        detect_pii=True
    )

    print_info(f"Risk Profile: {risk_profile.level.value}")
    print_info(f"Target Coverage: {(1 - risk_profile.alpha) * 100:.1f}%")
    print_info(f"Bootstrap Ensembles: {n_bootstraps}")
    print_info(f"Residual Window Size: {window_size}")
    console.print()

    # Prepare time series data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print_info(f"Time series length: {len(X)} samples")
    print_info(f"Number of features: {X.shape[1]}")
    console.print()

    try:
        # Temporal split (no shuffling)
        X_train, X_test, y_train, y_test = TimeSeriesValidator.split_temporal(
            X, y, train_ratio=train_ratio
        )

        print_info(f"Training set: {len(X_train)} samples")
        print_info(f"Test set: {len(X_test)} samples")
        console.print()

        # Initialize EnbPI
        print_info("Initializing EnbPI regressor...")
        enbpi = EnbPIRegressor(
            base_forecaster=base_forecaster,
            n_bootstraps=n_bootstraps,
            window_size=window_size,
            risk_profile=risk_profile
        )

        # Fit on training data
        print_info("Fitting ensemble models on training data...")
        enbpi.fit(X_train, y_train)

        print_success("EnbPI model fitted")
        console.print()

        # Generate predictions with intervals
        print_info("Generating predictions with intervals on test set...")
        result = enbpi.predict(X_test, return_intervals=True)

        print_success("Predictions complete")
        console.print()

        # Evaluate coverage
        print_info("Evaluating prediction interval coverage...")
        coverage_metrics = enbpi.evaluate_coverage(X_test, y_test)

        console.print()
        console.print("[bold]Coverage Evaluation:[/bold]")
        console.print(f"  Empirical Coverage: {format_percentage(coverage_metrics['empirical_coverage'])}")
        console.print(f"  Target Coverage: {format_percentage(coverage_metrics['target_coverage'])}")
        console.print(f"  Average Interval Width: {coverage_metrics['average_interval_width']:.4f}")
        console.print(f"  RMSE: {coverage_metrics['rmse']:.4f}")
        console.print(f"  Samples In Interval: {coverage_metrics['n_in_interval']} / {coverage_metrics['n_samples']}")
        console.print()

        # Coverage status
        if coverage_metrics['coverage_achieved']:
            print_success(f"✓ Coverage requirement MET ({format_percentage(coverage_metrics['empirical_coverage'])} >= {format_percentage(coverage_metrics['target_coverage'])})")
        else:
            print_warning(f"⚠ Coverage requirement NOT MET ({format_percentage(coverage_metrics['empirical_coverage'])} < {format_percentage(coverage_metrics['target_coverage'])})")

        # Display sample predictions
        if verbose and len(result.predictions) > 0:
            console.print()
            console.print("[bold]Sample Predictions (first 5):[/bold]")
            n_display = min(5, len(result.predictions))
            for i in range(n_display):
                pred = result.predictions[i]
                lower = result.lower_bounds[i]
                upper = result.upper_bounds[i]
                actual = y_test[i]
                in_interval = "✓" if lower <= actual <= upper else "✗"
                console.print(f"  Sample {i+1}: {in_interval}")
                console.print(f"    Prediction: {pred:.4f}")
                console.print(f"    Interval: [{lower:.4f}, {upper:.4f}]")
                console.print(f"    Actual: {actual:.4f}")

        # Save results
        if output:
            output.mkdir(parents=True, exist_ok=True)
            result_file = output / "forecast_results.txt"

            with open(result_file, 'w') as f:
                f.write(f"EnbPI Time Series Forecasting Results\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Risk Level: {risk_profile.level.value}\n")
                f.write(f"Target Coverage: {format_percentage(coverage_metrics['target_coverage'])}\n")
                f.write(f"Bootstrap Ensembles: {n_bootstraps}\n")
                f.write(f"Window Size: {window_size}\n\n")
                f.write(f"Coverage Evaluation:\n")
                f.write(f"  Empirical Coverage: {format_percentage(coverage_metrics['empirical_coverage'])}\n")
                f.write(f"  Average Interval Width: {coverage_metrics['average_interval_width']:.4f}\n")
                f.write(f"  RMSE: {coverage_metrics['rmse']:.4f}\n")
                f.write(f"  Samples In Interval: {coverage_metrics['n_in_interval']} / {coverage_metrics['n_samples']}\n")
                f.write(f"  Coverage Achieved: {coverage_metrics['coverage_achieved']}\n\n")

                f.write(f"Predictions:\n")
                for i in range(len(result.predictions)):
                    f.write(f"  Sample {i+1}: pred={result.predictions[i]:.4f}, ")
                    f.write(f"interval=[{result.lower_bounds[i]:.4f}, {result.upper_bounds[i]:.4f}], ")
                    f.write(f"actual={y_test[i]:.4f}\n")

            print_success(f"Results saved to {result_file}")

            # Save predictions to CSV
            predictions_file = output / "predictions.csv"
            predictions_df = pd.DataFrame({
                'prediction': result.predictions,
                'lower_bound': result.lower_bounds,
                'upper_bound': result.upper_bounds,
                'actual': y_test,
                'in_interval': (y_test >= result.lower_bounds) & (y_test <= result.upper_bounds)
            })
            predictions_df.to_csv(predictions_file, index=False)
            print_success(f"Predictions saved to {predictions_file}")

            # Save PII audit report if PII was detected and sanitized
            if audit_report and audit_report.get('action') != 'none':
                import json
                pii_audit_file = output / "pii_audit_report.json"
                with open(pii_audit_file, 'w') as f:
                    json.dump(audit_report, f, indent=2)
                print_success(f"PII audit report saved to {pii_audit_file}")

    except Exception as e:
        print_error(f"Time series forecasting failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("fairness")
def fairness(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to data with predictions"),
    demographics: Optional[Path] = typer.Option(None, "--demographics", help="Path to demographic data"),
):
    """
    Perform disparate impact analysis (PLACEHOLDER - Not yet implemented).

    Implements the 4/5ths rule for CFPB/EEOC compliance.
    """
    print_header("SPECTRUM BLUE - FAIRNESS ANALYSIS")
    print_warning("Fairness analysis not yet implemented")
    print_info("DisparateImpactAnalyzer module is planned for Phase 1")

    console.print(f"\nModel: {model}")
    console.print(f"Data: {data}")
    if demographics:
        console.print(f"Demographics: {demographics}")

    raise typer.Exit(0)


@app.command("harden")
def harden(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file"),
    calibration_data: Path = typer.Option(..., "--calibration-data", "-c", help="Path to calibration data"),
    strategy: str = typer.Option("ensemble_proxy", "--strategy", "-s",
                                  help="Hardening strategy: ensemble_proxy, output_quantizer, input_sanitizer, prediction_smoother, randomized"),
    output: Path = typer.Option(..., "--output", "-o", help="Output path for hardened model"),
    test_data: Optional[Path] = typer.Option(None, "--test-data", "-t", help="Test data for validation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Harden a model against adversarial attacks without retraining.

    Wraps an existing model with defense layers that improve robustness
    at inference time. No access to training data required.

    Strategies (empirically validated on California Housing):
      - ensemble_proxy: Best overall (+66% OM, +70% PS), 2.4pp accuracy cost
      - output_quantizer: Good for OutputManipulation (+18%), 0.6pp cost
      - input_sanitizer: Good for PredictionShift (+43%), may improve accuracy
      - prediction_smoother: Modest gains (+7-9%), 1.2pp cost, higher latency
      - randomized: Limited effectiveness (+3-7%), 2.3pp cost

    Example:
        spectrum blue harden --model ./model.pkl --calibration-data ./calib.csv \\
            --strategy ensemble_proxy --output ./hardened_model.pkl
    """
    import pickle
    from spectrum.blue.harden import harden_model
    from sklearn.metrics import r2_score, accuracy_score

    print_header("SPECTRUM BLUE - MODEL HARDENING")

    # Load model and data
    model_obj = load_model(str(model))
    df_calib, _, _ = load_data(str(calibration_data), detect_pii=False)

    X_calib = df_calib.iloc[:, :-1].values

    print_info(f"Base model: {type(model_obj).__name__}")
    print_info(f"Calibration samples: {len(X_calib)}")
    print_info(f"Strategy: {strategy}")
    console.print()

    # Validate strategy
    valid_strategies = ["ensemble_proxy", "output_quantizer", "input_sanitizer",
                        "prediction_smoother", "randomized"]
    if strategy not in valid_strategies:
        print_error(f"Invalid strategy: {strategy}")
        print_info(f"Valid options: {', '.join(valid_strategies)}")
        raise typer.Exit(1)

    try:
        # Apply hardening
        print_info("Applying hardening wrapper...")
        hardened = harden_model(model_obj, X_calib, strategy=strategy)

        print_success(f"Model hardened with {strategy}")
        console.print()

        # Show expected improvements
        improvements = {
            "ensemble_proxy": ("OutputManipulation: +66%", "PredictionShift: +70%", "Accuracy cost: ~2.4pp"),
            "output_quantizer": ("OutputManipulation: +18%", "PredictionShift: +4%", "Accuracy cost: ~0.6pp"),
            "input_sanitizer": ("OutputManipulation: -3%", "PredictionShift: +43%", "Accuracy cost: ~0pp"),
            "prediction_smoother": ("OutputManipulation: +7%", "PredictionShift: +9%", "Accuracy cost: ~1.2pp"),
            "randomized": ("OutputManipulation: +3%", "PredictionShift: +7%", "Accuracy cost: ~2.3pp"),
        }

        console.print("[bold]Expected Improvements (empirical):[/bold]")
        for improvement in improvements[strategy]:
            console.print(f"  {improvement}")
        console.print()

        # Validate on test data if provided
        if test_data:
            df_test, _, _ = load_data(str(test_data), detect_pii=False)
            X_test = df_test.iloc[:, :-1].values
            y_test = df_test.iloc[:, -1].values

            print_info("Validating on test data...")

            # Compare predictions
            base_preds = model_obj.predict(X_test)
            hardened_preds = hardened.predict(X_test)

            # Detect if regression or classification
            is_classifier = hasattr(model_obj, 'predict_proba')

            if is_classifier:
                base_score = accuracy_score(y_test, base_preds)
                hardened_score = accuracy_score(y_test, hardened_preds)
                metric_name = "Accuracy"
            else:
                base_score = r2_score(y_test, base_preds)
                hardened_score = r2_score(y_test, hardened_preds)
                metric_name = "R2"

            console.print(f"[bold]Validation Results:[/bold]")
            console.print(f"  Base {metric_name}: {base_score:.4f}")
            console.print(f"  Hardened {metric_name}: {hardened_score:.4f}")
            console.print(f"  Change: {(hardened_score - base_score) * 100:+.2f}pp")
            console.print()

        # Save hardened model
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'wb') as f:
            pickle.dump(hardened, f)

        print_success(f"Hardened model saved to {output}")

        if verbose:
            console.print()
            console.print("[bold]Wrapper details:[/bold]")
            console.print(f"  Type: {type(hardened).__name__}")
            console.print(f"  Base model: {type(hardened.base_model).__name__}")
            if hasattr(hardened, 'is_calibrated_'):
                console.print(f"  Calibrated: {hardened.is_calibrated_}")

    except Exception as e:
        print_error(f"Hardening failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("harden-evaluate")
def harden_evaluate(
    model: Path = typer.Option(..., "--model", "-m", help="Path to model file"),
    calibration_data: Path = typer.Option(..., "--calibration-data", "-c", help="Path to calibration data"),
    test_data: Path = typer.Option(..., "--test-data", "-t", help="Path to test data"),
    n_attack_samples: int = typer.Option(50, "--n-samples", "-n", help="Number of samples for attack testing"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
):
    """
    Evaluate all hardening strategies against adversarial attacks.

    Runs OutputManipulation and PredictionShift attacks against each
    wrapper strategy and reports comparative results.

    This is an empirical ablation study for your specific model.

    Example:
        spectrum blue harden-evaluate --model ./model.pkl \\
            --calibration-data ./calib.csv --test-data ./test.csv \\
            --n-samples 50
    """
    import time
    from spectrum.blue.harden import (
        InputSanitizer, EnsembleProxy, OutputQuantizer,
        PredictionSmoother, RandomizedWrapper
    )
    from sklearn.metrics import r2_score

    print_header("SPECTRUM BLUE - HARDENING EVALUATION")

    # Load model and data
    model_obj = load_model(str(model))
    df_calib, _, _ = load_data(str(calibration_data), detect_pii=False)
    df_test, _, _ = load_data(str(test_data), detect_pii=False)

    X_calib = df_calib.iloc[:, :-1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    print_info(f"Base model: {type(model_obj).__name__}")
    print_info(f"Calibration samples: {len(X_calib)}")
    print_info(f"Test samples: {len(X_test)}")
    print_info(f"Attack samples: {n_attack_samples}")
    console.print()

    # Import attack wrappers
    try:
        from spectrum.red.attack import OutputManipulationWrapper, PredictionShiftWrapper
        attacks_available = True
    except ImportError:
        print_warning("Attack wrappers not available. Skipping attack evaluation.")
        attacks_available = False

    # Define wrappers to test
    wrappers = [
        ("Baseline", model_obj),
        ("InputSanitizer", InputSanitizer(model_obj).calibrate(X_calib)),
        ("EnsembleProxy", EnsembleProxy(model_obj).calibrate(X_calib)),
        ("OutputQuantizer", OutputQuantizer(model_obj, n_levels=20).calibrate(X_calib)),
        ("PredictionSmoother", PredictionSmoother(model_obj, n_samples=10)),
        ("Randomized", RandomizedWrapper(model_obj)),
    ]

    results = []

    for name, wrapper in wrappers:
        print_info(f"Testing: {name}")

        result = {"name": name}

        # Measure accuracy
        preds = wrapper.predict(X_test)
        r2 = r2_score(y_test, preds)
        result["r2"] = r2

        # Measure latency
        start = time.time()
        for _ in range(10):
            _ = wrapper.predict(X_test[:100])
        latency = (time.time() - start) / 10 * 1000
        result["latency_ms"] = latency

        # Run attacks if available
        if attacks_available:
            X_attack = X_test[:n_attack_samples]

            try:
                om = OutputManipulationWrapper(wrapper, n_bins=5, max_iter=30, max_eval=2000, confidence_level=0.99)
                om_metrics = om.run(X_attack)
                result["om_attempts"] = om_metrics.mean_attempts_to_success
            except Exception as e:
                result["om_error"] = str(e)

            try:
                ps = PredictionShiftWrapper(wrapper, epsilon=1.0, max_iter=50, n_directions=30,
                                            success_threshold=0.10, confidence_level=0.99)
                ps_metrics = ps.run(X_attack)
                result["ps_attempts"] = ps_metrics.mean_attempts_to_success
            except Exception as e:
                result["ps_error"] = str(e)

        results.append(result)
        console.print(f"  R2: {r2:.4f}, Latency: {latency:.1f}ms")

    # Summary
    console.print()
    print_header("EVALUATION RESULTS")

    baseline_r2 = results[0]["r2"]
    baseline_om = results[0].get("om_attempts", 1)
    baseline_ps = results[0].get("ps_attempts", 1)

    console.print(f"{'Strategy':<20} {'R2 Drop':<10} {'OM Change':<12} {'PS Change':<12} {'Latency':<10}")
    console.print("-" * 64)

    for r in results:
        r2_drop = (baseline_r2 - r["r2"]) * 100

        om = r.get("om_attempts", float('nan'))
        if not np.isnan(om) and baseline_om > 0:
            om_change = f"{(om / baseline_om - 1) * 100:+.0f}%"
        else:
            om_change = "N/A"

        ps = r.get("ps_attempts", float('nan'))
        if not np.isnan(ps) and baseline_ps > 0:
            ps_change = f"{(ps / baseline_ps - 1) * 100:+.0f}%"
        else:
            ps_change = "N/A"

        console.print(f"{r['name']:<20} {r2_drop:+.1f}pp     {om_change:<12} {ps_change:<12} {r['latency_ms']:.0f}ms")

    # Recommendation
    console.print()
    if attacks_available:
        best = max(results[1:], key=lambda r: (
            r.get("om_attempts", 0) / max(baseline_om, 1) +
            r.get("ps_attempts", 0) / max(baseline_ps, 1)
        ))
        print_success(f"Recommended strategy: {best['name']}")

    # Save results
    if output:
        import json
        output.mkdir(parents=True, exist_ok=True)
        result_file = output / "hardening_evaluation.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print_success(f"Results saved to {result_file}")


if __name__ == "__main__":
    app()
