"""
Shared utilities for the spectrum CLI.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List
import pickle
import joblib
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from spectrum.infra.privacy import (
    PIIDetector,
    PIIDataSanitizer,
    PIIPolicy,
    PIISchema,
    PIIDetectionResult
)
from spectrum.infra.types import RiskProfile

console = Console()


def print_header(text: str):
    """Print a formatted section header."""
    console.print()
    console.print("=" * 80, style="bold blue")
    console.print(f"  {text}", style="bold blue")
    console.print("=" * 80, style="bold blue")
    console.print()


def print_success(message: str):
    """Print a success message."""
    console.print(f"✓ {message}", style="bold green")


def print_error(message: str):
    """Print an error message."""
    console.print(f"✗ {message}", style="bold red")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"⚠ {message}", style="bold yellow")


def print_info(message: str):
    """Print an info message."""
    console.print(f"ℹ {message}", style="bold cyan")


def load_model(model_path: str):
    """
    Load a trained model from file.
    
    Supports .pkl, .joblib, and .pickle extensions.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print_error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        if model_path.suffix in ['.pkl', '.pickle']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        else:
            print_error(f"Unsupported model format: {model_path.suffix}")
            print_info("Supported formats: .pkl, .pickle, .joblib")
            sys.exit(1)
        
        print_success(f"Model loaded: {model_path.name}")
        return model
    
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        sys.exit(1)


def load_data(
    data_path: str,
    risk_profile: Optional[RiskProfile] = None,
    pii_schema: Optional[PIISchema] = None,
    detect_pii: bool = True
) -> Tuple[pd.DataFrame, Optional[List[PIIDetectionResult]], Optional[dict]]:
    """
    Load data from CSV or Parquet file with optional PII detection and sanitization.

    :param data_path: Path to data file (.csv or .parquet)
    :param risk_profile: Optional RiskProfile to determine PII policy
    :param pii_schema: Optional explicit PII schema
    :param detect_pii: Whether to automatically detect PII (default: True)
    :return: (DataFrame, PII detection results, audit report)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        print_error(f"Data file not found: {data_path}")
        sys.exit(1)

    try:
        # Load data
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(data_path)
        else:
            print_error(f"Unsupported data format: {data_path.suffix}")
            print_info("Supported formats: .csv, .parquet")
            sys.exit(1)

        print_success(f"Data loaded: {data_path.name} ({len(df)} rows, {len(df.columns)} columns)")

        # PII Detection and Sanitization
        pii_results = None
        audit_report = None

        if detect_pii:
            # Initialize detector with optional schema
            detector = PIIDetector(schema=pii_schema or PIISchema())
            pii_results = detector.detect(df)

            if pii_results:
                print_warning(f"Detected {len(pii_results)} potential PII field(s)")
                display_pii_detections(pii_results)

                # Determine policy
                if risk_profile and risk_profile.pii_policy:
                    policy = risk_profile.pii_policy
                elif risk_profile:
                    # Use default policy based on risk level
                    from spectrum.infra.privacy import get_recommended_policy
                    policy = get_recommended_policy(risk_profile.level.value)
                else:
                    # Safe default: detect only
                    policy = PIIPolicy.DETECT_ONLY

                # Apply sanitization if needed
                if policy != PIIPolicy.DETECT_ONLY and policy != PIIPolicy.ALLOW:
                    sanitizer = PIIDataSanitizer(policy=policy)
                    df, audit_report = sanitizer.sanitize(df, pii_results)

                    if policy == PIIPolicy.PSEUDONYMIZE:
                        print_info(f"Applied PSEUDONYMIZE policy: {len(audit_report['fields_modified'])} field(s) hashed")
                    elif policy == PIIPolicy.REDACT:
                        print_info(f"Applied REDACT policy: {len(audit_report['fields_modified'])} field(s) removed")
            else:
                print_success("No PII detected in dataset")

        return df, pii_results, audit_report

    except Exception as e:
        print_error(f"Failed to load data: {e}")
        sys.exit(1)


def load_data_simple(data_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Parquet file (legacy interface without PII handling).

    For backward compatibility. Use load_data() for PII-aware loading.
    """
    df, _, _ = load_data(data_path, detect_pii=False)
    return df


def display_pii_detections(results: List[PIIDetectionResult]):
    """Display PII detection results in a formatted table."""
    table = Table(title="PII Detection Results", box=box.ROUNDED)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("PII Type", style="yellow")
    table.add_column("Confidence", style="white")
    table.add_column("Reason", style="dim")

    for result in results:
        confidence_str = f"{result.confidence:.0%}"
        table.add_row(
            result.field_name,
            result.pii_type.value,
            confidence_str,
            result.reason
        )

    console.print(table)


def create_summary_table(data: dict, title: str = "Summary") -> Table:
    """Create a rich table for displaying key-value pairs."""
    table = Table(title=title, box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    for key, value in data.items():
        table.add_row(key, str(value))
    
    return table


def format_percentage(value: float) -> str:
    """Format a float as a percentage."""
    return f"{value * 100:.2f}%"


def interpret_attack_success_rate(score: float) -> tuple[str, str]:
    """
    Interpret attack success rate according to specs.

    Lower values indicate better robustness.

    Returns (risk_level, interpretation)
    """
    if score <= 0.30:
        return "GREEN - Robust", "Strong robustness. Acceptable for high-risk deployment."
    elif score <= 0.60:
        return "YELLOW - Moderate", "Vulnerability detected. Recommend hardening before deployment."
    else:
        return "RED - Fragile", "Highly susceptible to adversarial manipulation. Not suitable for high-risk deployment."




def interpret_drift_status(max_psi: float, alert_threshold: float = 0.25, monitor_threshold: float = 0.10) -> tuple[str, str]:
    """
    Interpret drift status based on PSI thresholds.
    
    Returns (status_symbol, interpretation)
    """
    if max_psi < monitor_threshold:
        return "✓", "No significant drift detected"
    elif max_psi < alert_threshold:
        return "⚡", "Moderate drift detected - monitor closely"
    else:
        return "⚠", "Significant drift detected - immediate action required"
