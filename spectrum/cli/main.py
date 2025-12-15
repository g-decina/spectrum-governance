"""
spectrum - Main CLI entry point for ML Purple Teaming

Command-line interface for adversarial auditing and regulatory compliance
of machine learning systems.

Usage:
    spectrum [OPTIONS] COMMAND [ARGS]...

Commands:
    red     Adversarial testing and attack campaigns
    blue    Defensive evaluation and hardening
    lens    Governance infrastructure and reporting

Examples:
    # Run adversarial scan
    spectrum red scan --model ./model.pkl --data ./test.csv
    
    # Generate SHAP explanations
    spectrum blue explain --model ./model.pkl --data ./test.csv
    
    # Initialize audit session
    spectrum lens init --audit-name "Q1_2025_Audit" --client "Acme Corp"
"""

import typer
from pathlib import Path
from typing import Optional

from spectrum.cli import __version__
from spectrum.cli.utils import console
from spectrum.cli import red, blue, lens

# Create main app
app = typer.Typer(
    name="spectrum",
    help="ML Purple Teaming - Adversarial Auditing & Regulatory Compliance",
    add_completion=False,
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(red.app, name="red", help="Adversarial testing")
app.add_typer(blue.app, name="blue", help="Defensive evaluation")
app.add_typer(lens.app, name="lens", help="Governance & reporting")


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"spectrum version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v, -vv, -vvv)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
):
    """
    spectrum - ML Purple Teaming CLI
    
    A command-line interface for adversarial auditing and regulatory
    compliance of machine learning systems.
    
    Based on the ML Purple Teaming Field Manual specifications.
    """
    # Global options handling would go here
    # For now, these are just placeholders
    if config:
        # TODO: Load configuration file
        pass
    
    if verbose:
        # TODO: Set logging level based on verbosity
        pass
    
    if quiet:
        # TODO: Suppress output
        pass


if __name__ == "__main__":
    app()
