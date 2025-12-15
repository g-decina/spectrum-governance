"""
spectrum lens - Governance and reporting commands
"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from spectrum.cli.utils import (
    console, print_header, print_success, print_error, print_info, print_warning
)
from spectrum.lens.compliance_report import ComplianceReport
from spectrum.lens.report_builder import ReportBuilder
from spectrum.lens.metrics_aggregator import MetricsAggregator
from spectrum.infra.logger import RCIALogger

app = typer.Typer(help="Governance infrastructure and reporting")


@app.command("init")
def init(
    audit_name: str = typer.Option(..., "--audit-name", "-n", help="Unique identifier for this audit"),
    client: str = typer.Option(..., "--client", "-c", help="Client organization name"),
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to model being audited"),
    regulatory_context: str = typer.Option("", "--regulatory-context", "-r", help="Comma-separated regulations (EU_AI_ACT, CFPB, EEOC)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory for audit artifacts"),
):
    """
    Initialize a new audit session.
    
    Creates a directory structure and logging context for a governance audit.
    
    Example:
        spectrum lens init --audit-name "Q1_2025_Credit_Audit" \\
            --client "Acme Financial" --model ./model.pkl \\
            --regulatory-context "EU_AI_ACT,CFPB"
    """
    print_header("SPECTRUM LENS - AUDIT INITIALIZATION")
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path(f"./audits/{audit_name}")
    
    # Create directory structure
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        (output_dir / "artifacts").mkdir(exist_ok=True)
        
        print_success(f"Audit directory created: {output_dir}")
    except Exception as e:
        print_error(f"Failed to create audit directory: {e}")
        raise typer.Exit(1)
    
    # Initialize RCIA logger
    log_path = output_dir / "logs" / "rcia_audit.jsonl"
    try:
        logger = RCIALogger(log_path=str(log_path))
        print_success(f"RCIA logger initialized: {log_path}")
    except Exception as e:
        print_error(f"Failed to initialize logger: {e}")
        raise typer.Exit(1)
    
    # Create audit metadata
    metadata = {
        "audit_name": audit_name,
        "client": client,
        "model_path": str(model_path),
        "regulatory_context": regulatory_context.split(",") if regulatory_context else [],
        "created_at": datetime.now().isoformat(),
        "status": "ACTIVE"
    }
    
    metadata_file = output_dir / "audit_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_success(f"Metadata saved: {metadata_file}")
    console.print()
    
    # Display summary
    console.print("[bold]Audit Session Summary:[/bold]")
    console.print(f"  Name: {audit_name}")
    console.print(f"  Client: {client}")
    console.print(f"  Model: {model_path}")
    console.print(f"  Regulations: {regulatory_context or 'None specified'}")
    console.print(f"  Output: {output_dir}")
    console.print()
    
    print_success("Audit session initialized successfully")


@app.command("status")
def status(
    audit_session: Path = typer.Option(..., "--audit-session", "-a", help="Path to audit session directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed event log"),
):
    """
    Display status of an audit session.
    
    Shows metadata, progress, and recent events.
    
    Example:
        spectrum lens status --audit-session ./audits/Q1_2025_Credit_Audit
    """
    print_header("SPECTRUM LENS - AUDIT STATUS")
    
    # Load metadata
    metadata_file = audit_session / "audit_metadata.json"
    if not metadata_file.exists():
        print_error(f"No audit metadata found at {metadata_file}")
        print_info("Use 'spectrum lens init' to create a new audit session")
        raise typer.Exit(1)
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Display metadata
    console.print("[bold]Audit Session:[/bold]", metadata['audit_name'])
    console.print(f"  Client: {metadata['client']}")
    console.print(f"  Model: {metadata['model_path']}")
    console.print(f"  Created: {metadata['created_at']}")
    console.print(f"  Status: {metadata['status']}")
    console.print(f"  Regulations: {', '.join(metadata.get('regulatory_context', []))}")
    console.print()
    
    # Check log file
    log_path = audit_session / "logs" / "rcia_audit.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        
        console.print(f"[bold]RCIA Events:[/bold] {len(lines)} logged")
        
        if verbose and lines:
            console.print("\n[bold]Recent Events (last 10):[/bold]")
            for line in lines[-10:]:
                event = json.loads(line)
                console.print(f"  [{event['timestamp']}] {event['event_type']}")
    else:
        console.print("[bold]RCIA Events:[/bold] No log file found")


@app.command("generate-report")
def generate_report(
    audit_session: Path = typer.Option(..., "--audit-session", "-a", help="Path to audit session directory"),
    template: str = typer.Option("tier1_forensic_audit", "--template", "-t", help="Report template name"),
    format: str = typer.Option("docx", "--format", "-f", help="Output format: docx, html, pdf"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """
    Generate compliance report from audit session data.

    Available templates:
      - tier1_forensic_audit (General purpose)
      - eu_ai_act_annex_iv (EU AI Act Technical File)
      - cfpb_model_validation (CFPB lending compliance)
      - eeoc_selection_audit (Employment screening)

    Example:
        spectrum lens generate-report --audit-session ./audits/Q1_2025/ \\
            --template tier1_forensic_audit --format docx
    """
    print_header("SPECTRUM LENS - REPORT GENERATION")
    
    # Load metadata
    metadata_file = audit_session / "audit_metadata.json"
    if not metadata_file.exists():
        print_error(f"No audit metadata found at {metadata_file}")
        raise typer.Exit(1)
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print_info(f"Audit: {metadata['audit_name']}")
    print_info(f"Template: {template}")
    print_info(f"Format: {format}")
    console.print()
    
    # Set default output path
    if output is None:
        output = audit_session / "reports" / f"{template}_report.{format}"

    try:
        # Aggregate metrics from all CLI test outputs
        print_info("Aggregating metrics from audit artifacts...")
        aggregator = MetricsAggregator(audit_session)
        aggregated_data = aggregator.aggregate()

        # Save aggregated metrics for reference
        aggregator.save_aggregated_metrics()
        print_success("Metrics aggregated successfully")
        console.print()

        # Create report builder
        builder = ReportBuilder()

        # Generate report based on format
        if format == "docx":
            print_info("Generating professional DOCX report...")
            docx_path = builder.generate_docx(
                data=aggregated_data,  # Use aggregated data instead of metadata
                output_path=str(output),
                template_type=template
            )
            print_success(f"DOCX report generated: {output}")

        elif format == "html":
            # Warning about template availability
            if template not in ["tier1_forensic_audit"]:
                print_warning(f"HTML template '{template}' not found")
                print_info("Using fallback HTML generation")

            html = builder.generate_html(
                template_name=f"{template}.html",
                data=aggregated_data,  # Use aggregated data
                output_path=str(output)
            )
            print_success(f"HTML report generated: {output}")

        elif format == "pdf":
            print_warning("PDF generation not yet implemented")
            print_info("Generating HTML instead...")
            html = builder.generate_html(
                template_name=f"{template}.html",
                data=aggregated_data,  # Use aggregated data
                output_path=str(output.with_suffix('.html'))
            )
            print_success(f"HTML report generated: {output.with_suffix('.html')}")

        else:
            print_error(f"Unsupported format: {format}")
            print_info("Supported formats: docx, html, pdf (html fallback)")
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Report generation failed: {e}")
        if "python-docx" in str(e):
            print_info("Install python-docx with: pip install python-docx")
        raise typer.Exit(1)


@app.command("close")
def close(
    audit_session: Path = typer.Option(..., "--audit-session", "-a", help="Path to audit session directory"),
    archive: bool = typer.Option(False, "--archive", help="Create tar.gz archive"),
):
    """
    Close and finalize an audit session.
    
    Marks the session as completed and optionally creates an archive.
    
    Example:
        spectrum lens close --audit-session ./audits/Q1_2025/ --archive
    """
    print_header("SPECTRUM LENS - AUDIT CLOSURE")
    
    # Load metadata
    metadata_file = audit_session / "audit_metadata.json"
    if not metadata_file.exists():
        print_error(f"No audit metadata found at {metadata_file}")
        raise typer.Exit(1)
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Update status
    metadata['status'] = "COMPLETED"
    metadata['completed_at'] = datetime.now().isoformat()
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_success(f"Audit session closed: {metadata['audit_name']}")
    
    # Archive if requested
    if archive:
        import tarfile
        
        archive_name = f"{audit_session.name}.tar.gz"
        archive_path = audit_session.parent / archive_name
        
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(audit_session, arcname=audit_session.name)
            
            print_success(f"Archive created: {archive_path}")
        except Exception as e:
            print_error(f"Archive creation failed: {e}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
